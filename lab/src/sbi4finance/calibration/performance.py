"""Timing metadata for bounded numerical workers, without importing a backend.

Phases may nest. Their elapsed times are observations, not an additive cost
decomposition; in particular, callbacks can overlap accelerator work. Allocated
resource time is distinct from process CPU time and GPU utilization.
"""

from contextlib import contextmanager
from importlib import metadata
import os
import platform
import resource
import sys
import time


def synchronize_model(model):
    """Wait for model/optimizer arrays at worker phase boundaries, when on JAX.

    This is intentionally not called after every training batch. Keras owns its
    stateless training state while fitting; workers call this outside ``fit``.
    """
    if model is None or "jax" not in sys.modules:
        return
    import jax
    arrays = [variable.value for variable in model.variables]
    optimizer = getattr(model, "optimizer", None)
    if optimizer is not None:
        arrays.extend(variable.value for variable in optimizer.variables)
    jax.block_until_ready(arrays)


def synchronize_logs(logs):
    """Wait for the first compiled step's returned results, without copying weights."""
    if "jax" in sys.modules:
        import jax
        jax.block_until_ready(logs)


class PerformanceRecorder:
    """Record invocation-local wall/process clocks and explicitly scoped phases."""

    def __init__(self):
        self.started = self.now()
        self.phases = {}

    @staticmethod
    def now():
        return time.perf_counter(), time.process_time()

    def finish(self, name, started):
        ended = self.now()
        record = self.phases.setdefault(name, {"wall_seconds": 0.0, "process_cpu_seconds": 0.0, "calls": 0})
        record["wall_seconds"] += ended[0] - started[0]
        record["process_cpu_seconds"] += ended[1] - started[1]
        record["calls"] += 1

    @contextmanager
    def phase(self, name, *, synchronize=None):
        if synchronize is not None:
            synchronize()
        started = self.now()
        try:
            yield
        finally:
            if synchronize is not None:
                synchronize()
            self.finish(name, started)

    def report(self, *, work=None):
        ended = self.now()
        wall, cpu = ended[0] - self.started[0], ended[1] - self.started[1]
        def allocated_count(*names):
            for name in names:
                value = os.environ.get(name, "")
                if value.isdecimal() and int(value) > 0:
                    return int(value)
            return None
        cores = allocated_count("SLURM_CPUS_PER_TASK")
        gpus = allocated_count("SLURM_GPUS", "SLURM_GPUS_ON_NODE")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
        versions = {}
        for name in ("bayesflow", "keras", "jax", "numpy", "scipy"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        devices = None
        device_memory = None
        if "jax" in sys.modules:
            import jax
            devices = [str(device) for device in jax.devices()]
            device_memory = [{"device": str(device), "stats": device.memory_stats()} for device in jax.devices()]
        return {
            "format": "full_surface_performance_v1",
            "timing_scope": "current worker invocation only; previous or interrupted attempts are not included",
            "wall_seconds": wall, "process_cpu_seconds": cpu,
            "phases": dict(self.phases),
            "phases_are_additive": False,
            "phase_note": "fit includes validation, first-step compilation and checkpoint callbacks; nested phases overlap",
            "allocation": {"cpu_cores": cores, "gpu_count": gpus,
                           "cpu_seconds": None if cores is None else cores * wall,
                           "gpu_seconds": None if gpus is None else gpus * wall,
                           "source": "Slurm allocation environment, when a single integer is available; not utilization"},
            "hardware": {"system": platform.system(), "machine": platform.machine(),
                         "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "devices": devices,
                         "device_memory": device_memory},
            "thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE")},
            "runtime": {"python": platform.python_version(), **versions},
            "peak_process_rss_bytes": peak_bytes,
            "peak_rss_scope": "process lifetime high-water mark, including imports and earlier invocations",
            "work": work,
        }


def training_callback(recorder):
    """Time first-step compilation and each validation pass using Keras callbacks."""
    import keras

    class TrainingPhases(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.fit_started = None
            self.first_batch_started = None
            self.first_batch_complete = False
            self.validation_started = None

        def on_train_begin(self, logs=None):
            if not self.first_batch_complete and self.fit_started is not None:
                recorder.finish("first_fit_preparation_including_build", self.fit_started)

        def on_train_batch_begin(self, batch, logs=None):
            if not self.first_batch_complete:
                self.first_batch_started = recorder.now()

        def on_train_batch_end(self, batch, logs=None):
            if not self.first_batch_complete:
                synchronize_logs(logs)
                recorder.finish("first_training_batch_including_compile", self.first_batch_started)
                self.first_batch_complete = True

        def on_test_begin(self, logs=None):
            self.validation_started = recorder.now()

        def on_test_end(self, logs=None):
            synchronize_logs(logs)
            recorder.finish("validation", self.validation_started)

    return TrainingPhases()
