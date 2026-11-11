"""Fresh, temporary source identities for fake-kernel actual-layout tests.

Committed work orders describe immutable production runs. Reusing their source
hashes as unit-test fixtures makes any later runtime edit look like an attempt
to resume those runs. These helpers retain their financial panel and declared
ancestry while binding a separate test-only copy to the current checkout.
Production files and validators are never changed or bypassed.
"""
from importlib import metadata
from pathlib import Path

from sbi4finance.calibration import rough_actual_layout_pilot_portable as portable


EVIDENCE = portable.ROOT/"lab/evidence/pilot-results/2026-11-10/bergomi-observation-refresh"
HISTORICAL_PORTABLE = EVIDENCE/"preparation-portable/work-order.json"


def current_portable(root):
    """Copy the authoritative panel into a new current-source test identity."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    value = portable.read(HISTORICAL_PORTABLE)
    value["source_bindings"] = portable.source_bindings()
    value["execution_versions"] = {name: metadata.version(name) for name in ("numpy", "scipy")}
    value["test_fixture"] = "current-source metadata for fake-kernel unit tests only"
    path = root/"current-portable-work-order.json"
    portable.write_new(path, value)
    portable.validate(value)
    return value, path


def current_bank_plan(root, historical, bank):
    """Rebind a test-only plan and its parent to the current producer family."""
    root = Path(root)
    _, parent = current_portable(root)
    value = portable.read(historical)
    value["parent_work_order"] = portable.binding(parent)
    value["parent_work_order_value"] = portable.read(parent)
    value["source_bindings"] = bank.sources()
    value["versions"] = bank.versions()
    value["test_fixture"] = "current-source metadata for fake-kernel unit tests only"
    path = root/"current-bank-plan.json"
    portable.write_new(path, value)
    bank.validate(value)
    return value, path
