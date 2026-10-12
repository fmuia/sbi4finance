"""Select a validation observation from the simulator's own HKDE bank reader."""

import hashlib
from pathlib import Path

from .local import ArtifactError
from .options import hkde_panel_prices, prepare_hkde_panel


def load_hkde_bank_panel(path, *, example_id: str, checkpoint_task: dict):
    """Load one validation panel as local observations, with separate truth.

    The owning ``option_bank.load_bank`` verifies the whole bank. This small
    teaching bridge selects only validation IDs; it never exposes train/test
    observations as held-out evidence. No bank, simulator, or adapter is copied.
    Simulator dependencies are imported only when this function is called.
    """
    if not isinstance(example_id, str) or not example_id.startswith("validation:"):
        raise ArtifactError("select a validation:<ID> panel; train/test are outside this walkthrough")
    from ..option_bank import load_bank

    path = Path(path)
    manifest, splits = load_bank(path)
    task = manifest["configuration"]["task"]
    if task != checkpoint_task:
        raise ArtifactError("bank scientific task differs from the complete checkpoint task")
    split = splits["validation"]
    ids = split["example_id"].tolist()
    if example_id not in ids:
        raise ArtifactError(f"validation example not found: {example_id}")
    index = ids.index(example_id)
    metadata = {
        "artifact_id": f"hkde-bank-{example_id.replace(':', '-')}",
        "data_origin": "synthetic", "purpose": "validation",
        "conventions": {
            "quote_form": "single_premium", "price_unit": "currency",
            "exercise": "European", "settlement": "cash",
            "premium_basis": "per_underlying_unit",
        },
        "provenance": {
            "source": "option_bank_v1; simulator-owned reader",
            "bank_manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
            "validation_split_sha256": manifest["splits"]["validation"]["sha256"],
            "task_fingerprint": manifest["task_fingerprint"],
            "root_seed": manifest["configuration"]["root_seed"], "case_id": example_id,
        },
        "truth": dict(zip(task["parameter_names"], map(float, split["parameters"][index]), strict=True)),
    }
    records = [
        {"contract_id": f"c{j}", **contract, "price": float(price[0])}
        for j, (contract, price) in enumerate(zip(
            task["context"]["contracts"], split["observations"][index], strict=True))
    ]
    panel = prepare_hkde_panel(records, metadata=metadata, task=task)
    hkde_panel_prices(panel, checkpoint_task=checkpoint_task)
    return panel
