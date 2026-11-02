"""BayesFlow 2.1 inference on the complete 63-contract tutorial design.

The known-curve rough task conditions on its declared curve, as in the original
course. The joint rough model's curve values are auxiliary labels, never conditions.
Set encoders here represent complete surfaces, not missing-quote support.
"""

# Shared implementations are owned by the tutorial runtime.
from sbi4f_tutorial.shared.full_surface import (
    FAMILIES,
    ENCODERS,
    validate_task,
    uses_known_curve,
    observations,
    conditioning_data,
    training_data,
    fresh_noise,
    configuration,
    build_workflow,
    sample,
    load_checkpoint,
)
