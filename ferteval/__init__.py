"""ferteval — evaluation suite for Delphi-style fertility-sequence models.

Phase 1 (implemented): discrimination (AUC) and cohort-subgroup calibration,
adapted from the upstream gerstung-lab/Delphi `evaluate_auc.py` / notebook.

Phase 2 (interfaces reserved in `demography` and `sampling`): demographic
measures (completed cohort fertility, parity progression ratios, ASFR, ...)
computed identically on real and model-forecasted sequences.
"""

__version__ = "0.1.0"
