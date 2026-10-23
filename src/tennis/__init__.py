"""ATP match prediction project: DuckDB feature layer, Elo variants, ML models."""

# Frozen evaluation split. The test years are looked at only for milestone evaluations.
TRAIN_END_YEAR = 2021   # train: <= 2021
VAL_YEAR = 2022         # tuning / model selection
TEST_START_YEAR = 2023  # frozen test set: >= 2023

__version__ = "0.1.0"
