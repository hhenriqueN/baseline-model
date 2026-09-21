"""Central path definitions for the imitation-learning data pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../voos-piloto
DATA = ROOT / "data"

RAW_DIR = DATA / "raw"
MANIFESTS_DIR = DATA / "manifests"
VALIDATED_DIR = DATA / "validated"

PROCESSED_DIR = DATA / "processed"
PROCESSED_10HZ_DIR = PROCESSED_DIR / "10hz"
PROCESSED_FULL_EPISODES_DIR = PROCESSED_DIR / "full_episodes"
PROCESSED_INIT_TRANSIENTS_DIR = PROCESSED_DIR / "initialization_transients"
PROCESSED_EXCLUDED_DIR = PROCESSED_DIR / "excluded"

FEATURES_DIR = DATA / "features"
FEATURES_UNNORM_DIR = FEATURES_DIR / "unnormalized"
FEATURES_BASELINE_CORE_DIR = FEATURES_DIR / "baseline_core"
FEATURES_FULL_APPROACH_DIR = FEATURES_DIR / "full_approach"
FEATURES_RECOVERY_ABLATION_DIR = FEATURES_DIR / "recovery_ablation"

SPLITS_DIR = DATA / "splits"

SCALERS_DIR = DATA / "scalers"
SCALERS_FOLDS_DIR = SCALERS_DIR / "folds"
SCALERS_FINAL_DIR = SCALERS_DIR / "final_all_valid"

REPORTS_DIR = DATA / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CONFIG_DIR = DATA / "config"

ALL_DIRS = [
    RAW_DIR, MANIFESTS_DIR, VALIDATED_DIR,
    PROCESSED_10HZ_DIR, PROCESSED_FULL_EPISODES_DIR,
    PROCESSED_INIT_TRANSIENTS_DIR, PROCESSED_EXCLUDED_DIR,
    FEATURES_UNNORM_DIR, FEATURES_BASELINE_CORE_DIR,
    FEATURES_FULL_APPROACH_DIR, FEATURES_RECOVERY_ABLATION_DIR,
    SPLITS_DIR, SCALERS_FOLDS_DIR, SCALERS_FINAL_DIR,
    FIGURES_DIR, CONFIG_DIR,
]


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
