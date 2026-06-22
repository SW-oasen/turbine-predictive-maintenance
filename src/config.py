from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "db" / "turbofan_engine.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

MODEL_PATH = PROJECT_ROOT / "models"

#DATABASE_URL = "sqlite:///../db/turbofan_engine.db"
#RAW_DATA_PATH = "../data/raw/"

# Variante A: explizite Liste
DATASETS_META = [
  {
    "dataset_id": "FD001",
    "train_file": "train_FD001.txt",
    "test_file":  "test_FD001.txt",
    "rul_file":   "RUL_FD001.txt",
    "condition": "single-sea level",
    "fault_mode": "HPC"
  },
  {
    "dataset_id": "FD002",
    "train_file": "train_FD002.txt",
    "test_file":  "test_FD002.txt",
    "rul_file":   "RUL_FD002.txt",
    "condition": "multiple",
    "fault_mode": "HPC"
  },
  {
    "dataset_id": "FD003",
    "train_file": "train_FD003.txt",
    "test_file":  "test_FD003.txt",
    "rul_file":   "RUL_FD003.txt",
    "condition": "single-sea level",
    "fault_mode": "HPC+Fan"
  },
  {
    "dataset_id": "FD004",
    "train_file": "train_FD004.txt",
    "test_file":  "test_FD004.txt",
    "rul_file":   "RUL_FD004.txt",
    "condition": "multiple",
    "fault_mode": "HPC+Fan"
  }
]

OPTIONS = {
  "checkpoint": True,
  "drop_constant_sensors": True,
  "rolling_windows": [5, 20],
  "sensors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}