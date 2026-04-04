import os
import gc
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATE_FORMAT = "%Y-%m-%d-%H:%M"
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

def get_logger(name: str = "logger", write_to_file: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt=DATE_FORMAT)
        # Logs in console
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        # Logs in file
        if write_to_file:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file_name = re.sub(r"[^\w\-]+", "_", name.strip())
            log_file_path = os.path.join(_LOG_DIR, f"{log_file_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log")
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger

def choose_device(logger: logging.Logger)->str:
    if torch.backends.cuda.is_built():
        # usually on Windows machines with GPU
        device = "cuda"
    elif torch.backends.mps.is_built():
        # usually on MAC
        device = "mps"
    else:
        # if not we should use our CPU
        device = "cpu"
    logger.info(f"Chosen device: {device}")
    return device


def collect_garbage(device: str) -> None:
    """GPU garbage collection
    """
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def set_seed(seed: Optional[int] = None) -> None:
  """Set seed for reproducibility
  """
  seed  = 1234 if seed is None else seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], logger: logging.Logger) -> pd.DataFrame:

    metrics = [
                {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                    "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                    "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                }
            ]

    metrics_df = pd.DataFrame(metrics)
    logger.info(f"Evaluation metrics:\n{metrics_df}")
    logger.info(f"Classification report:\n{classification_report(y_true, y_pred, target_names=target_names, digits=3)}")
    return metrics_df


