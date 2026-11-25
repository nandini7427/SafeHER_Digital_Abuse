"""Structured JSONL training logger for SafeHER neural training pipelines.

Records epoch-level quantitative metrics, hyperparameters, and run summaries
while strictly protecting against saving raw image contents, private survivor data,
or unhashed file references.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class StructuredTrainingLogger:
    """JSON Lines training logger capturing structured quantitative epoch metrics."""

    def __init__(self, log_dir: Union[str, Path], run_name: Optional[str] = None):
        """Initialize StructuredTrainingLogger.

        Args:
            log_dir: Directory where JSONL log files will be persisted.
            run_name: Optional run prefix for log filenames.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{run_name}_" if run_name else ""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{prefix}metrics_{timestamp_str}.jsonl"
        self.summary_file = self.log_dir / f"{prefix}summary_{timestamp_str}.json"

        self.history: List[Dict[str, Any]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        train_precision: float,
        train_recall: float,
        val_loss: float,
        val_accuracy: float,
        val_precision: float,
        val_recall: float,
        learning_rate: float,
        epoch_duration_seconds: float,
        is_best: bool = False,
        early_stopping_counter: int = 0,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a single epoch's metrics to JSONL log file."""
        record: Dict[str, Any] = {
            "epoch": epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_accuracy, 5),
            "train_precision": round(train_precision, 5),
            "train_recall": round(train_recall, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_accuracy, 5),
            "val_precision": round(val_precision, 5),
            "val_recall": round(val_recall, 5),
            "learning_rate": float(f"{learning_rate:.6e}"),
            "epoch_duration_seconds": round(epoch_duration_seconds, 2),
            "is_best": is_best,
            "early_stopping_counter": early_stopping_counter,
        }

        if extra_metrics:
            for k, v in extra_metrics.items():
                if not isinstance(v, (bytes, bytearray)):
                    record[k] = v

        self.history.append(record)

        # Append to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} | Best: {is_best}"
        )

        return record

    def save_summary(self, summary_data: Dict[str, Any]) -> Path:
        """Persist structured summary JSON at the conclusion of training."""
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        return self.summary_file
