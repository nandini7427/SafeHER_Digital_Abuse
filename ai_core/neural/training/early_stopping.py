"""Early stopping monitor for neural training pipelines.

Tracks validation metrics across epochs to halt training when performance
plateaus, preventing overfitting and unneeded compute expenditure.
"""

from typing import Any, Dict, Optional


class EarlyStopping:
    """Early stopping monitor supporting metric minimization or maximization."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as improvement.
            mode: 'min' (e.g. for loss) or 'max' (e.g. for accuracy).
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def step(self, score: float, epoch: int) -> bool:
        """Evaluate metric score for the current epoch.

        Args:
            score: Metric value (e.g. validation loss).
            epoch: Current epoch index.

        Returns:
            is_best: True if this epoch achieved a new best metric score.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True

        is_improvement = False
        if self.mode == "min":
            if score < (self.best_score - self.min_delta):
                is_improvement = True
        else:
            if score > (self.best_score + self.min_delta):
                is_improvement = True

        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score", None)
        self.best_epoch = state.get("best_epoch", 0)
        self.early_stop = state.get("early_stop", False)
