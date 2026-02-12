"""Probes that bundle feature extraction with a classifier.

Each probe is a self-contained class that knows:
  1. How to extract features from a model (layer, pooling strategy)
  2. How to train and predict (sklearn classifier or end-to-end finetuning)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm


# --- Shared utilities ---


def pool_hidden_states(
    hidden: torch.Tensor, mask: torch.Tensor, method: str
) -> torch.Tensor:
    """Pool hidden states using the given method.

    Args:
        hidden: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) attention mask
        method: "mean" or "last"

    Returns:
        (batch, hidden_dim) pooled representations
    """
    if method == "mean":
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
    elif method == "last":
        seq_lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), seq_lengths]
    else:
        raise ValueError(f"Unknown pool method: {method}")


# --- Base class ---


class Probe(ABC):
    """Base class for probes.

    All probes expose a unified fit/predict interface. Subclasses differ
    in whether they use frozen feature extraction + sklearn or end-to-end
    finetuning with a classification head.
    """

    @property
    @abstractmethod
    def feature_key(self) -> str:
        """Unique key for caching and result naming."""
        ...

    @abstractmethod
    def fit(
        self,
        texts: list[str],
        labels: np.ndarray,
        model,
        tokenizer,
        *,
        cache_dir: Path,
        seed: int = 42,
    ) -> None:
        """Train the probe."""
        ...

    @abstractmethod
    def predict(
        self,
        texts: list[str],
        model,
        tokenizer,
        *,
        cache_dir: Path,
    ) -> np.ndarray:
        """Return predicted class labels."""
        ...

    @abstractmethod
    def predict_proba(
        self,
        texts: list[str],
        model,
        tokenizer,
        *,
        cache_dir: Path,
    ) -> np.ndarray:
        """Return predicted probabilities, shape (n_samples, n_classes)."""
        ...


# --- Hidden state probes (frozen features + sklearn) ---


class HiddenStateProbe(Probe):
    """Probe that extracts pooled hidden states from a specific layer.

    Subclasses set `layer`, `pool`, and implement `make_classifier`.
    Feature extraction results are cached as .npz files.
    """

    layer: int = -1
    pool: str = "mean"

    @property
    def feature_key(self) -> str:
        return f"layer{self.layer}_pool{self.pool}"

    @abstractmethod
    def make_classifier(self, seed: int = 42):
        """Return a fresh sklearn estimator."""
        ...

    def extract_features(
        self,
        model,
        tokenizer,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract pooled hidden state features for all texts."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Extracting hidden states")

        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[self.layer]
            mask = inputs["attention_mask"]
            pooled = pool_hidden_states(hidden, mask, self.pool)
            results.append(pooled.cpu().float().numpy())

        return np.concatenate(results, axis=0)

    def _get_or_extract_features(
        self,
        texts: list[str],
        model,
        tokenizer,
        cache_dir: Path,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Load cached features or extract them."""
        cache_path = cache_dir / f"hidden_states_{self.feature_key}.npz"
        if cache_path.exists():
            return np.load(cache_path)["hidden_states"]
        if model is None:
            raise RuntimeError(
                f"Features not cached at {cache_path} and no model provided"
            )
        X = self.extract_features(model, tokenizer, texts, batch_size=batch_size)
        np.savez(cache_path, hidden_states=X)
        return X

    def fit(self, texts, labels, model, tokenizer, *, cache_dir, seed=42):
        X = self._get_or_extract_features(texts, model, tokenizer, cache_dir)
        self._clf = self.make_classifier(seed=seed)
        self._clf.fit(X, labels)

    def predict(self, texts, model, tokenizer, *, cache_dir):
        X = self._get_or_extract_features(texts, model, tokenizer, cache_dir)
        return self._clf.predict(X)

    def predict_proba(self, texts, model, tokenizer, *, cache_dir):
        X = self._get_or_extract_features(texts, model, tokenizer, cache_dir)
        return self._clf.predict_proba(X)


# --- Finetuning probes (end-to-end with classification head) ---


class FinetuneProbe(Probe):
    """Probe that attaches a classification head and trains end-to-end.

    Supports full finetuning or LoRA. Caches model checkpoints.
    The base model is NOT modified in-place: LoRA uses merge_and_unload
    for teardown, and full finetuning works on a deepcopy.

    NOTE: full finetuning deepcopies the model, roughly doubling GPU memory.
    For a 1B model in fp32 this is ~4GB extra; for larger models consider LoRA.
    """

    def __init__(
        self,
        strategy: str = "lora",
        layer: int = -1,
        pool: str = "mean",
        lr: float = 1e-4,
        epochs: int = 3,
        batch_size: int = 16,
        num_classes: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: list[str] | None = None,
    ):
        self.strategy = strategy
        self.layer = layer
        self.pool = pool
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]

        self._head: nn.Module | None = None
        self._inference_model = None

    @property
    def feature_key(self) -> str:
        if self.strategy == "lora":
            return f"finetune_lora_r{self.lora_r}_layer{self.layer}_{self.pool}"
        return f"finetune_full_layer{self.layer}_{self.pool}"

    @property
    def _checkpoint_dir(self) -> str:
        return f"checkpoint_{self.feature_key}"

    def _make_head(self, hidden_size: int) -> nn.Module:
        return nn.Linear(hidden_size, self.num_classes)

    def _prepare_model(self, model):
        """Prepare model for training. Never modifies the original in-place.

        Returns the model to train (PEFT-wrapped or deepcopy).
        """
        if self.strategy == "lora":
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=0.05,
            )
            return get_peft_model(model, lora_config)
        else:
            return copy.deepcopy(model)

    def _teardown_model(self, train_model):
        """Clean up after training."""
        if self.strategy == "lora":
            # merge_and_unload restores the base model
            train_model.merge_and_unload()
        del train_model
        torch.cuda.empty_cache()

    def _save_checkpoint(self, path: Path, train_model, head: nn.Module):
        path.mkdir(parents=True, exist_ok=True)
        if self.strategy == "lora":
            train_model.save_pretrained(path / "lora_adapter")
        else:
            torch.save(train_model.state_dict(), path / "model.pt")
        torch.save(head.state_dict(), path / "head.pt")

    def _load_checkpoint(self, path: Path, model):
        """Load checkpoint into self._inference_model and self._head."""
        hidden_size = model.config.hidden_size
        self._head = self._make_head(hidden_size).to(model.device)
        self._head.load_state_dict(
            torch.load(path / "head.pt", map_location=model.device, weights_only=True)
        )
        if self.strategy == "lora":
            from peft import PeftModel

            self._inference_model = PeftModel.from_pretrained(
                model, path / "lora_adapter"
            )
        else:
            model.load_state_dict(
                torch.load(
                    path / "model.pt", map_location=model.device, weights_only=True
                )
            )
            self._inference_model = model

    def fit(self, texts, labels, model, tokenizer, *, cache_dir, seed=42):
        checkpoint_path = cache_dir / self._checkpoint_dir
        if checkpoint_path.exists():
            print(f"[{self.feature_key}] Checkpoint exists, skipping training")
            self._load_checkpoint(checkpoint_path, model)
            return

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch.manual_seed(seed)
        train_model = self._prepare_model(model)
        hidden_size = train_model.config.hidden_size
        head = self._make_head(hidden_size).to(model.device)

        # Higher LR for the randomly-initialized head
        optimizer = torch.optim.AdamW(
            [
                {"params": train_model.parameters(), "lr": self.lr},
                {"params": head.parameters(), "lr": self.lr * 10},
            ]
        )
        loss_fn = nn.CrossEntropyLoss()

        indices = np.arange(len(texts))
        for epoch in range(self.epochs):
            np.random.seed(seed + epoch)
            np.random.shuffle(indices)
            train_model.train()
            head.train()
            epoch_loss = 0.0
            n_batches = 0

            for start in tqdm(
                range(0, len(texts), self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
            ):
                batch_idx = indices[start : start + self.batch_size]
                batch_texts = [texts[i] for i in batch_idx]
                batch_labels = torch.tensor(
                    labels[batch_idx], dtype=torch.long, device=model.device
                )

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(model.device)

                outputs = train_model(**inputs, output_hidden_states=True)
                pooled = pool_hidden_states(
                    outputs.hidden_states[self.layer],
                    inputs["attention_mask"],
                    self.pool,
                )
                logits = head(pooled)
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            print(f"  Epoch {epoch + 1}/{self.epochs}, avg loss={epoch_loss / n_batches:.4f}")

        self._save_checkpoint(checkpoint_path, train_model, head)
        self._head = head
        self._inference_model = train_model

    def _forward_batched(self, texts, model, tokenizer, cache_dir):
        """Run batched inference, returning probability arrays."""
        checkpoint_path = cache_dir / self._checkpoint_dir
        if self._inference_model is None:
            self._load_checkpoint(checkpoint_path, model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._inference_model.eval()
        self._head.eval()
        all_probs = []

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(model.device)

                outputs = self._inference_model(**inputs, output_hidden_states=True)
                pooled = pool_hidden_states(
                    outputs.hidden_states[self.layer],
                    inputs["attention_mask"],
                    self.pool,
                )
                logits = self._head(pooled)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def predict(self, texts, model, tokenizer, *, cache_dir):
        proba = self._forward_batched(texts, model, tokenizer, cache_dir)
        return proba.argmax(axis=1)

    def predict_proba(self, texts, model, tokenizer, *, cache_dir):
        return self._forward_batched(texts, model, tokenizer, cache_dir)


# --- Concrete probes ---


class FinalLayerLinear(HiddenStateProbe):
    """Last hidden layer, mean pool, logistic regression."""

    layer = -1
    pool = "mean"

    def make_classifier(self, seed: int = 42):
        return LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", random_state=seed, n_jobs=-1,
        )


class FinalLayerLastTokenLinear(HiddenStateProbe):
    """Last hidden layer, last-token pool, logistic regression."""

    layer = -1
    pool = "last"

    def make_classifier(self, seed: int = 42):
        return LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", random_state=seed, n_jobs=-1,
        )


# --- Registry ---

# Values are either probe classes (instantiated with no args) or callables
# that return a Probe instance.
PROBES: dict[str, type[Probe] | callable] = {
    "final_layer_linear": FinalLayerLinear,
    "final_layer_last_token_linear": FinalLayerLastTokenLinear,
    "final_layer_lora": lambda: FinetuneProbe(strategy="lora"),
    "final_layer_full_finetune": lambda: FinetuneProbe(strategy="full"),
}


def get_probe(name: str) -> Probe:
    """Instantiate a probe by name."""
    if name not in PROBES:
        raise ValueError(f"Unknown probe '{name}'. Available: {list_probes()}")
    entry = PROBES[name]
    if isinstance(entry, type):
        return entry()
    return entry()  # callable (lambda)


def list_probes() -> list[str]:
    return list(PROBES.keys())
