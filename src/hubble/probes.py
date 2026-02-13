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
    ) -> np.ndarray:
        """Return predicted probabilities, shape (n_samples, n_classes)."""
        ...


# --- Hidden state probes (frozen features + sklearn) ---


class HiddenStateProbe(Probe):
    """Probe that extracts pooled hidden states from a specific layer.

    Subclasses set `layer` and `pool`.
    Feature extraction results are cached as .npz files.
    """

    layer: int = -1
    cache_path: Path | None = None

    @staticmethod
    def mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

    @staticmethod
    def last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), seq_lengths]

    pool = mean

    @property
    def feature_key(self) -> str:
        return f"layer{self.layer}_pool{self.pool.__name__}"

    def extract_features(
        self,
        model,
        tokenizer,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        cache_path: Path | None = None,
    ) -> np.ndarray:
        """Extract pooled hidden state features for all texts.

        If cache_path is provided, loads from cache if it exists,
        otherwise extracts and saves to cache.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f"[{self.feature_key}] Loading cached hidden states")
                return np.load(cache_path)["hidden_states"]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[{self.feature_key}] Extracting features...")
        results = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc="Extracting hidden states")

        for start in iterator:
            batch_texts = texts[start: start + batch_size]
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
            pooled = self.pool(hidden, mask)
            results.append(pooled.cpu().float().numpy())

        X = np.concatenate(results, axis=0)
        if cache_path is not None:
            np.savez(cache_path, hidden_states=X)
            print(f"[{self.feature_key}] Cached (shape={X.shape})")
        return X

    def _cache_path(self, cache_dir: Path) -> Path:
        return cache_dir / f"hidden_states_{self.feature_key}.npz"

    def fit(self, texts, labels, model, tokenizer, *, cache_dir, seed=42):
        X = self.extract_features(
            model, tokenizer, texts, cache_path=self._cache_path(cache_dir))
        self._clf = LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", random_state=seed, n_jobs=-1,
        )
        self._clf.fit(X, labels)

    def predict(self, texts, model, tokenizer, *, cache_dir):
        X = self.extract_features(
            model, tokenizer, texts, cache_path=self._cache_path(cache_dir))
        return self._clf.predict(X)

    def predict_proba(self, texts, model, tokenizer, *, cache_dir):
        X = self.extract_features(
            model, tokenizer, texts, cache_path=self._cache_path(cache_dir))
        return self._clf.predict_proba(X)


class FinalLayerLinear(HiddenStateProbe):
    """Last hidden layer, mean pool, logistic regression."""

    layer = -1
    pool = HiddenStateProbe.mean


class FinalLayerLastTokenLinear(HiddenStateProbe):
    """Last hidden layer, last-token pool, logistic regression."""

    layer = -1
    pool = HiddenStateProbe.last


# --- Finetuning probes (end-to-end with classification head) ---


class _ProbeModel(nn.Module):
    """Wraps base model + classification head so HF Trainer can train them jointly."""

    def __init__(self, model, head, layer, pool_fn):
        super().__init__()
        self.model = model
        self.head = head
        self.layer = layer
        self.pool_fn = pool_fn

    @property
    def config(self):
        return self.model.config

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled = self.pool_fn(outputs.hidden_states[self.layer], attention_mask)
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class _ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in self.encodings.keys()}
        item["labels"] = int(self.labels[idx])
        return item


class FinetuneProbe(Probe, ABC):
    """Base for probes that attach a classification head and train end-to-end.

    Subclasses implement _prepare_model, _save_checkpoint, _load_checkpoint.
    Training uses HuggingFace Trainer.
    """

    def __init__(
        self,
        layer: int = -1,
        pool=HiddenStateProbe.mean,
        lr: float = 1e-4,
        epochs: int = 3,
        batch_size: int = 16,
        num_classes: int = 2,
    ):
        self.layer = layer
        self.pool = pool
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

        self._head: nn.Module | None = None
        self._inference_model = None

    @property
    def _checkpoint_dir(self) -> str:
        return f"checkpoint_{self.feature_key}"

    def _make_head(self, hidden_size: int) -> nn.Module:
        return nn.Linear(hidden_size, self.num_classes)

    @abstractmethod
    def _prepare_model(self, model):
        """Return a trainable copy/wrapper of the model."""
        ...

    @abstractmethod
    def _save_checkpoint(self, path: Path, train_model, head: nn.Module):
        ...

    @abstractmethod
    def _load_checkpoint(self, path: Path, model):
        """Populate self._inference_model and self._head from a checkpoint."""
        ...

    def fit(self, texts, labels, model, tokenizer, *, cache_dir, seed=42):
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

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
        wrapper = _ProbeModel(train_model, head, self.layer, self.pool)

        encodings = tokenizer(list(texts), truncation=True, max_length=512)
        dataset = _ProbeDataset(encodings, labels)

        # Higher LR for the randomly-initialized head
        optimizer = torch.optim.AdamW([
            {"params": train_model.parameters(), "lr": self.lr},
            {"params": head.parameters(), "lr": self.lr * 10},
        ])

        training_args = TrainingArguments(
            output_dir=str(checkpoint_path / "_trainer"),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            seed=seed,
            logging_strategy="epoch",
            save_strategy="no",
            report_to="none",
        )
        trainer = Trainer(
            model=wrapper,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
            optimizers=(optimizer, None),
        )
        trainer.train()

        self._save_checkpoint(checkpoint_path, train_model, head)
        self._head = head
        self._inference_model = train_model

    def predict_proba(self, texts, model, tokenizer, *, cache_dir):
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
                batch_texts = texts[start: start + self.batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(model.device)

                outputs = self._inference_model(
                    **inputs, output_hidden_states=True)
                pooled = self.pool(
                    outputs.hidden_states[self.layer],
                    inputs["attention_mask"],
                )
                logits = self._head(pooled)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def predict(self, texts, model, tokenizer, *, cache_dir):
        return self.predict_proba(
            texts, model, tokenizer, cache_dir=cache_dir).argmax(axis=1)


class FullFinetuneProbe(FinetuneProbe):
    """Full finetuning: deepcopies the model.

    NOTE: roughly doubles GPU memory (e.g. ~4GB extra for a 1B model in fp32).
    """

    @property
    def feature_key(self) -> str:
        return f"finetune_full_layer{self.layer}_{self.pool.__name__}"

    def _prepare_model(self, model):
        return copy.deepcopy(model)

    def _save_checkpoint(self, path: Path, train_model, head: nn.Module):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(train_model.state_dict(), path / "model.pt")
        torch.save(head.state_dict(), path / "head.pt")

    def _load_checkpoint(self, path: Path, model):
        hidden_size = model.config.hidden_size
        self._head = self._make_head(hidden_size).to(model.device)
        self._head.load_state_dict(
            torch.load(path / "head.pt", map_location=model.device,
                       weights_only=True)
        )
        model.load_state_dict(
            torch.load(
                path / "model.pt", map_location=model.device, weights_only=True
            )
        )
        self._inference_model = model


class LoRAFinetuneProbe(FinetuneProbe):
    """LoRA finetuning via PEFT."""

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]

    @property
    def feature_key(self) -> str:
        return f"finetune_lora_r{self.lora_r}_layer{self.layer}_{self.pool.__name__}"

    def _prepare_model(self, model):
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=0.05,
        )
        return get_peft_model(model, lora_config)

    def _save_checkpoint(self, path: Path, train_model, head: nn.Module):
        path.mkdir(parents=True, exist_ok=True)
        train_model.save_pretrained(path / "lora_adapter")
        torch.save(head.state_dict(), path / "head.pt")

    def _load_checkpoint(self, path: Path, model):
        from peft import PeftModel

        hidden_size = model.config.hidden_size
        self._head = self._make_head(hidden_size).to(model.device)
        self._head.load_state_dict(
            torch.load(path / "head.pt", map_location=model.device,
                       weights_only=True)
        )
        self._inference_model = PeftModel.from_pretrained(
            model, path / "lora_adapter"
        )


# --- Registry ---

PROBES: dict[str, type[Probe] | callable] = {
    "final_layer_linear": FinalLayerLinear,
    "final_layer_last_token_linear": FinalLayerLastTokenLinear,
    "final_layer_lora": LoRAFinetuneProbe,
    "final_layer_full_finetune": FullFinetuneProbe,
}
