"""Hidden state extraction utilities for probing experiments."""

import numpy as np
import torch
from tqdm import tqdm


def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    layer: int = -1,
    pool: str = "mean",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract pooled hidden states from a causal LM.

    Args:
        model: HuggingFace CausalLM (already on device, in eval mode).
        tokenizer: Corresponding tokenizer.
        texts: List of raw text strings.
        batch_size: Examples per forward pass.
        layer: Which hidden layer to extract (-1 = last transformer layer).
        pool: "mean" (mean over non-padding tokens) or "last" (last non-padding token).
        show_progress: Show tqdm progress bar.

    Returns:
        np.ndarray of shape (len(texts), hidden_dim), dtype float32.
    """
    # Ensure pad token is set (common issue with Llama-based models)
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

        # hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer, index -1 = last transformer layer
        hidden = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)
        mask = inputs["attention_mask"]  # (batch, seq_len)

        if pool == "mean":
            # Mean-pool over non-padding tokens
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        elif pool == "last":
            # Last non-padding token: find index of last 1 in attention_mask
            seq_lengths = mask.sum(dim=1) - 1  # (batch,)
            pooled = hidden[torch.arange(hidden.size(0)), seq_lengths]
        else:
            raise ValueError(f"Unknown pool method: {pool}")

        results.append(pooled.cpu().float().numpy())

    return np.concatenate(results, axis=0)
