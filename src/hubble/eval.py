"""Inference utilities for evaluating Hubble models on WinoGrande-style tasks."""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_id: str, device: str = "cuda", dtype: torch.dtype | None = None):
    """Load a Hubble HF model and tokenizer.

    Args:
        dtype: Override precision. If None, auto-selects: fp32 for 1B models, bf16 for 8B.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if dtype is None:
        dtype = torch.bfloat16 if "8b" in model_id.lower() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def compute_suffix_logprob(model, tokenizer, prefix: str, suffix: str) -> float:
    """Compute average per-token log-prob of suffix conditioned on prefix.

    This is the standard WinoGrande evaluation: score each option by how likely
    the shared suffix is given prefix+option.

    Returns:
        Average log-prob per token (negative; higher = more likely).
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
    # Suffix tokens = everything after the prefix
    suffix_len = len(full_ids) - len(prefix_ids)
    if suffix_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)

    # Log-probs of each token, shifted: logits[t] predicts token[t+1]
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    target_ids = input_ids[0, 1:]

    # Gather log-probs for actual tokens
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    # Average over suffix tokens only
    suffix_log_probs = token_log_probs[-suffix_len:]
    return suffix_log_probs.mean().item()


def evaluate_winogrande_example(
    model, tokenizer, sentence: str, option1: str, option2: str
) -> dict:
    """Evaluate a single WinoGrande example.

    WinoGrande format: "sentence with _ placeholder"
    Evaluation: for each option, form prefix = sentence_before_blank + option,
    suffix = sentence_after_blank. Score by log-prob of suffix given prefix.

    Returns dict with logprob_option1, logprob_option2.
    """
    blank_idx = sentence.index("_")
    before_blank = sentence[:blank_idx]
    after_blank = sentence[blank_idx + 1:]  # includes leading space usually

    prefix1 = before_blank + option1
    prefix2 = before_blank + option2

    lp1 = compute_suffix_logprob(model, tokenizer, prefix1, after_blank)
    lp2 = compute_suffix_logprob(model, tokenizer, prefix2, after_blank)

    return {"logprob_option1": lp1, "logprob_option2": lp2}


def evaluate_winogrande_df(model, tokenizer, df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Evaluate all WinoGrande examples in a DataFrame.

    Adds columns: logprob_option1_{label}, logprob_option2_{label},
    acc_{label}, confidence_{label}.
    """
    lp1s, lp2s = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {model_label}"):
        result = evaluate_winogrande_example(
            model, tokenizer, row["sentence"], row["option1"], row["option2"]
        )
        lp1s.append(result["logprob_option1"])
        lp2s.append(result["logprob_option2"])

    df = df.copy()
    df[f"logprob_option1_{model_label}"] = lp1s
    df[f"logprob_option2_{model_label}"] = lp2s

    # Predicted answer: option with higher log-prob
    pred = (pd.Series(lp2s) > pd.Series(lp1s)).astype(int) + 1  # 1 or 2
    df[f"acc_{model_label}"] = (pred.values == df["answer"].values).astype(int)

    # Confidence: softmax of the two log-probs for the correct answer
    import numpy as np
    lp1_arr = np.array(lp1s)
    lp2_arr = np.array(lp2s)
    # Probability of option 1
    p1 = np.exp(lp1_arr) / (np.exp(lp1_arr) + np.exp(lp2_arr))
    p2 = 1 - p1
    correct_confidence = np.where(df["answer"].values == 1, p1, p2)
    df[f"confidence_{model_label}"] = correct_confidence

    return df
