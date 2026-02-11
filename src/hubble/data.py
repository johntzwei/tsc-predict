"""Data loading for Hubble perturbation datasets."""

import json
import pandas as pd
from datasets import load_dataset


def load_winogrande_perturbations(format: str = "infill") -> pd.DataFrame:
    """Load Hubble's WinoGrande perturbation dataset.

    Args:
        format: "infill" or "mcq"

    Returns:
        DataFrame with columns: orig_idx, sentence, option1, option2, answer,
        duplicates, paired_orig_idx, text, split
    """
    ds_name = f"allegrolab/testset_winogrande-{format}"
    ds = load_dataset(ds_name)

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            paired = json.loads(meta["paired_example"]) if meta["paired_example"] else None
            rows.append({
                "orig_idx": meta["orig_idx"],
                "sentence": meta["sentence"],
                "option1": meta["option1"],
                "option2": meta["option2"],
                "answer": int(meta["answer"]),
                "duplicates": example["duplicates"],
                "paired_orig_idx": paired["orig_idx"] if paired else None,
                "text": example["text"],
                "split": split_name,
                "format": format,
            })

    return pd.DataFrame(rows)
