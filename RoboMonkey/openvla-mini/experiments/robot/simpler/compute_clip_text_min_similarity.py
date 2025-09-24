#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import numpy as np
import clip  # OpenAI CLIP


def load_instructions(json_path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    if "instructions" not in data:
        raise ValueError("JSON missing 'instructions' field")
    return data["instructions"]


def batch_encode_texts(model, preprocess, device, texts: List[str], batch_size: int = 64) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            feats = model.encode_text(tokens)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            embeddings.append(feats)
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 512, device=device)


def cosine_similarity_matrix(embeds: torch.Tensor) -> torch.Tensor:
    # embeds expected normalized
    return embeds @ embeds.t()


def compute_min_similarities_for_task(
    model, preprocess, device, task_name: str, original: str, rephrases: List[str], include_task_label: bool
) -> Dict[str, float]:
    text_list: List[str] = []
    names: List[str] = []

    if include_task_label:
        text_list.append(task_name)
        names.append("task_label")

    if original:
        text_list.append(original)
        names.append("original")

    for idx, r in enumerate(rephrases or []):
        text_list.append(r)
        names.append(f"rephrase_{idx+1}")

    if len(text_list) < 2:
        return {"min_all_pairs": float("nan"), "min_vs_original": float("nan"), "count": len(text_list)}

    embeds = batch_encode_texts(model, preprocess, device, text_list)
    sims = cosine_similarity_matrix(embeds).cpu().numpy()

    # Mask diagonal
    n = sims.shape[0]
    mask = np.ones_like(sims, dtype=bool)
    np.fill_diagonal(mask, False)
    min_all_pairs = float(np.min(sims[mask])) if np.any(mask) else float("nan")

    # Min vs original only (if exists)
    min_vs_original = float("nan")
    if "original" in names:
        orig_idx = names.index("original")
        # all non-diagonal similarities for original
        other_idxs = [i for i in range(n) if i != orig_idx]
        if other_idxs:
            min_vs_original = float(np.min(sims[orig_idx, other_idxs]))

    return {
        "min_all_pairs": min_all_pairs,
        "min_vs_original": min_vs_original,
        "count": len(text_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute minimal CLIP text cosine similarity among originals/rephrases per task.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="./simpler_rephrased_final_eval.json",
        help="Path to JSON with 'instructions' mapping",
    )
    parser.add_argument(
        "--include_task_label",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Include the task key itself as an additional text to compare (default: True)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L/14",
        help="CLIP text model variant (e.g., 'ViT-B/32', 'ViT-L/14')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="",
        help="Optional CSV output path (per-task rows)",
    )
    args = parser.parse_args()

    instructions = load_instructions(args.json_path)

    device = args.device
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # Compute per-task
    rows: List[Tuple[str, float, float, int]] = []
    overall_mins: List[float] = []
    overall_mins_vs_original: List[float] = []

    for task_name, entry in instructions.items():
        original = entry.get("original", "")
        rephrases = entry.get("rephrases", [])

        result = compute_min_similarities_for_task(
            model, preprocess, device, task_name, original, rephrases, args.include_task_label
        )
        rows.append((
            task_name,
            result["min_all_pairs"],
            result["min_vs_original"],
            result["count"],
        ))
        if not np.isnan(result["min_all_pairs"]):
            overall_mins.append(result["min_all_pairs"])
        if not np.isnan(result["min_vs_original"]):
            overall_mins_vs_original.append(result["min_vs_original"])

    # Print summary
    print("\nPer-task minimal cosine similarities (CLIP text):")
    for task_name, min_all, min_vs_orig, count in rows:
        print(f"- {task_name}")
        print(f"  texts: {count}")
        print(f"  min_all_pairs: {min_all:.4f}" if np.isfinite(min_all) else "  min_all_pairs: nan")
        print(f"  min_vs_original: {min_vs_orig:.4f}" if np.isfinite(min_vs_orig) else "  min_vs_original: nan")

    if overall_mins:
        print("\nOverall stats:")
        print(f"  global_min_all_pairs: {float(np.min(overall_mins)):.4f}")
        print(f"  mean_min_all_pairs: {float(np.mean(overall_mins)):.4f}")
    if overall_mins_vs_original:
        print(f"  global_min_vs_original: {float(np.min(overall_mins_vs_original)):.4f}")
        print(f"  mean_min_vs_original: {float(np.mean(overall_mins_vs_original)):.4f}")

    # Optional CSV
    if args.csv_out:
        import csv

        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "min_all_pairs", "min_vs_original", "num_texts"])
            for task_name, min_all, min_vs_orig, count in rows:
                writer.writerow([task_name, min_all, min_vs_orig, count])
        print(f"\nWrote CSV to: {args.csv_out}")


if __name__ == "__main__":
    main()


