import argparse
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import torch
from utils.results import load_results

parser = argparse.ArgumentParser(
    description="Fuse results from multiple modalities",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("uni_modality_results", metavar="RESULTS_PT", nargs="+", type=Path)
parser.add_argument("fused_results", metavar="FUSED_RESULTS_PT", type=Path)
parser.add_argument(
    "--force", action="store_true", help="Overwrite exisiting fused_results if present."
)


def main(args):
    if args.fused_results.exists() and not args.force:
        print(f"{args.fused_results} already exists, use --force to overwrite.")
        sys.exit(-1)
    all_results: List[Dict[str, Any]] = [
        load_results(result_path) for result_path in args.uni_modality_results
    ]
    all_results = canonicalise_results_ordering(all_results)
    narration_ids = all_results[0]["narration_id"]
    check_narration_ids_match_across_results(all_results, narration_ids)
    fused_results = fuse(all_results)
    torch.save(decollate(fused_results), args.fused_results)


def decollate(collated_results: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    keys = collated_results.keys()
    n_entries = collated_results[next(iter(keys))].shape[0]

    return [{k: collated_results[k][i] for k in keys} for i in range(n_entries)]


def fuse(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys_to_fuse = {"verb_output", "noun_output"}
    first_result = all_results[0]
    keys_to_copy = set(first_result.keys()) - keys_to_fuse

    fused_results = dict()
    for key in keys_to_fuse:
        fused_results[key] = np.stack([r[key] for r in all_results], axis=0).mean(
            axis=0
        )

    for key in keys_to_copy:
        fused_results[key] = first_result[key]

    return fused_results


def canonicalise_results_ordering(
    all_results: List[Dict[str, np.ndarray]]
) -> List[Dict[str, np.ndarray]]:
    new_all_results = []
    for i in range(len(all_results)):
        results = all_results[i]
        narration_ids = results["narration_id"]
        sort_idxs = np.argsort(narration_ids)
        result = {k: vs[sort_idxs] for k, vs in results.items()}
        new_all_results.append(result)
    return new_all_results


def check_narration_ids_match_across_results(
    all_results: List[Dict[str, np.ndarray]], narration_ids: np.ndarray
) -> None:
    for other_results in all_results[1:]:
        if not (other_results["narration_id"] == narration_ids).all():
            raise ValueError("Narration IDs don't match across results.")


if __name__ == "__main__":
    main(parser.parse_args())
