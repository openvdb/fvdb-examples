# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Script to compute average deviation between two minimal_inference_stats.json files.
Usage: python compute_difference.py file1.json file2.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np


def load_stats_file(filepath: str, logger: logging.Logger) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and parse a minimal_inference_stats.json file.

    Args:
        filepath: Path to the JSON file to load.
        logger: Logger instance for error reporting here.

    Returns:
        Tuple of (per_sample_stats, global_stats) containing the parsed JSON data.
        If the file has old format (just a list), returns (data, empty_dict).

    Raises:
        SystemExit: If file is not found or contains invalid JSON.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with global_stats and per_sample_stats)
        if isinstance(data, list):
            # Old format - just a list of per-sample stats
            logger.info(f"Loading old format file: {filepath}")
            return data, {}
        elif isinstance(data, dict) and "per_sample_stats" in data:
            # New format - structured with global and per-sample stats
            logger.info(f"Loading new format file: {filepath}")
            global_stats = data.get("global_stats", {})
            per_sample_stats = data.get("per_sample_stats", [])
            return per_sample_stats, global_stats
        else:
            logger.error(f"Unexpected JSON structure in file '{filepath}'")
            sys.exit(1)

    except FileNotFoundError:
        logger.error(f"File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file '{filepath}': {e}")
        sys.exit(1)


def compute_deviations(
    stats1: List[Dict[str, Any]], stats2: List[Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """Compute deviations between corresponding entries in two stats files.

    Args:
        stats1: List of dictionaries from the first stats file.
        stats2: List of dictionaries from the second stats file.
        logger: Logger instance for warning messages.

    Returns:
        Dictionary containing average deviations for each numerical field,
        with both absolute and relative differences.
    """
    if len(stats1) != len(stats2):
        logger.warning(f"Files have different numbers of entries ({len(stats1)} vs {len(stats2)})")
        min_len = min(len(stats1), len(stats2))
        stats1 = stats1[:min_len]
        stats2 = stats2[:min_len]

    deviations = {
        "num_points": {"absolute": [], "relative": []},
        "output_feats_sum": {"absolute": [], "relative": []},
        "output_feats_last_element": {"absolute": [], "relative": []},
        "loss": {"absolute": [], "relative": []},
    }

    for i, (entry1, entry2) in enumerate(zip(stats1, stats2)):
        # Compute absolute and relative differences for numerical fields
        for field in deviations.keys():
            if field in entry1 and field in entry2:
                val1 = entry1[field]
                val2 = entry2[field]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Absolute difference
                    abs_deviation = abs(val1 - val2)
                    deviations[field]["absolute"].append(abs_deviation)

                    # Relative difference (avoid division by zero)
                    if abs(val1) > 0 and abs(val2) > 0:
                        rel_deviation = abs_deviation / max(abs(val1), abs(val2))
                    else:
                        rel_deviation = 0.0
                    deviations[field]["relative"].append(rel_deviation)

                else:
                    logger.warning(f"Non-numerical value found in field '{field}' at index {i}")

    # Compute average deviations
    avg_deviations = {}
    for field, diff_types in deviations.items():
        avg_deviations[field] = {}
        for diff_type, values in diff_types.items():
            if values:
                avg_deviations[field][diff_type] = np.mean(values)
            else:
                avg_deviations[field][diff_type] = 0.0

    return avg_deviations


def compute_global_deviations(
    global_stats1: Dict[str, Any], global_stats2: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """Compute deviations between global statistics from two files.

    Args:
        global_stats1: Global stats dictionary from the first file.
        global_stats2: Global stats dictionary from the second file.
        logger: Logger instance for warning messages.

    Returns:
        Dictionary containing deviations for global fields.
    """
    global_deviations = {}

    # Compare gradient vectors if present
    if "first_module_grad_last16" in global_stats1 and "first_module_grad_last16" in global_stats2:

        grad1 = global_stats1["first_module_grad_last16"]
        grad2 = global_stats2["first_module_grad_last16"]

        if isinstance(grad1, list) and isinstance(grad2, list) and len(grad1) == len(grad2):
            # Compute L2 norm of the difference vector
            diff_vec = [v1 - v2 for v1, v2 in zip(grad1, grad2)]
            abs_deviation = np.sqrt(sum(d * d for d in diff_vec))

            # Relative difference using L2 norms
            norm1 = np.sqrt(sum(v * v for v in grad1))
            norm2 = np.sqrt(sum(v * v for v in grad2))
            if norm1 > 0 and norm2 > 0:
                rel_deviation = abs_deviation / max(norm1, norm2)
            else:
                rel_deviation = 0.0

            global_deviations["first_module_grad_last16"] = {"absolute": abs_deviation, "relative": rel_deviation}

            logger.info(
                f"Global gradient deviation: absolute={abs_deviation:.6f}, relative={rel_deviation:.6f} ({rel_deviation*100:.2f}%)"
            )
        else:
            logger.warning("Gradient list format mismatch in global stats")

    # Compare other numerical global fields
    numerical_fields = ["total_samples", "batch_size"]
    for field in numerical_fields:
        if field in global_stats1 and field in global_stats2:
            val1 = global_stats1[field]
            val2 = global_stats2[field]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                abs_deviation = abs(val1 - val2)
                if abs(val1) > 0 and abs(val2) > 0:
                    rel_deviation = abs_deviation / max(abs(val1), abs(val2))
                else:
                    rel_deviation = 0.0

                global_deviations[field] = {"absolute": abs_deviation, "relative": rel_deviation}

    return global_deviations


def main():
    parser = argparse.ArgumentParser(
        description="Compute average deviation between two minimal_inference_stats.json files"
    )
    parser.add_argument("--stats_path_1", help="Path to first minimal_inference_stats.json file")
    parser.add_argument("--stats_path_2", help="Path to second minimal_inference_stats.json file")

    args = parser.parse_args()

    folder_for_stats_path = os.path.dirname(os.path.abspath(args.stats_path_1))
    logging_file_path = os.path.join(folder_for_stats_path, "compute_difference.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logging_file_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load both files
    stats1, global_stats1 = load_stats_file(args.stats_path_1, logger)
    stats2, global_stats2 = load_stats_file(args.stats_path_2, logger)

    logger.info(f"File 1 has {len(stats1)} per-sample entries")
    logger.info(f"File 2 has {len(stats2)} per-sample entries")

    # Compute per-sample deviations
    avg_deviations = compute_deviations(stats1, stats2, logger)

    # Compute global deviations if both files have global stats
    global_deviations = {}
    if global_stats1 and global_stats2:
        global_deviations = compute_global_deviations(global_stats1, global_stats2, logger)

    # Print results
    logger.info("\nPer-Sample Average Deviations:")
    logger.info("=" * 50)
    for field, diff_types in avg_deviations.items():
        logger.info(f"{field}:")
        for diff_type, avg_dev in diff_types.items():
            if diff_type == "relative":
                logger.info(f"  {diff_type:10s}: {avg_dev:.6f} ({avg_dev*100:.2f}%)")
            else:
                logger.info(f"  {diff_type:10s}: {avg_dev:.6f}")

    if global_deviations:
        logger.info("\nGlobal Statistics Deviations:")
        logger.info("=" * 50)
        for field, diff_types in global_deviations.items():
            logger.info(f"{field}:")
            for diff_type, dev in diff_types.items():
                if diff_type == "relative":
                    logger.info(f"  {diff_type:10s}: {dev:.6f} ({dev*100:.2f}%)")
                else:
                    logger.info(f"  {diff_type:10s}: {dev:.6f}")

    # Compute overall average deviations for per-sample stats
    overall_absolute = np.mean([diff_types["absolute"] for diff_types in avg_deviations.values()])
    overall_relative = np.mean([diff_types["relative"] for diff_types in avg_deviations.values()])
    logger.info("=" * 50)

    logger.info("\nOverall Per-Sample Averages:")
    logger.info(f"Absolute: {overall_absolute:.6f}")
    logger.info(f"Relative: {overall_relative:.6f} ({overall_relative*100:.2f}%)")


if __name__ == "__main__":
    main()

# scannet_samples_large.json
# python compute_difference.py --stats_path_1 data/scannet_samples_large_output.json --stats_path_2 data/scannet_samples_large_output_gt.json

# scannet_samples_small.json
# python compute_difference.py --stats_path_1 data/scannet_samples_small_output.json --stats_path_2 data/scannet_samples_small_output_gt.json
