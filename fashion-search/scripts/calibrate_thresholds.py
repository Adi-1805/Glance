"""
calibrate_thresholds.py

Calibrates the zero-shot attribute classifier's similarity threshold using
Fashionpedia's ground-truth (garment, color) annotations as validation labels.

Usage:
    python -m scripts.calibrate_thresholds \
        --fashionpedia-ann data/metadata/fashionpedia_annotations.json \
        --image-dir data/raw \
        --thresholds 0.18 0.20 0.22 0.24 0.26 0.28 0.30 \
        --output evaluation_report_calibration.md

What this does:
    1. Loads Fashionpedia's ground-truth per-image (garment, color) pairs.
    2. Maps Fashionpedia's fine-grained taxonomy onto our project vocab
       (indexer/vocab.py categories/colors) via CATEGORY_MAP / COLOR_MAP below.
    3. Re-runs the zero-shot attribute classifier at each candidate threshold
       (reusing indexer/attributes.py, NOT a reimplementation) so results are
       apples-to-apples with what actually gets indexed.
    4. Computes precision / recall / F1 per threshold and reports the best one.

NOTE: This only validates against images that came from Fashionpedia (i.e. have
ground-truth labels). Unsplash-sourced context/environment images have no
Fashionpedia labels and are excluded from calibration -- the chosen threshold
is simply applied to them at indexing time, same as everything else.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Reuse the actual project modules so calibration reflects real indexing behavior,
# not a reimplementation that could silently drift from it.
from indexer.attributes import extract_attributes  # TODO: confirm this is the real entry point in your attributes.py
from indexer.vocab import GARMENT_TYPES, COLORS       # TODO: confirm these names match your vocab.py


# ---------------------------------------------------------------------------
# 1. Taxonomy mapping: Fashionpedia -> project vocab
# ---------------------------------------------------------------------------
# Fashionpedia's category ontology (46 apparel categories) and attribute
# ontology (294 attributes, includes colors) are both more granular than our
# vocab. Map Fashionpedia's names onto our coarser vocab here.
#
# TODO: fill this out against your actual indexer/vocab.py vocabulary and the
# actual Fashionpedia category list in your annotation file. This is a
# starting point based on Fashionpedia's public category set -- verify names
# match your downloaded annotation file exactly (they are case-sensitive).

CATEGORY_MAP = {
    "coat": "coat",
    "jacket": "jacket",
    "cardigan": "sweater",
    "sweater": "sweater",
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "t-shirt",
    "dress": "dress",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat/raincoat": "raincoat",   # verify actual Fashionpedia label string
    "tie": "tie",
    "vest": "vest",
    "hood": "hoodie",
}

COLOR_MAP = {
    # Fashionpedia color attribute label -> project vocab color
    "yellow": "yellow",
    "red": "red",
    "blue": "blue",
    "white": "white",
    "black": "black",
    "grey": "gray",
    "gray": "gray",
    "green": "green",
    "orange": "orange",
    "brown": "brown",
    "pink": "pink",
    "purple": "purple",
    "beige": "beige",
    "navy": "navy",
}


def map_ground_truth_pair(fp_category: str, fp_color: str):
    """Map a single Fashionpedia (category, color) pair to project vocab.
    Returns None if either side has no mapping (i.e. out of scope for us)."""
    garment = CATEGORY_MAP.get(fp_category.strip().lower())
    color = COLOR_MAP.get(fp_color.strip().lower())
    if garment is None or color is None:
        return None
    return (garment, color)


# ---------------------------------------------------------------------------
# 2. Load Fashionpedia ground truth
# ---------------------------------------------------------------------------

def load_fashionpedia_ground_truth(ann_path: Path, image_dir: Path):
    """
    Parses a Fashionpedia-style COCO annotation file into:
        { image_filename: set of (garment, color) tuples }

    Expected input schema (standard Fashionpedia COCO export):
        {
          "images": [{"id": int, "file_name": str}, ...],
          "categories": [{"id": int, "name": str}, ...],
          "attributes": [{"id": int, "name": str}, ...],
          "annotations": [
              {"image_id": int, "category_id": int, "attribute_ids": [int, ...]},
              ...
          ]
        }

    TODO: if your downloaded annotation file has a different structure
    (e.g. already flattened, or from a HuggingFace `datasets` load rather
    than raw COCO JSON), adjust this loader accordingly. The rest of the
    script only depends on the returned dict shape, not this file format.
    """
    with open(ann_path, "r") as f:
        data = json.load(f)

    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    category_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    attribute_id_to_name = {a["id"]: a["name"] for a in data.get("attributes", [])}

    ground_truth = defaultdict(set)

    for ann in data["annotations"]:
        image_name = image_id_to_name.get(ann["image_id"])
        if image_name is None:
            continue

        # Only keep images we actually have on disk (i.e. in our indexed subset)
        if not (image_dir / image_name).exists():
            continue

        fp_category = category_id_to_name.get(ann["category_id"], "")
        fp_attr_ids = ann.get("attribute_ids", [])
        fp_colors = [
            attribute_id_to_name[aid]
            for aid in fp_attr_ids
            if aid in attribute_id_to_name and attribute_id_to_name[aid].lower() in COLOR_MAP
        ]

        for fp_color in fp_colors:
            mapped = map_ground_truth_pair(fp_category, fp_color)
            if mapped is not None:
                ground_truth[image_name].add(mapped)

    return ground_truth


# ---------------------------------------------------------------------------
# 3. Run the real classifier at a given threshold
# ---------------------------------------------------------------------------

def predict_attributes_for_image(image_path: Path, threshold: float):
    """
    Calls the actual project attribute extractor with a given threshold and
    returns a set of (garment, color) tuples, same shape as ground truth.

    TODO: adjust this call signature to match indexer/attributes.py's real
    function signature. This assumes extract_attributes(image_path, threshold)
    returns a list of dicts like [{"garment": "raincoat", "color": "yellow"}, ...],
    matching the format visible in your terminal output.
    """
    predicted = extract_attributes(str(image_path), threshold=threshold)
    return {(p["garment"], p["color"]) for p in predicted}


# ---------------------------------------------------------------------------
# 4. Precision / recall / F1 at each threshold
# ---------------------------------------------------------------------------

def evaluate_threshold(ground_truth: dict, image_dir: Path, threshold: float):
    tp = fp = fn = 0

    for image_name, gt_pairs in ground_truth.items():
        image_path = image_dir / image_name
        pred_pairs = predict_attributes_for_image(image_path, threshold)

        tp += len(gt_pairs & pred_pairs)
        fp += len(pred_pairs - gt_pairs)
        fn += len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate attribute classifier threshold using Fashionpedia ground truth.")
    parser.add_argument("--fashionpedia-ann", type=Path, required=True, help="Path to Fashionpedia COCO-style annotation JSON")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing the indexed images")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30],
                         help="Candidate similarity thresholds to sweep")
    parser.add_argument("--output", type=Path, default=Path("calibration_report.md"), help="Where to write the results report")
    args = parser.parse_args()

    print(f"Loading Fashionpedia ground truth from {args.fashionpedia_ann} ...")
    ground_truth = load_fashionpedia_ground_truth(args.fashionpedia_ann, args.image_dir)
    print(f"Loaded ground truth for {len(ground_truth)} images.")

    if len(ground_truth) == 0:
        print("WARNING: no ground-truth images matched files on disk. Check --image-dir "
              "and that CATEGORY_MAP / COLOR_MAP names match your annotation file's actual labels.")
        return

    results = []
    for threshold in args.thresholds:
        print(f"Evaluating threshold={threshold} ...")
        result = evaluate_threshold(ground_truth, args.image_dir, threshold)
        results.append(result)
        print(f"  precision={result['precision']}  recall={result['recall']}  f1={result['f1']}")

    best = max(results, key=lambda r: r["f1"])

    # Write markdown report
    lines = [
        "# Attribute Classifier Threshold Calibration",
        "",
        f"Validated against {len(ground_truth)} Fashionpedia-labeled images.",
        "",
        "| Threshold | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['threshold']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['tp']} | {r['fp']} | {r['fn']} |")

    lines += [
        "",
        f"**Best threshold by F1: {best['threshold']}** "
        f"(precision={best['precision']}, recall={best['recall']}, f1={best['f1']})",
        "",
        "Apply this threshold in `indexer/attributes.py` and re-run the indexer "
        "on the full dataset (including unlabeled Unsplash images, which were "
        "excluded from this calibration since Fashionpedia has no ground truth for them).",
    ]

    args.output.write_text("\n".join(lines))
    print(f"\nReport written to {args.output}")
    print(f"Best threshold: {best['threshold']} (F1={best['f1']})")


if __name__ == "__main__":
    main()