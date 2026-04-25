import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PRESET_MAPPINGS = {
    "lip6": {
        1: [1, 2, 4, 11, 13],    # head
        2: [5, 6, 7, 10, 12],    # torso
        3: [14],                 # left arm
        4: [15],                 # right arm
        5: [16, 18],             # left leg + left shoe
        6: [17, 19],             # right leg + right shoe
    },
    "atr6": {
        1: [1, 2, 3, 11, 17],    # head
        2: [4, 5, 6, 7, 8],      # torso
        3: [14],                 # left arm
        4: [15],                 # right arm
        5: [12, 9],              # left leg + left shoe
        6: [13, 10],             # right leg + right shoe
    },
    "fashn6": {
        1: [1, 2, 9, 11, 17],    # head: face, hair, hat, glasses, jewelry
        2: [3, 4, 5, 6, 7, 10, 16],  # torso + clothing around torso
        3: [12],                 # arms
        4: [13],                 # hands
        5: [14],                 # legs
        6: [15],                 # feet
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert human parsing labels into grouped semantic masks.")
    parser.add_argument("--input-root", required=True, help="Root directory containing raw parsing maps.")
    parser.add_argument("--output-root", required=True, help="Root directory to save grouped masks.")
    parser.add_argument("--preset", choices=sorted(PRESET_MAPPINGS.keys()), default="lip6")
    parser.add_argument("--suffix", default=".png", help="Parsing file suffix to scan.")
    return parser.parse_args()


def convert_mask(mask_array, mapping):
    grouped_mask = np.zeros_like(mask_array, dtype=np.uint8)
    for group_id, source_ids in mapping.items():
        source_mask = np.isin(mask_array, source_ids)
        grouped_mask[source_mask] = group_id
    return grouped_mask


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    mapping = PRESET_MAPPINGS[args.preset]

    mask_paths = sorted(input_root.rglob(f"*{args.suffix}"))
    if not mask_paths:
        raise FileNotFoundError(f"No parsing masks with suffix {args.suffix} were found under {input_root}")

    for mask_path in mask_paths:
        relative_path = mask_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mask_array = np.array(Image.open(mask_path), dtype=np.uint8)
        grouped_mask = convert_mask(mask_array, mapping)
        Image.fromarray(grouped_mask, mode='L').save(output_path)

    print(f"Converted {len(mask_paths)} masks from {input_root} to {output_root} using preset {args.preset}.")


if __name__ == "__main__":
    main()
