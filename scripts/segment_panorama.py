"""Create instance masks for a panorama with SAM 2 from box prompts.

Pano3DComposer takes instance masks as input; any segmenter works. This helper
prompts SAM 2 with one box per object (pixel coordinates of the panorama):

    boxes.json: {"bed_1": [x0, y0, x1, y1], "armchair_1": [...], ...}

A box with x0 > x1 wraps around the left/right seam of the panorama; it is segmented
on a copy rolled so that the object is whole, and the mask is rolled back.

    python scripts/segment_panorama.py --pano demo/lythwood_room/panoramas/lythwood_room.jpg \
        --boxes demo/lythwood_room/boxes.json --out demo/lythwood_room/masks/lythwood_room

Requires ``pip install sam2``; the checkpoint is downloaded from Hugging Face.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pano", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    args = parser.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = cv2.cvtColor(cv2.imread(str(args.pano)), cv2.COLOR_BGR2RGB)
    boxes = json.loads(args.boxes.read_text())
    predictor = SAM2ImagePredictor.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        width = image.shape[1]
        for name, (x0, y0, x1, y1) in boxes.items():
            shift = 0
            if x0 > x1:  # wraps around the seam: roll the box centre to the middle of the image
                shift = width // 2 - int((x0 + x1 + width) / 2) % width
                predictor.set_image(np.roll(image, shift, axis=1))
                x0, x1 = (x0 + shift) % width, (x1 + shift) % width
            masks, scores, _ = predictor.predict(box=np.array([x0, y0, x1, y1], dtype=np.float32), multimask_output=False)
            if shift:
                predictor.set_image(image)
            mask = masks[0] > 0
            # Keep the largest connected component to drop stray fragments (before rolling back,
            # where a wrapping object would be split in two).
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if count > 1:
                mask = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.roll(mask, -shift, axis=1)
            cv2.imwrite(str(args.out / f"{name}.png"), mask.astype(np.uint8) * 255)
            print(f"{name}: score={scores[0]:.3f} pixels={mask.sum()}")


if __name__ == "__main__":
    main()
