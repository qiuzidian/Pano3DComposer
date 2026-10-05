"""Run Pano3DComposer on a scene folder (see pano3d/data.py for the expected layout).

    python scripts/infer.py --scene demo/lythwood_room --out outputs/lythwood_room
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pano3d.data import load_scene  # noqa: E402
from pano3d.pipeline import Config, Pano3DComposer, save_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="scene folder with panoramas/ and masks/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pano3d.safetensors"))
    parser.add_argument("--generator", choices=["amodal3r", "trellis"], default="amodal3r")
    parser.add_argument("--pose-fusion", choices=["first", "weighted"], default="first",
                        help="'weighted' fuses all crop x rendering pose candidates by consistency-weighted SE(3) averaging")
    parser.add_argument("--no-refine", action="store_true", help="disable coarse-to-fine refinement")
    parser.add_argument("--no-upright", action="store_true", help="keep the full predicted rotation")
    parser.add_argument("--physics", action="store_true", help="resolve collisions between placed objects")
    parser.add_argument("--background", choices=["auto", "on", "off"], default="auto",
                        help="background Gaussians; 'auto' = only for a single panorama")
    parser.add_argument("--meshes-only", action="store_true", help="only export object meshes and transforms")
    parser.add_argument("--save-process", action="store_true", help="also dump intermediate results of every stage")
    parser.add_argument("--low-vram", action="store_true", help="keep idle models on the CPU")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        checkpoint=args.checkpoint,
        generator=args.generator,
        pose_fusion=args.pose_fusion,
        refine=not args.no_refine,
        upright=not args.no_upright,
        physics=args.physics,
        background={"auto": None, "on": True, "off": False}[args.background],
        low_vram=args.low_vram,
        seed=args.seed,
    )
    scene = load_scene(args.scene)
    print(f"{scene.name}: {len(scene.panoramas)} panorama(s), {len(scene.objects)} object(s)")
    if len(scene.panoramas) > 1 and args.background == "auto":
        args.meshes_only = True  # multi-view inputs export the aligned object meshes
    result = Pano3DComposer(config)(scene)
    save_result(result, args.out, save_gaussians=not args.meshes_only)
    if args.save_process:
        from pano3d.visualize import save_process

        save_process(scene, args.scene, result, args.out)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
