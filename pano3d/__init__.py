"""Pano3DComposer: feed-forward compositional 3D scene generation from panoramas."""

import sys
from pathlib import Path

# The vendored OmniVGGT package lives in third_party/omnivggt.
_OMNIVGGT = Path(__file__).resolve().parent.parent / "third_party" / "omnivggt"
if str(_OMNIVGGT) not in sys.path:
    sys.path.insert(0, str(_OMNIVGGT))
