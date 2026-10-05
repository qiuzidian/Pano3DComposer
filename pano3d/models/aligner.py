"""Object-to-world transformation predictor (depth-conditioned OmniVGGT).

Given renderings of a generated asset in its local frame and real crops of the
object in the world frame, OmniVGGT predicts all camera poses in one shared
(network) frame plus an anisotropic scale. Chaining the predicted relative pose
with the known cameras maps the asset from its local frame to the world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omnivggt.heads.camera_head import CameraHead
from omnivggt.models.omnivggt_aggregator import ZeroAggregator
from omnivggt.utils.geometry import closed_form_inverse_se3
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

from ..geometry import pose_distance, se3_inverse, weighted_se3_mean

INPUT_SIZE = 224  # resolution of the views fed to OmniVGGT


@dataclass
class PoseInputs:
    """A set of views with known intrinsics, (optional) extrinsics and depth."""

    rgb: torch.Tensor  # (S, 3, H, W) in [0, 1], white background
    depth: torch.Tensor  # (S, H, W) z-depth, 0 outside the object
    K: torch.Tensor  # (S, 3, 3) normalized intrinsics
    c2w: torch.Tensor  # (S, 4, 4) camera-to-world (zeros when unknown)

    def resized(self, size: int = INPUT_SIZE) -> "PoseInputs":
        rgb = F.interpolate(self.rgb, size=(size, size), mode="bilinear", align_corners=False)
        depth = F.interpolate(self.depth[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        return PoseInputs(rgb, depth, self.K, self.c2w)

    @staticmethod
    def cat(a: "PoseInputs", b: "PoseInputs") -> "PoseInputs":
        return PoseInputs(*(torch.cat([x, y]) for x, y in zip(vars(a).values(), vars(b).values())))


def _normalize_to_first(w2c: torch.Tensor) -> torch.Tensor:
    """Express (1, S, 3, 4) world-to-camera matrices relative to the first camera.

    With more than one camera the translations are also divided by the mean
    distance to the first camera, as done during training.
    """
    B, S = w2c.shape[:2]
    homog = torch.cat([w2c, torch.zeros(B, S, 1, 4, device=w2c.device)], dim=-2)
    homog[:, :, -1, -1] = 1.0
    rel = homog @ closed_form_inverse_se3(homog[:, 0]).unsqueeze(1)
    if S > 1:
        centers = rel[:, :, :3, 3]
        scale = (centers - centers[:, :1]).norm(dim=-1)[:, 1:].mean(dim=1, keepdim=True).clamp(min=1e-6)
        rel[:, :, :3, 3] /= scale.unsqueeze(-1)
    return rel[:, :, :3]


class OmniVGGTPoseNet(nn.Module):
    """OmniVGGT trunk + camera head with an extra anisotropic scale branch."""

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = ZeroAggregator(img_size=518, patch_size=14, embed_dim=1024, pose_hidden_dim=9)
        self.camera_head = CameraHead(dim_in=2 * 1024)
        self.amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], device: torch.device) -> "OmniVGGTPoseNet":
        with torch.device(device):
            net = cls()
        net.load_state_dict(state, strict=True)
        return net.to(device).eval()  # also moves buffers created outside the device context

    @torch.no_grad()
    def forward(self, views: PoseInputs, num_known: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict poses for all views.

        The first ``num_known`` views and the remaining views are two groups with
        independently normalized extrinsics (e.g. local renderings vs. world crops).
        Every view is conditioned on its camera token and depth.

        Returns:
            c2w: (S, 4, 4) camera-to-world poses in the network frame.
            scale: (S, 3) predicted anisotropic scale tokens.
        """
        S = views.rgb.shape[0]
        w2c = closed_form_inverse_se3(views.c2w)[None]
        w2c = torch.cat([_normalize_to_first(w2c[:, :num_known, :3]), _normalize_to_first(w2c[:, num_known:, :3])], 1)
        all_views = list(range(S))
        depth = views.depth[None, ..., None]
        with torch.autocast("cuda", dtype=self.amp_dtype):
            tokens, _ = self.aggregator(
                images=views.rgb[None],
                extrinsics=w2c,
                intrinsics=views.K[None],
                depth=depth,
                mask=(depth > 0)[..., 0],
                depth_gt_index=all_views,
                camera_gt_index=all_views,
            )
        with torch.autocast("cuda", enabled=False):
            pose_enc = self.camera_head(tokens)[-1]
            w2c_pred, _, scale = pose_encoding_to_extri_intri(pose_enc, views.rgb.shape[-2:])
        return closed_form_inverse_se3(w2c_pred[0]), scale[0]


def candidate_transforms(pred: torch.Tensor, local_c2w: torch.Tensor, world_c2w: torch.Tensor) -> torch.Tensor:
    """Local-to-world rigid transforms for every (world view u, local view v) pair.

    ``pred`` holds the predicted poses of the V local views followed by the n world
    views. The relative pose v -> u is coordinate-frame invariant, so applying it to
    the known local camera of v gives the local-frame camera of world view u:
    E_u^local = E_v^local (P_v)^-1 P_u. Then T_uv = E_u^world (E_u^local)^-1.

    Returns:
        (n, V, 4, 4) candidate transforms.
    """
    V = local_c2w.shape[0]
    pred_local, pred_world = pred[:V], pred[V:]
    local_of_world = local_c2w[None] @ se3_inverse(pred_local)[None] @ pred_world[:, None]  # (n, V, 4, 4)
    return world_c2w[:, None] @ se3_inverse(local_of_world)


def fuse_candidates(
    pred: torch.Tensor, local_c2w: torch.Tensor, world_c2w: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Consistency-weighted SE(3) fusion of all n x V candidate transforms.

    A candidate scores high when, applied to the local-frame cameras implied by every
    other (u', v') pair, it reproduces the known world cameras.
    """
    V = local_c2w.shape[0]
    candidates = candidate_transforms(pred, local_c2w, world_c2w).flatten(0, 1)  # (n*V, 4, 4)
    local_of_world = local_c2w[None] @ se3_inverse(pred[:V])[None] @ pred[V:, None]  # (n, V, 4, 4)
    implied_world = candidates[:, None, None] @ local_of_world[None]  # (n*V, n, V, 4, 4)
    errors = pose_distance(implied_world, world_c2w[None, :, None].expand_as(implied_world))
    weights = torch.softmax(-errors.flatten(1).mean(1) / temperature, dim=0)
    return weighted_se3_mean(candidates, weights)


def first_pair_transform(pred: torch.Tensor, local_c2w: torch.Tensor, world_c2w: torch.Tensor) -> torch.Tensor:
    """Transform from the first world view and the first local view only (conference version)."""
    return candidate_transforms(pred, local_c2w, world_c2w)[0, 0]


def relative_update(pred: torch.Tensor, world_c2w: torch.Tensor, num_world: int) -> torch.Tensor:
    """World-frame rigid correction from a refiner prediction.

    The refiner sees the n real crops followed by n renderings of the current
    placement from the same cameras; the predicted relative pose between the first
    rendering and the first crop, conjugated into the world frame, is the update.
    """
    delta = pred[0] @ se3_inverse(pred[num_world])
    return world_c2w[0] @ delta @ se3_inverse(world_c2w[0])
