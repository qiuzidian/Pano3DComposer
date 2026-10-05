"""A minimal 3D Gaussian container shared by objects and background."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

SH_C0 = 0.28209479177387814


def quaternion_to_matrix(q_xyzw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation matrices from quaternions stored in (x, y, z, w) order."""
    i, j, k, r = torch.unbind(q_xyzw, dim=-1)
    two_s = 2 / ((q_xyzw * q_xyzw).sum(dim=-1) + eps)
    m = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(m, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(scales: torch.Tensor, rotations_xyzw: torch.Tensor) -> torch.Tensor:
    R = quaternion_to_matrix(rotations_xyzw)
    S = scales.diag_embed()
    return R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)


@dataclass
class Gaussians:
    """N Gaussians in world space.

    ``harmonics`` has shape (N, 3, d_sh); only the DC band is used for rendering.
    ``scales``/``rotations`` describe the local shape and are only used for export;
    ``covariances`` is the source of truth (it can encode anisotropic object scaling).
    """

    means: torch.Tensor  # (N, 3)
    covariances: torch.Tensor  # (N, 3, 3)
    harmonics: torch.Tensor  # (N, 3, d_sh)
    opacities: torch.Tensor  # (N,)
    scales: torch.Tensor  # (N, 3)
    rotations: torch.Tensor  # (N, 4), xyzw

    @staticmethod
    def concat(items: list["Gaussians"]) -> "Gaussians":
        d_sh = max(g.harmonics.shape[-1] for g in items)
        harmonics = [torch.nn.functional.pad(g.harmonics, (0, d_sh - g.harmonics.shape[-1])) for g in items]
        return Gaussians(
            means=torch.cat([g.means for g in items]),
            covariances=torch.cat([g.covariances for g in items]),
            harmonics=torch.cat(harmonics),
            opacities=torch.cat([g.opacities for g in items]),
            scales=torch.cat([g.scales for g in items]),
            rotations=torch.cat([g.rotations for g in items]),
        )

    def transformed(self, T: torch.Tensor) -> "Gaussians":
        """Apply an affine transform T (4x4, rotation x anisotropic scale + translation).

        Only the DC color is kept, so the spherical harmonics need no rotation.
        """
        A, t = T[:3, :3].to(self.means), T[:3, 3].to(self.means)
        return Gaussians(
            means=self.means @ A.T + t,
            covariances=A @ self.covariances @ A.T,
            harmonics=self.harmonics,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.rotations,
        )

    def save_ply(self, path: str | Path) -> None:
        """Export in the standard 3DGS ply layout (DC color only).

        Scales and rotations are re-derived from the covariances, so anisotropic
        object scaling is preserved in the exported file.
        """
        from plyfile import PlyData, PlyElement

        cov = self.covariances.detach().double().cpu()
        eigval, eigvec = torch.linalg.eigh(cov)
        det = torch.linalg.det(eigvec)
        eigvec[..., :, 0] *= det.sign()[..., None]  # make it a proper rotation
        scales = eigval.clamp_min(1e-12).sqrt()
        quat_wxyz = _matrix_to_quaternion_wxyz(eigvec)

        dc = (self.harmonics[..., 0].detach().double().cpu()).numpy()
        opacity = self.opacities.detach().double().cpu().clamp(1e-6, 1 - 1e-6)
        attributes = {
            "x": self.means[:, 0], "y": self.means[:, 1], "z": self.means[:, 2],
            "f_dc_0": dc[:, 0], "f_dc_1": dc[:, 1], "f_dc_2": dc[:, 2],
            "opacity": torch.log(opacity / (1 - opacity)),
            "scale_0": scales[:, 0].log(), "scale_1": scales[:, 1].log(), "scale_2": scales[:, 2].log(),
            "rot_0": quat_wxyz[:, 0], "rot_1": quat_wxyz[:, 1], "rot_2": quat_wxyz[:, 2], "rot_3": quat_wxyz[:, 3],
        }
        dtype = [(name, "f4") for name in attributes]
        data = np.empty(self.means.shape[0], dtype=dtype)
        for name, value in attributes.items():
            data[name] = value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def _matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.stack(
        [
            1 + trace,
            1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ],
        dim=-1,
    ).clamp_min(0).sqrt() / 2
    q[..., 1] = torch.copysign(q[..., 1], m[..., 2, 1] - m[..., 1, 2])
    q[..., 2] = torch.copysign(q[..., 2], m[..., 0, 2] - m[..., 2, 0])
    q[..., 3] = torch.copysign(q[..., 3], m[..., 1, 0] - m[..., 0, 1])
    return q
