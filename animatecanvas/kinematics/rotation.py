from __future__ import annotations
from typing import Union, Any, Optional
import math
import numpy as np
import torch

# ---- SciPy: used for all NumPy code paths ----
from scipy.spatial.transform import Rotation as R

ArrayLike = Union[torch.Tensor, np.ndarray]
EPS = 1e-6


class ROTATION_TYPE:
    QUATERNION = "quaternion"
    EULER = "euler"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    AXIS_ANGLE = "axis_angle"


ROT_DIM = {
    ROTATION_TYPE.QUATERNION: 4,
    ROTATION_TYPE.EULER: 3,
    ROTATION_TYPE.ROTATION_6D: 6,
    ROTATION_TYPE.MATRIX: 9,  # also accepts (...,3,3)
    ROTATION_TYPE.AXIS_ANGLE: 3,
}


# ----------------------- Utilities -----------------------
def _is_numpy(x: ArrayLike) -> bool:
    return isinstance(x, np.ndarray)


def _as_numpy(x: Any) -> np.ndarray:
    """
    Convert to np.ndarray safely.
    - If `x` is a torch.Tensor that participates in autograd (requires_grad=True
      or has a non-None grad_fn), raise an error to avoid silently breaking gradients.
    - Otherwise, convert with detach().cpu().numpy() for tensors, or np.asarray for others.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        if x.requires_grad or (x.grad_fn is not None):
            raise RuntimeError(
                "Attempted to convert a Tensor that requires grad to NumPy, which "
                "would break the autograd graph. Use torch-only ops or explicitly "
                "detach: x = x.detach().cpu().numpy()."
            )
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _from_numpy_like(template: ArrayLike, arr: np.ndarray) -> ArrayLike:
    if isinstance(template, np.ndarray):
        return arr
    device = (
        template.device if isinstance(template, torch.Tensor) else torch.device("cpu")
    )
    return torch.from_numpy(arr).to(device=device, dtype=template.dtype)


def _scipy_from_rotvec(value: np.ndarray) -> R:
    return R.from_rotvec(np.asarray(value).reshape(-1, 3))


def _scipy_from_matrix(value: np.ndarray) -> R:
    return R.from_matrix(np.asarray(value).reshape(-1, 3, 3))


def _scipy_from_quat(value: np.ndarray) -> R:
    return R.from_quat(np.asarray(value).reshape(-1, 4))


def _reshape_matrix9(x: ArrayLike) -> ArrayLike:
    # accept (...,9) -> (...,3,3)
    if x.shape[-1] == 9:
        if _is_numpy(x):
            return x.reshape(x.shape[:-1] + (3, 3))
        else:
            return x.view(*x.shape[:-1], 3, 3)
    return x


def _normalize_np(v: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _normalize_torch(v: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    n = torch.linalg.norm(v, dim=dim, keepdim=True).clamp_min(eps)
    return v / n


def _stack_cols01_np(Rm: np.ndarray) -> np.ndarray:
    return np.concatenate([Rm[..., 0:3, 0], Rm[..., 0:3, 1]], axis=-1)


def _stack_cols01_torch(Rm: torch.Tensor) -> torch.Tensor:
    return torch.cat([Rm[..., 0:3, 0], Rm[..., 0:3, 1]], dim=-1)


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("Letter must be X/Y/Z")


# ------------- Torch helpers (equivalent to PyTorch3D logic) -------------
def _copysign_t(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Equivalent to torch.copysign(x, y), compatible with older versions
    return torch.sign(y) * torch.abs(x)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # Avoid negative zero / numerical error
    return torch.sqrt(torch.clamp(x, min=0.0))


def standardize_quaternion(
    quaternions: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Standardize unit quaternion to primary hemisphere (w>=0).
    """
    if _is_numpy(quaternions):
        return np.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# ---------------- Axis-angle (rotvec) <-> Matrix ----------------
def axis_angle_to_matrix(axis_angle: ArrayLike) -> ArrayLike:
    """
    axis_angle: (..., 3) rotation vector = axis * angle(rad)
    return: (..., 3, 3)
    """
    # NumPy: use SciPy
    if _is_numpy(axis_angle):
        return _scipy_from_rotvec(axis_angle).as_matrix().reshape(axis_angle.shape[:-1] + (3, 3))

    # Torch: Rodrigues formula (fast path, matches PyTorch3D)
    a = axis_angle
    shape = a.shape
    device, dtype = a.device, a.dtype
    angles = torch.norm(a, dim=-1, keepdim=True).unsqueeze(-1)  # (...,1,1)

    rx, ry, rz = a[..., 0], a[..., 1], a[..., 2]
    z = torch.zeros(shape[:-1], dtype=dtype, device=device)
    K = torch.stack([z, -rz, ry, rz, z, -rx, -ry, rx, z], dim=-1).view(
        shape[:-1] + (3, 3)
    )
    K2 = K @ K
    I = torch.eye(3, dtype=dtype, device=device)
    ang2 = angles * angles
    ang2 = torch.where(ang2 == 0, torch.ones_like(ang2), ang2)
    return (
        I.expand_as(K)
        + torch.sinc(angles / math.pi) * K
        + ((1 - torch.cos(angles)) / ang2) * K2
    )


def matrix_to_axis_angle(matrix: ArrayLike) -> ArrayLike:
    """(...,3,3/9) -> rotvec (...,3)"""
    M = _reshape_matrix9(matrix)
    if _is_numpy(M):
        return _scipy_from_matrix(M).as_rotvec().reshape(M.shape[:-2] + (3,))

    if M.size(-1) != 3 or M.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {M.shape}.")
    w = torch.stack(
        [
            M[..., 2, 1] - M[..., 1, 2],
            M[..., 0, 2] - M[..., 2, 0],
            M[..., 1, 0] - M[..., 0, 1],
        ],
        dim=-1,
    )
    w_norm = torch.norm(w, dim=-1, keepdim=True)
    tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    ang = torch.atan2(w_norm, tr - 1.0)  # (...,1)

    # near zero -> zero vector
    zeros = torch.zeros(3, dtype=M.dtype, device=M.device)
    w = torch.where(torch.isclose(ang, torch.zeros_like(ang)), zeros, w)

    # general case
    out = torch.empty_like(w)
    mask_pi = torch.isclose(ang.squeeze(-1), ang.new_full((1,), math.pi))
    mask_not = ~mask_pi
    out[mask_not] = 0.5 * w[mask_not] / torch.sinc(ang[mask_not] / math.pi)

    # near pi case: use first row of (R+I)/2 as direction, magnitude = angle
    if mask_pi.any():
        n = 0.5 * (
            M[mask_pi][..., 0, :] + torch.eye(1, 3, dtype=M.dtype, device=M.device)
        )
        n = n / torch.clamp(torch.norm(n, dim=-1, keepdim=True), min=EPS)
        out[mask_pi] = n * ang[mask_pi]
    return out


# ---------------- Quaternion (w,x,y,z) <-> Matrix ----------------
def quaternion_to_matrix(quaternions: ArrayLike) -> ArrayLike:
    """(...,4) -> (...,3,3) with real part first (w,x,y,z)."""
    if _is_numpy(quaternions):
        # SciPy expects [x,y,z,w]
        q = quaternions
        xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
        return _scipy_from_quat(xyzw).as_matrix().reshape(q.shape[:-1] + (3, 3))

    q = quaternions
    r, i, j, k = torch.unbind(q, -1)
    # Stable formulation matching PyTorch3D
    two_s = 2.0 / torch.clamp((q * q).sum(-1), min=EPS)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return o.view(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: ArrayLike) -> ArrayLike:
    """(...,3,3/9) -> (...,4) w,x,y,z"""
    M = _reshape_matrix9(matrix)
    if _is_numpy(M):
        # SciPy as_quat -> [x,y,z,w]
        xyzw = _scipy_from_matrix(M).as_quat().reshape(M.shape[:-2] + (4,))
        return np.concatenate([xyzw[..., 3:4], xyzw[..., 0:3]], axis=-1)

    if M.size(-1) != 3 or M.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {M.shape}.")

    batch = M.shape[:-2]
    m = M.reshape(batch + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, -1)

    # q_abs: four candidate positive roots
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Build r/i/j/k scaled candidates (avoid division by small numbers)
    cand = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    # Best-conditioned candidate (largest denominator)
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    cand = cand / (2.0 * q_abs[..., None].max(floor))
    idx = q_abs.argmax(dim=-1, keepdim=True)
    gather_idx = idx.unsqueeze(-1).expand(list(batch) + [1, 4])
    out = torch.gather(cand, -2, gather_idx).squeeze(-2)
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: ArrayLike) -> ArrayLike:
    """
    Quaternion(w,x,y,z) -> rotvec(...,3)
    """
    if _is_numpy(quaternions):
        q = quaternions
        xyzw = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
        return _scipy_from_quat(xyzw).as_rotvec().reshape(q.shape[:-1] + (3,))

    q = standardize_quaternion(quaternions)
    v_norm = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    half = torch.atan2(v_norm, q[..., :1])
    sh_over_a = 0.5 * torch.sinc(half / math.pi)
    return q[..., 1:] / sh_over_a


def axis_angle_to_quaternion(axis_angle: ArrayLike) -> ArrayLike:
    """
    rotvec(...,3) -> quaternion(w,x,y,z)
    """
    if _is_numpy(axis_angle):
        xyzw = _scipy_from_rotvec(axis_angle).as_quat().reshape(axis_angle.shape[:-1] + (4,))
        return np.concatenate([xyzw[..., 3:4], xyzw[..., 0:3]], axis=-1)

    a = axis_angle
    ang = torch.linalg.norm(a, dim=-1, keepdim=True)
    sh_over_a = 0.5 * torch.sinc(ang * 0.5 / math.pi)
    return torch.cat([torch.cos(ang * 0.5), a * sh_over_a], dim=-1)


# ---------------- Euler <-> Matrix ----------------
def _axis_angle_rotation_torch(axis: str, angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        Rm = torch.stack(
            [
                torch.stack([one, zero, zero], dim=-1),
                torch.stack([zero, c, -s], dim=-1),
                torch.stack([zero, s, c], dim=-1),
            ],
            dim=-2,
        )
    elif axis == "Y":
        Rm = torch.stack(
            [
                torch.stack([c, zero, s], dim=-1),
                torch.stack([zero, one, zero], dim=-1),
                torch.stack([-s, zero, c], dim=-1),
            ],
            dim=-2,
        )
    else:  # "Z"
        Rm = torch.stack(
            [
                torch.stack([c, -s, zero], dim=-1),
                torch.stack([s, c, zero], dim=-1),
                torch.stack([zero, zero, one], dim=-1),
            ],
            dim=-2,
        )
    return Rm


def _angle_from_tan_torch(
    axis: str, other_axis: str, data: torch.Tensor, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def euler_to_matrix(e: ArrayLike, order: str = "XYZ", deg: bool = False) -> ArrayLike:
    if len(order) != 3 or any(c not in "XYZ" for c in order):
        raise ValueError("order must be three letters in 'XYZ'")
    # NumPy：SciPy
    if _is_numpy(e):
        rotations = R.from_euler(order, e.reshape(-1, e.shape[-1]), degrees=deg)
        return rotations.as_matrix().reshape(e.shape[:-1] + (3, 3))

    # Torch: follows PyTorch3D
    ang = e if not deg else (e * math.pi / 180.0)
    mats = [
        _axis_angle_rotation_torch(c, a) for c, a in zip(order, torch.unbind(ang, -1))
    ]
    return mats[0] @ mats[1] @ mats[2]


def matrix_to_euler(
    matrix: ArrayLike, order: str = "XYZ", deg: bool = False
) -> ArrayLike:
    M = _reshape_matrix9(matrix)
    if len(order) != 3:
        raise ValueError("order must have 3 letters.")
    if order[1] in (order[0], order[2]):
        raise ValueError(f"Invalid order {order}.")
    for letter in order:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in order string.")

    # NumPy：SciPy
    if _is_numpy(M):
        return _scipy_from_matrix(M).as_euler(order, degrees=deg).reshape(M.shape[:-2] + (3,))

    if M.size(-1) != 3 or M.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {M.shape}.")
    i0 = _index_from_letter(order[0])
    i2 = _index_from_letter(order[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central = torch.asin(
            torch.clamp(M[..., i0, i2], -1.0, 1.0)
            * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central = torch.acos(torch.clamp(M[..., i0, i0], -1.0, 1.0))
    o0 = _angle_from_tan_torch(order[0], order[1], M[..., i2], False, tait_bryan)
    o2 = _angle_from_tan_torch(order[2], order[1], M[..., i0, :], True, tait_bryan)
    ang = torch.stack([o0, central, o2], dim=-1)
    if deg:
        ang = ang / math.pi * 180.0
    return ang


# Convenience wrappers
def euler_to_quaternion(
    e: ArrayLike, order: str = "XYZ", deg: bool = True
) -> ArrayLike:
    return matrix_to_quaternion(euler_to_matrix(e, order=order, deg=deg))


def quaternion_to_euler(
    quat: ArrayLike, order: str = "XYZ", deg: bool = False
) -> ArrayLike:
    return matrix_to_euler(quaternion_to_matrix(quat), order=order, deg=deg)


def quaternion_to_euler_deg(quat: ArrayLike, order: str = "XYZ") -> ArrayLike:
    return quaternion_to_euler(quat, order=order, deg=True)


def axis_angle_to_euler(
    axis_angle: ArrayLike, order: str = "XYZ", deg: bool = False
) -> ArrayLike:
    return matrix_to_euler(axis_angle_to_matrix(axis_angle), order=order, deg=deg)


def axis_angle_to_euler_deg(axis_angle: ArrayLike, order: str = "XYZ") -> ArrayLike:
    return axis_angle_to_euler(axis_angle, order=order, deg=True)


# ---------------- 6D (Zhou19) <-> Matrix ----------------
# Convention:
#   "column" (default): d6 = [col0(3), col1(3)] = [R00,R10,R20, R01,R11,R21]
#   "row":              d6 = R[:, :2].reshape(6)
#                       = [R00,R01, R10,R11, R20,R21]
# This historical name means row-wise flattening of the first two COLUMNS. It
# is used by motion135/HY-Motion/DART. MS272 instead stores R[:2, :].reshape(6)
# and is decoded in motion272.py with a dedicated function.
# Permutation between them (applied per 6-element vector):
#   column_to_row: [..., [0, 3, 1, 4, 2, 5]]
#   row_to_column: [..., [0, 2, 4, 1, 3, 5]]
_COL_TO_ROW = [0, 3, 1, 4, 2, 5]
_ROW_TO_COL = [0, 2, 4, 1, 3, 5]


def rotation_6d_to_matrix(d6: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert 6D rotation representation to rotation matrix.

    Args:
        d6: (..., 6) rotation 6D vector.
        convention: "column" (default) or "row". Specifies the layout of the input d6.
    """
    assert convention in ("column", "row"), f"convention must be 'column' or 'row', got '{convention}'"
    # If input is row-major, convert to column-major for internal processing
    if convention == "row":
        if _is_numpy(d6):
            d6 = d6[..., _ROW_TO_COL]
        else:
            d6 = d6[..., _ROW_TO_COL]

    if _is_numpy(d6):
        assert d6.shape[-1] == 6, "Need (...,6)"
        x_raw = d6[..., 0:3]
        y_raw = d6[..., 3:6]
        x = _normalize_np(x_raw, axis=-1)
        z = np.cross(x, y_raw, axis=-1)
        z = _normalize_np(z, axis=-1)
        y = np.cross(z, x, axis=-1)
        return np.stack([x, y, z], axis=-1)
    else:
        assert d6.shape[-1] == 6, "Need (...,6)"
        x_raw = d6[..., 0:3]
        y_raw = d6[..., 3:6]
        x = _normalize_torch(x_raw, dim=-1)
        z = torch.cross(x, y_raw, dim=-1)
        z = _normalize_torch(z, dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=-1)


def matrix_to_rotation_6d(matrix: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert rotation matrix to 6D rotation representation.

    Args:
        matrix: (..., 3, 3) or (..., 9) rotation matrix.
        convention: "column" (default) or "row". Specifies the layout of the output d6.
    """
    assert convention in ("column", "row"), f"convention must be 'column' or 'row', got '{convention}'"
    M = _reshape_matrix9(matrix)
    if _is_numpy(M):
        d6 = _stack_cols01_np(M)
    else:
        d6 = _stack_cols01_torch(M)
    # If row-major output is requested, permute from column to row
    if convention == "row":
        d6 = d6[..., _COL_TO_ROW]
    return d6


# Aliases
cont6d_to_matrix = rotation_6d_to_matrix
matrix_to_cont6d = matrix_to_rotation_6d


def quaternion_to_rotation_6d(quat: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert quaternion to 6D rotation representation.

    Args:
        quat: (..., 4) quaternion (w, x, y, z).
        convention: "column" (default) or "row". Specifies the layout of the output d6.
    """
    return matrix_to_rotation_6d(quaternion_to_matrix(quat), convention=convention)


def rotation_6d_to_quaternion(d6: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert 6D rotation representation to quaternion.

    Args:
        d6: (..., 6) rotation 6D vector.
        convention: "column" (default) or "row". Specifies the layout of the input d6.
    """
    return matrix_to_quaternion(rotation_6d_to_matrix(d6, convention=convention))


def axis_angle_to_rotation_6d(axis_angle: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert axis-angle to 6D rotation representation.

    Args:
        axis_angle: (..., 3) rotation vector.
        convention: "column" (default) or "row". Specifies the layout of the output d6.
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle), convention=convention)


def rotation_6d_to_axis_angle(d6: ArrayLike, convention: str = "column") -> ArrayLike:
    """Convert 6D rotation representation to axis-angle.

    Args:
        d6: (..., 6) rotation 6D vector.
        convention: "column" (default) or "row". Specifies the layout of the input d6.
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6, convention=convention))


def rotation_6d_to_euler(
    d6: ArrayLike, order: str = "XYZ", deg: bool = False, convention: str = "column"
) -> ArrayLike:
    """Convert 6D rotation representation to euler angles.

    Args:
        d6: (..., 6) rotation 6D vector.
        order: Euler angle order.
        deg: If True, return degrees.
        convention: "column" (default) or "row". Specifies the layout of the input d6.
    """
    return matrix_to_euler(rotation_6d_to_matrix(d6, convention=convention), order=order, deg=deg)


def rotation_6d_to_euler_deg(d6: ArrayLike, order: str = "XYZ", convention: str = "column") -> ArrayLike:
    return rotation_6d_to_euler(d6, order=order, deg=True, convention=convention)


# old alias
expmap_to_quaternion = axis_angle_to_quaternion


# ---------------- Generic converter ----------------
_VALID_TYPES = {
    "quaternion",
    "matrix",
    "axis_angle",
    "rotvec",
    "expmap",
    "euler",
    "rotation_6d",
    ROTATION_TYPE.ROTATION_6D,
    "cont6d",
    "joints",
}


def _normalize_type_name(t: str) -> str:
    t = t.lower()
    if t in (ROTATION_TYPE.ROTATION_6D, "rotation_6d", "cont6d"):
        return "rotation_6d"
    if t in ("rotvec", "expmap"):
        return "axis_angle"
    if t in ("mat", "matrix3x3"):
        return "matrix"
    return t


def rot_convert(
    from_rot: ArrayLike,
    from_type: str,
    to_type: str,
    *,
    order: str = "XYZ",
    deg: bool = False,
) -> ArrayLike:
    ft = _normalize_type_name(from_type)
    tt = _normalize_type_name(to_type)
    if ft not in _VALID_TYPES or tt not in _VALID_TYPES:
        raise ValueError(f"Unsupported type(s): {from_type} -> {to_type}")
    if ft == tt:
        return from_rot

    # NumPy + SciPy fast path (not used for torch)
    if _is_numpy(from_rot):

        def _to_rot(x: np.ndarray, t: str) -> "R":
            if t == "matrix":
                return R.from_matrix(_reshape_matrix9(x))
            if t == "quaternion":
                xyzw = np.concatenate([x[..., 1:4], x[..., 0:1]], axis=-1)
                return R.from_quat(xyzw)
            if t == "axis_angle":
                return R.from_rotvec(x)
            if t == "euler":
                return R.from_euler(order.lower(), x, degrees=deg)
            if t == "rotation_6d":
                return R.from_matrix(rotation_6d_to_matrix(x))
            raise ValueError(f"Unsupported from_type: {t}")

        def _from_rot(rot: "R", t: str) -> np.ndarray:
            if t == "matrix":
                return rot.as_matrix()
            if t == "quaternion":
                xyzw = rot.as_quat()
                return np.concatenate([xyzw[..., 3:4], xyzw[..., 0:3]], axis=-1)
            if t == "axis_angle":
                return rot.as_rotvec()
            if t == "euler":
                return rot.as_euler(order.lower(), degrees=deg)
            if t == "rotation_6d":
                Rm = rot.as_matrix()
                return _stack_cols01_np(Rm)
            raise ValueError(f"Unsupported to_type: {t}")

        r = _to_rot(from_rot, ft)
        return _from_rot(r, tt)

    # Torch: via matrix intermediate (logic matches PyTorch3D)
    def to_matrix(x: ArrayLike, typ: str) -> ArrayLike:
        if typ == "matrix":
            return _reshape_matrix9(x)
        if typ == "quaternion":
            return quaternion_to_matrix(x)
        if typ == "axis_angle":
            return axis_angle_to_matrix(x)
        if typ == "euler":
            return euler_to_matrix(x, order=order, deg=deg)
        if typ == "rotation_6d":
            return rotation_6d_to_matrix(x)
        raise ValueError(f"Unsupported from_type: {typ}")

    def from_matrix(Rm: ArrayLike, typ: str) -> ArrayLike:
        if typ == "matrix":
            return Rm
        if typ == "quaternion":
            return matrix_to_quaternion(Rm)
        if typ == "axis_angle":
            return matrix_to_axis_angle(Rm)
        if typ == "euler":
            return matrix_to_euler(Rm, order=order, deg=deg)
        if typ == "rotation_6d":
            return matrix_to_rotation_6d(Rm)
        raise ValueError(f"Unsupported to_type: {typ}")

    return from_matrix(to_matrix(from_rot, ft), tt)


# ---------------- Public 6D convention helpers ----------------
class Rot6DConvention:
    """Canonical names for the two 6D rotation layouts used across the repo.

    - ``COLUMN`` (math default): ``[R00,R10,R20, R01,R11,R21]`` (first two columns).
      Produced/consumed by every ``*_6d_*`` function here when ``convention="column"``.
    - ``ROW`` (training data / model I/O / checkpoints / MS272 rot block):
      ``[R00,R01, R10,R11, R20,R21]`` (first two rows).

    Use :func:`repack_6d` to convert a stored 6D vector between layouts, or pass
    ``convention=`` directly to the conversion functions.
    """

    COLUMN = "column"
    ROW = "row"


def repack_6d(d6: ArrayLike, src: str, dst: str) -> ArrayLike:
    """Re-pack a 6D rotation vector between ``"column"`` and ``"row"`` layouts.

    This only permutes the trailing 6 elements; it does NOT change the rotation.
    Use it when a tensor is *stored* in one convention but a consumer expects the
    other (e.g. column-major IK output feeding a row-major training pipeline).

    Args:
        d6: (..., 6) or (..., 6*N) packed per-joint 6D vectors. The permutation is
            applied on the last axis assuming contiguous 6-tuples, so pass either a
            (...,6) array or reshape (...,N,6) first.
        src: source layout, ``"column"`` or ``"row"``.
        dst: destination layout, ``"column"`` or ``"row"``.
    """
    assert src in ("column", "row"), f"src must be 'column'/'row', got {src!r}"
    assert dst in ("column", "row"), f"dst must be 'column'/'row', got {dst!r}"
    if src == dst:
        return d6
    assert d6.shape[-1] == 6, (
        "repack_6d expects a trailing dim of 6; reshape (...,N,6) before calling for "
        "per-joint vectors."
    )
    perm = _COL_TO_ROW if (src == "column" and dst == "row") else _ROW_TO_COL
    return d6[..., perm]


