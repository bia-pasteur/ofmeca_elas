"""Useful for the computation of mechanical quantities"""
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import unit_impulse


def parse_spacing(
    f_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *varargs: float | Sequence[float] | np.ndarray,
) -> list[float | np.ndarray]:
    """
    Parse spacing arguments for gradient calculation.

    This function mimics the behavior of `np.gradient` when handling
    scalar and array-like spacing arguments, ensuring proper broadcasting
    for each axis.

    Args:
        f_shape (tuple of int):
            Shape of the input array `f`.
        axes (tuple of int):
            Axes along which the gradient will be computed.
        *varargs (float | np.ndarray | Sequence):
            - No arguments: spacing = 1.0 along all axes.
            - One scalar: same spacing is applied to all axes.
            - One sequence/array per axis: custom spacing.

    Returns:
        list[float | np.ndarray]:
            Spacing per axis (scalar or 1D array).

    Raises:
        ValueError:
            If spacing array length does not match the axis length.
    """
    len_axes = len(axes)
    n = len(varargs)

    dx = (
        [1.0] * len_axes
        if n == 0
        else list(varargs) * len_axes
        if n == 1
        else list(varargs)
    )

    for i, d in enumerate(dx):
        arr = np.asanyarray(d)

        if arr.ndim == 0:
            dx[i] = float(arr)
        elif arr.ndim == 1:
            if len(arr) != f_shape[axes[i]]:
                raise ValueError("1D spacing must match axis length")

            diffx = np.diff(
                arr.astype(np.float64) if np.issubdtype(arr.dtype, np.integer) else arr
            )
            dx[i] = diffx[0] if np.allclose(diffx, diffx[0]) else diffx
        else:
            raise ValueError("Spacing must be scalar or 1D")

    return dx


def apply_neumann_bc(
    f: np.ndarray,
    gradient: np.ndarray,
    mask: np.ndarray,
    boundary_coords: np.ndarray,
    dx: list[float],
    axes: tuple[int, ...],
) -> None:
    """
    Apply Neumann boundary conditions by estimating one-sided differences.

    Args:
        f (np.ndarray): Input array.
        gradient (np.ndarray): Gradient array to be modified in-place.
        mask (np.ndarray): Binary mask indicating the domain of interest.
        boundary_coords (np.ndarray): Coordinates of boundary pixels.
        dx (list[float]): Spacing values for each axis.
        axes (tuple[int]): Axes along which gradients are computed.
    """
    dim = f.ndim
    units = [unit_impulse(dim, axis, dtype=int) for axis in axes]

    for coor in boundary_coords:
        for i, (_, ax_dx) in enumerate(zip(axes, dx)):
            offset = units[i] * ax_dx
            next_pixel = tuple((coor + offset).astype(int))
            prev_pixel = tuple((coor - offset).astype(int))

            if mask[next_pixel] == 1:
                val = f[next_pixel] - f[tuple(coor)]
            elif mask[prev_pixel] == 1:
                val = f[tuple(coor)] - f[prev_pixel]
            else:
                continue

            if gradient.ndim == f.ndim + 1:
                gradient[i][tuple(coor)] = val
            else:
                gradient[tuple(coor)] = val


def grad_domain(
    f: np.ndarray,
    mask: np.ndarray,
    *varargs: float | Sequence[float] | np.ndarray,
    axis: Any = None,
    boundary_condition: str = "Neumann",
) -> np.ndarray:
    """
    Computes the masked gradient of an N-dimensional array `f`
    with support for Neumann or Dirichlet boundary conditions.

    Args:
        f (np.ndarray): The input N-dimensional array.
        mask (np.ndarray): Binary mask (same shape as `f`).
        *varargs (float | Sequence[float] | np.ndarray): Spacing.
        axis (int | Sequence[int] | None): Axes along which gradient is computed.
        boundary_condition (str): 'Neumann' or 'Dirichlet'.

    Returns:
        np.ndarray: Gradient of `f` within the masked domain.
    """
    N = f.ndim
    axes = (
        tuple(range(N))
        if axis is None
        else np.core.numeric.normalize_axis_tuple(axis, N)
    )

    dx = parse_spacing(f.shape, axes, *varargs)

    # Compute regular gradient
    gradient = np.array(np.gradient(f, *varargs, axis=axis))

    # Identify boundary pixels
    boundary_mask = distance_transform_edt(mask) == 1
    boundary_coords = np.argwhere(boundary_mask)

    # Apply boundary condition
    if boundary_condition == "Neumann":
        apply_neumann_bc(f, gradient, mask, boundary_coords, dx, axes)
    elif boundary_condition == "Dirichlet":
        if gradient.ndim == f.ndim + 1:
            gradient[:, boundary_mask] = 0
        else:
            gradient[boundary_mask] = 0
    else:
        raise ValueError(
            "Unsupported boundary_condition: choose 'Neumann' or 'Dirichlet'"
        )

    return gradient * mask


def jacobian_mask(
    u: np.ndarray, spacing: Sequence[float], maskcell: np.ndarray
) -> np.ndarray:
    """
    Computes the Jacobian of the displacement field

    Args:
        u (np.ndarray): The displacement field
            Shape : (d, T, *S)

    Returns:
        np.ndarray: The Jacobian of the displacement field
            Shape : (d, d, T, *S)
    """

    dim = u.shape[0]
    jac = np.zeros((dim,) + u.shape)

    for i in range(dim):
        for j in range(dim):
            if u[i].ndim == maskcell.ndim + 1:
                mask = np.broadcast_to(maskcell, u[i].shape)
            else:
                mask = maskcell
            jac[i, j] = jac[i, j] = grad_domain(
                u[i], mask, spacing[j], axis=j + 1, boundary_condition="Neumann"
            )
    return jac


def strain_mask(
    u: np.ndarray, spacing: Sequence[float], maskcell: np.ndarray
) -> np.ndarray:
    """
    Computes the strain of the displacement field

    Args:
        u (np.ndarray): The displacement field
            Shape: (d, T, *S)

    Returns:
        np.ndarray: The strain of the displacement field
            Shape: (d, d, T, *S)
    """

    jac = jacobian_mask(u, spacing, maskcell)

    return 0.5 * (jac + np.swapaxes(jac, 0, 1))


def deformation(eps: np.ndarray) -> np.ndarray:
    """
    Computes the deformation from a vector field

    Args:
        u (np.ndarray): The vector field

    Returns:
        np.ndarray: the norm of the strain tensor at every pixel
    """

    # Return the norm of the strain tensor
    return np.linalg.norm(eps, axis=(0, 1))


def stress_mask(eps: np.ndarray, mu: float, lambda_: float) -> np.ndarray:
    """
    Computes the stress of the displacement field

    Args:
        u (np.ndarray): The displacement field
            Shape: (d, T, *S)
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter

    Returns:
        np.ndarray: the stress of the displacement field
            Shape: (d, d, T, *S)
    """

    dim = eps.shape[0]
    tr = np.trace(eps, axis1=0, axis2=1)
    s = np.zeros_like(eps)

    for ax1 in range(dim):
        for ax2 in range(dim):
            s[ax1, ax2] = 2 * mu * eps[ax1, ax2] + lambda_ * tr * (ax1 == ax2)
    return s


def compute_normals_from_mask_2d(mask: np.ndarray) -> np.ndarray:
    """
    Compute outward normal vectors to a binary 2D mask using the gradient.

    Args:
        mask (np.ndarray): Binary mask (Y, X) where 1 indicates the boundary zone.

    Returns:
        normals (np.ndarray): Array of normal vectors of shape (2, Y, X), normalized to unit length.
    """
    gy, gx = np.gradient(mask.astype(float))
    norm = np.sqrt(gx**2 + gy**2)
    nx = gx / (norm + 1e-12)
    ny = gy / (norm + 1e-12)

    normals = np.stack((ny, nx), axis=0)
    return normals


def compute_traction_2d(stress_field: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute traction t = sigma · n in 2D for a single timepoint.

    Args:
        stress_field (np.ndarray): Stress tensor field, shape (2, 2, Y, X)
        normals (np.ndarray): Normal vectors, shape (2, Y, X)

    Returns:
        traction (np.ndarray): Traction vector field, shape (2, Y, X)
    """
    d, _, Y, X = stress_field.shape
    traction = np.zeros((d, Y, X))

    for i in range(d):
        for j in range(d):
            traction[i] += stress_field[i, j] * normals[j]

    return traction
