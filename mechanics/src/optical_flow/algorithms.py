"""The optical flow algorithms"""

# pylint: disable=invalid-name
import warnings

import cv2
import numba
import numpy as np
import scipy.sparse as sp
import torch
from byotrack.implementation.optical_flow.opencv import OpenCVOpticalFlow
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.signal import convolve
from skimage.registration import (  # pylint: disable=no-name-in-module
    optical_flow_ilk,
    optical_flow_tvl1,
)
from skimage.transform import pyramid_reduce  # pylint: disable=no-name-in-module
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from mechanics.src.config import (
    FarnebackParams,
    FistaParams,
    HSParams,
    ILKParams,
    TVL1Params,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="scipy.sparse.linalg._eigen._svds"
)

### Functions related to FISTA Optical Flow with multiscale approach


def A(I):
    """
    Computes A such that A @ h = nabla_I ⋅ h

    Args:
        I (np.ndarray): The image whose gradient is computed (2D or 3D).

    Returns:
        sp.csr_matrix: Sparse matrix representing A.
    """
    grads = np.gradient(I)
    grads_flat = [np.ravel(g, order="C") for g in grads]

    A_blocks = [sp.diags(g, offsets=0, format="csr") for g in grads_flat]

    A_full = sp.hstack(A_blocks, format="csr")

    return A_full


@numba.njit()
def nabla_block(
    S: tuple[int, ...], axis=0, offset_i=0, offset_j=0
) -> tuple[list[int], list[int], list[int]]:
    """
    Constructs the sparse representation (row, col, value lists) of a discrete forward finite-difference
    operator along a single spatial axis for a given multidimensional grid.

    The function creates one "block" of the full gradient operator — i.e., the partial derivative
    with respect to the chosen axis — as lists of indices and values that can later be assembled
    into a sparse matrix.

    Args:
        S (Tuple[int, ...]): Shape of the multidimensional grid (e.g., `(ny, nx)` for a 2D image or `(nz, ny, nx)` for 3D).
        axis (int, optional): Axis along which to compute the discrete derivative.
            Defaults to 0.
        offset_i (int, optional): Row index offset for the resulting block in the global sparse matrix.
            Defaults to 0.
        offset_j (int, optional): Column index offset for the resulting block in the global sparse matrix.
            Defaults to 0.

    Returns:
        Tuple[List[int], List[int], List[int]]: Three lists `(rows, cols, values)` representing the COO-format entries of the
            discrete difference operator:
              - `rows[i]`: row index of entry i
              - `cols[i]`: column index of entry i
              - `values[i]`: numerical value (+1 or -1)
    """
    num_pixels = 1
    step_size = 1
    temp_size = 1
    for ax, size in enumerate(S):
        num_pixels *= size
        if ax > axis:
            step_size *= size
        if ax < axis:
            temp_size *= size

    rows = []
    cols = []
    values = []

    for i in range(num_pixels):
        rows.append(i + offset_i)
        cols.append(i + offset_j)
        values.append(1)

        k = (i % (step_size * S[axis])) // step_size

        if k + 1 < S[axis]:
            rows.append(i + offset_i)
            cols.append(i + step_size + offset_j)
            values.append(-1)

    return rows, cols, values


def nabla(C, S: tuple[int, ...]):
    """
    Constructs a full sparse gradient operator for a given multidimensional domain and
    number of channels.

    The operator maps flattened input fields of shape `(C, *S)` to their discrete gradients
    along each spatial axis, producing an array of shape `(C * len(S), *S)` when applied.

    Args:
        C (int):
            Number of channels (e.g., components of a vector field such as u_x, u_y, u_z).
        S (Tuple[int, ...]):
            Shape of the spatial domain (e.g., `(ny, nx)` for 2D, `(nz, ny, nx)` for 3D).

    Returns:
        sp.csr_array:
            Sparse matrix of shape `(C * len(S) * np.prod(S), C * np.prod(S))` in CSR format,
            representing the discrete gradient operator ∇ applied to a multi-channel field.

    Notes:
        - The operator is assembled by stacking the per-axis finite-difference blocks generated
          by `nabla_block`.
    """
    rows = []
    cols = []
    values = []
    num_pixels = np.prod(S, initial=1)
    for axis in range(len(S)):
        for channel in range(C):
            block_id = C * axis + channel
            rows_, cols_, values_ = nabla_block(
                S,
                axis=axis,
                offset_i=block_id * num_pixels,
                offset_j=channel * num_pixels,
            )
            rows.extend(rows_)
            cols.extend(cols_)
            values.extend(values_)

    return sp.coo_array(
        (values, (rows, cols)), shape=(len(S) * C * num_pixels, C * num_pixels)
    ).tocsr()


def compute_Q(I: np.ndarray, alpha: float, beta: float, nabla_for_L: sp.spmatrix):
    """
    Computed the matrix Q defined as 2*(A^T*A + alpha*2*∇^T*∇ + beta*2*H^T*H)

    Args:
        I (np.ndarray): Initial image to compute A with
        alpha (float): Regularization parameter for the gradient
        beta (float): Regularization parameter for the hessian
        nabla_for_L (sp.spmatrix): Matrix representation of the gradient operator

    Returns:
         sp.csr_array:
            Sparse matrix reqpresenting Q
    """
    A_mat = A(I)
    nabla_mat = nabla_for_L
    H_mat = nabla_mat.T @ nabla_mat

    Q = (
        A_mat.T @ A_mat
        + alpha**2 * (nabla_mat.T @ nabla_mat)
        + beta**2 * (H_mat.T @ H_mat)
    )
    return 2 * Q


def compute_lip(I: np.ndarray, alpha: float, beta: float, nabla_for_L: sp.spmatrix):
    """
    Computes the best Lipschitz constant L of the gradient of the function
    f(h) = norm_2(∇I2 ·(h - h0)+It)^2 + alpha * norm_2,1(∇h)^2 + beta * norm_2,1(Hh)^2
    L = sigma_max{2*(A^T*A + alpha*2*∇^T*∇ + beta*2*H^T*H)}

    Args:
        I (np.ndarray): The image whose gradient will be computed in A
        alpha (float): Regularization parameter controlling smoothness of the flow field.
        beta (float): Regularization parameter controlling smoothness of the gradient of the flow field.
        nabla_fot_L (np.ndarray):

    Returns:
        float: Theoretical Lipschitz constant of the gradient of f(h)
    """
    Q = compute_Q(I, alpha, beta, nabla_for_L)
    return sp.linalg.norm(Q, ord=2)


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warps a multi dimensionnal image using an optical flow

    Args:
        image (np.ndarray): Image to warp. Can optionnally have a channel dimension.
            Shape: (H, ...[, C])
        flow (np.ndarray): Optical flow. The first dimension is 2 for 2D images, 3 for 3D images, etc...
            Shape: (ndim, H, ...), dtype: float (32 or 64)

    Returns:
        np.ndarray: Warped image
            Shape: (H, ...[, C])
    """
    assert flow.ndim - 1 == flow.shape[0], (
        "First dimension should match the number of trailing dimensions"
    )

    if image.ndim == flow.ndim:
        *dims, channels = image.shape
        points = np.indices(dims, flow.dtype) + flow

        return np.concatenate(
            [
                map_coordinates(image[..., channel], points, mode="nearest", order=1)[
                    ..., None
                ]
                for channel in range(channels)
            ],
            axis=-1,
        )

    # No channels
    assert image.ndim == flow.ndim - 1

    return map_coordinates(
        image, np.indices(image.shape, flow.dtype) + flow, mode="nearest", order=1
    )


def resize_flow(flow, shape):
    """
    Rescales the values of the vector field (u, v) to the desired shape.
    The values of the output vector field are scaled to the new resolution.

    Args:
        flow (np.ndarray): The displacement field
            Shape: # compléter
        shape (iterable): Couple of integers representing the output shape

    Returns:
        np.ndarray: The resized and rescaled motion field
    """

    scale = [n / o for n, o in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)

    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]

    rflow = scale_factor * ndi.zoom(
        flow, [1] + scale, order=1, mode="nearest", prefilter=False
    )

    return rflow


def get_pyramid(image, downscale=2.0, nlevel=10, min_size=16):
    """
    Construct image pyramid.

    Args:
        img (np.ndarray): The image to be preprocessed (Gray scale or RGB)
            Shape: # A compléter
        downscale (float): The pyramid downscale factor
            Default: 2
        nlevel (int): The maximum number of pyramid levels
            Default: 10
        min_size (int): The minimum soze for any dimension of the pyramid levels
            Default: 16

    Returns :
        list[ndarray]: The coarse to fine images pyramid
    """

    pyramid = [image]
    size = min(image.shape)
    count = 1

    while (count < nlevel) and (size > downscale * min_size):
        J = pyramid_reduce(pyramid[-1], downscale)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]


def forward_diff(f: np.ndarray, axis: int) -> np.ndarray:
    """
    Computes the forward difference of an array along a given axis.
    For element i: diff[i] = f[i+1] - f[i]
    The last element along the axis is set to zero.

    Args:
        f (np.ndarray): Input array
        axis (int): Axis along which to compute the difference

    Returns:
        np.ndarray: Array of the same shape as f containing forward differences
    """
    if axis < 0 or axis >= f.ndim:
        raise ValueError(f"axis must be between 0 and {f.ndim}")

    diff = np.zeros_like(f)

    slc_start = [slice(None)] * f.ndim
    slc_end = [slice(None)] * f.ndim

    # Select slices for forward difference
    slc_start[axis] = slice(0, -1)
    slc_end[axis] = slice(1, None)

    # Forward difference : f[i+1] - f[i]
    diff[tuple(slc_start)] = f[tuple(slc_end)] - f[tuple(slc_start)]

    # Last element remains zero

    return diff


def backward_diff(f: np.ndarray, axis: int) -> np.ndarray:
    """
    Computes the backward difference of an array along a given axis.
    For element i: diff[i] = f[i] - f[i-1]
    Special handling at the boundaries:
        diff[0] = f[0]
        diff[-1] = -f[-2]

    Args:
        f (np.ndarray): Input array
        axis (int): Axis along which to compute the difference

    Returns:
        np.ndarray: Array of the same shape as f containing backward differences
    """
    if axis < 0 or axis >= f.ndim:
        raise ValueError(f"axis must be between 0 and {f.ndim - 1}")

    diff = np.zeros_like(f)

    slc_prev = [slice(None)] * f.ndim
    slc_curr = [slice(None)] * f.ndim

    slc_prev[axis] = slice(0, -1)
    slc_curr[axis] = slice(1, None)

    # Main backward difference: f[i] - f[i-1]
    diff[tuple(slc_curr)] = f[tuple(slc_curr)] - f[tuple(slc_prev)]

    # diff[0] = f[0]
    start_idx = [slice(None)] * f.ndim
    start_idx[axis] = 0
    diff[tuple(start_idx)] = f[tuple(start_idx)]

    # diff[-1] = -f[-2]
    end_idx = [slice(None)] * f.ndim
    end_idx[axis] = -1
    prev_idx = [slice(None)] * f.ndim
    prev_idx[axis] = -2
    diff[tuple(end_idx)] = -f[tuple(prev_idx)]

    return diff


def nabla_h(h: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """
    Computes the discrete gradient (forward differences) of a d-dimensional flow field.

    Args:
        h (np.ndarray): Input array
            Shape: (d, T, *S)
        gamma (float, optional): Weight for the temporal gradient

    Returns:
        np.ndarray: Array containing gradients of each component along each axis
    """

    d = h.shape[0]  # displacement components

    ndim = h.ndim - 1  # number of axes excluding displacement index

    grads = np.zeros((d, ndim) + h.shape[1:], dtype=h.dtype)

    for i in range(d):
        for axis in range(ndim):
            # Forward differences along each axis for each component, including time
            grads[i, axis] = forward_diff(h[i], axis=axis)
            if axis == 0 and gamma is not None:
                grads[i, axis] *= gamma

    return grads


def nabla_star_h(grads: np.ndarray) -> np.ndarray:
    """
    Computes the discrete divergence (backward differences) of a d-dimensional gradient field.
    This is the adjoint of nabla_h.

    Args:
        grads (np.ndarray): Gradient array
            Shape: (d, d, T, *S)

    Returns:
        np.ndarray: Array containing the divergence of each component
            Shape: (d, T, *S)
    """

    d = grads.shape[0]
    ndim = grads.shape[1]
    div = np.zeros((d,) + grads.shape[2:], dtype=grads.dtype)

    for i in range(d):
        for axis in range(ndim):
            # Backward difference along each axis, summed over components, inclusing time
            div[i] -= backward_diff(grads[i, axis], axis=axis)

    return div


def hessian_(
    h: np.ndarray, nablah=np.ndarray, gamma: float | None = None
) -> np.ndarray:
    """
    Computes the discrete Hessian of a d-dimensional flow field.
    Uses forward differences twice: first to get gradients, then to get second derivatives.

    Args:
        h (np.ndarray): Input array
            Shape: (d, T, *S)
        gamma (float): Weight for the temporal gradient

    Returns:
        np.ndarray: Array containing second derivatives along each axis
            Shape: (d, d, d, T, *S)
    """
    d = h.shape[0]
    ndim = h.ndim - 1

    hessian = np.zeros((d, ndim, ndim) + h.shape[1:], dtype=h.dtype)

    for i in range(d):
        for axisder in range(ndim):
            for axishess in range(ndim):
                # Forward difference of the first derivative gives second derivative
                hessian[i, axisder, axishess] = forward_diff(
                    nablah[i, axisder], axis=axishess
                )
                if axishess == 0 and gamma is not None:
                    hessian[i, axisder, axishess] *= gamma

    return hessian


def hessian_star_h(hess: np.ndarray) -> np.ndarray:
    """
    Computes the adjoint operator of the discrete Hessian

    Args:
        hess (np.ndarray): Discrete Hessian tensor
            Shape: (d, d, d, T, *S)

    Returns:
        np.ndarray: Array corresponding to the adjoint
                    action of the Hessian on the vector field.
            Shape: (d, T, *S)
    """
    d = hess.shape[0]
    ndim = hess.shape[1]
    spatial_shape = hess.shape[3:]

    adjoint = np.zeros((d,) + spatial_shape, dtype=hess.dtype)

    for i in range(d):
        for axis1 in range(ndim):
            for axis2 in range(ndim):
                # Apply backward difference twice:
                # - once along the direction of last derivative
                # - once along direction of first derivative
                adjoint[i] += -backward_diff(
                    -backward_diff(hess[i, axis1, axis2], axis=axis2), axis=axis1
                )

    return adjoint


def nabla_f(
    h: np.ndarray,
    h0: np.ndarray,
    It: np.ndarray,
    nablaI: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Computes the gradient of the energy functional f(h) at the current estimate h.
    The functional includes:
        - Data term: ||It + nablaI · (h - h0)||^2
        - Gradient regularization: alpha^2 * ||∇h||^2
        - Hessian regularization: beta^2 * ||∇²h||^2  (if beta ≠ 0)

    Args:
        h (np.ndarray): Current displacement field
            Shape: (d, *S)
        h0 (np.ndarray): Reference displacement field (outer iteration)
            Shape: (d, *S)
        It (np.ndarray): Temporal difference image (ref - warped(moving))
            Shape: (*S)
        nablaI (np.ndarray): Spatial gradients of the warped moving image
            Shape (d, *S)
        alpha (float | np.ndarray): Regularization weight for gradient smoothness.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.
        beta (float | np.ndarray): Regularization weight for Hessian smoothness.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.

    Returns:
        np.ndarray: Gradient of f(h)
            Shape: (d, *S)
    """
    # Gradient of data attached term
    proj = (nablaI * (h - h0)).sum(axis=0)
    grad_data = 2 * nablaI * (It + proj)

    # Gradient of the gradient regularization term
    nablah = nabla_h(h, gamma)

    # Handle spatially-varying alpha (array) vs uniform (scalar)
    if np.ndim(alpha) == 0:
        grad_reg = 2 * (alpha**2) * nabla_star_h(nablah)
    else:
        # alpha is an array: element-wise multiplication
        # Add None dimension to broadcast over channel dimension of nabla_star_h output
        grad_reg = 2 * (alpha**2)[None, ...] * nabla_star_h(nablah)

    # Gradient of the hessian regularization term
    grad_reg_hess = 0

    if np.ndim(beta) == 0:
        if beta != 0:
            hessianh = hessian_(h, nablah, gamma)
            grad_reg_hess = 2 * (beta**2) * hessian_star_h(hessianh)
    else:
        # beta is an array: check if any non-zero values
        if np.any(beta != 0):
            hessianh = hessian_(h, nablah, gamma)
            grad_reg_hess = 2 * (beta**2)[None, ...] * hessian_star_h(hessianh)

    return grad_data + grad_reg + grad_reg_hess


def p_L(
    h: np.ndarray,
    h0: np.ndarray,
    lipschitz_constant: float,
    It: np.ndarray,
    nablaI: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """
    Single proximal gradient for FISTA step.

    Args:
        h (np.ndarray): Current displacement field
            Shape: (d, *S)
        h0 (np.ndarray): Reference displacement field (outer iteration)
            Shape: (d, *S)
        lipschitz_constant (float): Lipschitz constant of ∇f
        It (np.ndarray): Temporal difference image
            Shape: (*S)
        alpha (float | np.ndarray): Regularization weight for gradient smoothness.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.
        beta (float | np.ndarray): Regularization weight for Hessian smoothness.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.
        nablaI (np.ndarray): Spatial gradients of warped moving image
            Shape: (d, *S)

    Returns:
        np.ndarray: Updated displacement field after one proximal step
            Shape: (d, *S)
    """
    nabla_f_ = nabla_f(h, h0, It, nablaI, alpha, beta, gamma)
    return h - (1 / lipschitz_constant) * nabla_f_


def _fista(
    ref: np.ndarray,
    moving: np.ndarray,
    h0: np.ndarray,
    alpha: float | np.ndarray = 0.01,
    beta: float | np.ndarray = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Computes the optical flow between two images (reference and moving) using
    FISTA algorithm with iterative warping

    Args:
        ref (np.ndarray): Reference image
            Shape: (*S)
        moving (np.ndarray): Moving image to be aligned
            Shape: (*S)
        h0 (np.ndarray): Initial displacement field
            Shape: (d, *S)
        alpha (float | np.ndarray): Weight for gradient regularization.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.
        beta (float | np.ndarray): Weight for Hessian regularization.
            Can be a scalar or an array of the same shape as the image for spatially-varying regularization.
        num_iter (int): Number of FISTA iterations per warp step
        num_warp (int): Number of warping updates
        eps (float): Stopping criterion threshold

    Returns:
        np.ndarray: Optimized displacement field h
            Shape: (d, *S)

    References:
        Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems.
        SIAM Journal on Imaging Sciences, 2(1), 183–202
    """

    ref = ref.astype(np.float32)
    moving = moving.astype(np.float32)
    h0 = h0.astype(np.float32)
    h = h0.copy()

    # FISTA auxiliary variable
    y = h.copy()

    # FISTA momentum term
    t = 1
    t_new = t

    H, W = ref.shape
    nabla_for_l = nabla(2, (H, W))

    # Ensure alpha and beta are arrays if they're not scalars, matching image shape
    if np.ndim(alpha) > 0:
        alpha = np.asarray(alpha)
        if alpha.shape != ref.shape:
            raise ValueError(
                f"alpha shape {alpha.shape} must match image shape {ref.shape}"
            )
    if np.ndim(beta) > 0:
        beta = np.asarray(beta)
        if beta.shape != ref.shape:
            raise ValueError(
                f"beta shape {beta.shape} must match image shape {ref.shape}"
            )

    for _ in range(num_warp):
        # Warp the moving image with respect to the current displacement estimate h0
        im2_warped = warp(moving, h0)
        # Compute spatial gradients of the warped image
        nablaI = np.array(np.gradient(im2_warped))
        # Compute temporal intensity difference between reference and warped image
        It = im2_warped - ref

        # For Lipschitz constant computation, use scalar values (max of arrays if needed)
        alpha_scalar = np.max(alpha) if np.ndim(alpha) > 0 else alpha
        beta_scalar = np.max(beta) if np.ndim(beta) > 0 else beta
        lipschitz_constant = compute_lip(
            im2_warped, alpha_scalar, beta_scalar, nabla_for_l
        )

        for _ in range(num_iter):
            # Proximal gradient update
            h_new = p_L(y, h0, lipschitz_constant, It, nablaI, alpha, beta)

            # Update momentum parameter
            t_new = (1 / 2) * (1 + np.sqrt(1 + 4 * (t**2)))

            # Extrapolation step
            factor = (t - 1) / t_new
            y = h_new + factor * (h_new - h)

            # Convergence check based on mean squared change in h
            if np.linalg.norm(y - h) ** 2 <= eps**2:
                break

            # Update h and t for new inner iteration
            h = h_new.copy()
            t = t_new.copy()

        # Update initial displacement field for next outer iteration
        h0 = h.copy()

    return h0


def fista(
    reference: np.ndarray,
    moving: np.ndarray,
    alpha: float | np.ndarray = 0.01,
    beta: float | np.ndarray = 0,
    num_iter: int = 100,
    num_warp: int = 2,
    num_pyramid: int = 10,
    pyramid_downscale: float | np.ndarray = 2.0,
    pyramid_min_size: int | np.ndarray = 16,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Computes optical flow between two images (reference and moving) using
    FISTA algorithm with a multi-scale image pyramid for efficient
    large-displacement handling.

    The algorithm works by:
        1. Building pyramids of the reference and moving images.
        2. Estimating the displacement field (flow) starting at the coarsest level.
        3. Progressively refining the flow at finer levels using FISTA with warping.

    Args:
        reference (np.ndarray): Reference image
            Shape: (*S)
        moving (np.ndarray): Moving image to be aligned to the reference
            Shape: (d, *S)
        alpha (float | np.ndarray, optional): Regularization weight for gradient smoothness.
            Can be a scalar or an array of the same shape as the reference image for spatially-varying regularization.
            When using a multi-scale pyramid, the array is automatically resized at each level.
            Default: 0.01
        beta (float | np.ndarray, optional): Regularization weight for Hessian smoothness.
            Can be a scalar or an array of the same shape as the reference image for spatially-varying regularization.
            When using a multi-scale pyramid, the array is automatically resized at each level.
            Default: 0
        num_iter (int, optional): Number of FISTA iterations per warp step (inner iterations)
            Default: 100
        num_warp (int, optional): Number of warp updates per pyramid level (outer iterations)
            Default: 2
        num_pyramid (int, optional): Maximum number of pyramid levels
            Default: 10
        pyramid_downscale (Union[float, np.ndarray], optional): Scaling factor or array for downsampling
                            at each pyramid level.
            Default: 2.0
        pyramid_min_size (Union[int, np.ndarray], optional): Minimum size of images in the pyramid
            Default: 16
        eps (float, optional): Stopping criterion threshold
            Default: 1e-5

    Returns:
        np.ndarray: Estimated displacement field h, same shape as input images

    References:
        Meinhardt-Llopis, E., & Sánchez, J. (2013). Horn-schunck optical flow with a multi-scale strategy.
        Image Processing on line.
    """

    # Input images must be of the same shape
    assert moving.shape == reference.shape

    # Dimension of the input images (e.g., 2D or 3D)
    d = reference.ndim

    # Check if alpha and beta are arrays and validate shapes
    alpha_is_array = np.ndim(alpha) > 0
    beta_is_array = np.ndim(beta) > 0

    if alpha_is_array:
        alpha = np.asarray(alpha)
        assert alpha.shape == reference.shape, (
            f"alpha shape {alpha.shape} must match reference shape {reference.shape}"
        )
    if beta_is_array:
        beta = np.asarray(beta)
        assert beta.shape == reference.shape, (
            f"beta shape {beta.shape} must match reference shape {reference.shape}"
        )

    # Build the image pyramids for both reference and moving images
    pyramid_ = list(
        zip(
            get_pyramid(reference, pyramid_downscale, num_pyramid, pyramid_min_size),
            get_pyramid(moving, pyramid_downscale, num_pyramid, pyramid_min_size),
        )
    )

    # Initialize displacement field (optical flow) at the coarsest pyramid level
    h0 = np.zeros(
        (d,) + pyramid_[0][0].shape
    )  # Shape matching the coarsest reference level

    # For coarsest level, resize alpha/beta arrays if needed
    pyr_ref_0, pyr_mov_0 = pyramid_[0]
    if alpha_is_array:
        alpha_level = ndi.zoom(
            alpha,
            [s / m for s, m in zip(pyr_ref_0.shape, reference.shape)],
            order=1,
            mode="nearest",
        )
    else:
        alpha_level = alpha
    if beta_is_array:
        beta_level = ndi.zoom(
            beta,
            [s / m for s, m in zip(pyr_ref_0.shape, reference.shape)],
            order=1,
            mode="nearest",
        )
    else:
        beta_level = beta

    # Compute optical flow at the coarsest pyramid level
    h = _fista(
        pyr_ref_0,
        pyr_mov_0,
        h0,
        alpha_level,
        beta_level,
        num_iter=num_iter,
        num_warp=num_warp,
        eps=eps,
    )

    # Progressively refine the optical flow up the pyramid levels
    for pyr_ref, pyr_mov in pyramid_[1:]:
        # Resize the flow field to the current pyramid level's dimensions
        h = resize_flow(h, pyr_ref.shape)

        # Resize alpha/beta arrays to current pyramid level if needed
        if alpha_is_array:
            alpha_level = ndi.zoom(
                alpha,
                [s / m for s, m in zip(pyr_ref.shape, reference.shape)],
                order=1,
                mode="nearest",
            )
        else:
            alpha_level = alpha
        if beta_is_array:
            beta_level = ndi.zoom(
                beta,
                [s / m for s, m in zip(pyr_ref.shape, reference.shape)],
                order=1,
                mode="nearest",
            )
        else:
            beta_level = beta

        # Update the flow field using the HS algorithm at the current pyramid level
        h = _fista(
            pyr_ref,
            pyr_mov,
            h,
            alpha_level,
            beta_level,
            num_iter=num_iter,
            num_warp=num_warp,
            eps=eps,
        )

    return h


### Functions related to Horn-Schunck optical flow with a multiscale approach


def _create_average_kernel(dimension: int):
    """
    Creates an average kernel for a specified dimension. The values of the discrete
    kernel for 2D can be found in the seminal paper by Horn-Schunck.

    Args:
        dimension (int): The dimension of the kernel. Acceptable values are:
                         1 (for a 1D kernel), 2 (for a 2D kernel), or 3 (for a 3D kernel).

    Returns:
        np.ndarray: A NumPy array representing the averaging kernel for the specified dimension.

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
            Artificial intelligence, 17(1-3), 185-203
    """
    # Create a 1D average kernel if dimension is 1
    if dimension == 1:
        # Returns a 1D array where non-zero elements are set to 1/6
        return np.array([1 / 2, 0, 1 / 2], dtype=np.float32)

    # Create a 2D average kernel if dimension is 2
    elif dimension == 2:
        return np.array(
            [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
            dtype=np.float32,
        )

    # Create a 3D average kernel if dimension is 3
    elif dimension == 3:
        return np.array(
            [
                [
                    [1 / 24, 1 / 12, 1 / 24],
                    [1 / 12, 1 / 6, 1 / 12],
                    [1 / 24, 1 / 12, 1 / 24],
                ],
                [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
                [
                    [1 / 24, 1 / 12, 1 / 24],
                    [1 / 12, 1 / 6, 1 / 12],
                    [1 / 24, 1 / 12, 1 / 24],
                ],
            ],
            dtype=np.float32,
        )

    else:
        raise ValueError("Dimension must be 1, 2, or 3.")


def _hs_optical_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    u0: np.ndarray,
    alpha: float,
    num_iter: 100,
    num_warp=2,
    eps=1e-5,
    w=1.0,
    dtype=np.float32,
):
    """
    Computes the optical flow between two images (reference and moving) using
    the Horn-Schunck (HS) algorithm with iterative warping.

    This algorithm applies to small displacements only and shall be paired with
    a multiscale strategy for larger displacements.

    The sparse system ensuing from the resolution of the discrete Euler-Lagrange equations is solved
    by Successive Over-Relaxation (SOR), which generalizes the Jacobi iteration initially presented in the
    seminal paper of Horn and Schunck.

    Args:
        reference (np.ndarray): The reference (static) image to which the moving image is aligned.
            Shape: (M, N,...)
        moving (np.ndarray): The moving image that is being warped towards the reference.
            Shape: (M, N,...)
        u0 (np.ndarray): The initial displacement field guess.
            Shape: Shape: (d, M, N,...)
        alpha (float): Regularization parameter controlling smoothness of the flow field.
        num_iter (int): Number of iterations for the HS algorithm.
            Default: 100
        num_warp (int, optional): Number of warping steps for iterative refinement.
            Default: 2
        eps (float, optional): Convergence tolerance. Algorithm stops if mean squared change in
                               displacement field is below this threshold.
            Default: 1e-5.
        w (float, optional): Relaxation parameter for SOR.  Values shall be in (0,2).
                            A value of 1 corresponds to Jacobi iteration.
            Default: 1.
        dtype (np.dtype, optional): Data type for computation (default is np.float32).

    Returns:
        np.ndarray: The computed displacement field (optical flow) that warps the moving image
                    towards the reference image, with shape matching u0.

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
            Artificial intelligence, 17(1-3), 185-203
    """
    # Input images must be of the same shape
    assert moving.shape == reference.shape

    # Check if relaxation parameter is in the admissible range
    if not 0 < w < 2:
        raise ValueError(f"Over relaxation parameter should be in (0,2). Found: {w}")

    # Dimension of input images (1D, 2D or 3D), assumed from reference image
    dim = reference.ndim

    # Cast input arrays to specified dtype for consistency in computation
    u0 = u0.astype(dtype)
    reference = reference.astype(dtype)
    moving = moving.astype(dtype)

    # Initialize displacement field u as zero matrix with same shape as u0
    u = u0.copy()

    # Create a Laplacian kernel for smoothing the flow field, based on image dimensions
    laplace_kernel = _create_average_kernel(reference.ndim)
    laplace_kernel /= np.sum(laplace_kernel)

    # Iterative warping loop, to refine the displacement field (optical flow)
    for _ in range(num_warp):
        # Warp the moving image according to the current displacement estimate u0
        im2_warped = warp(moving, u0)

        # Compute spatial gradients of the warped image
        nabla_I = np.array(np.gradient(im2_warped))

        # Compute temporal intensity difference between reference and warped image
        im_t = im2_warped - reference

        # Inner loop for the HS algorithm to iteratively refine the flow field
        for _ in range(num_iter):
            # Compute smoothed version of the current displacement field u using convolution
            u_average = convolve(
                np.pad(u, pad_width=((0, 0),) + tuple([(1, 1)] * dim), mode="edge"),
                laplace_kernel[None, ...],
                mode="valid",
            )

            # Derivative-based term used to update the displacement field, balancing data and smoothness
            der = ((nabla_I * (u_average - u0)).sum(axis=0) + im_t) / (
                (nabla_I * nabla_I).sum(axis=0) + alpha
            )

            # Update displacement field by subtracting gradient-scaled derivative
            u_new = u_average - nabla_I * der

            # Convergence check based on mean squared change in u
            if np.mean((u_new - u) ** 2) < eps**2:
                break

            # Update u with a weighted relaxation of u_new for stability and convergence control
            u = w * u_new + (1 - w) * u

        # Update initial displacement field for next warping step
        u0 = np.array(u, dtype=dtype)

    return u0


def hs_optical_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    alpha: float,
    num_iter=100,
    num_warp=2,
    num_pyramid=10,
    pyramid_downscale: float | np.ndarray = 2.0,
    pyramid_min_size: int | np.ndarray = 16,
    eps=1e-5,
    w=1.0,
):
    """
    Computes optical flow between two images (reference and moving) using the
    Horn-Schunck (HS) algorithm with a multi-scale image pyramid for efficient
    large-displacement handling.

    Args:
        reference (np.ndarray): The reference (static) image to which the moving image is aligned.
            Shape: (M, N,...)
        moving (np.ndarray): The moving image to be aligned with the reference image.
            Shape: (M, N,...)
        alpha (float): Regularization parameter that controls the smoothness of the computed flow.
        num_iter (int, optional): Number of iterations at each pyramid level for the HS algorithm.
            Default: 100
        num_warp (int, optional): Number of warping steps for each pyramid level.
            Default: 2
        num_pyramid (int, optional): Number of pyramid levels for multi-scale processing.
            Default: 10
        pyramid_downscale (Union[float, np.ndarray], optional): Scaling factor or array for downsampling
                             at each pyramid level.
            Default: 2
        pyramid_min_size (Union[int, np.ndarray], optional): Minimum size for images in the pyramid to stop downscaling.
            Default: 16
        eps (float, optional): Convergence threshold based on mean squared change in flow field.
            Default: 1e-5
        w (float, optional): Relaxation parameter for SOR.  Values shall be in (0,2).
                            A value of 1 corresponds to Jacobi iteration.
            Default: 1

    Returns:
        np.ndarray: The computed displacement field (optical flow) that aligns the moving image to the reference image.

    References:
        Meinhardt-Llopis, E., & Sánchez, J. (2013). Horn-schunck optical flow with a multi-scale strategy.
        Image Processing on line.
    """
    # Input images must be of the same shape
    assert moving.shape == reference.shape

    # Check if relaxation parameter is in the admissible range
    if not 0 < w < 2:
        raise ValueError(f"Over relaxation parameter should be in (0,2). Found: {w}")

    # Dimension of the input images (e.g., 2D or 3D)
    d = reference.ndim

    # Build the image pyramids for both reference and moving images
    pyramid_ = list(
        zip(
            get_pyramid(reference, pyramid_downscale, num_pyramid, pyramid_min_size),
            get_pyramid(moving, pyramid_downscale, num_pyramid, pyramid_min_size),
        )
    )

    # Initialize displacement field (optical flow) at the coarsest pyramid level
    u0 = np.zeros(
        (d,) + pyramid_[0][0].shape
    )  # Shape matching the coarsest reference level

    # Compute optical flow at the coarsest pyramid level
    u = _hs_optical_flow(
        pyramid_[0][0],
        pyramid_[0][1],
        u0,
        alpha,
        num_iter=num_iter,
        num_warp=num_warp,
        eps=eps,
        w=w,
    )

    # Progressively refine the optical flow up the pyramid levels
    for pyr_ref, pyr_mov in pyramid_[1:]:
        # Resize the flow field to the current pyramid level's dimensions
        u = resize_flow(u, pyr_ref.shape)

        # Update the flow field using the HS algorithm at the current pyramid level
        u = _hs_optical_flow(
            pyr_ref,
            pyr_mov,
            u,
            alpha,
            num_iter=num_iter,
            num_warp=num_warp,
            eps=eps,
            w=w,
        )

    return u


# Functions to run the algorithms on an film with several frames


def fista_of(
    img: np.ndarray, fista_params: FistaParams, global_flow: bool
) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of an image sequence using FISTA.

    This function iteratively estimates the displacement field between each pair of
    successive frames in a temporal image sequence, producing a dense optical flow field
    across time.

    Args:
        img (np.ndarray): Input image sequence
        fista_params (Namespace or FistaParams):
            Parameter object containing the configuration for FISTA-optical flow:
              - `num_iter` (int): Number of iterations per level.
              - `num_warp` (int): Number of warping refinements.
              - `alpha` (float): Regularization parameter for the gradient.
              - `beta` (float): Regularization parameter for the hessian.
              - `eps` (float): Convergence tolerance for early stopping.
              - `num_pyramid` (int): Number of scales in the image pyramid.
              - `pyramid_downscale` (float): Scale factor between pyramid levels.
              - `pyramid_min_size` (int): Minimum spatial size at the coarsest pyramid level.
        global_flow (bool): Global or local flow
              - True for flow between 0 and t for every t
              - False for flow between t and t+1 for every t

    Returns:
        np.ndarray:
            Estimated optical flow field `h_FISTA` with shape
            `(D, T-1, *S)`, where:
              - `D` is the dimensionality of the flow (e.g., 2 for 2D, 3 for 3D),
              - `T-1` corresponds to the number of computed frame-to-frame displacements,
              - `*S` matches the spatial dimensions of the input frames.
    """
    dimension = len(img[0].shape)

    frames = img.shape[0]
    if global_flow:
        h_FISTA = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
        for t in range(1, frames):
            h_FISTA[:, t] = fista(
                img[0],
                img[t],
                alpha=fista_params.alpha,
                beta=fista_params.beta,
                num_iter=fista_params.num_iter,
                num_warp=fista_params.num_warp,
                num_pyramid=fista_params.num_pyramid,
                pyramid_downscale=fista_params.pyramid_downscale,
                pyramid_min_size=fista_params.pyramid_min_size,
                eps=fista_params.eps,
            )
    else:
        h_FISTA = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
        for t in range(frames - 1):
            h_FISTA[:, t] = fista(
                img[t],
                img[t + 1],
                alpha=fista_params.alpha,
                beta=fista_params.beta,
                num_iter=fista_params.num_iter,
                num_warp=fista_params.num_warp,
                num_pyramid=fista_params.num_pyramid,
                pyramid_downscale=fista_params.pyramid_downscale,
                pyramid_min_size=fista_params.pyramid_min_size,
                eps=fista_params.eps,
            )
    return h_FISTA


def hs_of(img: np.ndarray, hs_params: HSParams, global_flow: bool) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of an image sequence using Horn and Schunck's algorithm.

    This function iteratively estimates the displacement field between each pair of
    successive frames in a temporal image sequence, producing a dense optical flow field
    across time.

    Args:
        img (np.ndarray): Input image sequence as a NumPy array of shape
        hs_params (Namespace or HSParams):
            Parameter object containing the configuration for HS:
              - `num_iter` (int): Number of iterations per level.
              - `num_warp` (int): Number of warping refinements.
              - `alpha` (float): Regularization parameter for the gradient.
              - `eps` (float): Convergence tolerance for early stopping.
              - `num_pyramid` (int): Number of scales in the image pyramid.
              - `pyramid_downscale` (float): Scale factor between pyramid levels.
              - `pyramid_min_size` (int): Minimum spatial size at the coarsest pyramid level.
              - `w` (float):  Relaxation parameter for SOR.  Values shall be in (0,2).
                    A value of 1 corresponds to Jacobi iteration.
        global_flow (bool): Global or local flow
              - True for flow between 0 and t for every t
              - False for flow between t and t+1 for every t

    Returns:
        np.ndarray:
            Estimated optical flow field `h_hs` with shape
            `(D, T-1, *S)`
    """
    dimension = len(img[0].shape)
    frames = img.shape[0]
    if global_flow:
        h_hs = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
        for t in range(1, frames):
            h_hs[:, t] = hs_optical_flow(
                img[0],
                img[t],
                alpha=hs_params.alpha,
                num_iter=hs_params.num_iter,
                num_warp=hs_params.num_warp,
                num_pyramid=hs_params.num_pyramid,
                pyramid_downscale=hs_params.pyramid_downscale,
                eps=hs_params.eps,
                w=hs_params.w,
            )
    else:
        h_hs = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
        for t in range(frames - 1):
            h_hs[:, t] = hs_optical_flow(
                img[t],
                img[t + 1],
                alpha=hs_params.alpha,
                num_iter=hs_params.num_iter,
                num_warp=hs_params.num_warp,
                num_pyramid=hs_params.num_pyramid,
                pyramid_downscale=hs_params.pyramid_downscale,
                eps=hs_params.eps,
                w=hs_params.w,
            )
    return h_hs


### Function related to Farneback Optical-Flow


def farneback(
    img: np.ndarray, fb_params: FarnebackParams, global_flow: bool
) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of an image sequence using Farneback's algorithm.

    This function iteratively estimates the displacement field between each pair of
    successive frames in a temporal image sequence, producing a dense optical flow field
    across time.

    Args:
        img (np.ndarray): Input image sequence
        farneback_params (Namespace or FarnebackParams):
            Parameter object containing the configuration for the Farneback solver, with fields:
              - `winSize` (int): Size of the averaging window used for polynomial expansion.
              - `pyrScale` (float): Image scale (<1) between pyramid levels.
              - `numLevels` (int): Number of pyramid layers.
              - `fastPyramids` (bool): Whether to use fast pyramid construction.
              - `numIters` (int): Number of iterations at each pyramid level.
              - `polyN` (int): Size of the pixel neighborhood for polynomial expansion.
              - `polySigma` (float): Standard deviation of Gaussian used for smoothing derivatives.
              - `flags` (int): Operation flags for OpenCV’s implementation.
        global_flow (bool): Global or local flow
              - True for flow between 0 and t for every t
              - False for flow between t and t+1 for every t


    Returns:
        np.ndarray:
            Estimated optical flow field `h_f` with shape
            `(D, T-1, *S)`, where:
              - `D` is the dimensionality of the flow (e.g., 2 for 2D, 3 for 3D),
              - `T-1` corresponds to the number of computed frame-to-frame displacements,
              - `*S` matches the spatial dimensions of the input frames.
    """
    dimension = len(img[0].shape)
    optflow = OpenCVOpticalFlow(
        cv2.FarnebackOpticalFlow.create(  # pylint: disable=no-member
            winSize=fb_params.winSize,
            pyrScale=fb_params.pyrScale,
            numLevels=fb_params.numLevels,
            fastPyramids=fb_params.fastPyramids,
            numIters=fb_params.numIters,
            polyN=fb_params.polyN,
            polySigma=fb_params.polySigma,
            flags=fb_params.flags,
        ),
        downscale=1,
    )
    frames = img.shape[0]
    if global_flow:
        h_f = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
        src = optflow.preprocess(img[0, ..., None])
        for t in range(1, frames):
            dst = optflow.preprocess(img[t, ..., None])
            h_f[:, t] = optflow.compute(src, dst)
            src = dst
    else:
        h_f = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
        for t in range(frames - 1):
            src = optflow.preprocess(img[t, ..., None])
            dst = optflow.preprocess(img[t + 1, ..., None])
            h_f[:, t] = optflow.compute(src, dst)
            src = dst
    return h_f


### Function related to TV-L1 optical flow


def tv_l1(img: np.ndarray, tvl1_params: TVL1Params, global_flow: bool) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of an image sequence using a TV-L1 approach.

    This function iteratively estimates the displacement field between each pair of
    successive frames in a temporal image sequence, producing a dense optical flow field
    across time.

    Args:
        img (np.ndarray): Input image sequence
        tvl1_params (Namespace or TVL1Params):
            Parameter object containing the configuration for the TV-L1 solver, with fields:
              - `attachment` (float): Data fidelity weight balancing the L1 term.
              - `tightness` (float): Regularization strength controlling smoothness.
              - `num_warp` (int): Number of warping steps per iteration.
              - `num_iter` (int): Number of optimization iterations.
              - `tol` (float): Convergence tolerance for early stopping.
              - `prefilter` (bool): Whether to apply Gaussian prefiltering for stability.
        global_flow (bool): Global or local flow
              - True for flow between 0 and t for every t
              - False for flow between t and t+1 for every t

    Returns:
        np.ndarray:
            Estimated optical flow field `h_tvl1` with shape
            `(D, T-1, *S)`, where:
              - `D` is the dimensionality of the flow (e.g., 2 for 2D, 3 for 3D),
              - `T-1` corresponds to the number of computed frame-to-frame displacements,
              - `*S` matches the spatial dimensions of the input frames.
    """
    dimension = len(img[0].shape)
    frames = img.shape[0]
    if global_flow:
        h_tvl1 = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
        for t in range(1, frames):
            h_tvl1[:, t] = optical_flow_tvl1(
                img[0],
                img[t],
                attachment=tvl1_params.attachment,
                tightness=tvl1_params.tightness,
                num_warp=tvl1_params.num_warp,
                num_iter=tvl1_params.num_iter,
                tol=tvl1_params.tol,
                prefilter=tvl1_params.prefilter,
            )
    else:
        h_tvl1 = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
        for t in range(frames - 1):
            h_tvl1[:, t] = optical_flow_tvl1(
                img[t],
                img[t + 1],
                attachment=tvl1_params.attachment,
                tightness=tvl1_params.tightness,
                num_warp=tvl1_params.num_warp,
                num_iter=tvl1_params.num_iter,
                tol=tvl1_params.tol,
                prefilter=tvl1_params.prefilter,
            )
    return h_tvl1


# Function related to ILK optical flow


def ilk(img: np.ndarray, ilk_params: ILKParams, global_flow: bool) -> np.ndarray:
    """
    Computes optical flow between consecutive frames of an image sequence using a TV-L1 approach.

    This function iteratively estimates the displacement field between each pair of
    successive frames in a temporal image sequence, producing a dense optical flow field
    across time.

    Args:
        img (np.ndarray): Input image sequence
        ilk_params (Namespace or ILKParams):
            Parameter object containing the configuration for the ILK solver, with fields:
              - `radius` (float): Radius of the local window used for gradient averaging.
              - `num_warp` (int): Number of image warping refinements per iteration.
              - `gaussian` (bool): Whether to use Gaussian weighting within the window.
              - `prefilter` (bool): Whether to prefilter images to suppress noise.
        global_flow (bool): Global or local flow
              - True for flow between 0 and t for every t
              - False for flow between t and t+1 for every t

    Returns:
        np.ndarray:
            Estimated optical flow field `h_ilk` with shape
            `(D, T-1, *S)`, where:
              - `D` is the dimensionality of the flow (e.g., 2 for 2D, 3 for 3D),
              - `T-1` corresponds to the number of computed frame-to-frame displacements,
              - `*S` matches the spatial dimensions of the input frames.
    """
    dimension = len(img[0].shape)
    h_ilk = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
    frames = img.shape[0]
    if global_flow:
        h_ilk = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
        for t in range(1, frames):
            h_ilk[:, t] = optical_flow_ilk(
                img[0],
                img[t],
                radius=ilk_params.radius,
                num_warp=ilk_params.num_warp,
                gaussian=ilk_params.gaussian,
                prefilter=ilk_params.prefilter,
            )
    else:
        h_ilk = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
        for t in range(frames - 1):
            h_ilk[:, t] = optical_flow_ilk(
                img[t],
                img[t + 1],
                radius=ilk_params.radius,
                num_warp=ilk_params.num_warp,
                gaussian=ilk_params.gaussian,
                prefilter=ilk_params.prefilter,
            )
    return h_ilk


# Functions related to PIV algorithm


# def inverse_transform_coordinates(x, y, u, v):
#     if y.ndim == 1:
#         y = y[::-1]
#     else:
#         y = y[::-1, :]
#     return x, y, u, v


# def get_dense_displacement(
#     frame_a,
#     frame_b,
#     window_size=8,
#     overlap=2,
#     search_area=16,
#     s2n_thresh=1.3,
#     method="cubic",
# ):
#     """
#     Get pixel-level dense displacement field
#     """

#     frame_a_norm = (
#         (frame_a - frame_a.min()) / (frame_a.max() - frame_a.min()) * 255
#     ).astype(np.int32)
#     frame_b_norm = (
#         (frame_b - frame_b.min()) / (frame_b.max() - frame_b.min()) * 255
#     ).astype(np.int32)

#     x, y, u, v, s2n = piv.simple_piv(
#         frame_a_norm,
#         frame_b_norm,
#         window_size=window_size,
#         overlap=overlap,
#         search_area_size=search_area,
#         dt=1.0,
#         validation_method="sig2noise",
#         plot=False,
#     )

#     x, y, u, v = inverse_transform_coordinates(x, y, u, v)

#     valid = s2n > s2n_thresh

#     x_valid = x[valid]
#     y_valid = y[valid]
#     u_valid = u[valid]
#     v_valid = v[valid]

#     h, w = frame_a.shape
#     y_dense, x_dense = np.mgrid[0:h:1, 0:w:1]

#     u_dense = griddata(
#         (x_valid, y_valid), u_valid, (x_dense, y_dense), method=method, fill_value=0
#     )
#     v_dense = griddata(
#         (x_valid, y_valid), v_valid, (x_dense, y_dense), method=method, fill_value=0
#     )

#     u_dense = np.nan_to_num(u_dense, nan=0.0)
#     v_dense = np.nan_to_num(v_dense, nan=0.0)

#     return u_dense, v_dense, (x, y, u, v, valid)


# def piv_algo(img: np.ndarray, piv_params: PIVParams, global_flow: bool) -> np.ndarray:
#     """
#     Computes dense displacement field between consecutive frames using PIV.

#     This function iteratively estimates the displacement field between each pair of
#     successive frames in a temporal image sequence, producing a dense optical flow field
#     across time.

#     Args:
#         img (np.ndarray): Input image sequence with shape (T, H, W)
#             where T is number of frames, H is height, W is width
#         piv_params (PIVParams):
#             Parameter object containing the configuration for PIV, with fields:
#               - `window_size` (int): Size of interrogation window
#               - `overlap` (int): Overlap of interrogation windows
#               - `search_area` (int): Size of search area
#               - `s2n_thresh` (float): Signal-to-noise ratio threshold
#               - `method` (str): Interpolation method ('cubic', 'linear', etc.)
#         global_flow (bool): Type of flow computation
#               - True for flow between frame 0 and frame t for every t
#               - False for flow between frame t and frame t+1 for every t

#     Returns:
#         np.ndarray:
#             Estimated displacement field `h_piv` with shape:
#               - `(2, T-1, H, W)` if global_flow=False
#               - `(2, T, H, W)` if global_flow=True

#             where:
#               - 2 represents (u, v) components
#               - T is number of frames
#               - H, W are spatial dimensions matching input frames
#     """

#     dimension = 2  # u, v components
#     frames = img.shape[0]

#     if global_flow:
#         h_piv = np.zeros((dimension,) + (img.shape[0],) + img.shape[1:])
#         for t in range(1, frames):
#             u_dense, v_dense, _ = get_dense_displacement(
#                 img[0],
#                 img[t],
#                 window_size=piv_params.window_size,
#                 overlap=piv_params.overlap,
#                 search_area=piv_params.search_area,
#                 s2n_thresh=piv_params.s2n_thresh,
#                 method=piv_params.method,
#             )
#             h_piv[0, t] = u_dense
#             h_piv[1, t] = v_dense
#     else:
#         h_piv = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])
#         for t in range(frames - 1):
#             u_dense, v_dense, _ = get_dense_displacement(
#                 img[t],
#                 img[t + 1],
#                 window_size=piv_params.window_size,
#                 overlap=piv_params.overlap,
#                 search_area=piv_params.search_area,
#                 s2n_thresh=piv_params.s2n_thresh,
#                 method=piv_params.method,
#             )
#             h_piv[0, t] = u_dense
#             h_piv[1, t] = v_dense

#     return h_piv


def raft_algo(img: np.ndarray) -> np.ndarray:
    """
    Computed the displacement field between the frames of an image
    of shape (2, H, W)

    Args:
        img (np.ndarray): The image

    Returns:
        np.ndarray:
            Estimated displacement field `h_raft` with shape: `(2, 1, H, W)`
    """

    def pad_to_multiple_8(arr):
        h, w = arr.shape

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")

        return arr_padded, h, w

    def unpad_flow(flow, orig_h, orig_w):
        return flow[:, :, :orig_h, :orig_w]

    dimension = 2

    frame_1_padded, h, w = pad_to_multiple_8(img[0])
    frame_2_padded, _, _ = pad_to_multiple_8(img[1])

    # frame_1 shape: (H, W) -> need (1, C, H, W)
    img1 = torch.from_numpy(frame_1_padded).unsqueeze(0).unsqueeze(0)  # (1, C, H, W)
    img2 = torch.from_numpy(frame_2_padded).unsqueeze(0).unsqueeze(0)  # (1, C, H, W)

    img1 = img1.repeat(1, 3, 1, 1)  # (1, 3, H, W)
    img2 = img2.repeat(1, 3, 1, 1)  # (1, 3, H, W)

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    img1, img2 = transforms(img1, img2)

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        list_of_flows = model(img1, img2)

    predicted_flow = list_of_flows[-1]
    predicted_flow_np = predicted_flow.cpu().numpy()

    predicted_flow_unpadded = unpad_flow(predicted_flow_np, h, w)

    h_raft = np.zeros((dimension,) + (img.shape[0] - 1,) + img.shape[1:])

    h_raft[0, 0] = predicted_flow_unpadded[0, 1]
    h_raft[1, 0] = predicted_flow_unpadded[0, 0]

    return h_raft
