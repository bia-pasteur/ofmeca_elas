"""Some utils functions"""

# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import tifffile
from scipy.interpolate import griddata
from scipy.io import loadmat
from shapely import Polygon, contains_xy

from mechanics.src.optical_flow.algorithms import fista, warp


def extract_E_from_folder(folder_name: str) -> float:
    """
    Extracts the Young's modulus (E) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted Young’s modulus `E`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `_E_` pattern.
    """
    match = re.search(r"_E_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract E from {folder_name}")
    return float(match.group(1))


def extract_T_from_folder(folder_name: str) -> float:
    """
    Extracts the applied Traction (T) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted applied Traction `T`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `T_` pattern.
    """
    match = re.search(r"T_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract T from {folder_name}")
    return float(match.group(1))


def extract_nu_from_folder(folder_name: str) -> float:
    """
    Extracts the Poisson's ratio (nu) value from a folder name of the format `'T_X_E_Y_nu_Z'`.

    Args:
        folder_name (str):
            Name of the folder containing the encoded parameter values.

    Returns:
        float:
            The extracted Poisson's ratio `nu`.

    Raises:
        ValueError:
            If the folder name does not contain a valid `_nu_` pattern.
    """
    match = re.search(r"_nu_([0-9]*\.?[0-9]+)", folder_name)
    if match is None:
        raise ValueError(f"Could not extract nu from {folder_name}")
    return float(match.group(1))


def find_experiment_folder(base_path: Path, T: float, E: float, nu: float) -> Path:
    """
    Searches recursively for an experiment folder matching the parameters T, E, and ν.

    The folder structure is expected to follow the convention `'T_X_E_Y_nu_Z'`.

    Args:
        base_path (Path):
            Base directory containing subfolders for different experiments.
        T (float):
            Tension or traction parameter to match.
        E (float):
            Young’s modulus to match.
        nu (float):
            Poisson’s ratio to match.

    Returns:
        Path:
            The path to the matching experiment folder.

    Raises:
        FileNotFoundError:
            If no folder matching the given parameters is found.
    """
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue
        for folder in subdir.iterdir():
            if not folder.is_dir():
                continue
            if folder.is_dir():
                t_val = extract_T_from_folder(folder.name)
                e_val = extract_E_from_folder(folder.name)
                nu_val = extract_nu_from_folder(folder.name)
                if (
                    np.isclose(t_val, T)
                    and np.isclose(e_val, E)
                    and np.isclose(nu_val, nu)
                ):
                    return folder
    raise FileNotFoundError(f"No folder found for T={T}, E={E}, nu={nu}")


def extract_std_from_file(file_name: str) -> float:
    """
    Extracts the noise standard deviation value from a file name of the format `'std_0p05_img.npy'`.

    Args:
        file_name (str):
            Name of the file containing the encoded noise level.

    Returns:
        float:
            The extracted noise standard deviation.

    Raises:
        ValueError:
            If the file name does not contain a valid `std_` pattern.
    """
    match = re.search(r"std_(\d+(?:p\d+)?)_img\.npy", file_name)
    if match is None:
        raise ValueError(f"Could not extract std from {file_name}")
    std_str = match.group(1).replace("p", ".")
    return float(std_str)


def get_all_stds_from_folder(folder_path: str) -> list[float]:
    """
    Scans a folder and retrieves all standard deviation values from file names matching the pattern `'std_*_img.npy'`.

    Args:
        folder_path (str):
            Path to the folder containing noisy image files.

    Returns:
        list[float]:
            Sorted list of extracted noise standard deviations.
    """
    stds = []
    for file_name in os.listdir(folder_path):
        if "std_" in file_name and file_name.endswith("_img.npy"):
            std = extract_std_from_file(file_name)
            stds.append(std)
    return sorted(stds)


def load_images_and_displacements(
    exp_folder: Path,
    mode: str = "original",
    load_wofv: bool = False,
    load_piv: bool = False,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray] | None,
    list[np.ndarray] | None,
]:
    exp_folder = Path(exp_folder)

    if mode == "original":
        img_files = sorted(
            exp_folder.rglob("*_img.npy"),
        )
        disp_files = sorted(
            exp_folder.rglob("*_ugt.npy"),
        )
        wofv_files = sorted(
            exp_folder.rglob("*_wofv.mat"),
        )
        piv_files = sorted(
            exp_folder.rglob("*_piv.mat"),
        )

    elif mode == "noisy":
        img_files = sorted(
            exp_folder.rglob("std_*_img.npy"),
        )
        disp_files = sorted(
            exp_folder.rglob("std_*_ugt.npy"),
        )
        wofv_files = sorted(
            exp_folder.rglob("std_*_wofv.mat"),
        )
        piv_files = sorted(
            exp_folder.rglob("std_*_piv.mat"),
        )

    else:
        raise ValueError("mode must be either 'original' or 'noisy'")

    images = [np.load(f) for f in img_files]
    displacements = [np.load(f) for f in disp_files]

    wofv_displacements = None
    piv_displacements = None

    if load_wofv:
        if len(wofv_files) != len(images):
            raise ValueError(
                "Number of wOFV files does not match the number of images: "
                f"{len(wofv_files)} WO-FV files vs {len(images)} images."
            )

        wofv_displacements = [load_clean_wofv_displacement(f) for f in wofv_files]

    if load_piv:
        if len(piv_files) != len(images):
            raise ValueError(
                "Number of PIV files does not match the number of images: "
                f"{len(piv_files)} PIV files vs {len(images)} images."
            )

        piv_displacements = [
            load_clean_piv_displacement(f, images[i]) for i, f in enumerate(piv_files)
        ]

    return images, displacements, wofv_displacements, piv_displacements


def load_clean_wofv_displacement(file):
    mat_data = loadmat(file)
    u_wofv = np.nan_to_num(mat_data["u_original"], nan=0.0)
    v_wofv = np.nan_to_num(mat_data["v_original"], nan=0.0)
    u_wofv = np.pad(u_wofv, ((0, 1), (0, 1)), mode="constant")
    v_wofv = np.pad(v_wofv, ((0, 1), (0, 1)), mode="constant")
    h_wofv = np.zeros((2, 1, *u_wofv.shape), dtype=u_wofv.dtype)

    h_wofv[1, 0] = u_wofv
    h_wofv[0, 0] = v_wofv

    return h_wofv


def load_clean_piv_displacement(file, img):
    mat_data = loadmat(file)
    u_piv = mat_data["u_original"]
    v_piv = mat_data["v_original"]
    x = mat_data["x"]
    y = mat_data["y"]

    x_flat = np.nan_to_num(x.flatten(), nan=0.0)
    y_flat = np.nan_to_num(y.flatten(), nan=0.0)
    u_flat = np.nan_to_num(u_piv.flatten(), nan=0.0)
    v_flat = np.nan_to_num(v_piv.flatten(), nan=0.0)

    img_h, img_w = img[0].shape

    xi = np.linspace(x.min(), x.max(), img_w)
    yi = np.linspace(y.min(), y.max(), img_h)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    u_dense = griddata((x_flat, y_flat), u_flat, (xi_grid, yi_grid), method="cubic")
    v_dense = griddata((x_flat, y_flat), v_flat, (xi_grid, yi_grid))

    h_piv = np.zeros((2,) + (img.shape[0],) + img.shape[1:])
    h_piv[1, 0] = u_dense
    h_piv[0, 0] = v_dense

    return h_piv


def compute_lame(E, nu):
    """
    Computes the Lamé parameters (μ and λ) from Young’s modulus and Poisson’s ratio.

    Args:
        E (float):
            Young’s modulus of the material.
        nu (float):
            Poisson’s ratio of the material.

    Returns:
        tuple[float, float]:
            - `mu_e` (float): Shear modulus (μ).
            - `lambda_e` (float): First Lamé parameter (λ).
    """
    lambda_e = E * nu / ((1.0 + nu) * (1.0 - (nu**2)))
    mu_e = E / (2.0 * (1.0 + nu))
    return mu_e, lambda_e


def rmse(u: np.ndarray, h: np.ndarray) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between two vector fields.

    Args:
        u (np.ndarray):
            Ground-truth field, shape `(d, H, W, ...)`.
        h (np.ndarray):
            Estimated or predicted field, same shape as `u`.

    Returns:
        float:
            The root mean square error value between `u` and `h`.
    """
    diff = u - h
    mse = np.mean(diff**2)
    return np.sqrt(mse)


def results_to_df(results: dict | list[dict]) -> pd.DataFrame:
    """
    Converts optical flow evaluation results into a formatted pandas DataFrame.

    Args:
        results (dict or list[dict]):
            Results containing RMSE values per optical flow method. Each entry should include keys such as:
              - `'mean_rmse_disp'`
              - `'mean_rmse_strain'`
              - `'mean_rmse_def'`
              - `'mean_rmse_stress'`
              - `'mean_rmse_traction'`

    Returns:
        pd.DataFrame:
            A formatted table with RMSE metrics for:
            displacement, strain, deformation, stress, and traction force,
            indexed by method name in the order:
            `["Proposed", "HS", "Farneback", "TV-L1", "ILK"]`.
    """
    name_map = {
        "fista": "Proposed",
        "hs": "HS",
        "farneback": "Farneback",
        "tv_l1": "TV-L1",
        "ilk": "ILK",
        "piv": "CC",
        "raft_algo": "RAFT",
        "wofv": "wOFV",
    }

    desired_order = [
        "Proposed",
        "HS",
        "Farneback",
        "TV-L1",
        "ILK",
        "CC",
        "RAFT",
        "wOFV",
    ]

    if isinstance(results, dict):
        results = [results]

    tables = []
    for res in results:
        if "mean_rmse_disp" in res:
            df = pd.DataFrame(
                {
                    "RMSE displacement": res["mean_rmse_disp"],
                    "RMSE strain": res["mean_rmse_strain"],
                    "RMSE deformation": res["mean_rmse_def"],
                    "RMSE stress": res["mean_rmse_stress"],
                    "RMSE traction force": res["mean_rmse_traction"],
                    "runtime": res["mean_runtime"],
                    "std RMSE displacement": res["std_rmse_disp"],
                    "std RMSE strain": res["std_rmse_strain"],
                    "std RMSE deformation": res["std_rmse_def"],
                    "std RMSE stress": res["std_rmse_stress"],
                    "std RMSE traction force": res["std_rmse_traction"],
                    "std runtime": res["std_runtime"],
                }
            )
            df.index = [name_map.get(k, k) for k in df.index]
            df = df[
                [
                    "RMSE displacement",
                    "RMSE strain",
                    "RMSE deformation",
                    "RMSE stress",
                    "RMSE traction force",
                    "runtime",
                    "std RMSE displacement",
                    "std RMSE strain",
                    "std RMSE deformation",
                    "std RMSE stress",
                    "std RMSE traction force",
                    "std runtime",
                ]
            ]
            tables.append(df)
        else:
            df = pd.DataFrame(
                {
                    "RMSE displacement": res[0]["rmse_flows"],
                    "RMSE strain": res[0]["rmse_strain"],
                    "RMSE deformation": res[0]["rmse_def"],
                    "RMSE stress": res[0]["rmse_stress"],
                    "RMSE traction force": res[0]["rmse_traction"],
                    "runtime": res[0]["runtime"],
                }
            )
            df.index = [name_map.get(k, k) for k in df.index]

            df = df[
                [
                    "RMSE displacement",
                    "RMSE strain",
                    "RMSE deformation",
                    "RMSE stress",
                    "RMSE traction force",
                    "runtime",
                ]
            ]
            tables.append(df)

    df_mean = pd.concat(tables).groupby(level=0).mean().round(4)

    df_mean = df_mean.reindex([m for m in desired_order if m in df_mean.index])

    return df_mean


def load_order_clean(image_path):
    """
    Loads a TIFF image, ensures it follows the TZYXC axis order and normalizes it

    Args:
        image_path (str): Path to the input TIFF image.

    Returns:
        tuple: A tuple containing the normalized image as a numpy array and the image's filename without extension.
    """
    # Store the name of the image for the plots
    image_name = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_name)[0]

    # Load the TIFF image
    with tifffile.TiffFile(image_path) as tif:
        image = tif.series[0].asarray()  # Read the image array

        # Check the axes order
        axes = tif.series[0].axes

        # The expected order is TZYXC
        expected_axes = ["T", "Z", "Y", "X", "C"]

        if "C" not in axes:
            image = image[..., np.newaxis]  # Add C as the last axis
            axes += "C"

        if "Z" not in axes:
            # Insert a singleton Z-dimension before Y
            y_index = axes.find("Y")
            image = np.expand_dims(image, axis=y_index)
            axes = axes[:y_index] + "Z" + axes[y_index:]

        if axes != expected_axes:
            # Swap axes based on the current axes order
            axis_order = {axis: idx for idx, axis in enumerate(axes)}

            # Create a mapping of the current axes to the target axes
            swap_order = [axis_order[axis] for axis in expected_axes]

            # Swap the axes to match TZYXC
            image = np.transpose(image, swap_order)

    # Normalize the image
    image = image / image.max()

    return image, image_name_without_ext


def generate_mask_on_micro_image(
    image: np.ndarray,
    active_contour: bool,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
):
    """
    Creates a mask on the cell in the image

    Args:
        image (np.ndarray): Image containing a single cell
        active_conotur (bool): Defined if the segmentation is done via active contour or via threshold
        center (tuple[float, float], optional): The center of the initial circle around the cell used for active_contours. Defaults to None.
        radius (float, optional): The radius of the initial circle around the cell used for active_contours. Defaults to None.
        alpha (float, optional): lpha parameter for the active contour function. Defaults to None.
        beta (float, optional): beta parameter for the active contour function. Defaults to None.
        gamma (float, optional): gamma parameter for the active contour function. Defaults to None.        c

    Returns:
        np.ndarray: The mask of the cell
    """

    if active_contour:
        radians = np.linspace(0, 2 * np.pi, 200)
        r = center[0] + radius * np.sin(radians)
        c = center[1] + radius * np.cos(radians)

        init = (np.array([r, c]).T)[:-1]

        snake = skimage.segmentation.active_contour(
            image, init, alpha=alpha, beta=beta, gamma=gamma
        )
        coords = list(zip(snake[:, 1], snake[:, 0]))
        polygon = Polygon(coords)

        x_coords = np.linspace(0, image.shape[1] - 1, num=image.shape[1] - 1)
        y_coords = np.linspace(0, image.shape[0] - 1, num=image.shape[0] - 1)

        mask_ = np.zeros_like(image)

        for x in x_coords:
            for y in y_coords:
                mask_[int(y), int(x)] = contains_xy(polygon, x, y)

    else:
        mask = image > 0.2
        mask__ = skimage.morphology.remove_small_objects(mask, 100)
        mask_ = skimage.morphology.remove_small_holes(mask__, 100)

    return mask_


def morozov(
    image: np.ndarray,
    num_iter_of: int,
    num_warp_of: int,
    num_pyramid_of: int,
    pyramid_downscale_of: float,
    homogeneous_patches: np.ndarray,
    alpha_init: float,
    step_size: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    c: float = 0.1,
) -> tuple[np.ndarray, float, float]:
    """
    Apply Morozov's discrepancy principle to estimate optimal regularization parameters for image registration.
    The regularization parameters are supposed to scale linearly to one another.
    Args:
        image (np.ndarray): The image. image[0] is the reference image (fixed), image[1] is the image to be registered (moving)
        num_iter_of (int): The number of iteration to perform in the optical flow algorithm
        num_warp_of (int): The number of warp to perform in the optical flow algorithm
        num_pyramid_of (int): The number of pyramids to create in the multi-scale approach of the optical flow algorihtm
        pyramid_downscale_of (float): Image scale between pyramid levels.
        homogeneous_patches (np.ndarray): Array of homogeneous patches for noise estimation.
        step_size (float): Step size for updating alpha.
        Default: 0.01.
        max_iter (int): Maximum number of iterations. Default: 100.
        tol (float): Tolerance for stopping criterion. Default: 1e-6.
        c (float): Constant to relate beta to alpha (beta = c * alpha). Default: 0.1.

    Returns:
        tuple[np.ndarray, float, float]:
        - u (np.ndarray): Estimated displacement field.
        - alpha (float): Optimal regularization parameter for the gradient term.
        - beta (float): Optimal regularization parameter for the Hessian term.
    """

    img1 = image[0]
    img2 = image[1]

    # Estimate the standard deviation of the noise
    noise_std = np.std(homogeneous_patches)
    discrepancy = noise_std**2 * img1.size

    print("discrepancy", discrepancy)

    # Initialize parameters
    alpha = alpha_init
    beta = c * alpha_init

    for _ in range(max_iter):
        # Solve for displacement field using FISTA
        u = fista(
            img1,
            img2,
            alpha,
            beta,
            num_iter=num_iter_of,
            num_warp=num_warp_of,
            num_pyramid=num_pyramid_of,
            pyramid_downscale=pyramid_downscale_of,
        )
        img2_warped = warp(img2, u)

        # Compute data term (discrepancy)
        data_term = np.linalg.norm(img1 - img2_warped) ** 2
        print("data term", data_term)
        print("abs diff", abs(data_term - discrepancy))
        # import matplotlib.pyplot as plt
        # plt.imshow(img1-img2_warped)
        # plt.colorbar()
        # plt.show()
        # Check Morozov's discrepancy principle
        if abs(data_term - discrepancy) < tol:
            break
        # Update alpha
        if data_term > discrepancy:
            alpha += step_size
        else:
            alpha -= step_size

        # Deduce beta
        beta = c * alpha
        print(alpha, beta)

    return u, alpha, beta


def remap(
    image: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    qt: bool = False,
):
    """
    Remaps the value of an image between 0 and 1.
    If vmin and vmax are None and qt is True, uses the 1% quantile of the image for 0 and the 99% in the image for 1.
    If vmin and vmax are None and qt is False, uses the minimum of the image for 0 and the maximum of the image for 1.
    If vmin and vmax are not None, maps vmin at 0 and vmax at 1.

    Args:
        image (np.ndarray): The image
        vmin (float, optional): The value in the image to be mapped at 0. Defaults to None.
        vmax (float, optional): The value in the image to be mapped at 1. Defaults to None.
        qt (bool): If we want to use the quantiles of the image rather than the extrema.

    Returns:
        np.ndarray: The new image with values between 0 and 1
    """

    if vmin is None or vmax is None:
        if qt:
            q1, q2 = np.quantile(image, q=[0.01, 0.99])
        else:
            q1, q2 = image.min(), image.max()
        if vmin is None:
            vmin = q1
        if vmax is None:
            vmax = q2

    return np.clip((image - vmin) / (vmax - vmin), 0, 1)
