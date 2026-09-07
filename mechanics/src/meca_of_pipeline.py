"""Useful to get the results of the mechanical computations on images stored in a dictionary"""

# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace

import copy
import time

import numpy as np
import scipy.ndimage as ndi

from mechanics.src.MCM.quantities_computation import (
    compute_normals_from_mask_2d,
    compute_traction_2d,
    deformation,
    strain_mask,
    stress_mask,
)
from mechanics.src.utils import rmse


def _initialize_results_image_gt(
    u_gt: np.ndarray, mu: float, lambda_: float
) -> tuple[dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray]:
    """
    Initialize the results dictionary with Ground-Truth values

    Args:
        u_gt (np.ndarray): Ground-Truth displacmeent field
        mu (float): First Lamé parameter
        lambda_ (float): Second Lamé parameter

    Returns:
        dict, tuple[float, float, float, float, float, np.ndarray], np.ndarray:

    """
    results_im = {}
    mask = u_gt[0, 0] != 0
    strain_gt = strain_mask(u_gt, [1, 1], mask)
    def_gt = deformation(strain_gt)
    stress_gt = stress_mask(strain_gt, mu, lambda_)

    mask_eroded_gt = ndi.binary_erosion(mask, iterations=3)
    inner_boundary_gt = mask_eroded_gt & (~ndi.binary_erosion(mask_eroded_gt))
    normals_gt = compute_normals_from_mask_2d(mask_eroded_gt)
    normals_gt[:, ~inner_boundary_gt] = 0
    traction_gt = compute_traction_2d(stress_gt[:, :, 0], -normals_gt)

    norm_disp = np.sqrt(np.mean(u_gt**2))
    norm_strain = np.sqrt(np.mean(strain_gt**2))
    norm_def = np.sqrt(np.mean(def_gt**2))
    norm_stress = np.sqrt(np.mean(stress_gt**2))
    norm_traction = np.sqrt(np.mean(traction_gt**2))

    results_im["flows"] = {"gt": u_gt}
    results_im["strain"] = {"gt": strain_gt}
    results_im["deformation"] = {"gt": def_gt}
    results_im["stress"] = {"gt": stress_gt}
    results_im["traction"] = {"gt": traction_gt}
    results_im["mask"] = mask
    results_im["rmse_flows"] = {}
    results_im["rmse_strain"] = {}
    results_im["rmse_def"] = {}
    results_im["rmse_stress"] = {}
    results_im["rmse_traction"] = {}
    results_im["runtime"] = {}

    return (
        results_im,
        (
            norm_disp,
            norm_strain,
            norm_def,
            norm_stress,
            norm_traction,
            normals_gt,
        ),
        mask,
    )


def _reg_params(
    method: callable,
    params: dict,
    mask: np.ndarray | None = None,
    reg_multiplier: float | None = None,
) -> dict:
    """
    Adapts the Optical Flow method parameters

    Args:
        method (callable): Optical flow function
        params (dict): Original paremeters
        mask (np.ndarray): Mask of the cell in the image of interest.
                           Used for FISTA regularization parameter map. Defaults to None.
        reg_multiplier (float | None, optional): Value to multiply the parameters.
                                                 Used in the noise experiment. Defaults to None.

    Returns:
        dict: Updated parameters.
    """
    current_params = copy.deepcopy(params)

    if "fista" in method.__name__:
        if reg_multiplier is not None:
            current_params.alpha *= reg_multiplier
            current_params.beta *= reg_multiplier

        alpha = current_params.alpha
        beta = current_params.beta

        current_params.alpha = np.where(
            ndi.binary_dilation(mask, iterations=1), alpha, alpha * 2.5
        )
        current_params.beta = np.where(
            ndi.binary_dilation(mask, iterations=1), beta, beta * 2.5
        )

    else:
        if reg_multiplier is not None:
            if "hs" in method.__name__:
                current_params.alpha *= reg_multiplier
            elif "farneback" in method.__name__:
                current_params.winSize = int(current_params.winSize * reg_multiplier)
            elif "ilk" in method.__name__:
                current_params.radius *= reg_multiplier
            elif "tvl1" in method.__name__:
                current_params.attachment *= reg_multiplier

    return current_params


def _update_results_method(
    results_im: dict,
    method_name: str,
    h: np.ndarray,
    mask: np.ndarray,
    time_method: float,
    mu: float,
    lambda_: float,
    norms_normals_gt: tuple,
) -> dict:
    """
    Update the results dictionary for one image with values related to one optical flow method.

    Args:
        results_im (dict): Results dictionary.
        method_name (str): Name fo the optical flow method.
        h (np.ndarray): Displacement field obtained using the optical flow method of interest.
        mask (np.ndarray): Mask of the cell in the image of interest.
        time_method (float): Run time of the algorithm.
        mu (float): First Lamé parameter.
        lambda_ (float): Second Lamé parameter.
        norms_normals_gt (tuple): Norms of the Ground-Truth mechanical values for RMSE computation
                                  and normals of the cell for traction computation.

    Returns:
        dict: Updated results dictionary.
    """
    norm_disp, norm_strain, norm_def, norm_stress, norm_traction, normals = (
        norms_normals_gt
    )

    h_mask = h * mask

    rmse_flow = rmse(h_mask, results_im["flows"]["gt"]) / norm_disp

    strain_of = strain_mask(h_mask, [1, 1], mask)
    rmse_strain = rmse(strain_of, results_im["strain"]["gt"]) / norm_strain

    def_of = deformation(strain_of)
    rmse_def = rmse(def_of, results_im["deformation"]["gt"]) / norm_def

    stress_of = stress_mask(strain_of, mu, lambda_)
    rmse_stress = rmse(stress_of, results_im["stress"]["gt"]) / norm_stress

    traction_of = compute_traction_2d(stress_of[:, :, 0], -normals)
    rmse_traction = rmse(traction_of, results_im["traction"]["gt"]) / norm_traction

    results_im["flows"][method_name] = h_mask
    results_im["strain"][method_name] = strain_of
    results_im["deformation"][method_name] = def_of
    results_im["stress"][method_name] = stress_of
    results_im["traction"][method_name] = traction_of
    results_im["runtime"][method_name] = time_method

    results_im["rmse_flows"][method_name] = rmse_flow
    results_im["rmse_strain"][method_name] = rmse_strain
    results_im["rmse_def"][method_name] = rmse_def
    results_im["rmse_stress"][method_name] = rmse_stress
    results_im["rmse_traction"][method_name] = rmse_traction

    return results_im


def _compute_mean_stds_values(results: dict, method_name: str) -> dict:
    """
    Compute the average values and their stds for all mechanical quantities
    for one OF method and add them to the results dictionary.

    Args:
        results (dict): Results dictionary.
        method_name (str): Name of the optical flow method.

    Returns:
        dict: Updated results dictionary.
    """
    disp_vals, strain_vals, def_vals, stress_vals, trac_vals, runtime_vals = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for nb, res in results.items():
        if not isinstance(nb, int):
            continue
        if res:
            disp_vals.append(res["rmse_flows"][method_name])
            strain_vals.append(res["rmse_strain"][method_name])
            def_vals.append(res["rmse_def"][method_name])
            stress_vals.append(res["rmse_stress"][method_name])
            trac_vals.append(res["rmse_traction"][method_name])
            runtime_vals.append(res["runtime"][method_name])

    results["mean_rmse_disp"][method_name] = np.mean(disp_vals)
    results["mean_rmse_strain"][method_name] = np.mean(strain_vals)
    results["mean_rmse_def"][method_name] = np.mean(def_vals)
    results["mean_rmse_stress"][method_name] = np.mean(stress_vals)
    results["mean_rmse_traction"][method_name] = np.mean(trac_vals)
    results["mean_runtime"][method_name] = np.mean(runtime_vals)

    results["std_rmse_disp"][method_name] = np.std(disp_vals)
    results["std_rmse_strain"][method_name] = np.std(strain_vals)
    results["std_rmse_def"][method_name] = np.std(def_vals)
    results["std_rmse_stress"][method_name] = np.std(stress_vals)
    results["std_rmse_traction"][method_name] = np.std(trac_vals)
    results["std_runtime"][method_name] = np.std(runtime_vals)

    return results


def compute_of_strain_traction(
    images: list[np.ndarray],
    displacements: list[np.ndarray],
    mu: float,
    lambda_: float,
    of_functions: list[callable],
    of_params: list[dict],
    wofv_displacements: list[np.ndarray] | None = None,
    piv_displacements: list[np.ndarray] | None = None,
    reg_multipliers: list[float] | None = None,
) -> dict:
    """
    Get Ground-Truth and compute optical-flow-based displacement, strain, deformation, stress, and traction fields on images.
    Compute average values and their stds for every measured quantity if more than one image is provided.

    Args:
        images (list[np.ndarray]): Images to analyze.
        displacements (list[np.ndarray]): Ground-Truth displacements on the images.
        mu (float): First Lamé parameter.
        lambda_ (float): Second Lamé parameter.
        of_functions (list[callable]): Optical flow functions to use during the analysis.
        of_params (list[dict]): Parameters for each optical flow function.
        wofv_displacements (list[np.ndarray] | None, optional): wOFV displacement computed using the PIVlab sofware. Defaults to None.
        piv_displacements (list[np.ndarray] | None, optional): PIV displacement computed using the PIVlab sofware. Defaults to None.
        reg_multipliers (list[float] | None, optional): Values to multiply the optical flow parameters.
                                                        Used in the noise experiment. Defaults to None.

    Returns:
        dict: Dictionary containing the Ground-Truth and optical flow based displacement, strain, deformation, stress, and traction fields on images
        as well as the RMSE value for every quantity.
    """
    results = {}

    for nb, image in enumerate(images):
        disp_gt = displacements[nb]
        results_im, norms_normals_gt, mask = _initialize_results_image_gt(
            disp_gt, mu, lambda_
        )

        for i, method in enumerate(of_functions):
            method_name = method.__name__.replace("_of", "")

            params_for_image = _reg_params(method, of_params[i], mask, reg_multipliers)

            start_time = time.time()
            h = method(image, params_for_image, global_flow=False)
            time_method = time.time() - start_time

            results_im = _update_results_method(
                results_im=results_im,
                method_name=method_name,
                h=h,
                mask=mask,
                time_method=time_method,
                mu=mu,
                lambda_=lambda_,
                norms_normals_gt=norms_normals_gt,
            )

        if wofv_displacements is not None:
            results_im = _update_results_method(
                results_im=results_im,
                method_name="wofv",
                h=wofv_displacements[nb],
                mask=mask,
                time_method=0.0,
                mu=mu,
                lambda_=lambda_,
                norms_normals_gt=norms_normals_gt,
            )

        if piv_displacements is not None:
            results_im = _update_results_method(
                results_im=results_im,
                method_name="piv",
                h=piv_displacements[nb],
                mask=mask,
                time_method=0.0,
                mu=mu,
                lambda_=lambda_,
                norms_normals_gt=norms_normals_gt,
            )

        results[nb] = results_im

    if len(images) > 1:
        results["mean_rmse_disp"] = {}
        results["mean_rmse_strain"] = {}
        results["mean_rmse_def"] = {}
        results["mean_rmse_stress"] = {}
        results["mean_rmse_traction"] = {}
        results["mean_runtime"] = {}
        results["std_rmse_disp"] = {}
        results["std_rmse_strain"] = {}
        results["std_rmse_def"] = {}
        results["std_rmse_stress"] = {}
        results["std_rmse_traction"] = {}
        results["std_runtime"] = {}

        for method in of_functions:
            method_name = method.__name__.replace("_of", "")

            _compute_mean_stds_values(results, method_name)

            if wofv_displacements is not None:
                _compute_mean_stds_values(results, "wofv")

            if piv_displacements is not None:
                _compute_mean_stds_values(results, "piv")

    return results


def compute_of_strain_traction_micro_img(
    image: np.ndarray,
    mask: np.ndarray,
    mu: float,
    lambda_: float,
    of_functions: list[callable],
    of_params: list[dict],
    global_flow: bool,
    wofv_displacement: np.ndarray | None = None,
) -> dict:
    """
    Compute optical-flow-based displacement, strain, deformation, stress, and traction fields on a microsocpy image.

    This function evaluates several optical flow (OF) methods on an image.
    It then computes the corresponding strain, deformation gradient, stress, and traction fields

    Args:
        image (np.ndarray): 2D grayscale image (float or uint) used as input to optical flow methods. The image must contain one single cell.
        mask (np.ndarray): Binary mask of the cell in the image
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter
        of_functions (List[Callable]): List of optical flow functions to evaluate.
        of_params (List[Dict]): List of parameter dictionaries corresponding to each function in `of_functions`.
        global_flow (bool): Used in optical flow computation to compute the flow between every image and the next or between the first image and every other.

    Returns
        dict:
            Dictionary containing, for each image index:

            - `"flows"`, `"strain"`, `"deformation"`, `"stress"`, `"traction"`:
            dictionaries with OF-based results per method.
            - `"mask"` : binary mask used for valid regions.
    """

    results = {}

    eroded_mask = ndi.binary_erosion(mask)
    inner_boundary_gt = eroded_mask & (~ndi.binary_erosion(eroded_mask))
    normals = compute_normals_from_mask_2d(eroded_mask)
    normals[:, ~inner_boundary_gt] = 0

    results["flows"] = {}
    results["strain"] = {}
    results["deformation"] = {}
    results["stress"] = {}
    results["traction"] = {}
    results["runtime"] = {}

    for i, method in enumerate(of_functions):
        method_name = method.__name__.replace("_of", "")
        start_time = time.time()
        h = method(image, of_params[i], global_flow)
        time_method = time.time() - start_time
        h_mask = h * mask

        strain_of = strain_mask(h, [1, 1], mask)

        def_of = deformation(strain_of)

        stress_of = stress_mask(strain_of, mu, lambda_)

        traction_of = compute_traction_2d(stress_of[:, :, 0], -normals)

        results["flows"][method_name] = h_mask
        results["strain"][method_name] = strain_of
        results["deformation"][method_name] = def_of
        results["stress"][method_name] = stress_of
        results["traction"][method_name] = traction_of
        results["runtime"][method_name] = time_method

    if wofv_displacement is not None:
        h_wofv = wofv_displacement[0]
        h_wofv_mask = h_wofv * mask

        strain_wofv = strain_mask(h_wofv_mask, [1, 1], mask)

        def_wofv = deformation(strain_wofv)

        stress_wofv = stress_mask(strain_wofv, mu, lambda_)

        traction_wofv = compute_traction_2d(stress_wofv[:, :, 0], -normals)

        results["flows"]["wofv"] = h_wofv_mask
        results["strain"]["wofv"] = strain_wofv
        results["deformation"]["wofv"] = def_wofv
        results["stress"]["wofv"] = stress_wofv
        results["traction"]["wofv"] = traction_wofv

    return results
