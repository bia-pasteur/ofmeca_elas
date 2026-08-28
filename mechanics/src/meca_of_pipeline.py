"""Useful to get the results of the mechanical computations on images stored in a dictionary"""

# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
import copy
import time
from collections.abc import Callable

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


def compute_of_strain_traction(
    images: list[np.ndarray],
    displacements: list[np.ndarray],
    mu: float,
    lambda_: float,
    of_functions: list[Callable],
    of_params: list[dict],
    global_flow: bool,
    wofv_displacements: list[np.ndarray] | None = None,
) -> dict:
    """
    Compute optical-flow-based displacement, strain, deformation, stress, and traction fields for
    a given set of images and displacements.
    The results are stored in a dictionary that will also incluse the average values in the case
    of several images and displacements provided.

    Args:
        images (List[np.ndarray]): List of 2D grayscale images (float or uint) used as inputs to optical flow methods.
        displacements (List[np.ndarray]): List of ground-truth displacement fields for each image.
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter
        of_functions (List[Callable]): List of optical flow functions to evaluate.
        of_params (List[Dict]): List of parameter dictionaries corresponding to each function in of_functions.
        global_flow (bool): Used in optical flow computation to compute the flow between every image and the next or between the first image and every other.
        wofv_displacements (Dict[int, np.ndarray], optional):
            Dictionary mapping image indices to PIVlab displacement fields (shape: (2, H, W)).
            If provided, wOFV results are included in the analysis alongside other OF methods.

    Returns:
        Dict:  Dictionary containing, for each image index:

            - `"flows"`, `"strain"`, `"deformation"`, `"stress"`, `"traction"`:
            dictionaries with ground-truth (`"gt"`) and OF-based results per method.
            - `"rmse_flows"`, `"rmse_strain"`, `"rmse_def"`, `"rmse_stress"`, `"rmse_traction"`:
            RMSE values comparing OF estimates to ground truth.
            - `"runtime` : the runtime of the OF algorithms
            - `"mask"` : binary mask used for valid regions (cell)

            If multiple images are provided, the following mean metrics are also computed:
            - `"mean_rmse_disp"`, `"mean_rmse_strain"`, `"mean_rmse_def"`,
            `"mean_rmse_stress"`, `"mean_rmse_traction"`, `"std_rmse_disp"`, `"std_rmse_strain"`, `std_rmse_def"`,
            `"std_rmse_stress"`, `"std_rmse_traction"`
    """
    results = {}

    for nb, image in enumerate(images):
        results[nb] = {}
        displacement = displacements[nb]
        disp_gt = displacement
        mask = disp_gt[0, 0] != 0
        strain_gt = strain_mask(displacement, [1, 1], mask)
        def_gt = deformation(strain_gt)
        stress_gt = stress_mask(strain_gt, mu, lambda_)

        mask_eroded_gt = ndi.binary_erosion(mask, iterations=2)
        inner_boundary_gt = mask_eroded_gt & (~ndi.binary_erosion(mask_eroded_gt))
        normals_gt = compute_normals_from_mask_2d(mask_eroded_gt)
        normals_gt[:, ~inner_boundary_gt] = 0
        traction_gt = compute_traction_2d(stress_gt[:, :, 0], -normals_gt)

        norm_disp = np.sqrt(np.mean(disp_gt**2))
        norm_strain = np.sqrt(np.mean(strain_gt**2))
        norm_def = np.sqrt(np.mean(def_gt**2))
        norm_stress = np.sqrt(np.mean(stress_gt**2))
        norm_traction = np.sqrt(np.mean(traction_gt**2))

        results[nb]["flows"] = {"gt": disp_gt}
        results[nb]["strain"] = {"gt": strain_gt}
        results[nb]["deformation"] = {"gt": def_gt}
        results[nb]["stress"] = {"gt": stress_gt}
        results[nb]["traction"] = {"gt": traction_gt}
        results[nb]["mask"] = mask
        results[nb]["rmse_flows"] = {}
        results[nb]["rmse_strain"] = {}
        results[nb]["rmse_def"] = {}
        results[nb]["rmse_stress"] = {}
        results[nb]["rmse_traction"] = {}
        results[nb]["runtime"] = {}

        for i, method in enumerate(of_functions):
            method_name = method.__name__.replace("_of", "")
            start_time = time.time()

            current_params = copy.deepcopy(of_params[i])

            if "fista" in method.__name__:
                alpha = current_params.alpha
                beta = current_params.beta

                #     # # Signed distance: positive inside mask, negative outside
                #     # dist_inside = ndi.distance_transform_edt(
                #     #     ndi.binary_dilation(mask, iterations=2)
                #     # )
                #     # dist_outside = ndi.distance_transform_edt(
                #     #     ~ndi.binary_dilation(mask, iterations=2)
                #     # )
                #     # signed_dist = dist_inside - dist_outside

                #     # # Sigmoid weight: ~1 deep inside mask, ~0 deep outside, smooth at boundary
                #     # scale = 1  # transition width in pixels
                #     # weight = 1 / (1 + np.exp(-signed_dist / scale))

                #     # current_params.alpha = weight * alpha + (1 - weight) * (alpha / 10)
                #     # current_params.beta = weight * beta + (1 - weight) * (beta / 10)

                #     # boundary = mask & (~ndi.binary_erosion(mask))

                #     # current_params.alpha = np.where(boundary, alpha, alpha / 2)
                #     # current_params.beta = np.where(boundary, beta, beta / 2)

                current_params.alpha = np.where(
                    ndi.binary_dilation(mask, iterations=1), alpha, alpha * 2
                )
                current_params.beta = np.where(
                    ndi.binary_dilation(mask, iterations=1), beta, beta * 2
                )

            h = method(image, current_params, global_flow)

            # h = method(image, of_params[i], global_flow)
            time_method = time.time() - start_time
            h_mask = h * mask
            rmse_flow = rmse(h_mask, disp_gt) / norm_disp

            strain_of = strain_mask(h_mask, [1, 1], mask)
            rmse_strain = rmse(strain_of, strain_gt) / norm_strain

            def_of = deformation(strain_of)
            rmse_def = rmse(def_of, def_gt) / norm_def

            stress_of = stress_mask(strain_of, mu, lambda_)
            rmse_stress = rmse(stress_of, stress_gt) / norm_stress

            traction_of = compute_traction_2d(stress_of[:, :, 0], -normals_gt)
            rmse_traction = rmse(traction_of, traction_gt) / norm_traction

            results[nb]["flows"][method_name] = h_mask
            results[nb]["strain"][method_name] = strain_of
            results[nb]["deformation"][method_name] = def_of
            results[nb]["stress"][method_name] = stress_of
            results[nb]["traction"][method_name] = traction_of
            results[nb]["runtime"][method_name] = time_method

            results[nb]["rmse_flows"][method_name] = rmse_flow  # * 100
            results[nb]["rmse_strain"][method_name] = rmse_strain  # * 100
            results[nb]["rmse_def"][method_name] = rmse_def  # * 100
            results[nb]["rmse_stress"][method_name] = rmse_stress
            results[nb]["rmse_traction"][method_name] = rmse_traction

        if wofv_displacements is not None:
            h_wofv = wofv_displacements[nb]
            h_wofv_mask = h_wofv * mask
            rmse_flow_wofv = rmse(h_wofv_mask, disp_gt) / norm_disp

            strain_wofv = strain_mask(h_wofv_mask, [1, 1], mask)
            rmse_strain_wofv = rmse(strain_wofv, strain_gt) / norm_strain

            def_wofv = deformation(strain_wofv)
            rmse_def_wofv = rmse(def_wofv, def_gt) / norm_def

            stress_wofv = stress_mask(strain_wofv, mu, lambda_)
            rmse_stress_wofv = rmse(stress_wofv, stress_gt) / norm_stress

            traction_wofv = compute_traction_2d(stress_wofv[:, :, 0], -normals_gt)
            rmse_traction_wofv = rmse(traction_wofv, traction_gt) / norm_traction

            results[nb]["flows"]["wofv"] = h_wofv_mask
            results[nb]["strain"]["wofv"] = strain_wofv
            results[nb]["deformation"]["wofv"] = def_wofv
            results[nb]["stress"]["wofv"] = stress_wofv
            results[nb]["traction"]["wofv"] = traction_wofv
            results[nb]["runtime"]["wofv"] = 0.0

            results[nb]["rmse_flows"]["wofv"] = rmse_flow_wofv
            results[nb]["rmse_strain"]["wofv"] = rmse_strain_wofv
            results[nb]["rmse_def"]["wofv"] = rmse_def_wofv
            results[nb]["rmse_stress"]["wofv"] = rmse_stress_wofv
            results[nb]["rmse_traction"]["wofv"] = rmse_traction_wofv

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
            m = method.__name__.replace("_of", "")
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
                    disp_vals.append(res["rmse_flows"][m])
                    strain_vals.append(res["rmse_strain"][m])
                    def_vals.append(res["rmse_def"][m])
                    stress_vals.append(res["rmse_stress"][m])
                    trac_vals.append(res["rmse_traction"][m])
                    runtime_vals.append(res["runtime"][m])

            results["mean_rmse_disp"][m] = np.mean(disp_vals)
            results["mean_rmse_strain"][m] = np.mean(strain_vals)
            results["mean_rmse_def"][m] = np.mean(def_vals)
            results["mean_rmse_stress"][m] = np.mean(stress_vals)
            results["mean_rmse_traction"][m] = np.mean(trac_vals)
            results["mean_runtime"][m] = np.mean(runtime_vals)

            results["std_rmse_disp"][m] = np.std(disp_vals)
            results["std_rmse_strain"][m] = np.std(strain_vals)
            results["std_rmse_def"][m] = np.std(def_vals)
            results["std_rmse_stress"][m] = np.std(stress_vals)
            results["std_rmse_traction"][m] = np.std(trac_vals)
            results["std_runtime"][m] = np.std(runtime_vals)

        if wofv_displacements is not None:
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
                if res and "wofv" in res["rmse_flows"]:
                    disp_vals.append(res["rmse_flows"]["wofv"])
                    strain_vals.append(res["rmse_strain"]["wofv"])
                    def_vals.append(res["rmse_def"]["wofv"])
                    stress_vals.append(res["rmse_stress"]["wofv"])
                    trac_vals.append(res["rmse_traction"]["wofv"])
                    runtime_vals.append(res["runtime"]["wofv"])

            if disp_vals:
                results["mean_rmse_disp"]["wofv"] = np.mean(disp_vals)
                results["mean_rmse_strain"]["wofv"] = np.mean(strain_vals)
                results["mean_rmse_def"]["wofv"] = np.mean(def_vals)
                results["mean_rmse_stress"]["wofv"] = np.mean(stress_vals)
                results["mean_rmse_traction"]["wofv"] = np.mean(trac_vals)
                results["mean_runtime"]["wofv"] = np.mean(runtime_vals)

                results["std_rmse_disp"]["wofv"] = np.std(disp_vals)
                results["std_rmse_strain"]["wofv"] = np.std(strain_vals)
                results["std_rmse_def"]["wofv"] = np.std(def_vals)
                results["std_rmse_stress"]["wofv"] = np.std(stress_vals)
                results["std_rmse_traction"]["wofv"] = np.std(trac_vals)
                results["std_runtime"]["wofv"] = np.std(runtime_vals)

    return results


def compute_of_strain_traction_micro_img(
    image: np.ndarray,
    mask: np.ndarray,
    mu: float,
    lambda_: float,
    of_functions: list[Callable],
    of_params: list[dict],
    global_flow: bool,
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

    return results
