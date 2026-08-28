"""Useful to run the regularization parameter sensitivity analysis"""

import copy
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import jsonargparse
import numpy as np

from mechanics.src.config import GeneralParams, OpticalFlowParams, RegExperiment
from mechanics.src.meca_of_pipeline import compute_of_strain_traction
from mechanics.src.optical_flow.algorithms import farneback, fista_of, hs_of, ilk, tv_l1
from mechanics.src.plot_functions import plot_reg
from mechanics.src.utils import compute_lame


def process_test_reg(
    exp_folder: Path,
    image_for_test_reg: int,
    mu: float,
    lambda_: float,
    of_for_computation: (list[Callable]),
    params_of: list[dict],
    global_flow: bool,
    factors_for_reg: list[float],
) -> dict:
    """
    Runs a regularization parameter sensitivity analysis for optical flow-based strain and traction estimation.

    This function tests how scaling the regularization parameters (e.g., alpha, beta, window size, etc.)
    affects the accuracy of mechanical computations. It loads a reference image and corresponding ground truth
    displacement field, applies different scaling factors to the
    optical flow parameters, and computes RMSE metrics for each method and scaling factor.

    Args:
        exp_folder (Path): Path to the experiment folder containing the test data files.
        image_for_test_reg (int): Index or identifier of the image/displacement pair to use for testing.
            Expects files named `<image_for_test_reg>_img.npy` and `<image_for_test_reg>_ugt.npy` in `exp_folder`.
        mu (float): Lamé parameter
        lambda_ (float): Lamé parameter
        of_for_computation (List[Callable]): List of optical flow functions to evaluate.
        params_of (List[Dict]): List of parameter objects or dictionaries corresponding to each optical flow method in `of_for_computation`.
        factors_for_reg (List[float]): List of multiplicative scaling factors applied to regularization-related parameters for sensitivity testing.

    Raises:
        FileNotFoundError:
            If the required image or ground truth displacement files are missing from `exp_folder`.

    Returns:
        dict:
            Nested dictionary containing RMSE values for each optical flow method and scaling factor.
            Structure:
            ```
            {
                'flow': {method_name: [rmse_values_per_factor], ...},
                'deformation': {method_name: [rmse_values_per_factor], ...},
                'traction': {method_name: [rmse_values_per_factor], ...}
            }
            ```
            - `flow`: RMSE of the optical flow fields.
            - `deformation`: RMSE of the reconstructed deformation fields.
            - `traction`: RMSE of the estimated traction fields.
    """
    print("\nRunning regularisation parameter mishandling analysis on a new image...")
    start_time = time.time()
    img_path = exp_folder / f"{image_for_test_reg}_img.npy"
    ugt_path = exp_folder / f"{image_for_test_reg}_ugt.npy"

    if not img_path.exists() or not ugt_path.exists():
        raise FileNotFoundError(
            f"Missing files for image_id={image_for_test_reg} in {exp_folder}"
        )

    images = [np.load(img_path)]
    displacements = [np.load(ugt_path)]

    method_names = []
    for _, method in enumerate(of_for_computation):
        method_names.append(method.__name__.replace("_of", ""))

    rmse_dict = {}
    rmse_dict["flow"] = {}
    rmse_dict["deformation"] = {}
    rmse_dict["traction"] = {}

    for m in method_names:
        rmse_dict["flow"][m] = []
        rmse_dict["deformation"][m] = []
        rmse_dict["traction"][m] = []

    for factor in factors_for_reg:
        params_for_computation = copy.deepcopy(params_of)
        params_for_computation[0].alpha *= factor
        params_for_computation[0].beta *= factor
        params_for_computation[1].alpha *= factor
        params_for_computation[2].winSize = int(
            params_for_computation[2].winSize * factor
        )
        params_for_computation[3].radius *= factor
        params_for_computation[4].tightness *= factor

        results_exp = compute_of_strain_traction(
            images=images,
            displacements=displacements,
            mu=mu,
            lambda_=lambda_,
            of_functions=of_for_computation,
            of_params=params_for_computation,
            global_flow=global_flow,
        )

        for m in method_names:
            rmse_dict["flow"][m].append(results_exp[0]["rmse_flows"][m])
            rmse_dict["deformation"][m].append(results_exp[0]["rmse_def"][m])
            rmse_dict["traction"][m].append(results_exp[0]["rmse_traction"][m])

    elapsed = time.time() - start_time
    print(
        f"Regularisation parameter mishandling analysis completed in {elapsed:.2f} seconds"
    )
    return rmse_dict


def main(
    optical_flow: OpticalFlowParams, general: GeneralParams, reg_exp: RegExperiment
):
    """
    Runs a regularization parameter sensitivity analysis for optical flow-based strain and traction estimation.

    Args:
        optical_flow (OpticalFlowParams): Configuration object containing parameter sets for each supported optical flow method
        general (GeneralParams): General configuration (mainly result storage)
        reg_exp (RegExperiment): Parameters of the regularization experiment (optical flow functions, T, E, nu...)

    Raises:
        ValueError:
            If the optical flow functions provided in the experiment class are not in the available optical flow functions.
    """

    of_methods = {
        "farneback": (farneback, optical_flow.farneback),
        "hs": (hs_of, optical_flow.hs),
        "tvl1": (tv_l1, optical_flow.tvl1),
        "ilk": (ilk, optical_flow.ilk),
        "fista": (fista_of, optical_flow.fista),
    }

    of_for_computation, params_for_computation = [], []

    for of_func_name in reg_exp.of_funcs:
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]
        of_for_computation.append(of_func)
        params_for_computation.append(of_params)

    rmse_acc_reg = defaultdict(lambda: defaultdict(list))

    exp_folder = Path(
        f"data/elas/experiment_1/T_{reg_exp.T}_E_{reg_exp.E}_nu_{reg_exp.nu}"
    )
    img_indices = sorted(
        (f.stem.replace("_img", "")) for f in exp_folder.glob("*_img.npy")
    )

    mu_reg, lambda_reg = compute_lame(reg_exp.E, reg_exp.nu)

    for imgind in img_indices:
        rmse_dict_reg = process_test_reg(
            exp_folder=exp_folder,
            image_for_test_reg=imgind,
            mu=mu_reg,
            lambda_=lambda_reg,
            of_for_computation=of_for_computation,
            params_of=params_for_computation,
            factors_for_reg=reg_exp.factors,
            global_flow=optical_flow.global_flow,
        )
        for key in ["deformation", "traction"]:
            for method in rmse_dict_reg[key]:
                rmse_acc_reg[key][method].append(rmse_dict_reg[key][method])

    rmse_mean_reg = {
        key: {
            method: np.mean(rmse_acc_reg[key][method], axis=0)
            for method in rmse_acc_reg[key]
        }
        for key in rmse_acc_reg
    }

    plot_reg(rmse_mean_reg, reg_exp.factors, Path(general.results_dir))


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
