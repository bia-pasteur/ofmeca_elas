"""Useful to search best optical flow parameters from the elastic simulation"""

import dataclasses
import itertools
import pickle
import time
from collections.abc import Callable
from pathlib import Path

import jsonargparse
import numpy as np

from mechanics.src.config import ElasticExperiment, GeneralParams, OpticalFlowParamsList
from mechanics.src.meca_of_pipeline import compute_of_strain_traction
from mechanics.src.optical_flow.algorithms import (
    farneback,
    fista_of,
    hs_of,
    ilk,
    tv_l1,
)
from mechanics.src.plot_functions import (
    save_table_rmse,
)
from mechanics.src.utils import (
    compute_lame,
    find_experiment_folder,
    load_clean_wofv_displacement,
    results_to_df,
    rmse,
)


def process_case(
    of_for_computation: list[Callable],
    params_for_computation: list[dict],
    global_flow: bool,
    T: float | None = None,
    E: float | None = None,
    nu: float | None = None,
    image_id: int | None = None,
) -> dict | list[dict]:
    """
    Process a specific simulation or experimental case by computing optical flow–based strain
    and traction fields, optionally saving visualization plots.

    This function can operate in several modes:
      - **Single image mode:** if `image_id`, `T`, `E`, and `nu` are specified, it loads a specific image
        and its ground-truth displacement field.
      - **Full parameter set mode:** if `T`, `E`, and `nu` are specified without `image_id`, it loads
        all images corresponding to these parameters.
      - **Batch experiment mode:** if `exp_ind` is specified (1, 2, or 3), it recursively processes
        all sub-cases of that experiment, identified by their `(T, E, nu)` folders.

    Args:
        elastic_params (ElasticExperiment): Parameters of the elastic simulation (T, E, nu, values for plotting...)
        results_dir (Path): Directory where results (plots and data) will be saved.
        of_for_computation (List[Callable]): List of optical flow algorithms (functions) to apply for displacement computation.
        params_for_computation (List[Dict]): List of parameter dictionaries corresponding to each optical flow method.
        global_flow (bool): Used in optical flow computation to compute the flow between every image and the next or between the first image and every other.
        exp_ind (int, optional): Experiment index (1, 2, or 3). If provided, the function iterates over
            all sub-cases `(T, E, nu)` contained in the corresponding experiment folder. Defaults to None.
        T (float, optional): Traction force magnitude for the wanted case. Defaults to None.
        E (float, optional): Young’s modulus of the cell for the wanted case. Default to None.
        nu (float, optional): Poisson's ratio of the cell for the wanted case. Defaults to None.
        image_id (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError:
            - If only a subset of `(T, E, nu)` is provided.
            - If `exp_ind` is not 1, 2, or 3.
            - If required parameters are missing for a specific image.
        FileNotFoundError:
            If the required image or displacement files do not exist in the expected folder.
    Returns:
        Dict | List[Dict]:
            - If processing a single case (image_id or (T, E, nu)), returns a dictionary containing
              the computed optical flow, strain, and traction results.
            - If exp_ind is provided, returns a list of results dictionaries for all sub-cases.
    """
    base_path = Path("data/elas")
    images = []
    displacements = []
    mu, lambda_ = compute_lame(500.0, 0.45)
    if image_id is not None:
        if E is None or T is None or nu is None:
            raise ValueError(
                f"Unspecified traction force T and/or Young's modulus E for simulation on image {image_id}"
            )
        else:
            exp_folder = find_experiment_folder(base_path, T, E, nu)
            img_path = exp_folder / f"{image_id}_img.npy"
            ugt_path = exp_folder / f"{image_id}_ugt.npy"

            if not img_path.exists() or not ugt_path.exists():
                raise FileNotFoundError(
                    f"Missing files for image_id={image_id} in {exp_folder}"
                )

            images = [np.load(img_path)]
            displacements = [np.load(ugt_path)]

            if of_for_computation == [None]:
                displacement = displacements[0]
                disp_gt = displacement
                mask = disp_gt[0, 0] != 0
                norm_disp = np.sqrt(np.mean(disp_gt**2))
                wofv_path = Path("data/fine_tuning")
                for sm in [20, 30, 40, 50, 60]:
                    h_wofv = load_clean_wofv_displacement(
                        wofv_path / f"smoothness_{sm}.mat"
                    )
                    h_wofv_mask = h_wofv * mask
                    rmse_flow_wofv = rmse(h_wofv_mask, disp_gt) / norm_disp

                    print("smoothness:", sm)
                    print("rmse dis:", rmse_flow_wofv)

    start_time = time.time()
    if image_id is not None:
        print(
            f"\nRunning analysis on T = {T}, E = {E}, nu = {nu} for image {image_id} ..."
        )

    results = compute_of_strain_traction(
        images=images,
        displacements=displacements,
        mu=mu,
        lambda_=lambda_,
        of_functions=of_for_computation,
        of_params=params_for_computation,
        global_flow=global_flow,
    )

    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")

    return results


def main(
    optical_flow_list: OpticalFlowParamsList,
    general: GeneralParams,
    elastic_exp: ElasticExperiment,
):
    """
    Main entry point for optical flow–based strain and traction analysis in order to get optimal
    parameters for optical flow algorithms.

    This function orchestrates the full processing pipeline:
      1. Initializes selected optical flow methods and their parameter sets.
      3. Runs process_case for experiment 1 for each combination of parameters to be tested.
      4. Saves the computed results as serialized `.pkl` files, CSV tables, and RMSE plots.

    Args:
        optical_flow_list (OpticalFlowParamsList): Configuration object containing parameter sets for each supported optical flow method.
                                                   Parameters can be lists where each parameter combination must be tested.
        general (GeneralParams): General configuration (mainly result storage)
        elastic_exp (ElasticExperiment): Parameters of the experiment of interest

    Raises:
        ValueError:
            - If an unknown optical flow method name is provided in `experiment.of_funcs`.
            - If experiment parameters are inconsistently defined (handled within `process_case`).
    """

    of_methods = {
        "farneback": (farneback, optical_flow_list.farneback_list),
        "hs": (hs_of, optical_flow_list.hs_list),
        "tvl1": (tv_l1, optical_flow_list.tvl1_list),
        "ilk": (ilk, optical_flow_list.ilk_list),
        "fista": (fista_of, optical_flow_list.fista_list),
        "wofv": (None, None),
    }

    for of_func_name in elastic_exp.of_funcs:
        print(of_func_name)
        if of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        of_func, of_params = of_methods[of_func_name]

        if of_func_name == "wofv":
            print(of_func)
            results_exp = process_case(
                of_for_computation=[of_func],
                params_for_computation={},
                T=100.0,
                E=500.0,
                nu=0.45,
                image_id="im_00_cell_003",
                global_flow=optical_flow_list.global_flow,
            )

        d = dataclasses.asdict(of_params)
        list_fields = {k: v for k, v in d.items() if isinstance(v, list)}
        fixed_fields = {k: v for k, v in d.items() if not isinstance(v, list)}
        cls = type(of_params)

        combos = (
            [
                cls(**{**fixed_fields, **dict(zip(list_fields, vals))})
                for vals in itertools.product(*list_fields.values())
            ]
            if list_fields
            else [of_params]
        )

        for params in combos:
            if all(
                x is None
                for x in (
                    elastic_exp.T,
                    elastic_exp.E,
                    elastic_exp.nu,
                    elastic_exp.exp_ind,
                    elastic_exp.image_id,
                )
            ):
                results_exp = process_case(
                    of_for_computation=[of_func],
                    params_for_computation=[params],
                    T=100.0,
                    E=500.0,
                    nu=0.45,
                    image_id="im_00_cell_003",
                    global_flow=optical_flow_list.global_flow,
                )

                with open(
                    Path(general.results_dir)
                    / "tables_dict"
                    / f"results_T_100.0_E_500.0_nu_0.45_im_3_{elastic_exp.of_funcs[0]}_{params}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(results_exp, f)
                df_exp = results_to_df(results_exp)
                df_exp.to_csv(
                    Path(general.results_dir)
                    / "tables_dict"
                    / f"mean_rmse_experiment_T_100.0_E_500.0_nu_0.45_im_3_{elastic_exp.of_funcs[0]}_{params}.csv",
                    index=True,
                )
                save_table_rmse(
                    df_exp,
                    Path(general.results_dir)
                    / "search_params"
                    / f"mean_rmse_{elastic_exp.of_funcs[0]}_{params}.png",
                )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
