"""Useful to compute mechanical quantities from the elastic simulation"""

import pickle
import time
from collections.abc import Callable
from pathlib import Path

import jsonargparse
import numpy as np

from mechanics.src.config import ElasticExperiment, GeneralParams, OpticalFlowParams
from mechanics.src.meca_of_pipeline import compute_of_strain_traction
from mechanics.src.optical_flow.algorithms import (
    farneback,
    fista_of,
    hs_of,
    ilk,
    raft_algo,
    tv_l1,
)
from mechanics.src.plot_functions import (
    save_of_strain_traction,
    save_scatter_comparison,
    save_table_rmse_with_std,
)
from mechanics.src.utils import (
    compute_lame,
    extract_E_from_folder,
    extract_nu_from_folder,
    extract_T_from_folder,
    find_experiment_folder,
    load_clean_piv_displacement,
    load_clean_wofv_displacement,
    load_images_and_displacements,
    results_to_df,
)


def process_case(
    elastic_params: ElasticExperiment,
    results_dir: Path,
    of_for_computation: list[Callable],
    params_for_computation: list[dict],
    include_wofv: bool = False,
    include_piv: bool = False,
    exp_ind: int | None = None,
    T: float | None = None,
    E: float | None = None,
    nu: float | None = None,
    image_id: int | None = None,
    cases_done: dict | None = None,
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
        wofv_base_path (Path, optional): Base path to PIVlab results. If provided, PIV displacements are loaded and included.
        exp_ind (int, optional): Experiment index (1, 2, or 3). If provided, the function iterates over
            all sub-cases `(T, E, nu)` contained in the corresponding experiment folder. Defaults to None.
        T (float, optional): Traction force magnitude for the wanted case. Defaults to None.
        E (float, optional): Young’s modulus of the cell for the wanted case. Default to None.
        nu (float, optional): Poisson’s ratio of the cell for the wanted case. Defaults to None.
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
    mu = 0
    lambda_ = 0
    if image_id is not None:
        if E is None or T is None or nu is None:
            raise ValueError(
                f"Unspecified traction force T and/or Young's modulus E for simulation on image {image_id}"
            )
        else:
            exp_folder = find_experiment_folder(base_path, T, E, nu)
            img_path = exp_folder / f"{image_id}_img.npy"
            ugt_path = exp_folder / f"{image_id}_ugt.npy"
            wofv_path = exp_folder / f"{image_id}.mat"
            piv_path = exp_folder / f"{image_id}.mat"

            if not img_path.exists() or not ugt_path.exists():
                raise FileNotFoundError(
                    f"Missing files for image_id={image_id} in {exp_folder}"
                )

            images = [np.load(img_path)]
            displacements = [np.load(ugt_path)]

            if include_wofv:
                wofv_displacements = [load_clean_wofv_displacement(wofv_path)]

            if include_piv:
                piv_displacements = [load_clean_piv_displacement(piv_path, images[0])]
            mu, lambda_ = compute_lame(E, nu)

    elif (E is not None and (T is None or nu is None)) or (
        E is None and (T is not None or nu is not None)
    ):
        raise ValueError(
            "All of E, T, and nu must be specified together (either all given or all None)."
        )

    elif E is not None and T is not None and nu is not None:
        exp_folder = find_experiment_folder(base_path, T, E, nu)
        images, displacements, wofv_displacements, piv_displacements = (
            load_images_and_displacements(
                exp_folder,
                mode="original",
                load_wofv=include_wofv,
                load_piv=include_piv,
            )
        )

        mu, lambda_ = compute_lame(E, nu)

    elif exp_ind is not None:
        if exp_ind == 1:
            exp = "experiment_1"
        elif exp_ind == 2:
            exp = "experiment_2"
        elif exp_ind == 3:
            exp = "experiment_3"
        else:
            raise ValueError("exp_ind can only be 1, 2 or 3")
        exp_folder = base_path / exp
        results_ = []
        for case_T_E_nu in sorted(exp_folder.iterdir()):
            if case_T_E_nu.name != ".DS_Store":
                print(
                    "\nAlready done ?",
                    "Yes" if case_T_E_nu.name in cases_done else "No",
                )
                if case_T_E_nu.name in cases_done:
                    results_.append(cases_done[case_T_E_nu.name])
                else:
                    E_case = extract_E_from_folder(case_T_E_nu.name)
                    T_case = extract_T_from_folder(case_T_E_nu.name)
                    nu_case = extract_nu_from_folder(case_T_E_nu.name)
                    results_case = process_case(
                        elastic_params,
                        results_dir,
                        of_for_computation,
                        params_for_computation,
                        include_wofv=include_wofv,
                        include_piv=include_piv,
                        T=T_case,
                        E=E_case,
                        nu=nu_case,
                    )
                    results_.append(results_case)
                    cases_done[case_T_E_nu.name] = results_case

        return results_, cases_done

    else:
        raise ValueError("Can't run with nothing specified")

    start_time = time.time()
    if image_id is not None:
        print(
            f"\nRunning analysis on T = {T}, E = {E}, nu = {nu} for image {image_id} ..."
        )
    elif T is not None:
        print(f"\nRunning analysis on T = {T}, E = {E}, nu = {nu} ...")

    results = compute_of_strain_traction(
        images=images,
        displacements=displacements,
        mu=mu,
        lambda_=lambda_,
        of_functions=of_for_computation,
        of_params=params_for_computation,
        wofv_displacements=wofv_displacements,
        piv_displacements=piv_displacements,
    )

    if image_id is not None:
        save_of_strain_traction(
            images=images,
            results=results,
            save_path=results_dir
            / "plots"
            / f"strain_traction_plot_E_{E}_T_{T}_nu_{nu}_im_{image_id}.png",
            implot=0,
            vmaxstrain=elastic_params.vmaxstrain,
            scale_flow=elastic_params.scale_flow,
            step_flow=elastic_params.step_flow,
            scale_traction=elastic_params.scale_traction,
            step_traction=elastic_params.step_traction,
            threshold_inf=elastic_params.threshold_inf,
            threshold_sup=elastic_params.threshold_sup,
            show=False,
        )

    else:
        if (
            T == elastic_params.T_for_plot
            and E == elastic_params.E_for_plot
            and nu == elastic_params.nu_for_plot
        ):
            save_of_strain_traction(
                images=images,
                results=results,
                save_path=results_dir
                / "plots"
                / f"strain_traction_plot_E_{E}_T_{T}_nu_{nu}_im_{elastic_params.implot}.png",
                implot=elastic_params.implot,
                vmaxstrain=elastic_params.vmaxstrain,
                scale_flow=elastic_params.scale_flow,
                step_flow=elastic_params.step_flow,
                scale_traction=elastic_params.scale_traction,
                step_traction=elastic_params.step_traction,
                threshold_inf=elastic_params.threshold_inf,
                threshold_sup=elastic_params.threshold_sup,
                show=False,
            )

    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")

    return results


def main(
    optical_flow: OpticalFlowParams,
    general: GeneralParams,
    elastic_exp: ElasticExperiment,
):
    """
    Main entry point for optical flow–based strain and traction analysis.

    This function orchestrates the full processing pipeline:
      1. Initializes selected optical flow methods and their parameter sets.
      3. Runs process_case either:
         - For all experiments (1, 2, 3) if no specific parameters are given, or
         - For a single experiment, case or image if exp_ind or (T, E, ν) and/or image_id are specified.
      4. Saves the computed results as serialized `.pkl` files, CSV tables, and RMSE plots.
      5. Optionally generates comparison scatter plots between experiments.

    Args:
        optical_flow (OpticalFlowParams): Configuration object containing parameter sets for each supported optical flow method
        general (GeneralParams): General configuration (mainly result storage)
        elastic_exp (ElasticExperiment): Parameters of the experiment of interest
        pivlab_path (Path, optional): Base path to PIVlab results. If provided, PIV displacements are loaded and included in the analysis.

    Raises:
        ValueError:
            - If an unknown optical flow method name is provided in `experiment.of_funcs`.
            - If experiment parameters are inconsistently defined (handled within `process_case`).
        FileNotFoundError:
            If required image or displacement files are missing for a given `(T, E, ν, image_id)`.
    """

    of_methods = {
        "farneback": (farneback, optical_flow.farneback),
        "hs": (hs_of, optical_flow.hs),
        "tvl1": (tv_l1, optical_flow.tvl1),
        "ilk": (ilk, optical_flow.ilk),
        "fista": (fista_of, optical_flow.fista),
        "raft": (raft_algo, {}),
    }

    of_for_computation, params_for_computation = [], []

    include_wofv = False
    include_piv = False
    for of_func_name in elastic_exp.of_funcs:
        if of_func_name == "wofv":
            include_wofv = True

        elif of_func_name == "piv":
            include_piv = True

        elif of_func_name not in of_methods:
            raise ValueError(f"Unknown optical flow method '{of_func_name}'")

        else:
            of_func, of_params = of_methods[of_func_name]
            of_for_computation.append(of_func)
            params_for_computation.append(of_params)

    if elastic_exp.scatter_comparison:
        dfs = []
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
        cases_done = {}
        for exp_ind in [1, 2, 3]:
            results_exp, cases_done = process_case(
                elastic_params=elastic_exp,
                results_dir=Path(general.results_dir),
                of_for_computation=of_for_computation,
                params_for_computation=params_for_computation,
                include_wofv=include_wofv,
                include_piv=include_piv,
                exp_ind=exp_ind,
                T=elastic_exp.T,
                E=elastic_exp.E,
                nu=elastic_exp.nu,
                image_id=elastic_exp.image_id,
                cases_done=cases_done,
            )
            with open(
                Path(general.results_dir)
                / "tables_dict"
                / f"results_exp_{exp_ind}.pkl",
                "wb",
            ) as f:
                pickle.dump(results_exp, f)
            df_exp = results_to_df(results_exp)
            df_exp.to_csv(
                Path(general.results_dir)
                / "tables_dict"
                / f"mean_rmse_experiment_{exp_ind}.csv",
                index=True,
            )
            save_table_rmse_with_std(
                df_exp,
                Path(general.results_dir)
                / "plots"
                / f"mean_rmse_experiment_{exp_ind}.png",
            )
            if elastic_exp.scatter_comparison:
                dfs.append(df_exp)
        if elastic_exp.scatter_comparison:
            save_scatter_comparison(dfs, Path(general.results_dir))

    else:
        results, _ = process_case(
            elastic_params=elastic_exp,
            results_dir=Path(general.results_dir),
            of_for_computation=of_for_computation,
            params_for_computation=params_for_computation,
            include_wofv=include_wofv,
            include_piv=include_piv,
            exp_ind=elastic_exp.exp_ind,
            T=elastic_exp.T,
            E=elastic_exp.E,
            nu=elastic_exp.nu,
            image_id=elastic_exp.image_id,
            cases_done={},
        )
        if elastic_exp.image_id is None:
            if elastic_exp.exp_ind is not None:
                path = f"experiment_{elastic_exp.exp_ind}"
            else:
                path = f"T_{elastic_exp.T}_E_{elastic_exp.E}_nu_{elastic_exp.nu}"

            with open(
                Path(general.results_dir) / "tables_dict" / f"{path}.pkl", "wb"
            ) as f:
                pickle.dump(results, f)

            df = results_to_df(results)
            df.to_csv(
                Path(general.results_dir) / "tables_dict" / f"mean_rmse_{path}.csv",
                index=True,
            )
            save_table_rmse_with_std(
                df, Path(general.results_dir) / "plots" / f"mean_rmse_{path}.png"
            )


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
