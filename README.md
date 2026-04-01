# Hessian regularized optical flow algorithm for mechanobiology

This project simulates micropipette aspiration experiments on elastic cells and analyzes them with several optical flow algorithms to estimate key mechanical quantities — displacement, strain, deformation, stress, and traction forces.

The goal is to evaluate and compare different optical flow methods for estimating cellular mechanics — identifying which one provides the most accurate and physically consistent results under varying noise levels and experimental conditions.

Analysis on real microscopy images is also possible with the pipeline, but no comparison with ground truth values is available.

# Overview 

The repository is devided in two principal parts : 

### 1)  Data generation

Generates synthetic images of deformed elastic cells using FEniCSx (finite element simulation).

FEniCSx is computationally heavy. Pre-generated datasets are available at [link to dataset].

### 2) Mechanical computations

Computes mechanical quantities on the images using multiple optical flow algorithms, computes the derived mechanical quantities, and compares the results.

It is possible to :

Run the full pipeline (generate data and perform the mechanical analysis), or
Run only one part (e.g., use pre-generated data for analysis).


# Installation and full pipeline execution 

Start by cloning the repository

`git clone git@github.com/bia-pasteur/ofmeca.git`

Then create a conda environment with python=3.12 and install fenicsx using 

`conda install -c conda-forge fenics-dolfinx`

Install requirements

`pip install -r requirements.txt`

Run the code to create the images and perform the analysis using 

`./run_all.sh`


# Creation of synthetic images

First create a conda environment with python=3.12 and install fenicsx using 

`conda install -c conda-forge fenics-dolfinx`

Install requirements

`pip install -r data_generation/requirements.txt`

## Original images

To create images of deforming elastic cells using real cell, we provide images containing several cells at `img_paths`, along with the segmentation masks at `masks_paths`.

There are three simulation experiments:

| Experiment | Variable | Fixed parameters | YAML keys |
|-------------|-----------|------------------|------------|
| 1 | Traction `T` | E, ν | `traction_zone`, `ym_for_t_nu`, `nu_for_t_ym` |
| 2 | Young’s modulus `E` | T, ν | `youngs_modulus`, `t_for_ym_nu`, `nu_for_t_ym` |
| 3 | Poisson’s ratio `ν` | T, E | `nu`, `t_for_ym_nu`, `ym_for_t_nu` |

Each setting is repeated for all cells present in the provided image.

Run the code to create these images using 

`python -m data_generation.examples.generate_elastic_datasets --config=data_generation/configs/elastic_params.yaml`

Images and associated displacement are saved under:

`data/elas/experiment_i/T_<T>_E_<E>_nu_<nu>/`

## Noisy images

To test the robustness to noise, Gaussian noise of increasing stds (defined by `noise_stds`) is added to all reference images determined by `traction_zone`, `ym`, `nu` in the `noise_params.yaml` file

Run the code to create these images using 

`python -m data_generation.examples.generate_noisy_elastic_datasets --config=data_generation/configs/noise_params.yaml`

Images and associated displacements are saved under:

`data/noise_experiment_T_<T>_E_<E>_nu_<nu>/img_<seed>)`

# Optical flow analysis and mechanical quantification

Install requirements 

`pip install -r mechanics/requirements.txt`

## Sythetic images

### Edit: `mechanics/elastic_params.yaml`.

- `of_funcs`: list of optical flow algorithms to test.

- Select images by:

    - Explicit IDs: (`T`, `E`, `nu`, `image_id`)

    - Whole experiment: (`exp_ind`)

- Or run all experiments by leaving these unspecified.

Results (RMSE tables, plots, etc.) are stored under `results/tables/` and `results/plots/`

### Plotting controls:

In `elastic_exp.yaml`, the section plot_parameters defines which image to visualize:

`T_for_plot`, `E_for_plot`, `nu_for_plot`, `implot`

### Regularization study:

Use `T`, `E`, `nu`, and `factors` in the `reg_exp.yaml` file to control the regularization experiments.

## Microscopy images

### Edit: `mechanics/micro_exp.yaml`.

- `of_funcs`: list of optical flow algorithms to apply.

- Select the image to process via `im` (1 or 2), which controls the region of interest extracted from the .tif file.

- Set the path to the .tif file via `path`.

Results (plots) are stored under `results/plots/`

### Running the scripts:

| Purpose | Command | Output |
|-------------|-----------|------------|
| Run experiments on regular images | `python -m mechanics.examples.run_elastic_exp --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/elastic_exp.yaml` | RMSE table + plots |
| Run regularization robustness study | `python -m mechanics.examples.run_elastic_noise --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/noise_exp.yaml` | Combined plot (no saved tables) |
| Run noise robustness study | `python -m mechanics.examples.run_elastic_reg --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/reg_exp.yaml` | Saved results + plots |
| Run regularization and noise robustness study | `python -m mechanics.examples.run_elastic_noise_reg --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/reg_exp.yaml` | Saved results + plots |