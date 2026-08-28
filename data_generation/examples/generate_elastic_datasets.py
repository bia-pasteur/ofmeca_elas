"""Useful to generate synthetic images of deforming cells"""

# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
import os
from pathlib import Path

import jsonargparse
import numpy as np
import scipy.ndimage as ndi
import tifffile

from data_generation.src.config import ElasticSimuParams
from data_generation.src.imaging.generator import create_image_simu, dirichlet
from data_generation.src.mesh.creation import gmsh_cell_from_image


def create_elastic_cell_image(
    img: np.ndarray,
    masks: np.ndarray,
    slices: list[tuple[slice, ...] | None],
    cell_ind: int,
    t_end: int,
    num_time_steps: int,
    traction_zone: float,
    youngs_modulus: float,
    nu: float,
    name: str,
    eta: float | None = 0,
):
    """
    Creates an image of a cell undergoing elastic deformation.
    The cell is obtained using an image of various cells and the mask associated
    with the cell of interest. It is then meshed and deformed by solving a linear
    elastic problem using the finite elements methods. The images of the cell before and after
    deformation are then recreated by getting the intensities of the pixels back from the original image.

    Args:
        img (np.ndarray): The original image of cells
        masks (np.ndarray): The mask of the cells in the image
        slices (List[Optional[Tuple[slice, ...]]]): A list of tuples of slices representing the bounding box of each
                                                    labeled object in the mask array. Each element corresponds to a
                                                    label (1-indexed), and is None if the label is not present in the mask.
        cell_ind (int): The index of the cell used to create the new image
        t_end (int): End time of the simulation
        num_time_steps (int): Numbe rof time steps of the simulation
        traction_zone (float): Traction applied on the cell
        youngs_modulus (float): Young's modulus of the cell
        nu (float): Poisson's ratio of the cell
        name (str): Name to save the image
        eta (Optional, float): Viscosity of the cell

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - u_ground_truth (np.ndarray): Displacement field array of shape (2, 1, H, W),
              representing ground-truth displacements between time steps.
            - img_final (np.ndarray): Normalized image (float array in [0, 1]) of shape (2, H, W)
              representing the cell in its initial condition and the deformed cell.
    """
    img_single_cell = np.pad(
        np.where(masks == cell_ind + 1, img, 0)[
            slices[cell_ind][0], slices[cell_ind][1]
        ],
        [50, 50],
        "constant",
    )
    u_list, warped_image_path = create_image_simu(
        gmsh_cell_from_image,
        dirichlet,
        t_end,
        num_time_steps,
        traction_zone,
        youngs_modulus,
        nu,
        eta,
        name,
        img=img_single_cell,
    )
    u_list = u_list.transpose(3, 0, 1, 2)
    u_gt = u_list[:, 1:]
    u_ground_truth = np.zeros_like(u_gt)
    u_ground_truth[0] = u_gt[1]
    u_ground_truth[1] = u_gt[0]

    img_final = tifffile.imread(warped_image_path)
    img_final = img_final / img_final.max()

    return u_ground_truth, img_final


def main(elas_simu: ElasticSimuParams):
    """
    Generates simulated displacement fields and corresponding images for a set of experiments.

    This function runs simulations across different parameter combinations for traction
    force (T), Young’s modulus (E), and Poisson’s ratio (nu), generating synthetic
    images and displacement fields. Results are saved as .npy files in structured
    experiment directories. The images are generated from already existing images of cells
    with masks associated to isolate each cell in the image and perform the simulations
    on every cell present in the original image.

    Args:
        elas_simu (ElasticSimuParams): Experiment configuration parameters

    Notes:
        Output files are saved in directories following the convention `experiment_{n}/T_{T}_E_{E}_nu_{nu}/`
        in files and `name_img.npy` for images and `name_ugt.npy` for displacement fields, where name is the name of the image.
        `n` associated with the experiment works as follows: 1 for varying T, 2 for varying E, 3 for varying nu.
    """

    # Experiment 1, varying T
    ym = float(elas_simu.ym_for_t_nu)
    nu = elas_simu.nu_for_ym_t

    for tzone in elas_simu.traction_zone:
        tzone = float(tzone)

        for i in range(len(elas_simu.img_paths)):
            img_path = elas_simu.img_paths[i]
            masks_path = Path(f"{elas_simu.masks_paths[i]}")

            img = tifffile.imread(img_path)
            masks = tifffile.imread(masks_path)
            slices = ndi.find_objects(masks)

            for cell_ind in range(masks.max()):
                if i == 1 and cell_ind == 3:
                    # Ignore cells that have bugs
                    continue
                else:
                    name = f"im_{i:02}_cell_{cell_ind:03}"

                    print(
                        f"\n Running simulation {name} for T={tzone}, E={ym}, nu={nu}"
                    )

                    u_gt, img_final = create_elastic_cell_image(
                        img=img,
                        masks=masks,
                        slices=slices,
                        cell_ind=cell_ind,
                        t_end=elas_simu.t_end,
                        num_time_steps=elas_simu.num_time_steps,
                        traction_zone=tzone,
                        youngs_modulus=ym,
                        nu=nu,
                        name=name,
                    )

                    output_dir = os.path.join(
                        "data/elas", f"experiment_1/T_{tzone}_E_{ym}_nu_{nu}"
                    )

                    os.makedirs(output_dir, exist_ok=True)

                    np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
                    np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

    # Experiment 2, varying E
    tzone = float(elas_simu.t_for_ym_nu)
    nu = elas_simu.nu_for_ym_t

    for ym in elas_simu.youngs_modulus:
        ym = float(ym)

        for i in range(len(elas_simu.img_paths)):
            img_path = elas_simu.img_paths[i]
            masks_path = elas_simu.masks_paths[i]
            img = tifffile.imread(img_path)
            masks = tifffile.imread(masks_path)
            slices = ndi.find_objects(masks)

            for cell_ind in range(masks.max()):
                if i == 1 and cell_ind == 3:
                    # Ignore cells that have bugs
                    continue
                else:
                    name = f"im_{i:02}_cell_{cell_ind:03}"

                    print(
                        f"\n Running simulation {name} for T={tzone}, E={ym}, nu={nu}"
                    )
                    u_gt, img_final = create_elastic_cell_image(
                        img=img,
                        masks=masks,
                        slices=slices,
                        cell_ind=cell_ind,
                        t_end=elas_simu.t_end,
                        num_time_steps=elas_simu.num_time_steps,
                        traction_zone=tzone,
                        youngs_modulus=ym,
                        nu=nu,
                        name=name,
                    )

                    output_dir = os.path.join(
                        "data/elas", f"experiment_2/T_{tzone}_E_{ym}_nu_{nu}"
                    )

                    os.makedirs(output_dir, exist_ok=True)

                    np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
                    np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)

    # Experiment 3, varying nu
    tzone = float(elas_simu.t_for_ym_nu)
    ym = float(elas_simu.ym_for_t_nu)

    for nu in elas_simu.nu:
        nu = float(nu)

        for i in range(len(elas_simu.img_paths)):
            img_path = elas_simu.img_paths[i]
            masks_path = elas_simu.masks_paths[i]
            img = tifffile.imread(img_path)
            masks = tifffile.imread(masks_path)
            slices = ndi.find_objects(masks)

            for cell_ind in range(masks.max()):
                if i == 1 and cell_ind == 3:
                    # Ignore cells that have bugs
                    continue
                else:
                    name = f"im_{i:02}_cell_{cell_ind:03}"

                    print(
                        f"\n Running simulation {name} for T={tzone}, E={ym}, nu={nu}"
                    )

                    u_gt, img_final = create_elastic_cell_image(
                        img=img,
                        masks=masks,
                        slices=slices,
                        cell_ind=cell_ind,
                        t_end=elas_simu.t_end,
                        num_time_steps=elas_simu.num_time_steps,
                        traction_zone=tzone,
                        youngs_modulus=ym,
                        nu=nu,
                        name=name,
                    )

                    output_dir = os.path.join(
                        "data/elas", f"experiment_3/T_{tzone}_E_{ym}_nu_{nu}"
                    )

                    os.makedirs(output_dir, exist_ok=True)

                    np.save(os.path.join(output_dir, f"{name}_ugt.npy"), u_gt)
                    np.save(os.path.join(output_dir, f"{name}_img.npy"), img_final)


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False)
