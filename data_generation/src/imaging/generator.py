"""Generator of the images"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
from typing import Callable, Union, List, Tuple, Optional, Dict
import os
import numpy as np
import tifffile
import skimage
import matplotlib.pyplot as plt
import dolfinx
from noise import pnoise2
from dolfinx import fem, default_scalar_type, mesh
from dolfinx.mesh import compute_midpoints
from petsc4py.PETSc import ScalarType # pylint: disable=no-name-in-module
from data_generation.src.mesh.creation import create_mesh_file, create_cell_like_shape
from data_generation.src.fem.solver import finite_elements_force_zone


def closest_cell(
    mesh: dolfinx.mesh.Mesh,
    point: np.ndarray,
    candidate_cells: np.ndarray
) -> int:
    """
    Find the cell (element) in the mesh that is closest to a given point,
    restricted to a list of candidate cells.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh.
        point (np.ndarray): Target point in physical coordinates, shape (3,).
        candidate_cells (np.ndarray): Array of candidate cell indices.

    Returns:
        int: Index of the cell in `candidate_cells` whose midpoint
             is closest to the given point.
    """
    midpoints = compute_midpoints(mesh, mesh.topology.dim, np.array(candidate_cells, dtype=np.int32))
    dists = np.linalg.norm(midpoints - point, axis=1)
    return candidate_cells[np.argmin(dists)]


def interpolation(
    mesh: dolfinx.mesh.Mesh,
    u: fem.Function,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Interpolates a finite element vector field u on a regular 2D or 3D grid.

    Args:
        mesh (dolfinx.mesh.Mesh): Computational mesh where u is defined.
        u (fem.Function): Vector-valued FE function (displacement field)
        x_range (Tuple[float, float]): (xmin, xmax) bounds of interpolation domain along x-axis.
        y_range (Tuple[float, float]): (ymin, ymax) bounds of interpolation domain along y-axis.
        z_range (Optional[Tuple[float, float]]): (zmin, zmax) bounds along z-axis. If None → 2D interpolation. Defaults to None.

    Returns:
        np.ndarray: Array of interpolated values.
            - Shape (H, W, 2) for 2D interpolation
            - Shape (H, W, D, 3) for 3D interpolation
    """

    H, W = y_range[1], x_range[1]
    
    x = np.linspace(x_range[0], x_range[1], W)  
    y = np.linspace(y_range[0], y_range[1], H) 
    
    if z_range is None:
        # 2D case
        X, Y = np.meshgrid(x, y, indexing='xy')
        pixels = np.stack((X, Y, np.zeros_like(X)), axis=-1)
        dim = 2
        val_shape = (H, W, 2)
    else:
        # 3D case
        D = z_range[1]
        z = np.linspace(z_range[0], z_range[1], D)  
        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
        pixels = np.stack((X, Y, Z), axis=-1)
        dim = 3
        val_shape = (H, W, D, 3)

    flattened_pix = pixels.reshape(-1, 3)
    
    # Build bounding box tree for the correct dimension
    bounding_box = dolfinx.geometry.bb_tree(mesh, dim)
    
    # Find colliding cells
    cells = dolfinx.geometry.compute_collisions_points(bounding_box, flattened_pix)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cells, flattened_pix)

    pixels_interpolation = []
    cells_interpolation = []

    for i, pixel in enumerate(flattened_pix):
        if len(colliding_cells.links(i)) > 0:
            best_cell = closest_cell(mesh, pixel, colliding_cells.links(i))
            pixels_interpolation.append(pixel)
            cells_interpolation.append(best_cell)

    pixels_interpolation = np.array(pixels_interpolation, dtype=np.float64)
    
    u_values = u.eval(pixels_interpolation, cells_interpolation)

    # Fill the output array
    val = np.zeros(val_shape)
    mask = np.zeros(val_shape)
    count = 0

    if dim == 2:
        for i in range(H):
            for j in range(W):
                if count < len(pixels_interpolation):
                    if np.array_equal(pixels[i, j], pixels_interpolation[count]):
                        val[i, j, :] = u_values[count, :2]
                        mask[i, j, :] = 1
                        count += 1
    else:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if count < len(pixels_interpolation):
                        if np.array_equal(pixels[i, j, k], pixels_interpolation[count]):
                            val[i, j, k, :] = u_values[count, :]
                            mask[i, j, k, :] = 1
                            count += 1

    return val, mask


def dirichlet(
    nodes: np.ndarray, 
    boundary_x: np.ndarray = None, 
    boundary_y: np.ndarray = None,
    zone_center: tuple = None, 
    zone_radius: float = None
) -> np.ndarray:
    """
    Determines if nodes are on the Dirichlet boundary. 
    For cells defined using random shapes, this means having coordinates matching boundary_x, boundary_y and 
    being outside the zone in the y-direction.
    For cells defined using real images, this is defined using the quantiles. 

    Args:
        nodes (np.ndarray): Coordinates of points on facets, shape (d, N)
        boundary_x (np.ndarray, optional): x-coordinates of the boundary points. Defaults to None.
        boundary_y (np.ndarray, optional): y-coordinates of the boundary points. Defaults to None.
        zone_center (tuple, optional): Center of the zone of aspiration. Defaults to None.
        zone_radius (float, optional): Radius of the zone of aspiration. Defaults to None.

    Returns:
        np.ndarray: Boolean array of length N, True means Dirichlet should apply
    """
    
    x, y, z = nodes
    n = x.shape[0]

    # If the mesh is defined from scratch using a random shape, the Dirichlet boundary is defined from the center of the zone
    if zone_center is not None:
        on_boundary = np.zeros(n, dtype=bool)
        for bx, by in zip(boundary_x, boundary_y):
            on_boundary |= (np.isclose(x, bx) & np.isclose(y, by))

        in_zone = (y > 0) & (x >= -zone_radius) & (x <= zone_radius)

        return np.logical_and(on_boundary, ~in_zone)
    
    # Else, the Dirichlet boundary is defined using the quantiles
    else : 
        n = x.shape[0]
        q20, q80 = np.quantile(x, [0.20, 0.8])
        q60 = np.quantile(y, 0.6)

        mask_in = (x >= q20) & (x <= q80) & (y >= q60)

        return ~mask_in 


def create_intensities_perlin(
    mesh: dolfinx.mesh.Mesh, 
    scale: float = 5.0, 
    grain: int = 3, 
    seed: int = 0
) -> dolfinx.fem.Function:
    """
    Create a discontinuous Galerkin (DG) function on the mesh where
    the cell-wise values follow a 2D Perlin noise pattern.

    This function assigns a Perlin noise intensity value to each cell of the mesh,
    normalized between 0 and 1, and stores it in a DG finite element function.

    Args:
        mesh (dolfinx.mesh.Mesh): The input computational mesh.
        scale (float): Frequency of the Perlin noise. Smaller values
            produce larger-scale (smoother) patterns.
        grain (int): Polynomial degree of the DG elements.
        seed (int): Random seed for the Perlin noise generator.

    Returns:
        dolfinx.fem.Function: DG function representing the normalized
            Perlin noise intensity field over the mesh.
    """
    Q = fem.functionspace(mesh, ("DG", grain))
    intensities = fem.Function(Q)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    coords = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells))

    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    coords_norm = np.zeros_like(coords)
    coords_norm[:,0] = (coords[:,0]-xmin)/(xmax-xmin)
    coords_norm[:,1] = (coords[:,1]-ymin)/(ymax-ymin)

    values_per_cell = np.zeros(num_cells)
    for i in range(num_cells):
        x = coords_norm[i,0] * scale
        y = coords_norm[i,1] * scale
        values_per_cell[i] = pnoise2(x, y, octaves=5, persistence=0.5, repeatx=1024, repeaty=1024, base=seed)
    min_val, max_val = np.percentile(values_per_cell, [1, 99])  
    values_per_cell = np.clip(values_per_cell, min_val, max_val)
    values_per_cell = (values_per_cell - min_val) / (max_val - min_val)

    dofmap = Q.dofmap
    intensities_vals = np.zeros_like(intensities.x.array)
    for cell_index in range(num_cells):
        dofs = dofmap.cell_dofs(cell_index)
        intensities_vals[dofs] = values_per_cell[cell_index]

    intensities.x.array[:] = intensities_vals
    return intensities


def map_intensities_from_image(
    mesh: dolfinx.mesh.Mesh, 
    img: np.ndarray
) -> dolfinx.fem.Function:
    """
    Create a discontinuous Galerkin (DG) function on the mesh where
    the cell-wise values follow the brightness pattern of a cell image

    Args:
        mesh (dolfinx.mesh.Mesh): The input computational mesh.
        img (np.ndarray): Image of a masked cell

    Returns:
        dolfinx.fem.Function: DG function representing the intensity of the cell image over the mesh.
    """
    Q = fem.functionspace(mesh, ("CG", 2))
    intensities = fem.Function(Q)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    coords = compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells))

    values_per_cell = np.zeros(num_cells)
    for i in range(num_cells):
        x = int(coords[i,0])
        y = int(coords[i,1])
        values_per_cell[i] = img[y,x]
    min_val, max_val = np.percentile(values_per_cell, [1, 99])  
    values_per_cell = np.clip(values_per_cell, min_val, max_val)
    values_per_cell = (values_per_cell - min_val) / (max_val - min_val)

    dofmap = Q.dofmap
    intensities_vals = np.zeros_like(intensities.x.array)
    for cell_index in range(num_cells):
        dofs = dofmap.cell_dofs(cell_index)
        intensities_vals[dofs] = values_per_cell[cell_index]

    intensities.x.array[:] = intensities_vals
    return intensities


def deform_mesh(
    mesh: dolfinx.mesh.Mesh, 
    u: dolfinx.fem.Function
) -> dolfinx.mesh.Mesh:
    """
    Deform the input mesh according to a given displacement field.

    The displacement field u is evaluated at the mesh geometry points,
    and each node of the mesh is updated by adding the corresponding
    displacement vector.

    Args:
        mesh (dolfinx.mesh.Mesh): The computational mesh to deform.
        u (dolfinx.fem.Function): Displacement field defined on the mesh.

    Returns:
        dolfinx.mesh.Mesh: The same mesh object with updated coordinates.
    """
    x = mesh.geometry.x
    # Initialise cell search
    bounding_box = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bounding_box, x)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x)
    for i, point in enumerate(x):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # Evaluate u
    u_values = u.eval(points_on_proc, cells)
    # Update mesh coordinates
    x += u_values

    return mesh


def plot_boundary_conditions(
    msh: dolfinx.mesh.Mesh, 
    intensities: dolfinx.fem.Function, 
    x_range: Tuple[float, float], 
    y_range: Tuple[float, float], 
    save_path: str
) -> None:
    """
    Plots the image of the cell with the associated boundayr conditions

    Args:
        msh (dolfinx.mesh.Mesh): The computational mesh
        intensities (dolfinx.fem.Function): DG function representing the intensity of the cell image over the mesh
        x_range (Tuple[float, float]): Spatial range in the x-direction.
        y_range (Tuple[float, float]): Spatial range in the y-direction.
        save_path (str): Path to the folder where the plot is saved 
    """
    original_image, mask_image = interpolation(msh, intensities, x_range, y_range)
    contours = skimage.measure.find_contours(mask_image[:,:,0])
    x = contours[0][:,1]
    y = contours[0][:,0]

    q20, q80 = np.quantile(x, [0.20, 0.8])
    q60 = np.quantile(y, 0.6)

    mask_in = (x >= q20) & (x <= q80) & (y >= q60)
    mask_out = ~mask_in
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax.imshow(original_image[:,:,0], 'gray')
    ax.scatter(x[mask_in], y[mask_in], color='red', s=1, label='Neumann')
    ax.scatter(x[mask_out], y[mask_out], color='blue', s=1, label='Dirichlet')
    ax.axis('off')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Figure saved to {save_path}")
    plt.close()
  
  
def create_image_simu(
    mesh_function: Callable,
    dirichlet_: Callable,
    t_end: float,
    num_time_steps: int,
    traction_zone: float,
    youngs_modulus: float,
    nu: float,
    eta: float,
    name: str,
    img: np.ndarray=None,
    x_range: Tuple[float, float]=None,
    y_range: Tuple[float, float]=None,
    physical_length: float=None,
    zone_radius: float=None,
    zone_center: tuple=None,
    z_range: Optional[Tuple[float, float]] = None,
    subparts: Optional[Dict] = None,
    num_points: Optional[float] = None,
    noise_amplitude: Optional[float] = None,
    num_fourier_modes: Optional[float] = None,
    lc=None,
    seed=1, 
    grain=2, 
    n=None,
    seed_image=None,
) -> Tuple[np.ndarray, str]:
    """
    Generate a sequence of simulated microscopy images based on a finite element 
    (FE) deformation simulation.

    Creates a mesh (either from an image or procedurally), applies a time-dependent
    displacement due to a local traction force, and interpolates the deformed intensity
    field on a regular grid to produce a synthetic 2D image sequence saved as a TIFF.
    If no image is provided, a Perlin noise texture is used as intensity field.

    Args:
        mesh_function (Callable): Function used to generate the Gmsh mesh. Either
            gmsh_cell_from_image or gmsh_cell_shape.
        dirichlet_ (Callable): Function defining Dirichlet boundary conditions on the mesh.
        t_end (float): Final simulation time (in seconds).
        num_time_steps (int): Number of time steps in the simulation.
        traction_zone (float): Magnitude of the applied traction force (in Pa).
        youngs_modulus (Union[float, List[float]]): Young's modulus of the cell (in Pa).
            Can be a list if subparts are defined.
        nu (Union[float, List[float]]): Poisson's ratio of the cell.
            Can be a list if subparts are defined.
        eta (Union[float, List[float]]): Viscosity coefficient(s) of the cell (in Pa.s).
            Can be a list if subparts are defined. Use 0 for purely elastic.
        name (str): Base name for the output TIFF file and saved figures.
        img (np.ndarray, optional): Binary mask image of a single cell, used to
            generate the mesh and intensity field. If None, a random shape and 
            Perlin noise texture are used instead. Defaults to None.
        x_range (Tuple[float, float], optional): Spatial range in the x-direction 
            for interpolation. Inferred from img or n if not provided. Defaults to None.
        y_range (Tuple[float, float], optional): Spatial range in the y-direction 
            for interpolation. Inferred from img or n if not provided. Defaults to None.
        physical_length (float, optional): Base radius of the randomly generated cell 
            shape. Required if img is None. Defaults to None.
        zone_radius (float, optional): Radius of the active traction force zone.
            Defaults to None.
        zone_center (tuple, optional): (x, y) center coordinates of the traction 
            force zone. Defaults to None.
        z_range (Tuple[float, float], optional): Spatial range in the z-direction.
            Defaults to None.
        subparts (Dict[int, Dict[str, float]], optional): Dictionary describing 
            mechanical subregions of the cell. Keys are integer subregion indices, 
            values are dicts with keys 'youngs_modulus' (float) and 'eta' (float).
            Example: {1: {'youngs_modulus': 500.0, 'eta': 100.0}}.
            Defaults to None.
        num_points (int, optional): Number of boundary points for the procedural 
            cell shape. Required if img is None. Defaults to None.
        noise_amplitude (float, optional): Amplitude of the Fourier noise applied 
            to the cell boundary. Required if img is None. Defaults to None.
        num_fourier_modes (int, optional): Number of Fourier modes for the boundary 
            perturbation. Required if img is None. Defaults to None.
        lc (float, optional): Characteristic mesh element length. 
            Defaults to 0.1 * physical_length if None.
        seed (int, optional): Random seed for mesh generation and Perlin noise 
            texture. Defaults to 1.
        grain (int, optional): Degree of the DG function space used for Perlin 
            noise intensities. Defaults to 2.
        n (int, optional): Number of grid points per axis for interpolation output.
            Used to set x_range and y_range when img is None. Defaults to None.
        seed_image (int, optional): Random seed used for ground truth subpart 
            placement and visualization. Defaults to None.

    Returns:
        Tuple[np.ndarray, str]:
            - u_list (np.ndarray): Displacement fields interpolated on a regular grid, 
              shape (num_time_steps, H, W, 2), dtype float32, where H and W are the 
              grid dimensions inferred from y_range and x_range.
            - warped_image_path (str): Path to the saved multi-frame TIFF file 
              containing the warped image sequence, with axes 'TYX'.
    """
    rng = np.random.default_rng(seed=seed)
    
    # Create the mesh
    x_coords, y_coords, msh = create_mesh_file(mesh_function, img, physical_length, rng, num_points, noise_amplitude, num_fourier_modes, lc)
    
    # Create the function space
    # Lagrange elements, degree 1 (linear elements), vector-valued function space (one function per spatial dimension (3))
    V = fem.functionspace(msh, ('Lagrange', 2, (msh.geometry.dim,)))
    traction_constant = fem.Constant(msh, default_scalar_type((0, traction_zone, 0)))
    
    # Definition of the initial condition 
    # We initialize everything at 0
    def cond_init(x):
        return np.array([0.0*x[0], 0.0*x[1], 0.0*x[2]], dtype=ScalarType)

    # Get displacement when the force is applied on the zone
    dis = finite_elements_force_zone(msh, V, cond_init, dirichlet_, t_end, num_time_steps, traction_constant,  youngs_modulus, nu, eta, x_coords, y_coords, zone_radius, zone_center, physical_length, subparts, seed_image)
    
    # Definition of the intensities on the mesh
    if img is None: 
        intensities = create_intensities_perlin(msh, seed=seed, grain=grain)
        x_range = [0, n]
        y_range = [0, n]
        
    else: 
        intensities = map_intensities_from_image(msh, img)
        x_range = [0, img.shape[1]]
        y_range = [0, img.shape[0]]
    
    #plot_boundary_conditions(msh, intensities, x_range, y_range, Path(f"results/figures/boundary_{name}"))
    
    H, W = y_range[1], x_range[1]
    warped_images = np.zeros((num_time_steps, H, W), dtype=np.float32)
    u_list = np.zeros((num_time_steps, H, W, 2), dtype=np.float32)

    uh_prev = fem.Function(V)
    u_total = fem.Function(V) 
    u_total.x.array[:] = 0.0

    coords_initial = msh.geometry.x.copy()
    coords_current = coords_initial.copy()

    for time in range(num_time_steps):
        uh = dis[time]
        du = fem.Function(V)
        du.x.array[:] = uh.x.array - uh_prev.x.array

        u_total.x.array[:] += du.x.array

        msh.geometry.x[:] = coords_initial
        u_, _ = interpolation(msh, u_total, x_range, y_range, z_range)
        u_list[time] = u_

        msh.geometry.x[:] = coords_current
        msh = deform_mesh(msh, du)
        coords_current = msh.geometry.x.copy()

        warped_im, _ = interpolation(msh, intensities, x_range, y_range)
        warped_images[time] = warped_im[:, :, 0]

        uh_prev.x.array[:] = uh.x.array
        
    if eta == 0.0:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data/raw/elas")
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data/raw/viscoelas")
        
    output_dir = os.path.join(base_dir, f"T_{traction_zone}_E_{youngs_modulus}_eta_{eta}_nu_{nu}")
    os.makedirs(output_dir, exist_ok=True)
    warped_image_path = os.path.join(output_dir, f"{name}.tiff")
    tifffile.imwrite(warped_image_path, warped_images, metadata={'axes': 'TYX'}, imagej=True)
    return u_list, warped_image_path