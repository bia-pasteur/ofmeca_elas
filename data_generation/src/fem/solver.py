"""Solves the FEM problem"""

# pylint: disable=invalid-name
from typing import Callable, List, Dict, Optional
import ufl
import numpy as np
import random
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, apply_lifting, set_bc
from ufl import inner, TrialFunction, TestFunction
from shapely import Polygon, contains_xy, minimum_bounding_radius
from shapely.affinity import translate
from petsc4py import PETSc
from data_generation.src.mesh.creation import create_cell_like_shape

def strain(
    u: ufl.Argument
) -> ufl.Form:
    """
    Computes the strain of the displacement field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
    Returns:
        ufl.Form: The strain tensor
            Shape: (d, d)
    """
    return ufl.sym(ufl.grad(u))


def stress_elas(
    u: ufl.Argument, 
    lambda_e: float, 
    mu_e: float
) -> ufl.Form:
    """
    Computes the elastic stress of the displacement field (Hooke's law)

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_e (float): Lamé elastic parameter
        mu_e (float): Lamé elastic parameter

    Returns:
        ufl.Form: The elastic stress tensor
            Shape: (d, d)
    """
    return lambda_e * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_e * strain(u)


def stress_visc(
    u: ufl.Argument, 
    lambda_v: float, 
    mu_v: float, 
    delta_t: float
) -> ufl.Form:
    """
    Computes the viscous stress tensor of the displacmeent field

    Args:
        u (ufl.Argument): Displacement field
            Shape: (d,)
        lambda_v (float): Lamé viscous parameter
        mu_v (float): Lamé viscous parameter
        delta_t (float): The time step 

    Returns:
        ufl.Form: The viscous stress tensor
            Shape: (d, d)
    """
    return (lambda_v/delta_t)*ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * (mu_v/delta_t) * strain(u)


def create_dirichlet(
    msh: mesh.Mesh, 
    dfacets: callable, 
    V: fem.FunctionSpace, 
    boundary_x: np.ndarray, 
    boundary_y: np.ndarray, 
    zone_radius: float = None, 
    zone_center: tuple = None
) -> fem.DirichletBC:
    """
    Creates a Dirichlet BC on the mesh boundary outside the zone of traction application.

    Args:
        msh (mesh.Mesh): Finite elements mesh
        dfacets (callable): Function defining the Dirichlet condition
        V (fem.FunctionSpace): Function space on the mesh
        boundary_x (np.ndarray): x coordinates of the boundary elements
        boundary_y (np.ndarray): y coordinates of the boundary elements
        zone_center (tuple, optional): Center of the zone of aspiration. Defaults to None.
        zone_radius (float, optional): Radius of the zone of aspiration. Defaults to None.

    Returns:
        fem.DirichletBC: _description_
    """
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim,
        lambda xx: dfacets(xx, boundary_x, boundary_y, zone_center, zone_radius)
    )

    u_dirichlet = np.array([0, 0, 0], dtype=default_scalar_type)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    return fem.dirichletbc(u_dirichlet, dofs, V)


def finite_elements_force_zone(    
    msh: dolfinx.mesh.Mesh,
    V: fem.FunctionSpace,
    cond_init: Callable[[np.ndarray], np.ndarray],
    dirichlet: Callable[[np.ndarray, tuple, float, float], np.ndarray],
    t_end: float,
    num_time_steps: int,
    traction_constant: fem.Function,
    youngs_modulus: float,
    nu: float,
    eta: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    zone_radius: float = None,
    zone_center: tuple = None,
    physical_length: float = None,
    subparts: Optional[Dict] = None, 
    seed: 1 = None
) -> List[np.ndarray]:
    """
    Performs a simulation of a Kelvin-Voigt viscoelastic or linear elastic deformation
    on a mesh representing a cell under constant traction force with possible variation 
    of the Young's modulus and viscosity inside the cell.

    Args:
        msh (mesh.Mesh): Finite element mesh of the computational domain
        V (fem.FunctionSpace): Function space for displacements
        cond_init (Callable): Initial condition function (displacement field)
        dirichlet (Callable): Function defining Dirichlet BCs at the pipette
        t_end (float): Total simulation time
        num_time_steps (int): Number of discrete time steps
        traction_constant (fem.Function): Traction force applied on the pipette boundary
        youngs_modulus (float): Young’s modulus of the cell, outside of eventual subparts
        nu (float): Poisson's ratio of the cell, outside of eventual subparts
        eta (float): Viscosity of the cell, outside of eventual subparts
        x_coords (np.ndarray): x-coordinates of the boundary of the cell
        y_coords (np.ndarray): y-coordinates of the boundary of the cell
        zone_radius (float, optional): If the simulation isn't run on real cell images, the radius of the zone of aspiration. Defaults to None.
        zone_center (tuple, optional): If the simulation isn't run on real cell images, the center of the zone of aspiration. Defaults to None.
        physical_length (float, optional): If the simulation isn't run on real cell images, the physical length of the. Defaults to None.
        subparts (Optional[Dict], optional): Dictionary of subpart definitions. Each value must contain: 
                                            - 'young's_modulus' (float): Young's modulus of the subpart
                                            - 'eta.             (float): viscoscisty of the subpart.
                                            Note : for now, the subparts are randomly generated. Defaults to None.
        seed (1, optional): Random seed for shape generation in the subparts. Defaults to None.

    Returns:
        List[np.ndarray]: Displacement field at each time step
    """
    # Define the time step
    delta_t = t_end/num_time_steps
    
    # Function to compute Lamé parameters
    def lame(E__, nu__, eta__):
        # Elastic Lamé parameters
        lambda_e = E__*nu__ / ((1+nu__)*(1-2*nu__))
        mu_e = E__ / (2*(1+nu__))

        # Viscous Lamé parameters
        lambda_v = -eta__/3
        mu_v = eta__/2
        return lambda_e, mu_e, lambda_v, mu_v

    # Assign interior elements a Young's modulus, Poisson's ratio and viscosity
    F = fem.functionspace(msh, ("DG", 0))
    youngs_modulus_ = fem.Function(F)
    eta_ = fem.Function(F)
    nu_ = fem.Function(F)
    youngs_modulus_.x.array[:] = youngs_modulus
    nu_.x.array[:] = nu
    eta_.x.array[:] = eta
    
    # If subparts of varying Young's modulus and viscosity are defined, create them
    if subparts is not None:
        polygon_cell = Polygon(zip(x_coords, y_coords))
        minx, miny, maxx, maxy = polygon_cell.bounds
        min_rad = minimum_bounding_radius(polygon_cell)
        sp_polygons = {}
        for key, subpart in subparts.items():
            rng = np.random.default_rng(seed=seed+key)
            x_sp, y_sp = create_cell_like_shape(
                50, base_radius=(min_rad/4), noise_amplitude=2, num_fourier_modes=30, rng=rng, r_min_ratio=0.1
            )
            x_c, y_c = random.uniform(minx, maxx), random.uniform(miny, maxy)
            sp_polygon = Polygon(list(zip(x_sp + x_c, y_sp + y_c)))
            sp_polygons[key] = sp_polygon
        
        rng = np.random.default_rng(seed=1000)
        minx, miny, maxx, maxy = polygon_cell.bounds
        
        for key in sp_polygons:
            for _ in range(10000):
                dx = rng.uniform(minx, maxx)
                dy = rng.uniform(miny, maxy)

                centroid = sp_polygons[key].centroid
                shifted = translate(sp_polygons[key], dx - centroid.x, dy - centroid.y)
                shifted_buffer = shifted.buffer(0.1)

                if not polygon_cell.contains(shifted_buffer):
                    continue
                
                if any(shifted_buffer.intersects(sp_polygons[i]) for i in sp_polygons if i != key):
                    continue
                
                sp_polygons[key] = shifted
                break 
            
        dim = msh.topology.dim
        for key, subpart in subparts.items():
            sp_polygon = sp_polygons[key]
            ym_sp = subpart['youngs_modulus']
            eta_sp = subpart['eta']
            
            def omega_sp(x, poly=sp_polygon):
                return contains_xy(poly, x[0], x[1])
            
            cells_i = dolfinx.mesh.locate_entities(msh, dim, omega_sp)
            youngs_modulus_.x.array[cells_i] = np.full_like(cells_i, ym_sp, dtype=default_scalar_type)
            eta_.x.array[cells_i] = np.full_like(cells_i, eta_sp, dtype=default_scalar_type)
            nu_.x.array[cells_i] = np.full_like(cells_i, nu, dtype=default_scalar_type)

    lambda_e, mu_e, lambda_v, mu_v = lame(youngs_modulus_, nu_, eta_)
    
    # Define a function for the old values
    u_old = fem.Function(V, name="Previous step")
    u_old.interpolate(cond_init)

    # Define solution variable, and interpolate initial solution
    uh = fem.Function(V, name="Solution variable")
    uh.interpolate(cond_init)
    
    # Definition of the integration measures
    ds = ufl.Measure("ds", domain=msh)
    dx = ufl.Measure("dx", domain=msh)
    displacement_history = []
    
    # Create dirichlet conditions 
    bc = create_dirichlet(msh, dirichlet, V, x_coords, y_coords, zone_radius, zone_center)
    # Create the trial and test function
    u, v = TrialFunction(V), TestFunction(V)

    # Weak form
    a = inner(stress_elas(u, lambda_e, mu_e) + stress_visc(u, lambda_v, mu_v, delta_t), strain(v))*dx
    L = inner(traction_constant, v)*ds + inner(stress_visc(u_old, lambda_v, mu_v, delta_t), strain(v))*dx
    # Convert the left hand side (LSH) and the right hand side (RHS) into a DOLFINx-compatible representation
    compiled_a = fem.form(a)
    compiled_L = fem.form(L)

    # Allocate memory for the solution
    b = fem.Function(V)

    # Assemble the stiffness matrix A from a and apply Dirichlet boundary conditions
    # Linear system : Au = b
    A = assemble_matrix(compiled_a, bcs=[bc])
    A.assemble()

    # Create the solver 
    solver = PETSc.KSP().create(msh.comm) # pylint: disable=no-member

    # Set the system matrix
    solver.setOperators(A)

    # Using direct factorization
    solver.setType(PETSc.KSP.Type.PREONLY) # pylint: disable=no-member

    # Using LU decomposition as a precondition
    # Fast for small to medium problems, memory-intensive
    solver.getPC().setType(PETSc.PC.Type.LU) # pylint: disable=no-member
    
    displacement_history.append(uh.copy())
    
    # Iteratively solve the problem
    for _ in range(num_time_steps):
        # Reset the RHS
        b.x.array[:] = 0
        # Assemble the RHS => construct Au = b
        assemble_vector(b.x.petsc_vec, compiled_L)
        # Apply the Dirichlet boundary conditions
        apply_lifting(b.x.petsc_vec, [compiled_a], [[bc]])
        set_bc(b.x.petsc_vec, [bc])
        # Solve the linear system
        solver.solve(b.x.petsc_vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        set_bc(uh.x.petsc_vec, [bc])
        # Update the old values
        u_old.x.array[:] = uh.x.array.copy()
        # Store displacement (either as an array or a copy of uh)
        #displacement_history.append(uh.x.array)
        displacement_history.append(uh.copy())
    return displacement_history