#%%
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary, CellType
from dolfinx.fem import (
    Function,
    dirichletbc,
    locate_dofs_topological,
    form,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from ufl import curl, TrialFunction, TestFunction, inner, dx, as_vector
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import boundary_marker, par_print
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import meshtags

comm = MPI.COMM_WORLD
degree = 1

n = 20
domain = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=CellType.hexahedron)

facet_dim = domain.topology.dim - 1

DG = fem.functionspace(domain, ("DG", 0))
alpha_value = 1.0
alpha = Function(DG)
alpha.interpolate(lambda x: np.where(x[0] <= 0.5, alpha_value, alpha_value))
beta = Function(DG)

size_beta = 4.0
beta_loc = size_beta/n
eps = 1e-12

beta.interpolate(
    lambda x: np.where(
        (np.abs(x[0] - 0.5) < beta_loc + eps) &
        (np.abs(x[1] - 0.5) < beta_loc + eps) &
        (np.abs(x[2] - 0.5) < beta_loc + eps),
        1.0, 0.0
    )
)

num_cells = domain.topology.index_map(domain.topology.dim).size_local
cell_indices = np.arange(num_cells, dtype=np.int32)

beta_cell_values = beta.x.array[:num_cells].astype(np.int32)

ct = meshtags(domain, domain.topology.dim, cell_indices, beta_cell_values)
with XDMFFile(domain.comm, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(ct, domain.geometry)

tdim = domain.topology.dim
fdim = tdim - 1

facets = locate_entities_boundary(
    domain, dim=(domain.topology.dim - 1), marker=boundary_marker
)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

V_CG = fem.functionspace(domain, ("CG", degree))

total_dofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs

A = TrialFunction(A_space)
v = TestFunction(A_space)

f = as_vector((1.0, 1.0, 1.0))

a = form(inner(alpha * curl(A), curl(v)) * dx + inner(beta * A, v) * dx)
L = form(inner(f, v) * dx)


# Boundary conditions

dofs = locate_dofs_topological(V=A_space, entity_dim=fdim, entities=facets)
u_bc = Function(A_space)
bc = dirichletbc(u_bc, dofs)

# Solver steps

A_mat = assemble_matrix(a, bcs=[bc])
A_mat.assemble()

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

uh = fem.Function(A_space)

ams_opts = {
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-10,
    "ksp_type": "cg",
    "ksp_monitor_true_residual": None,
    "pc_hypre_ams_cycle_type": 13,
    "pc_hypre_ams_tol": 0.0, # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1, #Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 25,
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_opts.items():
    opts[option] = value
opts.prefixPop()

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

G = discrete_gradient(V_CG._cpp_object, A_space._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

V_interior = fem.functionspace(domain, ("CG", degree))
interior_nodes_array = fem.Function(V_interior)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = V_interior.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tagged_cells = ct.find(1)# Conductive tags

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()

pc.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

# #Export interior nodes to XDMF for visualization
# with XDMFFile(domain.comm, "interior_nodes.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(interior_nodes_array)


if degree == 1:
    cvec_0 = Function(A_space)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(A_space)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(A_space)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )
    pc.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, A_space._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)


ksp.setFromOptions()
ksp.setUp()
pc.setUp()

ksp.solve(b, uh.x.petsc_vec)

# Output to bp

# X = fem.functionspace(domain, ("Discontinuous Lagrange", degree, (domain.geometry.dim,)))
# A_vis = fem.Function(X)
# A_vis.interpolate(uh)

# A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
# A_file.write(0.0)


res = A_mat * uh.x.petsc_vec - b
par_print(comm, f"Residual norm: {res.norm()}")

iterations = ksp.getIterationNumber()
par_print(comm, f"Number of iterations: {iterations}")

reason = ksp.getConvergedReason()
par_print(comm, f"Convergence reason: {reason}")


print("A_mat norm:", A_mat.norm())
print("b norm:", b.norm())
print("G.norm:", G.norm())
print("uh.x.norm():", np.linalg.norm(uh.x.array))
# %%
