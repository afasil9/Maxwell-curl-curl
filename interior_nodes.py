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

n = 15
domain = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=CellType.hexahedron)

facet_dim = domain.topology.dim - 1

DG = fem.functionspace(domain, ("DG", 0))
alpha = Function(DG)
alpha.interpolate(lambda x: np.where(x[0] <= 0.5, 1.0, 1.0))
beta = Function(DG)

size_beta = 4.0
beta_loc = size_beta/n
eps = 1e-12

beta.interpolate(
    lambda x: np.where(
        (np.abs(x[0] - 0.5) < beta_loc + eps) &
        (np.abs(x[1] - 0.5) < beta_loc + eps) &
        (np.abs(x[2] - 0.5) < beta_loc + eps),
        0.0, 1.0
    )
)

num_cells = domain.topology.index_map(domain.topology.dim).size_local
cell_indices = np.arange(num_cells, dtype=np.int32)
beta_cell_values = beta.x.array.astype(np.int32)

ct = meshtags(domain, domain.topology.dim, cell_indices, beta_cell_values)

with XDMFFile(domain.comm, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(ct, domain.geometry)

#%%
tdim = domain.topology.dim
fdim = tdim - 1

cell_idx = ct.find(1) # Find cells with beta 1
cell_to_vertex = domain.topology.connectivity(tdim, 0)

all_vertex_indices = []
for idx in cell_idx:
    all_vertex_indices.extend(cell_to_vertex.links(idx))
unique_vertex_indices = np.unique(all_vertex_indices)

CG = fem.functionspace(domain, ("CG", 1))
interior_nodes_array = fem.Function(CG)


total_dofs_local = CG.dofmap.index_map.size_local
all_dofs_local = set(range(total_dofs_local))

set_dofs_tag_1 = set(unique_vertex_indices)
set_dofs_tag_0 = all_dofs_local - set_dofs_tag_1
dofs_tag_0 = np.array(sorted(set_dofs_tag_0), dtype=np.int32)

interior_nodes_array.x.array[dofs_tag_0] = 1.0

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
    "ksp_atol": 1e-12,
    "ksp_rtol": 1e-12,
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    # "ksp_gmres_restart": 300,
    "pc_hypre_ams_cycle_type": 13,
    "pc_hypre_ams_tol": 0.0, # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1, #Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 5,
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

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

#%%
N = len(interior_nodes_array.x.array)
interior_nodes_vector = PETSc.Vec().create(comm=domain.comm)
interior_nodes_vector.setSizes(N)
interior_nodes_vector.setUp()

interior_nodes_vector.setValues(range(N), interior_nodes_array.x.array)
interior_nodes_vector.assemble()

pc.setHYPREAMSSetInteriorNodes(interior_nodes_vector)

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

X = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
A_vis = fem.Function(X)
A_vis.interpolate(uh)

A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_file.write(0.0)


res = A_mat * uh.x.petsc_vec - b
par_print(comm, f"Residual norm: {res.norm()}")

iterations = ksp.getIterationNumber()
par_print(comm, f"Number of iterations: {iterations}")

reason = ksp.getConvergedReason()
par_print(comm, f"Convergence reason: {reason}")


