import json
import sys

import numpy as np
import petsc4py
import scipy.sparse as sp
from basix.ufl import element
from dolfinx import fem
from dolfinx.common import Timer
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    locate_dofs_topological,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix, discrete_gradient, interpolation_matrix
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, as_vector, curl, dx, inner

from utils import L2_norm, boundary_marker, par_print

petsc4py.init(sys.argv)
PETSc.Log.begin()

comm = MPI.COMM_WORLD
degree = 1

n = 10
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

beta_non_conductive_value = 0.0

beta.interpolate(
    lambda x: np.where(
        (np.abs(x[0] - 0.5) < beta_loc + eps) &
        (np.abs(x[1] - 0.5) < beta_loc + eps) &
        (np.abs(x[2] - 0.5) < beta_loc + eps),
        1.0, beta_non_conductive_value
    )
)

num_cells = domain.topology.index_map(domain.topology.dim).size_local
cell_indices = np.arange(num_cells, dtype=np.int32)

beta_cell_values = beta.x.array[:num_cells]

has_zero_beta_region = domain.comm.allreduce(
    bool(np.any(beta_cell_values == 0)), op=MPI.LOR
)

beta_tags = (beta_cell_values > 0.5).astype(np.int32)


ct = meshtags(domain, domain.topology.dim, cell_indices, beta_tags)
# with XDMFFile(domain.comm, "mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_meshtags(ct, domain.geometry)

tdim = domain.topology.dim
fdim = tdim - 1

facets = locate_entities_boundary(
    domain, dim=(domain.topology.dim - 1), marker=boundary_marker
)

def set_interior_nodes(domain, ct, facets, fdim, degree, pc):
    V_interior = fem.functionspace(domain, ("CG", degree))
    interior_nodes_array = fem.Function(V_interior)

    interior_nodes_array.x.array[:] = 1.0
    interior_nodes_array.x.scatter_forward()

    dofmap = V_interior.dofmap
    num_dofs_per_cell = dofmap.dof_layout.num_dofs
    cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

    tagged_cells = ct.find(1)  # Conductive tags

    tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
    unique_dofs = np.unique(tagged_cell_dofs)

    interior_nodes_array.x.array[unique_dofs] = 0.0

    # Exclude the outer domain boundary from the interior node set
    boundary_dofs = locate_dofs_topological(V=V_interior, entity_dim=fdim, entities=facets)
    interior_nodes_array.x.array[boundary_dofs] = 0.0

    interior_nodes_array.x.scatter_forward()

    pc.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)


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
par_print(comm, "Assembling system")

with Timer("Setup: Assembling system") as a_mat_timer:
    A_mat = assemble_matrix(a, bcs=[bc])
    A_mat.assemble()
assembly_time = a_mat_timer.elapsed().total_seconds()

par_print(comm, "Assembling RHS")
with Timer("Setup: Assembling RHS") as rhs_timer:
    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
assembly_time_rhs = rhs_timer.elapsed().total_seconds()

uh = fem.Function(A_space)

ams_opts = {
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_type": "cg",
    "ksp_monitor_true_residual": None,
    "pc_hypre_ams_cycle_type": 1,
    "pc_hypre_ams_tol": 0.0, # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1, #Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 100,
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

par_print(comm, "Setting up preconditioner")
with Timer("Setup: Preconditioner") as timer_pc:
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

    opts = PETSc.Options()

    if has_zero_beta_region:
        opts["log_view"] = "ascii:log.txt"
    else:
        opts["log_view"] = "ascii:log_no_zero_beta.txt"


    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in ams_opts.items():
        opts[option] = value
    opts.prefixPop()

    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")

    G = discrete_gradient(V_CG, A_space)

    cols, vals = G.getRow(100)
    print("entries:", len(vals), " true nonzeros:", int(np.count_nonzero(np.abs(vals) > 1e-12)))

    G.assemble()
    ai, aj, av = G.getValuesCSR()
    M = sp.csr_matrix((av, aj, ai), shape=G.getSize())
    M.data[np.abs(M.data) < 1e-12] = 0.0
    M.eliminate_zeros()                       # now exactly 2/row
    print("avg nnz/row:", M.nnz / M.shape[0])

    G_clean = PETSc.Mat().createAIJ(size=M.shape,
                                    csr=(M.indptr, M.indices, M.data),
                                    comm=G.comm)
    G_clean.assemble()
    pc.setHYPREDiscreteGradient(G_clean)

    if has_zero_beta_region:
        par_print(comm, "Setting interior nodes for AMS preconditioner due to zero beta region.")
        set_interior_nodes(domain, ct, facets, fdim, degree, pc)


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
        Pi = interpolation_matrix(Vec_CG, A_space)
        Pi.assemble()

        # Attach discrete gradient to preconditioner
        pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

    ksp.setFromOptions()
    pc.setFromOptions()

    st_setup = PETSc.Log.Stage("PCSetUp")
    st_setup.push()
    ksp.setUp()
    pc.setUp()
    st_setup.pop()

pc_assembly_time = timer_pc.elapsed().total_seconds()

info = G.getInfo()
par_print(comm, f"G rows: {G.getSize()[0]}, nnz: {info['nz_used']}, avg/row: {info['nz_used']/G.getSize()[0]}")

with Timer("Solve") as timer_solve:
    st_solve = PETSc.Log.Stage("KSPSolve")
    st_solve.push()
    ksp.solve(b, uh.x.petsc_vec)
    st_solve.pop()
solve_time = timer_solve.elapsed().total_seconds()

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
par_print(comm, f"L2 norm of curl uh: {L2_norm(curl(uh))}")


timings = {
    "zero_beta_region": has_zero_beta_region,
    "assemble_matrix": assembly_time,
    "assemble_rhs": assembly_time_rhs,
    "assemble_preconditioner": pc_assembly_time,
    "solve": solve_time,
    "iterations": iterations,
}

if has_zero_beta_region:
    with open("timings_zero_beta.json", "w") as f:
        json.dump(timings, f, indent=4)
else:
    with open("timings.json", "w") as f:
        json.dump(timings, f, indent=4)

