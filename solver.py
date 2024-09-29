import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner, curl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, sin, pi, curl
from dolfinx.fem import (assemble_scalar, form, Function)
from matplotlib import pyplot as plt
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import petsc, Expression, locate_dofs_topological
from dolfinx.io import VTXWriter
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from basix.ufl import element
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from basix.ufl import element
from dolfinx import fem, io, la, default_scalar_type
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    cos,
    inner,
    cross,
)
import ufl
import sys

def monitor(ksp, its, rnorm):
        iteration_count.append(its)
        residual_norm.append(rnorm)
        print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))

comm = MPI.COMM_WORLD
def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

# from dolfinx.mesh import create_box

# nprocs = 2
# ndofs = 500000
# ntot = nprocs * ndofs

# n = round((ntot /4)**(1/3) -1)

# domain = create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 1, 1])], [n, n, n]) 


case = 3
degree = 1

n = 10

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
facet_dim = gdim - 1 #Topological dimension 

alpha_in = 1 #Magnetic Permeability
beta_in = 0 #Conductivity -> This is set to 0 for Magnetostatic problems

alpha = fem.Constant(domain, default_scalar_type(alpha_in))
beta = fem.Constant(domain, default_scalar_type(beta_in))

gdim = domain.geometry.dim
top_dim = gdim - 1 #Topological dimension 

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

total_dofs = A_space.dofmap.index_map.size_global * A_space.dofmap.index_map_bs
print(total_dofs)


A  = ufl.TrialFunction(A_space)
v = ufl.TestFunction(A_space)

facets = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                    marker=lambda x: np.isclose(x[0], 0.0)|np.isclose(x[0], 1.0)|np.isclose(x[1], 0.0)|np.isclose(x[1], 1.0)|
                                        np.isclose(x[2], 0.0)|np.isclose(x[2], 1.0))
dofs = fem.locate_dofs_topological(V=A_space, entity_dim=top_dim, entities=facets)

x = SpatialCoordinate(domain)
u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
f = curl(alpha * curl(u_e)) + beta * u_e

a = form(inner(alpha * curl(A), curl(v)) * dx + inner(beta * A, v) * dx)
L = form(inner(f,v) * dx)

u_bc_expr = Expression(u_e, A_space.element.interpolation_points())
u_bc = Function(A_space)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, dofs)

# Solver steps

A_mat = assemble_matrix(a, bcs = [bc])
A_mat.assemble()

sol = petsc.assemble_vector(L)
petsc.apply_lifting(sol, [a], bcs=[[bc]])
sol.ghostUpdate(addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(sol, [bc])

uh = fem.Function(A_space)

iteration_count = []
residual_norm = []

lu_opts = {"ksp_atol": 1e-10, 
           "ksp_rtol": 1e-10,
           "ksp_type": "gmres"}

ams_opts = {"pc_hypre_ams_cycle_type": 7,
                    "pc_hypre_ams_tol": 0,
                    "pc_hypre_ams_max_iter": 1,
                    "pc_hypre_ams_amg_beta_theta": 0.25,
                    "pc_hypre_ams_print_level": 1,
                    "ksp_atol": 1e-10, "ksp_rtol": 1e-10,
                    "ksp_type": "gmres",
                    "pc_hypre_ams_amg_alpha_options": "10,1,3",
                    "pc_hypre_ams_amg_beta_options": "10,1,3",
                    "pc_hypre_ams_print_level": 0
                    }

# Iterative Solver with Direct LU preconditioner

if case == 1:
    print("Case 1: Direct Solver")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setType('preonly')
    
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
   
    ksp.setUp()
    pc.setUp()

# Iterative Solver with AMS Preconditioner

if case == 2:
    print("Case 2: Iterative Solver with AMS Preconditioner")

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

    # Build discrete gradient
    V_CG = fem.functionspace(domain, ("CG", degree))._cpp_object
    G = discrete_gradient(V_CG, A_space._cpp_object)
    G.assemble()
    pc.setHYPREDiscreteGradient(G)

    if degree == 1:
        cvec_0 = Function(A_space)
        cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_1 = Function(A_space)
        cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.ones_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_2 = Function(A_space)
        cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.ones_like(x[0]))))
        pc.setHYPRESetEdgeConstantVectors(cvec_0.vector,
                                            cvec_1.vector,
                                            cvec_2.vector)
    else:
        Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
        Pi = interpolation_matrix(Vec_CG._cpp_object, A_space._cpp_object)
        Pi.assemble()

        # Attach discrete gradient to preconditioner
        pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

    pc.setHYPRESetBetaPoissonMatrix(None)

    ksp.setFromOptions()
    ksp.setUp()
    pc.setUp()
    # pc.setHYPRESetBetaPoissonMatrix(None)

if case == 3:
    
    print("Case 3: Iterative Solver with Direct Preconditioner")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
    
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in lu_opts.items():
        opts[option] = value
    opts.prefixPop()

    # iterations = ksp.getIterationNumber()

    pc = ksp.getPC()
    pc.setType("gamg")
    # pc.setFactorSolverType("mumps")
    
    ksp.setFromOptions()
    ksp.setUp()
    pc.setUp()


# ksp.setMonitor(monitor)
ksp.solve(sol, uh.vector)
res = A_mat * uh.vector - sol
# print("Residual norm: ", res.norm())

e = L2_norm(curl(uh - u_e))

par_print(comm, e)

# print("total dofs: ", total_dofs)
# print(f"||u - u_e||_L^2(Omega) = {e})")


