import sys
import numpy as np
from dolfinx.fem import (
    assemble_scalar,
    form,
    Expression,
    Function,
    functionspace,
)
from mpi4py import MPI
from ufl import dx, inner
from ufl.core.expr import Expr
import ufl


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(
        MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM)
    )


def monitor(ksp, its, rnorm):
    iteration_count = []
    residual_norm = []
    iteration_count.append(its)
    residual_norm.append(rnorm)
    print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))


def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)
    ]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])


def error_L2(uh, u_ex, degree_raise=4):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(inner(e_W, e_W) * dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


