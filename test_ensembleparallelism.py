import firedrake as fd
from pyop2.mpi import MPI
import pytest
import time

from ensemble import NewEnsemble


@pytest.mark.parallel(nprocs=6)
def test_ensemble_allreduce():
    manager = NewEnsemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    u_correct = fd.Function(V)
    u = fd.Function(V)
    usum = fd.Function(V)

    u_correct.interpolate(fd.sin(fd.pi*x)*fd.cos(fd.pi*y) + fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y) + fd.sin(3*fd.pi*x)*fd.cos(3*fd.pi*y))
    q = fd.Constant(manager.ensemble_comm.rank + 1)
    u.interpolate(fd.sin(q*fd.pi*x)*fd.cos(q*fd.pi*y))
    usum.assign(10)             # Check that the output gets zeroed.
    manager.allreduce(u, usum)

    assert fd.errornorm(u_correct, usum) < 1e-4

@pytest.mark.parallel(nprocs=6)
def test_ensemble_solvers():
    # this test uses linearity of the equation to solve two problems
    # with different RHS on different subcommunicators,
    # and compare the reduction with a problem solved with the sum
    # of the two RHS
    manager = NewEnsemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    f_combined = fd.Function(V)
    f_separate = fd.Function(V)

    f_combined.interpolate(fd.sin(fd.pi*x)*fd.cos(fd.pi*y) + fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y) + fd.sin(3*fd.pi*x)*fd.cos(3*fd.pi*y))
    q = fd.Constant(manager.ensemble_comm.rank + 1)
    f_separate.interpolate(fd.sin(q*fd.pi*x)*fd.cos(q*fd.pi*y))

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    a = (fd.inner(u, v) + fd.inner(fd.grad(u), fd.grad(v)))*fd.dx
    Lcombined = fd.inner(f_combined, v)*fd.dx
    Lseparate = fd.inner(f_separate, v)*fd.dx

    u_combined = fd.Function(V)
    u_separate = fd.Function(V)
    usum = fd.Function(V)

    params = {'ksp_type': 'preonly',
              'pc_type': 'redundant',
              "redundant_pc_type": "lu",
              "redundant_pc_factor_mat_solver_type": "mumps",
              "redundant_mat_mumps_icntl_14": 200}

    combinedProblem = fd.LinearVariationalProblem(a, Lcombined, u_combined)
    combinedSolver = fd.LinearVariationalSolver(combinedProblem,
                                                solver_parameters=params)

    separateProblem = fd.LinearVariationalProblem(a, Lseparate, u_separate)
    separateSolver = fd.LinearVariationalSolver(separateProblem,
                                                solver_parameters=params)

    combinedSolver.solve()
    separateSolver.solve()
    manager.allreduce(u_separate, usum)

    assert fd.errornorm(u_combined, usum) < 1e-4


def test_comm_manager():
    with pytest.raises(ValueError):
        NewEnsemble(fd.COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=3)
def test_comm_manager_parallel():
    with pytest.raises(ValueError):
        NewEnsemble(fd.COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=2)
def test_comm_manager_allreduce():
    manager = NewEnsemble(fd.COMM_WORLD, 1)

    mesh = fd.UnitSquareMesh(1, 1, comm=manager.global_comm)

    mesh2 = fd.UnitSquareMesh(2, 2, comm=manager.ensemble_comm)

    V = fd.FunctionSpace(mesh, "CG", 1)
    V2 = fd.FunctionSpace(mesh2, "CG", 1)

    f = fd.Function(V)
    f2 = fd.Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        manager.allreduce(f, f2)

    f3 = fd.Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        manager.allreduce(f3, f2)

    # same function communicator but different function spaces
    V3 = fd.FunctionSpace(mesh, "DG", 0)
    g = fd.Function(V3)
    with pytest.raises(ValueError):
        manager.allreduce(f, g)


@pytest.mark.parallel(nprocs=8)
def test_blocking_send_recv():
    nprocs_spatial = 2
    manager = NewEnsemble(fd.COMM_WORLD, nprocs_spatial)

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(V)
    x, y = fd.SpatialCoordinate(mesh)
    u_correct = fd.Function(V).interpolate(fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y))

    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        manager.send(u_correct, dest=1, tag=0)
        manager.recv(u, source=1, tag=1)

        # after receiving, u should be like u_correct
        assert fd.errornorm(u, u_correct) < 1e-8

    if ensemble_procno == 1:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8
        manager.recv(u, source=0, tag=0)
        manager.send(u, dest=0, tag=1)
        # after receiving, u should be like u_correct
        assert fd.errornorm(u, u_correct) < 1e-8

    if ensemble_procno != 0 and ensemble_procno != 1:
        # without receiving, u should be 0
        assert fd.norm(u) < 1e-8


@pytest.mark.parallel(nprocs=8)
def test_nonblocking_send_recv_mixed():
    nprocs_spatial = 2
    manager = NewEnsemble(fd.COMM_WORLD, nprocs_spatial)

    # Big mesh so we blow through the MPI eager message limit.
    mesh = fd.UnitSquareMesh(100, 100, comm=manager.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V*Q
    w = fd.Function(W)
    x, y = fd.SpatialCoordinate(mesh)
    u, v = w.split()
    u_expr = fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y)
    v_expr = x + y

    w_expect = fd.Function(W)
    u_expect, v_expect = w_expect.split()
    u_expect.interpolate(u_expr)
    v_expect.interpolate(v_expr)
    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        requests = manager.isend(w_expect, dest=1, tag=0)
        MPI.Request.waitall(requests)
    elif ensemble_procno == 1:
        # before receiving, u should be 0
        assert fd.norm(w) < 1e-8
        requests = manager.irecv(w, source=0, tag=0)
        # Bad check to see if the buffer has gone away.
        time.sleep(2)
        MPI.Request.waitall(requests)
        assert fd.errornorm(u, u_expect) < 1e-8
        assert fd.errornorm(v, v_expect) < 1e-8
    else:
        assert fd.norm(w) < 1e-8


@pytest.mark.parallel(nprocs=8)
def test_nonblocking_send_recv():
    nprocs_spatial = 2
    manager = NewEnsemble(fd.COMM_WORLD, nprocs_spatial)

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.Function(V)
    x, y = fd.SpatialCoordinate(mesh)
    u_expr = fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y)
    u_expect = fd.interpolate(u_expr, V)
    ensemble_procno = manager.ensemble_comm.rank

    if ensemble_procno == 0:
        requests = manager.isend(u_expect, dest=1, tag=0)
        MPI.Request.waitall(requests)
    elif ensemble_procno == 1:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8
        requests = manager.irecv(u, source=0, tag=0)
        MPI.Request.waitall(requests)
        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8
    else:
        assert fd.norm(u) < 1e-8
