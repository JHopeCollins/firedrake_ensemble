import firedrake as fd
from pyop2.mpi import MPI
import pytest

from operator import mul
from functools import reduce as fold

from ensemble import NewEnsemble


max_ncpts = 3
ncpts = [i for i in range(1, max_ncpts + 1)]

max_root = 2
roots = [i for i in range(0, max_root + 1)] + [None]


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("ncpt", ncpts)
def test_ensemble_allreduce(ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_size = manager.ensemble_comm.size
    ensemble_rank = manager.ensemble_comm.rank

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u_correct = fd.Function(W)
    u = fd.Function(W)
    usum = fd.Function(W)

    for v_correct, v, vsum in zip(u_correct.split(), u.split(), usum.split()):
        v_correct.assign(0)
        v.assign(0)
        vsum.assign(10)

    # initialise local function
    for cpt, v in enumerate(u.split()):
        v.interpolate(func(ensemble_rank, cpt))

    # calculate sum of all ranks
    for cpt, v in enumerate(u_correct.split()):
        for rank in range(ensemble_size):
            v.interpolate(v + func(rank, cpt))

    manager.allreduce(u, usum)

    assert fd.errornorm(u_correct, usum) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("ncpt", ncpts)
def test_ensemble_reduce(root, ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_size = manager.ensemble_comm.size
    ensemble_rank = manager.ensemble_comm.rank

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u_correct = fd.Function(W).assign(0)
    u = fd.Function(W).assign(0)
    usum = fd.Function(W).assign(10)

    for v_correct, v, vsum in zip(u_correct.split(), u.split(), usum.split()):
        v_correct.assign(0)
        v.assign(0)
        vsum.assign(10)

    usum0 = usum.copy(deepcopy=True)

    # initialise local function
    for cpt, v in enumerate(u.split()):
        v.interpolate(func(ensemble_rank, cpt))

    # calculate sum of all ranks
    for cpt, v in enumerate(u_correct.split()):
        for rank in range(ensemble_size):
            v.interpolate(v + func(rank, cpt))

    # check default root=0 works
    if root is None:
        manager.reduce(u, usum)
        root = 0
    else:
        manager.reduce(u, usum, root=root)

    # test
    if ensemble_rank == root:
        assert fd.errornorm(u_correct, usum) < 1e-4
    else:
        assert fd.errornorm(usum0, usum) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("ncpt", ncpts)
def test_ensemble_ireduce(root, ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_size = manager.ensemble_comm.size
    ensemble_rank = manager.ensemble_comm.rank

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u_correct = fd.Function(W).assign(0)
    u = fd.Function(W).assign(0)
    usum = fd.Function(W).assign(10)

    for v_correct, v, vsum in zip(u_correct.split(), u.split(), usum.split()):
        v_correct.assign(0)
        v.assign(0)
        vsum.assign(10)

    usum0 = usum.copy(deepcopy=True)

    # initialise local function
    for cpt, v in enumerate(u.split()):
        v.interpolate(func(ensemble_rank, cpt))

    # calculate sum of all ranks
    for cpt, v in enumerate(u_correct.split()):
        for rank in range(ensemble_size):
            v.interpolate(v + func(rank, cpt))

    # check default root=0 works
    if root is None:
        requests = manager.ireduce(u, usum)
        root = 0
    else:
        requests = manager.ireduce(u, usum, root=root)

    MPI.Request.Waitall(requests)

    # test
    if ensemble_rank == root:
        assert fd.errornorm(u_correct, usum) < 1e-4
    else:
        assert fd.errornorm(usum0, usum) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("ncpt", ncpts)
def test_ensemble_bcast(root, ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_rank = manager.ensemble_comm.rank

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u_correct = fd.Function(W)
    u = fd.Function(W)

    # initialise local function
    for cpt, v in enumerate(u.split()):
        v.interpolate(func(ensemble_rank, cpt))

    if root is None:
        manager.bcast(u)
        root = 0
    else:
        manager.bcast(u, root=root)

    # broadcasted function
    for cpt, v in enumerate(u_correct.split()):
        v.interpolate(func(root, cpt))

    assert fd.errornorm(u_correct, u) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("ncpt", ncpts)
def test_ensemble_ibcast(root, ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_rank = manager.ensemble_comm.rank

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u_correct = fd.Function(W)
    u = fd.Function(W)

    # initialise local function
    for cpt, v in enumerate(u.split()):
        v.interpolate(func(ensemble_rank, cpt))

    if root is None:
        requests = manager.ibcast(u)
        root = 0
    else:
        requests = manager.ibcast(u, root=root)

    MPI.Request.Waitall(requests)

    # broadcasted function
    for cpt, v in enumerate(u_correct.split()):
        v.interpolate(func(root, cpt))

    assert fd.errornorm(u_correct, u) < 1e-4


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

    # same size but different function spaces
    mesh4 = fd.UnitSquareMesh(4, 2, comm=manager.comm)
    mesh5 = fd.UnitSquareMesh(2, 4, comm=manager.comm)

    V4 = fd.FunctionSpace(mesh4, "DG", 0)
    V5 = fd.FunctionSpace(mesh5, "DG", 0)

    f4 = fd.Function(V4)
    f5 = fd.Function(V5)

    with pytest.raises(ValueError):
        manager.allreduce(f4, f5)


@pytest.mark.parallel(nprocs=2)
def test_comm_manager_reduce():
    manager = NewEnsemble(fd.COMM_WORLD, 1)

    mesh = fd.UnitSquareMesh(1, 1, comm=manager.global_comm)

    mesh2 = fd.UnitSquareMesh(2, 2, comm=manager.ensemble_comm)

    V = fd.FunctionSpace(mesh, "CG", 1)
    V2 = fd.FunctionSpace(mesh2, "CG", 1)

    f = fd.Function(V)
    f2 = fd.Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        manager.reduce(f, f2)

    f3 = fd.Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        manager.reduce(f3, f2)

    # same function communicator but different function spaces
    V3 = fd.FunctionSpace(mesh, "DG", 0)
    g = fd.Function(V3)
    with pytest.raises(ValueError):
        manager.reduce(f, g)

    # same size but different function spaces
    mesh4 = fd.UnitSquareMesh(4, 2, comm=manager.comm)
    mesh5 = fd.UnitSquareMesh(2, 4, comm=manager.comm)

    V4 = fd.FunctionSpace(mesh4, "DG", 0)
    V5 = fd.FunctionSpace(mesh5, "DG", 0)

    f4 = fd.Function(V4)
    f5 = fd.Function(V5)

    with pytest.raises(ValueError):
        manager.reduce(f4, f5)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("ncpt", ncpts)
def test_blocking_send_recv(ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_rank = manager.ensemble_comm.rank
    ensemble_size = manager.ensemble_comm.size

    rank0 = 0
    rank1 = 1

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u = fd.Function(W).assign(0)
    u_expect = fd.Function(W)

    # function to send
    for cpt, v in enumerate(u_expect.split()):
        v.interpolate(func(ensemble_size, cpt))

    if ensemble_rank == rank0:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        manager.send(u_expect, dest=rank1, tag=rank0)
        manager.recv(u, source=rank1, tag=rank1)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    elif ensemble_rank == rank1:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        manager.recv(u, source=rank0, tag=rank0)
        manager.send(u_expect, dest=rank0, tag=rank1)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    else:
        assert fd.norm(u) < 1e-8


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("ncpt", ncpts)
def test_nonblocking_send_recv(ncpt):
    manager = NewEnsemble(fd.COMM_WORLD, 2)
    ensemble_rank = manager.ensemble_comm.rank
    ensemble_size = manager.ensemble_comm.size

    rank0 = 0
    rank1 = 1

    mesh = fd.UnitSquareMesh(10, 10, comm=manager.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    u = fd.Function(W).assign(0)
    u_expect = fd.Function(W)

    # function to send
    for cpt, v in enumerate(u_expect.split()):
        v.interpolate(func(ensemble_size, cpt))

    if ensemble_rank == rank0:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        send_requests = manager.isend(u_expect, dest=rank1, tag=rank0)
        recv_requests = manager.irecv(u, source=rank1, tag=rank1)
        MPI.Request.waitall(send_requests)
        MPI.Request.waitall(recv_requests)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    elif ensemble_rank == rank1:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        send_requests = manager.isend(u_expect, dest=rank0, tag=rank1)
        recv_requests = manager.irecv(u, source=rank0, tag=rank0)
        MPI.Request.waitall(send_requests)
        MPI.Request.waitall(recv_requests)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    else:
        assert fd.norm(u) < 1e-8
