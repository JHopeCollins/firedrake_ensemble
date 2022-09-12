import firedrake as fd
from pyop2.mpi import MPI
import pytest

from operator import mul
from functools import reduce as fold

from ensemble import NewEnsemble


max_ncpts = 2
ncpts = [i for i in range(1, max_ncpts + 1)]

min_root = 1
max_root = 1
roots = [None] + [i for i in range(min_root, max_root + 1)]

# is_blocking = [True]
is_blocking = [True, False]


# unique profile on each mixed function component on each ensemble rank
def function_profile(x, y, rank, cpt):
    return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)


@pytest.fixture
def ensemble():
    if fd.COMM_WORLD.size == 1:
        return

    return NewEnsemble(fd.COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    if fd.COMM_WORLD.size == 1:
        return

    return fd.UnitSquareMesh(10, 10, comm=ensemble.comm)


# mixed function space
@pytest.fixture(params=ncpts)
def W(request, mesh):
    if fd.COMM_WORLD.size == 1:
        return

    V = fd.FunctionSpace(mesh, "CG", 1)
    return fold(mul, [V for _ in range(request.param)])


# initialise unique function on each rank
@pytest.fixture
def urank(ensemble, mesh, W):
    if fd.COMM_WORLD.size == 1:
        return

    rank = ensemble.ensemble_comm.rank
    u = fd.Function(W)
    x, y = fd.SpatialCoordinate(mesh)
    for cpt, v in enumerate(u.split()):
        v.interpolate(function_profile(x, y, rank, cpt))
    return u


# sum of urank across all ranks
@pytest.fixture
def urank_sum(ensemble, mesh, W):
    if fd.COMM_WORLD.size == 1:
        return

    u = fd.Function(W).assign(0)
    x, y = fd.SpatialCoordinate(mesh)
    for cpt, v in enumerate(u.split()):
        for rank in range(ensemble.ensemble_comm.size):
            v.interpolate(v + function_profile(x, y, rank, cpt))
    return u


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("blocking", is_blocking)
def test_ensemble_allreduce(ensemble, mesh, W, urank, urank_sum,
                            blocking):

    x, y = fd.SpatialCoordinate(mesh)

    u_reduce = fd.Function(W).assign(0)

    if blocking:
        ensemble.allreduce(urank, u_reduce)
    else:
        requests = ensemble.iallreduce(urank, u_reduce)
        MPI.Request.Waitall(requests)

    assert fd.errornorm(urank_sum, u_reduce) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("blocking", is_blocking)
def test_ensemble_reduce(ensemble, mesh, W, urank, urank_sum,
                         root, blocking):

    x, y = fd.SpatialCoordinate(mesh)

    u_reduce = fd.Function(W).assign(10)

    if blocking:
        reduction = ensemble.reduce
    else:
        reduction = ensemble.ireduce

    # check default root=0 works
    if root is None:
        requests = reduction(urank, u_reduce)
        root = 0
    else:
        requests = reduction(urank, u_reduce, root=root)

    if not blocking:
        MPI.Request.Waitall(requests)

    # only u_reduce on rank root should be modified
    if ensemble.ensemble_comm.rank == root:
        assert fd.errornorm(urank_sum, u_reduce) < 1e-4
    else:
        assert fd.errornorm(fd.Function(W).assign(10), u_reduce) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("root", roots)
@pytest.mark.parametrize("blocking", is_blocking)
def test_ensemble_bcast(ensemble, mesh, W, urank,
                        root, blocking):

    if blocking:
        bcast = ensemble.bcast
    else:
        bcast = ensemble.ibcast

    # check default root=0 works
    if root is None:
        requests = bcast(urank)
        root = 0
    else:
        requests = bcast(urank, root=root)

    if not blocking:
        MPI.Request.Waitall(requests)

    # broadcasted function
    u_correct = fd.Function(W)

    x, y = fd.SpatialCoordinate(mesh)

    for cpt, v in enumerate(u_correct.split()):
        v.interpolate(function_profile(x, y, root, cpt))

    assert fd.errornorm(u_correct, urank) < 1e-4


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("ncpt", ncpts)
@pytest.mark.parametrize("blocking", is_blocking)
def test_send_and_recv(ensemble, ncpt, blocking):
    ensemble_rank = ensemble.ensemble_comm.rank
    ensemble_size = ensemble.ensemble_comm.size

    rank0 = 0
    rank1 = 1

    mesh = fd.UnitSquareMesh(10, 10, comm=ensemble.comm)

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

    if blocking:
        send = ensemble.send
        recv = ensemble.recv
    else:
        send = ensemble.isend
        recv = ensemble.irecv

    if ensemble_rank == rank0:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        send_requests = send(u_expect, dest=rank1, tag=rank0)
        recv_requests = recv(u, source=rank1, tag=rank1)

        if not blocking:
            MPI.Request.waitall(send_requests)
            MPI.Request.waitall(recv_requests)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    elif ensemble_rank == rank1:
        # before receiving, u should be 0
        assert fd.norm(u) < 1e-8

        recv_requests = recv(u, source=rank0, tag=rank0)
        send_requests = send(u_expect, dest=rank0, tag=rank1)

        if not blocking:
            MPI.Request.waitall(send_requests)
            MPI.Request.waitall(recv_requests)

        # after receiving, u should be like u_expect
        assert fd.errornorm(u, u_expect) < 1e-8

    else:
        assert fd.norm(u) < 1e-8


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("ncpt", ncpts)
@pytest.mark.parametrize("blocking", is_blocking)
def test_sendrecv(ensemble, ncpt, blocking):
    ensemble_rank = ensemble.ensemble_comm.rank
    ensemble_size = ensemble.ensemble_comm.size

    src_rank = (ensemble_rank - 1) % ensemble_size
    dst_rank = (ensemble_rank + 1) % ensemble_size

    mesh = fd.UnitSquareMesh(10, 10, comm=ensemble.comm)

    x, y = fd.SpatialCoordinate(mesh)

    # unique function for each rank / component index pair
    def func(rank, cpt=0):
        return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)

    # mixed space of dimension ncpt
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = fold(mul, [V for _ in range(ncpt)])

    usend = fd.Function(W)
    urecv = fd.Function(W).assign(0)
    u_expect = fd.Function(W)

    for cpt, v in enumerate(usend.split()):
        v.interpolate(func(ensemble_rank, cpt))

    for cpt, v in enumerate(u_expect.split()):
        v.interpolate(func(src_rank, cpt))

    if blocking:
        sendrecv = ensemble.sendrecv
    else:
        sendrecv = ensemble.isendrecv

    requests = sendrecv(usend, dst_rank, sendtag=ensemble_rank,
                        frecv=urecv, source=src_rank, recvtag=src_rank)

    if not blocking:
        MPI.Request.Waitall(requests)

    # after receiving, u should be like u_expect
    assert fd.errornorm(urecv, u_expect) < 1e-8


@pytest.mark.parallel(nprocs=6)
def test_ensemble_solvers(ensemble):
    # this test uses linearity of the equation to solve two problems
    # with different RHS on different subcommunicators,
    # and compare the reduction with a problem solved with the sum
    # of the two RHS

    mesh = fd.UnitSquareMesh(10, 10, comm=ensemble.comm)

    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    f_combined = fd.Function(V)
    f_separate = fd.Function(V)

    f_combined.interpolate(fd.sin(fd.pi*x)*fd.cos(fd.pi*y) + fd.sin(2*fd.pi*x)*fd.cos(2*fd.pi*y) + fd.sin(3*fd.pi*x)*fd.cos(3*fd.pi*y))
    q = fd.Constant(ensemble.ensemble_comm.rank + 1)
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
    ensemble.allreduce(u_separate, usum)

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
    ensemble = NewEnsemble(fd.COMM_WORLD, 1)

    mesh = fd.UnitSquareMesh(1, 1, comm=ensemble.global_comm)

    mesh2 = fd.UnitSquareMesh(2, 2, comm=ensemble.ensemble_comm)

    V = fd.FunctionSpace(mesh, "CG", 1)
    V2 = fd.FunctionSpace(mesh2, "CG", 1)

    f = fd.Function(V)
    f2 = fd.Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        ensemble.allreduce(f, f2)

    f3 = fd.Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        ensemble.allreduce(f3, f2)

    # same function communicator but different function spaces
    V3 = fd.FunctionSpace(mesh, "DG", 0)
    g = fd.Function(V3)
    with pytest.raises(ValueError):
        ensemble.allreduce(f, g)

    # same size but different function spaces
    mesh4 = fd.UnitSquareMesh(4, 2, comm=ensemble.comm)
    mesh5 = fd.UnitSquareMesh(2, 4, comm=ensemble.comm)

    V4 = fd.FunctionSpace(mesh4, "DG", 0)
    V5 = fd.FunctionSpace(mesh5, "DG", 0)

    f4 = fd.Function(V4)
    f5 = fd.Function(V5)

    with pytest.raises(ValueError):
        ensemble.allreduce(f4, f5)


@pytest.mark.parallel(nprocs=2)
def test_comm_manager_reduce():
    ensemble = NewEnsemble(fd.COMM_WORLD, 1)

    mesh = fd.UnitSquareMesh(1, 1, comm=ensemble.global_comm)

    mesh2 = fd.UnitSquareMesh(2, 2, comm=ensemble.ensemble_comm)

    V = fd.FunctionSpace(mesh, "CG", 1)
    V2 = fd.FunctionSpace(mesh2, "CG", 1)

    f = fd.Function(V)
    f2 = fd.Function(V2)

    # different function communicators
    with pytest.raises(ValueError):
        ensemble.reduce(f, f2)

    f3 = fd.Function(V2)

    # same function communicators, but doesn't match ensembles spatial communicator
    with pytest.raises(ValueError):
        ensemble.reduce(f3, f2)

    # same function communicator but different function spaces
    V3 = fd.FunctionSpace(mesh, "DG", 0)
    g = fd.Function(V3)
    with pytest.raises(ValueError):
        ensemble.reduce(f, g)

    # same size but different function spaces
    mesh4 = fd.UnitSquareMesh(4, 2, comm=ensemble.comm)
    mesh5 = fd.UnitSquareMesh(2, 4, comm=ensemble.comm)

    V4 = fd.FunctionSpace(mesh4, "DG", 0)
    V5 = fd.FunctionSpace(mesh5, "DG", 0)

    f4 = fd.Function(V4)
    f5 = fd.Function(V5)

    with pytest.raises(ValueError):
        ensemble.reduce(f4, f5)
