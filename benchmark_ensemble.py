
from firedrake.petsc import PETSc
import firedrake as fd
from pyop2.mpi import MPI

import numpy as np
from operator import mul
from functools import reduce as fold

from ensemble import NewEnsemble as Ensemble

# unique profile on each mixed function component on each ensemble rank
def unique_function(mesh, rank, W):
    return fd.Function(W).assign(0)

    # Don't actually initialise anything because this muddies the flamegraph
    # def function_profile(x, y, rank, cpt):
    #     return fd.sin(cpt + (rank+1)*fd.pi*x)*fd.cos(cpt + (rank+1)*fd.pi*y)
    # u = fd.Function(W)
    # x, y = fd.SpatialCoordinate(mesh)
    # for cpt, v in enumerate(u.split()):
    #     v.interpolate(function_profile(x, y, rank, cpt))
    # return u


# mixed function space
def mixed_space(mesh, ncpt):
    V = fd.FunctionSpace(mesh, "CG", 1)
    return fold(mul, [V for _ in range(ncpt)])


@PETSc.Log.EventDecorator()
def send_and_recv(ensemble, usend, urecv,
                  new_method, blocking=True):

    ensemble_rank = ensemble.ensemble_comm.rank
    rank0 = 0
    rank1 = 1

    if blocking:
        if new_method:
            send = ensemble.send_new
            recv = ensemble.recv_new
        else:
            send = ensemble.send
            recv = ensemble.recv
    else:
        if new_method:
            return
        send = ensemble.isend
        recv = ensemble.irecv

    ensemble.global_comm.Barrier()
    start = MPI.Wtime()

    # send 0 -> 1
    with PETSc.Log.Event("__main__.send_and_recv.send_section"):
        if ensemble_rank == rank0:
            requests = send(usend, dest=rank1, tag=rank0)

        elif ensemble_rank == rank1:
            requests = recv(urecv, source=rank0, tag=rank0)

        if not blocking:
            MPI.Request.waitall(requests)

    ensemble.global_comm.Barrier()

    # send 1 -> 0
    with PETSc.Log.Event("__main__.send_and_recv.recv_section"):
        if ensemble_rank == rank0:
            requests = recv(urecv, source=rank1, tag=rank1)

        elif ensemble_rank == rank1:
            requests = send(usend, dest=rank0, tag=rank1)

        if not blocking:
            MPI.Request.waitall(requests)

    ensemble.global_comm.Barrier()
    end = MPI.Wtime()

    # Functions are all 0 so don't bother with this check
    # assert fd.errornorm(urecv, usend) < 1e-8

    return end - start


@PETSc.Log.EventDecorator()
def benchmark_send_and_recv(nspatial, nx, ncpts, nwarmups, nrepeats,
                            new_method, blocking=True):

    ensemble = Ensemble(fd.COMM_WORLD, nspatial)

    mesh = fd.UnitSquareMesh(nx, nx, comm=ensemble.comm)

    W = mixed_space(mesh, ncpts)

    ensemble_size = ensemble.ensemble_comm.size

    usend = unique_function(mesh, ensemble_size, W)
    urecv = fd.Function(W).assign(0)

    for _ in range(nwarmups):
        send_and_recv(ensemble, usend, urecv, new_method, blocking)

    durations = np.zeros(nrepeats)
    for i in range(nrepeats):
        durations[i] = send_and_recv(ensemble, usend, urecv, new_method, blocking)

    return (np.average(durations), np.std(durations))


if __name__ == "__main__":

    if fd.COMM_WORLD.size % 2 != 0:
        raise ValueError("Number of MPI ranks must be even")

    nspatial = fd.COMM_WORLD.size//2
    nx = 512
    ncpts = 20
    nwarmups = 0
    nrepeats = 100

    with PETSc.Log.Event("__main__.benchmark"):
        with PETSc.Log.Event("__main__.benchmark_old_method"):
            duration_old = benchmark_send_and_recv(nspatial, nx, ncpts, nwarmups, nrepeats,
                                                   new_method=False, blocking=True)

        with PETSc.Log.Event("__main__.benchmark_new_method"):
            duration_new = benchmark_send_and_recv(nspatial, nx, ncpts, nwarmups, nrepeats,
                                                   new_method=True, blocking=True)

    if fd.COMM_WORLD.rank == 1:
        print(f"ncpts = {ncpts} | ncells = {nx*nx}")
        print(f"Old duration:  average = {duration_old[0]} | std = {duration_old[1]}")
        print(f"New duration:  average = {duration_new[0]} | std = {duration_new[1]}")
