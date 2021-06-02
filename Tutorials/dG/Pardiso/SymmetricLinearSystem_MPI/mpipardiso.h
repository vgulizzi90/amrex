#ifndef MPIPARDISO_H_
#define MPIPARDISO_H_

#include <mpi.h>

/*
 * Internal API
 */
void pardiso_mpi_freeall_(void *handle[64]);
void pardiso_mpi_freefactor_(void *handle[64]);
void pardiso_mpi_analyze_(void *handle[64], int *actual_solveMPP);
void pardiso_mpi_notify_worker_finished_(void* handle[64], int *error);

/*
 * User-level API
 */
void pardiso_mpi_init_c_(void *handle[64], MPI_Comm comm, int *error);
void pardiso_mpi_init_ftn_(void *handle[64], MPI_Fint *ftn_comm, int *error);
void pardiso_mpi_worker_(void *handle[64], int *error);
void pardiso_mpi_finalize_(void *handle[64], int *error);

#endif /* MPIPARDISO_H_ */
