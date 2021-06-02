/* -------------------------------------------------------------------- */
/*      This program can be downloaded from the following site:         */
/*      http://www.pardiso-project.org                                  */
/*                                                                      */
/*  (C) Olaf Schenk, Departments of Mathematics and Computer Science    */
/*      University of Basel, Switzerland.                               */
/*      Email: olaf.schenk@unibas.ch                                    */
/*                                                                      */
/*      Example MPI program to show the use of the "PARDISO" routine    */
/*      on a scalable symmetric linear systems (Laplace equation)       */
/*      The program can be started with "mpirun -np p ./laplace n"      */
/*      in which "n" represents the discretization in one spatial       */
/*      direction and "p" is the numbers of MPI processes               */
/*      e.g. mpirun -np 2 ./laplace 100                                 */ 
/* -------------------------------------------------------------------- */

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include "laplace.h"
#include <assert.h>


#if defined(_OPENMP)
#include <omp.h>
#endif

#include <mpi.h>

#include "mpipardiso.h"

#define MAX_CPU_MODEL_NAME 160

#define MTX_FILE_FMT "Loading matrix file \"%s\"\n"
#define FMT_PARAM_INT "%-20s = %d\n"
#define FMT_PARAM_DBL "%-20s = %g\n"
#define FMT_PARAM_STRING "%-20s = %s\n"
#define FMT_GROUP_HOST "[process %d]\n"
#define FMT_GROUP "[%s]\n"
#define FMT_KEYVAL_STR_INT "%-20s = %d\n"
#define FMT_KEYVAL_STR_STR "%-20s = %s\n"

#define TIME_FMT "%-25s = %10.3f\n"
#define FMT_PARDISO_IPARAM "%-10s = %10d              # %s\n"
#define FMT_PARDISO_DPARAM "%-10s = %10g              # %s\n"
#define FMT_NORM "%-25s = %25.15e\n"

#define IPARAM(I) iparam[(I)-1]
#define DPARAM(I) dparam[(I)-1]

typedef struct
{
    char processorName[MPI_MAX_PROCESSOR_NAME];
    char cpu_model_name[MAX_CPU_MODEL_NAME];
    int omp_num_threads;
} host_info_message_t;


/* PARDISO prototype. */
extern void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
extern void pardiso_chkvec   (int *, int *, double *, int *);
extern void pardiso_printstats(int *, int *, double *, int *, int *, int *, double *, int *);

extern void pardisoinit(
        void *handle[64],
        const int *matrix_type,
        const int *solver,
        int iparam[64],
        double dparam[64],
        int *ierror);

extern void pardiso(
    void *handle[64],
    int *max_fac_store_in,
    int *matrix_number,
    int *matrix_type_in,
    int *ido,
    int *neqns_in,
    double *a,
    int *ia,
    int *ja,
    int *perm_user,
    int *nb,
    int iparam[64],
    int *msglvl,
    double *b,
    double *x,
    int *ierror,
    double dparam[64]);

double seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double) tv.tv_sec) + 1e-6 * tv.tv_usec;
}

static int malloc_error(void)
{
    printf("Error: malloc returned NULL\n");
    return 1;
}

/**
 * y = y - A*x
 */
static void SSP_matVecSymKernel2(
        int n,
        double *x,
        double *y,
        double *va,
        int *ja,
        int *ia)
{
    double s, v, xi;
    int i, j, k;

    assert(x != y);

    for (i = 0; i < n; i++)
    {
        xi = x[i];
        s = va[ia[i] - 1] * xi;
        for (k = ia[i]; k < ia[i + 1] - 1; k++)
        {
            j = ja[k] - 1;
            v = va[k];
            s += v * x[j];
            y[j] -= v * xi;
        }
        y[i] -= s;
    }
}


static int SSP_print_accuracy(
        int n,
        int *ia,
        int *ja,
        double *va,
        double *b,
        double *x)
{
    int i;
    double norm_res, norm_b;

    double *r = malloc(n * sizeof(double));
    if (r == NULL)
        return malloc_error();

    /* r = b - A * x */
    memcpy(r, b, n * sizeof(double));
    SSP_matVecSymKernel2(n, x, r, va, ja, ia);

    norm_res = 0.0;
    for (i = 0; i < n; ++i)
        norm_res += r[i] * r[i];
    norm_res = sqrt(norm_res);

    norm_b = 0.0;
    for (i = 0; i < n; ++i)
        norm_b += b[i] * b[i];
    norm_b = sqrt(norm_b);

    printf(FMT_NORM, "norm(b - A*x)/norm(b)", norm_res / norm_b);

    free(r);

    return 0;
}

static int pardiso_error(int ierror, int *iparam)
{
    printf("PARDISO returns with error code %d", ierror);
    switch (ierror)
    {
    case -1:
        printf(" (Input inconsistent)");
        break;
    case -2:
        printf(" (Not enough memory)");
        break;
    case -5:
        printf(" (Unclassified (internal) error)");
        break;
    case -10:
        printf(" (No license file found)");
        break;
    case -11:
        printf(" (License is expired)");
        break;
    case -12:
        printf(" (Wrong username or hostname)");
        break;
    }
    printf("\n");
    return 1;
}

/**
 * Get CPU model name from /proc/cpuinfo
 *
 * Stores the model name as a string in RESULT. RESULT_LEN is the size of array RESULT.
 */
static void get_cpu_model_name(char* result, int result_len)
{
    const char *field_name = "model name";
    char buf[256];
    FILE *f = NULL;
    char *colon, *newline;

    result[0] = '\0';
    f = fopen("/proc/cpuinfo", "r");
    if (f != NULL)
    {
        while (!feof(f))
        {
            fgets(buf, sizeof(buf) / sizeof(buf[0]), f);
            if (strncmp(field_name, buf, strlen(field_name)) == 0)
            {
                colon = strstr(buf, ":");
                if (colon != NULL)
                {
                    strncpy(result, colon + 2, result_len);
                    result[result_len - 1] = '\0';
                    newline = strstr(result, "\n");
                    if (newline != NULL)
                        *newline = '\0';
                    break;
                }
            }
        }
        fclose(f);
    }
}

static int print_welcome_msg(int myid, int mpi_num_procs)
{
    MPI_Status status;
    int proc_id;
    int mpiErr;
    int idummy;
    host_info_message_t msg;

    mpiErr = MPI_Get_processor_name(msg.processorName, &idummy);
    assert(mpiErr == 0);
    get_cpu_model_name(msg.cpu_model_name, MAX_CPU_MODEL_NAME);
#if defined(_OPENMP)
    msg.omp_num_threads = omp_get_max_threads();
#else
    msg.omp_num_threads = 1;
#endif

    for (proc_id = 0; proc_id < mpi_num_procs; ++proc_id)
    {
        if (myid == 0)
        {
            if (proc_id != 0)
                MPI_Recv(&msg, sizeof(msg), MPI_BYTE, proc_id, 0,
                        MPI_COMM_WORLD, &status);
            printf(FMT_GROUP_HOST, proc_id);
            printf(FMT_KEYVAL_STR_STR, "name", msg.processorName);
            printf(FMT_KEYVAL_STR_STR, "cpu model", msg.cpu_model_name);
            printf(FMT_KEYVAL_STR_INT, "omp_num_threads", msg.omp_num_threads);
        }
        else
        {
            if (proc_id == myid)
                MPI_Send(&msg, sizeof(msg), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return 0;
}

static void usage(char **argv)
{
    fprintf(stdout, "\n");
    fprintf(stdout, "Usage: mpirun -np i %s j // i=number of mpi processes, j=number of discretization points  \n\n", argv[0]);
}


void * _hh_malloc_laplace(size_t  count,  size_t  size)
{
        void * new = malloc (count * size);
        if (new == NULL)
        {
                printf("ERROR during memory allocation!\n");
                exit (7);
        }
        return  new;
}

void * _hh_calloc_laplace(size_t  count,  size_t  size)
{
        void * new = calloc (count, size);
        if (new == NULL)
        {
                printf("ERROR during cleared memory allocation!\n");
                exit (7);
        }
        return  new;
}


smat_laplace_t *  smat_new_laplace (int   m,
		   int    n,
		   int	    type)
{
	smat_laplace_t  *new_mat;

	mem_calloc (new_mat,  1,  smat_laplace_t);

	new_mat->m   	 = m;
	new_mat->n   	 = n;
	new_mat->nnz 	 = 0;

        if (type == 0 || type == 2)
	   new_mat->sym = 0;
        else
	   new_mat->sym = 1;

        if (type == 0 || type == 1)
	   new_mat->is_complex = 0;
        else
	   new_mat->is_complex = 1;

	mem_calloc (new_mat->ia,  m + 1,  int);

	new_mat->ja = NULL;
	new_mat->a  = NULL;

	return new_mat;
}


smat_laplace_t	* smat_new_nnz_struct_laplace (int	m,
			       int	n,
			       int	nnz,
			       int	type)
{
        /* type = 0 : unsymmetric, real     */
        /* type = 1 : symmetric,   real     */
        /* type = 2 : unsymmetric, is_complex  */
        /* type = 3 : symmetric,   is_complex  */

	smat_laplace_t  *new_mat = smat_new_laplace (m, n, type);

        assert(type == 0 || type == 1 || type == 2 || type == 1);

	new_mat->nnz = nnz;
	mem_alloc (new_mat->ja,  nnz+1,  int);

	return new_mat;
}

smat_laplace_t	* smat_new_nnz_laplace (int	m,
			int	n,
			int	nnz,
			int	type)
{
	smat_laplace_t  *new_mat = smat_new_nnz_struct_laplace (m, n, nnz, type);

	if (new_mat->is_complex == 0)
		mem_alloc (new_mat->a,  nnz+1,  double);
        else
		mem_alloc (new_mat->a,  2*nnz+1,  double);

	return new_mat;
}

smat_laplace_t * tst_gen_poisson_5_sym_laplace (int  n, double hx, double hx2, double c, double alpha)
{
        smat_laplace_t          *pm = smat_new_nnz_laplace(n*n, n*n, ((n - 1)*n) + ((2*n - 1)*n), 1);
        int              	i, nnz = 0;

        /* For n nxn blocks */
        for (i = 0; i < n; i++)
        {
                int  j;

                /* For each of the n rows */
                for (j = 0; j < n; j++)
                {
                        pm->ia[i*n + j] = nnz;

                        pm->ja[nnz] = i*n + j;
                        pm-> a[nnz]   = 4.0;
                        nnz++;

                        if (j < n-1)
                        {
                                pm->ja[nnz] = i*n + j + 1;
                                pm-> a[nnz  ] = -1.0;
                                nnz++;
                        }
                        if (i < n-1)
                        {
                                pm->ja[nnz] = (i+1)*n + j;
                                pm-> a[nnz  ] = -1.0; 
                                nnz++;
                        }

                }
        }
        pm->ia[n*n] = nnz;
        assert (pm->nnz == nnz);

        return  pm;
}

static int mpipardiso_driver(
        int n,
        int nnz,
        int* ia,
        int* ja,
        double *va,
        double *b,
        MPI_Comm comm)
{
    int mpi_rank;
    int mpi_size;

    /* Auxiliary variables. */
    char    *var;
    int	    i=0;

    int    mtype = -2;        /* Real symmetric indefinite matrix */


    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    if (mpi_rank == 0)
    {
        /*
         * Master process
         */

        double *x;
        double time_analyze = 0.0;
        double time_factorize = 0.0;
        double time_solve = 0.0;

        void  *handle[64];
        int    iparam[64];
        double dparam[64];
        int ido;
        int nb = 1; /* Only one right hand side.                */
        int Nmax_system = 1;
        int matrix_number = 1;
        int matrix_type = -2; /* Sparse symmetric matrix is indefinite.   */
        int solver = 0;       /* Use sparse direct method */
        int perm_user = 0;    /* Dummy pointer for user permutation.      */
        int msglvl = 1;
        int ierror = 0;       /* Initialize the Pardiso error flag        */

	int nrhs = 1;        /* Number of right hand sides  */

	/* Number of processors. */
    	int num_procs;

        /*
         * Setup Pardiso control parameters und initialize the solver's
         * internal address pointers. This is only necessary for the FIRST
         * call of the PARDISO solver.
         *
         * PARDISO license check and initialize solver
         */
        memset(dparam, 0, sizeof(dparam));
        memset(handle, 0, sizeof(handle));
        pardisoinit(handle, &matrix_type, &solver, iparam, dparam, &ierror);
	
	if (ierror != 0) 
    	{
		if (ierror == -10 )
         		printf("No license file found \n");
        	if (ierror == -11 )
           		printf("License is expired \n");
        	if (ierror == -12 )
           		printf("Wrong username or hostname \n");
    	}
    	else
        	printf("[PARDISO]: License check was successful ... \n");
        if (ierror)
            return pardiso_error(ierror, iparam);

   	/* Numbers of processors, value of OMP_NUM_THREADS */
    	var = getenv("OMP_NUM_THREADS");
    	if(var != NULL)
        	sscanf( var, "%d", &num_procs );
    	else {
        	printf("Set environment OMP_NUM_THREADS to 1");
        	exit(1);
    	}


        IPARAM(3) = num_procs;  /* maximal number of OpenMP tasks */
        IPARAM(8) = 0;         /* no iterative refinement */
        IPARAM(51) = 1;        /* use MPP factorization kernel */
        IPARAM(52) = mpi_size; /* number of nodes that we would like to use */

        printf(FMT_PARDISO_IPARAM, "IPARAM(3)", IPARAM(3),
                "Number of OpenMP tasks per host");
        printf(FMT_PARDISO_IPARAM, "IPARAM(11)", IPARAM(11), "Scaling");
        printf(FMT_PARDISO_IPARAM, "IPARAM(13)", IPARAM(13), "Matching");
        printf(FMT_PARDISO_IPARAM, "IPARAM(52)", IPARAM(52), "Number of hosts");

	/* -------------------------------------------------------------------- */
	/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
	/*     notation.                                                        */
	/* -------------------------------------------------------------------- */
    	for (i = 0; i < n+1; i++) 
        	ia[i] += 1;

    	for (i = 0; i < nnz; i++) 
        	ja[i] += 1;


	/* -------------------------------------------------------------------- */
	/*  .. pardiso_chk_matrix(...)                                          */
	/*     Checks the consistency of the given matrix.                      */
	/*     Use this functionality only for debugging purposes               */
	/* -------------------------------------------------------------------- */
	   
	pardiso_chkmatrix  (&mtype, &n, va, ia, ja, &ierror);
	if (ierror != 0) {
		printf("\nierror in consistency of matrix: %d", ierror);
		exit(1);
	}

	/* -------------------------------------------------------------------- */
	/* ..  pardiso_chkvec(...)                                              */
	/*     Checks the given vectors for infinite and NaN values             */
	/*     Input parameters (see PARDISO user manual for a description):    */
	/*     Use this functionality only for debugging purposes               */
	/* -------------------------------------------------------------------- */
	
	pardiso_chkvec (&n, &nrhs, b, &ierror);
	if (ierror != 0) {
		printf("\nierror  in right hand side: %d", ierror);
		exit(1);
	}

	/* -------------------------------------------------------------------- */
	/* .. pardiso_printstats(...)                                           */
	/*    prints information on the matrix to STDOUT.                       */
	/*    Use this functionality only for debugging purposes                */
	/* -------------------------------------------------------------------- */

	pardiso_printstats(&mtype, &n, va, ia, ja, &nrhs, b, &ierror);
	if (ierror != 0) {
		printf("\nierror right hand side: %d", ierror);
		exit(1);
	}

        /* Additional initialization for PARDISO MPI solver */
        pardiso_mpi_init_c_(handle, comm, &ierror);
        if (ierror)
            return pardiso_error(ierror, iparam);

        /* Allocate work arrays */
        x = malloc(n * sizeof(double));
        if (x == NULL)
            malloc_error();

        /* Analysis */
        time_analyze -= seconds();

        /*
         * Start symbolic factorization of the PARDISO solver.
         */
        ido = 11; /* perform only symbolic factorization */
        pardiso(handle, &Nmax_system, &matrix_number, &matrix_type, &ido, &n,
                va, ia, ja, &perm_user, &nb, iparam, &msglvl, b, x, &ierror, dparam);

        if (ierror)
            return pardiso_error(ierror, iparam);

        time_analyze += seconds();
        printf(TIME_FMT, "time analyze", time_analyze);

        printf(FMT_PARDISO_IPARAM, "IPARAM(18)", IPARAM(18),
                "Number of LU nonzeros");
        printf(FMT_PARDISO_IPARAM, "IPARAM(19)", IPARAM(19), "Number of FLOPS");
        printf(FMT_PARDISO_IPARAM, "IPARAM(15)", IPARAM(15), "Analysis memory");
        printf(FMT_PARDISO_IPARAM, "IPARAM(16)", IPARAM(16), "Structure memory");
        printf(FMT_PARDISO_IPARAM, "IPARAM(17)", IPARAM(17), "Factor memory");

        /*
         * Start numerical factorization of the PARDISO solver.
         */
        time_factorize = -seconds();

        ido = 22;
        pardiso(handle, &Nmax_system, &matrix_number, &matrix_type, &ido, &n,
                va, ia, ja, &perm_user, &nb, iparam, &msglvl, b, x, &ierror,
                dparam);

        if (ierror)
            return pardiso_error(ierror, iparam);

        time_factorize += seconds();
        printf(TIME_FMT, "time_factorize", time_factorize);

        printf(FMT_PARDISO_IPARAM, "IPARAM(14)", IPARAM(14), "Perturbed pivots");
        printf(FMT_PARDISO_IPARAM, "IPARAM(22)", IPARAM(22), "Positive pivots");
        printf(FMT_PARDISO_IPARAM, "IPARAM(23)", IPARAM(23), "Negative pivots");
        printf(FMT_PARDISO_DPARAM, "DPARAM(33)", DPARAM(33), "Determinant");

        /*
         * Start forward/backward substitution of the PARDISO solver.
         */
        time_solve = -seconds();

        ido = 33;
        pardiso(handle, &Nmax_system, &matrix_number, &matrix_type, &ido, &n,
                va, ia, ja, &perm_user, &nb, iparam, &msglvl, b, x, &ierror,
                dparam);

        if (ierror)
            return pardiso_error(ierror, iparam);

        time_solve += seconds();
        printf(TIME_FMT, "time_solve", time_solve);

        /*
         * Compute relative residual
         */
        SSP_print_accuracy(n, ia, ja, va, b, x);
        {
            double norm_b = 0.0;
            int i;
            for (i = 0; i < n; ++i)
                norm_b += b[i] * b[i];
            norm_b = sqrt(norm_b);

            printf(FMT_NORM, "norm(b)", norm_b);

            double norm_x = 0.0;
            for (i = 0; i < n; ++i)
                norm_x += x[i] * x[i];
            norm_x = sqrt(norm_x);

            printf(FMT_NORM, "norm(x)", norm_x);

        }

        /*
         * Free PARDISO internal memory
         */
        ido = -1;
        pardiso(handle, &Nmax_system, &matrix_number, &matrix_type, &ido, &n,
                va, ia, ja, &perm_user, &nb, iparam, &msglvl, b, x, &ierror,
                dparam);

        if (ierror)
            return pardiso_error(ierror, iparam);

        /* Additional finalization for MPI solver */
        pardiso_mpi_finalize_(handle, &ierror);

        if (ierror)
            return pardiso_error(ierror, iparam);

        free(x);
    }
    else /* mpi_rank != 0 */
    {
        /*
         * Worker process
         */
        void *handle[64];
        int ierror = 0;

        /* Initialization for PARDISO MPI solver */
        pardiso_mpi_init_c_(handle, comm, &ierror);
        if (ierror)
            return 1;

        /* Symbolic factorization */
        pardiso_mpi_worker_(handle, &ierror);
        if (ierror)
            return 1;

        /* Numerical factorization */
        pardiso_mpi_worker_(handle, &ierror);
        if (ierror)
            return 1;

        /* Triangular solve */
        pardiso_mpi_worker_(handle, &ierror);
        if (ierror)
            return 1;

        /* Free solver memory */
        pardiso_mpi_worker_(handle, &ierror);

        /* Finalization for MPI solver */
        pardiso_mpi_finalize_(handle, &ierror);
    }

    return 0;
}


int main(int argc, char **argv) 
{

    int    grid  = atoi(argv[1]); /* Number of discretization points in one spatial direction */
    int    nnz;
  
     /* RHS and solution vectors. */
    double*  b      = NULL;

    /* Number of processors. */
    int myid, n, mpi_num_procs;
    int ierr;
    int status = 0;

    /* Auxiliary variables. */
    int      i;
   
    double           Xmax  = 6400;
    double           Xmin  =  400;
    double           c;
    double           hx, hx2;
    /* damping of the material */
    double           alpha = 0.05;

    /* Matrix data. */
    smat_laplace_t* A = NULL;

    /* MPI */
    MPI_Init(&argc, &argv);

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    assert(ierr == 0);

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_procs);
    assert(ierr == 0);

    print_welcome_msg(myid, mpi_num_procs);

   /* Parse command line args and broadcast error status */
    if (myid == 0)
    {
        if (argc == 2)
            grid = atoi(argv[1]);
        else
        {
            usage(argv);
            status = 1;
        }
    }
 
   ierr = MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
    if (!status)
    {
	    /* Parse command line args and broadcast error status */
	    if (myid == 0)
	    {
		/* mesh size */
	 	hx    = (Xmax-Xmin)/(grid-1);        
		hx2   = 1/(hx*hx);
		n = grid * grid;
		b      = malloc (n * sizeof (double));
		if (b == NULL) 
		{
			printf("\nERROR during malloc of b");
			exit(9);
		} 
		/* Set right hand side b to one */
		for (i = 0; i < n; i++) 
			b[i]   = i;
		c = 0;
	    	A = tst_gen_poisson_5_sym_laplace(grid, hx, hx2, c, alpha);
		n = A->m;
		nnz = A->ia[n];
		mpipardiso_driver(n, nnz, A->ia, A->ja, A->a, b, MPI_COMM_WORLD);
	    }
	    else
	    {
		mpipardiso_driver(-1, -1, NULL, NULL, NULL, NULL, MPI_COMM_WORLD);
	    }
    }
    MPI_Finalize();
    return 0;
}
