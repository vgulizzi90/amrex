#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>

#include "blas/AMReX_blas.H"
#include "lapack/AMReX_lapack.H"

// TESTED ROUTINES ====================================================
extern "C"
{
    // BLAS LV 1
    extern int idamax_(int *, double *, int *);
    extern int drot_(int *, double *, int *, double *, int *, double *, double *);
    extern double dnrm2_(int *, double *, int *);

    // BLAS LV 2
    extern int dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
    
    // BLAS LV 3
    extern int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
    extern int dtrsm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *, int *);

    // LAPACK
    extern int dlaswp_(int *, double *, int *, int *, int *, int *, int *);
    extern int dgetf2_(int *, int *, double *, int *, int *, int *);
    extern int dgetrf_(int *, int *, double *, int *, int *, int *);
    extern int dgetrs_(const char *, int *, int *, double *, int *, int *, double *, int *, int *);
    extern int dgetri_(int *, double *, int *, int *, double *, int *, int *);
    extern int dpotrf_(const char *, int *, double *, int *, int *);
    extern int dpotrs_(char *, int *, int *, double *, int *, double *, int *, int *);

    extern int dgeev_(char *, char *, int *, double *, int *, double *, double *, double *, int *, double *, int *, double *, int *, int *);
    extern int dsyev_(char *, char *, int *, double *, int *, double *, double *, int *, int *);

}
// ====================================================================

// PRINT ARRAYS =======================================================
void dPrint2D(const std::string & name, int Nr, int Nc, const double * a)
{
    std::cout << name << ":" << std::endl;
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c) std::cout << std::scientific << std::setprecision(5) << std::setw(12) << a[r+c*Nr] << " ";
        std::cout << std::endl;
    }
}
// ====================================================================

// COMPARE ARRAYS =====================================================
int dCompare(const int & size, const double * A, const double * B, const double & tol)
{
    double diff[size];
    for (int k = 0; k < size; ++k) diff[k] = std::abs(A[k]-B[k]);
    if (*std::max_element(diff, diff+size) > tol) return -1;
    else return 0;
}
// ====================================================================

// ERROR MESSAGE ======================================================
void ErrMsg(const std::string & region, const std::string routine)
{
    std::cout << "TEST FAILED: " << region << " - " << routine << std::endl;
    exit(-1);
}
// ====================================================================

int main()
{
// RANDOM NUMBER GENERATOR  ===========================================
unsigned seed(std::chrono::system_clock::now().time_since_epoch().count());
std::default_random_engine gen(seed);
std::uniform_real_distribution<double> distr(0.0, 1.0);
// ====================================================================

// ####################################################################
// BLAS LV 1 ##########################################################
// ####################################################################
// IDAMAX =============================================================
{
    int N = 5;

    double X[N];
    int id, idRef;
    int INCX = 1;

    // INIT
    for (int n = 0; n < N; ++n) X[n] = distr(gen);

    // CALL BLAS
    idRef = idamax_(&N, X, &INCX);

    // CALL AMREX_LINALG
    id = amrex::linalg::idamax_(&N, X, &INCX);

    // COMPUTE THE DIFFERENCE
    if (idRef != id)
    {
        dPrint2D("X", 1, N, X);
        std::cout << "idRef: " << idRef << std::endl;
        std::cout << "id: " << id << std::endl;

        ErrMsg("BLAS LV 1", "IDAMAX");
    }
}
// ====================================================================
// DROT ===============================================================
{
    int N = 5;

    double dX[N], dY[N], dXRef[N], dYRef[N];
    double c = distr(gen);
    double s = distr(gen);
    int INCX = 1, INCY = 1;

    // INIT
    for (int n = 0; n < N; ++n)
    {
        dX[n] = distr(gen);
        dY[n] = distr(gen);
        dXRef[n] = dX[n];
        dYRef[n] = dY[n];
    }

    // CALL BLAS
    drot_(&N, dXRef, &INCX, dYRef, &INCY, &c, &s);

    // CALL AMREX_LINALG
    amrex::linalg::drot_(&N, dX, &INCX, dY, &INCY, &c, &s);

    // COMPUTE THE DIFFERENCE
    if ((dCompare(N, dX, dXRef, 1.0e-12) == -1) ||
        (dCompare(N, dY, dYRef, 1.0e-12) == -1))
    {
        dPrint2D("dX", 1, N, dX);
        dPrint2D("dXRef", 1, N, dXRef);
        dPrint2D("dY", 1, N, dY);
        dPrint2D("dYRef", 1, N, dYRef);

        ErrMsg("BLAS LV 1", "DROT");
    }
}
// ====================================================================
// DNRM2 ==============================================================
{
    int N = 5;

    double X[N], nX, XRef[N], nXRef;
    int INCX = 1;

    // INIT
    for (int n = 0; n < N; ++n)
    {
        X[n] = distr(gen);
        XRef[n] = X[n];
    }

    // CALL BLAS
    nXRef = dnrm2_(&N, XRef, &INCX);

    // CALL AMREX_LINALG
    nX = amrex::linalg::dnrm2_(&N, X, &INCX);

    // COMPUTE THE DIFFERENCE
    if (std::abs(nX-nXRef) > 1.0e-12)
    {
        dPrint2D("X", 1, N, X);
        dPrint2D("XRef", 1, N, XRef);
        std::cout << "nX: " << nX << std::endl;
        std::cout << "nXRef: " << nXRef << std::endl;

        ErrMsg("BLAS LV 1", "DNRM2");
    }
}
// ====================================================================
// ####################################################################
// ####################################################################




// ####################################################################
// BLAS LV 2 ##########################################################
// ####################################################################
// DGEMV ==============================================================
{
    int M = 6;
    int N = 4;

    char TRANS = (distr(gen) > 0.5) ? 'T' : 'N';
    double ALPHA = distr(gen);
    double BETA = distr(gen);
    int INCX = 1;
    int INCY = 1;
    double A[M*N], X[N], Y[M], YRef[M], Y0[M];

    // INIT
    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
        A[m+M*n] = distr(gen);

    for (int n = 0; n < N; ++n) X[n] = distr(gen);

    for (int m = 0; m < M; ++m)
    {

        Y0[m] = distr(gen);
        YRef[m] = Y0[m];
        Y[m] = Y0[m];
    }

    // CALL BLAS
    dgemv_(&TRANS, &M, &N, &ALPHA, A, &M, X, &INCX, &BETA, YRef, &INCY);

    // CALL AMREX_LINALG
    amrex::linalg::dgemv_(&TRANS, &M, &N, &ALPHA, A, &M, X, &INCX, &BETA, Y, &INCY);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M, Y, YRef, 1.0e-12) == -1)
    {
        dPrint2D("Y0", 1, M, Y0);
        dPrint2D("YRef", 1, M, YRef);
        dPrint2D("Y", 1, M, Y);

        ErrMsg("BLAS LV 2", "DGEMV");
    }
}
// ====================================================================
// ####################################################################
// ####################################################################




// ####################################################################
// BLAS LV 3 ##########################################################
// ####################################################################
// DGEMM ==============================================================
{
    int M = 6;
    int K = 5;
    int N = 4;

    char TRANSA = 'N';
    char TRANSB = 'T';
    double ALPHA = distr(gen);
    double BETA = distr(gen);
    double A[M*K], B[K*N], C[M*N], CRef[M*N], C0[M*N];

    // INIT
    for (int k = 0; k < K; ++k)
    for (int m = 0; m < M; ++m)
        A[m+M*k] = distr(gen);
    
    for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
        B[k+K*n] = distr(gen);

    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        C0[m+M*n] = distr(gen);
        CRef[m+M*n] = C0[m];
        C[m+M*n] = C0[m];
    }

    // CALL BLAS
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &M, B, &K, &BETA, CRef, &M);

    // CALL AMREX_LINALG
    amrex::linalg::dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &M, B, &K, &BETA, C, &M);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*N, C, CRef, 1.0e-12) == -1)
    {
        dPrint2D("A", M, K, A);
        dPrint2D("B", K, N, B);
        
        dPrint2D("C0", M, N, C0);
        dPrint2D("CRef", M, N, CRef);
        dPrint2D("C", M, N, C);

        ErrMsg("BLAS LV 3", "DGEMM");
    }
}
// ====================================================================
// DTRSM ==============================================================
{
    int M = 6;
    int N = 4;

    char SIDE = 'L';
    char UPLO = 'U';
    char TRANSA = 'T';
    char DIAG = 'N';
    double ALPHA = distr(gen);
    double A[M*M], X[M*N], XRef[M*N], X0[M*N];
    int LDA = (SIDE == 'L') ? M : N;

    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        X0[m+M*n] = distr(gen);
        XRef[m+M*n] = X0[m];
        X[m+M*n] = X0[m];
    }

    if (SIDE == 'L')
    {
        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
            A[m+M*n] = distr(gen);
        }
    }
    else
    {
        for (int n = 0; n < N; ++n)
        for (int m = 0; m < N; ++m)
        {
            A[m+N*n] = distr(gen);
        }  
    }

    // CALL BLAS
    dtrsm_(&SIDE, &UPLO, &TRANSA, &DIAG, &M, &N, &ALPHA, A, &LDA, XRef, &M);

    // CALL AMREX_LINALG
    amrex::linalg::dtrsm_(&SIDE, &UPLO, &TRANSA, &DIAG, &M, &N, &ALPHA, A, &LDA, X, &M);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*N, X, XRef, 1.0e-12) == -1)
    {
        if (SIDE == 'L')
        {
            dPrint2D("A", M, M, A);
            dPrint2D("B", M, N, X0);
        }
        else
        {
            dPrint2D("A", N, N, A);
            dPrint2D("B", M, N, X0);
        }
        
        dPrint2D("XRef", M, N, XRef);
        dPrint2D("X", M, N, X);

        ErrMsg("BLAS LV 3", "DTRSM");
    }
}
// ====================================================================
// ####################################################################
// ####################################################################




// ####################################################################
// LAPACK #############################################################
// ####################################################################
// DLASWP =============================================================
{
    int M = 6;
    int N = 4;

    double A0[M*N], A[M*N], ARef[M*N];
    int K1 = 1;
    int K2 = M-1;
    int IPIV[M];
    int INCX = 1;

    // INIT
    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    for (int m = 0; m < M; ++m)
    {
        IPIV[m] = M-m-1;
    }

    // CALL LAPACK
    dlaswp_(&N, ARef, &M, &K1, &K2, IPIV, &INCX);

    // CALL AMREX_LINALG
    amrex::linalg::dlaswp_(&N, A, &M, &K1, &K2, IPIV, &INCX);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*N, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, N, A0);
        dPrint2D("ARef", M, N, ARef);
        dPrint2D("A", M, N, A);

        ErrMsg("LAPACK", "DLASWP");
    }
}
// ====================================================================

// DGETF2 =============================================================
{
    int M = 6;
    int N = 4;

    double A0[M*N], A[M*N], ARef[M*N];
    int IPIV[std::min(M,N)];
    int INFO;

    // INIT
    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    // CALL LAPACK
    dgetf2_(&M, &N, ARef, &M, IPIV, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dgetf2_(&M, &N, A, &M, IPIV, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*N, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, N, A0);
        dPrint2D("ARef", M, N, ARef);
        dPrint2D("A", M, N, A);

        ErrMsg("LAPACK", "DGETF2");
    }
}
// ====================================================================

// DGETRF =============================================================
{
    int M = 6;
    int N = 4;

    double A0[M*N], A[M*N], ARef[M*N];
    int IPIV[std::min(M,N)];
    int INFO;

    // INIT
    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    // CALL LAPACK
    dgetrf_(&M, &N, ARef, &M, IPIV, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dgetrf_(&M, &N, A, &M, IPIV, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*N, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, N, A0);
        dPrint2D("ARef", M, N, ARef);
        dPrint2D("A", M, N, A);

        ErrMsg("LAPACK", "DGETRF");
    }
}
// ====================================================================

// DGETRS =============================================================
{
    int M = 6;
    int N = 4;

    char TRANS = (distr(gen) > 0.5) ? 'T' : 'N';
    double A0[M*M], A[M*M], ARef[M*M], X[M*N], XRef[M*N], X0[M*N];
    int IPIV[M];
    int INFO;

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        X0[m+M*n] = distr(gen);
        X[m+M*n] = X0[m+M*n];
        XRef[m+M*n] = X0[m+M*n];
    }

    // CALL LAPACK
    dgetrf_(&M, &M, ARef, &M, IPIV, &INFO);
    dgetrs_(&TRANS, &M, &N, ARef, &M, IPIV, XRef, &M, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dgetrf_(&M, &M, A, &M, IPIV, &INFO);
    amrex::linalg::dgetrs_(&TRANS, &M, &N, A, &M, IPIV, X, &M, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("ARef", M, M, ARef);
        dPrint2D("A", M, M, A);

        ErrMsg("LAPACK", "DGETRF of DGETRS");
    }
    
    if (dCompare(M*N, X, XRef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, M, A0);
        
        dPrint2D("X0", M, N, X0);

        dPrint2D("XRef", M, N, XRef);
        dPrint2D("X", M, N, X);

        ErrMsg("LAPACK", "DGETRS");
    }
}
// ====================================================================

// DGETRI =============================================================
{
    int M = 7;

    double A0[M*M], A[M*M], ARef[M*M];
    int IPIV[M];
    int INFO;
    int LWORK = M*M;
    double WORK[LWORK];

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    // CALL LAPACK
    dgetrf_(&M, &M, ARef, &M, IPIV, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dgetrf_(&M, &M, A, &M, IPIV, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("ARef", M, M, ARef);
        dPrint2D("A", M, M, A);

        ErrMsg("LAPACK", "DGETRF of DGETRI");
    }

    // CALL LAPACK
    dgetri_(&M, ARef, &M, IPIV, WORK, &LWORK, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dgetri_(&M, A, &M, IPIV, WORK, &LWORK, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("ARef", M, M, ARef);
        dPrint2D("A", M, M, A);

        ErrMsg("LAPACK", "DGETRI");
    }
}
// ====================================================================

// DPOTRF =============================================================
{
    int M = 7;

    double A0[M*M], A[M*M], ARef[M*M];
    char UPLO = (distr(gen) > 0.5) ? 'U' : 'L';
    int INFO;

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
    }

    for (int n = 0; n < M; ++n)
    for (int m = n+1; m < M; ++m)
    {
        A0[m+M*n] = A0[n+M*m];
    }

    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    // CALL LAPACK
    dpotrf_(&UPLO, &M, ARef, &M, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dpotrf_(&UPLO, &M, A, &M, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, M, A0);
        dPrint2D("ARef", M, M, ARef);
        dPrint2D("A", M, M, A);

        ErrMsg("LAPACK", "DPOTRF");
    }   
}
// ====================================================================

// DPOTRS =============================================================
{
    int M = 6;
    int N = 4;
    
    double A0[M*M], A[M*M], ARef[M*M], X[M*N], XRef[M*N], X0[M*N];
    char UPLO = (distr(gen) > 0.5) ? 'U' : 'L';
    int INFO;

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        A[m+n*M] = A0[m+M*n];
        ARef[m+n*M] = A0[m+M*n];
    }

    for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
    {
        X0[m+M*n] = distr(gen);
        X[m+M*n] = X0[m+M*n];
        XRef[m+M*n] = X0[m+M*n];
    }

    // CALL LAPACK
    dpotrf_(&UPLO, &M, ARef, &M, &INFO);
    dpotrs_(&UPLO, &M, &N, ARef, &M, XRef, &M, &INFO);

    // CALL AMREX_LINALG
    amrex::linalg::dpotrf_(&UPLO, &M, A, &M, &INFO);
    amrex::linalg::dpotrs_(&UPLO, &M, &N, A, &M, X, &M, &INFO);

    // COMPUTE THE DIFFERENCE
    if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
    {
        dPrint2D("ARef", M, M, ARef);
        dPrint2D("A", M, M, A);

        ErrMsg("LAPACK", "DPOTRF of DPOTRS");
    }
    
    if (dCompare(M*N, X, XRef, 1.0e-12) == -1)
    {
        dPrint2D("A0", M, M, A0);
        
        dPrint2D("X0", M, N, X0);

        dPrint2D("XRef", M, N, XRef);
        dPrint2D("X", M, N, X);

        ErrMsg("LAPACK", "DPOTRS");
    }
}
// ====================================================================

// DGEEV ==============================================================
{
    int M = 2;

    char JOBVL = 'V', JOBVR = 'V';
    double A0[M*M], A[M*M], ARef[M*M];
    int LWORK = 6*M;
    double WORK[LWORK], WORKRef[LWORK];
    int INFO, INFORef;
    double wRe[M], wIm[M], wReRef[M], wImRef[M];
    double VL[M*M], VR[M*M], VLRef[M*M], VRRef[M*M];

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        ARef[m+M*n] = A0[m+M*n];
        A[m+M*n] = A0[m+M*n];
    }

    // CALL LAPACK
    dgeev_(&JOBVL, &JOBVR, &M, ARef, &M, wReRef, wImRef, VLRef, &M, VRRef, &M, WORKRef, &LWORK, &INFORef);

    // CALL AMREX_LINALG
    amrex::linalg::dgeev_(&JOBVL, &JOBVR, &M, A, &M, wRe, wIm, VL, &M, VR, &M, WORK, &LWORK, &INFO);
    
    // COMPUTE THE DIFFERENCE
    if ((dCompare(M, wRe, wReRef, 1.0e-12) == -1) ||
        (dCompare(M, wIm, wImRef, 1.0e-12) == -1) ||
        (dCompare(M*M, VL, VLRef, 1.0e-12) == -1) ||
        (dCompare(M*M, VR, VRRef, 1.0e-12) == -1))
    {
        dPrint2D("A", M, M, A0);

        dPrint2D("wReRef", 1, M, wReRef);
        dPrint2D("wRe", 1, M, wRe);
        dPrint2D("wImRef", 1, M, wImRef);
        dPrint2D("wIm", 1, M, wIm);
        dPrint2D("VLRef", M, M, VLRef);
        dPrint2D("VL", M, M, VL);
        dPrint2D("VRRef", M, M, VRRef);
        dPrint2D("VR", M, M, VR);

        ErrMsg("LAPACK", "DGEEV");
    }
}
// ====================================================================

// DSYEV ==============================================================
{
    int M = 4;

    char JOB = 'V';
    char UPLO = (distr(gen) > 0.5) ? 'U' : 'L';
    double A0[M*M], A[M*M], ARef[M*M];
    int LWORK = 3*M;
    double WORK[LWORK], WORKRef[LWORK];
    int INFO, INFORef;
    double w[M], wRef[M];
    double V[M*M], VRef[M*M];

    // INIT
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        A0[m+M*n] = distr(gen);
        ARef[m+M*n] = A0[m+M*n];
        A[m+M*n] = A0[m+M*n];
    }

    // CALL LAPACK
    dsyev_(&JOB, &UPLO, &M, ARef, &M, wRef, WORKRef, &LWORK, &INFORef);

    // CALL AMREX_LINALG
    amrex::linalg::dsyev_(&JOB, &UPLO, &M, A, &M, w, WORK, &LWORK, &INFO);
    
    for (int n = 0; n < M; ++n)
    for (int m = 0; m < M; ++m)
    {
        VRef[m+M*n] = ARef[m+M*n];
        V[m+M*n] = A[m+M*n];
    }
    
    // COMPUTE THE DIFFERENCE
    if ((dCompare(M, w, wRef, 1.0e-12) == -1) ||
        (dCompare(M*M, V, VRef, 1.0e-12) == -1))
    {
        dPrint2D("A", M, M, A0);

        dPrint2D("w", 1, M, w);
        dPrint2D("wRef", 1, M, wRef);
        dPrint2D("VRef", M, M, VRef);
        dPrint2D("V", M, M, V);

        ErrMsg("LAPACK", "DSYEV");
    }
}
// ====================================================================
// ####################################################################
// ####################################################################

}