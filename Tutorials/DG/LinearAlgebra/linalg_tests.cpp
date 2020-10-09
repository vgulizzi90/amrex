#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>

// TESTED ROUTINES ====================================================
extern "C"
{
    // BLAS LV 1
    extern bool lsame_(char *, char *);
    extern int idamax_(int *, double *, int *);
    extern double ddot_(const int *, const double *, const int *, const double *, const int *);

    // BLAS LV 2
    extern int dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
    extern int dtrmv_(char *, char *, char *, int *, double *, int *, double *, int *);
    extern int dtrti2_(const char *, const char *, int *, double *, int *, int *);
    
    // BLAS LV 3
    extern int dtrsm_(char *, char *, char *, char *, int *, int *, double *, double *, int *, double *, int *);

    // LAPACK
    extern int dlaswp_(int *, double *, int *, int *, int *, int *, int *);
    extern int dgetf2_(int *, int *, double *, int *, int *, int *);
    extern int dgetrf_(int *, int *, double *, int *, int *, int *);
    extern int dgetrs_(const char *, int *, int *, double *, int *, int *, double *, int *, int *);
    extern int dgetri_(int *, double *, int *, int *, double *, int *, int *);
    extern int dpotf2_(const char *, int *, double *, int *, int *);
    extern int dpotrf_(const char *, int *, double *, int *, int *);
    extern int dpotrs_(char *, int *, int *, double *, int *, double *, int *, int *);

}
// ====================================================================

#define AMREX_GPU_HOST_DEVICE
#define AMREX_FORCE_INLINE __inline__
#define Real double
#define Abort() exit(-1)

#include "../../../Src/DG/AMReX_DG_LinAlg.H"

// AUXILIARY ROUTINES =================================================
void iPrint2D(const std::string & name, int Nr, int Nc, const int * a)
{
    std::cout << name << ":" << std::endl;
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c) std::cout << a[r+c*Nr] << " ";
        std::cout << std::endl;
    }
}
void dPrint2D(const std::string & name, int Nr, int Nc, const double * a)
{
    std::cout << name << ":" << std::endl;
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c) std::cout << std::scientific << std::setprecision(5) << std::setw(12) << a[r+c*Nr] << " ";
        std::cout << std::endl;
    }
}
int dCompare(const int size, const double * A, const double * B, const double tol)
{
    double diff[size];
    for (int k = 0; k < size; ++k) diff[k] = std::abs(A[k]-B[k]);
    if (*std::max_element(diff, diff+size) > tol) return -1;
    else return 0;
}
double dMaxDiff(const int size, const double * A, const double * B)
{
    double diff[size];
    for (int k = 0; k < size; ++k) diff[k] = std::abs(A[k]-B[k]);
    return (*std::max_element(diff, diff+size));
}
void ErrMsg(const std::string & region, const std::string routine)
{
    std::cout << "TEST FAILED: " << region << " - " << routine << std::endl;
    exit(-1);
}

void Eye(const int N, Real * A)
{
    std::fill(A, A+N*N, 0.0);
    for (int n = 0; n < N; ++n)
    {
        A[n+n*N] = 1.0;
    }
}
// ====================================================================


// MAIN PROGRAM
int main()
{
    // RANDOM NUMBER GENERATOR  =======================================
    unsigned seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    // ================================================================

    // ################################################################
    // BLAS LV 1 ######################################################
    // ################################################################
    // IDAMAX
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
        id = amrex::DG::linalg::idamax(N, X, INCX);

        // COMPUTE THE DIFFERENCE
        if (idRef != (id+1))
        {
            dPrint2D("X", 1, N, X);
            std::cout << "idRef: " << idRef << std::endl;
            std::cout << "id: " << id << std::endl;

            ErrMsg("BLAS LV 1", "IDAMAX");
        }
    }

    // DDOT
    {
        int N = 54;

        double X[N], Y[N];
        int INCX = 1;
        int INCY = 1;
        double resRef, res;

        // INIT
        for (int n = 0; n < N; ++n)
        {
            X[n] = distr(gen);
            Y[n] = distr(gen);
        }

        // CALL BLAS
        resRef = ddot_(&N, X, &INCX, Y, &INCY);

        // CALL AMREX_LINALG
        res = amrex::DG::linalg::ddot(N, X, INCX, Y, INCY);

        // COMPUTE THE DIFFERENCE
        if (std::abs(resRef-res) > 1.0e-12)
        {
            std::cout << "resRef: " << resRef << std::endl;
            std::cout << "res: " << res << std::endl;

            ErrMsg("BLAS LV 1", "DDOT");
        }
    }
    // ################################################################
    // ################################################################



    // ################################################################
    // BLAS LV 2 ######################################################
    // ################################################################
    // DGEMV
    {
        char TRANS = 'T';
        int M = 76;
        int N = 54;
        double alpha = 12.0;
        double A[M*N];
        int LDA = M;
        double X[N];
        int INCX = 1;
        double beta = 2.0;
        double YRef[M], Y[M], Y0[M];
        int INCY = 1;

        // INIT
        for (int n = 0; n < M*N; ++n)
        {
            A[n] = distr(gen);
        }
        for (int n = 0; n < N; ++n)
        {
            X[n] = distr(gen);
        }
        for (int n = 0; n < M; ++n)
        {
            Y[n] = distr(gen);
            YRef[n] = Y[n];
            Y0[n] = Y[n];
        }

        // CALL BLAS
        dgemv_(&TRANS, &M, &N, &alpha, A, &LDA, X, &INCX, &beta, YRef, &INCY);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dgemv(TRANS, M, N, alpha, A, LDA, X, INCX, beta, Y, INCY);

        // COMPUTE THE DIFFERENCE
        if (dCompare(N, Y, YRef, 1.0e-12) == -1)
        {
            dPrint2D("Y0", N, 1, Y0);
            dPrint2D("YRef", N, 1, YRef);
            dPrint2D("Y", N, 1, Y);

            ErrMsg("BLAS LV 2", "DGEMV");
        }
    }
    
    // DTRMV
    {
        int N = 5;
        int LDA = 5;

        double A[N*N];
        double X0[N], X[N], XRef[N];
        char UPLO = 'L';
        char TRANS = 'T';
        char DIAG = 'U';
        int INCX = 1;

        // INIT
        for (int n = 0; n < N*N; ++n)
        {
            A[n] = distr(gen);
        }
        //Eye(N, A);
        for (int n = 0; n < N; ++n)
        {
            XRef[n] = distr(gen);
            X[n] = XRef[n];
            X0[n] = XRef[n];
        }

        // CALL BLAS
        dtrmv_(&UPLO, &TRANS, &DIAG, &LDA, A, &LDA, XRef, &INCX);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dtrmv(UPLO, TRANS, DIAG, LDA, A, LDA, X, INCX);

        // COMPUTE THE DIFFERENCE
        if (dCompare(N, X, XRef, 1.0e-12) == -1)
        {
            dPrint2D("A", N, N, A);
            dPrint2D("X0", N, 1, X0);
            dPrint2D("XRef", N, 1, XRef);
            dPrint2D("X", N, 1, X);

            ErrMsg("BLAS LV 2", "DTRMV");
        }
    }

    // DTRTI2
    {
        int N = 5;

        double A0[N*N], A[N*N], ARef[N*N];
        char UPLO = 'L';
        char DIAG = 'N';
        int INFO;

        // INIT
        for (int n = 0; n < N*N; ++n)
        {
            A0[n] = distr(gen);
            A[n] = A0[n];
            ARef[n] = A0[n];
        }

        // CALL LAPACK
        dtrti2_(&UPLO, &DIAG, &N, ARef, &N, &INFO);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dtrti2(UPLO, DIAG, N, A, N, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(N*N, A, ARef, 1.0e-12) == -1)
        {
            dPrint2D("A0", N, N, A0);
            dPrint2D("ARef", N, N, ARef);
            dPrint2D("A", N, N, A);

            ErrMsg("BLAS LV 2", "DTRTI2");
        }
    }
    // ################################################################
    // ################################################################



    // ################################################################
    // BLAS LV 3 ######################################################
    // ################################################################
    // DTRSM
    {
        int M = 6;
        int N = 4;
        int LDA = M;
        int LDB = M;

        char SIDE = 'L';
        char UPLO = 'L';
        char TRANS = 'T';
        char DIAG = 'U';
        Real alpha = 1.0;//distr(gen);
        int sizeA = LDA*((SIDE == 'L') ? M : N);
        int sizeB = LDB*N;
        Real A[sizeA], B[sizeB], X[sizeB], XRef[sizeB];

        // INIT
        for (int n = 0; n < sizeA; ++n)
        {
            A[n] = distr(gen);
        }
        //Eye(M, A); A[0] = 1.0; A[LDA] = 1.0;
        for (int n = 0; n < sizeB; ++n)
        {
            B[n] = distr(gen);
            XRef[n] = B[n];
            X[n] = B[n];
        }

        // CALL BLAS
        dtrsm_(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &alpha, A, &LDA, XRef, &LDB);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dtrsm(SIDE, UPLO, TRANS, DIAG, M, N, alpha, A, LDA, X, LDB);

        // COMPUTE THE DIFFERENCE
        if (dCompare(sizeB, X, XRef, 1.0e-12) == -1)
        {
            dPrint2D("A", LDA, ((SIDE == 'L') ? M : N), A);
            dPrint2D("B", LDB, N, B);
            dPrint2D("XRef", LDB, N, XRef);
            dPrint2D("X", LDB, N, X);

            ErrMsg("LAPACK", "DTRSM");
        }
    }
    // ################################################################
    // ################################################################



    // ################################################################
    // LAPACK #########################################################
    // ################################################################
    // DLASWP
    {
        int M = 6;
        int N = 4;

        double A0[M*N], A[M*N], ARef[M*N];
        int INCX = -1;
        int K1 = 1;
        int K2 = M;
        int sizeIPIV = K1+(K2-K1)*std::abs(INCX);
        int IPIVRef[sizeIPIV], IPIV[sizeIPIV];

        // INIT
        for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m)
        {
            A0[m+M*n] = distr(gen);
            A[m+n*M] = A0[m+M*n];
            ARef[m+n*M] = A0[m+M*n];
        }
        
        IPIVRef[0] = 6; IPIV[0] = 6-1;
        IPIVRef[1] = 4; IPIV[1] = 4-1;
        IPIVRef[2] = 6; IPIV[2] = 6-1;
        IPIVRef[3] = 5; IPIV[3] = 5-1;
        IPIVRef[4] = 6; IPIV[4] = 6-1;
        IPIVRef[5] = 6; IPIV[5] = 6-1;

        // CALL LAPACK
        dlaswp_(&N, ARef, &M, &K1, &K2, IPIVRef, &INCX);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dlaswp(N, A, M, K1-1, K2-1, IPIV, INCX);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*N, A, ARef, 1.0e-12) == -1)
        {
            dPrint2D("A0", M, N, A0);
            iPrint2D("IPIVRef", 1, sizeIPIV, IPIVRef);
            iPrint2D("IPIV", 1, sizeIPIV, IPIV);
            dPrint2D("ARef", M, N, ARef);
            dPrint2D("A", M, N, A);

            ErrMsg("LAPACK", "DLASWP");
        }
    }

    // DGETF2
    {
        int M = 4;
        int N = 6;

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
        amrex::DG::linalg::dgetf2(M, N, A, M, IPIV, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*N, A, ARef, 1.0e-12) == -1)
        {
            dPrint2D("A0", M, N, A0);
            dPrint2D("ARef", M, N, ARef);
            dPrint2D("A", M, N, A);

            ErrMsg("LAPACK", "DGETF2");
        }
    }

    // DGETRS
    {
        int M = 6;
        int N = 4;

        char TRANS = 'N';
        double A0[M*M], A[M*M], ARef[M*M], X[M*N], XRef[M*N], X0[M*N];
        int IPIVRef[M], IPIV[M];
        int INFORef, INFO;

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
        dgetrf_(&M, &M, ARef, &M, IPIVRef, &INFORef);
        dgetrs_(&TRANS, &M, &N, ARef, &M, IPIVRef, XRef, &M, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dgetrf(M, M, A, M, IPIV, INFO);
        amrex::DG::linalg::dgetrs(TRANS, M, N, A, M, IPIV, X, M, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
        {
            iPrint2D("IPIVRef", 1, M, IPIVRef);
            iPrint2D("IPIV", 1, M, IPIV);

            dPrint2D("ARef", M, M, ARef);
            dPrint2D("A", M, M, A);

            ErrMsg("LAPACK", "DGETRF of DGETRS");
        }
        
        if (dCompare(M*N, X, XRef, 1.0e-12) == -1)
        {
            std::cout << "INFORef: " << INFORef << std::endl;
            std::cout << "INFO: " << INFO << std::endl;
            iPrint2D("IPIVRef", 1, M, IPIVRef);
            iPrint2D("IPIV", 1, M, IPIV);

            dPrint2D("A0", M, M, A0);
            
            dPrint2D("X0", M, N, X0);

            dPrint2D("XRef", M, N, XRef);
            dPrint2D("X", M, N, X);

            ErrMsg("LAPACK", "DGETRS");
        }
    }

    // DGETRI
    {
        int M = 3;

        double A0[M*M], A[M*M], ARef[M*M];
        int IPIV[M], IPIVRef[M];
        int INFORef, INFO;
        int LWORK = M*M;
        double WORK[LWORK];

        // INIT
        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
            A0[m+M*n] = (m == n) ? m+1 : 0.0;// distr(gen);
            A[m+n*M] = A0[m+M*n];
            ARef[m+n*M] = A0[m+M*n];
        }

        // CALL LAPACK
        dgetrf_(&M, &M, ARef, &M, IPIVRef, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dgetrf(M, M, A, M, IPIV, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
        {
            std::cout << "INFORef: " << INFORef << std::endl;
            std::cout << "INFO: " << INFO << std::endl;
            iPrint2D("IPIVRef", 1, M, IPIVRef);
            iPrint2D("IPIV", 1, M, IPIV);
            dPrint2D("ARef", M, M, ARef);
            dPrint2D("A", M, M, A);

            ErrMsg("LAPACK", "DGETRF of DGETRI");
        }

        // CALL LAPACK
        dgetri_(&M, ARef, &M, IPIVRef, WORK, &LWORK, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dgetri(M, A, M, IPIV, WORK, LWORK, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
        {
            std::cout << "INFORef: " << INFORef << std::endl;
            std::cout << "INFO: " << INFO << std::endl;
            iPrint2D("IPIVRef", 1, M, IPIVRef);
            iPrint2D("IPIV", 1, M, IPIV);
            dPrint2D("ARef", M, M, ARef);
            dPrint2D("A", M, M, A);

            ErrMsg("LAPACK", "DGETRI");
        }
    }
    
    // DPOTRF
    {
        int M = 25;

        double A0[M*M], A[M*M], ARef[M*M];
        char UPLO = 'L';
        int INFORef, INFO;

        // INIT
        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
            A0[m+M*n] = distr(gen);
        }
        for (int n = 0; n < M; ++n)
        {
            {
                int m = n;
                A0[m+M*n] *= 100.0;
            }
            for (int m = n+1; m < M; ++m)
            {
                A0[m+M*n] = 0.5*(A0[m+M*n]+A0[n+M*m]);
                A0[n+M*m] = A0[m+M*n];
            }
        }

        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
            A[m+n*M] = A0[m+M*n];
            ARef[m+n*M] = A0[m+M*n];
        }

        // CALL LAPACK
        dpotf2_(&UPLO, &M, ARef, &M, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dpotrf(UPLO, M, A, M, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
        {
            std::cout << "INFORef: " << INFORef << std::endl;
            std::cout << "INFO: " << INFO << std::endl;
            dPrint2D("A0", M, M, A0);
            dPrint2D("ARef", M, M, ARef);
            dPrint2D("A", M, M, A);

            std::cout << "max diff: " << dMaxDiff(M*M, A, ARef) << std::endl;

            ErrMsg("LAPACK", "DPOTRF");
        }   
    }

    // DPOTRS
    {
        int M = 25;
        int N = 4;
        
        double A0[M*M], A[M*M], ARef[M*M], X[M*N], XRef[M*N], X0[M*N];
        char UPLO = 'U';
        int INFORef, INFO;

        // INIT
        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
            A0[m+M*n] = distr(gen);
        }
        for (int n = 0; n < M; ++n)
        {
            {
                int m = n;
                A0[m+M*n] *= 10.0;
            }
            for (int m = n+1; m < M; ++m)
            {
                A0[m+M*n] = 0.5*(A0[m+M*n]+A0[n+M*m]);
                A0[n+M*m] = A0[m+M*n];
            }
        }

        for (int n = 0; n < M; ++n)
        for (int m = 0; m < M; ++m)
        {
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
        dpotrf_(&UPLO, &M, ARef, &M, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dpotrf(UPLO, M, A, M, INFO);

        // COMPUTE THE DIFFERENCE
        if (dCompare(M*M, A, ARef, 1.0e-12) == -1)
        {
            std::cout << "INFORef: " << INFORef << std::endl;
            std::cout << "INFO: " << INFO << std::endl;
            dPrint2D("ARef", M, M, ARef);
            dPrint2D("A", M, M, A);
            std::cout << "max diff: " << dMaxDiff(M*M, A, ARef) << std::endl;

            ErrMsg("LAPACK", "DPOTRF of DPOTRS");
        }

        // CALL LAPACK
        dpotrs_(&UPLO, &M, &N, ARef, &M, XRef, &M, &INFORef);

        // CALL AMREX_LINALG
        amrex::DG::linalg::dpotrs(UPLO, M, N, A, M, X, M, INFO);
        
        if (dCompare(M*N, X, XRef, 1.0e-12) == -1)
        {
            dPrint2D("A0", M, M, A0);
            
            dPrint2D("X0", M, N, X0);

            dPrint2D("XRef", M, N, XRef);
            dPrint2D("X", M, N, X);

            ErrMsg("LAPACK", "DPOTRS");
        }
    }
    // ################################################################
    // ################################################################
}