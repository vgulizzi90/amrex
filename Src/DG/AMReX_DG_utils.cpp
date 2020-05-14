// AMReX_DG_utils.cpp

#include <AMReX_DG_utils.H>

namespace amrex
{
namespace DG_utils
{

// PRINT-TO-SCREEN ####################################################
void PrintIntArray2D(int Nr, int Nc, const int * a)
{
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c) Print() << a[r+c*Nr] << " ";
        Print() << std::endl;
    }
}

void InlinePrintRealArray(int N, const Real * a)
{
    for (int r = 0; r < N; ++r)
    {
        Print() << std::scientific << std::setprecision(5) << std::setw(12) << a[r] << " ";
    }
}

void PrintRealArray2D(int Nr, int Nc, const Real * a)
{
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c) Print() << std::scientific << std::setprecision(5) << std::setw(12) << a[r+c*Nr] << " ";
        Print() << std::endl;
    }
}

void PrintDiagRealArray2D(int Nr, int Nc, const Real * a)
{
    int N = std::min(Nr, Nc);
    for (int r = 0; r < N; ++r)
    {
        Print() << a[r+r*Nr] << std::endl;
    }
}
// ####################################################################


// SPECIAL ARRAYS #####################################################
AMREX_GPU_HOST_DEVICE
void linspace(const Real& a, const Real& b, const int& n, Real* x)
{
    const Real dx = (b-a)/(n-1);

    for (int k = 0; k < n; ++k) x[k] = a+dx*k;
}
// ####################################################################


// SIMPLE ARRAY OPERATIONS ############################################
AMREX_GPU_HOST_DEVICE
void copy(const int & n, const Real * src, Real * dst)
{
    for (int r = 0; r < n; ++r) dst[r] = src[r];
}

AMREX_GPU_HOST_DEVICE
Real sum(const int& n, const Real* src)
{
    Real res;
    res = 0.0;
    for (int r = 0; r < n; ++r) res += src[r];
    
    return res;
}
// ####################################################################


// ALGEBRA ############################################################
// SPECIAL MATRICES ===================================================
AMREX_GPU_HOST_DEVICE
void eye(const int& n, Real* I)
{
    for (int c = 0; c < n; ++c)
    for (int r = 0; r < n; ++r)
        I[r+c*n] = 0.0;
    
    for (int r = 0; r < n; ++r)
        I[r+r*n] = 1.0;
}

AMREX_GPU_HOST_DEVICE
void ones(const int& Nr, const int& Nc, Real* dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[r+c*Nr] = 1.0;
}
AMREX_GPU_HOST_DEVICE
void negative_ones(const int& Nr, const int& Nc, Real* dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[r+c*Nr] = -1.0;
}
// ====================================================================
// TRANSPOSE ==========================================================
AMREX_GPU_HOST_DEVICE
void transpose(const int & Nr, const int & Nc, const Real * src, Real * dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[c+r*Nc] = src[r+c*Nr];
}
// ====================================================================
// ELEMENT-WISE OPERATIONS ============================================
AMREX_GPU_HOST_DEVICE
void sum(Real * dst, const int & n, const Real * src)
{
    for (int r = 0; r < n; ++r) dst[r] += src[r];
}
AMREX_GPU_HOST_DEVICE
void sum(const int& Nr, const int& Nc, const Real* src1, const Real* src2, Real* dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[r+c*Nr] = src1[r+c*Nr]+src2[r+c*Nr];
}
AMREX_GPU_HOST_DEVICE
void multiply(const int& Nr, const int& Nc, const Real* src1, const Real & src2, Real* dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[r+c*Nr] = src1[r+c*Nr]*src2;
}
AMREX_GPU_HOST_DEVICE
void multiply(const int& Nr, const int& Nc, const Real* src1, const Real* src2, Real* dst)
{
    for (int c = 0; c < Nc; ++c)
    for (int r = 0; r < Nr; ++r)
        dst[r+c*Nr] = src1[r+c*Nr]*src2[r+c*Nr];
}
// ====================================================================
// MATRIX-MATRIX MULTIPLICATION =======================================
AMREX_GPU_HOST_DEVICE
void matmul(const int& Nr, const int& Nc, const int& Nrhs, const Real* A, const Real* B, Real* X)
{
    for (int k = 0; k < Nrhs; ++k)
    for (int i = 0; i < Nr; ++i)
        X[i+k*Nr] = 0.0;

    for (int k = 0; k < Nrhs; ++k)
    for (int i = 0; i < Nr; ++i)
    for (int j = 0; j < Nc; ++j)
        X[i+k*Nr] += A[i+j*Nr]*B[j+k*Nc];
}
// ====================================================================
// MATRIX-MATRIX KRONECKER PRODUCT ====================================
AMREX_GPU_HOST_DEVICE
void kron(const int& Nr1, const int& Nc1, const int& Nr2, const int& Nc2, const Real* src1, const Real* src2, Real* dst)
{
    for (int c1 = 0; c1 < Nc1; ++c1)
    for (int c2 = 0; c2 < Nc2; ++c2)
    for (int r1 = 0; r1 < Nr1; ++r1)
    for (int r2 = 0; r2 < Nr2; ++r2)
    {
        dst[r1*Nr2+r2+(c2+c1*Nc2)*Nr1*Nr2] = src1[r1+c1*Nr1]*src2[r2+c2*Nr2];
    }
}
// ====================================================================
// INVERSE OF A MATRIX ================================================
void matinv(const int & n, const Real * A, Real * invA)
{
    // VARIABLES
    int N = n;
    int info;
    int ipiv[n];
    int lwork = n*n;
    Real work[lwork];

    // INIT
    for (int k = 0; k < n*n; ++k)
    {
        invA[k] = A[k];
    }

    // CALL LAPACK ROUTINES
    linalg::dgetrf_(&N, &N, invA, &N, ipiv, &info);
    linalg::dgetri_(&N, invA, &N, ipiv, work, &lwork, &info);
}
// ====================================================================

// NORM OF A VECTOR ===================================================
AMREX_GPU_HOST_DEVICE
Real norm2(const int & n, const Real * src)
{
    Real tmp = src[0]*src[0];
    for (int i = 1; i < n; ++i) tmp += src[i]*src[i];

    return sqrt(tmp);
}
// ====================================================================

// CHOLESKY DECOMPOSITION OF A POSITIVE-DEFINITE MATRIX ===============
void Cholesky(const int & n, const Real * A, Real * Ch)
{
    Real work[n*n];
    Cholesky(n, A, Ch, work);
}

AMREX_GPU_HOST_DEVICE
void Cholesky(const int & n, const Real * A, Real * Ch, Real * work)
{
    // VARIABLES --------------------------------
    Real sum;

    int r, c, k;
    // ------------------------------------------
    // INIT VARIABLES
    for (k = 0; k < n*n; ++k) work[k] = 0.0;
    // ------------------------------------------

    // COMPUTE THE DECOMPOSITION --------------------------------------
    for (r = 0; r < n; ++r)
    for (c = 0; c < (r+1); ++c)
    {
        sum = 0.0;
        for (k = 0; k < c; ++k) sum += work[r+k*n]*work[c+k*n];
            
        work[r+c*n] = (r == c) ? sqrt(A[r+r*n]-sum) : (1.0/work[c+c*n]*(A[r+c*n]-sum));
    }

    for (r = 0; r < n; ++r)
    {
        for (c = 0; c <= r; ++c) Ch[r+c*n] = work[r+c*n];

        for (c = r+1; c < n; ++c) Ch[r+c*n] = 0.0;
    }
    // ----------------------------------------------------------------
}
// ====================================================================

// SOLVE A LINEAR SYSTEM OF EQUATIONS A*x = b GIVEN THE CHOLESKY ======
// DECOMPOSITION OF A =================================================
AMREX_GPU_HOST_DEVICE
void Cholesky_solve(const int& n, const int & n_rhs, const Real * Ch, const Real * b, Real * x)
{
    for (int r = 0; r < n; ++r) x[r] = b[r];
    Cholesky_solve_overwrite(n, n_rhs, Ch, x);
}

AMREX_GPU_HOST_DEVICE
void Cholesky_solve_overwrite(const int & n, const int & n_rhs, const Real * Ch, Real * x)
{
    // VARIABLES
    Real sum;

    for (int rh = 0; rh < n_rhs; ++rh)
    {
        // Solve Ch*y = b
        for (int r = 0; r < n; ++r)
        {
            sum = x[r+rh*n];
            
            for (int l = 0; l < r; ++l) sum -= Ch[r+l*n]*x[l+rh*n];
            
            x[r+rh*n] = sum/Ch[r+r*n];
        }

        // Solve Ch.T*x = y
        for (int r = n-1; r >= 0; --r)
        {
            sum = x[r+rh*n];
            
            for (int l = r+1; l < n; ++l) sum -= Ch[l+r*n]*x[l+rh*n];
            
            x[r+rh*n] = sum/Ch[r+r*n];
        }
    }

}
// ====================================================================

// COMPUTE EIGENVALUES AND EIGENVECTOR OF A REAL SYMMETRIC MATRIX =====
// LAPACK INTERFACES --------------------------------------------------
// GENERAL MATRICES
void eig(char VL_flag, char VR_flag,
         int N, Real * A, int LDA,
         Real * wRe, Real * wIm,
         Real * VL, int LDVL,
         Real * VR, int LDVR,
         Real * WORK, int nWORK,
         int info)
{
    linalg::dgeev_(&VL_flag, &VR_flag, &N, A, &LDA, wRe, wIm, VL, &LDVL, VR, &LDVR, WORK, &nWORK, &info);
}

// SYMMETRIC MATRICES
void eig(char V_flag, char UPLO_flag,
         int N, Real * A, int LDA,
         Real * w,
         Real * WORK, int nWORK,
         int info)
{
    linalg::dsyev_(&V_flag, &UPLO_flag, &N, A, &LDA, w, WORK, &nWORK, &info);
}
// --------------------------------------------------------------------

// GENERAL MATRICES
void eig(const bool & compute_VL, const bool & compute_VR,
         const int & N, const Real * A_,
         Real * wRe, Real * wIm, Real * VL, Real * VR)
{
    // PARAMETERS
    const char VL_flag = compute_VL ? 'V' : 'N';
    const char VR_flag = compute_VR ? 'V' : 'N';
    const int nWORK = 6*N;

    // VARIABLES
    Real A[N*N];
    Real WORK[nWORK];
    int info = 0;

    //transpose(N, N, A_, A);
    for (int k = 0; k < N*N; ++k) A[k] = A_[k];

    eig(VL_flag, VR_flag, N, A, N, wRe, wIm, VL, N, VR, N, WORK, nWORK, info);
}

void eig(const int & N, const Real * A, Real * wRe, Real * wIm)
{
    eig(false, false, N, A, wRe, wIm, nullptr, nullptr);
}

// SYMMETRIC MATRICES
void eig(const bool & compute_V,
         const int & N, const Real * A_,
         Real * w, Real * v)
{
    // PARAMETERS
    const char V_flag = compute_V ? 'V' : 'N';
    const char UPLO_flag = 'U';
    const int nWORK = 6*N;

    // VARIABLES
    Real A[N*N];
    Real WORK[nWORK];
    int info = 0;

    for (int k = 0; k < N*N; ++k) A[k] = A_[k];

    eig(V_flag, UPLO_flag, N, A, N, w, WORK, nWORK, info);
}

void eig(const int & N, const Real * A, Real * w)
{
    eig(false, N, A, w, nullptr);
}
// ====================================================================
// ####################################################################

} // namespace DG_utils
} // namespace amrex