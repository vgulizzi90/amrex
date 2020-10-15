//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_LinAlg.H
 * \brief Implementations of linear algebra routines.
*/

#include <AMReX_DG_LinAlg.H>

namespace amrex
{
namespace DG
{
namespace linalg
{
// ####################################################################
// BLAS ROUTINES ######################################################
// ####################################################################


// ####################################################################
// ####################################################################



// ####################################################################
// LAPACK ROUTINES ####################################################
// ####################################################################

/**
 * \brief LU factorization of a general m-by-n matrix a.
 *
 * This routine is taken from LAPACK.
 *
 * The factorization has the form
 * a = P*L*U
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements (lower
 * trapezoidal trapezoidal if m > n), and U is upper triangular (upper trapezoidal if m < n).
 *
 * \param[in] m: number of rows;
 * \param[in] n: number of columns;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] ipiv: integer array with dimension min(m,n) containing the pivot indices;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *                   > 0: if info = k, U(k,k) is exactly zero.
 *
*/
AMREX_GPU_HOST_DEVICE
void dgetf2(const int m, const int n, Real * a, const int lda, int * ipiv, int & info)
{
    // Quick return if possible
    if ((m == 0) || (n == 0)) return;

    const Real one = 1.0, zero = 0.0;
    const int mn = std::min(m, n);

    // Test the input parameters
    info = 0;
    if (m < 0) info = -1;
    else if (n < 0) info = -2;
    else if (lda < std::max(1,m)) info = -4;
    if (info != 0)
    {
        xerbla("dgetf2 ", info);
        return;
    }

    // Compute machine safe minimum
    const Real sfmin = std::numeric_limits<Real>::min();

    for (int j = 0; j < mn; ++j)
    {
        // Find pivot and test for singularity
        const int jp = j+idamax(m-j, &a[j+j*lda], 1);

        ipiv[j] = jp;

        if (a[jp+j*lda] != zero)
        {
            // Apply the interchange to columns 1:n
            if (jp != j) dswap(n, &a[j], lda, &a[jp], lda);

            // Compute elements j+1:m-1 of j-th column
            if (j < m)
            {
                if (fabs(a[j+j*lda]) >= sfmin)
                {
                    dscal(m-j-1, one/a[j+j*lda], &a[j+1+j*lda], 1);
                }
                else
                {
                    for (int i = 0; i < (m-j-1); ++i)
                    {
                        a[j+i+j*lda] /= a[j+j*lda];
                    }
                }
            }
        }
        else if (info == 0)
        {
            info = j;
        }

        if (j < mn)
        {
            // Update trailing submatrix
            dger(m-j-1, n-j-1, -one, &a[j+1+j*lda], 1, &a[j+(j+1)*lda], lda, &a[j+1+(j+1)*lda], lda);
        }
    }
}

/**
 * \brief LU factorization of a general m-by-n matrix a.
 *
 * This routine is taken from LAPACK.
 *
 * Unlike the LAPACK version, we always use the unblocked non-recursive code.
 *
 * The factorization has the form
 * a = P*L*U
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements (lower
 * trapezoidal trapezoidal if m > n), and U is upper triangular (upper trapezoidal if m < n).
 *
 * \param[in] m: number of rows;
 * \param[in] n: number of columns;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] ipiv: integer array with dimension min(m,n) containing the pivot indices;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *                   > 0: if info = k, U(k,k) is exactly zero.
 *
*/
AMREX_GPU_HOST_DEVICE
void dgetrf(const int m, const int n, Real * a, const int lda, int * ipiv, int & info)
{
    dgetf2(m, n, a, lda, ipiv, info);
}

/**
 * \brief Solves a system of linear equations a*x = b or a^T*x = b with a general n-by-n matrix a
 * using the LU factorization computed by dgetrf.
 *
 * This routine is taken from LAPACK.
 *
 * \param[in] trans: specifies the operation to be performed as follows:
 *                   trans == 'N' => a*x = b
 *                   trans == 'T' => a^T*x = b
 * \param[in] n: order of the matrix a;
 * \param[in] nrhs: number of right-hand sides;
 * \param[in] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[in] ipiv: integer array containing the pivot indices from dgetrf;
 * \param[inout] b: real array with dimensions (ldb,nrhs);
 * \param[in] ldb: leading dimension of array b;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *
*/
AMREX_GPU_HOST_DEVICE
void dgetrs(const char trans, const int n, const int nrhs, const Real * a, const int lda,
            const int * ipiv, Real * b, const int ldb, int & info)
{
    // Quick return if possible
    if ((n == 0) || (nrhs == 0)) return;

    const Real one = 1.0;
    const bool notran = (trans == 'N');

    // Test the input parameters
    info = 0;
    if ((!notran) && (trans != 'T')) info = -1;
    else if (n < 0) info = -2;
    else if (nrhs < 0) info = -3;
    else if (lda < std::max(1, n)) info = -5;
    else if (ldb < std::max(1, n)) info = -8;
    if (info != 0)
    {
        xerbla("dgetrs ", -info);
        return;
    }

    // Solve a*x = b
    if (notran)
    {
        // Apply row interchanges to the right hand sides
        dlaswp(nrhs, b, ldb, 0, n-1, ipiv, 1);

        // Solve L*x = b, overwriting b with x
        dtrsm('L', 'L', 'N', 'U', n, nrhs, one, a, lda, b, ldb);

        // Solve U*x = b, overwriting b with x
        dtrsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);

    }
    // Solve a^T*x = b
    else
    {
        // Solve U^T*x = b, overwriting b with x
        dtrsm('L', 'U', 'T', 'N', n, nrhs, one, a, lda, b, ldb);
        
        // Solve L^T*x = b, overwriting b with x
        dtrsm('L', 'L', 'T', 'U', n, nrhs, one, a, lda, b, ldb);

        // Apply row interchanges to the solution vectors
        dlaswp(nrhs, b, ldb, 0, n-1, ipiv, -1);
    }
}

/**
 * \brief Computes the inverse of a matrix using the LU factorization computed by dgetrf
 *
 * This routine is taken from LAPACK.
 *
 * Unlike the LAPACK version, we always use the unblocked code.
 *
 * \param[in] n: order of the matrix a;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[in] ipiv: integer array containing the pivot indices from dgetrf;
 * \param[out] work: real array with dimension max(1,lwork);
 * \param[in] lwork: dimension of the array work;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *                   > 0: if info = k, U(k,k) is exactly zero.
 *
*/
AMREX_GPU_HOST_DEVICE
void dgetri(const int n, Real * a, const int lda, int * ipiv,
            Real * work, const int lwork, int & info)
{
    // Quick return if possible
    if (n == 0) return;

    const Real one = 1.0, zero = 0.0;
    const bool lquery = (lwork == -1);

    int iws, /*jb,*/ jp, ldwork, lwkopt, nb, nbmin/*, nn*/;

    // We are NOT calling ilaenv!
    nb = 1;
    lwkopt = n*nb;
    work[0] = lwkopt;

    // Test the input parameters
    info = 0;
    if (n < 0) info = -1;
    else if (lda < std::max(1,n)) info = -3;
    else if ((lwork < std::max(1,n)) && (!lquery)) info = -6;
    if (info != 0)
    {
        xerbla("dgetri ", -info);
        return;
    }
    else if (lquery)
    {
        return;
    }

    // Form inv(U). If INFO > 0 from dtrtri, then U is singular and the
    // inverse cannot be computed
    dtrtri('U', 'N', n, a, lda, info);
    if (info > 0) return;

    nbmin = 2;
    ldwork = n;
    if ((nb > 1) && (nb < n))
    {
        // We are not supposed to end up in here;
        xerbla("dgetri - to be implemented ", 0);
    }
    else
    {
        iws = n;
    }

    // Solve the equation inv(a)*L = inv(U) for inv(a)
    // Use unblocked code
    if ((nb < nbmin) || (nb >= n))
    {
        for (int j = (n-1); j >= 0; --j)
        {
            // Copy current column of L to WORK and replace with zeros
            for (int i = (j+1); i < n; ++i)
            {
                work[i] = a[i+j*lda];
                a[i+j*lda] = zero;
            }

            // Compute current column of inv(a)
            if (j < n)
            {
                dgemv('N', n, n-j-1, -one, &a[(j+1)*lda], lda, &work[j+1], 1, one, &a[j*lda], 1);
            }
        }
    }
    // Use blocked code
    else
    {
        // We are not supposed to end up in here;
        xerbla("dgetri - to be implemented ", 1);
    }

    // Apply column interchanges
    for (int j = (n-2); j >= 0; --j)
    {
        jp = ipiv[j];
        if (jp != j)
        {
            dswap(n, &a[j*lda], 1, &a[jp*lda], 1);
        }
    }
    work[0] = iws;
}

/**
 * \brief Cholesky factorization of a real symmetric positive definite matrix A
 *
 * This routine is taken from LAPACK.
 *
 * The factorization has the form
 *
 * a = U^T*U, if uplo = 'U',
 *
 * or
 *
 * a = L*L^T, if uplo = 'L'.
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] n: order of the matrix a;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *                   > 0: if info = k, the leading minor of order k is not positive definite,
 *                        and the factorization could not be completed;
 *
*/
AMREX_GPU_HOST_DEVICE
void dpotf2(const char uplo, const int n, Real * a, const int lda, int & info)
{
    // Quick return if possible
    if (n == 0) return;

    const Real one = 1.0, zero = 0.0;
    const bool upper = (uplo == 'U');

    Real ajj;

    // Test the input parameters
    info = 0;
    if ((!upper) && (uplo != 'L')) info = -1;
    else if (n < 0) info = -2;
    else if (lda < std::max(1, n)) info = -4;
    if (info != 0)
    {
        xerbla("dpotf2 ", -info);
        return;
    }

    // Compute the Cholesky factorization a = U^T*U
    if (upper)
    {
        for (int j = 0; j < n; ++j)
        {
            // Compute U(j,j) and test for non-positive-definiteness
            ajj = a[j+j*lda]-ddot(j, &a[j*lda], 1, &a[j*lda], 1);
            if ((ajj <= zero) || std::isnan(ajj))
            {
                a[j+j*lda] = ajj;
                info = j;
                return;
            }
            ajj = std::sqrt(ajj);
            a[j+j*lda] = ajj;

            // Compute elements j+1:n-1 of row j
            if (j < (n-1))
            {
                dgemv('T', j, n-j-1, -one, &a[(j+1)*lda], lda, &a[j*lda], 1, one, &a[j+(j+1)*lda], lda);
                dscal(n-j-1, one/ajj, &a[j+(j+1)*lda], lda);
            }
        }
    }
    // Compute the Cholesky factorization a = L*L^T
    else
    {
        for (int j = 0; j < n; ++j)
        {
            // Compute L(j,j) and test for non-positive-definiteness
            ajj = a[j+j*lda]-ddot(j, &a[j], lda, &a[j], lda);
            if ((ajj <= zero) || std::isnan(ajj))
            {
                a[j+j*lda] = ajj;
                info = j;
                return;
            }
            ajj = std::sqrt(ajj);
            a[j+j*lda] = ajj;

            // Compute elements j+1:n-1 of column j
            if (j < (n-1))
            {
                dgemv('N', n-j-1, j, -one, &a[j+1], lda, &a[j], lda, one, &a[j+1+j*lda], 1);
                dscal(n-j-1, one/ajj, &a[j+1+j*lda], 1);
            }
        }
    }
}

/**
 * \brief Cholesky factorization of a real symmetric positive definite matrix A
 *
 * This routine is taken from LAPACK.
 *
 * Unlike the LAPACK version, we always use the unblocked non-recursive code.
 *
 * The factorization has the form
 *
 * a = U^T*U, if uplo = 'U',
 *
 * or
 *
 * a = L*L^T, if uplo = 'L'.
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] n: order of the matrix a;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *                   > 0: if info = k, the leading minor of order k is not positive definite,
 *                        and the factorization could not be completed;
 *
*/
AMREX_GPU_HOST_DEVICE
void dpotrf(const char uplo, const int n, Real * a, const int lda, int & info)
{
    dpotf2(uplo, n, a, lda, info);
}

/**
 * \brief Solves a system of linear equations a*x = b with a symmetric positive definite matrix a
 * using the Cholesky factorization a = U**T*U or a = L*L**T computed by dpotrf.
 *
 * This routine is taken from LAPACK.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] n: order of the matrix a;
 * \param[in] nrhs: number of right-hand sides;
 * \param[in] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[inout] b: real array with dimensions (ldb,nrhs);
 * \param[in] ldb: leading dimension of array b;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *
*/
AMREX_GPU_HOST_DEVICE
void dpotrs(const char uplo, const int n, const int nrhs,
            const Real * a, const int lda,
            Real * b, const int ldb,
            int & info)
{
    // Quick return if possible
    if ((n == 0) || (nrhs == 0)) return;

    const Real one = 1.0;
    const bool upper = (uplo == 'U');

    // Test the input parameters
    info = 0;
    if ((!upper) && (uplo != 'L')) info = -1;
    else if (n < 0) info = -2;
    else if (nrhs < 0) info = -3;
    else if (lda < std::max(1, n)) info = -5;
    else if (ldb < std::max(1, n)) info = -7;
    if (info != 0)
    {
        xerbla("dpotrs ", -info);
        return;
    }

    // Solve a*x = b where a = U^T*U
    if (upper)
    {
        // Solve U^T*x = b, overwriting b with x
        dtrsm('L', 'U', 'T', 'N', n, nrhs, one, a, lda, b, ldb);

        // Solve U*x = b, overwriting b with x
        dtrsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
    }
    // Solve a*x = b where a = L*L^T
    else
    {
        // Solve L*x = b, overwriting b with x
        dtrsm('L', 'L', 'N', 'N', n, nrhs, one, a, lda, b, ldb);

        // Solve L^T*x = b, overwriting b with x
        dtrsm('L', 'L', 'T', 'N', n, nrhs, one, a, lda, b, ldb);
    }
}
// ####################################################################
// ####################################################################



// ####################################################################
// DERIVED ROUTINES ###################################################
// ####################################################################

// ####################################################################
// ####################################################################

} // namespace linalg
} // namespace DG
} // namespace amrex