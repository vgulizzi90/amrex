//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_LinAlg.H
 * \brief Linear algebra routines.
*/

#include <AMReX_Gpu.H>

#include <AMReX_REAL.H>

#ifndef AMREX_DG_LINALG_H_
#define AMREX_DG_LINALG_H_

#define THIRD (1.0/3.0)

namespace amrex
{
namespace DG
{
namespace linalg
{
// ####################################################################
// BLAS ROUTINES ######################################################
// ####################################################################
/**
 * \brief Error handler.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] s: the routine that called xerbla;
 * \param[int] info: the position of the invalid parameter in the parameter list of the calling
 *                   routine.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void xerbla(const char * s, int info)
{
    printf("On entry to %s, parameter number %d had an illegal value.\n", s, info);
    Abort();
}

/**
 * \brief Finds the index of element having max absolute value.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] n: number of elements in input vector;
 * \param[in] dx: input real array;
 * \param[in] incx: storage spacing between elements of dx;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int idamax(const int n, const Real * dx, const int incx)
{
    int ix;
    Real dmax;
    int res;

    res = -1;
    if ((n < 1) || (incx <= 0)) return res;

    res = 0;
    if (n == 1) return res;

    if (incx == 1)
    {
        dmax = fabs(dx[0]);
        for (int i = 1; i < n; ++i)
        {
            if (fabs(dx[i]) > dmax)
            {
                res = i;
                dmax = fabs(dx[i]);
            }
        }
    }
    else
    {
        ix = 0;
        dmax = fabs(dx[ix]);
        ix += incx;
        for (int i = 1; i < n; ++i)
        {
            if (fabs(dx[ix]) > dmax)
            {
                res = i;
                dmax = fabs(dx[ix]);
            }
            ix += incx;
        }
    }

    return res;
}

/**
 * \brief Interchanges two vectors.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] n: number of element in input vectors;
 * \param[inout] dx: 1st input real array;
 * \param[in] incx: storage spacing between elements of dx;
 * \param[inout] dy: 2nd input real array;
 * \param[in] incy: storage spacing between elements of dy;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dswap(const int n, Real * dx, const int incx, Real * dy, const int incy)
{
    // Quick return if possible
    if (n <= 0) return;

    Real tmp;

    if ((incx == 1) && (incy == 1))
    {
        const int m = n%3;

        if (m != 0)
        {
            for (int i = 0; i < m; ++i)
            {
                tmp = dx[i];
                dx[i] = dy[i];
                dy[i] = tmp;
            }
            if (n <= 3) return;
        }

        for (int i = m; i < n; i += 3)
        {
            tmp = dx[i];
            dx[i] = dy[i];
            dy[i] = tmp;
            tmp = dx[i+1];
            dx[i+1] = dy[i+1];
            dy[i+1] = tmp;
            tmp = dx[i+2];
            dx[i+2] = dy[i+2];
            dy[i+2] = tmp;
        }
    }
    else
    {
        int ix, iy;

        ix = (incx < 0) ? (-n+1)*incx : 0;
        iy = (incy < 0) ? (-n+1)*incy : 0;

        for (int i = 0; i < n; ++i)
        {
            tmp = dx[ix];
            dx[ix] = dy[iy];
            dy[iy] = tmp;
            ix += incx;
            iy += incy;
        }
    }
}

/**
 * \brief Scales a real array by a constant.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] n: number of element in input vector;
 * \param[in] da: scalar used to scale the array;
 * \param[inout] dx: real array;
 * \param[in] incx: storage spacing between elements of dx;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dscal(const int n, const Real da, Real * dx, const int incx)
{
    // Quick return if possible
    if ((n <= 0) || (incx <= 0)) return;

    if (incx == 1)
    {
        const int m = n%5;
        
        if (m != 0)
        {
            for (int i = 0; i < m; ++i)
            {
                dx[i] *= da;
            }
            if (n <= 5) return;
        }

        for (int i = m; i < n; i += 5)
        {
            dx[i] *= da;
            dx[i+1] *= da;
            dx[i+2] *= da;
            dx[i+3] *= da;
            dx[i+4] *= da;
        }
    }
    else
    {
        const int n_inc = n*incx;
        
        for (int i = 0; i < n_inc; i += incx)
        {
            dx[i] *= da;
        }
    }
}

/**
 * \brief Performs the rank 1 operation: a += alpha*x*y^T
 *
 * This routine is taken from BLAS.
 *
 * \param[in] m: number of rows of the matrix a. Must be at least zero;
 * \param[in] n: number of columns of the matrix a. Must be at least zero;
 * \param[in] alpha: multiplying scalar;
 * \param[in] x: real array;
 * \param[in] incx: increment for the elements of x. Must not be zero;
 * \param[in] y: real array;
 * \param[in] incy: increment for the elements of y. Must not be zero;
 * \param[inout] a: Before entry, the leading m by n part of the array a must contain the matrix of
 *                  coefficients. On exit, a is overwritten by the updated matrix;
 * \param[in] lda: first dimension of a;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dger(const int m, const int n,
          const Real alpha, const Real * x, const int incx, const Real * y, const int incy, 
          Real * a, const int lda)
{
    const Real zero = 0.0;

    // Quick return if possible
    if ((m == 0) || (n == 0) || (alpha == zero)) return;

    int info;
    int jy, kx, ix;
    Real tmp;

    // Test the input parameters
    info = 0;
    if (m < 0) info = 1;
    else if (n < 0) info = 2;
    else if (incx == 0) info = 5;
    else if (incy == 0) info = 7;
    else if (lda < std::max(1,m)) info = 9;
    if (info != 0)
    {
        xerbla("dger ", info);
        return;
    }

    jy = (incy > 0) ? 0 : -(n-1)*incy;

    if (incx == 1)
    {
        for (int j = 0; j < n; ++j)
        {
            if (y[jy] != zero)
            {
                tmp = alpha*y[jy];

                for (int i = 0; i < m; ++i)
                {
                    a[i+j*lda] += x[i]*tmp;
                }
            }
            jy += incy;
        }
    }
    else
    {
        kx = (incx > 0) ? 0 : -(m-1)*incx;

        for (int j = 0; j < n; ++j)
        {
            if (y[jy] != zero)
            {
                tmp = alpha*y[jy];
                ix = kx;

                for (int i = 0; i < m; ++i)
                {
                    a[i+j*lda] += x[ix]*tmp;
                    ix += incx;
                }
            }
            jy += incy;
        }
    }
}

/**
 * \brief Forms the dot product of two vectors
 *
 * This routine is taken from BLAS.
 *
 * \param[in] n: number of elements in input vectors;
 * \param[in] dx: input array;
 * \param[in] incx: storage spacing between elements of dx;
 * \param[in] dy: input array;
 * \param[in] incy: storage spacing between elements of dy;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real ddot(const int n, const Real * dx, const int incx, const Real * dy, const int incy)
{
    int ix, iy, m;
    Real tmp, res;

    res = 0.0;
    tmp = 0.0;

    if (n == 0) return res;

    if ((incx == 1) && (incy == 1))
    {
        m = n%5;

        if (m != 0)
        {
            for (int i = 0; i < m; ++i)
            {
                tmp += dx[i]*dy[i];
            }
            
            if (n < 5)
            {
                res = tmp;
                return tmp;
            }
        }

        for (int i = m; i < n; i += 5)
        {
            tmp += dx[i]*dy[i];
            tmp += dx[i+1]*dy[i+1];
            tmp += dx[i+2]*dy[i+2];
            tmp += dx[i+3]*dy[i+3];
            tmp += dx[i+4]*dy[i+4];
        }
    }
    else
    {
        ix = (incx < 0) ? (-n+1)*incx : 0;
        iy = (incy < 0) ? (-n+1)*incy : 0;

        for (int i = 0; i < n; ++i)
        {
            tmp += dx[ix]*dy[iy];
            ix += incx;
            iy += incy;
        }
    }
    
    res = tmp;

    return res;
}

/**
 * \brief Performs matrix-vector operations.
 *
 * Performs one of the matrix-vector operations
 *
 * x := a*x
 *
 * or
 *
 * x := a^T*x
 *
 * This routine is taken from BLAS.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] trans: specifies the operation to be performed as follows:
 *                   trans == 'N' => x := a*x
 *                   trans == 'T' => x := a^T*x
 * \param[in] diag: specifies how a is read:
 *                  diag == 'N' => a is non-unit triangular;
 *                  diag == 'U' => a is unit triangular;
 * \param[in] n: order of matrix a;
 * \param[in] a: array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[inout] x: input array that will be modified;
 * \param[in] incx: increment for the elements of x. Must not be zero;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dtrmv(const char uplo, const char trans, const char diag, const int n,
           const Real * a, const int lda, Real * x, const int incx)
{
    const Real zero = 0.0;

    // Quick return if possible
    if (n == 0) return;

    const bool nounit = (diag == 'N');

    int info, ix, jx, kx;
    Real tmp;

    // Test the input parameters
    info = 0;
    if ((uplo != 'U') && (uplo != 'L')) info = 1;
    else if ((trans != 'N') && (trans != 'T')) info = 2;
    else if ((diag != 'U') && (diag != 'N')) info = 3;
    else if (n < 0) info = 4;
    else if (lda < std::max(1,n)) info = 6;
    else if (incx == 0) info = 8;
    if (info != 0)
    {
        xerbla("dtrmv ", info);
        return;
    }

    kx = (incx <= 0) ? -(n-1)*incx : 0;
    
    // Start the operations. In this version the elements of a are
    // accessed sequentially with one pass through a.
    
    // Form x := a*x
    if (trans == 'N')
    {
        if (uplo == 'U')
        {
            if (incx == 1)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (x[j] != zero)
                    {
                        tmp = x[j];
                        for (int i = 0; i < j; ++i)
                        {
                            x[i] += tmp*a[i+j*lda];
                        }
                        if (nounit) x[j] *= a[j+j*lda];
                    }
                }
            }
            else
            {
                jx = kx;

                for (int j = 0; j < n; ++j)
                {
                    if (x[jx] != zero)
                    {
                        tmp = x[jx];
                        ix = kx;
                        for (int i = 0; i < j; ++i)
                        {
                            x[ix] += tmp*a[i+j*lda];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= a[j+j*lda];
                    }

                    jx += incx;
                }
            }
        }
        else
        {
            if (incx == 1)
            {
                for (int j = (n-1); j >= 0; --j)
                {
                    if (x[j] != zero)
                    {
                        tmp = x[j];
                        for (int i = (n-1); i > j; --i)
                        {
                            x[i] += tmp*a[i+j*lda];
                        }
                        if (nounit) x[j] *= a[j+j*lda];
                    }
                }
            }
            else
            {
                kx += (n-1)*incx;
                jx = kx;

                for (int j = (n-1); j >= 0; --j)
                {
                    if (x[jx] != zero)
                    {
                        tmp = x[jx];
                        ix = kx;
                        for (int i = (n-1); i > j; --i)
                        {
                            x[ix] += tmp*a[i+j*lda];
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= a[j+j*lda];
                    }

                    jx -= incx;
                }
            }
        }
    }
    // Form x := a^T*x
    else
    {
        if (uplo == 'U')
        {
            if (incx == 1)
            {
                for (int j = (n-1); j >= 0; --j)
                {
                    tmp = x[j];
                    if (nounit) tmp *= a[j+j*lda];

                    for (int i = (j-1); i >= 0; --i)
                    {
                        tmp += a[i+j*lda]*x[i];
                    }

                    x[j] = tmp;
                }
            }
            else
            {
                jx = kx+(n-1)*incx;

                for (int j = (n-1); j >= 0; --j)
                {
                    tmp = x[jx];
                    ix = jx;
                    if (nounit) tmp *= a[j+j*lda];

                    for (int i = (j-1); i >= 0; --i)
                    {
                        ix -= incx;
                        tmp += a[i+j*lda]*x[ix];
                    }

                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
        else
        {
            if (incx == 1)
            {
                for (int j = 0; j < n; ++j)
                {
                    tmp = x[j];
                    if (nounit) tmp *= a[j+j*lda];

                    for (int i = j+1; i < n; ++i)
                    {
                        tmp += a[i+j*lda]*x[i];
                    }

                    x[j] = tmp;
                }
            }
            else
            {
                jx = kx;

                for (int j = 0; j < n; ++j)
                {
                    tmp = x[jx];
                    ix = jx;
                    if (nounit) tmp *= a[j+j*lda];

                    for (int i = j+1; i < n; ++i)
                    {
                        ix += incx;
                        tmp += a[i+j*lda]*x[ix];
                    }

                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

/**
 * \brief Performs matrix-vector operations.
 *
 * Performs one of the matrix-vector operations
 *
 * y := alpha*a*x+beta*y
 *
 * or
 *
 * y := alpha*a^T*x+beta*y
 *
 * This routine is taken from BLAS.
 *
 * \param[in] trans: specifies the operation to be performed as follows:
 *                   trans == 'N' => y := alpha*a*x+beta*y
 *                   trans == 'T' => y := alpha*a^T*x+beta*y
 * \param[in] m: number of rows of matrix a;
 * \param[in] n: number of columns of matrix a;
 * \param[in] alpha: multiplying scalar;
 * \param[in] a: Before entry, the leading m by n part of the array a must contain the matrix of
 *               coefficients;
 * \param[in] lda: leading dimension of array a;
 * \param[in] x: input array;
 * \param[in] incx: increment for the elements of x. Must not be zero;
 * \param[in] beta: multiplying scalar;
 * \param[inout] y: output array;
 * \param[in] incy: increment for the elements of x. Must not be zero;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgemv(const char trans, const int m, const int n,
           const Real alpha,
           const Real * a, const int lda,
           const Real * x, const int incx,
           const Real beta,
           Real * y, const int incy)
{
    const Real one = 1.0, zero = 0.0;

    // Quick return if possible
    if ((m == 0) || (n == 0) || ((alpha == zero) && (beta == one))) return;

    int info;
    int ix, iy, jx, jy, kx, ky, lenx, leny;
    Real tmp;

    // Test the input parameters
    info = 0;
    if ((trans != 'N') && (trans != 'T')) info = 1;
    else if (m < 0) info = 2;
    else if (n < 0)  info = 3;
    else if (lda < std::max(1,m)) info = 6;
    else if (incx == 0) info = 8;
    else if (incy == 0) info = 11;
    if (info != 0)
    {
        xerbla("dgemv ", info);
        return;
    }

    // Set lenx and leny, the lengths of the vectors x and y, and set
    // up the start points in x and y
    if (trans == 'N')
    {
        lenx = n;
        leny = m;
    }
    else
    {
        lenx = m;
        leny = n;
    }
    kx = (incx > 0) ? 0 : -(lenx-1)*incx;
    ky = (incy > 0) ? 0 : -(leny-1)*incy;

    // Start the operations. In this version the elements of a are
    // accessed sequentially with one pass through a

    // First form y := beta*y
    if (beta != one)
    {
        if (incy == 1)
        {
            if (beta == zero)
            {
                for (int i = 0; i < leny; ++i)
                {
                    y[i] = zero;
                }
            }
            else
            {
                for (int i = 0; i < leny; ++i)
                {
                    y[i] *= beta;
                }
            }
        }
        else
        {
            iy = ky;

            if (beta == zero)
            {
                for (int i = 0; i < leny; ++i)
                {
                    y[iy] = zero;
                    iy += incy;
                }
            }
            else
            {
                for (int i = 0; i < leny; ++i)
                {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
    }

    if (alpha == zero) return;

    // Form y := alpha*a*x+y
    if (trans == 'N')
    {
        jx = kx;

        if (incy == 1)
        {
            for (int j = 0; j < n; ++j)
            {
                tmp = alpha*x[jx];

                for (int i = 0; i < m; ++i)
                {
                    y[i] += tmp*a[i+j*lda];
                }

                jx += incx;
            }
        }
        else
        {
            for (int j = 0; j < n; ++j)
            {
                tmp = alpha*x[jx];
                
                iy = ky;
                
                for (int i = 0; i < m; ++i)
                {
                    y[iy] += tmp*a[i+j*lda];
                    iy += incy;
                }

                jx += incx;
            }
        }
    }
    // Form y := alpha*a^T*x+y
    else
    {
        jy = ky;

        if (incx == 1)
        {
            for (int j = 0; j < n; ++j)
            {
                tmp = zero;

                for (int i = 0; i < m; ++i)
                {
                    tmp += a[i+j*lda]*x[i];
                }

                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
        else
        {
            for (int j = 0; j < n; ++j)
            {
                tmp = zero;

                ix = kx;

                for (int i = 0; i < m; ++i)
                {
                    tmp += a[i+j*lda]*x[ix];
                    ix += incx;
                }

                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
    }
}

/**
 * \brief Performs matrix-matrix operations.
 *
 * Performs one of the matrix-matrix operations
 *
 * c := alpha*op(a)*op(b)+beta*c
 *
 * where op(a) is one of a or a^T.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] transa: specifies the operation to be performed as follows:
 *                    transa == 'N' => op(a) = a
 *                    transa == 'T' => op(a) = a^T
 * \param[in] transb: specifies the operation to be performed as follows:
 *                    transb == 'N' => op(b) = b
 *                    transb == 'T' => op(b) = b^T
 * \param[in] m: number of rows of the matrix op(a) and the matrix c;
 * \param[in] n: number of columns of the matrix op(b) and of the matrix c;
 * \param[in] k: number of columns of the matrix op(a) and number of rows of the matrix op(b);
 * \param[in] alpha: multiplying scalar;
 * \param[in] a: Before entry with transa == 'N', the leading m by k part of the array a must
 *               contain the matrix a, otherwise the leading k by m part of the array a must contain
 *               the matrix a;
 * \param[in] lda: first dimension of a;
 * \param[in] b: Before entry with transb == 'N', the leading k by n part of the array b must
 *               contain the matrix b, otherwise the leading n by k part of the array b must contain
 *               the matrix b;
 * \param[in] ldb: first dimension of b;
 * \param[in] beta: multiplying scalar;
 * \param[inout] c: output array;
 * \param[in] ldc: first dimension of c;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgemm(const char transa, const char transb, const int m, const int n, const int k,
           const Real alpha, const Real * a, const int lda, 
           const Real * b, const int ldb, const Real beta,
           Real * c, const int ldc)
{
    const Real one = 1.0, zero = 0.0;

    // Quick return if possible
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one))) return;

    const bool nota = (transa == 'N');
    const bool notb = (transb == 'N');

    int info, nrowa, nrowb;
    Real tmp;

    if (nota) 
    {
        nrowa = m;
    }
    else
    {
        nrowa = k;
    }

    if (notb)
    {
        nrowb = k;
    }
    else
    {
        nrowb = n;
    }

    // Test the input parameters
    info = 0;
    if ((!nota) && (transa != 'T')) info = 1;
    else if ((!notb) && (transb != 'T')) info = 2;
    else if (m < 0) info = 3;
    else if (n < 0) info = 4;
    else if (k < 0) info = 5;
    else if (lda < std::max(1,nrowa)) info = 8;
    else if (ldb < std::max(1,nrowb)) info = 10;
    else if (ldc < std::max(1,m)) info = 13;
    if (info != 0)
    {
        xerbla("dgemm ", info);
        return;
    }

    if (alpha == zero)
    {
        if (beta == zero)
        {
            for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
            {
                c[i+j*ldc] = zero;
            }
        }
        else
        {
            for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
            {
                c[i+j*ldc] *= beta;
            }
        }
        return;
    }

    // Start the operations
    if (notb)
    {
        // Form c := alpha*a*b+beta*c
        if (nota)
        {
            for (int j = 0; j < n; ++j)
            {
                if (beta == 0)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] = zero;
                    }
                }
                else if (beta != one)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] *= beta;
                    }
                }

                for (int l = 0; l < k; ++l)
                {
                    tmp = alpha*b[l+j*ldb];
                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] += tmp*a[i+l*lda];
                    }
                }
            }
        }
        // Form c := alpha*a^T*b+beta*c
        else
        {
            for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
            {
                tmp = zero;
                for (int l = 0; l < k; ++l)
                {
                    tmp += a[l+i*lda]*b[l+j*ldb];
                }
                if (beta == zero)
                {
                    c[i+j*ldc] = alpha*tmp;
                }
                else
                {
                    c[i+j*ldc] = alpha*tmp+beta*c[i+j*ldc];
                }
            }
        }
    }
    else
    {
        // Form c := alpha*a*b^T+beta*c
        if (nota)
        {
            for (int j = 0; j < n; ++j)
            {
                if (beta == zero)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] = zero;
                    }
                }
                else if (beta != one)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] *= beta;
                    }
                }

                for (int l = 0; l < k; ++l)
                {
                    tmp = alpha*b[j+l*ldb];

                    for (int i = 0; i < m; ++i)
                    {
                        c[i+j*ldc] += tmp*a[i+l*lda];
                    }
                }
            }
        }
        // Form c := alpha*a^T*b^T+beta*c
        else
        {
            for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
            {
                tmp = zero;
                for (int l = 0; l < k; ++l)
                {
                    tmp += a[l+i*lda]*b[j+l*ldb];
                }

                if (beta == zero)
                {
                    c[i+j*ldc] = alpha*tmp;
                }
                else
                {
                    c[i+j*ldc] = alpha*tmp+beta*c[i+j*ldc];
                }
            }
        }
    }
}

/**
 * \brief Computes the inverse of a real upper or lower triangular matrix a.
 *
 * This routine is taken from BLAS.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] diag: specifies how a is read:
 *                  diag == 'N' => a is non-unit triangular;
 *                  diag == 'U' => a is unit triangular;
 * \param[in] n: order of the matrix;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dtrti2(const char uplo, const char diag, const int n, Real * a, const int lda, int & info)
{
    const Real one = 1.0;

    const bool upper = (uplo == 'U');
    const bool nounit = (diag == 'N');

    Real ajj;

    // Test the input parameters
    info = 0;
    if (!upper && (uplo != 'L')) info = -1;
    else if (!nounit && (diag != 'U')) info = -2;
    else if (n < 0) info = -3;
    else if (lda < std::max(1, n)) info = -5;
    if (info != 0)
    {
        xerbla("dtrti2 ", info);
        return;
    }

    // Compute inverse of upper triangular matrix
    if (upper)
    {
        for (int j = 0; j < n; ++j)
        {
            if (nounit)
            {
                a[j+j*lda] = one/a[j+j*lda];
                ajj = -a[j+j*lda];
            }
            else
            {
                ajj = -one;
            }

            // Compute elements 0:j-1 of j-th column
            dtrmv('U', 'N', diag, j, a, lda, &a[j*lda], 1);
            dscal(j, ajj, &a[j*lda], 1);
        }
    }
    // Compute inverse of lower triangular matrix
    else
    {
        for (int j = (n-1); j >= 0; --j)
        {
            if (nounit)
            {
                a[j+j*lda] = one/a[j+j*lda];
                ajj = -a[j+j*lda];
            }
            else
            {
                ajj = -one;
            }

            if (j < n)
            {
                // Compute elements j+1:n-1 of j-th column
                dtrmv('L', 'N', diag, n-j-1, &a[j+1+(j+1)*lda], lda, &a[j+1+j*lda], 1);
                dscal(n-j-1, ajj, &a[j+1+j*lda], 1);
            }
        }
    }
}

/**
 * \brief Computes the inverse of a real upper or lower triangular matrix a.
 *
 * This routine is taken from BLAS.
 *
 * Unlike the BLAS version, we always use the unblocked code.
 *
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] diag: specifies how a is read:
 *                  diag == 'N' => a is non-unit triangular;
 *                  diag == 'U' => a is unit triangular;
 * \param[in] n: order of the matrix;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[out] info: = 0: successful exit
 *                   < 0: if info = -k, the k-th argument had an illegal value;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dtrtri(const char uplo, const char diag, const int n, Real * a, const int lda, int & info)
{
    dtrti2(uplo, diag, n, a, lda, info);
}

/**
 * \brief Solves a matrix equations
 *
 * This routine is taken from BLAS.
 *
 * The routine solves one of the matrix equations
 *
 * op(a)*x = alpha*b
 *
 * or
 *
 * x*op(a) = alpha*b
 *
 * where alpha is a scalar, x and b are m by n matrices, A is a unit, or non-unit,  upper or lower
 * triangular matrix and op(a) is one of op(a) = a or op(a) = a^T.
 *
 * The matrix x is overwritten on b.
 *
 * \param[in] side: specifies whether op(a) appears on the left or right of x as follows:
 *                  side == 'L' => op(a)*x = alpha*b;
 *                  uplo == 'R' => x*op(a) = alpha*b;
 * \param[in] uplo: specifies how a is read:
 *                  uplo == 'U' => a is upper triangular;
 *                  uplo == 'L' => a is lower triangular;
 * \param[in] trans: specifies op(a) as follows:
 *                   transa == 'N' => op(a) = a
 *                   transa == 'T' => op(a) = a^T
 * \param[in] diag: specifies how a is read:
 *                  diag == 'N' => a is non-unit triangular;
 *                  diag == 'U' => a is unit triangular;
 * \param[in] m: number of rows of b;
 * \param[in] n: number of columns of b;
 * \param[in] alpha: multiplying scalar;
 * \param[in] a: input array with dimensions (lda, k), where k is m when side == 'L' and k is n when
                 side == 'R'.
 * \param[in] lda: leading dimension of array a;
 * \param[inout] b: array with dimensions (ldb, n). On exit it is overwritten with the solution.
 * \param[in] ldb: leading dimension of array b;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dtrsm(const char side, const char uplo, const char trans, const char diag, 
           const int m, const int n, const Real alpha, const Real * a, const int lda,
           Real * b, const int ldb)
{
    // Quick return if possible
    if ((m == 0) || (n == 0)) return;

    const Real one = 1.0, zero = 0.0;

    const bool lside = (side == 'L');
    const bool nounit = (diag == 'N');
    const bool upper = (uplo == 'U');
    const int nrowa = lside ? m : n;

    int info;
    Real tmp;

    // Test the input parameters
    info = 0;
    if ((!lside) && (side != 'R')) info = 1;
    else if ((!upper) && (uplo != 'L')) info = 2;
    else if ((trans != 'N') && (trans != 'T')) info = 3;
    else if ((diag != 'N') && (diag != 'U')) info = 4;
    else if (m < 0) info = 5;
    else if (n < 0) info = 6;
    else if (lda < std::max(1,nrowa)) info = 9;
    else if (ldb < std::max(1,m)) info = 11;
    if (info != 0)
    {
        xerbla("dtrsm ", info);
        return;
    }

    if (alpha == zero)
    {
        for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
        {
            b[i+j*ldb] = zero;
        }
    }

    // Start the operations
    
    if (lside)
    {
        // Form b := alpha*inv(a)*b
        if (trans == 'N')
        {
            if (upper)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= alpha;
                        }
                    }

                    for (int k = (m-1); k >= 0; --k)
                    {
                        if (b[k+j*ldb] != zero)
                        {
                            if (nounit) b[k+j*ldb] = b[k+j*ldb]/a[k+k*lda];
                            for (int i = 0; i < k; ++i)
                            {
                                b[i+j*ldb] -= b[k+j*ldb]*a[i+k*lda];
                            }
                        }
                    }
                }
            }
            else
            {
                for (int j = 0; j < n; ++j)
                {
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= alpha;
                        }
                    }

                    for (int k = 0; k < m; ++k)
                    {
                        if (b[k+j*ldb] != zero)
                        {
                            if (nounit) b[k+j*ldb] = b[k+j*ldb]/a[k+k*lda];
                            for (int i = (k+1); i < m; ++i)
                            {
                                b[i+j*ldb] -= b[k+j*ldb]*a[i+k*lda];
                            }
                        }
                    }
                }
            }
        }
        // Form b := alpha*inv(a^T)*b
        else
        {
            if (upper)
            {
                for (int j = 0; j < n; ++j)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        tmp = alpha*b[i+j*ldb];
                        for (int k = 0; k < i; ++k)
                        {
                            tmp -= a[k+i*lda]*b[k+j*ldb];
                        }
                        if (nounit) tmp = tmp/a[i+i*lda];
                        b[i+j*ldb] = tmp;
                    }
                }
            }
            else
            {
                for (int j = 0; j < n; ++j)
                {
                    for (int i = (m-1); i >= 0; --i)
                    {
                        tmp = alpha*b[i+j*ldb];
                        for (int k = (i+1); k < m; ++k)
                        {
                            tmp -= a[k+i*lda]*b[k+j*ldb];
                        }
                        if (nounit) tmp = tmp/a[i+i*lda];
                        b[i+j*ldb] = tmp;
                    }
                }
            }
        }
    }
    else
    {
        // Form b := alpha*b*inv(a)
        if (trans == 'N')
        {
            if (upper)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= alpha;
                        }
                    }
                    for (int k = 0; k < j; ++k)
                    {
                        if (a[k+j*lda] != zero)
                        {
                            for (int i = 0; i < m; ++i)
                            {
                                b[i+j*ldb] -= a[k+j*lda]*b[i+k*ldb];
                            }
                        }
                    }
                    if (nounit)
                    {
                        tmp = one/a[j+j*lda];
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= tmp;
                        }
                    }
                }
            }
            else
            {
                for (int j = (n-1); j >= 0; --j)
                {
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= alpha;
                        }
                    }
                    for (int k = (j+1); k < n; ++k)
                    {
                        if (a[k+j*lda] != zero)
                        {
                            for (int i = 0; i < m; ++i)
                            {
                                b[i+j*ldb] -= a[k+j*lda]*b[i+k*ldb];
                            }
                        }
                    }
                    if (nounit)
                    {
                        tmp = one/a[j+j*lda];
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+j*ldb] *= tmp;
                        }
                    }
                }
            }
        }
        // Form b := alpha*b*inv(a^T)
        else
        {
            if (upper)
            {
                for (int k = (n-1); k >= 0; --k)
                {
                    if (nounit)
                    {
                        tmp = one/a[k+k*lda];
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+k*ldb] *= tmp;
                        }
                    }
                    for (int j = 0; j < (k-1); ++j)
                    {
                        if (a[j+k*lda] != zero)
                        {
                            tmp = a[j+k*lda];
                            for (int i = 0; i < m; ++i)
                            {
                                b[i+j*ldb] -= tmp*b[i+k*ldb];
                            }
                        }
                    }
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+k*ldb] *= alpha;
                        }
                    }
                }
            }
            else
            {
                for (int k = 0; k < n; ++k)
                {
                    if (nounit)
                    {
                        tmp = one/a[k+k*lda];
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+k*ldb] *= tmp;
                        }
                    }
                    for (int j = (k+1); j < n; ++j)
                    {
                        if (a[j+k*lda] != zero)
                        {
                            tmp = a[j+k*lda];
                            for (int i = 0; i < m; ++i)
                            {
                                b[i+j*ldb] -= tmp*b[i+k*ldb];
                            }
                        }
                    }
                    if (alpha != one)
                    {
                        for (int i = 0; i < m; ++i)
                        {
                            b[i+k*ldb] *= alpha;
                        }
                    }
                }
            }
        }
    }
}
// ####################################################################
// ####################################################################



// ####################################################################
// LAPACK ROUTINES ####################################################
// ####################################################################
/**
 * \brief Row interchanges on a general rectangular matrix.
 *
 * This routine is taken from LAPACK.
 *
 * \param[in] n: number of columns of matrix a;
 * \param[inout] a: real array with dimensions (lda,n);
 * \param[in] lda: leading dimension of array a;
 * \param[in] k1: The first element of ipiv for which a row interchange will be done;
 * \param[in] k2: (k2-k1) is the number of elements of ipiv for which a row interchange will be
 *                done;
 * \param[in] ipiv: integer array with dimension (k1+(k2-k1)*abs(incx)) containing the pivot
 *                  indices;
 * \param[in] incx: increment between successive values of ipiv. If incx is negative, the pivots
 *                  are applied in reverse order;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dlaswp(const int n, Real * a, const int lda,
            const int k1, const int k2, const int * ipiv, const int incx)
{
    int i1, i2, inc, ip, ix, ix0, n32;
    Real tmp;

    // Interchange row i with row ipiv(k1+(i-k1)*abs(incx)) for each of
    // rows k1 through k2
    if (incx > 0)
    {
        ix0 = k1;
        i1 = k1;
        i2 = k2;
        inc = 1;
    }
    else if (incx < 0)
    {
        ix0 = k1+(k1-k2)*incx;
        i1 = k2;
        i2 = k1;
        inc = -1;
    }
    else
    {
        return;
    }

    n32 = (n/32)*32;
    if (n32 != 0)
    {
        for (int j = 0; j < n32; j += 32)
        {
            ix = ix0;
            for (int i = i1; i < i2; i += inc)
            {
                ip = ipiv[ix];
                if (ip != i)
                {
                    for (int k = j; k < (j+32); ++k)
                    {
                        tmp = a[i+k*lda];
                        a[i+k*lda] = a[ip+k*lda];
                        a[ip+k*lda] = tmp;
                    }
                }
                ix += incx;
            }
        }
    }
    if (n32 != n)
    {
        ix = ix0;
        for (int i = i1; (incx < 0) ? (i >= i2) : (i < i2); i += inc)
        {
            ip = ipiv[ix];
            if (ip != i)
            {
                for (int k = n32; k < n; ++k)
                {
                    tmp = a[i+k*lda];
                    a[i+k*lda] = a[ip+k*lda];
                    a[ip+k*lda] = tmp;
                }
            }
            ix += incx;
        }
    }
}

AMREX_GPU_HOST_DEVICE
void dgetf2(const int m, const int n, Real * a, const int lda, int * ipiv, int & info);

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
 * This is the recursive version of the algorithm.
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
/*
AMREX_GPU_HOST_DEVICE
void dgetrf2(const int m, const int n, Real * a, const int lda, int * ipiv, int & info)
{
    xerbla("dgetrf2 - still not working! ", 0);
}
*/

AMREX_GPU_HOST_DEVICE
void dgetrf(const int m, const int n, Real * a, const int lda, int * ipiv, int & info);

AMREX_GPU_HOST_DEVICE
void dgetrs(const char trans, const int n, const int nrhs, const Real * a, const int lda,
            const int * ipiv, Real * b, const int ldb, int & info);

AMREX_GPU_HOST_DEVICE
void dgetri(const int n, Real * a, const int lda, int * ipiv,
            Real * work, const int lwork, int & info);

AMREX_GPU_HOST_DEVICE
void dpotf2(const char uplo, const int n, Real * a, const int lda, int & info);

AMREX_GPU_HOST_DEVICE
void dpotrf(const char uplo, const int n, Real * a, const int lda, int & info);

AMREX_GPU_HOST_DEVICE
void dpotrs(const char uplo, const int n, const int nrhs,
            const Real * a, const int lda,
            Real * b, const int ldb,
            int & info);
// ####################################################################
// ####################################################################



// ####################################################################
// AUXILIARY ROUTINES #################################################
// ####################################################################
/**
 * \brief Fill a buffer with zeros.
 *
 * \param[in] m: length of the buffer;
 * \param[in] a: Pointer to memory containing the values to be filled;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void zeros(const int m, Real * a)
{
    for (int i = 0; i < m; ++i)
    {
        a[i] = 0.0;
    }
}

/**
 * \brief Fill a buffer with entries of an identity matrix.
 *
 * \param[in] m: number of rows (and columns of the matrix);
 * \param[in] a: Pointer to memory containing the values to be filled;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eye(const int m, Real * a)
{
    for (int i = 0; i < m*m; ++i)
    {
        a[i] = 0.0;
    }
    for (int i = 0; i < m; ++i)
    {
        a[i+i*m] = 1.0;
    }
}

/**
 * \brief Check whether a buffer contains not-a-number values.
 *
 * \param[in] m: length of the buffer;
 * \param[in] a: Pointer to memory containing the values to be checked;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool contains_nan(const int m, Real * a)
{
    for (int i = 0; i < m; ++i)
    {
        if (std::isnan(a[i])) return true;
    }
    return false;
}

/**
 * \brief Performs matrix-matrix operations.
 *
 * Performs the matrix-matrix operations
 *
 * c := a*b
 *
 * \param[in] m: number of rows of the matrix a and the matrix c;
 * \param[in] k: number of columns of the matrix a and number of rows of the matrix b;
 * \param[in] n: number of columns of the matrix b and of the matrix c;
 * \param[in] alpha: multiplying scalar;
 * \param[in] a: Pointer to memory containing the matrix a;
 * \param[in] b: Pointer to memory containing the matrix b;
 * \param[out] c: Pointer to memory that will contain the matrix c;
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void matmul(const int m, const int k, const int n, const Real * a, const Real * b, Real * c)
{
    for (int r = 0; r < n; ++r)
    for (int i = 0; i < m; ++i)
    {
        c[i+r*m] = 0.0;
    }

    for (int r = 0; r < n; ++r)
    for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
    {
        c[i+r*m] += a[i+j*m]*b[j+r*k];
    }
}

/**
 * \brief Dot product of two vectors in 3D space.
 * 
 * \param[in] v1: pointer to memory containing the first vector's components.
 * \param[in] v2: pointer to memory containing the second vector's components.
 *
 * \return the dot product between the two vectors.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real dot3(const Real * v1, const Real * v2)
{
    Real res;
    res =  v1[0]*v2[0];
    res += v1[1]*v2[1];
    res += v1[2]*v2[2];

    return res;
}

/**
 * \brief Cross product between two vector in 3D space.
 * 
 * \param[in] v1: pointer to memory containing the first vector's components.
 * \param[in] v2: pointer to memory containing the second vector's components.
 * \param[out] res: pointer to memory that will be filled with the result.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cross3(const Real * v1, const Real * v2, Real * res)
{
    res[0] = v1[1]*v2[2]-v1[2]*v2[1];
    res[1] = v1[2]*v2[0]-v1[0]*v2[2];
    res[2] = v1[0]*v2[1]-v1[1]*v2[0];
}
// ####################################################################
// ####################################################################


// ####################################################################
// ROUTINES FOR 2X2 MATRICES ##########################################
// ####################################################################
/**
 * \brief Solve a 2x2 symmetric linear system with n right-hand sides.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dpo2s(const int n, const Real * a, const Real * b, Real * x)
{
    const Real id = 1.0/(a[0+0*2]*a[1+1*2]-a[0+1*2]*a[0+1*2]);

    for (int c = 0; c < n; ++c)
    {
        x[0+c*2] = ( a[1+1*2]*b[0+c*2]-a[0+1*2]*b[1+c*2])*id;
        x[1+c*2] = (-a[0+1*2]*b[0+c*2]+a[0+0*2]*b[1+c*2])*id;
    }
}

/**
 * \brief Compute eigenvalues and (right) eigenvectors of a 2x2 symmetric matrix.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dsy2ev(const Real * a, Real * lam, Real * ev)
{
    const Real a11 = a[0+0*2];
    const Real a12 = a[0+1*2];
    const Real a22 = a[1+1*2];
    const bool a12_is_zero = (std::abs(a12/a11) < 1.0e-12);
    const bool a11_equals_a22 = (std::abs(a11/a22-1.0) < 1.0e-12);

    Real tmp;

    if (a12_is_zero)
    {
        lam[0] = a11;
        lam[1] = a22;
        ev[0+0*2] = 1.0; ev[0+1*2] = 0.0;
        ev[1+0*2] = 0.0; ev[1+1*2] = 1.0;
    }
    else if (a11_equals_a22)
    {
        lam[0] = a11-a22;
        lam[1] = a11+a22;

        tmp = 0.5*std::sqrt(2.0);
        ev[0+0*2] =  tmp; ev[0+1*2] = tmp;
        ev[1+0*2] = -tmp; ev[1+1*2] = tmp;
    }
    else
    {
        tmp = std::sqrt(4.0*a12*a12+(a11-a22)*(a11-a22));
        lam[0] = 0.5*(a11+a22-tmp);
        lam[1] = 0.5*(a11+a22+tmp);

        //tmp = 0.5*std::atan(2.0*a12/(a11-a22));
        //if (tmp < 0.0) tmp += 0.5*M_PI;
        //ev[0+0*2] = std::cos(tmp); ev[0+1*2] = std::sin(tmp);
        //ev[1+0*2] = -ev[0+1*2];    ev[1+1*2] = ev[0+0*2];

        ev[0+0*2] = (a11-a22-tmp)/(2.0*a12);
        ev[1+0*2] = 1.0;

        tmp = 1.0/std::sqrt(ev[0+0*2]*ev[0+0*2]+ev[1+0*2]*ev[1+0*2]);

        ev[0+0*2] *= tmp;
        ev[1+0*2] *= tmp;

        ev[0+1*2] = -ev[1+0*2];
        ev[1+1*2] = ev[0+0*2];

    }
}
// ####################################################################
// ####################################################################



// ####################################################################
// ROUTINES FOR 3x3 MATRICES ##########################################
// ####################################################################
/**
 * \brief Solve a 3x3 symmetric linear system with n right-hand sides.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dpo3s(const int n, const Real * a, const Real * b, Real * x)
{
    const Real id = 1.0/(    a[0+0*3]*a[1+1*3]*a[2+2*3]+
                         2.0*a[0+1*3]*a[0+2*3]*a[1+2*3]-
                             a[0+2*3]*a[0+2*3]*a[1+1*3]-
                             a[0+0*3]*a[1+2*3]*a[1+2*3]-
                             a[0+1*3]*a[0+1*3]*a[2+2*3]);

    for (int c = 0; c < n; ++c)
    {
        x[0+c*3]  = (a[1+1*3]*a[2+2*3]-a[1+2*3]*a[1+2*3])*b[0+c*3];
        x[1+c*3]  = (a[0+2*3]*a[1+2*3]-a[0+1*3]*a[2+2*3])*b[0+c*3];
        x[2+c*3]  = (a[0+1*3]*a[1+2*3]-a[0+2*3]*a[1+1*3])*b[0+c*3];

        x[0+c*3] += (a[0+2*3]*a[1+2*3]-a[0+1*3]*a[2+2*3])*b[1+c*3];
        x[1+c*3] += (a[0+0*3]*a[2+2*3]-a[0+2*3]*a[0+2*3])*b[1+c*3];
        x[2+c*3] += (a[0+1*3]*a[0+2*3]-a[0+0*3]*a[1+2*3])*b[1+c*3];

        x[0+c*3] += (a[0+1*3]*a[1+2*3]-a[0+2*3]*a[1+1*3])*b[2+c*3];
        x[1+c*3] += (a[0+1*3]*a[0+2*3]-a[0+0*3]*a[1+2*3])*b[2+c*3];
        x[2+c*3] += (a[0+0*3]*a[1+1*3]-a[0+1*3]*a[0+1*3])*b[2+c*3];

        x[0+c*3] *= id;
        x[1+c*3] *= id;
        x[2+c*3] *= id;
    }
}

/**
 * \brief Compute eigenvalues and (right) eigenvectors of a 3x3 symmetric matrix.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dsy3ev_old(const Real * a, Real * lam, Real * ev)
{
    const Real a11 = a[0+0*3];
    const Real a12 = a[0+1*3];
    const Real a13 = a[0+2*3];
    const Real a22 = a[1+1*3];
    const Real a23 = a[1+2*3];
    const Real a33 = a[2+2*3];

    const Real q = THIRD*(a11+a22+a33);
    const Real p1 = a12*a12+a13*a13+a23*a23;
    const bool a_is_diagonal = (std::sqrt(p1)/q < 1.0e-12);
    const Real p = std::sqrt(a11*a11+a22*a22+a33*a33+3.0*p1-a22*a33-a11*a22-a11*a33)*THIRD;

    if (a_is_diagonal)
    {
        lam[0] = a11;
        lam[1] = a22;
        lam[2] = a33;
        ev[0+0*3] = 1.0; ev[0+1*3] = 0.0; ev[0+2*3] = 0.0;
        ev[1+0*3] = 0.0; ev[1+1*3] = 1.0; ev[1+2*3] = 0.0;
        ev[2+0*3] = 0.0; ev[2+1*3] = 0.0; ev[2+2*3] = 1.0;
    }
    else
    {
        int i;
        Real tmp;
        Real b[9];

        tmp = 1.0/p;
        b[0+0*3] = (a11-q)*tmp;
        b[0+1*3] = (a12  )*tmp;
        b[0+2*3] = (a13  )*tmp;
        b[1+1*3] = (a22-q)*tmp;
        b[1+2*3] = (a23  )*tmp;
        b[2+2*3] = (a33-q)*tmp;
        
        tmp = 0.5*(    -b[0+2*3]*b[0+2*3]*b[1+1*3]
                   +2.0*b[0+1*3]*b[0+2*3]*b[1+2*3]
                       -b[0+0*3]*b[1+2*3]*b[1+2*3]
                       -b[0+1*3]*b[0+1*3]*b[2+2*3]
                       +b[0+0*3]*b[1+1*3]*b[2+2*3]);

        if (tmp <= -1.0)
        {
            tmp = M_PI*THIRD;
        }
        else if (tmp >= +1.0)
        {
            tmp = 0.0;
        }
        else
        {
            tmp = std::acos(tmp)*THIRD;
        }

        lam[0] = q+2.0*p*std::cos(tmp+2.0*M_PI*THIRD);
        lam[2] = q+2.0*p*std::cos(tmp);
        lam[1] = 3.0*q-lam[0]-lam[2];

        // Compute eigenvector corresponding to lam[0] ----------------
        b[0+0*3] = a11-lam[0]; b[0+1*3] = a12;        b[0+2*3] = a13;
        b[1+0*3] = a12;        b[1+1*3] = a22-lam[0]; b[1+2*3] = a23;
        b[2+0*3] = a13;        b[2+1*3] = a23;        b[2+2*3] = a33-lam[0];
        
        cross3(&b[0*3], &b[1*3], &ev[0*3]);
        cross3(&b[0*3], &b[2*3], &ev[1*3]);
        cross3(&b[1*3], &b[2*3], &ev[2*3]);

        b[0] = dot3(&ev[0*3], &ev[0*3]);
        b[1] = dot3(&ev[1*3], &ev[1*3]);
        b[2] = dot3(&ev[2*3], &ev[2*3]);

        tmp = b[0];

        i = 0;
        if (b[1] > tmp)
        {
            tmp = b[1];
            i = 1;
        }
        if (b[2] > tmp)
        {
            i = 2;
        }

        if (i == 0)
        {
            tmp = 1.0/std::sqrt(b[0]);
            ev[0] *= tmp;
            ev[1] *= tmp;
            ev[2] *= tmp;
        }
        else if (i == 1)
        {
            tmp = 1.0/std::sqrt(b[1]);
            ev[0] = tmp*ev[0+1*3];
            ev[1] = tmp*ev[1+1*3];
            ev[2] = tmp*ev[2+1*3];

        }
        else
        {
            tmp = 1.0/std::sqrt(b[2]);
            ev[0] = tmp*ev[0+2*3];
            ev[1] = tmp*ev[1+2*3];
            ev[2] = tmp*ev[2+2*3];
        }
        // ------------------------------------------------------------

        // Compute eigenvector corresponding to lam[1] ----------------
        if (std::abs(ev[0]) > std::abs(ev[1]))
        {
            tmp = 1.0/std::sqrt(ev[0]*ev[0]+ev[2]*ev[2]);
            ev[0+1*3] = -ev[2]*tmp;
            ev[1+1*3] = 0.0;
            ev[2+1*3] = +ev[0]*tmp;
        }
        else
        {
            tmp = 1.0/std::sqrt(ev[1]*ev[1]+ev[2]*ev[2]);
            ev[0+1*3] = 0.0;
            ev[1+1*3] = +ev[2]*tmp;
            ev[2+1*3] = -ev[1]*tmp;
        }
        cross3(&ev[0*3], &ev[1*3], &ev[2*3]);

        b[0+0*3] = a11*ev[0+1*3]+a12*ev[1+1*3]+a13*ev[2+1*3];
        b[1+0*3] = a12*ev[0+1*3]+a22*ev[1+1*3]+a23*ev[2+1*3];
        b[2+0*3] = a13*ev[0+1*3]+a23*ev[1+1*3]+a33*ev[2+1*3];

        b[0+1*3] = a11*ev[0+2*3]+a12*ev[1+2*3]+a13*ev[2+2*3];
        b[1+1*3] = a12*ev[0+2*3]+a22*ev[1+2*3]+a23*ev[2+2*3];
        b[2+1*3] = a13*ev[0+2*3]+a23*ev[1+2*3]+a33*ev[2+2*3];

        b[0+2*3] = dot3(&ev[1*3], &b[0*3])-lam[1];
        b[1+2*3] = dot3(&ev[1*3], &b[1*3]);
        b[2+2*3] = dot3(&ev[2*3], &b[1*3])-lam[1];

        if (std::abs(b[0+2*3]) >= std::abs(b[2+2*3]))
        {
            tmp = std::max(std::abs(b[0+2*3]), std::abs(b[1+2*3]));

            if (tmp > 0.0)
            {
                if (std::abs(b[0+2*3]) > std::abs(b[1+2*3]))
                {
                    b[1+2*3] /= b[0+2*3];
                    b[0+2*3] = 1.0/std::sqrt(1.0+b[1+2*3]*b[1+2*3]);
                    b[1+2*3] *= b[0+2*3];
                }
                else
                {
                    b[0+2*3] /= b[1+2*3];
                    b[1+2*3] = 1.0/std::sqrt(1.0+b[0+2*3]*b[0+2*3]);
                    b[0+2*3] *= b[1+2*3];
                }
                
                ev[0+1*3] = b[1+2*3]*ev[0+1*3]-b[0+2*3]*ev[0+2*3];
                ev[1+1*3] = b[1+2*3]*ev[1+1*3]-b[0+2*3]*ev[1+2*3];
                ev[2+1*3] = b[1+2*3]*ev[2+1*3]-b[0+2*3]*ev[2+2*3];
            }
            else
            {
                // &ev[1*3] already contains the eigenvector
            }
        }
        else
        {
            tmp = std::max(std::abs(b[2+2*3]), std::abs(b[1+2*3]));

            if (tmp > 0.0)
            {
                if (std::abs(b[2+2*3]) > std::abs(b[1+2*3]))
                {
                    b[1+2*3] /= b[2+2*3];
                    b[2+2*3] = 1.0/std::sqrt(1.0+b[1+2*3]*b[1+2*3]);
                    b[1+2*3] *= b[2+2*3];
                }
                else
                {
                    b[2+2*3] /= b[1+2*3];
                    b[1+2*3] = 1.0/std::sqrt(1.0+b[2+2*3]*b[2+2*3]);
                    b[2+2*3] *= b[1+2*3];
                }

                ev[0+1*3] = b[2+2*3]*ev[0+1*3]-b[1+2*3]*ev[0+2*3];
                ev[1+1*3] = b[2+2*3]*ev[1+1*3]-b[1+2*3]*ev[1+2*3];
                ev[2+1*3] = b[2+2*3]*ev[2+1*3]-b[1+2*3]*ev[2+2*3];
            }
            else
            {
                // &ev[1*3] already contains the eigenvector
            }
        }
        // ------------------------------------------------------------

        // Compute eigenvector corresponding to lam[2] ----------------
        cross3(&ev[0*3], &ev[1*3], &ev[2*3]);
        // ------------------------------------------------------------

        // TRANSPOSE --------------------------------------------------
        //tmp = ev[0+1*3]; ev[0+1*3] = ev[1+0*3]; ev[1+0*3] = tmp;
        //tmp = ev[0+2*3]; ev[0+2*3] = ev[2+0*3]; ev[2+0*3] = tmp;
        //tmp = ev[1+2*3]; ev[1+2*3] = ev[2+1*3]; ev[2+1*3] = tmp;
        // ------------------------------------------------------------
/**/
Print() << "q: " << q << std::endl;
Print() << "p: " << p << std::endl;
Print() << "lam[0]: " << lam[0] << std::endl;
Print() << "lam[1]: " << lam[1] << std::endl;
Print() << "lam[2]: " << lam[2] << std::endl;
Print() << "ev[0]: " << ev[0] << " " << ev[1] << " " << ev[2] << std::endl;
Print() << "ev[1]: " << ev[0+1*3] << " " << ev[1+1*3] << " " << ev[2+1*3] << std::endl;
Print() << "ev[2]: " << ev[0+2*3] << " " << ev[1+2*3] << " " << ev[2+2*3] << std::endl;
/**/
    }
}

/**
 * \brief Compute eigenvalues and (right) eigenvectors of a 3x3 symmetric matrix.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dsy3ev(const Real * a, Real * lam, Real * ev)
{
    const Real a11 = a[0+0*3];
    const Real a12 = a[0+1*3];
    const Real a13 = a[0+2*3];
    const Real a22 = a[1+1*3];
    const Real a23 = a[1+2*3];
    const Real a33 = a[2+2*3];

    const bool a_is_diagonal = (std::sqrt(a12*a12+a13*a13+a23*a23)/(a11+a22+a33) < 1.0e-12);

    if (a_is_diagonal)
    {
        lam[0] = a11;
        lam[1] = a22;
        lam[2] = a33;
        ev[0+0*3] = 1.0; ev[0+1*3] = 0.0; ev[0+2*3] = 0.0;
        ev[1+0*3] = 0.0; ev[1+1*3] = 1.0; ev[1+2*3] = 0.0;
        ev[2+0*3] = 0.0; ev[2+1*3] = 0.0; ev[2+2*3] = 1.0;
    }
    else
    {
        // EIGENVALUES ------------------------------------------------
        {
            Real p, q, ph, tmp;

            // Temporarily store the invariants
            ev[0] = a11+a22+a33;
            ev[1] = a12*a12+a13*a13+a23*a23-a11*a22-a22*a33-a33*a11;
            ev[2] = a11*a22*a33+2.0*a12*a23*a13-a11*a23*a23-a12*a12*a33-a13*a13*a22;

            p = -ev[1]-THIRD*ev[0]*ev[0];
            q = -ev[2]-(2.0/27.0)*ev[0]*ev[0]*ev[0]-THIRD*ev[0]*ev[1];
            
            tmp = std::sqrt(THIRD*std::abs(p));
            ph = tmp*tmp*tmp;
            ph = -0.5*q/ph;
            ph = std::min(1.0, std::max(-1.0, ph));
            ph = std::acos(ph);

            // Eigenvalues
            lam[0] = THIRD*ev[0]-2.0*tmp*std::cos(THIRD*(ph-M_PI));
            lam[1] = THIRD*ev[0]-2.0*tmp*std::cos(THIRD*(ph+M_PI));
            lam[2] = ev[0]-lam[0]-lam[1];
        }
        // ------------------------------------------------------------

        // CHECK WHETHER WE HAVE COINCIDENT EIGENVALUES ---------------
        bool coincident_eigenvalues;
        
        {
            int k;

            {
                const Real tmp = 1.0e-12*std::abs(a11+a22+a33);

                coincident_eigenvalues = false;
                k = 0;

                if (std::abs(lam[0]-lam[1]) < tmp)
                {
                    coincident_eigenvalues = true;
                    k = 2;
                }
                else if (std::abs(lam[0]-lam[2]) < tmp)
                {
                    coincident_eigenvalues = true;
                    k = 1;
                }
                else if (std::abs(lam[1]-lam[2]) < tmp)
                {
                    coincident_eigenvalues = true;
                    k = 0;
                }
            }

            if (coincident_eigenvalues)
            {
                Real tmp;

                if (k != 0)
                {
                    tmp = lam[0];
                    lam[0] = lam[k];
                    lam[k] = tmp;
                }
            }
        }
        // ------------------------------------------------------------

        // EIGENVECTORS -----------------------------------------------
        const int k_max = coincident_eigenvalues ? 1 : 3;
        for (int k = 0; k < k_max; ++k)
        {
            const Real b11 = a11-lam[k], b22 = a22-lam[k], b33 = a33-lam[k];
            Real tmp, c12, c13, c23;
            int i;

            c12 = a12*a12-b11*b22;
            c13 = a13*a13-b11*b33;
            c23 = a23*a23-b22*b33;

            tmp = c12;
            i = 12;
            if (std::abs(c13) > std::abs(tmp))
            {
                tmp = c13;
                i = 13;
            }
            if (std::abs(c23) > std::abs(tmp))
            {
                tmp = c23;
                i = 23;
            }

            if (i == 12)
            {
                tmp = 1.0/c12;
                c13 = (a13*b22-a12*a23)*tmp;
                c23 = (a23*b11-a12*a13)*tmp;
                ev[2+k*3] = std::sqrt(1.0-c13*c13-c23*c23);
                ev[0+k*3] = c13*ev[2+k*3];
                ev[1+k*3] = c23*ev[2+k*3];
            }
            else if (i == 13)
            {
                tmp = 1.0/c13;
                c12 = (a12*b33-a13*a23)*tmp;
                c23 = (a23*b11-a12*a13)*tmp;
                ev[1+k*3] = std::sqrt(1.0-c12*c12-c23*c23);
                ev[0+k*3] = c12*ev[1+k*3];
                ev[2+k*3] = c23*ev[1+k*3];
            }
            else
            {
                tmp = 1.0/c23;
                c12 = (a12*b33-a13*a23)*tmp;
                c13 = (a13*b22-a12*a23)*tmp;
                ev[0+k*3] = std::sqrt(1.0-c12*c12-c13*c13);
                ev[1+k*3] = c12*ev[0+k*3];
                ev[2+k*3] = c13*ev[0+k*3];
            }
        }

        if (coincident_eigenvalues)
        {
            Real tmp;

            if (std::abs(ev[0]) > std::abs(ev[1]))
            {
                tmp = 1.0/std::sqrt(ev[0]*ev[0]+ev[2]*ev[2]);
                ev[0+1*3] = -ev[2]*tmp;
                ev[1+1*3] = 0.0;
                ev[2+1*3] = +ev[0]*tmp;
            }
            else
            {
                tmp = 1.0/std::sqrt(ev[1]*ev[1]+ev[2]*ev[2]);
                ev[0+1*3] = 0.0;
                ev[1+1*3] = +ev[2]*tmp;
                ev[2+1*3] = -ev[1]*tmp;
            }
            cross3(&ev[0*3], &ev[1*3], &ev[2*3]);
        }
        // ------------------------------------------------------------
        
/*
Print() << "lam[0]: " << lam[0] << std::endl;
Print() << "lam[1]: " << lam[1] << std::endl;
Print() << "lam[2]: " << lam[2] << std::endl;
Print() << "ev[0]: " << ev[0] << " " << ev[1] << " " << ev[2] << std::endl;
Print() << "ev[1]: " << ev[0+1*3] << " " << ev[1+1*3] << " " << ev[2+1*3] << std::endl;
Print() << "ev[2]: " << ev[0+2*3] << " " << ev[1+2*3] << " " << ev[2+2*3] << std::endl;
*/
    }
}
// ####################################################################
// ####################################################################

} // namespace linalg
} // namespace DG
} // namespace amrex

#endif