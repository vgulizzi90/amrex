//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_LinAlg.H
 * \brief Linear algebra routines.
*/

#ifndef AMREX_DG_LINALG_H_
#define AMREX_DG_LINALG_H_

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
 * \param[in] a: Before entry with transb == 'N', the leading k by n part of the array b must
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

} // namespace linalg
} // namespace DG
} // namespace amrex

#endif