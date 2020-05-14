#ifndef BL_LAPACK_DLAPY2_H_
#define BL_LAPACK_DLAPY2_H_

/* dlapy2.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "../blas/f2c.h"
#include "../blas/blaswrap.h"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
doublereal dlapy2_(doublereal *x, doublereal *y)
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Local variables */
    doublereal w, z__, xabs, yabs;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary */
/*  overflow. */

/*  Arguments */
/*  ========= */

/*  X       (input) DOUBLE PRECISION */
/*  Y       (input) DOUBLE PRECISION */
/*          X and Y specify the values x and y. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    xabs = std::abs(*x);
    yabs = std::abs(*y);
    w = std::max(xabs,yabs);
    z__ = std::min(xabs,yabs);
    if (z__ == 0.) {
	ret_val = w;
    } else {
/* Computing 2nd power */
	d__1 = z__ / w;
	ret_val = w * std::sqrt(d__1 * d__1 + 1.);
    }
    return ret_val;

/*     End of DLAPY2 */

} /* dlapy2_ */


#endif