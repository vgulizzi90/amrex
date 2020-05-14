// AMReX_blas.H

#ifndef BL_BLAS_H_
#define BL_BLAS_H_

#include "f2c.h"
#include "blaswrap.h"

namespace amrex
{
namespace linalg
{

// BLAS LV 1
#include "idamax.h"
#include "lsame.h"
#include "dswap.h"
#include "dscal.h"

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int xerbla_(const char *, integer *) {return 0;}

#include "dger.h"
#include "ddot.h"
#include "drot.h"
#include "dnrm2.h"
#include "daxpy.h"
#include "dcopy.h"

// BLAS LV 2
#include "dgemv.h"
#include "dtrmv.h"
#include "dsyr2.h"
#include "dsymv.h"

// BLAS LV 3
#include "dgemm.h"
#include "dtrmm.h"
#include "dtrsm.h"
#include "dsyrk.h"
#include "dsyr2k.h"

}
}

#endif