// AMReX_blas.H

#ifndef BL_LAPACK_H_
#define BL_LAPACK_H_

namespace amrex
{
namespace linalg
{

// AUXILIARY ROUTINES
#include "dlamch.h"

#include "ieeeck.h"
#include "iparmq.h"

integer ilaenv_(const integer *ispec, const char *name__, const integer *n1, const integer *n2, const integer *n3, const integer *n4);

#include "dlaisnan.h"
#include "disnan.h"
#include "dlaswp.h"
#include "dlapy2.h"
#include "dlabad.h"
#include "dlassq.h"
#include "dlange.h"
#include "iladlc.h"
#include "iladlr.h"
#include "dlarf.h"
#include "dlarfg.h"
#include "dlacpy.h"

int dlahr2_(integer *n, integer *k, integer *nb, doublereal *
	a, integer *lda, doublereal *tau, doublereal *t, integer *ldt, 
	doublereal *y, integer *ldy);

int dlarfb_(const char *side, const char *trans, const char *direct, const char *
	storev, integer *m, integer *n, integer *k, doublereal *v, integer *
	ldv, doublereal *t, integer *ldt, doublereal *c__, integer *ldc, 
	doublereal *work, integer *ldwork);

#include "dlascl.h"
#include "dlartg.h"
#include "dlarft.h"
#include "dlaset.h"
#include "dlasy2.h"
#include "dlarfx.h"
#include "dlanv2.h"

int dlaexc_(logical *wantq, integer *n, doublereal *t, 
	integer *ldt, doublereal *q, integer *ldq, integer *j1, integer *n1, 
	integer *n2, doublereal *work, integer *info);

#include "dlahqr.h"

#include "dladiv.h"
#include "dlaln2.h"
#include "dlae2.h"
#include "dlanst.h"
#include "dlansy.h"
#include "dlaev2.h"
#include "dlasr.h"

int dlatrd_(const char *uplo, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *e, doublereal *tau, doublereal *w, 
	integer *ldw);

// LAPACK
int dgetf2_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

int dgetrf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

int dgetrs_(const char *trans, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info);

int dtrti2_(const char *uplo, const char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);

int dtrtri_(const char *uplo, const char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);
int dgetri_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, doublereal *work, integer *lwork, integer *info);

int dpotf2_(const char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

int dpotrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

int dpotrs_(const char *uplo, const integer *n, const integer *nrhs, 
	const doublereal *a, const integer *lda, doublereal *b, const integer *ldb, integer * info);

int dgebal_(const char *job, integer *n, doublereal *a, integer *
	lda, integer *ilo, integer *ihi, doublereal *scale, integer *info);

int dgehd2_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *info);

int dgehrd_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

int dorg2r_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

int dorgqr_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

int dorghr_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

int dtrexc_(const char *compq, integer *n, doublereal *t, integer *
	ldt, doublereal *q, integer *ldq, integer *ifst, integer *ilst, 
	doublereal *work, integer *info);

int dorm2r_(const char *side, const char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

int dormqr_(const char *side, const char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

int dormhr_(const char *side, const char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, doublereal *a, integer *lda, doublereal *
	tau, doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

int dtrevc_(const char *side, const char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, integer *mm, integer *m, 
	doublereal *work, integer *info);

int dgebak_(const char *job, const char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *scale, integer *m, doublereal *v, integer *
	ldv, integer *info);

int dlasrt_(const char *id, integer *n, doublereal *d__, integer *info);

int dsterf_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

int dorg2l_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

int dorgql_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

int dorgtr_(const char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

int dsteqr_(const char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

int dsytd2_(const char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, integer *info);

int dsytrd_(const char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, doublereal *
	work, integer *lwork, integer *info);

int dsyev_(const char *jobz, const char *uplo, integer *n, doublereal *a, 
	 integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	integer *info);

// AUXILIARY ROUTINES
#include "dlaqr1.h"

int dlaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, doublereal 
	*sr, doublereal *si, doublereal *h__, integer *ldh, integer *iloz, 
	integer *ihiz, doublereal *z__, integer *ldz, doublereal *v, integer *
	ldv, doublereal *u, integer *ldu, integer *nv, doublereal *wv, 
	integer *ldwv, integer *nh, doublereal *wh, integer *ldwh);

int dlaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

int dlaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

int dlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

int dlaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

int dhseqr_(const char *job, const char *compz, integer *n, integer *ilo, 
	 integer *ihi, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *info);

// LAPACK
int dgeev_(const char *jobvl, const char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *wr, doublereal *wi, doublereal *vl, 
	integer *ldvl, doublereal *vr, integer *ldvr, doublereal *work, 
	integer *lwork, integer *info);

}
}

#endif