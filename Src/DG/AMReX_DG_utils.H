// AMReX_DG_utils.H

#ifndef BL_DG_UTILS_H_
#define BL_DG_UTILS_H_

#include <AMReX_MultiFab.H>

#include <AMReX_blas.H>
#include <AMReX_lapack.H>

namespace amrex
{
namespace DG_utils
{

// PRINT-TO-SCREEN ####################################################
void PrintIntArray2D(int Nr, int Nc, const int *);

void InlinePrintRealArray(int N, const Real *);
void PrintRealArray2D(int Nr, int Nc, const Real *);
void PrintDiagRealArray2D(int Nr, int Nc, const Real *);
// ####################################################################

// SPECIAL ARRAYS #####################################################
AMREX_GPU_HOST_DEVICE
void linspace(const Real&, const Real&, const int&, Real*);
// ####################################################################

// SIMPLE ARRAY OPERATIONS ############################################
AMREX_GPU_HOST_DEVICE
void copy(const int & n, const Real * src, Real * dst);

AMREX_GPU_HOST_DEVICE
Real sum(const int& n, const Real* src);
// ####################################################################

// ALGEBRA ############################################################
// MATRIX OPERATIONS ON ARRAYS ========================================
AMREX_GPU_HOST_DEVICE
void eye(const int&, Real*);
AMREX_GPU_HOST_DEVICE
void ones(const int& Nr, const int& Nc, Real* dst);
AMREX_GPU_HOST_DEVICE
void negative_ones(const int& Nr, const int& Nc, Real* dst);

AMREX_GPU_HOST_DEVICE
void transpose(const int & Nr, const int & Nc, const Real * src, Real * dst);

AMREX_GPU_HOST_DEVICE
void sum(Real * dst, const int & n, const Real * src);
AMREX_GPU_HOST_DEVICE
void sum(const int& Nr, const int& Nc, const Real* src1, const Real* src2, Real* dst);
AMREX_GPU_HOST_DEVICE
void multiply(const int& Nr, const int& Nc, const Real* src1, const Real & src2, Real* dst);
AMREX_GPU_HOST_DEVICE
void multiply(const int& Nr, const int& Nc, const Real* src1, const Real* src2, Real* dst);

AMREX_GPU_HOST_DEVICE
void matmul(const int&, const int&, const int&, const Real*, const Real*, Real*);
AMREX_GPU_HOST_DEVICE
void kron(const int& Nr1, const int& Nc1, const int& Nr2, const int& Nc2, const Real* src1, const Real* src2, Real* dst);

void matinv(const int&, const Real*, Real*);

AMREX_GPU_HOST_DEVICE
Real norm2(const int & n, const Real * src);

void Cholesky(const int &, const Real *, Real *);
AMREX_GPU_HOST_DEVICE
void Cholesky(const int &, const Real *, Real *, Real *);

AMREX_GPU_HOST_DEVICE
void Cholesky_solve(const int& n, const int & n_rhs, const Real * Ch, const Real * b, Real * x);
AMREX_GPU_HOST_DEVICE
void Cholesky_solve_overwrite(const int& n, const int & n_rhs, const Real * Ch, Real * x);
// ====================================================================

// EIGENVALUE PROBLEMS ================================================
// LAPACK INTERFACE ---------------------------------------------------
void eig(char VL_flag, char VR_flag, int N, Real * A, int LDA, Real * wRe, Real * wIm, Real * VL, int LDVL, Real * VR, int LDVR, Real * WORK, int nWORK, int info);
void eig(char V_flag, char UPLO_flag, int N, Real * A, int LDA, Real * w, Real * WORK, int nWORK, int info);
// --------------------------------------------------------------------

// GENERAL MATRICES ---------------------------------------------------
void eig(const bool & compute_VL, const bool & compute_VR, const int & N, const Real * A_, Real * wRe, Real * wIm, Real * VL, Real * VR);
void eig(const int & N, const Real * A, Real * wRe, Real * wIm);
// --------------------------------------------------------------------

// SYMMETRIC MATRICES -------------------------------------------------
void eig(const bool & compute_V, const int & N, const Real * A_, Real * w, Real * v);
void eig(const int & N, const Real * A, Real * w);
// --------------------------------------------------------------------
// ====================================================================
// ####################################################################

} // namespace DG_utils
} // namespace amrex

#endif