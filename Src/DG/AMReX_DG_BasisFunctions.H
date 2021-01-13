//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_BasisFunctions.H
 * \brief Routines for computing the basis functions for discontinuous Galerkin methods.
*/

#include <AMReX_REAL.H>

#ifndef AMREX_DG_BASIS_FUNCTIONS_H_
#define AMREX_DG_BASIS_FUNCTIONS_H_

#define DG_BASIS_FUNCTIONS_TAYLOR -1
#define DG_BASIS_FUNCTIONS_LEGENDRE_P 0
#define DG_BASIS_FUNCTIONS_CHEBYSHEV_T 1

#define DG_BASIS_FUNCTIONS 0

namespace amrex
{
namespace DG
{

// ####################################################################
// ONE-DIMENSIONAL BASIS FUNCTIONS ####################################
// ####################################################################
/**
 * \brief Basis functions up to degree L for N points in the interval [xlo,xhi].
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_BF_1d_Table_NxL(const int N, const int L,
                        const T * x, const Real xlo, const Real xhi,
                        T * BF)
{
    // LOCAL PARAMETERS ==========
    const Real dx = (xhi-xlo);
    const Real xm = 0.5*(xhi+xlo);
    const Real J = 2.0/dx;
    // ===========================

#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_TAYLOR)
    // TAYLOR EXPANSION ===============================================
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0)
    {
        for (int n = 0; n < N; ++n) BF[n+N] = (x[n]-xm);
    }
    for (int l = 2; l <= L; ++l)
    {
        const Real il = 1.0/l;
        for (int n = 0; n < N; ++n)
        {
            BF[n+l*N] = il*(x[n]-xm)*BF[n+(l-1)*N];
        }   
    }
    // ================================================================

#elif (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    // LEGENDRE POLYNOMIALS ===========================================
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0) for (int n = 0; n < N; ++n) BF[n+N] = J*(x[n]-xm);
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        BF[n+l*N] = ((2*l-1)*(J*(x[n]-xm))*BF[n+(l-1)*N]-(l-1)*BF[n+(l-2)*N])/l;
    // ================================================================

#else
    Abort("AMReX_DG_BasisFunctions.H - DG_BF_1d_Table_NxL - Unexpected set for basis functions.");
#endif
}

/**
 * \brief Basis functions up to degree L for N points in the interval [xlo,xhi].
 *
 * This functions is the same as DG_BF_1d_Table_NxL but it gets compiled for the host only.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 *
*/
template <typename T>
void DG_BF_1d_Table_NxL__host__(const int N, const int L,
                                const T * x, const Real xlo, const Real xhi,
                                T * BF)
{
    // LOCAL PARAMETERS ==========
    const Real dx = (xhi-xlo);
    const Real xm = 0.5*(xhi+xlo);
    const Real J = 2.0/dx;
    // ===========================

#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_TAYLOR)
    // TAYLOR EXPANSION ===============================================
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0)
    {
        for (int n = 0; n < N; ++n) BF[n+N] = (x[n]-xm);
    }
    for (int l = 2; l <= L; ++l)
    {
        const Real il = 1.0/l;
        for (int n = 0; n < N; ++n)
        {
            BF[n+l*N] = il*(x[n]-xm)*BF[n+(l-1)*N];
        }   
    }
    // ================================================================

#elif (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    // LEGENDRE POLYNOMIALS ===========================================
    for (int n = 0; n < N; ++n) BF[n] = 1.0;
    if (L > 0) for (int n = 0; n < N; ++n) BF[n+N] = J*(x[n]-xm);
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        BF[n+l*N] = ((2*l-1)*(J*(x[n]-xm))*BF[n+(l-1)*N]-(l-1)*BF[n+(l-2)*N])/l;
    // ================================================================
#else
    Abort("AMReX_DG_BasisFunctions.H - DG_BF_1d_Table_NxL__host__ - Unexpected set for basis functions.");
#endif
}

/**
 * \brief Basis functions and derivatives up to degree L for N points in the interval [xlo,xhi].
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 * \param[out] dBF: pointer to memory storing the derivatives of the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_BF_1d_Table_NxL(const int N, const int L,
                        const T * x, const Real xlo, const Real xhi,
                        T * BF, T * dBF)
{
    // LOCAL PARAMETERS ==========
    const Real xm = 0.5*(xhi+xlo);
    const Real J = 2.0/(xhi-xlo);
    // ===========================

#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_TAYLOR)
    // TAYLOR EXPANSION ===============================================
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);
    
    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0)
    {
        for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    }
    for (int l = 2; l <= L; ++l)
    {
        const Real il = 1.0/l;
        for (int n = 0; n < N; ++n)
        {
            dBF[n+l*N] = il*(BF[n+(l-1)*N]+(x[n]-xm)*dBF[n+(l-1)*N]);
        }   
    }
    // ================================================================

#elif (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    // LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES =====================
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);

    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0) for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] = (2*l-1)*BF[n+(l-1)*N]+dBF[n+(l-2)*N];

    for (int l = 1; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] *= J;
    // ================================================================
#else
    Abort("AMReX_DG_BasisFunctions.H - DG_BF_1d_Table_NxL - Unexpected set for basis functions.");
#endif
}

/**
 * \brief Basis functions and derivatives up to degree L for N points in the interval [xlo,xhi].
 *
 * This functions is the same as DG_BF_1d_Table_NxL but it gets compiled for the host only.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 * \param[out] dBF: pointer to memory storing the derivatives of the basis functions;
 *
*/
template <typename T>
void DG_BF_1d_Table_NxL__host__(const int N, const int L,
                                const T * x, const Real xlo, const Real xhi,
                                T * BF, T * dBF)
{
    // LOCAL PARAMETERS ==========
    const Real xm = 0.5*(xhi+xlo);
    const Real J = 2.0/(xhi-xlo);
    // ===========================

#if (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_TAYLOR)
    // TAYLOR EXPANSION ===============================================
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);
    
    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0)
    {
        for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    }
    for (int l = 2; l <= L; ++l)
    {
        const Real il = 1.0/l;
        for (int n = 0; n < N; ++n)
        {
            dBF[n+l*N] = il*(BF[n+(l-1)*N]+(x[n]-xm)*dBF[n+(l-1)*N]);
        }   
    }
    // ================================================================

#elif (DG_BASIS_FUNCTIONS == DG_BASIS_FUNCTIONS_LEGENDRE_P)
    // LEGENDRE POLYNOMIALS AND THEIR DERIVATIVES =====================
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF);

    for (int n = 0; n < N; ++n) dBF[n] = 0.0;
    if (L > 0) for (int n = 0; n < N; ++n) dBF[n+N] = 1.0;
    
    for (int l = 2; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] = (2*l-1)*BF[n+(l-1)*N]+dBF[n+(l-2)*N];

    for (int l = 1; l <= L; ++l)
    for (int n = 0; n < N; ++n)
        dBF[n+l*N] *= J;
    // ================================================================
#else
    Abort("AMReX_DG_BasisFunctions.H - DG_BF_1d_Table_NxL__host__ - Unexpected set for basis functions.");
#endif
}
// ####################################################################
// ####################################################################



// ####################################################################
// SPACE BASIS FUNCTIONS ##############################################
// ####################################################################
/**
 * \brief 1D space basis functions.
 *
 * This function just calls DG_BF_1d_Table_NxL.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL(const int N, const int L,
                              const T * x, const Real xlo, const Real xhi,
                              T * BF)
{
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF);
}

/**
 * \brief 1D space basis functions and derivatives.
 *
 * This function just calls DG_BF_1d_Table_NxL.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x: pointer to memory containing points coordinates;
 * \param[in] xlo, xhi: ends of basis functions's domain;
 * \param[out] BF: pointer to memory storing the basis functions;
 * \param[out] dBF: pointer to memory storing the derivatives of the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL(const int N, const int L,
                              const T * x, const Real xlo, const Real xhi,
                              T * BF, T * dBF)
{
    DG_BF_1d_Table_NxL(N, L, x, xlo, xhi, BF, dBF);
}

/**
 * \brief 2D space basis functions using tensor-products of 1d basis functions.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x1: pointer to memory containing points 1st coordinates;
 * \param[in] x2: pointer to memory containing points 2nd coordinates;
 * \param[in] xlo, xhi: pointers to memory containing the ends of the basis functions' domain;
 * \param[out] BF1_1d, BF2_1d: pointers to memory that store the temporary 1d basis functions;
 * \param[out] BF: pointer to memory storing the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_2d_BF_Table_NxL(const int N, const int L,
                              const T * x1, const T * x2, const Real * xlo, const Real * xhi,
                              T * BF1_1d, T * BF2_1d,
                              T * BF)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
    }
}

/**
 * \brief 2D space basis functions and derivatives using tensor-products of 1d basis functions.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x1: pointer to memory containing points 1st coordinates;
 * \param[in] x2: pointer to memory containing points 2nd coordinates;
 * \param[in] xlo, xhi: pointers to memory containing the ends of the basis functions' domain;
 * \param[out] BF1_1d, BF2_1d: pointers to memory that store the temporary 1d basis functions;
 * \param[out] dBF1_1d, dBF2_1d: pointers to memory that store the temporary 1d basis functions
 *                               derivatives;
 * \param[out] BF: pointer to memory storing the basis functions;
 * \param[out] dBFdx1, dBFdx2: pointer to memory storing the basis functions derivatives;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_2d_BF_Table_NxL(const int N, const int L,
                              const T * x1, const T * x2, const Real * xlo, const Real * xhi,
                              T * BF1_1d, T * dBF1_1d, T * BF2_1d, T * dBF2_1d,
                              T * BF, T * dBFdx1, T * dBFdx2)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2];
    }
}

/**
 * \brief 3D space basis functions using tensor-products of 1d basis functions.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x1: pointer to memory containing points 1st coordinates;
 * \param[in] x2: pointer to memory containing points 2nd coordinates;
 * \param[in] x3: pointer to memory containing points 3rd coordinates;
 * \param[in] xlo, xhi: pointers to memory containing the ends of the basis functions' domain;
 * \param[out] BF1_1d, BF2_1d, BF3_1d: pointers to memory that store the temporary 1d basis functions;
 * \param[out] BF: pointer to memory storing the basis functions;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_3d_BF_Table_NxL(const int N, const int L,
                              const T * x1, const T * x2, const T * x3, const Real * xlo, const Real * xhi,
                              T * BF1_1d, T * BF2_1d, T * BF3_1d,
                              T * BF)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d);
    DG_BF_1d_Table_NxL(N, L, x3, xlo[2], xhi[2], BF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
    }
}

/**
 * \brief 3D space basis functions and derivatives using tensor-products of 1d basis functions.
 *
 * \param[in] N: number of points;
 * \param[in] L: max degree;
 * \param[in] x1: pointer to memory containing points 1st coordinates;
 * \param[in] x2: pointer to memory containing points 2nd coordinates;
 * \param[in] x3: pointer to memory containing points 3rd coordinates;
 * \param[in] xlo, xhi: pointers to memory containing the ends of the basis functions' domain;
 * \param[out] BF1_1d, BF2_1d, BF3_1d: pointers to memory that store the temporary 1d basis
                                       functions;
 * \param[out] dBF1_1d, dBF2_1d, dBF3_1d: pointers to memory that store the temporary 1d basis
 *                                        functions derivatives;
 * \param[out] BF: pointer to memory storing the basis functions;
 * \param[out] dBFdx1, dBFdx2, dBFdx3: pointer to memory storing the basis functions derivatives;
 *
*/
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_3d_BF_Table_NxL(const int N, const int L,
                              const T * x1, const T * x2, const T * x3, const Real * xlo, const Real * xhi,
                              T * BF1_1d, T * dBF1_1d, T * BF2_1d, T * dBF2_1d, T * BF3_1d, T * dBF3_1d,
                              T * BF, T * dBFdx1, T * dBFdx2, T * dBFdx3)
{
    DG_BF_1d_Table_NxL(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);
    DG_BF_1d_Table_NxL(N, L, x3, xlo[2], xhi[2], BF3_1d, dBF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2]*BF3_1d[pos3];
        dBFdx3[pos] = BF1_1d[pos1]*BF2_1d[pos2]*dBF3_1d[pos3];
    }
}

template <typename T>
void DG_space_1d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x, const Real xlo, const Real xhi,
                                      T * BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF);
}
template <typename T>
AMREX_GPU_HOST_DEVICE
void DG_space_1d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x, const Real xlo, const Real xhi,
                                      T * BF, T * dBF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x, xlo, xhi, BF, dBF);
}

template <typename T>
void DG_space_2d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x1, const T * x2, const Real * xlo, const Real * xhi,
                                      T * BF1_1d, T * dBF1_1d, T * BF2_1d, T * dBF2_1d,
                                      T * BF, T * dBFdx1, T * dBFdx2)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2];
    }
}
template <typename T>
void DG_space_2d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x1, const T * x2, const Real * xlo, const Real * xhi,
                                      T * BF1_1d, T * BF2_1d,
                                      T * BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d);

    int pos, pos1, pos2;

    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2];
    }
}

template <typename T>
void DG_space_3d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x1, const T * x2, const T * x3, const Real * xlo, const Real * xhi,
                                      T * BF1_1d, T * dBF1_1d, T * BF2_1d, T * dBF2_1d, T * BF3_1d, T * dBF3_1d,
                                      T * BF, T * dBFdx1, T * dBFdx2, T * dBFdx3)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d, dBF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d, dBF2_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x3, xlo[2], xhi[2], BF3_1d, dBF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx1[pos] = dBF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
        dBFdx2[pos] = BF1_1d[pos1]*dBF2_1d[pos2]*BF3_1d[pos3];
        dBFdx3[pos] = BF1_1d[pos1]*BF2_1d[pos2]*dBF3_1d[pos3];
    }
}

template <typename T>
void DG_space_3d_BF_Table_NxL__host__(const int N, const int L,
                                      const T * x1, const T * x2, const T * x3, const Real * xlo, const Real * xhi,
                                      T * BF1_1d, T * BF2_1d, T * BF3_1d, T * BF)
{
    DG_BF_1d_Table_NxL__host__(N, L, x1, xlo[0], xhi[0], BF1_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x2, xlo[1], xhi[1], BF2_1d);
    DG_BF_1d_Table_NxL__host__(N, L, x3, xlo[2], xhi[2], BF3_1d);

    int pos, pos1, pos2, pos3;

    for (int l3 = 0; l3 <= L; ++l3)
    for (int l2 = 0; l2 <= L; ++l2)
    for (int l1 = 0; l1 <= L; ++l1)
    for (int n = 0; n < N; ++n)
    {
        pos = n+(l1+l2*(1+L)+l3*(1+L)*(1+L))*N;
        pos1 = n+l1*N;
        pos2 = n+l2*N;
        pos3 = n+l3*N;

        BF[pos] = BF1_1d[pos1]*BF2_1d[pos2]*BF3_1d[pos3];
    }
}
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif