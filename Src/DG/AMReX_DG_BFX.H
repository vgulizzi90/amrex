//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_BFX.H
 * \brief Contains contants and data structures to evaluate the DG solution in time and space.
*/

#ifndef AMREX_DG_BFX_H_
#define AMREX_DG_BFX_H_

#include <AMReX_DG_BasisFunctions.H>

namespace amrex
{
namespace DG
{

/**
 * \brief This struct evaluates the basis functions in space.
 *
*/
template <int N, int DG_MAX_p, int DG_MAX_Np>
struct DG_space_BF
{
    // DATA MEMBERS ===================================================
    const int p, Np;
    const Real * lo;
    const Real * hi;

    Real BF[DG_MAX_Np];
    AMREX_D_TERM
    (
        Real P1[1+DG_MAX_p];,
        Real P2[1+DG_MAX_p];,
        Real P3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dP1[1+DG_MAX_p];,
        Real dP2[1+DG_MAX_p];,
        Real dP3[1+DG_MAX_p];
    )
    AMREX_D_TERM
    (
        Real dBFdx1[DG_MAX_Np];,
        Real dBFdx2[DG_MAX_Np];,
        Real dBFdx3[DG_MAX_Np];
    )
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    DG_space_BF(const int p_):
    p{p_},
    Np{AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))},
    lo{nullptr}, hi{nullptr}
    {}

    AMREX_GPU_HOST_DEVICE
    DG_space_BF(const int p_, const Real * lo_, const Real * hi_):
    p{p_},
    Np{AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))},
    lo{lo_}, hi{hi_}
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_support(const Real * lo_, const Real * hi_)
    {
        this->lo = lo_;
        this->hi = hi_;
    }
    // ================================================================

    // EVAL THE BASIS FUNCTIONS =======================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_BF_table(const Real * x)
    {

#if (AMREX_SPACEDIM == 1)
        DG_space_1d_BF_Table_NxL(1, this->p,
                                 x, this->lo[0], this->hi[0],
                                 this->BF, this->dBFdx1);
#endif
#if (AMREX_SPACEDIM == 2)
        DG_space_2d_BF_Table_NxL(1, this->p,
                                 &x[0], &x[1], this->lo, this->hi,
                                 this->P1, this->dP1, this->P2, this->dP2,
                                 this->BF, this->dBFdx1, this->dBFdx2);
#endif
#if (AMREX_SPACEDIM == 3)
        DG_space_3d_BF_Table_NxL(1, this->p,
                                 &x[0], &x[1], &x[2], this->lo, this->hi,
                                 this->P1, this->dP1, this->P2, this->dP2, this->P3, this->dP3,
                                 this->BF, this->dBFdx1, this->dBFdx2, this->dBFdx3);
#endif
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_BF_only_table(const Real * x)
    {
#if (AMREX_SPACEDIM == 1)
        DG_space_1d_BF_Table_NxL(1, this->p,
                                 x, this->lo[0], this->hi[0],
                                 this->BF);
#endif
#if (AMREX_SPACEDIM == 2)
        DG_space_2d_BF_Table_NxL(1, this->p,
                                 &x[0], &x[1], this->lo, this->hi,
                                 this->P1, this->P2,
                                 this->BF);
#endif
#if (AMREX_SPACEDIM == 3)
        DG_space_3d_BF_Table_NxL(1, this->p,
                                 &x[0], &x[1], &x[2], this->lo, this->hi,
                                 this->P1, this->P2, this->P3,
                                 this->BF);
#endif
    }
    // ================================================================
};

typedef DG_space_BF<AMREX_SPACEDIM, __DG_PHI_MAX_SPACE_p__, __DG_PHI_MAX_SPACE_Np__> DG_PHI_space_BF;
typedef DG_space_BF<AMREX_SPACEDIM, __DG_SOL_MAX_SPACE_p__, __DG_SOL_MAX_SPACE_Np__> DG_SOL_space_BF;

/**
 * \brief This struct is derived from DG_space_BF and allows to compute the solution fields.
 *
*/
template <int N, int DG_MAX_p, int DG_MAX_Np>
struct DG_space_BFX
:
public DG_space_BF<N, DG_MAX_p, DG_MAX_Np>
{
    // DATA MEMBERS ===================================================
    Array4<Real const> const & fab;
    int i, j, k;
    int u;
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    DG_space_BFX(const int p_, const Real * lo_, const Real * hi_,
                 Array4<Real const> const & fab_,
                 const int i_, const int j_, const int k_)
    :
    DG_space_BF<N, DG_MAX_p, DG_MAX_Np>(p_, lo_, hi_),
    fab{fab_},
    i{i_}, j{j_}, k{k_},
    u{-1}
    {}

    AMREX_GPU_HOST_DEVICE
    DG_space_BFX(const int p_,
                 Array4<Real const> const & fab_)
    :
    DG_space_BF<N, DG_MAX_p, DG_MAX_Np>(p_),
    fab{fab_},
    u{-1}
    {}
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_fab_ptrs(const int i_, const int j_, const int k_)
    {
        this->i = i_;
        this->j = j_;
        this->k = k_;
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_u(const int u_)
    {
        this->u = u_;
    }
    // ================================================================

    // EVAL THE SOLUTION AT LOCATION x ================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval_unsafe()
    {
        Real res;

        res = 0.0;
        for (int c = 0; c < (this->Np); ++c)
        {
            res += this->BF[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
        }

        return res;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval(const Real * x)
    {
        this->eval_BF_table(x);

        return this->eval_unsafe();
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_unsafe(const int u_lo, const int u_hi, Real * SOL)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            SOL[this->u] = this->eval_unsafe();
        }
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval(const Real * x, const int u_lo, const int u_hi, Real * SOL)
    {
        this->eval_BF_table(x);

        this->eval_unsafe(u_lo, u_hi, SOL);
    }
    // ================================================================

    // EVAL THE SOLUTION'S GRADIENT AT LOCATION x =====================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad_unsafe(Real * res)
    {
        AMREX_D_TERM
        (
            res[0] = 0.0;,
            res[1] = 0.0;,
            res[2] = 0.0;
        )

        for (int c = 0; c < (this->Np); ++c)
        {
            AMREX_D_TERM
            (
                res[0] += this->dBFdx1[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[1] += this->dBFdx2[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[2] += this->dBFdx3[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
            )
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad(const Real * x, Real * res)
    {
        this->_eval_BF_table_(x);

        this->_eval_grad_unsafe_(res);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad_unsafe(const int u_lo, const int u_hi, Real * SOL_grad)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            this->eval_grad_unsafe(&SOL_grad[(this->u)*AMREX_SPACEDIM]);
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad(const Real * x, const int u_lo, const int u_hi, Real * SOL_grad)
    {
        this->_eval_BF_table_(x);

        this->eval_grad_unsafe(u_lo, u_hi, SOL_grad);
    }
    // ================================================================
};

typedef DG_space_BFX<AMREX_SPACEDIM, __DG_PHI_MAX_SPACE_p__, __DG_PHI_MAX_SPACE_Np__> DG_PHI_space_BFX;
typedef DG_space_BFX<AMREX_SPACEDIM, __DG_SOL_MAX_SPACE_p__, __DG_SOL_MAX_SPACE_Np__> DG_SOL_space_BFX;

} // namespace DG
} // namespace amrex

#endif