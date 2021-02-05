//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_BFX.H
 * \brief Contains contants and data structures to evaluate the DG solution in time and space.
*/

#ifndef AMREX_DG_BFX_H_
#define AMREX_DG_BFX_H_

#include <algoim_quad.hpp>

#include <AMReX_DG_Constants.H>
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
    {
#ifdef AMREX_DEBUG
        this->self_check();
#endif
    }

    AMREX_GPU_HOST_DEVICE
    DG_space_BF(const int p_, const Real * lo_, const Real * hi_):
    p{p_},
    Np{AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))},
    lo{lo_}, hi{hi_}
    {
#ifdef AMREX_DEBUG
        this->self_check();
#endif
    }
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void self_check() const
    {
        if (p > DG_MAX_p)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_DG_BFX.H - DG_space_BF.self_check()\n";
            msg += "| The order of the basis functions exceeds the maximum allowed number.\n";
            msg += "| p       : "+std::to_string(p)+".\n";
            msg += "| DG_MAX_p: "+std::to_string(DG_MAX_p)+".\n";
            Abort(msg);
        }
    }
    
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
 * \brief This struct computes the solution fields using the dG coefficients
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

    // EVAL THE SOLUTION AT x USING BASIS FUNCTIONS UP TO p_tr ORDER ==
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval_unsafe_trunc(const int p_tr)
    {
        const int Np_tr = AMREX_D_PICK((1+p_tr),(1+p_tr)*(1+p_tr),(1+p_tr)*(1+p_tr)*(1+p_tr));

        Real res;

        res = 0.0;
        for (int c = 0; c < Np_tr; ++c)
        {
#if (AMREX_SPACEDIM == 1)
            const int cc = c;
#endif
#if (AMREX_SPACEDIM == 2)
            const int l2 = c/(1+p_tr);
            const int l1 = (c-l2*(1+p_tr));
            const int cc = (l1+l2*(1+this->p));
#endif
#if (AMREX_SPACEDIM == 3)
            const int l3 = c/((1+p_tr)*(1+p_tr));
            const int l2 = (c-l3*(1+p_tr)*(1+p_tr))/(1+p_tr);
            const int l1 = (c-l3*(1+p_tr)*(1+p_tr)-l2*(1+p_tr));
            const int cc = (l1+l2*(1+this->p)+l3*(1+this->p)*(1+this->p));
#endif
            res += this->BF[cc]*this->fab(this->i,this->j,this->k,cc+(this->u)*(this->Np));
        }

        return res;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval_trunc(const int p_tr, const Real * x)
    {
        this->eval_BF_table(x);

        return this->eval_unsafe_trunc(p_tr);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_unsafe_trunc(const int p_tr, const int u_lo, const int u_hi, Real * SOL)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            SOL[this->u] = this->eval_unsafe_trunc(p_tr);
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_trunc(const int p_tr, const Real * x, const int u_lo, const int u_hi, Real * SOL)
    {
        this->eval_BF_table(x);

        this->eval_unsafe_trunc(p_tr, u_lo, u_hi, SOL);
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
        this->eval_BF_table(x);

        this->eval_grad_unsafe(res);
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
        this->eval_BF_table(x);

        this->eval_grad_unsafe(u_lo, u_hi, SOL_grad);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_un(const Real * x, Real * un)
    {
        this->eval_grad(x, un);

#if (AMREX_SPACEDIM == 1)
        const Real tmp = 1.0/std::sqrt(un[0]*un[0]);
#endif
#if (AMREX_SPACEDIM == 2)
        const Real tmp = 1.0/std::sqrt(un[0]*un[0]+un[1]*un[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const Real tmp = 1.0/std::sqrt(un[0]*un[0]+un[1]*un[1]+un[2]*un[2]);
#endif
        AMREX_D_TERM
        (
            un[0] *= tmp;,
            un[1] *= tmp;,
            un[2] *= tmp;
        )
    }
    // ================================================================
};

typedef DG_space_BFX<AMREX_SPACEDIM, __DG_PHI_MAX_SPACE_p__, __DG_PHI_MAX_SPACE_Np__> DG_PHI_space_BFX;
typedef DG_space_BFX<AMREX_SPACEDIM, __DG_SOL_MAX_SPACE_p__, __DG_SOL_MAX_SPACE_Np__> DG_SOL_space_BFX;


/**
 * \brief This struct computes the linearly reconstructed solution fields.
 *
*/
template <int N, int DG_MAX_p, int DG_MAX_Np>
struct DG_space_RX
{
    // DATA MEMBERS ===================================================
    Array4<Real const> const & fab;
    Array4<Real const> const & slopes_fab;
    const Real * xc;
    int i, j, k;
    int u;
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    DG_space_RX(Array4<Real const> const & fab_, Array4<Real const> const & slopes_fab_,
                const Real * xc_,
                const int i_, const int j_, const int k_)
    :
    fab{fab_},
    slopes_fab{slopes_fab_},
    xc{xc_},
    i{i_}, j{j_}, k{k_},
    u{-1}
    {}
    // ================================================================

    // MISC ===========================================================
    // ================================================================

    // EVAL THE SOLUTION AT LOCATION x ================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval(const Real * x)
    {
        Real res;

        res = this->fab(this->i,this->j,this->k,this->u);
        AMREX_D_TERM
        (
            res += this->slopes_fab(this->i,this->j,this->k,0+AMREX_SPACEDIM*(this->u))*(x[0]-this->xc[0]);,
            res += this->slopes_fab(this->i,this->j,this->k,1+AMREX_SPACEDIM*(this->u))*(x[1]-this->xc[1]);,
            res += this->slopes_fab(this->i,this->j,this->k,2+AMREX_SPACEDIM*(this->u))*(x[2]-this->xc[2]);
        )

        return res;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval(const Real * x, const int u_lo, const int u_hi, Real * SOL)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            SOL[this->u] = this->eval(x);
        }
    }
    // ================================================================
};

typedef DG_space_RX<AMREX_SPACEDIM, __DG_SOL_MAX_SPACE_p__, __DG_SOL_MAX_SPACE_Np__> DG_SOL_space_RX;


/**
 * \brief This is an implementation of DG_space_BFX for algoim.
 *
*/
template <int N, int DG_MAX_p, int DG_MAX_Np>
struct algoim_DG_space_BFX
{
    // DATA MEMBERS ===================================================
    const int p, Np;
    const Real * lo;
    const Real * hi;
    Array4<Real const> const & fab;
    int i, j, k;
    int u;
    Real sgn;
    // ================================================================

    // CONSTRUCTORS ===================================================
    algoim_DG_space_BFX(const int p_, const Real * lo_, const Real * hi_,
                        Array4<Real const> const & fab_,
                        const int i_, const int j_, const int k_)
    :
    p{p_},
    Np{AMREX_D_PICK((1+p_),(1+p_)*(1+p_),(1+p_)*(1+p_)*(1+p_))},
    lo{lo_}, hi{hi_},
    fab{fab_},
    i{i_}, j{j_}, k{k_},
    u{-1},
    sgn{-1.0}
    {
#ifdef AMREX_DEBUG
        this->self_check();
#endif
    }
    // ================================================================

    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void self_check() const
    {
        if (p > DG_MAX_p)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_DG_BFX.H - algoim_DG_space_BFX.self_check()\n";
            msg += "| The order of the basis functions exceeds the maximum allowed number.\n";
            msg += "| p       : "+std::to_string(p)+".\n";
            msg += "| DG_MAX_p: "+std::to_string(DG_MAX_p)+".\n";
            Abort(msg);
        }
    }

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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_sign(const int sgn_)
    {
        this->sgn = sgn_;
    }
    // ================================================================

    // METHODS REQUIRED BY ALGOIM =====================================
    template <typename T>
    T operator() (const GpuArray<T, N> & x) const
    {
        T T_BF[DG_MAX_Np];
        AMREX_D_TERM
        (
            ,
            T T_P1[1+DG_MAX_p];
            T T_P2[1+DG_MAX_p];,
            T T_P3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            ,
            T T_dP1[1+DG_MAX_p];
            T T_dP2[1+DG_MAX_p];,
            T T_dP3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dBFdx1[DG_MAX_Np];,
            T T_dBFdx2[DG_MAX_Np];,
            T T_dBFdx3[DG_MAX_Np];
        )
        T res;
        
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], this->lo[0], this->hi[0],
                                             T_BF, T_dBFdx1);,
            DG_space_2d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2,
                                             T_BF, T_dBFdx1, T_dBFdx2);,
            DG_space_3d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], &x[2], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2, T_P3, T_dP3,
                                             T_BF, T_dBFdx1, T_dBFdx2, T_dBFdx3);
        )

        res = 0.0;
        for (int c = 0; c < (this->Np); ++c)
        {
            res += T_BF[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
        }

        res *= -this->sgn;

        return res;
    }

    template <typename T>
    GpuArray<T, N> grad(const GpuArray<T, N> & x) const
    {
        T T_BF[DG_MAX_Np];
        AMREX_D_TERM
        (
            ,
            T T_P1[1+DG_MAX_p];
            T T_P2[1+DG_MAX_p];,
            T T_P3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            ,
            T T_dP1[1+DG_MAX_p];
            T T_dP2[1+DG_MAX_p];,
            T T_dP3[1+DG_MAX_p];
        )
        AMREX_D_TERM
        (
            T T_dBFdx1[DG_MAX_Np];,
            T T_dBFdx2[DG_MAX_Np];,
            T T_dBFdx3[DG_MAX_Np];
        )
        GpuArray<T, N> res;
        
        AMREX_D_PICK
        (
            DG_space_1d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], this->lo[0], this->hi[0],
                                             T_BF, T_dBFdx1);,
            DG_space_2d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2,
                                             T_BF, T_dBFdx1, T_dBFdx2);,
            DG_space_3d_BF_Table_NxL__host__(1, this->p,
                                             &x[0], &x[1], &x[2], this->lo, this->hi,
                                             T_P1, T_dP1, T_P2, T_dP2, T_P3, T_dP3,
                                             T_BF, T_dBFdx1, T_dBFdx2, T_dBFdx3);
        )

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
                res[0] += T_dBFdx1[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[1] += T_dBFdx2[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));,
                res[2] += T_dBFdx3[c]*this->fab(this->i,this->j,this->k,c+(this->u)*(this->Np));
            )
        }

        AMREX_D_TERM
        (
            res[0] *= -this->sgn;,
            res[1] *= -this->sgn;,
            res[2] *= -this->sgn;
        )

        return res;
    }
    // ================================================================
};

typedef algoim_DG_space_BFX<AMREX_SPACEDIM, __DG_PHI_MAX_SPACE_p__, __DG_PHI_MAX_SPACE_Np__> algoim_DG_phi_space_BFX;

} // namespace DG
} // namespace amrex

#endif