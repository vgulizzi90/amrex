//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_BFX.H
 * \brief Contains contants and data structures to evaluate the DG solution in time and space.
*/

#ifndef AMREX_DG_BFX_H_
#define AMREX_DG_BFX_H_

#include <AMReX_dG_BasisFunctions.H>

#define __DG_MAX_SPACE_p__ 3
#if (AMREX_SPACEDIM == 1)
#define __DG_MAX_SPACE_Np__ (1+__DG_MAX_SPACE_p__)
#endif
#if (AMREX_SPACEDIM == 2)
#define __DG_MAX_SPACE_Np__ ((1+__DG_MAX_SPACE_p__)*(1+__DG_MAX_SPACE_p__))
#endif
#if (AMREX_SPACEDIM == 3)
#define __DG_MAX_SPACE_Np__ ((1+__DG_MAX_SPACE_p__)*(1+__DG_MAX_SPACE_p__)*(1+__DG_MAX_SPACE_p__))
#endif

namespace amrex
{
namespace dG
{

template <int DG_MAX_p, int DG_MAX_Np>
struct dG_space_bf_base
{
    // DATA MEMBERS ===================================================
    const int p, Np;

    Real lo[AMREX_SPACEDIM], hi[AMREX_SPACEDIM];
    Real BF[DG_MAX_Np];
    Real AMREX_D_DECL(P1[1+DG_MAX_p], P2[1+DG_MAX_p], P3[1+DG_MAX_p]);
    Real AMREX_D_DECL(dP1[1+DG_MAX_p], dP2[1+DG_MAX_p], dP3[1+DG_MAX_p]);
    Real AMREX_D_DECL(dBFdx1[DG_MAX_Np], dBFdx2[DG_MAX_Np], dBFdx3[DG_MAX_Np]);
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMREX_GPU_HOST_DEVICE
    dG_space_bf_base(const Real * prob_lo, const Real * dx,
                     const int i, const int j, const int k, const short etype,
                     const int p_):
    p{p_},
    Np{AMREX_D_TERM((1+p_),*(1+p_),*(1+p_))}
    {
        this->set_support(prob_lo, dx, i, j, k, etype);
#ifdef AMREX_DEBUG
        this->self_check();
#endif
    }
    // ================================================================


    // READERS ========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void self_check() const
    {
        if (p > DG_MAX_p)
        {
#ifdef AMREX_USE_GPU
            Abort("ERROR: AMReX_dG_BFX.H - The order of the basis functions exceeds the maximum allowed number");
#else
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_BFX.H - dG_space_bf_base.self_check()\n";
            msg += "| The order of the basis functions exceeds the maximum allowed number.\n";
            msg += "| p       : "+std::to_string(p)+".\n";
            msg += "| DG_MAX_p: "+std::to_string(DG_MAX_p)+".\n";
            Abort(msg);
#endif
        }
    }
    // ================================================================


    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_support(const Real * prob_lo, const Real * dx, const int i, const int j, const int k, const short etype)
    {
        int bf_i, bf_j, bf_k;
        BF_CELL(i, j, k, etype, bf_i, bf_j, bf_k);

        AMREX_D_TERM
        (
            this->lo[0] = prob_lo[0]+bf_i*dx[0];,
            this->lo[1] = prob_lo[1]+bf_j*dx[1];,
            this->lo[2] = prob_lo[2]+bf_k*dx[2];
        )
        AMREX_D_TERM
        (
            this->hi[0] = this->lo[0]+dx[0];,
            this->hi[1] = this->lo[1]+dx[1];,
            this->hi[2] = this->lo[2]+dx[2];
        )
    }
    // ================================================================


    // BASIS FUNCTIONS EVALUATION =====================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_bf_only_table(const Real * x)
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

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_bf_table(const Real * x)
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
    // ================================================================

};

typedef dG_space_bf_base<__DG_MAX_SPACE_p__, __DG_MAX_SPACE_Np__> dG_space_bf;


template <int DG_MAX_p, int DG_MAX_Np>
struct dG_space_bfx_base
:
public dG_space_bf_base<DG_MAX_p, DG_MAX_Np>
{
    // DATA MEMBERS ===================================================
    int i, j, k;
    int u;
    Array4<Real const> const & fab;
    // ================================================================


    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    dG_space_bfx_base(const Real * prob_lo, const Real * dx,
                      const int i_, const int j_, const int k_, const short etype,
                      const int p_,
                      Array4<Real const> const & fab_)
    :
    dG_space_bf_base<DG_MAX_p, DG_MAX_Np>(prob_lo, dx, i_, j_, k_, etype, p_),
    i{i_}, j{j_}, k{k_},
    u{-1},
    fab{fab_}
    {}

    AMREX_GPU_HOST_DEVICE
    dG_space_bfx_base(const Real * prob_lo, const Real * dx,
                      const int i_, const int j_, const int k_,
                      const int p_,
                      Array4<Real const> const & fab_)
    :
    dG_space_bf_base<DG_MAX_p, DG_MAX_Np>(prob_lo, dx, i_, j_, k_, __DG_CELL_TYPE_ENTIRE__, p_),
    i{i_}, j{j_}, k{k_},
    u{-1},
    fab{fab_}
    {}
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
        this->eval_bf_only_table(x);

        return this->eval_unsafe();
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_unsafe(const int u_lo, const int u_hi, Real * U)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            U[this->u] = this->eval_unsafe();
        }
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval(const Real * x, const int u_lo, const int u_hi, Real * U)
    {
        this->eval_bf_only_table(x);

        this->eval_unsafe(u_lo, u_hi, U);
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
        this->eval_bf_table(x);

        this->eval_grad_unsafe(res);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad_unsafe(const int u_lo, const int u_hi, Real * U_grad)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            this->eval_grad_unsafe(&U_grad[(this->u)*AMREX_SPACEDIM]);
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_grad(const Real * x, const int u_lo, const int u_hi, Real * U_grad)
    {
        this->eval_bf_table(x);

        this->eval_grad_unsafe(u_lo, u_hi, U_grad);
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

typedef dG_space_bfx_base<__DG_MAX_SPACE_p__, __DG_MAX_SPACE_Np__> dG_space_bfx;


struct dG_space_rx
{
    // DATA MEMBERS ===================================================
    Real xc[AMREX_SPACEDIM];
    int i, j, k;
    int u;
    Array4<Real const> const & fab;
    Array4<Real const> const & slopes_fab;
    // ================================================================
    

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    dG_space_rx(const Real * xc_, const int i_, const int j_, const int k_,
                Array4<Real const> const & fab_,
                Array4<Real const> const & slopes_fab_)
    :
    xc{AMREX_D_DECL(xc_[0], xc_[1], xc_[2])},
    i{i_}, j{j_}, k{k_},
    u{-1},
    fab{fab_},
    slopes_fab{slopes_fab_}
    {}

    AMREX_GPU_HOST_DEVICE
    dG_space_rx(const int i_, const int j_, const int k_,
                Array4<Real const> const & fab_,
                Array4<Real const> const & slopes_fab_)
    :
    xc{AMREX_D_DECL(0.0, 0.0, 0.0)},
    i{i_}, j{j_}, k{k_},
    u{-1},
    fab{fab_},
    slopes_fab{slopes_fab_}
    {}
    // ================================================================


    // MISC ===========================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set_centroid(const Real * xc_)
    {
        AMREX_D_TERM
        (
            this->xc[0] = xc_[0];,
            this->xc[1] = xc_[1];,
            this->xc[2] = xc_[2];
        )
    }
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
    void eval(const Real * x, const int u_lo, const int u_hi, Real * U)
    {
        for (this->u = u_lo; this->u < u_hi; ++this->u)
        {
            U[this->u] = this->eval(x);
        }
    }
    // ================================================================
};

} // namespace DG
} // namespace amrex

#endif