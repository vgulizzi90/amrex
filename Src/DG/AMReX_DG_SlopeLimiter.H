// AMReX_DG_SlopeLimiter.H

#ifndef BL_DG_SLOPE_LIMITER_H_
#define BL_DG_SLOPE_LIMITER_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

#define DG_SL_THETA 1.0

namespace amrex
{
namespace DG
{
// ####################################################################
// DG SLOPE LIMITER ROUTINES ##########################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_FV(const Real t,
                          const ImplicitMesh<N_PHI, N_DOM> & mesh,
                          const MatrixFactory<N_PHI, N_DOM> & /*matfactory*/,
                          const MultiFab & X,
                          MultiFab & dX,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_FV(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    // ================================================================

    // EVAL THE PRIMARY VARIABLES AND THEIR GRADIENTS =================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            if (elm_is_not_empty)
            {
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM];
                Real Uh[N_U], dUh[N_U], dCh_left[N_U], dCh_right[N_U], dCh[N_U];

                // CELL-CENTERED INFO ---------------------------------
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )
                for (int ru = 0; ru < N_U; ++ru)
                {
                    Uh[ru] = X_fab(i,j,k,ru);
                }
                // ----------------------------------------------------

                // X1 DIRECTION ---------------------------------------
                // LEFT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i,j,k,ru)-X_fab(i-1,j,k,ru);
                }
                IBVP.F_DU2DC_1(dom, t, x, Uh, dUh, dCh_left);
                // RIGHT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i+1,j,k,ru)-X_fab(i,j,k,ru);
                }
                IBVP.F_DU2DC_1(dom, t, x, Uh, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU_1(dom, t, x, Uh, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,0+AMREX_SPACEDIM*ru) = dUh[ru]/dx[0];
                }
                // ----------------------------------------------------

#if (AMREX_SPACEDIM > 1)
                // X2 -------------------------------------------------
                // LEFT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i,j,k,ru)-X_fab(i,j-1,k,ru);
                }
                IBVP.F_DU2DC_2(dom, t, x, Uh, dUh, dCh_left);
                // RIGHT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i,j+1,k,ru)-X_fab(i,j,k,ru);
                }
                IBVP.F_DU2DC_2(dom, t, x, Uh, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU_2(dom, t, x, Uh, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,1+AMREX_SPACEDIM*ru) = dUh[ru]/dx[1];
                }
                // ----------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 -------------------------------------------------
                // LEFT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i,j,k,ru)-X_fab(i,j,k-1,ru);
                }
                IBVP.F_DU2DC_3(dom, t, x, Uh, dUh, dCh_left);
                // RIGHT DIFFERENCES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = X_fab(i,j,k+1,ru)-X_fab(i,j,k,ru);
                }
                IBVP.F_DU2DC_3(dom, t, x, Uh, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU_3(dom, t, x, Uh, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,2+AMREX_SPACEDIM*ru) = dUh[ru]/dx[2];
                }
                // ----------------------------------------------------
#endif
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
/*
Print() << "dUh: "; DG_utils::PrintRealArray2D(1, N_U, dUh); 
Print() << "dCh_left: "; DG_utils::PrintRealArray2D(1, N_U, dCh_left);
Print() << "dCh_right: "; DG_utils::PrintRealArray2D(1, N_U, dCh_right);
Print() << "dCh: "; DG_utils::PrintRealArray2D(1, N_U, dCh);
Print() << "dUh: "; DG_utils::PrintRealArray2D(1, N_U, dUh);

Print() << "HERE WE ARE" << std::endl;
exit(-1);
*/
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_FV_LS(const Real t,
                             const ImplicitMesh<N_PHI, N_DOM> & mesh,
                             const MatrixFactory<N_PHI, N_DOM> & matfactory,
                             const MultiFab & X,
                             MultiFab & dX,
                             const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_FV_LS(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    const int ngr = 1;

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    // ================================================================

    // EVAL THE PRIMARY VARIABLES AND THEIR GRADIENTS =================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            if (elm_is_valid)
            {


Print() << "HERE WE ARE " << std::endl;
exit(-1);
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}


template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_DG(const Real t,
                          const ImplicitMesh<N_PHI, N_DOM> & mesh,
                          const MatrixFactory<N_PHI, N_DOM> & matfactory,
                          MultiFab & X,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_DG(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------


Print() << "HERE WE ARE - ApplySlopeLimiter_DG" << std::endl;
exit(-1);
}
// ####################################################################



// ####################################################################
// DG SLOPE LIMITER CLASS #############################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct SlopeLimiter
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // REFINEMENT RATIO
    IntVect ref_ratio;

    // IMPLICIT MESH
    ImplicitMesh<N_PHI, N_DOM> mesh;

    // MATRIX FACTORY
    MatrixFactory<N_PHI, N_DOM> matfactory;

    // SOLUTION MULTIFABS
    MultiFab X_old, X_new, dXdt;
    charMultiFab tags;
    // ################################################################

    // INITIALIZATION #################################################
    void Define(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                const MatrixFactory<N_PHI, N_DOM> & coarse_matfactory,
                const InputReader & inputs, const int d)
    {
        // REFINEMENT RATIO ===========================================
        const int p = inputs.dG[d].space_p;
        
        AMREX_D_TERM
        (
            this->ref_ratio[0] = (inputs.dG[d].limiter.freeze_refinement[0] != 0) ? 1 : (2*p+1);,
            this->ref_ratio[1] = (inputs.dG[d].limiter.freeze_refinement[1] != 0) ? 1 : (2*p+1);,
            this->ref_ratio[2] = (inputs.dG[d].limiter.freeze_refinement[2] != 0) ? 1 : (2*p+1);
        )
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->mesh.DefineForSlopeLimiter(coarse_mesh, this->ref_ratio, inputs, d);
        // ============================================================

        // MATRIX FACTORY =============================================
        this->matfactory.DefineForSlopeLimiter(coarse_matfactory, this->ref_ratio,
                                               this->mesh.cc_ba, this->mesh.dm, inputs, d);
        // ============================================================

        // SOLUTION MULTIFABS =========================================
        {
            const int n_comp = (this->matfactory.cube.Np)*N_U;

            this->X_old.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_old = 0.0;

            this->X_new.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_new = 0.0;

            this->dXdt.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->dXdt = 0.0;
        }

        this->tags.define(this->mesh.cc_ba, this->mesh.dm, N_DOM, this->dom_data_ngr);
        this->tags = SOUND_ELEMENT;
        // ============================================================
    }
    // ################################################################

    // SET/UPDATE TAGS FROM COARSE MESH ###############################
    void SetTags(const charMultiFab & coarse_tags)
    {
        // PARAMETERS =================================================
        const IntVect rr = this->ref_ratio;
        // ============================================================

        // ============================================================
        for (MFIter mfi(this->tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<char const> const & coarse_tags_fab = coarse_tags.array(mfi);

            Array4<char> const & tags_fab = this->tags.array(mfi);
                
            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // COARSE GRID'S ELEMENT
#if (AMREX_SPACEDIM == 1)
#endif
#if (AMREX_SPACEDIM == 2)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = k;
#endif
#if (AMREX_SPACEDIM == 3)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = (k-pk)/rr[2];
#endif
    
                if (coarse_tags_fab(ci,cj,ck,dom) == TROUBLED_ELEMENT)
                {
                    tags_fab(i,j,k,dom) = SOUND_ELEMENT;
                }
                else
                {
                    tags_fab(i,j,k,dom) = TROUBLED_ELEMENT;
                }
            });
            Gpu::synchronize();
        }
        this->tags.FillBoundary(this->mesh.geom.periodicity());
        // ============================================================
    }

    void UpdateTags(const charMultiFab & coarse_tags)
    {
        // PARAMETERS =================================================
        const IntVect rr = this->ref_ratio;
        // ============================================================

        // ============================================================
        for (MFIter mfi(this->tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<char const> const & coarse_tags_fab = coarse_tags.array(mfi);

            Array4<char> const & tags_fab = this->tags.array(mfi);
                
            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // COARSE GRID'S ELEMENT
#if (AMREX_SPACEDIM == 1)
#endif
#if (AMREX_SPACEDIM == 2)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = k;
#endif
#if (AMREX_SPACEDIM == 3)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = (k-pk)/rr[2];
#endif
                const bool elm_is_tagged = (tags_fab(i,j,k,dom) == TROUBLED_ELEMENT);
                const bool coarse_elm_is_troubled = (coarse_tags_fab(ci,cj,ck,dom) == TROUBLED_ELEMENT);

                if (elm_is_tagged && coarse_elm_is_troubled)
                {
                    tags_fab(i,j,k,dom) = SOUND_ELEMENT;
                }
            });
            Gpu::synchronize();
        }
        this->tags.FillBoundary(this->mesh.geom.periodicity());
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif