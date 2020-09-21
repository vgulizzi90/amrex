// AMReX_DG_SlopeLimiter.H

#ifndef BL_DG_SLOPE_LIMITER_H_
#define BL_DG_SLOPE_LIMITER_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

namespace amrex
{
namespace DG
{

// ####################################################################
// DG BASE CLASS ######################################################
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