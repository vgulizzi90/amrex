//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_ImplicitMesh.cpp
 * \brief Contains implementations of functions for ImplicitMesh objects.
*/

#include <AMReX_DG_ImplicitMesh.H>

namespace amrex
{
namespace DG
{

/**
 * \brief Interpolate the level set functions from coarse grids to fine grids.
*/
void InterpolateLevelsets(const int N_PHI,
                          const Geometry & c_geom,
                          const StandardRectangle<AMREX_SPACEDIM> & c_std_elem,
                          const MultiFab & c_PHI,
                          const IntVect rr,
                          const Geometry & f_geom,
                          const StandardRectangle<AMREX_SPACEDIM> & f_std_elem,
                          MultiFab & f_PHI)
{
    // PARAMETERS =====================================================
    // DG
    const int c_sNp = c_std_elem.Np;
    const int f_sNp = f_std_elem.Np;

    // INTERPOLATOR
    const Real * I_ptr = f_std_elem.I.data();

    // PARALLEL COPY?
    const bool parallel_copy_is_needed = (!isMFIterSafe(f_PHI, c_PHI));
    // ================================================================

    // VARIABLES ======================================================
    MultiFab safe_c_PHI;
    const MultiFab * safe_c_PHI_ptr;
    // ================================================================

    // CHECK CONSISTENCY ==============================================
    if (f_std_elem.I.size() != (f_sNp*c_sNp*AMREX_D_TERM(rr[0],*rr[1],*irr[2])))
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_ImplicitMesh.cpp - InterpolateLevelsets\n";
        msg += "| The fine and the coarse standard elements are not consistent.\n";
        msg += "| f_std_elem.I.size(): "+std::to_string(f_std_elem.I.size())+".\n";
        Abort(msg);
    }
    // ================================================================

    // SAFE COARSE DATA ===============================================
    if (parallel_copy_is_needed)
    {
        const BoxArray safe_ba = coarsen(f_PHI.boxarray, rr);
        const DistributionMapping & f_dm = f_PHI.distributionMap;
        const int n_comp = c_PHI.n_comp;
        const IntVect & n_grow = c_PHI.n_grow;

        safe_c_PHI.define(safe_ba, f_dm, n_comp, n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_PHI.ParallelCopy(c_PHI, 0, 0, n_comp, n_grow, n_grow, c_geom.periodicity());

        safe_c_PHI_ptr = &safe_c_PHI;
    }
    else
    {
        safe_c_PHI_ptr = &c_PHI;
    }
    // ================================================================

    // INTERPOLATE THE LEVEL SET FROM A COARSE GRID ===================
    for (MFIter mfi(f_PHI); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();

        Array4<Real const> const & c_PHI_fab = safe_c_PHI_ptr->array(mfi);

        Array4<Real> const & f_PHI_fab = f_PHI.array(mfi);
            
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // COARSE GRID'S ELEMENT
#if (AMREX_SPACEDIM == 1)
Print() << "InterpolateLevelsets: AMREX_SPACEDIM == 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int ci = (i-pi)/rr[0];
            const int cj = (j-pj)/rr[1];
            const int ck = k;
            const int pos = (pi+pj*rr[0])*f_sNp*c_sNp;
#endif
#if (AMREX_SPACEDIM == 3)
Print() << "InterpolateLevelsets: AMREX_SPACEDIM == 3" << std::endl;
exit(-1);
#endif
            // INITIALIZE
            for (int ru = 0; ru < N_PHI; ++ru)
            for (int rs = 0; rs < f_sNp; ++rs)
            {
                f_PHI_fab(i,j,k,rs+ru*f_sNp) = 0.0;
            }

            // USE THE INTERPOLATION OPERATOR
            for (int ru = 0; ru < N_PHI; ++ru)
            for (int cs = 0; cs < c_sNp; ++cs)
            for (int rs = 0; rs < f_sNp; ++rs)
            {
                f_PHI_fab(i,j,k,rs+ru*f_sNp) += I_ptr[pos+rs+cs*f_sNp]*c_PHI_fab(ci,cj,ck,cs+ru*c_sNp);
            }

        });
        Gpu::synchronize();
    }
    f_PHI.FillBoundary(f_geom.periodicity());
    // ================================================================

#ifdef AMREX_DEBUG
    // ================================================================
    if (f_PHI.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_ImplicitMesh.cpp - InterpolateLevelsets\n";
        msg += "| f_PHI contain nans.\n";
        Abort(msg);
    }
    // ================================================================
#endif
}

} // namespace DG
} // namespace amrex