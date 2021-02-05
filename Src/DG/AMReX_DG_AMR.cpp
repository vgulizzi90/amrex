//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_AMR.cpp
 * \brief Contains implementations of functions for AMR operations.
*/

#include <AMReX_DG_AMR.H>

namespace amrex
{
namespace DG
{
namespace AMR
{

// ####################################################################
// INTERPOLATION/RESTRICTION ##########################################
// ####################################################################
/**
 * \brief Interpolate a coarse solution onto a fine solution.
*/
void Interpolate(const int N_SOL,
                 const Gpu::ManagedVector<int> & Sol2Dom,
                 const ImplicitMesh & c_mesh,
                 const MatrixFactory & c_matfactory,
                 const MultiFab & c_X,
                 const IntVect rr,
                 const ImplicitMesh & f_mesh,
                 const MatrixFactory & f_matfactory,
                 MultiFab & f_X)
{
    // PARAMETERS =====================================================
    const BoxArray & f_ba = f_X.boxarray;
    const DistributionMapping & f_dm = f_X.distributionMap;
    const IntVect n_grow = f_X.n_grow;

    const int c_n_comp = c_X.n_comp;

    const BoxArray c_f_ba = coarsen(f_ba, rr);

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    const int * Sol2Dom_ptr = Sol2Dom.data();

    // IMPLICIT FINE MESH
#ifdef AMREX_USE_CUDA
    const Real * space_Dom_I_mem_ptr = f_matfactory.space_Dom_I_device_mem.data();
#else
    const Real * space_Dom_I_mem_ptr = f_matfactory.space_Dom_I_host_mem.data();
#endif

    // DG
    const int c_sNp = c_matfactory.std_elem.Np;
    const int f_sNp = f_matfactory.std_elem.Np;

    // BOX ARRAY AND DISTRIBUTION MAPPING
    const bool parallel_copy_is_needed = (!isMFIterSafe(f_X, c_X));
    // ================================================================

    // CHECK ==========================================================
#ifdef AMREX_DEBUG
    if (f_X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Interpolate\n";
        msg += "| f_X contains nans (on entry).\n";
        Abort(msg);
    }
    if (c_X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Interpolate\n";
        msg += "| c_X contains nans (on entry).\n";
        Abort(msg);
    }
#endif
    // ================================================================

    // VARIABLES ======================================================
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ================================================================

    // INITIALIZATION =================================================
    f_X = 0.0;
    // ================================================================

    // NEED PARALLEL COPY =============================================
    if (parallel_copy_is_needed)
    {
        safe_c_X.define(c_f_ba, f_dm, c_n_comp, n_grow);
        safe_c_X = 0.0;
        safe_c_X.ParallelCopy(c_X, 0, 0, c_n_comp, n_grow, n_grow, c_mesh.geom.periodicity());

        safe_c_X_ptr = &safe_c_X;
    }
    else
    {
        safe_c_X_ptr = &c_X;
    }
    // ================================================================

    // LOOP OVER THE FINE MESH' ELEMENTS ==============================
    for (MFIter mfi(f_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        // COARSE MESH
        Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);

        // INTERPOLATION OPERATOR
        Array4<long const> const & space_Dom_I_pos_fab = f_matfactory.space_Dom_I_pos.array(mfi);

        // FINE MESH
        Array4<short const> const & f_eType_fab = f_mesh.eType.array(mfi);
        Array4<Real> const & f_X_fab = f_X.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int f_i, int f_j, int f_k, int ru) noexcept
        {
            // ELEMENT TYPE
            const int dom = Sol2Dom_ptr[ru];
            const short f_etype = f_eType_fab(f_i,f_j,f_k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(f_etype))
            {
                // LOCAL PARAMETERS
                const long pos = space_Dom_I_pos_fab(f_i,f_j,f_k,dom);
                const Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                // LOCAL VARIABLES
                int c_i, c_j, c_k;

                // INDICES OF THE COARSE CELL
                FINE_TO_COARSE(f_i, f_j, f_k, rr, c_i, c_j, c_k);

                // INTERPOLATE
                for (int cs = 0; cs < c_sNp; ++cs)
                for (int rs = 0; rs < f_sNp; ++rs)
                {
                    f_X_fab(f_i,f_j,f_k,rs+ru*f_sNp) += I_ptr[rs+cs*f_sNp]*c_X_fab(c_i,c_j,c_k,cs+ru*c_sNp);
                }
            }
        });
        Gpu::synchronize();
    }
    f_X.FillBoundary(f_mesh.geom.periodicity());
    // ================================================================

    AddSmallElementsContribution(f_mesh, f_matfactory, N_SOL, Sol2Dom, f_X);
    MultiplyByInverseMassMatrix(f_mesh, f_matfactory, N_SOL, Sol2Dom, f_X);
}

/**
 * \brief Restrict a fine solution onto a coarse solution.
*/
void Restrict(const int N_SOL,
              const Gpu::ManagedVector<int> & Sol2Dom,
              const ImplicitMesh & f_mesh,
              const MatrixFactory & f_matfactory,
              const MultiFab & f_X,
              const IntVect rr,
              const ImplicitMesh & c_mesh,
              const MatrixFactory & c_matfactory,
              const iMultiFab & c_mask,
              MultiFab & c_X)
{
    // PARAMETERS =====================================================
    const BoxArray & f_ba = f_X.boxarray;
    const DistributionMapping & f_dm = f_X.distributionMap;
    //const int f_n_comp = f_X.n_comp;
    const IntVect n_grow = f_X.n_grow;

    const BoxArray & c_ba = c_X.boxarray;
    const DistributionMapping & c_dm = c_X.distributionMap;
    const int c_n_comp = c_X.n_comp;

    const BoxArray c_f_ba = coarsen(f_ba, rr);

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    const int * Sol2Dom_ptr = Sol2Dom.data();

    // IMPLICIT FINE MESH
#ifdef AMREX_USE_CUDA
    const Real * space_Dom_I_mem_ptr = f_matfactory.space_Dom_I_device_mem.data();
#else
    const Real * space_Dom_I_mem_ptr = f_matfactory.space_Dom_I_host_mem.data();
#endif

    // DG
    const int c_sNp = c_matfactory.std_elem.Np;
    const int f_sNp = f_matfactory.std_elem.Np;
    // ================================================================

    // CHECK ==========================================================
#ifdef AMREX_DEBUG
    if (f_X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
        msg += "| f_X contains nans (on entry).\n";
        Abort(msg);
    }
    if (c_X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
        msg += "| c_X contains nans (on entry).\n";
        Abort(msg);
    }
#endif
    // ================================================================

    // VARIABLES ======================================================
    MultiFab mf2(c_f_ba, f_dm, c_n_comp, n_grow);
    // ================================================================
    
    /*
    // AT THE FINE MESH ===============================================
    // 
    // First compute: mf1 := f_MM*f_X
    // 
    // ================================================================
    mf1 = 0.0;
    MultiFab::Copy(mf1, f_X, 0, 0, f_n_comp, n_grow);
    MultiplyByMassMatrix(f_mesh, f_matfactory, N_SOL, Sol2Dom, mf1);

#ifdef AMREX_DEBUG
    if (mf1.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
        msg += "| mf1 contains nans at step 1.\n";
        Abort(msg);
    }
#endif
    // ================================================================
    */

    // AT THE COARSE MESH COVERED BY THE FINE MESH ====================
    // 
    // First compute: mf2 := f_I_c^T*f_X
    // 
    // ================================================================
    mf2 = 0.0;

    for (MFIter mfi(mf2); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & f_eType_fab = f_mesh.eType.array(mfi);
        Array4<long const> const & space_Dom_I_pos_fab = f_matfactory.space_Dom_I_pos.array(mfi);

        Array4<Real const> const & f_X_fab = f_X.array(mfi);
        
        Array4<Real> const & mf2_fab = mf2.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int c_i, int c_j, int c_k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = Sol2Dom_ptr[ru];
#if (AMREX_SPACEDIM == 1)
            const Dim3 lo = {c_i*rr[0], 0, 0};
            const Dim3 hi = {(c_i+1)*rr[0], 1, 1};
#endif
#if (AMREX_SPACEDIM == 2)
            const Dim3 lo = {c_i*rr[0], c_j*rr[1], 0};
            const Dim3 hi = {(c_i+1)*rr[0], (c_j+1)*rr[1], 1};
#endif
#if (AMREX_SPACEDIM == 3)
            const Dim3 lo = {c_i*rr[0], c_j*rr[1], c_k*rr[2]};
            const Dim3 hi = {(c_i+1)*rr[0], (c_j+1)*rr[1], (c_k+1)*rr[2]};
#endif
            
            // LOOP OVER THE CELLS OF THE SUBGRID
            for (int f_k = lo.z; f_k < hi.z; ++f_k)
            for (int f_j = lo.y; f_j < hi.y; ++f_j)
            for (int f_i = lo.x; f_i < hi.x; ++f_i)
            {
                const short f_etype = f_eType_fab(f_i,f_j,f_k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(f_etype))
                {
                    const long pos = space_Dom_I_pos_fab(f_i,f_j,f_k,dom);
                    const Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                    for (int rs = 0; rs < c_sNp; ++rs)
                    for (int cs = 0; cs < f_sNp; ++cs)
                    {
                        mf2_fab(c_i,c_j,c_k,rs+ru*c_sNp) += I_ptr[cs+rs*f_sNp]*f_X_fab(f_i,f_j,f_k,cs+ru*f_sNp);
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    mf2.FillBoundary(f_mesh.geom.periodicity());

#ifdef AMREX_DEBUG
    if (mf2.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
        msg += "| mf2 contains nans.\n";
        Abort(msg);
    }
#endif
    // ================================================================

    // AT THE COARSE MESH =============================================
    // 
    // Finally compute: c_X := c_MM^(-1)*mf2
    // 
    // ================================================================
    if (isMFIterSafe(f_X, c_X))
    {
        AddSmallElementsContribution(c_mesh, c_matfactory, N_SOL, Sol2Dom, mf2);
        MultiplyByInverseMassMatrix(c_mesh, c_matfactory, N_SOL, Sol2Dom, mf2);
        
        CopyWithMask(mf2, c_X, c_mask, c_n_comp, c_mesh.geom.periodicity());

#ifdef AMREX_DEBUG
        if (c_X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
            msg += "| c_X contains nans (MFIter is safe).\n";
            Abort(msg);
        }
#endif
    }
    else
    {
        MultiFab mf1(c_ba, c_dm, c_n_comp, n_grow);
        mf1 = 0.0;
        mf1.ParallelCopy(mf2, 0, 0, c_n_comp, n_grow, n_grow, c_mesh.geom.periodicity());
        AddSmallElementsContribution(c_mesh, c_matfactory, N_SOL, Sol2Dom, mf1);
        MultiplyByInverseMassMatrix(c_mesh, c_matfactory, N_SOL, Sol2Dom, mf1);

        CopyWithMask(mf1, c_X, c_mask, c_n_comp, c_mesh.geom.periodicity());

#ifdef AMREX_DEBUG
        if (mf1.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
            msg += "| mf1 contains nans at step 3.\n";
            Abort(msg);
        }
        if (c_X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_AMR.cpp - Restrict\n";
            msg += "| c_X contains nans at step 3.\n";
            Abort(msg);
        }
#endif
    }
    // ================================================================
}

// ####################################################################
// ####################################################################

} // namespace AMR
} // namespace DG
} // namespace amrex
