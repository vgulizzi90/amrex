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
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Interpolate(const int, const Gpu::ManagedVector<int> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    const BoxArray & f_ba = f_X.boxarray;
    const DistributionMapping & f_dm = f_X.distributionMap;

    const int c_eType_n_comp = c_mesh.eType.n_comp;
    const IntVect c_eType_n_grow = c_mesh.eType.n_grow;
    const int c_X_n_comp = c_X.n_comp;
    const IntVect c_X_n_grow = c_X.n_grow;

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
    shortMultiFab safe_c_eType;
    const shortMultiFab * safe_c_eType_ptr;

    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ================================================================

    // INITIALIZATION =================================================
    f_X = 0.0;
    // ================================================================

    // NEED PARALLEL COPY =============================================
    if (parallel_copy_is_needed)
    {
        safe_c_eType.define(c_f_ba, f_dm, c_eType_n_comp, c_eType_n_grow);
        safe_c_eType = 0;
        safe_c_eType.ParallelCopy(c_mesh.eType, 0, 0, c_eType_n_comp, c_eType_n_grow, c_eType_n_grow, c_mesh.geom.periodicity());

        safe_c_eType_ptr = &safe_c_eType;
        
        safe_c_X.define(c_f_ba, f_dm, c_X_n_comp, c_X_n_grow);
        safe_c_X = 0.0;
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_mesh.geom.periodicity());

        safe_c_X_ptr = &safe_c_X;
    }
    else
    {
        safe_c_eType_ptr = &c_mesh.eType;
        safe_c_X_ptr = &c_X;
    }
    // ================================================================

    // LOOP OVER THE FINE MESH' ELEMENTS ==============================
    for (MFIter mfi(f_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        // COARSE MESH
        Array4<short const> const & c_eType_fab = safe_c_eType_ptr->array(mfi);
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
                short c_etype;
                int c_BF_i, c_BF_j, c_BF_k;

                // INDICES OF THE COARSE CELL
                FINE_TO_COARSE(f_i, f_j, f_k, rr, c_i, c_j, c_k);
                c_etype = c_eType_fab(c_i,c_j,c_k,ELM_TYPE(dom));
                BF_CELL(c_i, c_j, c_k, c_etype, c_BF_i, c_BF_j, c_BF_k);

                // INTERPOLATE
                for (int cs = 0; cs < c_sNp; ++cs)
                for (int rs = 0; rs < f_sNp; ++rs)
                {
                    f_X_fab(f_i,f_j,f_k,rs+ru*f_sNp) += I_ptr[rs+cs*f_sNp]*c_X_fab(c_BF_i,c_BF_j,c_BF_k,cs+ru*c_sNp);
                }
            }
        });
        Gpu::synchronize();
    }
    f_X.FillBoundary(f_mesh.geom.periodicity());

    // HANDLE GHOST CELLS
    for (MFIter mfi(f_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();

        // COARSE MESH
        Array4<short const> const & c_eType_fab = safe_c_eType_ptr->array(mfi);
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

            if (ELM_IS_NOT_EMPTY(f_etype) && ELM_IS_GHOST(f_etype))
            {
                // LOCAL PARAMETERS
                const long pos = space_Dom_I_pos_fab(f_i,f_j,f_k,dom);
                const Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                // LOCAL VARIABLES
                int c_i, c_j, c_k;
                short c_etype;
                int c_BF_i, c_BF_j, c_BF_k;

                // INDICES OF THE COARSE CELL
                FINE_TO_COARSE(f_i, f_j, f_k, rr, c_i, c_j, c_k);
                c_etype = c_eType_fab(c_i,c_j,c_k,ELM_TYPE(dom));
                BF_CELL(c_i, c_j, c_k, c_etype, c_BF_i, c_BF_j, c_BF_k);

                // INTERPOLATE
                for (int cs = 0; cs < c_sNp; ++cs)
                for (int rs = 0; rs < f_sNp; ++rs)
                {
                    f_X_fab(f_i,f_j,f_k,rs+ru*f_sNp) += I_ptr[rs+cs*f_sNp]*c_X_fab(c_BF_i,c_BF_j,c_BF_k,cs+ru*c_sNp);
                }
            }
        });
        Gpu::synchronize();
    }
    f_X.FillBoundary(f_mesh.geom.periodicity());
    // ================================================================
/*
if (!c_X_fab.contains(c_BF_i, c_BF_j, c_BF_k))
{

Print() << "fine_fbx: " << bx << std::endl;
Print() << "fine_vbx: " << vbx << std::endl;
Print() << "     fine: " << IntVect(f_i, f_j, f_k) << std::endl;
Print() << "   coarse: " << IntVect(c_i,c_j,c_k) << std::endl;
Print() << "BF coarse: " << IntVect(c_BF_i, c_BF_j, c_BF_k) << std::endl;

const bool f_elm_is_ghost = ELM_IS_GHOST(f_etype);
Print() << "fine elm is ghost? " << (f_elm_is_ghost ? "yes" : "no") << std::endl;

Print() << "We might have a problem!" << std::endl;
exit(-1);
}
*/

    // CHECK ==========================================================
#ifdef AMREX_DEBUG
    if (f_X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_AMR.cpp - Interpolate\n";
        msg += "| f_X contains nans (after interpolation).\n";
        Abort(msg);
    }
#endif
    // ================================================================

    AddSmallElementsContribution(f_mesh, f_matfactory, N_SOL, Sol2Dom, f_X);
    MultiplyByInverseMassMatrix(f_mesh, f_matfactory, N_SOL, Sol2Dom, f_X, true);
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
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Restrict(const int, const Gpu::ManagedVector<int> &, ....)");
    // ----------------------------------------------------------------

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
                int f_BF_i, f_BF_j, f_BF_k;

                BF_CELL(f_i, f_j, f_k, f_etype, f_BF_i, f_BF_j, f_BF_k);

                if (ELM_IS_NOT_EMPTY(f_etype))
                {
                    const long pos = space_Dom_I_pos_fab(f_i,f_j,f_k,dom);
                    const Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                    for (int rs = 0; rs < c_sNp; ++rs)
                    for (int cs = 0; cs < f_sNp; ++cs)
                    {
                        mf2_fab(c_i,c_j,c_k,rs+ru*c_sNp) += I_ptr[cs+rs*f_sNp]*f_X_fab(f_BF_i,f_BF_j,f_BF_k,cs+ru*f_sNp);
                    }
/*
if (c_i == __i__ && c_j == __j__ && ru == 0)
{
Print() << "f_i, f_j: " << f_i << "," << f_j << std::endl;
Print() << "I_ptr" << std::endl;
IO::PrintRealArray2D(f_sNp, c_sNp, I_ptr);
}
*/
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

    /*
    IO::PrintMultiFabEntry(c_X, __i__, __j__, 0, 0);
    exit(-1);
    */
    // ================================================================
}

// ####################################################################
// ####################################################################



// ####################################################################
// AUXILIARY FUNCTIONS ################################################
// ####################################################################
/**
 * \brief Check that merging does not occur between a masked cell and an unmasked cells or viceversa.
*/
void CheckMergeLeaking(const ImplicitMesh & mesh, const int N_DOM, const iMultiFab & mask)
{
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_small = ELM_IS_SMALL(etype);
            const bool elm_is_extended = ELM_IS_EXTENDED(etype);
            const bool elm_is_ghost = ELM_IS_GHOST(etype);
            const bool cell_is_masked = CELL_IS_MASKED(mask_fab(i,j,k));

            if (elm_is_small)
            {
                const int merged_b = etype/10;
                int nbr_i, nbr_j, nbr_k, nbr_b;

                NBR_CELL(i, j, k, merged_b, nbr_i, nbr_j, nbr_k, nbr_b);

                const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                const bool nbr_is_ghost = ELM_IS_GHOST(nbr_etype);
                const bool nbr_cell_is_masked = CELL_IS_MASKED(mask_fab(nbr_i,nbr_j,nbr_k));

                if (elm_is_ghost && !nbr_is_ghost)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                    msg += "| A ghost cell should not be merged with a non-ghost cell.\n";
                    msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                    msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                    Abort(msg);
                }
                if (!elm_is_ghost && nbr_is_ghost)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                    msg += "| A non-ghost cell should not be merged with a ghost cell.\n";
                    msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                    msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                    Abort(msg);
                }
                if (cell_is_masked && !nbr_cell_is_masked)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                    msg += "| A masked cell should not be merged with a unmasked cell.\n";
                    msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                    msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                    Abort(msg);
                }
                if (!cell_is_masked && nbr_cell_is_masked)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                    msg += "| An unmasked cell should not be merged with a masked cell.\n";
                    msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                    msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                    Abort(msg);
                }
            }
            else if (elm_is_extended)
            {
                int nbr_i, nbr_j, nbr_k, nbr_b;

                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const short nbr_merged_b = (ELM_IS_SMALL(nbr_etype)) ? (nbr_etype/10) : -1;
                    const bool nbr_is_ghost = ELM_IS_GHOST(nbr_etype);
                    const bool nbr_cell_is_masked = CELL_IS_MASKED(mask_fab(nbr_i,nbr_j,nbr_k));

                    if (nbr_merged_b == nbr_b)
                    {
                        if (elm_is_ghost && !nbr_is_ghost)
                        {
                            std::string msg;
                            msg  = "\n";
                            msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                            msg += "| A ghost cell should not be extended onto a non-ghost cell.\n";
                            msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                            msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                            Abort(msg);
                        }
                        if (!elm_is_ghost && nbr_is_ghost)
                        {
                            std::string msg;
                            msg  = "\n";
                            msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                            msg += "| A non-ghost cell should not be extended onto a ghost cell.\n";
                            msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                            msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                            Abort(msg);
                        }
                        if (cell_is_masked && !nbr_cell_is_masked)
                        {
                            std::string msg;
                            msg  = "\n";
                            msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                            msg += "| A masked cell should not be extended onto a unmasked cell.\n";
                            msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                            msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                            Abort(msg);
                        }
                        if (!cell_is_masked && nbr_cell_is_masked)
                        {
                            std::string msg;
                            msg  = "\n";
                            msg += "ERROR: AMReX_DG_AMR.cpp - CheckMergeLeaking\n";
                            msg += "| An unmasked cell should not be extended onto a masked cell.\n";
                            msg += "|     elm("+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"): "+ELM_TYPE_DESCRIPTION(etype)+"\n";
                            msg += "| nbr_elm("+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"): "+ELM_TYPE_DESCRIPTION(nbr_etype)+"\n";
                            Abort(msg);
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
}
// ####################################################################
// ####################################################################

} // namespace AMR
} // namespace DG
} // namespace amrex
