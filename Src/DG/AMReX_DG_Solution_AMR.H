// AMReX_DG_Solution_AMR.H

#ifndef BL_DG_SOLUTION_AMR_H_
#define BL_DG_SOLUTION_AMR_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

namespace amrex
{
namespace DG
{
namespace AMR
{
// CHECK CONSISTENCY BETWEEN LEVELS AND REFINEMENT RATIO ##############
template<int N_PHI, int N_DOM>
void CheckLevelsConsistency(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                            const IntVect rr,
                            const ImplicitMesh<N_PHI, N_DOM> & fine_mesh)
{
    const bool bad_rr[AMREX_SPACEDIM] = 
    {
        AMREX_D_DECL
        (
            std::abs(fine_mesh.dx[0]*rr[0]-coarse_mesh.dx[0]) > fine_mesh.rtol,
            std::abs(fine_mesh.dx[1]*rr[1]-coarse_mesh.dx[1]) > fine_mesh.rtol,
            std::abs(fine_mesh.dx[2]*rr[2]-coarse_mesh.dx[2]) > fine_mesh.rtol
        )
    };

    if (AMREX_D_TERM(bad_rr[0], || bad_rr[1], || bad_rr[2]))
    {
        Print() << std::endl;
        Print() << "ERROR: AMReX_DG_Solution_AMR.H - CheckLevelsConsistency" << std::endl;
        Print() << "| The input meshes are not consistent with the refinement ratios:" << std::endl;
        Print() << "| coarse_mesh.dx = "; DG_utils::PrintRealArray2D(1, AMREX_SPACEDIM, coarse_mesh.dx.data());
        Print() << "| fine_mesh.dx = "; DG_utils::PrintRealArray2D(1, AMREX_SPACEDIM, fine_mesh.dx.data());
        Print() << "| Refinement ratios = " << rr << std::endl;
        Print() << std::endl;
        exit(-1);
    }   
}
// ####################################################################

// INTERPOLATE LEVEL SETS FROM COARSE GRID TO FINE GRID ###############
template<int N_PHI, int N_DOM>
void InterpolateLevelSets(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                          const IntVect rr,
                          ImplicitMesh<N_PHI, N_DOM> & fine_mesh)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("InterpolateLevelSets(const ImplicitMesh<N_PHI, N_DOM> &, const IntVect, ....)");
    // ----------------------------------------------------------------

    // QUICK CHECK ====================================================
    CheckLevelsConsistency<N_PHI, N_DOM>(coarse_mesh, rr, fine_mesh);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID (FINE MESH)
    const GpuArray<Real, AMREX_SPACEDIM> fine_dx = fine_mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> fine_prob_lo = fine_mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> fine_J = fine_mesh.J;

    // STANDARD ELEMENT (FINE MESH)
    const int fine_sNp = fine_mesh.cube.Np;
    const Real * I_ptr = fine_mesh.cube.I.data();

    // STANDARD ELEMENT (COARSE MESH)
    const int coarse_sNp = coarse_mesh.cube.Np;
    // ================================================================

    // INITIALIZATION =================================================
    fine_mesh.PHI = 0.0;
    // ================================================================

    // INTERPOLATE THE LEVEL SETS FROM THE COARSE GRID ================
    for (MFIter mfi(fine_mesh.PHI); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();

        Array4<Real const> const & coarse_PHI_fab = coarse_mesh.PHI.array(mfi);

        Array4<Real> const & fine_PHI_fab = fine_mesh.PHI.array(mfi);
            
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
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
            const int pos = (pi+pj*rr[0])*fine_sNp*coarse_sNp;
#endif
#if (AMREX_SPACEDIM == 3)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
            const int ci = (i-pi)/rr[0];
            const int cj = (j-pj)/rr[1];
            const int ck = (k-pk)/rr[2];
            const int pos = (pi+pj*rr[0]+pk*rr[0]*rr[1])*fine_sNp*coarse_sNp;
#endif
            // USE THE INTERPOLATION OPERATOR
            for (int ru = 0; ru < N_PHI; ++ru)
            for (int cs = 0; cs < coarse_sNp; ++cs)
            for (int rs = 0; rs < fine_sNp; ++rs)
            {
                fine_PHI_fab(i,j,k,rs+ru*fine_sNp) += I_ptr[pos+rs+cs*fine_sNp]*coarse_PHI_fab(ci,cj,ck,cs+ru*coarse_sNp);
            }
        });
        Gpu::synchronize();
    }
    fine_mesh.PHI.FillBoundary(fine_mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// EVAL INTERPOLATION OPERATOR BETWEEN TWO IMPLICITLY-DEFINED MESHES ##
template<int N_PHI, int N_DOM>
void EvalInterpolationOperator(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                               const MatrixFactory<N_PHI, N_DOM> & coarse_matfactory,
                               const IntVect rr,
                               const ImplicitMesh<N_PHI, N_DOM> & fine_mesh,
                               MatrixFactory<N_PHI, N_DOM> & fine_matfactory)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalInterpolationOperator(const ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // QUICK CHECK ====================================================
    CheckLevelsConsistency<N_PHI, N_DOM>(coarse_mesh, rr, fine_mesh);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID (COARSE MESH)
    const GpuArray<Real, AMREX_SPACEDIM> coarse_dx = coarse_mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> coarse_prob_lo = coarse_mesh.prob_lo;

    // GRID (FINE MESH)
    const GpuArray<Real, AMREX_SPACEDIM> fine_dx = fine_mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> fine_prob_lo = fine_mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> fine_J = fine_mesh.J;
    const Real fine_aJ = fine_mesh.aJ, fine_iaJ = fine_mesh.iaJ;

    // STANDARD ELEMENT (COARSE MESH)
    const int coarse_sp = coarse_matfactory.cube.p;
    const int coarse_sNp = coarse_matfactory.cube.Np;

    // STANDARD ELEMENT (FINE MESH)
    const int fine_sp = fine_matfactory.cube.p;
    const int fine_sNp = fine_matfactory.cube.Np;

    // IMPLICIT MESH (FINE MESH)
    const Real * eDomQuad_mem_ptr = fine_mesh.eDomQuad_host_mem.data();
#ifdef AMREX_USE_CUDA
    const Real * space_MMCh_mem_ptr = fine_matfactory.space_MMCh_device_mem.data();
#else
    const Real * space_MMCh_mem_ptr = fine_matfactory.space_MMCh_host_mem.data();
#endif
    // ================================================================

    // VARIABLES ======================================================
    long space_Dom_I_mem_pos;
    // ================================================================

    // INIT THE INTERPOLATION OPERATORS FOR THE STANDARD ELEMENT ======
    fine_matfactory.cube.InitInterpolation(coarse_matfactory.cube, rr);
    // ================================================================

    // INITIALIZATION =================================================
    space_Dom_I_mem_pos = 0L;
    // ================================================================

    // STANDARD ELEMENT INFO ==========================================
    // DOMAIN
    {
        // Size (we could have used fine_matfactory.cube.I.size())
        const int size = fine_sNp*coarse_sNp*AMREX_D_PICK(rr[0], rr[0]*rr[1], rr[0]*rr[1]*rr[2]);

        // Resize memory
        fine_matfactory.space_Dom_I_host_mem.resize(size);

        // Store data
        for (int k = 0; k < size; ++k)
        {
            fine_matfactory.space_Dom_I_host_mem[k] = fine_matfactory.cube.I[k];
        }

        // Update memory
        space_Dom_I_mem_pos += size;
    }
    // ================================================================

    // STORE THE IMPLICIT MESH INFO: DOMAIN (CELL-CENTERED) DATA ======
    for (MFIter mfi(fine_mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);
        
        // FINE MESH
        Array4<short const> const & fine_eType_fab = fine_mesh.eType.array(mfi);

        Array4<int const> const & fine_eDomQuad_Nq_fab = fine_mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & fine_eDomQuad_pos_fab = fine_mesh.eDomQuad_pos.array(mfi);

        // COARSE MESH
        Array4<short const> const & coarse_eType_fab = coarse_mesh.eType.array(mfi);

        // INTERPOLATION OPERATOR
        Array4<long> const & space_Dom_I_pos_fab = fine_matfactory.space_Dom_I_pos.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS (FINE GRID'S ELEMENT)
            const Real fine_cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(fine_prob_lo[0]+(i+0.5)*fine_dx[0],
                                                                   fine_prob_lo[1]+(j+0.5)*fine_dx[1],
                                                                   fine_prob_lo[2]+(k+0.5)*fine_dx[2])};
            const short fine_etype = fine_eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool fine_elm_is_empty = (fine_etype == DG_ELM_TYPE_EMPTY);
            const bool fine_elm_is_entire = (fine_etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool fine_elm_is_large = (fine_etype%10 == DG_ELM_TYPE_LARGE);
            const bool fine_elm_is_small = (fine_etype%10 == DG_ELM_TYPE_SMALL);
            const bool fine_elm_is_partial = (fine_elm_is_large || fine_elm_is_small);
            const bool fine_elm_is_entire_extended = fine_elm_is_entire && (fine_etype/10 == 1);
            const bool fine_elm_is_entire_but_not_extended = fine_elm_is_entire && (fine_etype/10 != 1);

            // LOCAL PARAMETERS (COARSE GRID'S ELEMENT)
#if (AMREX_SPACEDIM == 1)
#endif
#if (AMREX_SPACEDIM == 2)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int ci = (i-pi)/rr[0];
            const int cj = (j-pj)/rr[1];
            const int ck = k;
            const int I_pos = (pi+pj*rr[0])*fine_sNp*coarse_sNp;
#endif
#if (AMREX_SPACEDIM == 3)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
            const int ci = (i-pi)/rr[0];
            const int cj = (j-pj)/rr[1];
            const int ck = (k-pk)/rr[2];
            const int I_pos = (pi+pj*rr[0]+pk*rr[0]*rr[1])*fine_sNp*coarse_sNp;
#endif
            const int coarse_etype = coarse_eType_fab(ci,cj,ck,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool coarse_elm_is_not_small = (coarse_etype%10 != DG_ELM_TYPE_SMALL);

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], w;
            Real I[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];
            Real fine_BF_lo[AMREX_SPACEDIM], fine_BF_hi[AMREX_SPACEDIM];
            Real coarse_BF_lo[AMREX_SPACEDIM], coarse_BF_hi[AMREX_SPACEDIM];

            // BASIS FUNCTIONS SUPPORT (FINE GRID'S ELEMENT)
            BF_CELL(&fine_prob_lo[0], &fine_dx[0], i, j, k, fine_etype, fine_BF_lo, fine_BF_hi);

            // Data structures used to compute the basis functions
            DG_sol_space_BF fine_bf(fine_BF_lo, fine_BF_hi, fine_sp);

            // BASIS FUNCTIONS SUPPORT (COARSE GRID'S ELEMENT)
            BF_CELL(&coarse_prob_lo[0], &coarse_dx[0], ci, cj, ck, coarse_etype, coarse_BF_lo, coarse_BF_hi);

            // Data structures used to compute the basis functions
            DG_sol_space_BF coarse_bf(coarse_BF_lo, coarse_BF_hi, coarse_sp);

            // EMPTY ELEMENTS -----------------------------------------
            if (fine_elm_is_empty)
            {
                // Store memory offset
                space_Dom_I_pos_fab(i,j,k,dom) = -1L;
            }
            // --------------------------------------------------------
            // ENTIRE ELEMENTS ----------------------------------------
            else if (fine_elm_is_entire_but_not_extended && coarse_elm_is_not_small)
            {
                // Store memory offset
                space_Dom_I_pos_fab(i,j,k,dom) = I_pos;
            }
            // --------------------------------------------------------
            // PARTIAL ELEMENTS ---------------------------------------
            else if (fine_elm_is_partial || fine_elm_is_entire_extended)
            {
                // QUADRATURE INFO
                const int dom_Nq = fine_eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                const long pos = fine_eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                const Real * xptr = &eDomQuad_mem_ptr[pos];
                long qi_pos;

                // Store memory offset
                space_Dom_I_pos_fab(i,j,k,DG_DOM_BF_INFO_N_COMP_PER_DOM*dom) = space_Dom_I_mem_pos;

                // Expand memory
                fine_matfactory.space_Dom_I_host_mem.resize(space_Dom_I_mem_pos+fine_sNp*coarse_sNp);

                // Store data
                std::fill(I, I+fine_sNp*coarse_sNp, 0.0);

                for (int qi = 0; qi < dom_Nq; ++qi)
                {
                    qi_pos = (AMREX_SPACEDIM+1)*qi;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[qi_pos+0]; x[0] = fine_J[0]*x[0]+fine_cell_m[0];,
                        x[1] = xptr[qi_pos+1]; x[1] = fine_J[1]*x[1]+fine_cell_m[1];,
                        x[2] = xptr[qi_pos+2]; x[2] = fine_J[2]*x[2]+fine_cell_m[2];
                    )
                    w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*fine_aJ;

                    fine_bf._eval_BF_table_(x);
                    coarse_bf._eval_BF_table_(x);

                    for (int cs = 0; cs < coarse_sNp; ++cs)
                    for (int rs = 0; rs < fine_sNp; ++rs)
                    {
                        I[rs+cs*fine_sNp] += fine_bf.BF[rs]*coarse_bf.BF[cs]*w;
                    }
                }

                for (int rs = 0; rs < (fine_sNp*coarse_sNp); ++rs)
                {
                    fine_matfactory.space_Dom_I_host_mem[space_Dom_I_mem_pos+rs] = I[rs];
                }

                // Update memory
                space_Dom_I_mem_pos += fine_sNp*coarse_sNp;
            }
            // --------------------------------------------------------
        }
    }
    
    // STORE MEMORY INFO
    fine_matfactory.mem_for_space_Dom_I = space_Dom_I_mem_pos;
    // ================================================================

    // COPY MEMORY TO THE DEVICE ======================================
#ifdef AMREX_USE_CUDA
    fine_matfactory.space_Dom_I_device_mem.resize(fine_matfactory.space_Dom_I_host_mem.size());

    Gpu::copy(Gpu::hostToDevice, fine_matfactory.space_Dom_I_host_mem.begin(),
                                 fine_matfactory.space_Dom_I_host_mem.end(),
                                 fine_matfactory.space_Dom_I_device_mem.begin());
#endif
    // ================================================================

    // MULTIPLY BY THE INVERSE OF THE MASS MATRIX =====================
#ifdef AMREX_USE_CUDA
    Real * space_Dom_I_mem_ptr = fine_matfactory.space_Dom_I_device_mem.data();
#else
    Real * space_Dom_I_mem_ptr = fine_matfactory.space_Dom_I_host_mem.data();
#endif

    for (MFIter mfi(fine_mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);
        
        // FINE MESH
        Array4<short const> const & fine_eType_fab = fine_mesh.eType.array(mfi);

        Array4<long const> const & space_MMCh_pos_fab = fine_matfactory.space_MMCh_pos.array(mfi);

        // INTERPOLATION OPERATOR
        Array4<long> const & space_Dom_I_pos_fab = fine_matfactory.space_Dom_I_pos.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS (FINE GRID'S ELEMENT)
            const short fine_etype = fine_eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool fine_elm_is_entire = (fine_etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool fine_elm_is_large = (fine_etype%10 == DG_ELM_TYPE_LARGE);
            const bool fine_elm_is_small = (fine_etype%10 == DG_ELM_TYPE_SMALL);
            const bool fine_elm_is_entire_extended = fine_elm_is_entire && (fine_etype/10 == 1);

            if (fine_elm_is_large || fine_elm_is_small || fine_elm_is_entire_extended)
            {
                // INTERPOLATION OPERATOR
                const long pos = space_Dom_I_pos_fab(i,j,k,dom);
                Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                int BF_i, BF_j, BF_k;
                BF_CELL(i, j, k, fine_etype, BF_i, BF_j, BF_k);

                // MASS MATRIX
                const long MMCh_pos = space_MMCh_pos_fab(BF_i,BF_j,BF_k,dom);
                const Real * space_MMCh_ptr = &space_MMCh_mem_ptr[MMCh_pos];

                // Copy from memory. Remember that the mass matrix is
                // that of an element defined over the
                // [-1,1]^AMREX_SPACEDIM domain.
                for (int rs = 0; rs < (fine_sNp*coarse_sNp); ++rs)
                {
                    I_ptr[rs] *= fine_iaJ;
                }

                // Actual matrix operation (using mass matrix' Cholesky
                // decomposition)
                DG_utils::Cholesky_solve_overwrite(fine_sNp, coarse_sNp, space_MMCh_ptr, I_ptr);
            }
        });
        Gpu::synchronize();
    }
    // ================================================================

}
// ####################################################################

// INTERPOLATE SOLUTION FROM COARSE IMPLICIT MESH TO FINE IMPLICIT MESH
template<int N_PHI, int N_DOM, int N_U>
void InterpolateSolution(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                         const MatrixFactory<N_PHI, N_DOM> & coarse_matfactory,
                         const MultiFab & coarse_X,
                         const IntVect rr,
                         const ImplicitMesh<N_PHI, N_DOM> & fine_mesh,
                         const MatrixFactory<N_PHI, N_DOM> & fine_matfactory,
                         MultiFab & fine_X)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("InterpolateSolution(const ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // QUICK CHECK ====================================================
    CheckLevelsConsistency<N_PHI, N_DOM>(coarse_mesh, rr, fine_mesh);
    // ================================================================

Print() << "HERE WE ARE InterpolateSolution" << std::endl;
exit(-1);

}
// ####################################################################

// RESTRICT SOLUTION FROM FINE IMPLICIT MESH TO COARSE IMPLICIT MESH ##
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RestrictSolution(const ImplicitMesh<N_PHI, N_DOM> & fine_mesh,
                      const MatrixFactory<N_PHI, N_DOM> & fine_matfactory,
                      const MultiFab & fine_X,
                      const IntVect rr,
                      const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                      const MatrixFactory<N_PHI, N_DOM> & coarse_matfactory,
                      const charMultiFab & coarse_tags,
                      MultiFab & coarse_X,
                      const IBVP_CLASS & IBVP,
                      const bool stop_flag = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RestrictSolution(const ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // QUICK CHECK ====================================================
    CheckLevelsConsistency<N_PHI, N_DOM>(coarse_mesh, rr, fine_mesh);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID (COARSE MESH)
    const Real coarse_iaJ = coarse_mesh.iaJ;
    
    // GRID (FINE MESH)
    const Real fine_aJ = fine_mesh.aJ;
    
    // STANDARD ELEMENT (COARSE MESH)
    const int coarse_sNp = coarse_matfactory.cube.Np;
    
    // STANDARD ELEMENT (FINE MESH)
    const int fine_sNp = fine_matfactory.cube.Np;

    // IMPLICIT COARSE MESH
#ifdef AMREX_USE_CUDA
        const Real * coarse_space_MMCh_mem_ptr = coarse_matfactory.space_MMCh_device_mem.data();
#else
        const Real * coarse_space_MMCh_mem_ptr = coarse_matfactory.space_MMCh_host_mem.data();
#endif

    // IMPLICIT FINE MESH
#ifdef AMREX_USE_CUDA
        const Real * fine_space_MMCh_mem_ptr = fine_matfactory.space_MMCh_device_mem.data();
        const Real * space_Dom_I_mem_ptr = fine_matfactory.space_Dom_I_device_mem.data();
#else
        const Real * fine_space_MMCh_mem_ptr = fine_matfactory.space_MMCh_host_mem.data();
        const Real * space_Dom_I_mem_ptr = fine_matfactory.space_Dom_I_host_mem.data();
#endif
    // ================================================================

    // VARIABLES ======================================================
    
    // AUXILIARY MULTIFAB
    MultiFab aux_fine_X(fine_mesh.cc_ba, fine_mesh.dm, fine_X.nComp(), fine_X.nGrow());
    // ================================================================

    // INITIALIZATION =================================================
    aux_fine_X = 0.0;
    // ================================================================

    // LOOP OVER THE FINE MESH' ELEMENTS ==============================
    // 
    // We first compute: fMX = fM*fX
    // 
    // ================================================================
    for (MFIter mfi(fine_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = fine_mesh.eType.array(mfi);
        Array4<long const> const & space_MMCh_pos_fab = fine_matfactory.space_MMCh_pos.array(mfi);
        Array4<Real const> const & X_fab = fine_X.array(mfi);
        Array4<Real> const & aux_X_fab = aux_fine_X.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            if (elm_is_valid)
            {
                // MASS MATRIX
                const long pos = space_MMCh_pos_fab(i,j,k,dom);
                const Real * space_MMCh_ptr = &fine_space_MMCh_mem_ptr[pos];

                // HERE WE ARE RELYING ON THE FACT THAT fine_sNp = 1
                const Real c2 = space_MMCh_ptr[0]*space_MMCh_ptr[0];

                aux_X_fab(i,j,k,ru) = c2*fine_aJ*X_fab(i,j,k,ru);
            }
        });
        Gpu::synchronize();
    }
    aux_fine_X.FillBoundary(fine_mesh.geom.periodicity());

    if (fine_sNp != 1)
    {
        Print() << "ERROR: AMReX_DG_Solution_AMR.H - RestrictSolution" << std::endl;
        Print() << "| fine_sNp != 1 must be implemented." << std::endl;
        Print() << "| In the routine above (see where this message is raised) we need to" << std::endl;
        Print() << "| multiply by the mass matrix by we have its Cholesky decomposition" << std::endl;
        Print() << "| only." << std::endl;
        exit(-1);
    }
    // ================================================================

    // LOOP OVER THE COARSE MESH' ELEMENTS ============================
    // 
    // We compute: cX = cM^{-1}*fIc^T*fMX
    // 
    // ================================================================
    for (MFIter mfi(coarse_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        // FINE MESH
        Array4<short const> const & fine_eType_fab = fine_mesh.eType.array(mfi);
        Array4<Real const> const & aux_fine_X_fab = aux_fine_X.array(mfi);

        // INTERPOLATOR OPERATOR
        Array4<long const> const & space_Dom_I_pos_fab = fine_matfactory.space_Dom_I_pos.array(mfi);

        // COARSE MESH
        Array4<short const> const & coarse_eType_fab = coarse_mesh.eType.array(mfi);
        Array4<char const> const & coarse_tags_fab = coarse_tags.array(mfi);
        Array4<long const> const & coarse_space_MMCh_pos_fab = coarse_matfactory.space_MMCh_pos.array(mfi);
        Array4<Real> const & coarse_X_fab = coarse_X.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int ci, int cj, int ck, int ru) noexcept
        {
            // PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short coarse_etype = coarse_eType_fab(ci,cj,ck,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool coarse_elm_is_entire = (coarse_etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool coarse_elm_is_large = (coarse_etype%10 == DG_ELM_TYPE_LARGE);
            const bool coarse_elm_is_valid = coarse_elm_is_entire || coarse_elm_is_large;
            const bool coarse_elm_is_tagged = (coarse_tags_fab(ci,cj,ck,dom) == TAGGED_ELEMENT);

            if (coarse_elm_is_valid && coarse_elm_is_tagged)
            {
#if (AMREX_SPACEDIM == 2)
                const Dim3 lo = {ci*rr[0], cj*rr[1], 0};
                const Dim3 hi = {(ci+1)*rr[0], (cj+1)*rr[1], 1};
#endif
#if (AMREX_SPACEDIM == 3)
                const Dim3 lo = {ci*rr[0], cj*rr[1], ck*rr[2]};
                const Dim3 hi = {(ci+1)*rr[0], (cj+1)*rr[1], (ck+1)*rr[2]};
#endif
                // LOCAL VARIABLES
                Real Xh[DG_SOL_MAX_SPACE_Np];

                // INIT COARSE SOLUTION VECTOR
                for (int rs = 0; rs < coarse_sNp; ++rs)
                {
                    Xh[rs] = 0.0;
                }

                // LOOP OVER THE CELLS OF THE SUBGRID
                for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j < hi.y; ++j)
                for (int i = lo.x; i < hi.x; ++i)
                {
                    const int fine_etype = fine_eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    const bool fine_elm_is_not_empty = (fine_etype != DG_ELM_TYPE_EMPTY);

                    int BF_i, BF_j, BF_k;
                    BF_CELL(i, j, k, fine_etype, BF_i, BF_j, BF_k);

                    if (fine_elm_is_not_empty)
                    {
                        const long pos = space_Dom_I_pos_fab(i,j,k,dom);
                        const Real * I_ptr = &space_Dom_I_mem_ptr[pos];

                        for (int rs = 0; rs < coarse_sNp; ++rs)
                        for (int cs = 0; cs < fine_sNp; ++cs)
                        {
                             Xh[rs] += I_ptr[cs+rs*fine_sNp]*aux_fine_X_fab(BF_i,BF_j,BF_k,cs+ru*fine_sNp);
                        }
                    }
                }

                // MULTIPLY BY INVERSE MASS MATRIX
                {
                    const long pos = coarse_space_MMCh_pos_fab(ci,cj,ck,dom);
                    const Real * coarse_space_MMCh_ptr = &coarse_space_MMCh_mem_ptr[pos];

                    // Remember that the mass matrix is that of an
                    // element defined over the
                    // [-1,1]^AMREX_SPACEDIM domain.
                    for (int rs = 0; rs < coarse_sNp; ++rs)
                    {
                        Xh[rs] *= coarse_iaJ;
                    }

                    // Actual matrix operation (using mass matrix'
                    // Cholesky decomposition)
                    DG_utils::Cholesky_solve_overwrite(coarse_sNp, 1, coarse_space_MMCh_ptr, Xh);

                    // REPLACE COARSE SOLUTION
                    for (int rs = 0; rs < coarse_sNp; ++rs)
                    {
                        coarse_X_fab(ci,cj,ck,rs+ru*coarse_sNp) = Xh[rs];
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    coarse_X.FillBoundary(coarse_mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

} // namespace AMR
} // namespace DG
} // namespace amrex

#endif