//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_AMR.H
 * \brief Basis data structures and routines for AMR computations.
*/

#ifndef AMREX_DG_AMR_H_
#define AMREX_DG_AMR_H_

#include <AMReX_MultiFabUtil.H>
#include <AMReX_AmrCore.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_dG_Utils.H>
#include <AMReX_dG_Mesh.H>
#include <AMReX_dG_Mesh_Output.H>
#include <AMReX_dG_Solution.H>
#include <AMReX_dG_Solution_Output.H>

namespace amrex
{
namespace dG
{

namespace amr_utils
{
// AUXILIARY OPERATIONS ###############################################
// ####################################################################



// INTERPOLATION OPERATIONS ###########################################
/**
 * \brief Eval and store the interpolation operator between a fine and a coarse mesh.
 *
 * \param[in] c_geom: coarse amrex Geometry object.
 * \param[in] c_mesh: coarse Mesh object.
 * \param[in] c_solution: coarse Solution object.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine amrex Geometry object.
 * \param[in] f_mesh: fine Mesh object.
 * \param[inout] f_solution: fine Solution object; it will contain the interpolation operator.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void eval_space_interpolation(const Geometry & c_geom, const Mesh & c_mesh, const Solution & c_solution,
                              const IntVect rr,
                              const Geometry & f_geom, const Mesh & f_mesh, Solution & f_solution,
                              const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("amrex::dG::amr_utils::eval_space_interpolation(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> c_dx = c_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> f_dx = f_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = f_geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int c_sp = c_solution.params.space_p;
    const int f_sp = f_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    const int f_sNp = AMREX_D_TERM((1+f_sp),*(1+f_sp),*(1+f_sp));

    // QUADRATURE
    const int c_qo_reg = c_mesh.params.quadrature_order_regular_elements;
    const int f_qo_reg = f_mesh.params.quadrature_order_regular_elements;
    const int qo_reg = amrex::max(c_qo_reg, f_qo_reg);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    StandardRectangle<AMREX_SPACEDIM> f_std_elm;
    long elm_I_mem_pos;

    sMultiFab safe_c_elm_type;
    const sMultiFab * safe_c_elm_type_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(c_solution.elm_type, f_solution.elm_type))
    {
        safe_c_elm_type_ptr = &c_solution.elm_type;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_mesh.ba, rr);
        const DistributionMapping & f_dm = f_mesh.dm;
        const int c_elm_type_n_comp = c_solution.elm_type.n_comp;
        const IntVect c_elm_type_n_grow = c_solution.elm_type.n_grow;

        safe_c_elm_type.define(safe_c_ba, f_dm, c_elm_type_n_comp, c_elm_type_n_grow);
        safe_c_elm_type.ParallelCopy(c_solution.elm_type, 0, 0, c_elm_type_n_comp, c_elm_type_n_grow, c_elm_type_n_grow, c_geom.periodicity());

        safe_c_elm_type_ptr = &safe_c_elm_type;
    }
    // ----------------------------------------------------------------

    // INIT THE MULTIFAB ----------------------------------------------
    f_solution.elm_I_info.define(f_mesh.ba, f_mesh.dm, n_domains, f_mesh.dom_data_n_grow);
    f_solution.elm_I_info = -1L;
    // ----------------------------------------------------------------

    // INITIALIZE THE STANDARD ELEMENT --------------------------------
    f_std_elm.set_quadrature(&f_dx[0], qo_reg);
    f_std_elm.set_interpolation_operator(c_sp, rr, f_sp);
    // ----------------------------------------------------------------

    // FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO ---------
    elm_I_mem_pos = 0L;

    {
        const int size = f_sNp*c_sNp*(AMREX_D_TERM(rr[0],*rr[1],*rr[2]));

        // Resize memory
        f_solution.elm_I_host_mem.resize(size);

        // Store data
        std::copy(f_std_elm.I.begin(), f_std_elm.I.end(), f_solution.elm_I_host_mem.data());

        // Update memory
        elm_I_mem_pos += size;
    }
    // ----------------------------------------------------------------

    // EVAL AND STORE THE INTERPOLATION OPERATOR ----------------------
    Real const * f_cell_dom_quad_mem_ptr = f_mesh.cell_dom_quad_host_mem.data();

    for (MFIter mfi(f_solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        // Coarse mesh
        Array4<short const> const & c_elm_type_fab = safe_c_elm_type_ptr->array(mfi);
        
        // Fine mesh
        Array4<long const> const & f_cell_dom_quad_info_fab = f_mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & f_elm_type_fab = f_solution.elm_type.array(mfi);

        // Interpolation operator
        Array4<long> const & elm_I_info_fab = f_solution.elm_I_info.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // INDICES OF THE COARSE CELL
#if (AMREX_SPACEDIM == 1)
Print() << "eval_space_interpolation_unsafe: AMREX_SPACEDIM == 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int c_i = (i-pi)/rr[0];
            const int c_j = (j-pj)/rr[1];
            const int c_k = k;
            const int I_pos = (pi+pj*rr[0])*f_sNp*c_sNp;
#endif
#if (AMREX_SPACEDIM == 3)
            const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
            const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
            const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
            const int c_i = (i-pi)/rr[0];
            const int c_j = (j-pj)/rr[1];
            const int c_k = (k-pk)/rr[2];
            const int I_pos = (pi+pj*rr[0]+pk*rr[0]*rr[1])*f_sNp*c_sNp;
#endif

            // LOCAL PARAMETERS
            const Real f_cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*f_dx[0],
                                                                 prob_lo[1]+j*f_dx[1],
                                                                 prob_lo[2]+k*f_dx[2])};
            const short f_etype = f_elm_type_fab(i,j,k,ELM_TYPE(dom));
            const short c_etype = c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom));

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], w;
            Real I[__DG_MAX_SPACE_Np__*__DG_MAX_SPACE_Np__];

            // BASIS FUNCTIONS
            space_elm_bf f_bf(&prob_lo[0], &f_dx[0], i, j, k, f_etype, f_sp);
            space_elm_bf c_bf(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_etype, c_sp);

            // EMPTY ELEMENT
            if (ELM_IS_EMPTY(f_etype))
            {
                // Store memory offset
                elm_I_info_fab(i,j,k,dom) = -1L;
            }
            // ENTIRE ELEMENT
            else if (ELM_IS_ENTIRE(f_etype) && !ELM_IS_SMALL(c_etype))
            {
                // Store memory offset
                elm_I_info_fab(i,j,k,dom) = I_pos;
            }
            else if (ELM_IS_LARGE(f_etype) || ELM_IS_SMALL(f_etype) || (ELM_IS_ENTIRE(f_etype) && ELM_IS_SMALL(c_etype)))
            {
                // DOMAIN QUADRATURE INFO
                const int dom_Nq = f_cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = f_cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &f_cell_dom_quad_mem_ptr[pos];

                // Store memory offset
                elm_I_info_fab(i,j,k,dom) = elm_I_mem_pos;

                // Expand memory
                f_solution.elm_I_host_mem.resize(elm_I_mem_pos+f_sNp*c_sNp);

                // Store data
                std::fill(I, I+f_sNp*c_sNp, 0.0);
                for (int q = 0; q < dom_Nq; ++q)
                {
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+f_cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+f_cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+f_cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    f_bf.eval_bf_only_table(x);
                    c_bf.eval_bf_only_table(x);

                    for (int cs = 0; cs < c_sNp; ++cs)
                    for (int rs = 0; rs < f_sNp; ++rs)
                    {
                        I[rs+cs*f_sNp] += f_bf.BF[rs]*c_bf.BF[cs]*w;
                    }
                }

                for (int rs = 0; rs < (f_sNp*c_sNp); ++rs)
                {
                    f_solution.elm_I_host_mem[elm_I_mem_pos+rs] = I[rs];
                }

                // Update memory
                elm_I_mem_pos += f_sNp*c_sNp;
            }
        }
    }
    // ----------------------------------------------------------------

#ifdef AMREX_USE_GPU
    // COPY TO DEVICE -------------------------------------------------
    f_solution.elm_I_dev_mem.resize(f_solution.elm_I_host_mem.size());

    Gpu::copy(Gpu::hostToDevice, f_solution.elm_I_host_mem.begin(), f_solution.elm_I_host_mem.end(), f_solution.elm_I_dev_mem.begin());
    // ----------------------------------------------------------------
#endif
}

/**
 * \brief Interpolate from a coarse solution to a fine solution.
 *
 * \param[in] c_geom: coarse amrex Geometry object.
 * \param[in] c_solution: coarse Solution object.
 * \param[in] c_X: MultiFab containing coarse solution.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine amrex Geometry object.
 * \param[in] f_solution: fine Solution object.
 * \param[out] f_X: MultiFab that will contain the fine solution.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void interpolate(const Geometry & c_geom, const Solution & c_solution, const MultiFab & c_X,
                 const IntVect rr,
                 const Geometry & f_geom, const Solution & f_solution, MultiFab & f_X,
                 const IBVP & ibvp,
                 const bool use_fabbox = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("amrex::dG::amr_utils::interpolate(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // NUMBER OF UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

    // ORDER
    const int c_sp = c_solution.params.space_p;
    const int f_sp = f_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    const int f_sNp = AMREX_D_TERM((1+f_sp),*(1+f_sp),*(1+f_sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(c_X, f_X))
    {
        safe_c_X_ptr = &c_X;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_X.boxarray, rr);
        const DistributionMapping & f_dm = f_X.distributionMap;
        const int c_X_n_comp = c_X.n_comp;
        const IntVect c_X_n_grow = c_X.n_grow;

        safe_c_X.define(safe_c_ba, f_dm, c_X_n_comp, c_X_n_grow);
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_geom.periodicity());

        safe_c_X_ptr = &safe_c_X;
    }
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    f_X = 0.0;
    // ----------------------------------------------------------------

    // LOOP OVER THE FINE MESH' ELEMENTS ------------------------------
#ifdef AMREX_USE_GPU
    const Real * elm_I_mem_ptr = f_solution.elm_I_dev_mem.data();
#else
    const Real * elm_I_mem_ptr = f_solution.elm_I_host_mem.data();
#endif

    for (MFIter mfi(f_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = use_fabbox ? mfi.fabbox() : mfi.validbox();

        // Coarse mesh
        Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);

        // Interpolation operator
        Array4<long const> const & elm_I_info_fab = f_solution.elm_I_info.array(mfi);

        // Fine mesh
        Array4<Real> const & f_X_fab = f_X.array(mfi);

        ParallelFor(bx, n_unknown_fields,
        [=] AMREX_GPU_DEVICE (int f_i, int f_j, int f_k, int ru) noexcept
        {
            // ELEMENT TYPE (USING elm_I_info_fab)
            const int dom = ibvp.unknown_field_parent_domain(ru);
            const long pos = elm_I_info_fab(f_i,f_j,f_k,dom);

            if (pos != -1L)
            {
                // LOCAL PARAMETERS
                const Real * I_ptr = &elm_I_mem_ptr[pos];

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
    f_X.FillBoundary(f_geom.periodicity());
    // ----------------------------------------------------------------

    f_solution.add_small_elements_contribution(f_geom, f_X, ibvp);
    f_solution.multiply_by_inverse_space_mass_matrix(f_geom, f_X, ibvp, use_fabbox);
    f_solution.copy_solution_to_small_elements(f_geom, f_X, ibvp);
}

/**
 * \brief Remake an existing solution using information from a coarse solution.
 *
 * \param[in] t: time.
 * \param[in] c_geom: coarse amrex Geometry object.
 * \param[in] c_solution: coarse Solution object.
 * \param[in] c_mesh: coarse Mesh object.
 * \param[in] c_X: MultiFab containing coarse solution.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine amrex Geometry object.
 * \param[in] f_mesh: fine Mesh object.
 * \param[in] f_solution: fine Solution object.
 * \param[inout] f_X: MultiFab that will contain the fine solution.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void remake_solution(const Real t,
                     const Geometry & c_geom, const Solution & c_solution, const Mesh & c_mesh, const MultiFab & c_X,
                     const IntVect rr,
                     const Geometry & f_geom, const Solution & f_solution, const Mesh & f_mesh, MultiFab & f_X,
                     const IBVP & ibvp,
                     const bool include_ghost_cells = false)
{
    // VARIABLES ------------------------------------------------------
    MultiFab f_X_new(f_mesh.ba, f_mesh.dm, f_X.n_comp, f_X.n_grow);
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    f_X_new = 0.0;
    // ----------------------------------------------------------------

    // INTERPOLATE FROM COARSE ----------------------------------------
    interpolate(c_geom, c_solution, c_X, rr, f_geom, f_solution, f_X_new, ibvp, include_ghost_cells);
    // ----------------------------------------------------------------

    // OVERWRITE WITH OLD SOLUTION ------------------------------------
    f_X_new.ParallelCopy(f_X, 0, 0, f_X.n_comp, f_X.n_grow, f_X.n_grow, f_geom.periodicity());
    f_X_new.FillBoundary(f_geom.periodicity());
    f_solution.copy_solution_to_small_elements(f_geom, f_X_new, ibvp);
    // ----------------------------------------------------------------

    // SWAP NEW AND OLD -----------------------------------------------
    std::swap(f_X, f_X_new);
    // ----------------------------------------------------------------
}
// ####################################################################



// RESTRICTION OPERATIONS #############################################
/**
 * \brief Restrict from a fine solution to a coarse solution.
 *
 * \param[in] f_geom: fine amrex Geometry object.
 * \param[in] f_solution: fine Solution object.
 * \param[in] f_X: MultiFab containing the fine solution.
 * \param[in] rr: refinement ratio.
 * \param[in] c_geom: coarse amrex Geometry object.
 * \param[in] c_solution: coarse Solution object.
 * \param[inout] c_X: MultiFab that will contain the updated coarse solution.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void restrict(const Geometry & f_geom, const Solution & f_solution, const MultiFab & f_X,
              const IntVect rr,
              const Geometry & c_geom, const Solution & c_solution, MultiFab & c_X, const iMultiFab & c_mask,
              const IBVP & ibvp,
              const bool use_fabbox = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("amrex::dG::amr_utils::restrict(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // NUMBER OF UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

    // ORDER
    const int f_sp = f_solution.params.space_p;
    const int c_sp = c_solution.params.space_p;
    const int f_sNp = AMREX_D_TERM((1+f_sp),*(1+f_sp),*(1+f_sp));
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));

    // MULTIFABS
    const BoxArray coarsened_f_ba = coarsen(f_X.boxarray, rr);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    MultiFab RX(coarsened_f_ba, f_X.distributionMap, c_X.n_comp, c_X.n_grow);
    MultiFab safe_RX;
    MultiFab * safe_RX_ptr;
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    RX = 0.0;
    // ----------------------------------------------------------------

    // AT THE COARSE MESH COVERED BY THE FINE MESH --------------------
    //
    // Compute: RX := I^T*f_X
    //
    // ----------------------------------------------------------------
#ifdef AMREX_USE_GPU
    const Real * elm_I_mem_ptr = f_solution.elm_I_dev_mem.data();
#else
    const Real * elm_I_mem_ptr = f_solution.elm_I_host_mem.data();
#endif

    for (MFIter mfi(RX); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        // Fine mesh
        Array4<Real const> const & f_X_fab = f_X.array(mfi);

        // Interpolation operator
        Array4<long const> const & elm_I_info_fab = f_solution.elm_I_info.array(mfi);

        // Coarse mesh
        Array4<Real> const & ITX_fab = RX.array(mfi);

        ParallelFor(bx, n_unknown_fields,
        [=] AMREX_GPU_DEVICE (int c_i, int c_j, int c_k, int ru) noexcept
        {
            // DOMAIN / FINER MESH GRID
            const int dom = ibvp.unknown_field_parent_domain(ru);
#if (AMREX_SPACEDIM == 1)
            const Dim3 f_lo = {c_i*rr[0], 0, 0};
            const Dim3 f_hi = {(c_i+1)*rr[0], 1, 1};
#endif
#if (AMREX_SPACEDIM == 2)
            const Dim3 f_lo = {c_i*rr[0], c_j*rr[1], 0};
            const Dim3 f_hi = {(c_i+1)*rr[0], (c_j+1)*rr[1], 1};
#endif
#if (AMREX_SPACEDIM == 3)
            const Dim3 f_lo = {c_i*rr[0], c_j*rr[1], c_k*rr[2]};
            const Dim3 f_hi = {(c_i+1)*rr[0], (c_j+1)*rr[1], (c_k+1)*rr[2]};
#endif

            // LOOP OVER THE CELLS OF THE FINER MESH
            for (int f_k = f_lo.z; f_k < f_hi.z; ++f_k)
            for (int f_j = f_lo.y; f_j < f_hi.y; ++f_j)
            for (int f_i = f_lo.x; f_i < f_hi.x; ++f_i)
            {
                // ELEMENT TYPE (USING elm_I_info_fab)
                const long pos = elm_I_info_fab(f_i,f_j,f_k,dom);

                if (pos != -1L)
                {
                    // LOCAL PARAMETERS
                    const Real * I_ptr = &elm_I_mem_ptr[pos];

                    for (int rs = 0; rs < c_sNp; ++rs)
                    for (int cs = 0; cs < f_sNp; ++cs)
                    {
                        ITX_fab(c_i,c_j,c_k,rs+ru*c_sNp) += I_ptr[cs+rs*f_sNp]*f_X_fab(f_i,f_j,f_k,cs+ru*f_sNp);
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    RX.FillBoundary(c_geom.periodicity());
    // ----------------------------------------------------------------

    // AT THE COARSE MESH COVERED BY THE FINE MESH --------------------
    // 
    // Compute: RX := c_MM^(-1)*RX
    // 
    // ----------------------------------------------------------------
    if (isMFIterSafe(f_X, c_X))
    {
        safe_RX_ptr = &RX;
    }
    else
    {
        safe_RX.define(c_X.boxarray, c_X.distributionMap, RX.n_comp, RX.n_grow);
        safe_RX = 0.0;
        safe_RX.ParallelCopy(RX, 0, 0, RX.n_comp, RX.n_grow, RX.n_grow, c_geom.periodicity());

        safe_RX_ptr = &safe_RX;
    }

    c_solution.add_small_elements_contribution(c_geom, *safe_RX_ptr, ibvp);
    c_solution.multiply_by_inverse_space_mass_matrix(c_geom, *safe_RX_ptr, ibvp);
    c_solution.copy_solution_to_small_elements(c_geom, *safe_RX_ptr, ibvp);
    // ----------------------------------------------------------------

    // REPLACE RESTRICTED SOLUTION IN c_X -----------------------------
    for (MFIter mfi(c_X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & RX_fab = safe_RX_ptr->array(mfi);
        Array4<int const> const & c_mask_fab = c_mask.array(mfi);

        Array4<Real> const & c_X_fab = c_X.array(mfi);

        ParallelFor(bx, c_X.n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int c) noexcept
        {
            if (CELL_IS_MASKED(c_mask_fab(i,j,k)))
            {
                c_X_fab(i,j,k,c) = RX_fab(i,j,k,c);
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------
}
// ####################################################################

} // amr_utils


namespace amr
{
// ADAPTIVE MESH REFINEMENT CLASS #####################################
struct SinglePatch
:
public AmrCore
{
    // DATA MEMBERS ===================================================
    InputReaderSinglePatch inputs;
    Vector<Mesh *> meshes;
    Vector<Solution *> solutions;
    
    Vector<iMultiFab> masks;
    // ================================================================


    // CONSTRUCTOR ====================================================
    SinglePatch();
    // ================================================================


    // DESTRUCTOR =====================================================
    ~SinglePatch();
    // ================================================================


    // INITIALIZATION =================================================
    void init_inputs();
    void init();
    // ================================================================


    // READERS ========================================================
    bool advance_in_time_continues(const int n, const Real t) const;

    int get_largest_dG_space_p() const;
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.MakeNewLevelFromScratch" << std::endl;
exit(-1);
    }

    /**
     * \brief Update the masks between levels.
     *
    */
    void update_masks()
    {
        for (int lev = 1; lev <= this->finest_level; ++lev)
        {
            const sMultiFab & c_cell_type = this->meshes[lev-1]->cell_type;
            const sMultiFab & f_cell_type = this->meshes[lev]->cell_type;
            
            this->masks[lev-1] = makeFineMask(c_cell_type, f_cell_type,
                                              c_cell_type.n_grow, this->ref_ratio[lev-1],
                                              this->Geom(lev-1).periodicity(),
                                              __DG_CELL_NOT_MASKED__, __DG_CELL_MASKED__);
        }
    }
    // ================================================================


    // TAG CELLS FOR REFINEMENT =======================================
    virtual void ErrorEst(int /*lev*/, TagBoxArray & /*tags*/, Real /*time*/, int /*ngr*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.ErrorEst" << std::endl;
exit(-1);
    }

    virtual void ManualTagsPlacement(int lev, TagBoxArray & tags, const Vector<IntVect> & bf_lev) override
    {
        // PARAMETERS -------------------------------------------------
        const Solution & solution = *this->solutions[lev];
        const int n_domains = solution.elm_type.n_comp/__DG_ELM_TYPE_N_COMP_PER_DOM__;
        const char tagged = TagBox::BUF;
        // ------------------------------------------------------------

        // TAG SMALL CELLS THAT ARE MERGED WITH TAGGED CELLS ----------
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);

            Array4<char> const & tags_fab = tags.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    // ELEMENT TYPE
                    const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_not_tagged = (tags_fab(i,j,k) != tagged);

                    if (ELM_IS_SMALL(etype) && elm_is_not_tagged)
                    {
                        // LOCAL PARAMETERS
                        const int merged_b = etype/10;

                        // LOCAL VARIABLES
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        bool nbr_is_tagged;

                        NBR_CELL(i, j, k, merged_b, nbr_i, nbr_j, nbr_k, nbr_b);

                        nbr_is_tagged = (tags_fab(nbr_i,nbr_j,nbr_k) == tagged);

                        if (nbr_is_tagged)
                        {
                            tags_fab(i,j,k) = tagged;
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        tags.FillBoundary(this->geom[lev].periodicity());
        // ------------------------------------------------------------
    }
    // ================================================================

    
    // MAKE NEW LEVEL FROM A COARSE LEVEL =============================
    virtual void MakeNewLevelFromCoarse(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.MakeNewLevelFromCoarse" << std::endl;
exit(-1);
    }
    // ================================================================


    // REMAKE AN EXISTING LEVEL =======================================
    virtual void RemakeLevel(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.RemakeLevel" << std::endl;
exit(-1);
    }
    // ================================================================


    // CLEAR LEVEL ====================================================
    virtual void ClearLevel(int /*lev*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.ClearLevel" << std::endl;
exit(-1);
    }
    // ================================================================


    // CHECK QUADRATURE RULES =========================================
    void check_quadrature_rules(const int n_domains,
                                const Real * exact_volume,
                                const Real * exact_surface) const;
    void check_quadrature_rules(const Real exact_volume,
                                const Real exact_surface) const;
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error among the levels using the L_inf norm.
     *
     * \param[in] t: time.
     * \param[in] X: a vector of MultiFabs containing the solution coefficients.
     * \param[out] err: L_inf-based error computed over the levels.
     * \param[out] err_norm: L_inf-based error normalization computed over the levels.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_error_L_inf(const Real t, const Vector<MultiFab> & X, Real & err, Real & err_norm, const IBVP & ibvp) const
    {
        err = 0.0;
        err_norm = 0.0;

        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const Solution & solution = *this->solutions[lev];
            const iMultiFab & mask = this->masks[lev];
            
            Real err_lev, err_norm_lev;

            if ((solution.params.space_p == 0) && solution.params.FV_is_active)
            {
                solution.eval_FV_error_L_inf(t, this->geom[lev], mesh, X[lev], mask, err_lev, err_norm_lev, ibvp);
            }
            else
            {
                solution.eval_error_L_inf(t, this->geom[lev], mesh, X[lev], mask, err_lev, err_norm_lev, ibvp);
            }

            err = amrex::max(err, err_lev);
            err_norm = amrex::max(err_norm, err_norm_lev);
        }
    }
    // ================================================================


    // INTERPOLATION OPERATIONS =======================================
    /**
     * \brief Eval the space interpolation operator at the level lev.
     *
     * \param[in] lev: considered level (Must be > 0).
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_space_interpolation(const int lev, const IBVP & ibvp)
    {
        const Geometry & c_geom = this->geom[lev-1];
        const Mesh & c_mesh = *this->meshes[lev-1];
        const Solution & c_solution = *this->solutions[lev-1];
        const IntVect rr = this->refRatio(lev-1);
        const Geometry & f_geom = this->geom[lev];
        const Mesh & f_mesh = *this->meshes[lev];
        Solution & f_solution = *this->solutions[lev];

        amr_utils::eval_space_interpolation(c_geom, c_mesh, c_solution,
                                            rr,
                                            f_geom, f_mesh, f_solution,
                                            ibvp);
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void make_step_output_folder(const int n, const Real t) const;
    
    /**
     * \brief Export meshes to VTK format.
     *
     * \param[in] n: time step index.
     * \param[in] filename_root: root of the filename containing the mesh output.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_mesh_to_VTK(const int n, const std::string & filename_root, const IBVP & ibvp) const
    {
        // EXPORT THE MESHES ------------------------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            mesh_io::export_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                   this->geom[lev], mesh, mask, ibvp);
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Export the meshes quadrature points to VTK format.
     *
     * \param[in] n: time step index.
     * \param[in] filename_root: root of the filename containing the mesh output.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_quadrature_points_to_VTK(const int n, const std::string & filename_root, const IBVP & ibvp) const
    {
        // EXPORT THE MESHES QUADRATURE POINTS ------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            mesh_io::export_quadrature_points_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                     this->geom[lev], mesh, mask, ibvp);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
};
// ####################################################################

} // namespace amr
} // namespace dG
} // namespace amrex

#endif