//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution_RungeKutta.H
 * \brief Functions to advance solutions in time using Runge-Kutta algorithms.
*/

#ifndef AMREX_DG_SOLUTION_RUNGE_KUTTA_H_
#define AMREX_DG_SOLUTION_RUNGE_KUTTA_H_

#include <AMReX_VTK_Utils.H>

#include <AMReX_dG_Solution.H>

namespace amrex
{
namespace dG
{

namespace hyperbolic_pde
{
namespace explicit_RK
{

// DG METHOD: DOMAIN CONTRIBUTION #####################################
/**
 * \brief Add the domain contribution to the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{V} BF(x)_{,k}^T F_k(t, x, U) - BF(x)^T R(t, x, U)
 *
*/
template <typename IBVP>
void add_dom_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::add_dom_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS --------------------------------------
#ifdef AMREX_USE_GPU
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
                Real AMREX_D_DECL(F1[__DG_MAX_N_UNKNOWN_FIELDS__], F2[__DG_MAX_N_UNKNOWN_FIELDS__], F3[__DG_MAX_N_UNKNOWN_FIELDS__]);
                //Real R[__DG_MAX_N_UNKNOWN_FIELDS__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                // EVAL THE INTEGRAL
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // Solution at (t, x)
                    bfu.eval_bf_table(x);
                    bfu.eval_unsafe(u_lo, u_hi, U);

                    // Fluxes/Reactive terms as functions of (t, x, U)
                    //ibvp.eval_FR(dom, t, x, U, AMREX_D_DECL(F1, F2, F3), R);
                    ibvp.eval_F(dom, t, x, U, AMREX_D_DECL(F1, F2, F3));

                    // Integral contribution
                    for (int ru = u_lo; ru < u_hi; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*F1[ru];,
                            integrand += bfu.dBFdx2[rs]*F2[ru];,
                            integrand += bfu.dBFdx3[rs]*F3[ru];
                        )
                        //integrand -= bfu.BF[rs]*R[ru];
                        
                        dXdt_fab(i,j,k,rs+ru*sNp) += integrand*w;
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

/* // hpAMR */
    // CONCENTRATED SOURCE TERM ---------------------------------------
    const Real th = 10.0*M_PI/180.0;
    const Real tth = std::tan(th);
    const Real x1S = 1720.0;
    const GpuArray<Real, AMREX_SPACEDIM> xs = {AMREX_D_DECL(x1S, 2000.0+tth*x1S, 0.0)};
    const Real alpha = 1.0;

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(cell_lo[0]+dx[0],
                                                                   cell_lo[1]+dx[1],
                                                                   cell_lo[2]+dx[2])};

                // FLAG TO CHECK WHETHER WE ADD THE CONTRIBUTION
                Real cond;
                AMREX_D_TERM
                (
                    cond =         ((cell_lo[0]-1.0e-5*dx[0]) <= xs[0]) && (xs[0] <= (cell_hi[0]+1.0e-5*dx[0]));,
                    cond = cond && ((cell_lo[1]-1.0e-5*dx[1]) <= xs[1]) && (xs[1] <= (cell_hi[1]+1.0e-5*dx[1]));,
                    cond = cond && ((cell_lo[2]-1.0e-5*dx[2]) <= xs[2]) && (xs[2] <= (cell_hi[2]+1.0e-5*dx[2]));
                )

                if (cond)
                {
//Print(ParallelDescriptor::MyProc()) << "Verifying cond: " << i << " " << j << std::endl;

                    // LOCAL VARIABLES
                    Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
                    Real S[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // Solution at (t, xs)
                    bfu.eval(&xs[0], u_lo, u_hi, U);

                    // Source as a function of (t, xs, U)
                    ibvp.eval_S(dom, t, &xs[0], U, S);

                    // Contribution
                    for (int ru = u_lo; ru < u_hi; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        dXdt_fab(i,j,k,rs+ru*sNp) += bfu.BF[rs]*alpha*S[ru];
                    }   
                }
            }
        });
        Gpu::synchronize();
    }
//exit(-1);
    // ----------------------------------------------------------------
/**/
}
// ####################################################################



// DG METHOD: BOUNDARY CONTRIBUTION ###################################
template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_BCS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int sNp = bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        bfu.eval(x, u_lo, u_hi, U);

        // Numerical flux as a functions of (t, x, un, U)
        ibvp.eval_NFn_BCs(dom, t, x, un, U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        for (int rs = 0; rs < sNp; ++rs)
        {
            integrand = bfu.BF[rs]*NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+rs+ru*sNp) += integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & m_bfu, space_elm_bfx & p_bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int m_sNp = m_bfu.Np;
    const int p_sNp = p_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_bfu.eval(x, u_lo, u_hi, m_U);
        p_bfu.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            for (int rs = 0; rs < m_sNp; ++rs)
            {
                integrand = m_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*m_sNp) += integrand*w;
            }
            for (int rs = 0; rs < p_sNp; ++rs)
            {
                integrand = p_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,FX_offset+rs+ru*p_sNp) -= integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the intraphase.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
*/
template <typename IBVP>
void eval_bou_fluxes(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                     const MultiFab & X, Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::eval_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);
            
            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;
                    short m_etype, p_etype;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    m_etype = elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                    p_etype = elm_type_fab(pi,pj,pk,ELM_TYPE(dom));

                    // UNKNOWN FIELDS
                    space_elm_bfx m_bfu(&prob_lo[0], &dx[0], mi, mj, mk, m_etype, sp, X_fab);
                    space_elm_bfx p_bfu(&prob_lo[0], &dx[0], pi, pj, pk, p_etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Eval the boundary fluxes at the intraphase between two dG levels.
 *
 * \param[in] t: time.
 * \param[in] c_geom: coarse geometry.
 * \param[in] c_mesh: coarse mesh.
 * \param[in] c_solution: coarse solution.
 * \param[in] c_X: MultiFab object containing the dG coefficients at the coarse level.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine geometry.
 * \param[in] f_mesh: fine mesh.
 * \param[in] f_solution: fine solution.
 * \param[in] f_X: MultiFab object containing the dG coefficients at the fine level.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used in multi-level applications.
 *
*/
template <typename IBVP>
void eval_dG_dG_bou_fluxes(const Real t,
                           const Geometry & c_geom, const Mesh & /*c_mesh*/, const Solution & c_solution, const MultiFab & c_X,
                           const IntVect rr,
                           const Geometry & f_geom, const Mesh & f_mesh, const Solution & f_solution, const MultiFab & f_X,
                           Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::eval_dG_dG_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> c_dx = c_geom.CellSizeArray();
    
    const Box domain = f_geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> f_dx = f_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = f_geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = f_geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int f_sp = f_solution.params.space_p;
    //const int f_sNp = AMREX_D_TERM((1+f_sp),*(1+f_sp),*(1+f_sp));
    const int c_sp = c_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    sMultiFab safe_c_elm_type;
    const sMultiFab * safe_c_elm_type_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(f_X, c_X))
    {
        safe_c_elm_type_ptr = &c_solution.elm_type;
        safe_c_X_ptr = &c_X;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_mesh.ba, rr);
        const DistributionMapping & f_dm = f_mesh.dm;
        const int c_elm_type_n_comp = c_solution.elm_type.n_comp;
        const IntVect c_elm_type_n_grow = c_solution.elm_type.n_grow;
        const int c_X_n_comp = c_X.n_comp;
        const IntVect c_X_n_grow = c_X.n_grow;

        safe_c_elm_type.define(safe_c_ba, f_dm, c_elm_type_n_comp, c_elm_type_n_grow);
        safe_c_elm_type.ParallelCopy(c_solution.elm_type, 0, 0, c_elm_type_n_comp, c_elm_type_n_grow, c_elm_type_n_grow, c_geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, c_X_n_comp, c_X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_geom.periodicity());

        safe_c_elm_type_ptr = &safe_c_elm_type;
        safe_c_X_ptr = &safe_c_X;
    }
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(rr[d1]*rr[d2]);
#endif

#ifdef AMREX_USE_GPU
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            // Coarse mesh
            Array4<short const> const & c_elm_type_fab = safe_c_elm_type_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);
            
            // Fine mesh
            Array4<long const> const & f_cell_bou_quad_info_fab = f_mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & f_elm_type_fab = f_solution.elm_type.array(mfi);
            Array4<short const> const & f_elm_bou_type_fab = f_solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & f_X_fab = f_X.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short f_ebtype = f_elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(f_ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*f_dx[0],
                                                                       prob_lo[1]+fj*f_dx[1],
                                                                       prob_lo[2]+fk*f_dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &f_cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    const short m_etype = f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                    const short p_etype = f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom));
                    const bool m_is_ghost = ELM_IS_GHOST(f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom)));
                    const bool p_is_ghost = ELM_IS_GHOST(f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom)));

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS
                        space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS
                        space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (m_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(mi, mj, mk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (p_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(pi, pj, pk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,FX_offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Add the boundary contribution to the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[in] FX: an array of MultiFab objects containing the intraphase fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * In addition to summing the contribution from the intraphase boundaries, the following integral is
 * evaluated at the embedded interfaces: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U)
 *
 * or
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
*/
template <typename IBVP>
void add_bou_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, const Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::add_bou_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // EVAL THE INTERFACE BOUNDARY INTEGRALS --------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_LARGE_OR_SMALL(etype))
            {
                // NEIGHBORING DOMAIN
                const int nbr_dom = ibvp.get_nbr_dom(dom);

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

                if (nbr_dom == -1)
                {
                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        bfu.eval(x, u_lo, u_hi, U);

                        // Numerical flux as a functions of (t, x, un, U)
                        ibvp.eval_NFn_EB(dom, t, x, un, U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = bfu.BF[rs]*NFn[ru];
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
                else
                {
                    // ELEMENT TYPE (NEIGHBORING DOMAIN)
                    const short nbr_etype = elm_type_fab(i,j,k,ELM_TYPE(nbr_dom));

                    // LOCAL VARIABLES (NEIGHBORING DOMAIN)
                    Real nbr_U[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    space_elm_bfx nbr_bfu(&prob_lo[0], &dx[0], i, j, k, nbr_etype, sp, X_fab);
                    int u_lo, u_hi, nbr_u_lo, nbr_u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                    ibvp.domain_unknown_fields_index_bounds(nbr_dom, nbr_u_lo, nbr_u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        bfu.eval(x, u_lo, u_hi, U);
                        nbr_bfu.eval(x, nbr_u_lo, nbr_u_hi, nbr_U);

                        // Numerical flux as a functions of (t, x, un, U^-, U^+)
                        ibvp.eval_NFn_EB(dom, nbr_dom, t, x, un, U, nbr_U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = bfu.BF[rs]*NFn[ru];
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // ADD THE INTRAPHASE BOUNDARY INTEGRALS --------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = u_lo; ru < u_hi; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        dXdt_fab(i,j,k,rs+ru*sNp) -= (FX_fab(fi,fj,fk,rs+ru*sNp)+FX_fab(i,j,k,FX_offset+rs+ru*sNp));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    dXdt.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################



// FV METHOD: BOUNDARY CONTRIBUTION ###################################
template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_BCS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & rx, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        rx.eval(x, u_lo, u_hi, U);

        // Numerical flux as a functions of (t, x, un, U)
        ibvp.eval_NFn_BCs(dom, t, x, un, U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) += integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & m_ru, space_elm_rx & p_ru, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_ru.eval(x, u_lo, u_hi, m_U);
        p_ru.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) -= integrand*w;
        }
    }
}


/**
 * \brief Eval the boundary fluxes at the intraphase.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the solution states.
 * \param[in] dX: MultiFab object containing the solution slopes.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
*/
template <typename IBVP>
void eval_FV_bou_fluxes(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                        const MultiFab & X, const MultiFab & dX, Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::eval_FV_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);
            Array4<Real const> const & dX_fab = dX.array(mfi);
            
            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);

                    // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                    space_elm_rx m_ru(mi, mj, mk, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        m_ru.xc[0] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                        m_ru.xc[1] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                        m_ru.xc[2] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                    )
                    space_elm_rx p_ru(pi, pj, pk, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        p_ru.xc[0] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                        p_ru.xc[1] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                        p_ru.xc[2] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                    )
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & m_bfu, space_elm_rx & p_ru, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int m_sNp = m_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_bfu.eval(x, u_lo, u_hi, m_U);
        p_ru.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            for (int rs = 0; rs < m_sNp; ++rs)
            {
                integrand = m_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*m_sNp) += integrand*w;
            }
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) -= integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & m_ru, space_elm_bfx & p_bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int p_sNp = p_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_ru.eval(x, u_lo, u_hi, m_U);
        p_bfu.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            for (int rs = 0; rs < p_sNp; ++rs)
            {
                integrand = p_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,FX_offset+rs+ru*p_sNp) -= integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the intraphase between a dG level and a FV level.
 *
 * \param[in] t: time.
 * \param[in] c_geom: coarse geometry.
 * \param[in] c_mesh: coarse mesh.
 * \param[in] c_solution: coarse solution.
 * \param[in] c_X: MultiFab object containing the dG coefficients at the coarse level.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine geometry.
 * \param[in] f_mesh: fine mesh.
 * \param[in] f_solution: fine solution.
 * \param[in] f_X: MultiFab object containing the solution states at the fine level.
 * \param[in] f_dX: MultiFab object containing the solution slopes at the fine level.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used in multi-level applications.
 *
*/
template <typename IBVP>
void eval_dG_FV_bou_fluxes(const Real t,
                           const Geometry & c_geom, const Mesh & /*c_mesh*/, const Solution & c_solution, const MultiFab & c_X,
                           const IntVect rr,
                           const Geometry & f_geom, const Mesh & f_mesh, const Solution & f_solution, const MultiFab & f_X, const MultiFab & f_dX,
                           Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::eval_dG_FV_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> c_dx = c_geom.CellSizeArray();
    
    const Box domain = f_geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> f_dx = f_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = f_geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = f_geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int c_sp = c_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    sMultiFab safe_c_elm_type;
    const sMultiFab * safe_c_elm_type_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(f_X, c_X))
    {
        safe_c_elm_type_ptr = &c_solution.elm_type;
        safe_c_X_ptr = &c_X;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_mesh.ba, rr);
        const DistributionMapping & f_dm = f_mesh.dm;
        const int c_elm_type_n_comp = c_solution.elm_type.n_comp;
        const IntVect c_elm_type_n_grow = c_solution.elm_type.n_grow;
        const int c_X_n_comp = c_X.n_comp;
        const IntVect c_X_n_grow = c_X.n_grow;

        safe_c_elm_type.define(safe_c_ba, f_dm, c_elm_type_n_comp, c_elm_type_n_grow);
        safe_c_elm_type.ParallelCopy(c_solution.elm_type, 0, 0, c_elm_type_n_comp, c_elm_type_n_grow, c_elm_type_n_grow, c_geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, c_X_n_comp, c_X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_geom.periodicity());

        safe_c_elm_type_ptr = &safe_c_elm_type;
        safe_c_X_ptr = &safe_c_X;
    }
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(rr[d1]*rr[d2]);
#endif

#ifdef AMREX_USE_GPU
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            // Coarse mesh
            Array4<short const> const & c_elm_type_fab = safe_c_elm_type_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);
            
            // Fine mesh
            Array4<long const> const & f_cell_bou_quad_info_fab = f_mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<Real const> const & f_elm_centroid_fab = f_solution.elm_centroid.array(mfi);
            Array4<short const> const & f_elm_type_fab = f_solution.elm_type.array(mfi);
            Array4<short const> const & f_elm_bou_type_fab = f_solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & f_X_fab = f_X.array(mfi);
            Array4<Real const> const & f_dX_fab = f_dX.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short f_ebtype = f_elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(f_ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*f_dx[0],
                                                                       prob_lo[1]+fj*f_dx[1],
                                                                       prob_lo[2]+fk*f_dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &f_cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    const bool m_is_ghost = ELM_IS_GHOST(f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom)));
                    const bool p_is_ghost = ELM_IS_GHOST(f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom)));

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                        space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                        AMREX_D_TERM
                        (
                            p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                            p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                            p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                        )

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                        space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                        AMREX_D_TERM
                        (
                            m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                            m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                            m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                        )

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (m_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(mi, mj, mk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);
                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                                p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                                p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                            )

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (p_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(pi, pj, pk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                                m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                                m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                            )
                            // UNKNOWN FIELDS
                            space_elm_bfx p_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,FX_offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                                m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                                m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                            )
                            space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                                p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                                p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                            )

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Add the boundary contribution to the time derivative of solution states.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the solution states.
 * \param[in] dX: MultiFab object containing the solution slopes.
 * \param[in] FX: an array of MultiFab objects containing the intraphase fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * In addition to summing the contribution from the intraphase boundaries, the following integral is
 * evaluated at the interfaces: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
*/
template <typename IBVP>
void add_FV_bou_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                                const MultiFab & X, const MultiFab & dX, const Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_pde::explicit_RK::add_FV_bou_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // EVAL THE INTERFACE BOUNDARY INTEGRALS --------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_LARGE_OR_SMALL(etype))
            {
                // NEIGHBORING DOMAIN
                const int nbr_dom = ibvp.get_nbr_dom(dom);

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

                if (nbr_dom == -1)
                {
                    // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                    space_elm_rx rx(i, j, k, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        rx.xc[0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        rx.xc[1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        rx.xc[2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        rx.eval(x, u_lo, u_hi, U);

                        // Numerical flux as a functions of (t, x, un, U)
                        ibvp.eval_NFn_EB(dom, t, x, un, U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        {
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                }
                else
                {
Abort("dG::hyperbolic_pde::explicit_RK::add_FV_bou_contrib_to_dXdt - nbr_dom != -1");
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // ADD THE INTRAPHASE BOUNDARY INTEGRALS --------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = u_lo; ru < u_hi; ++ru)
                    {
                        dXdt_fab(i,j,k,ru) -= (FX_fab(fi,fj,fk,ru)+FX_fab(i,j,k,FX_offset+ru));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    dXdt.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace explicit_RK
} // namespace hyperbolic_pde



namespace parabolic_pde
{
namespace explicit_RK
{

// DG METHOD: DOMAIN CONTRIBUTION #####################################
/**
 * \brief Add the domain contribution to the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{V} BF(x)_{,k}^T D_k(t, x, U, U_{,l}) + BF(x)^T R(t, x, U, U_{,l})
 *
*/
template <typename IBVP>
void add_dom_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::add_dom_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS --------------------------------------
#ifdef AMREX_USE_GPU
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real integrand;
                Real U[__DG_MAX_NU__];
                Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
                Real AMREX_D_DECL(D1[__DG_MAX_NU__], D2[__DG_MAX_NU__], D3[__DG_MAX_NU__]);
                Real R[__DG_MAX_NU__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                // EVAL THE INTEGRAL
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // Solution at (t, x)
                    bfu.eval_bf_table(x);
                    bfu.eval_unsafe(u_lo, u_hi, U);
                    bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(U_1, U_2, U_3));

                    // Diffusion/Reaction terms as functions of
                    // (t, x, U, U_{,k})
                    ibvp.eval_DR(dom, t, x, U, AMREX_D_DECL(U_1, U_2, U_3), AMREX_D_DECL(D1, D2, D3), R);

                    // Integral contribution
                    for (int ru = u_lo; ru < u_hi; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*D1[ru];,
                            integrand += bfu.dBFdx2[rs]*D2[ru];,
                            integrand += bfu.dBFdx3[rs]*D3[ru];
                        )
                        integrand += bfu.BF[rs]*R[ru];
                        
                        dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------
}
// ####################################################################



// DG METHOD: BOUNDARY CONTRIBUTION ###################################
template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_BCS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int sNp = bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_NU__];
    Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(ND1[__DG_MAX_NU__], ND2[__DG_MAX_NU__], ND3[__DG_MAX_NU__]);
    Real NFn[__DG_MAX_NU__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        bfu.eval_bf_table(x);
        bfu.eval_unsafe(u_lo, u_hi, U);
        bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(U_1, U_2, U_3));

        // Numerical flux as functions of (t, x, un, U, U_{,k})
        ibvp.eval_NF_BCs(dom, t, x, un, U, AMREX_D_DECL(U_1, U_2, U_3), AMREX_D_DECL(ND1, ND2, ND3), NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        for (int rs = 0; rs < sNp; ++rs)
        {
            AMREX_D_TERM
            (
                integrand  = bfu.dBFdx1[rs]*ND1[ru];,
                integrand += bfu.dBFdx2[rs]*ND2[ru];,
                integrand += bfu.dBFdx3[rs]*ND3[ru];
            )
            integrand += bfu.BF[rs]*NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+rs+ru*sNp) += integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & m_bfu, space_elm_bfx & p_bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int m_sNp = m_bfu.Np;
    const int p_sNp = p_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_NU__], p_U[__DG_MAX_NU__];
    Real AMREX_D_DECL(m_U_1[__DG_MAX_NU__], m_U_2[__DG_MAX_NU__], m_U_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(p_U_1[__DG_MAX_NU__], p_U_2[__DG_MAX_NU__], p_U_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(m_ND1[__DG_MAX_NU__], m_ND2[__DG_MAX_NU__], m_ND3[__DG_MAX_NU__]);
    Real m_NFn[__DG_MAX_NU__];
    Real AMREX_D_DECL(p_ND1[__DG_MAX_NU__], p_ND2[__DG_MAX_NU__], p_ND3[__DG_MAX_NU__]);
    Real p_NFn[__DG_MAX_NU__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_bfu.eval_bf_table(x);
        m_bfu.eval_unsafe(u_lo, u_hi, m_U);
        m_bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(m_U_1, m_U_2, m_U_3));

        p_bfu.eval_bf_table(x);
        p_bfu.eval_unsafe(u_lo, u_hi, p_U);
        p_bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(p_U_1, p_U_2, p_U_3));

        // Numerical flux as functions of (t, x, un, U^-, U_{,k}^-, U^+, U_{,k}^+)
        ibvp.eval_NF_intraphase(dom, t, x, un, m_U, AMREX_D_DECL(m_U_1, m_U_2, m_U_3),
                                               p_U, AMREX_D_DECL(p_U_1, p_U_2, p_U_3),
                                               AMREX_D_DECL(m_ND1, m_ND2, m_ND3), m_NFn,
                                               AMREX_D_DECL(p_ND1, p_ND2, p_ND3), p_NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            for (int rs = 0; rs < m_sNp; ++rs)
            {
                AMREX_D_TERM
                (
                    integrand  = m_bfu.dBFdx1[rs]*m_ND1[ru];,
                    integrand += m_bfu.dBFdx2[rs]*m_ND2[ru];,
                    integrand += m_bfu.dBFdx3[rs]*m_ND3[ru];
                )
                integrand += m_bfu.BF[rs]*m_NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*m_sNp) += integrand*w;
            }
            for (int rs = 0; rs < p_sNp; ++rs)
            {
                AMREX_D_TERM
                (
                    integrand  = p_bfu.dBFdx1[rs]*p_ND1[ru];,
                    integrand += p_bfu.dBFdx2[rs]*p_ND2[ru];,
                    integrand += p_bfu.dBFdx3[rs]*p_ND3[ru];
                )
                integrand += p_bfu.BF[rs]*p_NFn[ru];
                FX_fab(fi,fj,fk,FX_offset+rs+ru*p_sNp) += integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the intraphase.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
*/
template <typename IBVP>
void eval_bou_fluxes(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                     const MultiFab & X, Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::eval_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);
            
            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;
                    short m_etype, p_etype;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    m_etype = elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                    p_etype = elm_type_fab(pi,pj,pk,ELM_TYPE(dom));

                    // UNKNOWN FIELDS
                    space_elm_bfx m_bfu(&prob_lo[0], &dx[0], mi, mj, mk, m_etype, sp, X_fab);
                    space_elm_bfx p_bfu(&prob_lo[0], &dx[0], pi, pj, pk, p_etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Eval the boundary fluxes at the intraphase between two dG levels.
 *
 * \param[in] t: time.
 * \param[in] c_geom: coarse geometry.
 * \param[in] c_mesh: coarse mesh.
 * \param[in] c_solution: coarse solution.
 * \param[in] c_X: MultiFab object containing the dG coefficients at the coarse level.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine geometry.
 * \param[in] f_mesh: fine mesh.
 * \param[in] f_solution: fine solution.
 * \param[in] f_X: MultiFab object containing the dG coefficients at the fine level.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used in multi-level applications.
 *
*/
template <typename IBVP>
void eval_dG_dG_bou_fluxes(const Real t,
                           const Geometry & c_geom, const Mesh & /*c_mesh*/, const Solution & c_solution, const MultiFab & c_X,
                           const IntVect rr,
                           const Geometry & f_geom, const Mesh & f_mesh, const Solution & f_solution, const MultiFab & f_X,
                           Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::eval_dG_dG_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> c_dx = c_geom.CellSizeArray();
    
    const Box domain = f_geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> f_dx = f_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = f_geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = f_geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int f_sp = f_solution.params.space_p;
    //const int f_sNp = AMREX_D_TERM((1+f_sp),*(1+f_sp),*(1+f_sp));
    const int c_sp = c_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    sMultiFab safe_c_elm_type;
    const sMultiFab * safe_c_elm_type_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(f_X, c_X))
    {
        safe_c_elm_type_ptr = &c_solution.elm_type;
        safe_c_X_ptr = &c_X;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_mesh.ba, rr);
        const DistributionMapping & f_dm = f_mesh.dm;
        const int c_elm_type_n_comp = c_solution.elm_type.n_comp;
        const IntVect c_elm_type_n_grow = c_solution.elm_type.n_grow;
        const int c_X_n_comp = c_X.n_comp;
        const IntVect c_X_n_grow = c_X.n_grow;

        safe_c_elm_type.define(safe_c_ba, f_dm, c_elm_type_n_comp, c_elm_type_n_grow);
        safe_c_elm_type.ParallelCopy(c_solution.elm_type, 0, 0, c_elm_type_n_comp, c_elm_type_n_grow, c_elm_type_n_grow, c_geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, c_X_n_comp, c_X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_geom.periodicity());

        safe_c_elm_type_ptr = &safe_c_elm_type;
        safe_c_X_ptr = &safe_c_X;
    }
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(rr[d1]*rr[d2]);
#endif

#ifdef AMREX_USE_GPU
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            // Coarse mesh
            Array4<short const> const & c_elm_type_fab = safe_c_elm_type_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);
            
            // Fine mesh
            Array4<long const> const & f_cell_bou_quad_info_fab = f_mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & f_elm_type_fab = f_solution.elm_type.array(mfi);
            Array4<short const> const & f_elm_bou_type_fab = f_solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & f_X_fab = f_X.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short f_ebtype = f_elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(f_ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*f_dx[0],
                                                                       prob_lo[1]+fj*f_dx[1],
                                                                       prob_lo[2]+fk*f_dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &f_cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    const short m_etype = f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                    const short p_etype = f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom));
                    const bool m_is_ghost = ELM_IS_GHOST(f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom)));
                    const bool p_is_ghost = ELM_IS_GHOST(f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom)));

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS
                        space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS
                        space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (m_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(mi, mj, mk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (p_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(pi, pj, pk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,FX_offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &f_dx[0], mi, mj, mk, m_etype, f_sp, f_X_fab);
                            space_elm_bfx p_bfu(&prob_lo[0], &f_dx[0], pi, pj, pk, p_etype, f_sp, f_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Add the boundary contribution to the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[in] FX: an array of MultiFab objects containing the intraphase fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * In addition to summing the contribution from the intraphase boundaries, the following integral is
 * evaluated at the embedded interfaces: 
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U)
 *
 * or
 *
 * int_{dV} BF(x)^T NFn(t, x, un, U^-, U^+)
 *
*/
template <typename IBVP>
void add_bou_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, const Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::add_bou_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // EVAL THE INTERFACE BOUNDARY INTEGRALS --------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_LARGE_OR_SMALL(etype))
            {
                // NEIGHBORING DOMAIN
                const int nbr_dom = ibvp.get_nbr_dom(dom);

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

                if (nbr_dom == -1)
                {
Print() << "parabolic_pde::explicit_central_difference::add_bou_contrib_to_dXdt - dom == -1" << std::endl;
exit(-1);
                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        bfu.eval(x, u_lo, u_hi, U);

                        // Numerical flux as a functions of (t, x, un, U)
                        ibvp.eval_NFn_EB(dom, t, x, un, U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = bfu.BF[rs]*NFn[ru];
// The sign should be +
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
                else
                {
Print() << "parabolic_pde::explicit_central_difference::add_bou_contrib_to_dXdt - dom != -1" << std::endl;
exit(-1);
                    // ELEMENT TYPE (NEIGHBORING DOMAIN)
                    const short nbr_etype = elm_type_fab(i,j,k,ELM_TYPE(nbr_dom));

                    // LOCAL VARIABLES (NEIGHBORING DOMAIN)
                    Real nbr_U[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    space_elm_bfx nbr_bfu(&prob_lo[0], &dx[0], i, j, k, nbr_etype, sp, X_fab);
                    int u_lo, u_hi, nbr_u_lo, nbr_u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                    ibvp.domain_unknown_fields_index_bounds(nbr_dom, nbr_u_lo, nbr_u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        bfu.eval(x, u_lo, u_hi, U);
                        nbr_bfu.eval(x, nbr_u_lo, nbr_u_hi, nbr_U);

                        // Numerical flux as a functions of (t, x, un, U^-, U^+)
                        ibvp.eval_NFn_EB(dom, nbr_dom, t, x, un, U, nbr_U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = bfu.BF[rs]*NFn[ru];
// The sign should be +
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // ADD THE INTRAPHASE BOUNDARY INTEGRALS --------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = u_lo; ru < u_hi; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        dXdt_fab(i,j,k,rs+ru*sNp) += (FX_fab(fi,fj,fk,rs+ru*sNp)+FX_fab(i,j,k,FX_offset+rs+ru*sNp));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    dXdt.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################



// FV METHOD: BOUNDARY CONTRIBUTION ###################################
template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_BCS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & rx, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        rx.eval(x, u_lo, u_hi, U);

        // Numerical flux as a functions of (t, x, un, U)
        ibvp.eval_NFn_BCs(dom, t, x, un, U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) += integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & m_ru, space_elm_rx & p_ru, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_ru.eval(x, u_lo, u_hi, m_U);
        p_ru.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) -= integrand*w;
        }
    }
}


/**
 * \brief Eval the boundary fluxes at the intraphase.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the solution states.
 * \param[in] dX: MultiFab object containing the solution slopes.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
*/
template <typename IBVP>
void eval_FV_bou_fluxes(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                        const MultiFab & X, const MultiFab & dX, Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::eval_FV_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);
            Array4<Real const> const & dX_fab = dX.array(mfi);
            
            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);

                    // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                    space_elm_rx m_ru(mi, mj, mk, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        m_ru.xc[0] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                        m_ru.xc[1] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                        m_ru.xc[2] = elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                    )
                    space_elm_rx p_ru(pi, pj, pk, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        p_ru.xc[0] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                        p_ru.xc[1] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                        p_ru.xc[2] = elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                    )
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_bfx & m_bfu, space_elm_rx & p_ru, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int m_sNp = m_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_bfu.eval(x, u_lo, u_hi, m_U);
        p_ru.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            for (int rs = 0; rs < m_sNp; ++rs)
            {
                integrand = m_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*m_sNp) += integrand*w;
            }
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,FX_offset+ru) -= integrand*w;
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FX_ICS(const int dom, const Real t,
            const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
            space_elm_rx & m_ru, space_elm_bfx & p_bfu, const int u_lo, const int u_hi,
            const int fi, const int fj, const int fk,
            Array4<Real> const & FX_fab, const int FX_offset,
            const IBVP & ibvp)
{
    // LOCAL PARAMETERS
    const int p_sNp = p_bfu.Np;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

    // EVAL THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // Quadrature point and weight
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        AMREX_D_TERM
        (
            x[0] = x_ptr[x_pos+0]+face_lo[0];,
            x[1] = x_ptr[x_pos+1]+face_lo[1];,
            x[2] = x_ptr[x_pos+2]+face_lo[2];
        )
        w = x_ptr[x_pos+AMREX_SPACEDIM];

        // Solution at (t, x)
        m_ru.eval(x, u_lo, u_hi, m_U);
        p_bfu.eval(x, u_lo, u_hi, p_U);

        // Numerical flux as a functions of (t, x, un, U^-, U^+)
        ibvp.eval_NFn_intraphase(dom, t, x, un, m_U, p_U, NFn);

        // Integral contribution
        for (int ru = u_lo; ru < u_hi; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            for (int rs = 0; rs < p_sNp; ++rs)
            {
                integrand = p_bfu.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,FX_offset+rs+ru*p_sNp) -= integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the intraphase between a dG level and a FV level.
 *
 * \param[in] t: time.
 * \param[in] c_geom: coarse geometry.
 * \param[in] c_mesh: coarse mesh.
 * \param[in] c_solution: coarse solution.
 * \param[in] c_X: MultiFab object containing the dG coefficients at the coarse level.
 * \param[in] rr: refinement ratio.
 * \param[in] f_geom: fine geometry.
 * \param[in] f_mesh: fine mesh.
 * \param[in] f_solution: fine solution.
 * \param[in] f_X: MultiFab object containing the solution states at the fine level.
 * \param[in] f_dX: MultiFab object containing the solution slopes at the fine level.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
 * This routine is intended to be used in multi-level applications.
 *
*/
template <typename IBVP>
void eval_dG_FV_bou_fluxes(const Real t,
                           const Geometry & c_geom, const Mesh & /*c_mesh*/, const Solution & c_solution, const MultiFab & c_X,
                           const IntVect rr,
                           const Geometry & f_geom, const Mesh & f_mesh, const Solution & f_solution, const MultiFab & f_X, const MultiFab & f_dX,
                           Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::eval_dG_FV_bou_fluxes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> c_dx = c_geom.CellSizeArray();
    
    const Box domain = f_geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> f_dx = f_geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = f_geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = f_geom.isPeriodicArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int c_sp = c_solution.params.space_p;
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    sMultiFab safe_c_elm_type;
    const sMultiFab * safe_c_elm_type_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ----------------------------------------------------------------

    // PARALLEL COPY --------------------------------------------------
    if (isMFIterSafe(f_X, c_X))
    {
        safe_c_elm_type_ptr = &c_solution.elm_type;
        safe_c_X_ptr = &c_X;
    }
    else
    {
        const BoxArray safe_c_ba = coarsen(f_mesh.ba, rr);
        const DistributionMapping & f_dm = f_mesh.dm;
        const int c_elm_type_n_comp = c_solution.elm_type.n_comp;
        const IntVect c_elm_type_n_grow = c_solution.elm_type.n_grow;
        const int c_X_n_comp = c_X.n_comp;
        const IntVect c_X_n_grow = c_X.n_grow;

        safe_c_elm_type.define(safe_c_ba, f_dm, c_elm_type_n_comp, c_elm_type_n_grow);
        safe_c_elm_type.ParallelCopy(c_solution.elm_type, 0, 0, c_elm_type_n_comp, c_elm_type_n_grow, c_elm_type_n_grow, c_geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, c_X_n_comp, c_X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, c_X_n_comp, c_X_n_grow, c_X_n_grow, c_geom.periodicity());

        safe_c_elm_type_ptr = &safe_c_elm_type;
        safe_c_X_ptr = &safe_c_X;
    }
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        FX[dir] = 0.0;
    }
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(rr[d1]*rr[d2]);
#endif

#ifdef AMREX_USE_GPU
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * f_cell_bou_quad_mem_ptr = f_mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            // Coarse mesh
            Array4<short const> const & c_elm_type_fab = safe_c_elm_type_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);
            
            // Fine mesh
            Array4<long const> const & f_cell_bou_quad_info_fab = f_mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<Real const> const & f_elm_centroid_fab = f_solution.elm_centroid.array(mfi);
            Array4<short const> const & f_elm_type_fab = f_solution.elm_type.array(mfi);
            Array4<short const> const & f_elm_bou_type_fab = f_solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & f_X_fab = f_X.array(mfi);
            Array4<Real const> const & f_dX_fab = f_dX.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short f_ebtype = f_elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(f_ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*f_dx[0],
                                                                       prob_lo[1]+fj*f_dx[1],
                                                                       prob_lo[2]+fk*f_dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = f_cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &f_cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    const bool m_is_ghost = ELM_IS_GHOST(f_elm_type_fab(mi,mj,mk,ELM_TYPE(dom)));
                    const bool p_is_ghost = ELM_IS_GHOST(f_elm_type_fab(pi,pj,pk,ELM_TYPE(dom)));

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                        space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                        AMREX_D_TERM
                        (
                            p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                            p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                            p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                        )

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                        space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                        AMREX_D_TERM
                        (
                            m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                            m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                            m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                        )

                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        FX_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, u_lo, u_hi, fi, fj, fk, FX_fab, 0, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (m_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(mi, mj, mk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS
                            space_elm_bfx m_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);
                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                                p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                                p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                            )

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (p_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(pi, pj, pk, rr, c_i, c_j, c_k);

                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                                m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                                m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                            )
                            // UNKNOWN FIELDS
                            space_elm_bfx p_bfu(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_elm_type_fab(c_i,c_j,c_k,ELM_TYPE(dom)), c_sp, c_X_fab);

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_bfu, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);

                            // NEEDED TO USE average_down_faces
                            for (int ru = u_lo; ru < u_hi; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,FX_offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                            space_elm_rx m_ru(mi, mj, mk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                m_ru.xc[0] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X1(dom));,
                                m_ru.xc[1] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X2(dom));,
                                m_ru.xc[2] = f_elm_centroid_fab(mi,mj,mk,ELM_CENTROID_X3(dom));
                            )
                            space_elm_rx p_ru(pi, pj, pk, f_X_fab, f_dX_fab);
                            AMREX_D_TERM
                            (
                                p_ru.xc[0] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X1(dom));,
                                p_ru.xc[1] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X2(dom));,
                                p_ru.xc[2] = f_elm_centroid_fab(pi,pj,pk,ELM_CENTROID_X3(dom));
                            )

                            // EVAL THE INTEGRAL
                            FX_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_ru, p_ru, u_lo, u_hi, fi, fj, fk, FX_fab, FX_offset, ibvp);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Add the boundary contribution to the time derivative of solution states.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the solution states.
 * \param[in] dX: MultiFab object containing the solution slopes.
 * \param[in] FX: an array of MultiFab objects containing the intraphase fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] dXdt: MultiFab objects that will be modified with domain contribution to the time
 *                  derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * In addition to summing the contribution from the intraphase boundaries, the following integral is
 * evaluated at the interfaces: 
 *
 * int_{dV} BF(x)^T NF(t, x, un, U^-, U^+)
 *
*/
template <typename IBVP>
void add_FV_bou_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                                const MultiFab & X, const MultiFab & dX, const Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, MultiFab & dXdt, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::parabolic_pde::explicit_RK::add_FV_bou_contrib_to_dXdt(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // EVAL THE INTERFACE BOUNDARY INTEGRALS --------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_LARGE_OR_SMALL(etype))
            {
                // NEIGHBORING DOMAIN
                const int nbr_dom = ibvp.get_nbr_dom(dom);

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__], NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

                if (nbr_dom == -1)
                {
                    // UNKNOWN FIELDS (WITH RECONSTRUCTION)
                    space_elm_rx rx(i, j, k, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        rx.xc[0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        rx.xc[1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        rx.xc[2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // Quadrature point, weight and unit normal
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // Solution at (t, x)
                        rx.eval(x, u_lo, u_hi, U);

                        // Numerical flux as a functions of (t, x, un, U)
                        ibvp.eval_NFn_EB(dom, t, x, un, U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        {
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                }
                else
                {
Abort("dG::parabolic_pde::explicit_RK::add_FV_bou_contrib_to_dXdt - nbr_dom != -1");
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // ADD THE INTRAPHASE BOUNDARY INTEGRALS --------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    // UKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = u_lo; ru < u_hi; ++ru)
                    {
                        dXdt_fab(i,j,k,ru) -= (FX_fab(fi,fj,fk,ru)+FX_fab(i,j,k,FX_offset+ru));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    dXdt.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace explicit_RK
} // namespace parabolic_pde

} // namespace dG
} // namespace amrex

#endif