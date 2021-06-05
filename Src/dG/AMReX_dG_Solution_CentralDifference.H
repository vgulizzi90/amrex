//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution_CentralDifference.H
 * \brief Functions to advance solutions in time using central difference.
*/

#ifndef AMREX_DG_SOLUTION_CENTRAL_DIFFERENCE_H_
#define AMREX_DG_SOLUTION_CENTRAL_DIFFERENCE_H_

#include <AMReX_VTK_Utils.H>

#include <AMReX_dG_Solution.H>

namespace amrex
{
namespace dG
{

namespace hyperbolic_2nd_order_pde
{
namespace explicit_central_difference
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
 * \param[in] d2Xdt2: MultiFab objects that will be modified with domain contribution to the time
 *                    derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * The following integral is evaluated: 
 *
 * int_{V} - BF(x)_{,k}^T D_k(t, x, U, U_{,l}) - BF(x)^T R(t, x, U, U_{,k})
 *
*/
template <typename IBVP>
void add_dom_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, MultiFab & d2Xdt2, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::explicit_central_difference::add_dom_contrib_to_dXdt(...)");
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

        Array4<Real> const & d2Xdt2_fab = d2Xdt2.array(mfi);

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
                Real AMREX_D_DECL(U_1[__DG_MAX_N_UNKNOWN_FIELDS__], U_2[__DG_MAX_N_UNKNOWN_FIELDS__], U_3[__DG_MAX_N_UNKNOWN_FIELDS__]);
                Real AMREX_D_DECL(D1[__DG_MAX_N_UNKNOWN_FIELDS__], D2[__DG_MAX_N_UNKNOWN_FIELDS__], D3[__DG_MAX_N_UNKNOWN_FIELDS__]);
                Real R[__DG_MAX_N_UNKNOWN_FIELDS__];

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
                    //ibvp.eval_D(dom, t, x, U, AMREX_D_DECL(U_1, U_2, U_3), AMREX_D_DECL(D1, D2, D3));
                    //ibvp.eval_R(dom, t, x, U, AMREX_D_DECL(U_1, U_2, U_3), R);
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
                        
                        d2Xdt2_fab(i,j,k,rs+ru*sNp) -= integrand*w;
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
    Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
    Real AMREX_D_DECL(U_1[__DG_MAX_N_UNKNOWN_FIELDS__], U_2[__DG_MAX_N_UNKNOWN_FIELDS__], U_3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real AMREX_D_DECL(ND1[__DG_MAX_N_UNKNOWN_FIELDS__], ND2[__DG_MAX_N_UNKNOWN_FIELDS__], ND3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

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
    Real m_U[__DG_MAX_N_UNKNOWN_FIELDS__], p_U[__DG_MAX_N_UNKNOWN_FIELDS__];
    Real AMREX_D_DECL(m_U_1[__DG_MAX_N_UNKNOWN_FIELDS__], m_U_2[__DG_MAX_N_UNKNOWN_FIELDS__], m_U_3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real AMREX_D_DECL(p_U_1[__DG_MAX_N_UNKNOWN_FIELDS__], p_U_2[__DG_MAX_N_UNKNOWN_FIELDS__], p_U_3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real AMREX_D_DECL(m_ND1[__DG_MAX_N_UNKNOWN_FIELDS__], m_ND2[__DG_MAX_N_UNKNOWN_FIELDS__], m_ND3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real m_NFn[__DG_MAX_N_UNKNOWN_FIELDS__];
    Real AMREX_D_DECL(p_ND1[__DG_MAX_N_UNKNOWN_FIELDS__], p_ND2[__DG_MAX_N_UNKNOWN_FIELDS__], p_ND3[__DG_MAX_N_UNKNOWN_FIELDS__]);
    Real p_NFn[__DG_MAX_N_UNKNOWN_FIELDS__];

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
 * int_{dV} BF(x)_{,k}^T NDn_k(t, x, un, U^-, U_{,k}^-, U^+, U_{,k}^+) + BF(x)^T NFn(t, x, un, U^-, U_{,k}^-, U^+, U_{,k}^+)
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
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::explicit_central_difference::eval_bou_fluxes(...)");
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
 * \brief Add the boundary contribution to the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: MultiFab object containing the dG coefficients.
 * \param[in] FX: an array of MultiFab objects containing the intraphase fluxes.
 * \param[in] FX_offset: an integer used for accessing the FX components.
 * \param[in] d2Xdt2: MultiFab objects that will be modified with domain contribution to the time
 *                    derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
 * In addition to summing the contribution from the intraphase boundaries, the following integral is
 * evaluated at the embedded interfaces: 
 *
 * int_{dV} BF(x)_{,k}^T NDn_k(t, x, un, U, U_{,k}) + BF(x)^T NFn(t, x, un, U, U_{,k})
 *
 * or
 *
 * int_{dV} BF(x)_{,k}^T NDn_k(t, x, un, U^-, U_{,k}^-, U^+, U_{,k}^+) + BF(x)^T NFn(t, x, un, U^-, U_{,k}^-, U^+, U_{,k}^+)
 *
*/
template <typename IBVP>
void add_bou_contrib_to_dXdt(const Real t, const Geometry & geom, const Mesh & mesh, const Solution & solution,
                             const MultiFab & X, const Array<MultiFab, AMREX_SPACEDIM> & FX, const int FX_offset, MultiFab & d2Xdt2, const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::explicit_central_difference::add_bou_contrib_to_dXdt(...)");
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

        Array4<Real> const & d2Xdt2_fab = d2Xdt2.array(mfi);

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
                        //ibvp.eval_NFn_EB(dom, t, x, un, U, NFn);

                        // Integral contribution
                        for (int ru = u_lo; ru < u_hi; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = bfu.BF[rs]*NFn[ru];
                            //d2Xdt2_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
                else
                {
Print() << "hyperbolic_2nd_order_pde::explicit_central_difference::add_bou_contrib_to_dXdt - dom != -1" << std::endl;
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

            Array4<Real> const & d2Xdt2_fab = d2Xdt2.array(mfi);

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
                        d2Xdt2_fab(i,j,k,rs+ru*sNp) += (FX_fab(fi,fj,fk,rs+ru*sNp)+FX_fab(i,j,k,FX_offset+rs+ru*sNp));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    d2Xdt2.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace explicit_central_difference
} // namespace hyperbolic_2nd_order_pde

} // namespace dG
} // namespace amrex

#endif