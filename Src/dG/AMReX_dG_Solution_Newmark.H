//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution_Newmark.H
 * \brief Functions to advance solutions in time using implicit Newmark scheme.
*/

#ifndef AMREX_DG_SOLUTION_NEWMARK_H_
#define AMREX_DG_SOLUTION_NEWMARK_H_

#include <AMReX_dG_Solution.H>

namespace amrex
{
namespace dG
{

namespace hyperbolic_2nd_order_pde
{
namespace implicit_Newmark
{

// DG METHOD: RESIDUALS AND JACOBIAN ##################################
/**
 * \brief Add domain contribution to the residuals (and Jacobian) of the implicit Newmark scheme.
 *
 * \param[in] cb: parameters of the Newmark algorithm.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] X a MultiFab containing the current dG coefficients.
 * \param[in] Xd a MultiFab containing the current dG coefficients time derivatives.
 * \param[in] Res: Vector containing the residuals.
 * \param[in] ia: Vector containing the number of non-zeros at each row of the Jacobian.
 * \param[in] Jac: Vector containing the non-zeros at each row of the Jacobian.
 * \param[in] csr_dom_info: a iMultiFab containing info about how to access the sparse data
 *                          structures when domain integration is performed.
 * \param[in] csr_bou_info: an Array of iMultiFabs containing info about how to access the sparse
 *                          data structures when boundary integration is performed.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
*/
template <typename IBVP>
void add_dom_contrib_to_residuals_sym(const Real cb, const Real t,
                                      const Geometry & geom, const Mesh & mesh, const Solution & solution,
                                      const MultiFab & X, const MultiFab & /*Xd*/,
                                      Vector<Real> & Res, const Vector<int> & ia, Vector<Real> & Jac, const iMultiFab & csr_dom_info,
                                      const IBVP & ibvp,
                                      const bool update_Jacobian = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::implicit_Newmark::add_dom_contrib_to_residuals_sym(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    const int max_Nu = solution.get_max_number_of_unknown_fields_per_domain(ibvp);

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

    // 0- vs 1-BASED INDEXING
    const int idx = ia[0];
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS: VALID ELEMENTS ----------------------
    // This part CAN be parallelized (in the sense of OpenMP) because
    // valid elements access different memory locations.
    // ----------------------------------------------------------------
#ifdef AMREX_USE_GPU
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
    Real * Res_mem_ptr = Res.data();
    const int * ia_mem_ptr = ia.data();
    Real * Jac_mem_ptr = Jac.data();
#else
    const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
    Real * Res_mem_ptr = Res.data();
    const int * ia_mem_ptr = ia.data();
    Real * Jac_mem_ptr = Jac.data();
#endif

    for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);

            if (elm_is_valid)
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
                Real M[__DG_MAX_NU2__];
#if (AMREX_SPACEDIM == 1)
                Real dD1dU_1[__DG_MAX_NU2__];
#endif
#if (AMREX_SPACEDIM == 2)
                Real dD1dU_1[__DG_MAX_NU2__], dD1dU_2[__DG_MAX_NU2__];
                Real dD2dU_1[__DG_MAX_NU2__], dD2dU_2[__DG_MAX_NU2__];
#endif
#if (AMREX_SPACEDIM == 3)
                Real dD1dU_1[__DG_MAX_NU2__], dD1dU_2[__DG_MAX_NU2__], dD1dU_3[__DG_MAX_NU2__];
                Real dD2dU_1[__DG_MAX_NU2__], dD2dU_2[__DG_MAX_NU2__], dD2dU_3[__DG_MAX_NU2__];
                Real dD3dU_1[__DG_MAX_NU2__], dD3dU_2[__DG_MAX_NU2__], dD3dU_3[__DG_MAX_NU2__];
#endif
                Real AMREX_D_DECL(dD1dU[__DG_MAX_NU2__], dD2dU[__DG_MAX_NU2__], dD3dU[__DG_MAX_NU2__]);
                Real AMREX_D_DECL(dRdU_1[__DG_MAX_NU2__], dRdU_2[__DG_MAX_NU2__], dRdU_3[__DG_MAX_NU2__]);
                Real dRdU[__DG_MAX_NU2__];
                
                Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
                Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                // EVAL THE INTEGRAL
                std::fill(RB, RB+Nup, 0.0);
                std::fill(JB, JB+Nup*Nup, 0.0);
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

                    for (int ru = 0; ru < Nu; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*D1[ru];,
                            integrand += bfu.dBFdx2[rs]*D2[ru];,
                            integrand += bfu.dBFdx3[rs]*D3[ru];
                        )
                        integrand += bfu.BF[rs]*R[ru];

                        const int rm = sNp*ru+rs;
                        RB[rm] += integrand*w;
                    }

                    if (update_Jacobian)
                    {
                        // Mass matrix at (t, x)
                        ibvp.eval_M(dom, t, x, M);

                        // Gradients of Diffusion/Reaction terms with
                        // respect to U and U_{,k} as functions of
                        // (t, x, U, U_{,k})
#if (AMREX_SPACEDIM == 1)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1,
                                        dD1dU_1, dD1dU,
                                        dRdU_1, dRdU);
#endif
#if (AMREX_SPACEDIM == 2)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1, U_2,
                                        dD1dU_1, dD1dU_2, dD2dU_1, dD2dU_2, dD1dU, dD2dU,
                                        dRdU_1, dRdU_2, dRdU);
#endif
#if (AMREX_SPACEDIM == 3)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1, U_2, U_3,
                                        dD1dU_1, dD1dU_2, dD1dU_3, dD2dU_1, dD2dU_2, dD2dU_3, dD3dU_1, dD3dU_2, dD3dU_3, dD1dU, dD2dU, dD3dU,
                                        dRdU_1, dRdU_2, dRdU_3, dRdU);
#endif

                        for (int ru = 0; ru < Nu; ++ru)
                        for (int cu = 0; cu < Nu; ++cu)
                        for (int rs = 0; rs < sNp; ++rs)
                        for (int cs = 0; cs < sNp; ++cs)
                        {
                            integrand = 0.0;
#if (AMREX_SPACEDIM == 1)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*dD2dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx2[rs]*  dD2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dRdU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*dD2dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx2[rs]*  dD2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx3[rs]*dD3dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx3[rs]*dD3dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx3[rs]*dD3dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx3[rs]*  dD3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dRdU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]* dRdU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
                            integrand *= cb;
                            integrand += bfu.BF[rs]*M[ru+cu*Nu]*bfu.BF[cs];

                            const int rm = sNp*ru+rs;
                            const int cm = sNp*cu+cs;
                            JB[rm+cm*Nup] += integrand*w;
                        }
                    }
                }

                // CONTRIBUTION TO THE GLOBAL SYSTEM
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                for (int r = 0; r < Nup; ++r)
                {
                    Res_mem_ptr[posX+r] += cb*RB[r];
                }

                if (update_Jacobian)
                {
                    for (int r = 0; r < Nup; ++r)
                    {
                        const int posJ = ia_mem_ptr[posX+r]-idx;
                        
                        for (int c = r; c < Nup; ++c)
                        {
                            // We access JB by column (i.e. use JB[c+r*Nup]
                            // instead of JB[r+c*Nup]) because of symmetry
                            Jac_mem_ptr[posJ+c-r] += JB[c+r*Nup];
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS: SMALL ELEMENTS ----------------------
    // This part CANNOT be parallelized (in the sense of OpenMP)
    // because multiple small elements can be merged with (and thus
    // access the memory of) the same valid element.
    // ----------------------------------------------------------------
    for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_small = ELM_IS_SMALL(etype);

            if (elm_is_small)
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
                Real M[__DG_MAX_NU2__];
#if (AMREX_SPACEDIM == 1)
                Real dD1dU_1[__DG_MAX_NU2__];
#endif
#if (AMREX_SPACEDIM == 2)
                Real dD1dU_1[__DG_MAX_NU2__], dD1dU_2[__DG_MAX_NU2__];
                Real dD2dU_1[__DG_MAX_NU2__], dD2dU_2[__DG_MAX_NU2__];
#endif
#if (AMREX_SPACEDIM == 3)
                Real dD1dU_1[__DG_MAX_NU2__], dD1dU_2[__DG_MAX_NU2__], dD1dU_3[__DG_MAX_NU2__];
                Real dD2dU_1[__DG_MAX_NU2__], dD2dU_2[__DG_MAX_NU2__], dD2dU_3[__DG_MAX_NU2__];
                Real dD3dU_1[__DG_MAX_NU2__], dD3dU_2[__DG_MAX_NU2__], dD3dU_3[__DG_MAX_NU2__];
#endif
                Real AMREX_D_DECL(dD1dU[__DG_MAX_NU2__], dD2dU[__DG_MAX_NU2__], dD3dU[__DG_MAX_NU2__]);
                Real AMREX_D_DECL(dRdU_1[__DG_MAX_NU2__], dRdU_2[__DG_MAX_NU2__], dRdU_3[__DG_MAX_NU2__]);
                Real dRdU[__DG_MAX_NU2__];
                
                Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
                Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                // EVAL THE INTEGRAL
                std::fill(RB, RB+Nup, 0.0);
                std::fill(JB, JB+Nup*Nup, 0.0);
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

                    for (int ru = 0; ru < Nu; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*D1[ru];,
                            integrand += bfu.dBFdx2[rs]*D2[ru];,
                            integrand += bfu.dBFdx3[rs]*D3[ru];
                        )
                        integrand += bfu.BF[rs]*R[ru];

                        const int rm = sNp*ru+rs;
                        RB[rm] += integrand*w;
                    }

                    if (update_Jacobian)
                    {
                        // Mass matrix at (t, x)
                        ibvp.eval_M(dom, t, x, M);

                        // Gradients of Diffusion/Reaction terms with
                        // respect to U and U_{,k} as functions of
                        // (t, x, U, U_{,k})
#if (AMREX_SPACEDIM == 1)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1,
                                        dD1dU_1, dD1dU,
                                        dRdU_1, dRdU);
#endif
#if (AMREX_SPACEDIM == 2)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1, U_2,
                                        dD1dU_1, dD1dU_2, dD2dU_1, dD2dU_2, dD1dU, dD2dU,
                                        dRdU_1, dRdU_2, dRdU);
#endif
#if (AMREX_SPACEDIM == 3)
                        ibvp.eval_dDRdU(dom, t, x, U, U_1, U_2, U_3,
                                        dD1dU_1, dD1dU_2, dD1dU_3, dD2dU_1, dD2dU_2, dD2dU_3, dD3dU_1, dD3dU_2, dD3dU_3, dD1dU, dD2dU, dD3dU,
                                        dRdU_1, dRdU_2, dRdU_3, dRdU);
#endif

                        for (int ru = 0; ru < Nu; ++ru)
                        for (int cu = 0; cu < Nu; ++cu)
                        for (int rs = 0; rs < sNp; ++rs)
                        for (int cs = 0; cs < sNp; ++cs)
                        {
                            integrand = 0.0;
#if (AMREX_SPACEDIM == 1)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*dD2dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx2[rs]*  dD2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dRdU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                            integrand += bfu.dBFdx1[rs]*dD1dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx1[rs]*dD1dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx1[rs]*  dD1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*dD2dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx2[rs]*dD2dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx2[rs]*  dD2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx3[rs]*dD3dU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand += bfu.dBFdx3[rs]*dD3dU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand += bfu.dBFdx3[rs]*dD3dU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand += bfu.dBFdx3[rs]*  dD3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dRdU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dRdU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]* dRdU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand +=     bfu.BF[rs]*   dRdU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
                            integrand *= cb;
                            integrand += bfu.BF[rs]*M[ru+cu*Nu]*bfu.BF[cs];

                            const int rm = sNp*ru+rs;
                            const int cm = sNp*cu+cs;
                            JB[rm+cm*Nup] += integrand*w;
                        }
                    }
                }

                // CONTRIBUTION TO THE GLOBAL SYSTEM
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                for (int r = 0; r < Nup; ++r)
                {
                    Res_mem_ptr[posX+r] += cb*RB[r];
                }

                if (update_Jacobian)
                {
                    for (int r = 0; r < Nup; ++r)
                    {
                        const int posJ = ia_mem_ptr[posX+r]-idx;
                        
                        for (int c = r; c < Nup; ++c)
                        {
                            // We access JB by column (i.e. use JB[c+r*Nup]
                            // instead of JB[r+c*Nup]) because of symmetry
                            Jac_mem_ptr[posJ+c-r] += JB[c+r*Nup];
                        }
                    }
                }
            }
        }
    }
    // ----------------------------------------------------------------
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void RES_BCS(const Real cb, const int dom, const Real t,
             const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
             space_elm_bfx & bfu, const int u_lo, const int u_hi,
             Real * Res_mem_ptr, const int * ia_mem_ptr, const int idx, Real * Jac_mem_ptr,
             const int posX,
             const IBVP & ibvp,
             const bool update_Jacobian = false)
{
    // LOCAL PARAMETERS
    const int sNp = bfu.Np;
    const int Nu = u_hi-u_lo;
    const int Nup = Nu*sNp;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_NU__];
    Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(ND1[__DG_MAX_NU__], ND2[__DG_MAX_NU__], ND3[__DG_MAX_NU__]);
    Real NFn[__DG_MAX_NU__];
    Real AMREX_D_DECL(dND1dU[__DG_MAX_NU2__], dND2dU[__DG_MAX_NU2__], dND3dU[__DG_MAX_NU2__]);
    Real AMREX_D_DECL(dNFndU_1[__DG_MAX_NU2__], dNFndU_2[__DG_MAX_NU2__], dNFndU_3[__DG_MAX_NU2__]);
    Real dNFndU[__DG_MAX_NU2__];
    Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
    Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

    // EVAL THE INTEGRAL
    std::fill(RB, RB+Nup, 0.0);
    std::fill(JB, JB+Nup*Nup, 0.0);
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

        for (int ru = 0; ru < Nu; ++ru)
        for (int rs = 0; rs < sNp; ++rs)
        {
            AMREX_D_TERM
            (
                integrand  = bfu.dBFdx1[rs]*ND1[ru];,
                integrand += bfu.dBFdx2[rs]*ND2[ru];,
                integrand += bfu.dBFdx3[rs]*ND3[ru];
            )
            integrand += bfu.BF[rs]*NFn[ru];

            const int rm = sNp*ru+rs;
            RB[rm] += integrand*w;
        }

        if (update_Jacobian)
        {
            // Gradients of Numerical flux with respect to U and U_{,k}
            // as functions of (t, x, un, U, U_{,k})
            ibvp.eval_dNFdU_BCs(dom, t, x, un,
                                U, AMREX_D_DECL(U_1, U_2, U_3),
                                AMREX_D_DECL(dND1dU, dND2dU, dND3dU), 
                                AMREX_D_DECL(dNFndU_1, dNFndU_2, dNFndU_3),
                                dNFndU);

            for (int ru = 0; ru < Nu; ++ru)
            for (int cu = 0; cu < Nu; ++cu)
            for (int rs = 0; rs < sNp; ++rs)
            for (int cs = 0; cs < sNp; ++cs)
            {
                integrand = 0.0;        
#if (AMREX_SPACEDIM == 1)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx3[rs]*   dND3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]* dNFndU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif

                const int rm = sNp*ru+rs;
                const int cm = sNp*cu+cs;

                JB[rm+cm*Nup] += integrand*w;
            }
        }
    }

    // CONTRIBUTION TO THE GLOBAL SYSTEM
    for (int r = 0; r < Nup; ++r)
    {
        Res_mem_ptr[posX+r] -= cb*RB[r];
    }

    if (update_Jacobian)
    {
        for (int r = 0; r < Nup; ++r)
        {
            const int posJ = ia_mem_ptr[posX+r]-idx;
            
            for (int c = r; c < Nup; ++c)
            {
                // We access JB by column (i.e. use JB[c+r*Nup]
                // instead of JB[r+c*Nup]) because of symmetry
                Jac_mem_ptr[posJ+c-r] -= cb*JB[c+r*Nup];
            }
        }
    }
}

template <typename IBVP>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void RES_ICS(const Real cb, const int dom, const Real t,
             const Real * face_lo, const int bou_Nq, const Real * x_ptr, const Real * un,
             space_elm_bfx & bfu, space_elm_bfx & nbr_bfu, const int u_lo, const int u_hi,
             Real * Res_mem_ptr, const int * ia_mem_ptr, const int idx, Real * Jac_mem_ptr,
             const int posX, const int nbr_posX, const int nnz_offset,
             const IBVP & ibvp,
             const bool update_Jacobian = false)
{
    // LOCAL PARAMETERS
    const int sNp = bfu.Np;
    const int nbr_sNp = nbr_bfu.Np;
    const int Nu = u_hi-u_lo;
    const int Nup = Nu*sNp;
    const int nbr_Nup = Nu*nbr_sNp;

    // LOCAL VARIABLES
    Real x[AMREX_SPACEDIM], w;
    Real integrand;
    Real U[__DG_MAX_NU__], Up[__DG_MAX_NU__];
    Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(Up_1[__DG_MAX_NU__], Up_2[__DG_MAX_NU__], Up_3[__DG_MAX_NU__]);
    Real AMREX_D_DECL(ND1[__DG_MAX_NU__], ND2[__DG_MAX_NU__], ND3[__DG_MAX_NU__]);
    Real NFn[__DG_MAX_NU__];
    Real AMREX_D_DECL(dND1dU[__DG_MAX_NU2__], dND2dU[__DG_MAX_NU2__], dND3dU[__DG_MAX_NU2__]);
    Real AMREX_D_DECL(dND1dUp[__DG_MAX_NU2__], dND2dUp[__DG_MAX_NU2__], dND3dUp[__DG_MAX_NU2__]);
    Real AMREX_D_DECL(dNFndU_1[__DG_MAX_NU2__], dNFndU_2[__DG_MAX_NU2__], dNFndU_3[__DG_MAX_NU2__]);
    Real dNFndU[__DG_MAX_NU2__];
    Real AMREX_D_DECL(dNFndUp_1[__DG_MAX_NU2__], dNFndUp_2[__DG_MAX_NU2__], dNFndUp_3[__DG_MAX_NU2__]);
    Real dNFndUp[__DG_MAX_NU2__];
    Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
    Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
    Real JBp[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

    // EVAL THE INTEGRAL
    std::fill(RB, RB+Nup, 0.0);
    std::fill(JB, JB+Nup*Nup, 0.0);
    std::fill(JBp, JBp+Nup*nbr_Nup, 0.0);
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

        nbr_bfu.eval_bf_table(x);
        nbr_bfu.eval_unsafe(u_lo, u_hi, Up);
        nbr_bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(Up_1, Up_2, Up_3));

        // Numerical flux as functions of (t, x, un, U, U_{,k}, U', U_{,k}')
        ibvp.eval_NF_intraphase(dom, t, x, un,
                                U, AMREX_D_DECL(U_1, U_2, U_3),
                                Up, AMREX_D_DECL(Up_1, Up_2, Up_3),
                                AMREX_D_DECL(ND1, ND2, ND3), NFn);

        for (int ru = 0; ru < Nu; ++ru)
        for (int rs = 0; rs < sNp; ++rs)
        {
            AMREX_D_TERM
            (
                integrand  = bfu.dBFdx1[rs]*ND1[ru];,
                integrand += bfu.dBFdx2[rs]*ND2[ru];,
                integrand += bfu.dBFdx3[rs]*ND3[ru];
            )
            integrand += bfu.BF[rs]*NFn[ru];

            const int rm = sNp*ru+rs;
            RB[rm] += integrand*w;
        }

        if (update_Jacobian)
        {
            // Gradients of Numerical flux with respect to U, U_{,k}, U'
            // and U_{,k}' as functions of
            // (t, x, un, U, U_{,k}, U', U_{,k}')
            ibvp.eval_dNFdU_intraphase(dom, t, x, un,
                                       U, AMREX_D_DECL(U_1, U_2, U_3),
                                       Up, AMREX_D_DECL(Up_1, Up_2, Up_3),
                                       AMREX_D_DECL(dND1dU, dND2dU, dND3dU),
                                       AMREX_D_DECL(dND1dUp, dND2dUp, dND3dUp), 
                                       AMREX_D_DECL(dNFndU_1, dNFndU_2, dNFndU_3),
                                       dNFndU,
                                       AMREX_D_DECL(dNFndUp_1, dNFndUp_2, dNFndUp_3),
                                       dNFndUp);

            for (int ru = 0; ru < Nu; ++ru)
            for (int cu = 0; cu < Nu; ++cu)
            for (int rs = 0; rs < sNp; ++rs)
            for (int cs = 0; cs < sNp; ++cs)
            {
                integrand = 0.0;
#if (AMREX_SPACEDIM == 1)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand += bfu.dBFdx3[rs]*   dND3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]* dNFndU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
                const int rm = sNp*ru+rs;
                const int cm = sNp*cu+cs;
                JB[rm+cm*Nup] += integrand*w;
            }

            for (int ru = 0; ru < Nu; ++ru)
            for (int cu = 0; cu < Nu; ++cu)
            for (int rs = 0; rs < sNp; ++rs)
            for (int cs = 0; cs < nbr_sNp; ++cs)
            {
                integrand = 0.0;
#if (AMREX_SPACEDIM == 1)
                integrand += bfu.dBFdx1[rs]*   dND1dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndUp_1[ru+cu*Nu]*nbr_bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]*   dNFndUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                integrand += bfu.dBFdx1[rs]*   dND1dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndUp_1[ru+cu*Nu]*nbr_bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndUp_2[ru+cu*Nu]*nbr_bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]*   dNFndUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                integrand += bfu.dBFdx1[rs]*   dND1dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand += bfu.dBFdx2[rs]*   dND2dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand += bfu.dBFdx3[rs]*   dND3dUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
                integrand +=     bfu.BF[rs]* dNFndUp_1[ru+cu*Nu]*nbr_bfu.dBFdx1[cs];
                integrand +=     bfu.BF[rs]* dNFndUp_2[ru+cu*Nu]*nbr_bfu.dBFdx2[cs];
                integrand +=     bfu.BF[rs]* dNFndUp_3[ru+cu*Nu]*nbr_bfu.dBFdx3[cs];
                integrand +=     bfu.BF[rs]*   dNFndUp[ru+cu*Nu]*nbr_bfu.BF[cs]    ;
#endif
                const int rm = sNp*ru+rs;
                const int cm = nbr_sNp*cu+cs;
                JBp[rm+cm*nbr_Nup] += integrand*w;
            }
        }
    }

    // CONTRIBUTION TO THE GLOBAL SYSTEM
    for (int r = 0; r < Nup; ++r)
    {
        Res_mem_ptr[posX+r] -= cb*RB[r];
    }

    if (update_Jacobian)
    {
        const int delta_pos = nnz_offset-(ia_mem_ptr[posX]-idx);

        if (nbr_posX > posX)
        {
            for (int r = 0; r < Nup; ++r)
            {
                const int posJ = ia_mem_ptr[posX+r]-idx;
                
                for (int c = r; c < Nup; ++c)
                {
                    // We access JB by column (i.e. use JB[c+r*Nup]
                    // instead of JB[r+c*Nup]) because of symmetry
                    Jac_mem_ptr[posJ+c-r] -= cb*JB[c+r*Nup];
                }
                
                for (int c = 0; c < nbr_Nup; ++c)
                {
                    Jac_mem_ptr[delta_pos+posJ+c-r] -= cb*JBp[r+c*nbr_Nup];
                }
            }
        }
        else
        {
            for (int r = 0; r < Nup; ++r)
            {
                const int posJ = ia_mem_ptr[posX+r]-idx;
                
                for (int c = r; c < Nup; ++c)
                {
                    // We access JB by column (i.e. use JB[c+r*Nup]
                    // instead of JB[r+c*Nup]) because of symmetry
                    Jac_mem_ptr[posJ+c-r] -= cb*JB[c+r*Nup];
                }
            }   
        }
    }
}

/**
 * \brief Add boundary contribution to the residuals (and Jacobian) of the implicit Newmark scheme.
 *
 * \param[in] cb: parameters of the Newmark algorithm.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] X a MultiFab containing the current dG coefficients.
 * \param[in] Xd a MultiFab containing the current dG coefficients time derivatives.
 * \param[in] Res: Vector containing the residuals.
 * \param[in] ia: Vector containing the number of non-zeros at each row of the Jacobian.
 * \param[in] Jac: Vector containing the non-zeros at each row of the Jacobian.
 * \param[in] csr_dom_info: a iMultiFab containing info about how to access the sparse data
 *                          structures when domain integration is performed.
 * \param[in] csr_bou_info: an Array of iMultiFabs containing info about how to access the sparse
 *                          data structures when boundary integration is performed.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
*/
template <typename IBVP>
void add_bou_contrib_to_residuals_sym(const Real cb, const Real t,
                                      const Geometry & geom, const Mesh & mesh, const Solution & solution,
                                      const MultiFab & X, const MultiFab & /*Xd*/,
                                      Vector<Real> & Res, const Vector<int> & ia, Vector<Real> & Jac, const iMultiFab & csr_dom_info, const Array<iMultiFab, AMREX_SPACEDIM> & csr_bou_info,
                                      const IBVP & ibvp,
                                      const bool update_Jacobian = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::implicit_Newmark::add_bou_contrib_to_residuals_sym(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    const int max_Nu = solution.get_max_number_of_unknown_fields_per_domain(ibvp);

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

    // 0- vs 1-BASED INDEXING
    const int idx = ia[0];
    // ----------------------------------------------------------------

    // EVAL THE BOUNDARY INTEGRALS ------------------------------------
#ifdef AMREX_USE_GPU
        Real * Res_mem_ptr = Res.data();
        int const * ia_mem_ptr = ia.data();
        Real * Jac_mem_ptr = Jac.data();
#else
        Real * Res_mem_ptr = Res.data();
        int const * ia_mem_ptr = ia.data();
        Real * Jac_mem_ptr = Jac.data();
#endif
    for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
#ifdef AMREX_USE_GPU
            Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
            Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif
            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<int const> const & csr_bou_info_fab = csr_bou_info[dir].array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // POSITION IN THE GLOBAL SYSTEM
                    const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    for (int b = 2*dir; b < 2*(dir+1); ++b)
                    {
                        // OUTER UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = 2.0*(b%2)-1.0;

                        // GRID FACE
                        int fi, fj, fk;
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                        if (ELM_BOU_IS_VALID(ebtype))
                        {
                            // LOCAL PARAMETERS
                            const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                               prob_lo[1]+fj*dx[1],
                                                                               prob_lo[2]+fk*dx[2])};

                            // QUADRATURE INFO
                            const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                            const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                            const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                            // BOUNDARY CONDITIONS
                            if (ELM_BOU_IS_WALL(ebtype))
                            {
                                // EVAL THE INTEGRAL
                                RES_BCS(cb, dom, t, face_lo, bou_Nq, x_ptr, un, bfu, u_lo, u_hi, Res_mem_ptr, ia_mem_ptr, idx, Jac_mem_ptr, posX, ibvp, update_Jacobian);
                            }
                            // INTRAPHASE CONDITIONS
                            else
                            {
                                // NEIGHBORING CELL
                                int nbr_i, nbr_j, nbr_k, nbr_b;
                                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                                const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                                space_elm_bfx nbr_bfu(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, sp, X_fab);

                                // POSITION IN THE GLOBAL SYSTEM
                                const int nbr_posX = csr_dom_info_fab(nbr_i,nbr_j,nbr_k,CSR_DOM_INFO_POS_X(dom));
                                const int nnz_offset = csr_bou_info_fab(fi,fj,fk,CSR_BOU_INFO_NNZ_OFFSET(dom, 0));

                                RES_ICS(cb, dom, t, face_lo, bou_Nq, x_ptr, un, bfu, nbr_bfu, u_lo, u_hi, Res_mem_ptr, ia_mem_ptr, idx, Jac_mem_ptr, posX, nbr_posX, nnz_offset, ibvp, update_Jacobian);
                            }
                        }
                    }
                }
            }
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Add domain contribution to the residuals (and Jacobian) of the implicit Newmark scheme.
 *
 * \param[in] cb: parameters of the Newmark algorithm.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] X a MultiFab containing the current dG coefficients.
 * \param[in] Xd a MultiFab containing the current dG coefficients time derivatives.
 * \param[in] Res: Vector containing the residuals.
 * \param[in] ia: Vector containing the number of non-zeros at each row of the Jacobian.
 * \param[in] Jac: Vector containing the non-zeros at each row of the Jacobian.
 * \param[in] csr_dom_info: a iMultiFab containing info about how to access the sparse data
 *                          structures when domain integration is performed.
 * \param[in] csr_bou_info: an Array of iMultiFabs containing info about how to access the sparse
 *                          data structures when boundary integration is performed.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
*/
template <typename IBVP>
void add_int_bou_contrib_to_residuals_sym(const Real cb, const Real t,
                                          const Geometry & geom, const Mesh & mesh, const Solution & solution,
                                          const MultiFab & X, const MultiFab & /*Xd*/,
                                          Vector<Real> & Res, const Vector<int> & ia, Vector<Real> & Jac, const iMultiFab & csr_dom_info,
                                          const IBVP & ibvp,
                                          const bool update_Jacobian = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::implicit_Newmark::add_int_bou_contrib_to_residuals_sym(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    const int max_Nu = solution.get_max_number_of_unknown_fields_per_domain(ibvp);

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

    // 0- vs 1-BASED INDEXING
    const int idx = ia[0];
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS: VALID ELEMENTS ----------------------
    // This part CAN be parallelized (in the sense of OpenMP) because
    // valid elements access different memory locations.
    // ----------------------------------------------------------------
#ifdef AMREX_USE_GPU
    Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
    Real * Res_mem_ptr = Res.data();
    const int * ia_mem_ptr = ia.data();
    Real * Jac_mem_ptr = Jac.data();
#else
    Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
    Real * Res_mem_ptr = Res.data();
    const int * ia_mem_ptr = ia.data();
    Real * Jac_mem_ptr = Jac.data();
#endif

    for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);

            if (elm_is_valid)
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_NU__];
                Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
                Real AMREX_D_DECL(ND1[__DG_MAX_NU__], ND2[__DG_MAX_NU__], ND3[__DG_MAX_NU__]);
                Real NFn[__DG_MAX_NU__];
                Real AMREX_D_DECL(dND1dU[__DG_MAX_NU2__], dND2dU[__DG_MAX_NU2__], dND3dU[__DG_MAX_NU2__]);
                Real AMREX_D_DECL(dNFndU_1[__DG_MAX_NU2__], dNFndU_2[__DG_MAX_NU2__], dNFndU_3[__DG_MAX_NU2__]);
                Real dNFndU[__DG_MAX_NU2__];
                
                Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
                Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                // EVAL THE INTEGRAL
                std::fill(RB, RB+Nup, 0.0);
                std::fill(JB, JB+Nup*Nup, 0.0);
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
                    bfu.eval_bf_table(x);
                    bfu.eval_unsafe(u_lo, u_hi, U);
                    bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(U_1, U_2, U_3));

                    // Numerical flux as functions of (t, x, un, U, U_{,k})
                    // (t, x, U, U_{,k})
                    ibvp.eval_NF_EB(dom, t, x, un, U, AMREX_D_DECL(U_1, U_2, U_3), AMREX_D_DECL(ND1, ND2, ND3), NFn);

                    for (int ru = 0; ru < Nu; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*ND1[ru];,
                            integrand += bfu.dBFdx2[rs]*ND2[ru];,
                            integrand += bfu.dBFdx3[rs]*ND3[ru];
                        )
                        integrand += bfu.BF[rs]*NFn[ru];

                        const int rm = sNp*ru+rs;
                        RB[rm] += integrand*w;
                    }

                    if (update_Jacobian)
                    {
                        // Gradients of Numerical flux with respect to U and U_{,k}
                        // as functions of (t, x, un, U, U_{,k})
                        ibvp.eval_dNFdU_EB(dom, t, x, un,
                                           U, AMREX_D_DECL(U_1, U_2, U_3),
                                           AMREX_D_DECL(dND1dU, dND2dU, dND3dU), 
                                           AMREX_D_DECL(dNFndU_1, dNFndU_2, dNFndU_3),
                                           dNFndU);

                        for (int ru = 0; ru < Nu; ++ru)
                        for (int cu = 0; cu < Nu; ++cu)
                        for (int rs = 0; rs < sNp; ++rs)
                        for (int cs = 0; cs < sNp; ++cs)
                        {
                            integrand = 0.0;        
#if (AMREX_SPACEDIM == 1)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx3[rs]*   dND3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif

                            const int rm = sNp*ru+rs;
                            const int cm = sNp*cu+cs;

                            JB[rm+cm*Nup] += integrand*w;
                        }
                    }
                }

                // CONTRIBUTION TO THE GLOBAL SYSTEM
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                for (int r = 0; r < Nup; ++r)
                {
                    Res_mem_ptr[posX+r] -= cb*RB[r];
                }

                if (update_Jacobian)
                {
                    for (int r = 0; r < Nup; ++r)
                    {
                        const int posJ = ia_mem_ptr[posX+r]-idx;
                        
                        for (int c = r; c < Nup; ++c)
                        {
                            // We access JB by column (i.e. use JB[c+r*Nup]
                            // instead of JB[r+c*Nup]) because of symmetry
                            Jac_mem_ptr[posJ+c-r] -= cb*JB[c+r*Nup];
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // EVAL THE DOMAIN INTEGRALS: SMALL ELEMENTS ----------------------
    // This part CANNOT be parallelized (in the sense of OpenMP)
    // because multiple small elements can be merged with (and thus
    // access the memory of) the same valid element.
    // ----------------------------------------------------------------
    for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_small = ELM_IS_SMALL(etype);

            if (elm_is_small)
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real U[__DG_MAX_NU__];
                Real AMREX_D_DECL(U_1[__DG_MAX_NU__], U_2[__DG_MAX_NU__], U_3[__DG_MAX_NU__]);
                Real AMREX_D_DECL(ND1[__DG_MAX_NU__], ND2[__DG_MAX_NU__], ND3[__DG_MAX_NU__]);
                Real NFn[__DG_MAX_NU__];
                Real AMREX_D_DECL(dND1dU[__DG_MAX_NU2__], dND2dU[__DG_MAX_NU2__], dND3dU[__DG_MAX_NU2__]);
                Real AMREX_D_DECL(dNFndU_1[__DG_MAX_NU2__], dNFndU_2[__DG_MAX_NU2__], dNFndU_3[__DG_MAX_NU2__]);
                Real dNFndU[__DG_MAX_NU2__];
                
                Real RB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__];
                Real JB[__DG_MAX_SPACE_Np__*__DG_MAX_NU__*__DG_MAX_SPACE_Np__*__DG_MAX_NU__];

                // UNKNOWN FIELDS
                space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                // EVAL THE INTEGRAL
                std::fill(RB, RB+Nup, 0.0);
                std::fill(JB, JB+Nup*Nup, 0.0);
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
                    bfu.eval_bf_table(x);
                    bfu.eval_unsafe(u_lo, u_hi, U);
                    bfu.eval_derivatives_unsafe(u_lo, u_hi, AMREX_D_DECL(U_1, U_2, U_3));

                    // Numerical flux as functions of (t, x, un, U, U_{,k})
                    // (t, x, U, U_{,k})
                    ibvp.eval_NF_EB(dom, t, x, un, U, AMREX_D_DECL(U_1, U_2, U_3), AMREX_D_DECL(ND1, ND2, ND3), NFn);

                    for (int ru = 0; ru < Nu; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        AMREX_D_TERM
                        (
                            integrand  = bfu.dBFdx1[rs]*ND1[ru];,
                            integrand += bfu.dBFdx2[rs]*ND2[ru];,
                            integrand += bfu.dBFdx3[rs]*ND3[ru];
                        )
                        integrand += bfu.BF[rs]*NFn[ru];

                        const int rm = sNp*ru+rs;
                        RB[rm] += integrand*w;
                    }

                    if (update_Jacobian)
                    {
                        // Gradients of Numerical flux with respect to U and U_{,k}
                        // as functions of (t, x, un, U, U_{,k})
                        ibvp.eval_dNFdU_EB(dom, t, x, un,
                                           U, AMREX_D_DECL(U_1, U_2, U_3),
                                           AMREX_D_DECL(dND1dU, dND2dU, dND3dU), 
                                           AMREX_D_DECL(dNFndU_1, dNFndU_2, dNFndU_3),
                                           dNFndU);

                        for (int ru = 0; ru < Nu; ++ru)
                        for (int cu = 0; cu < Nu; ++cu)
                        for (int rs = 0; rs < sNp; ++rs)
                        for (int cs = 0; cs < sNp; ++cs)
                        {
                            integrand = 0.0;        
#if (AMREX_SPACEDIM == 1)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 2)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif
#if (AMREX_SPACEDIM == 3)
                            integrand += bfu.dBFdx1[rs]*   dND1dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx2[rs]*   dND2dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand += bfu.dBFdx3[rs]*   dND3dU[ru+cu*Nu]*bfu.BF[cs]    ;
                            integrand +=     bfu.BF[rs]* dNFndU_1[ru+cu*Nu]*bfu.dBFdx1[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_2[ru+cu*Nu]*bfu.dBFdx2[cs];
                            integrand +=     bfu.BF[rs]* dNFndU_3[ru+cu*Nu]*bfu.dBFdx3[cs];
                            integrand +=     bfu.BF[rs]*   dNFndU[ru+cu*Nu]*bfu.BF[cs]    ;
#endif

                            const int rm = sNp*ru+rs;
                            const int cm = sNp*cu+cs;

                            JB[rm+cm*Nup] += integrand*w;
                        }
                    }
                }

                // CONTRIBUTION TO THE GLOBAL SYSTEM
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                for (int r = 0; r < Nup; ++r)
                {
                    Res_mem_ptr[posX+r] -= cb*RB[r];
                }

                if (update_Jacobian)
                {
                    for (int r = 0; r < Nup; ++r)
                    {
                        const int posJ = ia_mem_ptr[posX+r]-idx;
                        
                        for (int c = r; c < Nup; ++c)
                        {
                            // We access JB by column (i.e. use JB[c+r*Nup]
                            // instead of JB[r+c*Nup]) because of symmetry
                            Jac_mem_ptr[posJ+c-r] -= cb*JB[c+r*Nup];
                        }
                    }
                }
            }
        }
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Eval the residuals (and Jacobian) of the implicit Newmark scheme.
 *
 * \param[in] gamma, beta: parameters of the Newmark algorithm.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] MCh: a MultiFab objects containing the Cholesky decomposition of the pde mass matrix.
 * \param[in] X_old a MultiFab containing the old dG coefficients.
 * \param[in] Xd_old a MultiFab containing the old dG coefficients time derivatives.
 * \param[in] X a MultiFab containing the current dG coefficients.
 * \param[in] Xd a MultiFab containing the current dG coefficients time derivatives.
 * \param[inout] Res: Vector containing the residuals.
 * \param[in] ia: Vector containing the number of non-zeros at each row of the Jacobian.
 * \param[inout] Jac: Vector containing the non-zeros at each row of the Jacobian.
 * \param[in] csr_dom_info: a iMultiFab containing info about how to access the sparse data
 *                          structures when domain integration is performed.
 * \param[in] csr_bou_info: an Array of iMultiFabs containing info about how to access the sparse
 *                          data structures when boundary integration is performed.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
*/
template <typename IBVP>
void eval_residuals_sym(const Real gamma, const Real beta, const Real t, const Real dt,
                       const Geometry & geom, const Mesh & mesh, const Solution & solution, const iMultiFab & mask,
                       const MultiFab & MCh, const MultiFab & X_old, const MultiFab & Xd_old, const MultiFab & X, const MultiFab & Xd,
                       Vector<Real> & Res, const Vector<int> & ia, Vector<Real> & Jac, const iMultiFab & csr_dom_info, const Array<iMultiFab, AMREX_SPACEDIM> & csr_bou_info,
                       const IBVP & ibvp,
                       const bool update_Jacobian = false)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::implicit_Newmark::eval_residuals_sym(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    MultiFab X1(X_old.boxarray, X_old.distributionMap, X_old.n_comp, X_old.n_grow);
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    // X1 := M*(X-X_old-dt*Xd_old)
    MultiFab::LinComb(X1, +1.0, X, 0, -1.0, X_old, 0, 0, X.n_comp, X.n_grow);
    MultiFab::Saxpy(X1, -dt, Xd_old, 0, 0, X.n_comp, X.n_grow);

    solution.multiply_by_pde_mass_matrix(geom, MCh, X1, ibvp);
    solution.copy_solution_to_small_elements(geom, X1, ibvp);
    // ----------------------------------------------------------------

    // CONTRIBUTION TO THE RESIDUALS ----------------------------------
#ifdef AMREX_USE_GPU
    Real * Res_mem_ptr = Res.data();
#else
    Real * Res_mem_ptr = Res.data();
#endif
    for (MFIter mfi(X1); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<int const> const & mask_fab = mask.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real const> const & X1_fab = X1.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);

            if (cell_is_not_masked && elm_is_valid)
            {
                // LOCAL PARAMETERS
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                // UNKNOWN FIELDS
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                for (int r = 0; r < Nup; ++r)
                {
                    Res_mem_ptr[posX+r] = X1_fab(i,j,k,u_lo*sNp+r);
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // DOMAIN CONTRIBUTION --------------------------------------------
    add_dom_contrib_to_residuals_sym(beta*dt*dt, t, geom, mesh, solution, X, Xd, Res, ia, Jac, csr_dom_info, ibvp, update_Jacobian);
    add_dom_contrib_to_residuals_sym((0.5-beta)*dt*dt, t, geom, mesh, solution, X_old, Xd_old, Res, ia, Jac, csr_dom_info, ibvp, false);
    // ----------------------------------------------------------------

    // BOUNDARY CONTRIBUTION ------------------------------------------
    add_bou_contrib_to_residuals_sym(beta*dt*dt, t, geom, mesh, solution, X, Xd, Res, ia, Jac, csr_dom_info, csr_bou_info, ibvp, update_Jacobian);
    add_bou_contrib_to_residuals_sym((0.5-beta)*dt*dt, t, geom, mesh, solution, X_old, Xd_old, Res, ia, Jac, csr_dom_info, csr_bou_info, ibvp, false);
    // ----------------------------------------------------------------

    // INTERNAL BOUNDARIES CONTRIBUTION -------------------------------
    add_int_bou_contrib_to_residuals_sym(beta*dt*dt, t, geom, mesh, solution, X, Xd, Res, ia, Jac, csr_dom_info, ibvp, update_Jacobian);
    add_int_bou_contrib_to_residuals_sym((0.5-beta)*dt*dt, t, geom, mesh, solution, X_old, Xd_old, Res, ia, Jac, csr_dom_info, ibvp, false);
    // ----------------------------------------------------------------

    /* DEBUG
    Print() << "dt: " << dt << std::endl;
    Print() << "beta*dt*dt: " << beta*dt*dt << std::endl;
    Print() << "Res: " << std::endl;
    io::print_real_array_2d(Res.size(), 1, Res.data());
    Print() << "Jac: " << std::endl;
    int pos = 0;
    for (int r = 0; r < Res.size(); ++r)
    {
        io::print_real_array_2d(1, Res.size()-r, &Jac[pos]);
        pos += Res.size()-r;
    }

    Print() << "eval_residuals_sym" << std::endl;
    exit(-1);
    */
}

/**
 * \brief Eval the velocity for the implicit Newmark scheme.
 *
 * \param[in] gamma: parameters of the Newmark algorithm.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object; contains information about embedded geometries.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] MCh: a MultiFab objects containing the Cholesky decomposition of the pde mass matrix.
 * \param[in] X_old a MultiFab containing the old dG coefficients.
 * \param[in] Xd_old a MultiFab containing the old dG coefficients time derivatives.
 * \param[in] X a MultiFab containing the current dG coefficients.
 * \param[out] Xd a MultiFab containing that will contain the updated dG coefficients time
 *             derivatives.
 * \param[inout] Res: Vector containing the residuals.
 * \param[in] ia: Vector containing the number of non-zeros at each row of the Jacobian.
 * \param[in] Jac: Vector containing the non-zeros at each row of the Jacobian.
 * \param[in] csr_dom_info: a iMultiFab containing info about how to access the sparse data
 *                          structures when domain integration is performed.
 * \param[in] ibvp: initial boundary value problem object.
 *
 *
*/
template <typename IBVP>
void eval_Xd_sym(const Real gamma, const Real t, const Real dt,
                 const Geometry & geom, const Mesh & mesh, const Solution & solution, const iMultiFab & mask,
                 const MultiFab & MCh, const MultiFab & X_old, const MultiFab & Xd_old, const MultiFab & X, MultiFab & Xd,
                 Vector<Real> & Res, const iMultiFab & csr_dom_info, const Array<iMultiFab, AMREX_SPACEDIM> & csr_bou_info,
                 const IBVP & ibvp)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("dG::hyperbolic_2nd_order_pde::implicit_Newmark::eval_Xd_sym(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();

    // ORDER
    const int sp = solution.params.space_p;
    const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<int> dummy_ia(1);
    Vector<Real> dummy_Jac;
    // ----------------------------------------------------------------

    // ONLY ACCESSED, NOT USED ----------------------------------------
    dummy_ia[0] = 0;
    // ----------------------------------------------------------------

    // DOMAIN CONTRIBUTION --------------------------------------------
    add_dom_contrib_to_residuals_sym(gamma*dt, t, geom, mesh, solution, X, Xd, Res, dummy_ia, dummy_Jac, csr_dom_info, ibvp, false);
    add_dom_contrib_to_residuals_sym((1.0-gamma)*dt, t, geom, mesh, solution, X_old, Xd_old, Res, dummy_ia, dummy_Jac, csr_dom_info, ibvp, false);
    // ----------------------------------------------------------------

    // BOUNDARY CONTRIBUTION ------------------------------------------
    add_bou_contrib_to_residuals_sym(gamma*dt, t, geom, mesh, solution, X, Xd, Res, dummy_ia, dummy_Jac, csr_dom_info, csr_bou_info, ibvp, false);
    add_bou_contrib_to_residuals_sym((1.0-gamma)*dt, t, geom, mesh, solution, X_old, Xd_old, Res, dummy_ia, dummy_Jac, csr_dom_info, csr_bou_info, ibvp, false);
    // ----------------------------------------------------------------

    // COPY TO MULTIFAB -----------------------------------------------
#ifdef AMREX_USE_GPU
    Real * Res_mem_ptr = Res.data();
#else
    Real * Res_mem_ptr = Res.data();
#endif
    for (MFIter mfi(Xd); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<int const> const & mask_fab = mask.array(mfi);
        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
        Array4<Real> const & Xd_fab = Xd.array(mfi);

        ParallelFor(bx, n_domains,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);

            if (cell_is_not_masked && elm_is_valid)
            {
                // LOCAL PARAMETERS
                const int posX = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_POS_X(dom));

                // UNKNOWN FIELDS
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int Nup = Nu*sNp;

                for (int r = 0; r < Nup; ++r)
                {
                    Xd_fab(i,j,k,u_lo*sNp+r) = -Res_mem_ptr[posX+r];
                }
            }
        });
        Gpu::synchronize();
    }
    // ----------------------------------------------------------------

    // UPDATE ---------------------------------------------------------
    solution.multiply_by_inverse_pde_mass_matrix(geom, MCh, Xd, ibvp);
    MultiFab::Saxpy(Xd, 1.0, Xd_old, 0, 0, Xd.n_comp, Xd.n_grow);
    solution.copy_solution_to_small_elements(geom, Xd, ibvp);
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace implicit_Newmark
} // namespace hyperbolic_2nd_order_pde

} // namespace dG
} // namespace amrex

#endif