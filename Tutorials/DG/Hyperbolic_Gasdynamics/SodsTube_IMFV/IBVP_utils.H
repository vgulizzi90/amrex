//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define DG_N_SOL (1+AMREX_SPACEDIM+1)

#define DG_LIMIT_CP

#define DG_SL_THETA 1.5

//#define DG_LIMIT_BY_LS

#define LS_N_NBR AMREX_D_PICK(2, 8, 26)
#define STENCIL_N_NBR AMREX_D_PICK(2, 4, 6)

namespace amrex
{
namespace DG
{

/**
 * \brief Compute the maximum time step. 
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // DOMAINS
    const int dom = 0;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.std_elem.p;
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dt(X.boxarray, X.distributionMap, 1, 0);
    dt = std::numeric_limits<Real>::max();

    Real dt_min;
    // ================================================================

    // EVAL THE MINIMUM TIME STEP =====================================
    for (MFIter mfi(dt); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
            const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));

            // QUADRATURE INFO
            const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
            const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
            const Real * xptr = &eDomQuad_mem_ptr[pos];
            long x_pos;

            // LOCAL VARIABLES
            int BF_i, BF_j, BF_k;
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
            Real elm_dt, x_dt;
            Real x[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL];

            // SUPPORT OF THE BASIS FUNCTIONS
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
            
            // SOLUTION
            DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

            // LOOP OVER THE QUAD POINTS
            elm_dt = std::numeric_limits<Real>::max();
            for (int q = 0; q < dom_Nq; ++q)
            {
                x_pos = (AMREX_SPACEDIM+1)*q;
                AMREX_D_TERM
                (
                    x[0] = xptr[x_pos+0]+cell_lo[0];,
                    x[1] = xptr[x_pos+1]+cell_lo[1];,
                    x[2] = xptr[x_pos+2]+cell_lo[2];
                )

                // EVAL SOLUTION
                sol.eval_BF_only_table(x);
                sol.eval_unsafe(0, N_SOL, SOL);

                // EVAL DT
                x_dt = IBVP.F_DT(&dx[0], t, x, SOL);
                elm_dt = std::min(elm_dt, x_dt);
            }
            
            elm_dt = (elm_is_large) ? (vf*elm_dt) : elm_dt;
            dt_fab(i,j,k,0) = std::min(dt_fab(i,j,k,0), elm_dt);
        });
        Gpu::synchronize();
    }

    dt_min = dt.min(0);
    // ================================================================

    return dt_min;
}

/**
 * \brief Fill invalid cells (i.e. boundary/empty/small) using boundary conditions.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains unknown solution fields at the cell centers. On
 *                  exit, the cells in the ghost rows will have their values updated.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * SOL, const Real * bcs_SOL) const;
 *                  void F_PHI_BCS(const Real t, const Real * x, const Real * un,
 *                                 const Real * SOL, const Real * bcs_SOL) const;
*/
template <typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const ImplicitMesh & mesh,
                      const int N_SOL,
                      MultiFab & X,
                      const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // NEIGHBORS INFO IN STENCIL
    const GpuArray<int, STENCIL_N_NBR> aux_table_i =
    {
#if (AMREX_SPACEDIM == 1)
        -1, +1
#endif
#if (AMREX_SPACEDIM == 2)
        -1, +1, 0, 0
#endif
#if (AMREX_SPACEDIM == 3)
        -1, +1, 0, 0, 0, 0
#endif
    };
    const GpuArray<int, STENCIL_N_NBR> aux_table_j =
    {
#if (AMREX_SPACEDIM == 1)
        0, 0
#endif
#if (AMREX_SPACEDIM == 2)
        0, 0, -1, +1
#endif
#if (AMREX_SPACEDIM == 3)
        0, 0, -1, +1, 0, 0
#endif
    };
    const GpuArray<int, STENCIL_N_NBR> aux_table_k = 
    {
#if (AMREX_SPACEDIM == 1)
        0, 0
#endif
#if (AMREX_SPACEDIM == 2)
        0, 0, 0, 0
#endif
#if (AMREX_SPACEDIM == 3)
        0, 0, 0, 0, -1, +1
#endif
    };
    
    // DOMAINS
    const int dom = 0;

    // DOMAIN-TO-LEVELSETS CORRESPONDENCE
    int const * Dom2Phi_ptr = mesh.Dom2Phi.data();

    // DG
    const int sp = mesh.std_elem.p;
    // ================================================================

    // CHECK ==========================================================
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillInvalidCells\n";
        msg += "| X contains nans (On entry).\n";
        Warning(msg);
    }
    // ================================================================

    // FILL BOUNDARY CELLS ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            int BF_i, BF_j, BF_k;
            Real SOL[DG_N_SOL], bcs_SOL[DG_N_SOL];

            if (elm_is_not_empty)
            {
                // CURRENT CELL SOLUTION
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                }

                // UPDATE GRID BOUNDARY VALUES
                if ((i == domain.smallEnd(0)) && (is_periodic[0] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i-1+0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k  +0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] = -1.0;,
                        un[1] =  0.0;,
                        un[2] =  0.0;
                    )
                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i-1,j,k,ru) = bcs_SOL[ru];
                    }
                }
                if ((i == domain.bigEnd(0)) && (is_periodic[0] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i+1+0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k  +0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] = +1.0;,
                        un[1] =  0.0;,
                        un[2] =  0.0;
                    )

                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i+1,j,k,ru) = bcs_SOL[ru];
                    }
                }
#if (AMREX_SPACEDIM > 1)
                if (j == domain.smallEnd(1))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j-1+0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k  +0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] = -1.0;,
                        un[2] =  0.0;
                    )

                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i,j-1,k,ru) = bcs_SOL[ru];
                    }
                }
                if (j == domain.bigEnd(1))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j+1+0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k  +0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] = +1.0;,
                        un[2] =  0.0;
                    )

                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i,j+1,k,ru) = bcs_SOL[ru];
                    }
                }
#endif
#if (AMREX_SPACEDIM > 2)
                if (k == domain.smallEnd(2))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k-1+0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] =  0.0;,
                        un[2] = -1.0;
                    )

                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i,j,k-1,ru) = bcs_SOL[ru];
                    }
                }
                if (k == domain.bigEnd(2))
                {
                    AMREX_D_TERM
                    (
                        x[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                        x[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                        x[2] = prob_lo[2]+(k+1+0.5)*dx[2];
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] =  0.0;,
                        un[2] = +1.0;
                    )

                    IBVP.F_BCS(t, x, un, SOL, bcs_SOL);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        X_fab(i,j,k+1,ru) = bcs_SOL[ru];
                    }
                }
#endif
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillInvalidCells\n";
        msg += "| X contains nans (After filling the ghost cells).\n";
        Warning(msg);
    }
    // ================================================================

    // FILL EMPTY/SMALL CELLS =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_empty = (etype == __DG_ELM_TYPE_EMPTY__);
            const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
            const bool elm_is_not_valid = (elm_is_empty || elm_is_small);

            const int phi_sign = Dom2Phi_ptr[2*dom+1];

            // LOCAL VARIABLES
            Real xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL], bcs_SOL[DG_N_SOL];

            // LEVELSETS
            DG_PHI_space_BFX phi(sp, cell_lo, cell_hi, PHI_fab, i, j, k);
            phi.set_u(Dom2Phi_ptr[2*dom]);

            // CHECK WETHER THE ELEMENT HAS NO VALID NEIGHBORS --------
            bool elm_is_not_isolated;
            elm_is_not_isolated = false;

            for (int n = 0; n < STENCIL_N_NBR; ++n)
            {
                const int nbr_i = i+aux_table_i[n];
                const int nbr_j = j+aux_table_j[n];
                const int nbr_k = k+aux_table_k[n];
                const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                const bool nbr_is_entire = (nbr_etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool nbr_is_large = (nbr_etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool nbr_is_valid = (nbr_is_entire || nbr_is_large);
                if (nbr_is_valid) elm_is_not_isolated = true;
            }
            // --------------------------------------------------------
            
            if (elm_is_not_valid && elm_is_not_isolated)
            {
                // CELL'S CENTROID
                AMREX_D_TERM
                (
                    xc[0] = 0.5*(cell_lo[0]+cell_hi[0]);,
                    xc[1] = 0.5*(cell_lo[1]+cell_hi[1]);,
                    xc[2] = 0.5*(cell_lo[2]+cell_hi[2]);
                )

                // LEVEL SET GRADIENT DIRECTION
                phi.eval_un(xc, un);
                AMREX_D_TERM
                (
                    un[0] *= -phi_sign;,
                    un[1] *= -phi_sign;,
                    un[2] *= -phi_sign;
                )

                // MOVE TO THE CLOSE VALID CELL
                AMREX_D_TERM
                (
                    xc[0] -= un[0]*dx[0];,
                    xc[1] -= un[1]*dx[1];,
                    xc[2] -= un[2]*dx[2];
                )

#if (AMREX_SPACEDIM == 1)
                const int nbr_i = (int) std::floor((xc[0]-prob_lo[0])/dx[0]);
                const int nbr_j = j;
                const int nbr_k = k;
#endif
#if (AMREX_SPACEDIM == 2)
                const int nbr_i = (int) std::floor((xc[0]-prob_lo[0])/dx[0]);
                const int nbr_j = (int) std::floor((xc[1]-prob_lo[1])/dx[1]);
                const int nbr_k = k;
#endif
#if (AMREX_SPACEDIM == 3)
                const int nbr_i = (int) std::floor((xc[0]-prob_lo[0])/dx[0]);
                const int nbr_j = (int) std::floor((xc[1]-prob_lo[1])/dx[1]);
                const int nbr_k = (int) std::floor((xc[2]-prob_lo[2])/dx[2]);
#endif

                const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                const bool nbr_is_entire = (nbr_etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool nbr_is_large = (nbr_etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool nbr_is_valid = (nbr_is_entire || nbr_is_large);

                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                }

                IBVP.F_PHI_BCS(t, xc, un, SOL, bcs_SOL);
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    X_fab(i,j,k,ru) = bcs_SOL[ru];
                }

                if (!nbr_is_valid)
                {
                    /*
                    Print() << "(" << i << "," << j << "," << k << ") ---> (" << nbr_i << "," << nbr_j << "," << nbr_k << ") ---> bad neighbor" << std::endl;
                    Print() << "    SOL: "; IO::PrintRealArray2D(1, DG_N_SOL, SOL);
                    Print() << "bcs_SOL: "; IO::PrintRealArray2D(1, DG_N_SOL, bcs_SOL);
                    //exit(-1);
                    */
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillInvalidCells\n";
        msg += "| X contains nans (after filling the invalid cells).\n";
        Warning(msg);
    }
    // ================================================================
}

/**
 * \brief Limit slopes.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void LimitSlopes(const int N_SOL, const Real * dCL, const Real * dCR, Real * dC)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        const Real dc = 0.5*(dCL[ru]+dCR[ru]);
        const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
        const Real slope = DG_SL_THETA*std::min(std::abs(dCL[ru]), std::abs(dCR[ru]));
        const Real lim = (dCL[ru]*dCR[ru] >= 0.0) ? slope : 0.0;
        dC[ru] = sgn*std::min(lim, std::abs(dc));
    }
}

/**
 * \brief Eval limited slopes and store them. They will be used to reconstruct the solution.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dX: a MultiFab object that contains the limited slopes for the unknown solution
 *                 fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_DU2DC(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DU,
 *                               Real * DC) const;
 *                  void F_DC2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DC,
 *                               Real * DU) const;
 *
 *                  If DG_LIMIT_CP is defined, the class must also have:
 *
 *                  void F_U2P(const Real t, const Real * x,
 *                             const Real * SOL, Real * P) const;
 *                  void F_DP2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DP,
 *                               Real * DU) const;
*/
template <typename IBVP_CLASS>
void EvalLimitedSlopes(const Real t,
                       const ImplicitMesh & mesh,
                       const int N_SOL,
                       const MultiFab & X,
                       MultiFab & dX,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // DOMAIN
    const int dom = 0;
    // ================================================================

    // INITIALIZATION
    dX = 0.0;
    // ==============
    
    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ((etype%10 == __DG_ELM_TYPE_ENTIRE__) ||
                                       (etype%10 == __DG_ELM_TYPE_LARGE__));
            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], dSOL[DG_N_SOL];
                Real dCL[DG_N_SOL], dCR[DG_N_SOL], dC[DG_N_SOL];
#ifdef DG_LIMIT_CP
                Real P[DG_N_SOL], nbr_P[DG_N_SOL], dP[DG_N_SOL];
#endif

                // CURRENT CELL SPACE LOCATION
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // CURRENT CELL SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(i,j,k,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, SOL, P);
#endif

                // X1 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[0] = +1.0;

                // LEFT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i-1,j,k,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

                // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = SOL[ru]-nbr_SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = P[ru]-nbr_P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCL);

                // RIGHT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i+1,j,k,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

                // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = nbr_SOL[ru]-SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = nbr_P[ru]-P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCR);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                LimitSlopes(N_SOL, dCL, dCR, dC);

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(t, x, un, SOL, dC, dSOL);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dX_fab(i,j,k,0+AMREX_SPACEDIM*ru) = dSOL[ru]/dx[0];
                }
                // ----------------------------------------------------
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[1] = +1.0;

                // LEFT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i,j-1,k,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

                // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = SOL[ru]-nbr_SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = P[ru]-nbr_P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCL);

                // RIGHT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i,j+1,k,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

                // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = nbr_SOL[ru]-SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = nbr_P[ru]-P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCR);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                LimitSlopes(N_SOL, dCL, dCR, dC);

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(t, x, un, SOL, dC, dSOL);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dX_fab(i,j,k,1+AMREX_SPACEDIM*ru) = dSOL[ru]/dx[1];
                }
                // ----------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[2] = +1.0;

                // LEFT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i,j,k-1,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

                // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = SOL[ru]-nbr_SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = P[ru]-nbr_P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCL);

                // RIGHT DIFFERENCES
                // Neighbor cell info
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(i,j,k+1,ru);
                }
#ifdef DG_LIMIT_CP
                IBVP.F_U2P(t, x, nbr_SOL, nbr_P);
#endif

            // Characteristics
#ifndef DG_LIMIT_CP
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOL[ru] = nbr_SOL[ru]-SOL[ru];
                }
#else
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dP[ru] = nbr_P[ru]-P[ru];
                }
                IBVP.F_DP2DU(t, x, SOL, dP, dSOL);
#endif
                IBVP.F_DU2DC(t, x, un, SOL, dSOL, dCR);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                LimitSlopes(N_SOL, dCL, dCR, dC);

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(t, x, un, SOL, dC, dSOL);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dX_fab(i,j,k,2+AMREX_SPACEDIM*ru) = dSOL[ru]/dx[2];
                }
                // ----------------------------------------------------
#endif
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (dX.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopes\n";
        msg += "| dX contains nans.\n";
        amrex::Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Solve a 2x2 symmetric linear system with n_rhs right-hand sides.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void SolveSym2x2(const int n_rhs, const Real * S, const Real * B, Real * X)
{
    const Real inv_delta = 1.0/(S[0+0*2]*S[1+1*2]-S[0+1*2]*S[0+1*2]);

    for (int rh = 0; rh < n_rhs; ++rh)
    {
        X[0+rh*2] = ( S[1+1*2]*B[0+rh*2]-S[0+1*2]*B[1+rh*2])*inv_delta;
        X[1+rh*2] = (-S[0+1*2]*B[0+rh*2]+S[0+0*2]*B[1+rh*2])*inv_delta;
    }
}

/**
 * \brief Solve a 3x3 symmetric linear system with n_rhs right-hand sides.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void SolveSym3x3(const int n_rhs, const Real * S, const Real * B, Real * X)
{
    const Real inv_delta = 1.0/(    S[0+0*3]*S[1+1*3]*S[2+2*3]+
                                2.0*S[0+1*3]*S[0+2*3]*S[1+2*3]-
                                    S[0+2*3]*S[0+2*3]*S[1+1*3]-
                                    S[0+0*3]*S[1+2*3]*S[1+2*3]-
                                    S[0+1*3]*S[0+1*3]*S[2+2*3]);

    for (int rh = 0; rh < n_rhs; ++rh)
    {
        X[0+rh*3]  = (S[1+1*3]*S[2+2*3]-S[1+2*3]*S[1+2*3])*B[0+rh*3];
        X[1+rh*3]  = (S[0+2*3]*S[1+2*3]-S[0+1*3]*S[2+2*3])*B[0+rh*3];
        X[2+rh*3]  = (S[0+1*3]*S[1+2*3]-S[0+2*3]*S[1+1*3])*B[0+rh*3];

        X[0+rh*3] += (S[0+2*3]*S[1+2*3]-S[0+1*3]*S[2+2*3])*B[1+rh*3];
        X[1+rh*3] += (S[0+0*3]*S[2+2*3]-S[0+2*3]*S[0+2*3])*B[1+rh*3];
        X[2+rh*3] += (S[0+1*3]*S[0+2*3]-S[0+0*3]*S[1+2*3])*B[1+rh*3];

        X[0+rh*3] += (S[0+1*3]*S[1+2*3]-S[0+2*3]*S[1+1*3])*B[2+rh*3];
        X[1+rh*3] += (S[0+1*3]*S[0+2*3]-S[0+0*3]*S[1+2*3])*B[2+rh*3];
        X[2+rh*3] += (S[0+0*3]*S[1+1*3]-S[0+1*3]*S[0+1*3])*B[2+rh*3];

        X[0+rh*3] *= inv_delta;
        X[1+rh*3] *= inv_delta;
        X[2+rh*3] *= inv_delta;
    }
}

/**
 * \brief Eval limited slopes and store them. They will be used to reconstruct the solution.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dX: a MultiFab object that contains the limited slopes for the unknown solution
 *                 fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_DU2DC(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DU,
 *                               Real * DC) const;
 *                  void F_DC2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DC,
 *                               Real * DU) const;
 *                  void F_U2P(const Real t, const Real * x,
 *                             const Real * SOL, Real * P) const;
 *                  void F_DP2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DP,
 *                               Real * DU) const;
*/
template <typename IBVP_CLASS>
void EvalLimitedSlopesByLeastSquares(const Real t,
                                     const ImplicitMesh & mesh,
                                     const int N_SOL,
                                     const MultiFab & X,
                                     MultiFab & dX,
                                     const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    
    // DOMAINS
    const int dom = 0;

    // NEIGHBORS INFO FOR LEAST-SQUARE SLOPE LIMITING
    const GpuArray<int, LS_N_NBR> aux_table_i =
    {
#if (AMREX_SPACEDIM == 1)
        -1, +1
#endif
#if (AMREX_SPACEDIM == 2)
        -1, -1, -1, 0, 0, +1, +1, +1
#endif
#if (AMREX_SPACEDIM == 3)
        -1, -1, -1, 0, 0, 0, +1, +1, +1,
        -1, -1, -1, 0,    0, +1, +1, +1,
        -1, -1, -1, 0, 0, 0, +1, +1, +1,
#endif
    };
    const GpuArray<int, LS_N_NBR> aux_table_j =
    {
#if (AMREX_SPACEDIM == 1)
        0, 0
#endif
#if (AMREX_SPACEDIM == 2)
        -1, 0, +1, -1, +1, -1, 0, +1
#endif
#if (AMREX_SPACEDIM == 3)
        -1, 0, +1, -1, 0, +1, -1, 0, +1,
        -1, 0, +1, -1,    +1, -1, 0, +1,
        -1, 0, +1, -1, 0, +1, -1, 0, +1
#endif
    };
    const GpuArray<int, LS_N_NBR> aux_table_k = 
    {
#if (AMREX_SPACEDIM == 1)
        0, 0
#endif
#if (AMREX_SPACEDIM == 2)
        0, 0, 0, 0, 0, 0, 0, 0
#endif
#if (AMREX_SPACEDIM == 3)
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
         0,  0,  0,  0,      0,  0,  0,  0,
        +1, +1, +1, +1, +1, +1, +1, +1, +1
#endif
    };
    // ================================================================

    // INITIALIZATION
    dX = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ((etype%10 == __DG_ELM_TYPE_ENTIRE__) ||
                                       (etype%10 == __DG_ELM_TYPE_LARGE__));

            // LOCAL VARIABLES
            Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM], dxc[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL], P[DG_N_SOL];
            Real nbr_SOL[DG_N_SOL], nbr_P[DG_N_SOL];
            Real dSOL[DG_N_SOL];
            
            Real S[AMREX_SPACEDIM*AMREX_SPACEDIM];
            Real B[AMREX_SPACEDIM*DG_N_SOL];
            Real dP[AMREX_SPACEDIM*DG_N_SOL];
            Real min_P[DG_N_SOL], max_P[DG_N_SOL];
            Real alpha[DG_N_SOL], rP[DG_N_SOL];

            if (elm_is_valid)
            {
                // CURRENT CELL CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                )

                // CURRENT CELL SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(i,j,k,ru);
                }
                IBVP.F_U2P(t, xc, SOL, P);
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    min_P[ru] = P[ru];
                    max_P[ru] = P[ru];
                }

                // INIT LEAST-SQUARE MATRIX AND RHS
                for (int r = 0; r < AMREX_SPACEDIM*AMREX_SPACEDIM; ++r)
                {
                    S[r] = 0.0;
                }
                for (int r = 0; r < AMREX_SPACEDIM*DG_N_SOL; ++r)
                {
                    B[r] = 0.0;
                }

                // LOOP OVER THE NEIGHBORS
                for (int n = 0; n < LS_N_NBR; ++n)
                {
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_valid = ((nbr_etype%10 == __DG_ELM_TYPE_ENTIRE__) ||
                                               (nbr_etype%10 == __DG_ELM_TYPE_LARGE__));
                    if (!nbr_is_valid) continue;
                    if ((nbr_i < domain.smallEnd(0)) || (nbr_i > domain.bigEnd(0))) continue;
#if (AMREX_SPACEDIM > 1)
                    if ((nbr_j < domain.smallEnd(1)) || (nbr_j > domain.bigEnd(1))) continue;
#endif
#if (AMREX_SPACEDIM > 2)
                    if ((nbr_k < domain.smallEnd(2)) || (nbr_k > domain.bigEnd(2))) continue;
#endif

                    // NEIGHBOR CELL CENTROID
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                    )

                    // NEIGHBOR CELL SOLUTION
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        nbr_SOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                    }
                    IBVP.F_U2P(t, nbr_xc, nbr_SOL, nbr_P);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        min_P[ru] = std::min(min_P[ru], nbr_P[ru]);
                        max_P[ru] = std::max(max_P[ru], nbr_P[ru]);
                    }

                    // FILL LEAST-SQUARE MATRIX AND RHS
#if (AMREX_SPACEDIM == 2)
                    S[0+0*2] += (nbr_xc[0]-xc[0])*(nbr_xc[0]-xc[0]);
                    S[0+1*2] += (nbr_xc[0]-xc[0])*(nbr_xc[1]-xc[1]);
                    S[1+1*2] += (nbr_xc[1]-xc[1])*(nbr_xc[1]-xc[1]);

                    for (int rh = 0; rh < DG_N_SOL; ++rh)
                    {
                        B[0+rh*2] += (nbr_xc[0]-xc[0])*(nbr_P[rh]-P[rh]);
                        B[1+rh*2] += (nbr_xc[1]-xc[1])*(nbr_P[rh]-P[rh]);
                    }
#endif
#if (AMREX_SPACEDIM == 3)
                    S[0+0*3] += (nbr_xc[0]-xc[0])*(nbr_xc[0]-xc[0]);
                    S[0+1*3] += (nbr_xc[0]-xc[0])*(nbr_xc[1]-xc[1]);
                    S[0+2*3] += (nbr_xc[0]-xc[0])*(nbr_xc[2]-xc[2]);
                    S[1+1*3] += (nbr_xc[1]-xc[1])*(nbr_xc[1]-xc[1]);
                    S[1+2*3] += (nbr_xc[2]-xc[2])*(nbr_xc[2]-xc[2]);
                    S[2+2*3] += (nbr_xc[2]-xc[2])*(nbr_xc[2]-xc[2]);

                    for (int rh = 0; rh < DG_N_SOL; ++rh)
                    {
                        B[0+rh*3] += (nbr_xc[0]-xc[0])*(nbr_P[rh]-P[rh]);
                        B[1+rh*3] += (nbr_xc[1]-xc[1])*(nbr_P[rh]-P[rh]);
                        B[2+rh*3] += (nbr_xc[2]-xc[2])*(nbr_P[rh]-P[rh]);
                    }
#endif
                }

                // SYMMETRY CONDITION FOR LEAST-SQUARE MATRIX 
#if (AMREX_SPACEDIM == 2)
                S[1+0*2] = S[0+1*2];
#endif
#if (AMREX_SPACEDIM == 3)
                S[1+0*3] = S[0+1*3];
                S[2+0*3] = S[0+2*3];
                S[2+1*3] = S[1+2*3];
#endif
                // SOLVE SYSTEM
#if (AMREX_SPACEDIM == 2)
                SolveSym2x2(DG_N_SOL, S, B, dP);
#endif
#if (AMREX_SPACEDIM == 3)
                SolveSym3x3(DG_N_SOL, S, B, dP);
#endif
                // LIMIT THE SLOPES
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    alpha[ru] = 1.0;
                }
                for (int n = 0; n < LS_N_NBR; ++n)
                {
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_valid = ((nbr_etype%10 == __DG_ELM_TYPE_ENTIRE__) ||
                                               (nbr_etype%10 == __DG_ELM_TYPE_LARGE__));
                    if (!nbr_is_valid) continue;
                    if ((nbr_i < domain.smallEnd(0)) || (nbr_i > domain.bigEnd(0))) continue;
#if (AMREX_SPACEDIM > 1)
                    if ((nbr_j < domain.smallEnd(1)) || (nbr_j > domain.bigEnd(1))) continue;
#endif
#if (AMREX_SPACEDIM > 2)
                    if ((nbr_k < domain.smallEnd(2)) || (nbr_k > domain.bigEnd(2))) continue;
#endif
                    // NEIGHBOR CELL CENTROID
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                    )

                    // NEIGHBOR CELL SOLUTION
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        nbr_SOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                    }
                    IBVP.F_U2P(t, nbr_xc, nbr_SOL, nbr_P);

                    // DELTA BETWEEN NEIGHBOR CELL AND CELL CENTROIDS
                    AMREX_D_TERM
                    (
                        dxc[0] = nbr_xc[0]-xc[0];,
                        dxc[1] = nbr_xc[1]-xc[1];,
                        dxc[2] = nbr_xc[2]-xc[2];
                    )

                    // RECONSTRUCT THE SOLUTION AND EVAL THE LIMITER
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        rP[ru] = P[ru];
                        AMREX_D_TERM
                        (
                            rP[ru] += dP[0+ru*AMREX_SPACEDIM]*dxc[0];,
                            rP[ru] += dP[1+ru*AMREX_SPACEDIM]*dxc[1];,
                            rP[ru] += dP[2+ru*AMREX_SPACEDIM]*dxc[2];
                        )
                        if (rP[ru] > nbr_P[ru])
                        {
                            alpha[ru] = std::min(alpha[ru], (max_P[ru]-nbr_P[ru])/(rP[ru]-nbr_P[ru]));
                        }
                        else if (rP[ru] < nbr_P[ru])
                        {
                            alpha[ru] = std::min(alpha[ru], (min_P[ru]-nbr_P[ru])/(rP[ru]-nbr_P[ru]));
                        }
                    }   
                }

                // FINAL SLOPES
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    AMREX_D_TERM
                    (
                        dP[0+ru*AMREX_SPACEDIM] *= alpha[ru];,
                        dP[1+ru*AMREX_SPACEDIM] *= alpha[ru];,
                        dP[2+ru*AMREX_SPACEDIM] *= alpha[ru];
                    )
                }

                // COPY TO THE MULTIFAB
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    // We use P as a tmp variable
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        P[ru] = dP[dir+ru*AMREX_SPACEDIM];
                    }

                    IBVP.F_DP2DU(t, xc, SOL, P, dSOL);

                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dX_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = dSOL[ru];
                    }
                }
            }
            
        });
        Gpu::synchronize();

    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V_{,i}^T NFn
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] dX: a MultiFab object that contains the limited slopes for the unknown solution
 *                fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, const Real * nbr_SOL, Real * NFn) const;
 *                  void F_NF_BCS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, Real * NFn) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const ImplicitMesh & mesh,
                          const int N_SOL,
                          const MultiFab & X,
                          const MultiFab & dX,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // DOMAIN
    const int dom = 0;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        AMREX_D_TERM
        (
            Array4<int const> const & eBouQuad_Nq_fab_0 = mesh.eBouQuad_Nq[0].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab_0 = mesh.eBouQuad_pos[0].array(mfi);,
            Array4<int const> const & eBouQuad_Nq_fab_1 = mesh.eBouQuad_Nq[1].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab_1 = mesh.eBouQuad_pos[1].array(mfi);,
            Array4<int const> const & eBouQuad_Nq_fab_2 = mesh.eBouQuad_Nq[2].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab_2 = mesh.eBouQuad_pos[2].array(mfi);
        )
        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // CHECK WHETHER IS NOT EMPTY
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

            if (elm_is_not_empty)
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], NFn[DG_N_SOL];
                Real integrand;
                
                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

#ifdef DG_LIMIT_BY_LS
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )
#else
                AMREX_D_TERM
                (
                    xc[0] = prob_lo[0]+(BF_i+0.5)*dx[0];,
                    xc[1] = prob_lo[1]+(BF_j+0.5)*dx[1];,
                    xc[2] = prob_lo[2]+(BF_k+0.5)*dx[2];
                )
#endif

                // CELL BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    const int dir = b/2;

                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    int fi, fj, fk;
                    int bou_Nq;
                    long pos;
                    bool intraphase;

                    // NEIGHBOR CELL INFO
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // Conditions for NOT computing the flux
                    // contribution:
                    // i) the current cell extends to the neighboring
                    //    cell;
                    // ii) the current cell is small and the
                    //     neighboring cell extends into it;
                    // iii) the neighboring cell is empty (it means
                    //      that the cell must be separated by the zero
                    //      level set of a function; this case is
                    //      handled by the INTERPHASE case.
                    //
                    const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                    const bool nbr_is_empty = (nbr_etype == __DG_ELM_TYPE_EMPTY__);
                    if (cells_are_merged || nbr_is_empty) continue;

                    // SUPPORT OF THE BASIS FUNCTIONS (NEIGHBOR CELL)
                    BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                    
#ifdef DG_LIMIT_BY_LS
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                    )
#else
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_BF_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_BF_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_BF_k+0.5)*dx[2];
                    )
#endif

                    // WHICH FACE
                    GRID_FACE(i, j, k, b, fi, fj, fk);

                    // QUADRATURE INFO
                    intraphase = true;
                    if (dir == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_0(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

                        if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_1(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

                        if ((fj == domain.smallEnd(dir) || fj == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_2(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

                        if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }
                    }
#endif
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = 2.0*(b%2)-1.0;

                    // INTRAPHASE -------------------------------------
                    if (intraphase)
                    {
                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0]+cell_lo[0];,
                                x[1] = xptr[x_pos+1]+cell_lo[1];,
                                x[2] = xptr[x_pos+2]+cell_lo[2];
                            )
                            w = xptr[x_pos+AMREX_SPACEDIM];
                            
                            // SOLUTION
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                SOL[cu] = X_fab(BF_i,BF_j,BF_k,cu);
//#ifdef DG_LIMIT_BY_LS
                                AMREX_D_TERM
                                (
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
//#else
//                                SOL[cu] += dX_fab(BF_i,BF_j,BF_k,dir+AMREX_SPACEDIM*cu)*(x[dir]-xc[dir]);
//#endif

                                nbr_SOL[cu] = X_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,cu);
//#ifdef DG_LIMIT_BY_LS
                                AMREX_D_TERM
                                (
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
//#else
//                                nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,dir+AMREX_SPACEDIM*cu)*(x[dir]-nbr_xc[dir]);
//#endif
                            }

                            // NUMERICAL FLUX FOR INTRAPHASE
                            IBVP.F_NF_ICS(t, x, un, SOL, nbr_SOL, NFn);

                            // INTEGRAL CONTRIBUTION
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                    // ------------------------------------------------
                    // BOUNDARY CONDITIONS: GRID BOUNDARY -------------
                    else
                    {
                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0]+cell_lo[0];,
                                x[1] = xptr[x_pos+1]+cell_lo[1];,
                                x[2] = xptr[x_pos+2]+cell_lo[2];
                            )
                            w = xptr[x_pos+AMREX_SPACEDIM];
                            
                            // SOLUTION
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                SOL[cu] = X_fab(BF_i,BF_j,BF_k,cu);
                                AMREX_D_TERM
                                (
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

                            // NUMERICAL FLUX FOR GRID'S BOUNDARY
                            IBVP.F_NF_BCS(t, x, un, SOL, NFn);

                            // INTEGRAL CONTRIBUTION
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                    // ------------------------------------------------
                }

                // BOUNDARY CONDITIONS: ZERO LEVEL OF THE LEVELSET ----
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // EVAL THE INTEGRAL
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT AND WEIGHT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0];,
                            x[1] = xptr[x_pos+1];,
                            x[2] = xptr[x_pos+2];
                        )
                        w = xptr[x_pos+AMREX_SPACEDIM];

                        // UNIT NORMAL
                        AMREX_D_TERM
                        (
                            un[0] = xptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = xptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = xptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        // SOLUTION
                        for (int cu = 0; cu < DG_N_SOL; ++cu)
                        {
                            SOL[cu] = X_fab(BF_i,BF_j,BF_k,cu);
                            AMREX_D_TERM
                            (
                                SOL[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                SOL[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                SOL[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                            )
                        }

                        // NUMERICAL FLUX FOR INTRAPHASE
                        IBVP.F_NF_PHI_BCS(t, x, un, SOL, NFn);

                        // INTEGRAL CONTRIBUTION
                        integrand = NFn[ru];
                        dXdt_fab(i,j,k,ru) -= integrand*w;
                    }
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (dXdt.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
        msg += "| dXdt contains nans.\n";
        amrex::Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Eval dXdt.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_DU2DC(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DU,
 *                               Real * DC) const;
 *                  void F_DC2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DC, 
 *                               Real * DU) const;
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, const Real * nbr_SOL, Real * NFn) const;
 *                  void F_U2P(const Real t, const Real * x,
 *                             const Real * SOL, Real * P) const;
 *                  void F_DP2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DP,
 *                               Real * DU) const;
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const ImplicitMesh & mesh,
               const MatrixFactory & matfactory,
               const int N_SOL,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    // DOMAIN
    const int dom = 0;
    const Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, dom);
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dX(ba, dm, AMREX_SPACEDIM*N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL LIMITED SLOPES ============================================
#ifdef DG_LIMIT_BY_LS
    EvalLimitedSlopesByLeastSquares(t, mesh, N_SOL, X, dX, IBVP);
#else
    EvalLimitedSlopes(t, mesh, N_SOL, X, dX, IBVP);
#endif
    // ================================================================

    // EVAL THE TIME DERIVATIVES ======================================
    Eval_dXdt_BouContrib(t, mesh, N_SOL, X, dX, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    MultiplyByInverseMassMatrix(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using 2nd-order RungeKutta scheme
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients will be updated according to
 *                  a 2nd-order RungeKutta algorithm.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * SOL, const Real * bcs_SOL) const;
 *                  void F_PHI_BCS(const Real t, const Real * x, const Real * un,
 *                                 const Real * SOL, const Real * bcs_SOL) const;
 *                  void F_DU2DC(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DU,
 *                               Real * DC) const;
 *                  void F_DC2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DC, 
 *                               Real * DU) const;
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, const Real * nbr_SOL, Real * NFn) const;
 *                  void F_U2P(const Real t, const Real * x,
 *                             const Real * SOL, Real * P) const;
 *                  void F_DP2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DP,
 *                               Real * DU) const;
*/
template <typename IBVP_CLASS>
void TakeTimeStep(const Real dt, const Real t,
                  const ImplicitMesh & mesh,
                  const MatrixFactory & matfactory,
                  const int N_SOL,
                  MultiFab & X_new,
                  const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X_new.boxarray;
    const DistributionMapping & dm = X_new.distributionMap;

    // NUMBER OF COMPONENTS AND GHOST ROWS
    const int n_comp = X_new.n_comp;
    const IntVect ngr = X_new.n_grow;
    // ================================================================
    // NOTE: In this case, the number of components coincides with the
    //       number of unknown solution fields, namely N_SOL.
    // ================================================================

    // VARIABLES ======================================================
    MultiFab X_old(ba, dm, n_comp, ngr);
    MultiFab X1(ba, dm, n_comp, ngr);
    MultiFab dXdt(ba, dm, n_comp, ngr);
    // ================================================================

    // COPY NEW TO OLD ================================================
    MultiFab::Copy(X_old, X_new, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 1 =====================================================
    FillInvalidCells(t, mesh, N_SOL, X_old, IBVP);

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, mesh, matfactory, N_SOL, X_old, dXdt, IBVP);

    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillInvalidCells(t, mesh, N_SOL, X1, IBVP);

    // dXdt := dXdt(t+dt, X1)
    Eval_dXdt(t+dt, mesh, matfactory, N_SOL, X1, dXdt, IBVP);

    // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
    MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
    MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // CHECK SOLUTION =================================================
    if (X_new.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - TakeTimeStep\n";
        msg += "| Solution contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}


} // namespace DG
} // namespace amrex