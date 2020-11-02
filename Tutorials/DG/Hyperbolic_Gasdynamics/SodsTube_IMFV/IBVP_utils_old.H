//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define DG_N_SOL (1+AMREX_SPACEDIM+1)

#define DG_LIMIT_CP

#define DG_SL_THETA 2.0

#define DG_LIMIT_BY_LS

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
 * \brief Fill cells at the grid's boundaries using boundary conditions.
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
*/
template <typename IBVP_CLASS>
void FillBoundaryCells(const Real t,
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
    
    // DOMAINS
    const int dom = 0;
    // ================================================================

    // CHECK ==========================================================
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillBoundaryCells\n";
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
        msg += "ERROR: IBVP_utils.H - FillBoundaryCells\n";
        msg += "| X contains nans.\n";
        Warning(msg);
    }
    // ================================================================
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
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

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

/*
if (i == 35  && j == 25)
{
Print() << "un: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
Print() << "nbr_i: " << nbr_i << std::endl;
Print() << "nbr_j: " << nbr_j << std::endl;
Print() << "SOL: "; IO::PrintRealArray2D(1, N_SOL, SOL);
Print() << "bcs_SOL: "; IO::PrintRealArray2D(1, N_SOL, bcs_SOL);
Print() << "HELLO!" << std::endl;
exit(-1);
}
*/

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


/** \brief Fill Least-square matrix and right-hand side with zeros.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void InitLS(Real * S, Real * B)
{
    for (int r = 0; r < AMREX_SPACEDIM*AMREX_SPACEDIM; ++r)
    {
        S[r] = 0.0;
    }
    for (int r = 0; r < AMREX_SPACEDIM*DG_N_SOL; ++r)
    {
        B[r] = 0.0;
    }
}

/** \brief Add contribution to the Least-square matrix and right-hand side.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2LS(Real * S, Real * B, const Real * xc, const Real * SOL, const Real * xb, const Real * ls_SOL)
{
#if (AMREX_SPACEDIM == 2)
    S[0+0*2] += (xb[0]-xc[0])*(xb[0]-xc[0]);
    S[0+1*2] += (xb[0]-xc[0])*(xb[1]-xc[1]);
    S[1+0*2]  = S[0+1*2];
    S[1+1*2] += (xb[1]-xc[1])*(xb[1]-xc[1]);

    for (int rh = 0; rh < DG_N_SOL; ++rh)
    {
        B[0+rh*2] += (xb[0]-xc[0])*(ls_SOL[rh]-SOL[rh]);
        B[1+rh*2] += (xb[1]-xc[1])*(ls_SOL[rh]-SOL[rh]);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    S[0+0*3] += (xb[0]-xc[0])*(xb[0]-xc[0]);
    S[0+1*3] += (xb[0]-xc[0])*(xb[1]-xc[1]);
    S[0+2*3] += (xb[0]-xc[0])*(xb[2]-xc[2]);
    S[1+0*3]  = S[0+1*3];
    S[2+0*3]  = S[0+2*3];
    S[1+1*3] += (xb[1]-xc[1])*(xb[1]-xc[1]);
    S[1+2*3] += (xb[2]-xc[2])*(xb[2]-xc[2]);
    S[2+1*3]  = S[1+2*3];
    S[2+2*3] += (xb[2]-xc[2])*(xb[2]-xc[2]);

    for (int rh = 0; rh < DG_N_SOL; ++rh)
    {
        B[0+rh*3] += (xb[0]-xc[0])*(ls_SOL[rh]-SOL[rh]);
        B[1+rh*3] += (xb[1]-xc[1])*(ls_SOL[rh]-SOL[rh]);
        B[2+rh*3] += (xb[2]-xc[2])*(ls_SOL[rh]-SOL[rh]);
    }
#endif
}

/** \brief Init the lists of minimum and maximum values.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void InitMinMaxLS(const int N_SOL, Real * min_x, Real * max_x, Real * min_SOL, Real * max_SOL, const Real * xc, const Real * SOL)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        AMREX_D_TERM
        (
            min_x[0] = xc[0];,
            min_x[1] = xc[1];,
            min_x[2] = xc[2];
        )
        AMREX_D_TERM
        (
            max_x[0] = xc[0];,
            max_x[1] = xc[1];,
            max_x[2] = xc[2];
        )

        min_SOL[ru] = SOL[ru];
        max_SOL[ru] = SOL[ru];
    }
}

/** \brief Update the lists of minimum and maximum values.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void UpdateMinMaxLS(const int N_SOL, Real * min_x, Real * max_x, Real * min_SOL, Real * max_SOL, const Real * xb, const Real * SOL)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        min_SOL[ru] = std::min(min_SOL[ru], SOL[ru]);
        max_SOL[ru] = std::max(max_SOL[ru], SOL[ru]);

        AMREX_D_TERM
        (
            min_x[0] = std::min(min_x[0], xb[0]);,
            min_x[1] = std::min(min_x[1], xb[1]);,
            min_x[2] = std::min(min_x[2], xb[2]);
        )
        AMREX_D_TERM
        (
            max_x[0] = std::max(max_x[0], xb[0]);,
            max_x[1] = std::max(max_x[1], xb[1]);,
            max_x[2] = std::max(max_x[2], xb[2]);
        )
    }
}

/** \brief Reconstruct solution fields using input slopes.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void TestReconstruction(const int N_SOL, const Real * xc, const Real * x, const Real * SOL, const Real * D, Real * rSOL)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        rSOL[ru] = SOL[ru];
        AMREX_D_TERM
        (
            rSOL[ru] += D[0+ru*AMREX_SPACEDIM]*(x[0]-xc[0]);,
            rSOL[ru] += D[1+ru*AMREX_SPACEDIM]*(x[1]-xc[1]);,
            rSOL[ru] += D[2+ru*AMREX_SPACEDIM]*(x[2]-xc[2]);
        )
    }
}

/** \brief Reconstruct min solution fields using input slopes.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ReconstructMin(const int N_SOL, const Real * xc, const Real * min_x, const Real * max_x, const Real * SOL, const Real * D, Real * rSOL)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        rSOL[ru] = SOL[ru];
        if (D[0+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[0+ru*AMREX_SPACEDIM]*(min_x[0]-xc[0]);
        else                              rSOL[ru] += D[0+ru*AMREX_SPACEDIM]*(max_x[0]-xc[0]);
#if (AMREX_SPACEDIM > 1)
        if (D[1+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[1+ru*AMREX_SPACEDIM]*(min_x[1]-xc[1]);
        else                              rSOL[ru] += D[1+ru*AMREX_SPACEDIM]*(max_x[1]-xc[1]);
#endif
#if (AMREX_SPACEDIM > 2)
        if (D[2+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[2+ru*AMREX_SPACEDIM]*(min_x[2]-xc[2]);
        else                              rSOL[ru] += D[2+ru*AMREX_SPACEDIM]*(max_x[2]-xc[2]);
#endif
    }
}

/** \brief Reconstruct max solution fields using input slopes.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ReconstructMax(const int N_SOL, const Real * xc, const Real * min_x, const Real * max_x, const Real * SOL, const Real * D, Real * rSOL)
{
    for (int ru = 0; ru < N_SOL; ++ru)
    {
        rSOL[ru] = SOL[ru];
        if (D[0+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[0+ru*AMREX_SPACEDIM]*(max_x[0]-xc[0]);
        else                              rSOL[ru] += D[0+ru*AMREX_SPACEDIM]*(min_x[0]-xc[0]);
#if (AMREX_SPACEDIM > 1)
        if (D[1+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[1+ru*AMREX_SPACEDIM]*(max_x[1]-xc[1]);
        else                              rSOL[ru] += D[1+ru*AMREX_SPACEDIM]*(min_x[1]-xc[1]);
#endif
#if (AMREX_SPACEDIM > 2)
        if (D[2+ru*AMREX_SPACEDIM] > 0.0) rSOL[ru] += D[2+ru*AMREX_SPACEDIM]*(max_x[2]-xc[2]);
        else                              rSOL[ru] += D[2+ru*AMREX_SPACEDIM]*(min_x[2]-xc[2]);
#endif
    }
}

/** \brief Limit slopes.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void LimitSlopesLS(const int N_SOL, const Real * xc, const Real * min_x, const Real * max_x,
                   const Real * SOL, const Real * min_SOL, const Real * max_SOL,
                   Real * D)
{
    Real rSOL[DG_N_SOL], alpha[DG_N_SOL];

    ReconstructMin(N_SOL, xc, min_x, max_x, SOL, D, rSOL);
    for (int ru = 0; ru < DG_N_SOL; ++ru)
    {
        if (rSOL[ru] < min_SOL[ru])
        {
            alpha[ru] = (min_SOL[ru]-SOL[ru])/(rSOL[ru]-SOL[ru]);
        }
        else
        {
            alpha[ru] = 1.0;
        }
    }

    ReconstructMax(N_SOL, xc, min_x, max_x, SOL, D, rSOL);
    for (int ru = 0; ru < DG_N_SOL; ++ru)
    {
        if (rSOL[ru] > max_SOL[ru])
        {
            alpha[ru] = std::min(alpha[ru], (max_SOL[ru]-SOL[ru])/(rSOL[ru]-SOL[ru]));
        }
    }

    for (int ru = 0; ru < N_SOL; ++ru)
    {
        AMREX_D_TERM
        (
            D[0+ru*AMREX_SPACEDIM] *= alpha[ru];,
            D[1+ru*AMREX_SPACEDIM] *= alpha[ru];,
            D[2+ru*AMREX_SPACEDIM] *= alpha[ru];
        )
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
void EvalLimitedSlopesByLeastSquares2(const Real t,
                                      const ImplicitMesh & mesh,
                                      const int N_SOL,
                                      const MultiFab & X,
                                      MultiFab & dX,
                                      const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();
    
    // DOMAINS
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

    // VARIABLES ======================================================
    // AUXILIARY MULTIFABS
    Array<MultiFab, AMREX_SPACEDIM> bX;
    MultiFab ibX;
    // ================================================================

    // INITIALIZATION =================================================
    dX = 0.0;

    // AUXILIARY MULTIFABS
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        bX[dir].define(mesh.fc_ba[dir], mesh.dm, DG_N_SOL, 0);
        bX[dir] = 0.0;
    }
    ibX.define(mesh.cc_ba, mesh.dm, DG_N_SOL, 0);
    ibX = 0.0;
    // ================================================================

    // EVAL THE BOUNDARY LEAST-SQUARE VALUES ==========================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        for (MFIter mfi(bX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<Real const> const & eBouInfo_fab = mesh.eBouInfo[dir].array(mfi);
            
            Array4<Real const> const & X_fab = X.array(mfi);

            Array4<Real> const & bX_fab = bX[dir].array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk) noexcept
            {
                // CHECK WHETHER THE BOUNDARY IS VALID
                const short ebtype = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));
                const bool eb_is_valid = ((ebtype != __DG_ELM_BOU_TYPE_EMPTY__) &&
                                          (ebtype != __DG_ELM_BOU_TYPE_PARTIAL_INVALID__));

                if (eb_is_valid)
                {
                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk, i, j, k;
                    int BF_mi, BF_mj, BF_mk, BF_pi, BF_pj, BF_pk;
                    bool intraphase;
                    Real xb[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                    Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], SOL_RP[DG_N_SOL];

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE2NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);

                    // BOUNDARY CENTROID
                    AMREX_D_TERM
                    (
                        xb[0] = eBouInfo_fab(fi,fj,fk,ELM_BOU_CENTROID_X1(dom));,
                        xb[1] = eBouInfo_fab(fi,fj,fk,ELM_BOU_CENTROID_X2(dom));,
                        xb[2] = eBouInfo_fab(fi,fj,fk,ELM_BOU_CENTROID_X3(dom));
                    )

                    // INIT UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)

                    // INTRAPHASE VS GRID BOUNDARY
                    intraphase = true;
                    if ((dir == 0) && (is_periodic[dir] == 0))
                    {
                        if (fi == domain.smallEnd(dir))
                        {
                            intraphase = false;
                            i = pi;
                            j = pj;
                            k = pk;
                            un[dir] = -1.0;
                        }
                        else if (fi == (domain.bigEnd(dir)+1))
                        {
                            intraphase = false;
                            i = mi;
                            j = mj;
                            k = mk;
                            un[dir] = +1.0;
                        }
                    }
#if (AMREX_SPACEDIM > 1)
                    else if ((dir == 1) && (is_periodic[dir] == 0))
                    {
                        if (fj == domain.smallEnd(dir))
                        {
                            intraphase = false;
                            i = pi;
                            j = pj;
                            k = pk;
                            un[dir] = -1.0;
                        }
                        else if (fj == (domain.bigEnd(dir)+1))
                        {
                            intraphase = false;
                            i = mi;
                            j = mj;
                            k = mk;
                            un[dir] = +1.0;
                        }
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else if ((dir == 2) && (is_periodic[dir] == 0))
                    {
                        if (fk == domain.smallEnd(dir))
                        {
                            intraphase = false;
                            i = pi;
                            j = pj;
                            k = pk;
                            un[dir] = -1.0;
                        }
                        else if (fk == (domain.bigEnd(dir)+1))
                        {
                            intraphase = false;
                            i = mi;
                            j = mj;
                            k = mk;
                            un[dir] = +1.0;
                        }
                    }
#endif
                    else
                    {
                        un[dir] = +1.0;
                    }

                    // INTRAPHASE -------------------------------------
                    if (intraphase)
                    {
                        // CELLS INFO
                        const short m_etype = eType_fab(mi,mj,mk,ELM_TYPE(dom));
                        const short p_etype = eType_fab(pi,pj,pk,ELM_TYPE(dom));

                        // SUPPORT OF THE BASIS FUNCTIONS
                        BF_CELL(mi, mj, mk, m_etype, BF_mi, BF_mj, BF_mk);
                        BF_CELL(pi, pj, pk, p_etype, BF_pi, BF_pj, BF_pk);

                        // VALID CELL SOLUTION
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            SOL[ru] = X_fab(BF_mi,BF_mj,BF_mk,ru);
                            nbr_SOL[ru] = X_fab(BF_pi,BF_pj,BF_pk,ru);
                        }
                    }
                    // ------------------------------------------------
                    // GRID BOUNDARY ----------------------------------
                    else
                    {
                        int BF_i, BF_j, BF_k;

                        // VALID CELL INFO
                        const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                        // SUPPORT OF THE BASIS FUNCTIONS (VALID CELL)
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                        // VALID CELL SOLUTION
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                        }

                        // BCS
                        IBVP.F_BCS(t, xb, un, SOL, nbr_SOL);
                    }
                    // ------------------------------------------------

                    // RIEMANN SOLVER SOLUTION
                    IBVP.F_RIEMANN(t, xb, un, SOL, nbr_SOL, SOL_RP);

                    // STORE SOLUTION
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        bX_fab(fi,fj,fk,ru) = SOL_RP[ru];
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================

    // EVAL THE INTERNAL BOUNDARY LEAST-SQUARE VALUES =================
    for (MFIter mfi(ibX); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eIntBouInfo_fab = mesh.eIntBouInfo.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & ibX_fab = ibX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER THE ELEMENT IS PARTIAL
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
            const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
            const bool elm_is_partial = (elm_is_large || elm_is_small);

            if (elm_is_partial)
            {
                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xb[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], bcs_SOL[DG_N_SOL], SOL_RP[DG_N_SOL];

                // BOUNDARY CENTROID
                AMREX_D_TERM
                (
                    xb[0] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X1(dom));,
                    xb[1] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X2(dom));,
                    xb[2] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X3(dom));
                )

                // INIT UNIT NORMAL
                AMREX_D_TERM
                (
                    un[0] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN1(dom));,
                    un[1] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN2(dom));,
                    un[2] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN3(dom));
                )

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // VALID CELL SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                }

                // BCS
                IBVP.F_PHI_BCS(t, xb, un, SOL, bcs_SOL);

                // RIEMANN SOLVER SOLUTION
                IBVP.F_RIEMANN(t, xb, un, SOL, bcs_SOL, SOL_RP);

                // STORE SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    ibX_fab(i,j,k,ru) = SOL_RP[ru];
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================

    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        AMREX_D_TERM
        (
            Array4<short const> const & eBouType_fab_0 = mesh.eBouType[0].array(mfi);
            Array4<Real const> const & eBouInfo_fab_0 = mesh.eBouInfo[0].array(mfi);,
            Array4<short const> const & eBouType_fab_1 = mesh.eBouType[1].array(mfi);
            Array4<Real const> const & eBouInfo_fab_1 = mesh.eBouInfo[1].array(mfi);,
            Array4<short const> const & eBouType_fab_2 = mesh.eBouType[2].array(mfi);
            Array4<Real const> const & eBouInfo_fab_2 = mesh.eBouInfo[2].array(mfi);
        )
        Array4<Real const> const & eIntBouInfo_fab = mesh.eIntBouInfo.array(mfi);

        AMREX_D_TERM
        (
            Array4<Real const> const & bX_fab_0 = bX[0].array(mfi);,
            Array4<Real const> const & bX_fab_1 = bX[1].array(mfi);,
            Array4<Real const> const & bX_fab_2 = bX[2].array(mfi);
        )
        Array4<Real const> const & ibX_fab = ibX.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

            if (elm_is_not_empty)
            {
                // LOCAL PARAMETERS
                const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
                const bool elm_is_partial = (elm_is_large || elm_is_small);

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM], xb[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], ls_SOL[DG_N_SOL], lim_SOL[DG_N_SOL], dSOL[DG_N_SOL];
#ifdef DG_LIMIT_CP
                Real P[DG_N_SOL], ls_P[DG_N_SOL], lim_P[DG_N_SOL];
#endif
                Real S[AMREX_SPACEDIM*AMREX_SPACEDIM];
                Real B[AMREX_SPACEDIM*DG_N_SOL];
                Real D[AMREX_SPACEDIM*DG_N_SOL];
                Real min_SOL[DG_N_SOL], max_SOL[DG_N_SOL], max_x[AMREX_SPACEDIM], min_x[AMREX_SPACEDIM];
                Real rSOL[DG_N_SOL];

                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CENTROID (CURRENT CELL)
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )

                // CURRENT CELL SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                }
#ifdef DG_LIMIT_CP
                // USE PRIMARY VARIABLES
                IBVP.F_U2P(t, xc, SOL, P);
#endif
                // INIT LEAST-SQUARE MATRIX AND RHS
                InitLS(S, B);

                // INIT MIN MAX LISTS
#ifdef DG_LIMIT_CP
                InitMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xc, P);
#else
                InitMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xc, SOL);
#endif

                // LOOP OVER THE BOUNDARIES ---------------------------
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    const int dir = b/2;

                    int fi, fj, fk;
                    short ebtype;
                    bool intraphase, eb_is_valid;

                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    short nbr_etype;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;

                    // WHICH FACE
                    GRID_FACE(i, j, k, b, fi, fj, fk);

                    // CHECK WHETHER BOUNDARY IS VALID AND DETERMINE
                    // INTRAPHASE VS GRID BOUNDARY
                    intraphase = true;
                    if (dir == 0)
                    {
                        if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }

                        ebtype = eBouType_fab_0(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        if ((fj == domain.smallEnd(dir) || fj == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }

                        ebtype = eBouType_fab_1(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }

                        ebtype = eBouType_fab_2(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }
#endif
                    eb_is_valid = ((ebtype != __DG_ELM_BOU_TYPE_EMPTY__) &&
                                   (ebtype != __DG_ELM_BOU_TYPE_PARTIAL_INVALID__));

                    if (eb_is_valid)
                    {
                        if (dir == 0)
                        {
                            // BOUNDARY CENTROID
                            AMREX_D_TERM
                            (
                                xb[0] = eBouInfo_fab_0(fi,fj,fk,ELM_BOU_CENTROID_X1(dom));,
                                xb[1] = eBouInfo_fab_0(fi,fj,fk,ELM_BOU_CENTROID_X2(dom));,
                                xb[2] = eBouInfo_fab_0(fi,fj,fk,ELM_BOU_CENTROID_X3(dom));
                            )

                            // GET LEAST-SQUARE SOLUTION
                            for (int ru = 0; ru < N_SOL; ++ru)
                            {
                                ls_SOL[ru] = bX_fab_0(fi,fj,fk,ru);
                            }
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            // BOUNDARY CENTROID
                            AMREX_D_TERM
                            (
                                xb[0] = eBouInfo_fab_1(fi,fj,fk,ELM_BOU_CENTROID_X1(dom));,
                                xb[1] = eBouInfo_fab_1(fi,fj,fk,ELM_BOU_CENTROID_X2(dom));,
                                xb[2] = eBouInfo_fab_1(fi,fj,fk,ELM_BOU_CENTROID_X3(dom));
                            )

                            // GET LEAST-SQUARE SOLUTION
                            for (int ru = 0; ru < N_SOL; ++ru)
                            {
                                ls_SOL[ru] = bX_fab_1(fi,fj,fk,ru);
                            }
                        }
#endif
#if (AMREX_SPACEDIM > 2)
                        else if (dir == 2)
                        {
                            // BOUNDARY CENTROID
                            AMREX_D_TERM
                            (
                                xb[0] = eBouInfo_fab_2(fi,fj,fk,ELM_BOU_CENTROID_X1(dom));,
                                xb[1] = eBouInfo_fab_2(fi,fj,fk,ELM_BOU_CENTROID_X2(dom));,
                                xb[2] = eBouInfo_fab_2(fi,fj,fk,ELM_BOU_CENTROID_X3(dom));
                            )

                            // GET LEAST-SQUARE SOLUTION
                            for (int ru = 0; ru < N_SOL; ++ru)
                            {
                                ls_SOL[ru] = bX_fab_2(fi,fj,fk,ru);
                            }
                        }
#endif
#ifdef DG_LIMIT_CP
                        // USE PRIMARY VARIABLES
                        IBVP.F_U2P(t, xb, ls_SOL, ls_P);
                        Add2LS(S, B, xc, P, xb, ls_P);
#else
                        Add2LS(S, B, xc, SOL, xb, ls_SOL);
#endif

                        // GET LIMITING VALUES
                        if (intraphase)
                        {
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                            BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            for (int ru = 0; ru < N_SOL; ++ru)
                            {
                                lim_SOL[ru] = X_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ru);
                            }
                        }
                        else
                        {
                            AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                            un[dir] = 2.0*(b%2)-1.0;
                            IBVP.F_BCS(t, xb, un, SOL, lim_SOL);
                        }

#ifdef DG_LIMIT_CP
                        IBVP.F_U2P(t, xb, lim_SOL, lim_P);
                        UpdateMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xb, lim_P);
#else
                        UpdateMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xb, lim_SOL);
#endif

                    }
                }
                // ----------------------------------------------------

                // INTERNAL BOUNDARY (IF ANY) -------------------------
                if (elm_is_partial)
                {
                    // BOUNDARY CENTROID
                    AMREX_D_TERM
                    (
                        xb[0] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X1(dom));,
                        xb[1] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X2(dom));,
                        xb[2] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X3(dom));
                    )
                    // GET LEAST-SQUARE SOLUTION
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        ls_SOL[ru] = ibX_fab(i,j,k,ru);
                    }
#ifdef DG_LIMIT_CP
                    // USE PRIMARY VARIABLES
                    IBVP.F_U2P(t, xb, ls_SOL, ls_P);
                    Add2LS(S, B, xc, P, xb, ls_P);
#else
                    Add2LS(S, B, xc, SOL, xb, ls_SOL);
#endif

                    // GET LIMITING VALUE
                    {
                        // INIT UNIT NORMAL
                        AMREX_D_TERM
                        (
                            un[0] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN1(dom));,
                            un[1] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN2(dom));,
                            un[2] = eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN3(dom));
                        )
                        IBVP.F_PHI_BCS(t, xb, un, SOL, lim_SOL);
                    }
#ifdef DG_LIMIT_CP
                    IBVP.F_U2P(t, xb, lim_SOL, lim_P);
                    UpdateMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xb, lim_P);
#else
                    UpdateMinMaxLS(DG_N_SOL, min_x, max_x, min_SOL, max_SOL, xb, lim_SOL);
#endif
                }
                // ----------------------------------------------------

                // SOLVE SYSTEM ---------------------------------------
#if (AMREX_SPACEDIM == 2)
                SolveSym2x2(DG_N_SOL, S, B, D);
#endif
#if (AMREX_SPACEDIM == 3)
                SolveSym3x3(DG_N_SOL, S, B, D);
#endif
                // ----------------------------------------------------

                // LIMIT SLOPES ---------------------------------------
#ifdef DG_LIMIT_CP
                LimitSlopesLS(DG_N_SOL, xc, min_x, max_x, P, min_SOL, max_SOL, D);
#else
                LimitSlopesLS(DG_N_SOL, xc, min_x, max_x, SOL, min_SOL, max_SOL, D);
#endif
                // ----------------------------------------------------

                // COPY TO THE MULTIFAB -------------------------------
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
#ifdef DG_LIMIT_CP
                    // We use P as a tmp variable
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        P[ru] = D[dir+ru*AMREX_SPACEDIM];
                    }

                    IBVP.F_DP2DU(t, xc, SOL, P, dSOL);

                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dX_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = dSOL[ru];
                    }
#else
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dX_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = D[dir+ru*AMREX_SPACEDIM];
                    }
#endif
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();

    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // HANDLE THE EXTENDED ELEMENTS ===================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
            const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

            if (elm_is_extended)
            {
                // LOCAL VARIABLES
                Real dSOL[AMREX_SPACEDIM*DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_b;
                Real nbr_dSOL[AMREX_SPACEDIM*DG_N_SOL];

                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    AMREX_D_TERM
                    (
                        dSOL[0+ru*AMREX_SPACEDIM] = dX_fab(i,j,k,0+AMREX_SPACEDIM*ru);,
                        dSOL[1+ru*AMREX_SPACEDIM] = dX_fab(i,j,k,1+AMREX_SPACEDIM*ru);,
                        dSOL[2+ru*AMREX_SPACEDIM] = dX_fab(i,j,k,2+AMREX_SPACEDIM*ru);
                    )
                }

                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    // NEIGHBOR CELL INFO
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // CHECK WHETHER BOUNDARY IS VALID
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            AMREX_D_TERM
                            (
                                nbr_dSOL[0+ru*AMREX_SPACEDIM] = dX_fab(nbr_i,nbr_j,nbr_k,0+AMREX_SPACEDIM*ru);,
                                nbr_dSOL[1+ru*AMREX_SPACEDIM] = dX_fab(nbr_i,nbr_j,nbr_k,1+AMREX_SPACEDIM*ru);,
                                nbr_dSOL[2+ru*AMREX_SPACEDIM] = dX_fab(nbr_i,nbr_j,nbr_k,2+AMREX_SPACEDIM*ru);
                            )
                        }

                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            AMREX_D_TERM
                            (
                                if ((dSOL[0+ru*AMREX_SPACEDIM] > 0.0) && (nbr_dSOL[0+ru*AMREX_SPACEDIM] > 0.0))
                                {
                                    dSOL[0+ru*AMREX_SPACEDIM] = std::min(dSOL[0+ru*AMREX_SPACEDIM], nbr_dSOL[0+ru*AMREX_SPACEDIM]);
                                }
                                else if ((dSOL[0+ru*AMREX_SPACEDIM] < 0.0) && (nbr_dSOL[0+ru*AMREX_SPACEDIM] < 0.0))
                                {
                                    dSOL[0+ru*AMREX_SPACEDIM] = std::max(dSOL[0+ru*AMREX_SPACEDIM], nbr_dSOL[0+ru*AMREX_SPACEDIM]);
                                }
                                else
                                {
                                    dSOL[0+ru*AMREX_SPACEDIM] = 0.0;
                                },
                                if ((dSOL[1+ru*AMREX_SPACEDIM] > 0.0) && (nbr_dSOL[1+ru*AMREX_SPACEDIM] > 0.0))
                                {
                                    dSOL[1+ru*AMREX_SPACEDIM] = std::min(dSOL[1+ru*AMREX_SPACEDIM], nbr_dSOL[1+ru*AMREX_SPACEDIM]);
                                }
                                else if ((dSOL[1+ru*AMREX_SPACEDIM] < 0.0) && (nbr_dSOL[1+ru*AMREX_SPACEDIM] < 0.0))
                                {
                                    dSOL[1+ru*AMREX_SPACEDIM] = std::max(dSOL[1+ru*AMREX_SPACEDIM], nbr_dSOL[1+ru*AMREX_SPACEDIM]);
                                }
                                else
                                {
                                    dSOL[1+ru*AMREX_SPACEDIM] = 0.0;
                                },
                                if ((dSOL[2+ru*AMREX_SPACEDIM] > 0.0) && (nbr_dSOL[2+ru*AMREX_SPACEDIM] > 0.0))
                                {
                                    dSOL[2+ru*AMREX_SPACEDIM] = std::min(dSOL[2+ru*AMREX_SPACEDIM], nbr_dSOL[2+ru*AMREX_SPACEDIM]);
                                }
                                else if ((dSOL[2+ru*AMREX_SPACEDIM] < 0.0) && (nbr_dSOL[2+ru*AMREX_SPACEDIM] < 0.0))
                                {
                                    dSOL[2+ru*AMREX_SPACEDIM] = std::max(dSOL[2+ru*AMREX_SPACEDIM], nbr_dSOL[2+ru*AMREX_SPACEDIM]);
                                }
                                else
                                {
                                    dSOL[2+ru*AMREX_SPACEDIM] = 0.0;
                                }
                            )
                        }
                    }
                }
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
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopesByLeastSquares2\n";
        msg += "| dX contains nans.\n";
        amrex::Abort(msg);
    }
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
    //EvalLimitedSlopesByLeastSquares(t, mesh, N_SOL, X, dX, IBVP);
    //EvalLimitedSlopesByLeastSquares2(t, mesh, N_SOL, X, dX, IBVP);
    EvalLimitedSlopesByLeastSquares3(t, mesh, N_SOL, X, dX, IBVP);
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
    FillBoundaryCells(t, mesh, N_SOL, X_old, IBVP);
#ifdef DG_LIMIT_BY_LS
#else
    FillInvalidCells(t, mesh, N_SOL, X_old, IBVP);
#endif

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, mesh, matfactory, N_SOL, X_old, dXdt, IBVP);

    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillBoundaryCells(t, mesh, N_SOL, X1, IBVP);
#ifdef DG_LIMIT_BY_LS
#else
    FillInvalidCells(t, mesh, N_SOL, X1, IBVP);
#endif

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



/*
if ((i == 34 && j == 26))
{
Print() << "(" << i << "," << j << "," << k << ")" << std::endl;
Print() << "D (limited): " << std::endl;
IO::PrintRealArray2D(AMREX_SPACEDIM, DG_N_SOL, dSOL);
Print() << "(" << nbr_i << "," << nbr_j << "," << nbr_k << ")" << std::endl;
Print() << "nbr_D (limited): " << std::endl;
IO::PrintRealArray2D(AMREX_SPACEDIM, DG_N_SOL, nbr_dSOL);
Print() << "HERE WE ARE - EvalLimitedSlopesByLeastSquares2" << std::endl << std::endl;
exit(-1); 
}
*/
/*
if ((i == 34 && j == 26) || (i == 34 && j == 25))
{
Print() << "(" << i << "," << j << "," << k << ")->(" << BF_i << "," << BF_j << "," << BF_k << ")" << std::endl;
Print() << "xc: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, xc);
Print() << "SOL: "; IO::PrintRealArray2D(1, DG_N_SOL, SOL);
Print() << "P: "; IO::PrintRealArray2D(1, DG_N_SOL, P);
Print() << "min_SOL: "; IO::PrintRealArray2D(1, DG_N_SOL, min_SOL);
Print() << "max_SOL: "; IO::PrintRealArray2D(1, DG_N_SOL, max_SOL);
Print() << "D (limited): " << std::endl;
IO::PrintRealArray2D(AMREX_SPACEDIM, DG_N_SOL, D);
Print() << "HERE WE ARE - EvalLimitedSlopesByLeastSquares2" << std::endl << std::endl;
if ((i == 34 && j == 26)) exit(-1); 
}
*/


template <typename IBVP_CLASS>
void EvalLimitedSlopesByLeastSquares_____(const Real t,
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
            // CHECK WHETHER IS VALID
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ((etype%10 == __DG_ELM_TYPE_ENTIRE__) ||
                                       (etype%10 == __DG_ELM_TYPE_LARGE__));

            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                int n_valid_nbr;
                Real x[AMREX_SPACEDIM], g[AMREX_SPACEDIM];
                Real gU[DG_N_SOL], gP[DG_N_SOL];
                Real auxSOL[DG_N_SOL], auxP[DG_N_SOL];
                Real S[AMREX_SPACEDIM*AMREX_SPACEDIM];
                Real LAM[AMREX_SPACEDIM], EV[AMREX_SPACEDIM*AMREX_SPACEDIM];
                Real un[AMREX_SPACEDIM];
                Real dUL[DG_N_SOL*AMREX_SPACEDIM], dUR[DG_N_SOL*AMREX_SPACEDIM], dU[DG_N_SOL*AMREX_SPACEDIM];
                Real dCL[DG_N_SOL*AMREX_SPACEDIM], dCR[DG_N_SOL*AMREX_SPACEDIM], dC[DG_N_SOL*AMREX_SPACEDIM];
                Real y[AMREX_SPACEDIM], y2L[AMREX_SPACEDIM], y2R[AMREX_SPACEDIM];

                // CENTROID AND AVERAGE OF THE LS CLOUD OF POINTS -----
                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    x[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                    x[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                    x[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                )
                AMREX_D_TERM
                (
                    g[0] = x[0];,
                    g[1] = x[1];,
                    g[2] = x[2];
                )

                // CURRENT ELEMENT SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    gU[ru] = X_fab(i,j,k,ru);
                }
#ifdef DG_LIMIT_P
                IBVP.F_U2P(t, x, gU, gP);
#endif

                // NEIGHBORS
                n_valid_nbr = 0;
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

                    // NUMBER OF VALID NEIGHBORS
                    n_valid_nbr += 1;

                    // NEIGHBOR ELEMENT CENTROID
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        g[0] = (g[0]*n_valid_nbr+x[0])/(n_valid_nbr+1);,
                        g[1] = (g[1]*n_valid_nbr+x[1])/(n_valid_nbr+1);,
                        g[2] = (g[2]*n_valid_nbr+x[2])/(n_valid_nbr+1);
                    )
                    
                    // NEIGHBOR ELEMENT SOLUTION
                    /*
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        auxSOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                    }
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        gU[ru] = (gU[ru]*n_valid_nbr+auxSOL[ru])/(n_valid_nbr+1);
                    }
#ifdef DG_LIMIT_P
                    IBVP.F_U2P(t, x, auxSOL, auxP);
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        gP[ru] = (gP[ru]*n_valid_nbr+auxP[ru])/(n_valid_nbr+1);
                    }
#endif
                    */
                }
                // ----------------------------------------------------

                if (n_valid_nbr >= AMREX_SPACEDIM)
                {
                    // FILL THE MATRIX S ------------------------------
                    linalg::zeros(AMREX_SPACEDIM*AMREX_SPACEDIM, S);

                    // CURRENT ELEMENT
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    Add2S(S, g, x);

                    // NEIGHBORS
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

                        // NEIGHBOR ELEMENT CENTROID
                        AMREX_D_TERM
                        (
                            x[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                            x[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                            x[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                        )
                        Add2S(S, g, x);
                    }
                    // ------------------------------------------------

                    // DIAGONALIZE ------------------------------------
#if (AMREX_SPACEDIM == 2)
                    linalg::dsy2ev(AMREX_SPACEDIM, S, LAM, EV);
#endif
#if (AMREX_SPACEDIM == 3)
                    linalg::dsy3ev(AMREX_SPACEDIM, S, LAM, EV);
#endif
                    // ------------------------------------------------

                    // EVAL THE LEFT AND RIGHT DIFFERENCES ------------
                    AMREX_D_TERM
                    (
                        y2L[0] = 0.0; y2R[0] = 0.0;,
                        y2L[1] = 0.0; y2R[1] = 0.0;,
                        y2L[2] = 0.0; y2R[2] = 0.0;
                    )
                    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dUL[ru+dim*DG_N_SOL] = 0.0;
                        dUR[ru+dim*DG_N_SOL] = 0.0;
                    }

                    // CURRENT ELEMENT CENTROID
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    ApplyR(EV, g, x, y);

                    // CURRENT ELEMENT SOLUTION
#ifdef DG_LIMIT_P
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        auxSOL[ru] = X_fab(i,j,k,ru);
                    }
                    IBVP.F_U2P(t, x, auxSOL, auxP);

                    // ADD CONTRIBUTION
                    Add2dU(DG_N_SOL, gP, y, auxP, y2L, y2R, dUL, dUR);
#else
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        auxSOL[ru] = X_fab(i,j,k,ru);
                    }
                    
                    // ADD CONTRIBUTION
                    Add2dU(DG_N_SOL, gU, y, auxSOL, y2L, y2R, dUL, dUR);
#endif

                    // NEIGHBORS
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

                        // NEIGHBOR ELEMENT CENTROID
                        AMREX_D_TERM
                        (
                            x[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                            x[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                            x[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                        )
                        ApplyR(EV, g, x, y);
                        
                        // NEIGHBOR ELEMENT SOLUTION
#ifdef DG_LIMIT_P
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            auxSOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                        }
                        IBVP.F_U2P(t, x, auxSOL, auxP);

                        // ADD CONTRIBUTION
                        Add2dU(DG_N_SOL, gP, y, auxP, y2L, y2R, dUL, dUR);
#else
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            auxSOL[ru] = X_fab(nbr_i,nbr_j,nbr_k,ru);
                        }
                        
                        // ADD CONTRIBUTION
                        Add2dU(DG_N_SOL, gU, y, auxSOL, y2L, y2R, dUL, dUR);
#endif
                    }

                    // EVAL DIFFERENCES
                    // We use un as a temporary variable
                    AMREX_D_TERM
                    (
                        un[0] = 1.0/y2L[0];,
                        un[1] = 1.0/y2L[1];,
                        un[2] = 1.0/y2L[2];
                    )
                    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dUL[ru+dim*DG_N_SOL] *= un[dim];
                    }
                    // We use un as a temporary variable
                    AMREX_D_TERM
                    (
                        un[0] = 1.0/y2R[0];,
                        un[1] = 1.0/y2R[1];,
                        un[2] = 1.0/y2R[2];
                    )
                    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dUR[ru+dim*DG_N_SOL] *= un[dim];
                    }
                    // ------------------------------------------------

                    // LIMITED SLOPES OF THE CHARACTERISTICS ----------
                    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                    {
                        AMREX_D_TERM
                        (
                            un[0] = EV[dim+0*AMREX_SPACEDIM];,
                            un[1] = EV[dim+1*AMREX_SPACEDIM];,
                            un[2] = EV[dim+2*AMREX_SPACEDIM];
                        )
#ifdef DG_LIMIT_P
                        IBVP.F_DP2DU(t, g, gU, &dUL[dim*DG_N_SOL], auxSOL);
                        IBVP.F_DU2DC(t, g, un, gU, auxSOL, &dCL[dim*DG_N_SOL]);

                        IBVP.F_DP2DU(t, g, gU, &dUR[dim*DG_N_SOL], auxSOL);
                        IBVP.F_DU2DC(t, g, un, gU, auxSOL, &dCR[dim*DG_N_SOL]);
#else
                        IBVP.F_DU2DC(t, g, un, gU, &dUL[dim*DG_N_SOL], &dCL[dim*DG_N_SOL]);
                        IBVP.F_DU2DC(t, g, un, gU, &dUR[dim*DG_N_SOL], &dCR[dim*DG_N_SOL]);
#endif

if (linalg::contains_nan(DG_N_SOL*AMREX_SPACEDIM, dCL))
{
Print() << "etype(" << i << "," << j << "," << k << "): " << etype << std::endl;
Print() << "x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
Print() << "g: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, g);
Print() << "un: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
Print() << "gU: "; IO::PrintRealArray2D(1, DG_N_SOL, gU);
Print() << "gP: "; IO::PrintRealArray2D(1, DG_N_SOL, gP);
Print() << "n_valid_nbr: " << n_valid_nbr << std::endl;
Print() << "dUL: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, 1, &dUL[dim*DG_N_SOL]);
Print() << "dUR: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, 1, &dUR[dim*DG_N_SOL]);
Print() << "dCL: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, 1, &dCL[dim*DG_N_SOL]);
Print() << "dCR: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, 1, &dCR[dim*DG_N_SOL]);
exit(-1);
}

                        LimitSlopes(N_SOL, &dCL[dim*DG_N_SOL], &dCR[dim*DG_N_SOL], &dC[dim*DG_N_SOL]);

                        IBVP.F_DC2DU(t, g, un, gU, &dC[dim*DG_N_SOL], &dU[dim*DG_N_SOL]);
                    }
                    
                    for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                    for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dX_fab(i,j,k,d1+AMREX_SPACEDIM*ru) += EV[d1+d2*AMREX_SPACEDIM]*dU[ru+d2*DG_N_SOL];
                    }
                    // ------------------------------------------------

//if (linalg::contains_nan(AMREX_SPACEDIM*AMREX_SPACEDIM, EV) ||
//    linalg::contains_nan(DG_N_SOL*AMREX_SPACEDIM, dU))
/*
if (i == 32 && j == 25)
{
AMREX_D_TERM
(
    x[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
    x[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
    x[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
)
for (int ru = 0; ru < N_SOL; ++ru)
{
    auxSOL[ru] = X_fab(i,j,k,ru);
}
IBVP.F_U2P(t, x, auxSOL, auxP);

Print() << "etype(" << i << "," << j << "," << k << "): " << etype << std::endl;
Print() << "x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
Print() << "U: "; IO::PrintRealArray2D(1, DG_N_SOL, auxSOL);
Print() << "P: "; IO::PrintRealArray2D(1, DG_N_SOL, auxP);
Print() << "g: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, g);
Print() << "gU: "; IO::PrintRealArray2D(1, DG_N_SOL, gU);
Print() << "gP: "; IO::PrintRealArray2D(1, DG_N_SOL, gP);
Print() << "n_valid_nbr: " << n_valid_nbr << std::endl;

Print() << "S: " << std::endl; IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, S);
Print() << "LAM: " << std::endl; IO::PrintRealArray2D(1, AMREX_SPACEDIM, LAM);
Print() << "EV: " << std::endl; IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, EV);

Print() << "theta: " << std::acos(EV[0])*180.0/M_PI << std::endl;

Print() << "y2L: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2L);
Print() << "y2R: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2R);

Print() << "dUL: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dUL);
Print() << "dUR: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dUR);

Print() << "dCL: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dCL);
Print() << "dCR: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dCR);
Print() << "dC: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dC);

Print() << "dU: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dU);

for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
for (int ru = 0; ru < N_SOL; ++ru)
{
    dU[ru+d1*DG_N_SOL] = dX_fab(i,j,k,d1+AMREX_SPACEDIM*ru);
}
Print() << "dU: " << std::endl; IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dU);

Print() << "HERE WE ARE - EvalLimitedSlopesByLeastSquares" << std::endl;
exit(-1);
}
*/
                }
                else
                {

Print() << "etype: " << etype << std::endl;
Print() << "g: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, g);
Print() << "gU: "; IO::PrintRealArray2D(1, DG_N_SOL, gU);
Print() << "gP: "; IO::PrintRealArray2D(1, DG_N_SOL, gP);
Print() << "n_valid_nbr: " << n_valid_nbr << std::endl;
Print() << "HERE WE ARE - EvalLimitedSlopesByLeastSquares - n_valid_nbr < 2" << std::endl;
exit(-1);
                }
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
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopesByLeastSquares\n";
        msg += "| dX contains nans.\n";
        amrex::Abort(msg);
    }
    // ================================================================
}