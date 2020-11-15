//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define DG_N_SOL (1+AMREX_SPACEDIM+1)

#define DG_SL_THETA 2.0

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
                if ((j == domain.smallEnd(1)) && (is_periodic[1] == 0))
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
                if ((j == domain.bigEnd(1)) && (is_periodic[1] == 0))
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
                if ((k == domain.smallEnd(2)) && (is_periodic[2] == 0))
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
                if ((k == domain.bigEnd(2)) && (is_periodic[2] == 0))
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

/** \brief Add contribution to the Least-square matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2S(Real * S, const Real * g, const Real * x)
{
#if (AMREX_SPACEDIM == 2)
    S[0+0*2] += (x[0]-g[0])*(x[0]-g[0]);
    S[0+1*2] += (x[0]-g[0])*(x[1]-g[1]);
    S[1+0*2]  = S[0+1*2];
    S[1+1*2] += (x[1]-g[1])*(x[1]-g[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    S[0+0*3] += (x[0]-g[0])*(x[0]-g[0]);
    S[0+1*3] += (x[0]-g[0])*(x[1]-g[1]);
    S[0+2*3] += (x[0]-g[0])*(x[2]-g[2]);
    S[1+0*3]  = S[0+1*3];
    S[2+0*3]  = S[0+2*3];
    S[1+1*3] += (x[1]-g[1])*(x[1]-g[1]);
    S[1+2*3] += (x[2]-g[2])*(x[2]-g[2]);
    S[2+1*3]  = S[1+2*3];
    S[2+2*3] += (x[2]-g[2])*(x[2]-g[2]);
#endif
}

/** \brief Multiply by rotation matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyR(const Real * R, const Real * g, const Real * x, Real * y)
{
#if (AMREX_SPACEDIM == 2)
    y[0] = R[0+0*2]*(x[0]-g[0])+R[0+1*2]*(x[1]-g[1]);
    y[1] = R[1+0*2]*(x[0]-g[0])+R[1+1*2]*(x[1]-g[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    y[0] = R[0+0*3]*(x[0]-g[0])+R[0+1*3]*(x[1]-g[1])+R[0+2*3]*(x[2]-g[2]);
    y[1] = R[1+0*3]*(x[0]-g[0])+R[1+1*3]*(x[1]-g[1])+R[1+2*3]*(x[2]-g[2]);
    y[2] = R[2+0*3]*(x[0]-g[0])+R[2+1*3]*(x[1]-g[1])+R[2+2*3]*(x[2]-g[2]);
#endif
}

/** \brief Add contribution to the left and right differences.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2dU(const int N_SOL, const Real * gU, const Real * y, const Real * U, Real * y2L, Real * y2R, Real * dUL, Real * dUR)
{
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        if (y[dim] < 0.0)
        {
            y2L[dim] += y[dim]*y[dim];

            for (int ru = 0; ru < N_SOL; ++ru)
            {
                dUL[ru+dim*N_SOL] += (U[ru]-gU[ru])*y[dim];
            }
        }
        else
        {
            y2R[dim] += y[dim]*y[dim];

            for (int ru = 0; ru < N_SOL; ++ru)
            {
                dUR[ru+dim*N_SOL] += (U[ru]-gU[ru])*y[dim];
            }
        }
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
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    
    // DOMAINS
    const int dom = 0;

    // NEIGHBORS INFO FOR LEAST-SQUARE SLOPE LIMITING
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

    // DOMAIN-TO-LEVELSETS CORRESPONDENCE
    int const * Dom2Phi_ptr = mesh.Dom2Phi.data();

    // DG
    const int sp = mesh.std_elem.p;

    // NEAREST POINT SEARCH
    const int it_max = 10;
    const Real atol = 1.0e-3;
    const Real rtol = atol*AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);

    // AUXILIARY MULTIFABS FOR LIMITING
    MultiFab alpha(X.boxarray, X.distributionMap, DG_N_SOL, X.n_grow);
    MultiFab bounds(X.boxarray, X.distributionMap, 2*DG_N_SOL, X.n_grow);

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // INITIALIZATION
    dX = 0.0;
    alpha = 0.0;
    bounds = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS VALID
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);
            const bool elm_is_large_or_extended = ELM_IS_LARGE_OR_EXTENDED(etype);

            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], nxc[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], P[DG_N_SOL];
                Real S[AMREX_SPACEDIM*AMREX_SPACEDIM], LAM[AMREX_SPACEDIM], EV[AMREX_SPACEDIM*AMREX_SPACEDIM];
                Real nbr_xc[AMREX_SPACEDIM], nbr_SOL[DG_N_SOL], nbr_P[DG_N_SOL];
                Real un[AMREX_SPACEDIM];
                Real y[AMREX_SPACEDIM], y2L[AMREX_SPACEDIM], y2R[AMREX_SPACEDIM];
                Real dUL[DG_N_SOL*AMREX_SPACEDIM], dUR[DG_N_SOL*AMREX_SPACEDIM], dU[DG_N_SOL*AMREX_SPACEDIM];
                Real dCL[DG_N_SOL*AMREX_SPACEDIM], dCR[DG_N_SOL*AMREX_SPACEDIM], dC[DG_N_SOL*AMREX_SPACEDIM];

                // LEVEL SET
                DG_PHI_space_BFX phi(sp, PHI_fab);
                phi.set_u(Dom2Phi_ptr[2*dom]);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                )
                // CURRENT ELEMENT SOLUTION
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(i,j,k,ru);
                }
                IBVP.F_U2P(t, xc, SOL, P);

                // FILL THE MATRIX S ----------------------------------
                linalg::zeros(AMREX_SPACEDIM*AMREX_SPACEDIM, S);

                // MIRRORED IMAGE OF THE ELEMENT CENTROID
                if (elm_is_large_or_extended)
                {
                    int nbr_i = i;
                    int nbr_j = j;
                    int nbr_k = k;
                    NEAREST_POINT(&prob_lo[0], &dx[0], phi, xc, nbr_i, nbr_j, nbr_k, nxc, rtol, it_max);
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 2.0*nxc[0]-xc[0];,
                        nbr_xc[1] = 2.0*nxc[1]-xc[1];,
                        nbr_xc[2] = 2.0*nxc[2]-xc[2];
                    )
                    Add2S(S, xc, nbr_xc);
                }

                // NEIGHBORS
                for (int n = 0; n < STENCIL_N_NBR; ++n)
                {
                    const int dir = n/2;
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                    // NEIGHBOR ELEMENT CENTROID
                    if (ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        int BF_i, BF_j, BF_k;

                        BF_i = nbr_i;
                        BF_j = nbr_j;
                        BF_k = nbr_k;

                        if (ELM_IS_SMALL(nbr_etype))
                        {
                            Real nbr_vf, nbr_nbr_vf;
                            
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME_FRACTION(dom));

                            for (int nn = 0; nn < STENCIL_N_NBR; ++nn)
                            {
                                if ((nn/2) != dir)
                                {
                                    int n_i = nbr_i+aux_table_i[nn];
                                    int n_j = nbr_j+aux_table_j[nn];
                                    int n_k = nbr_k+aux_table_k[nn];
                                    nbr_nbr_vf = eInfo_fab(n_i,n_j,n_k,ELM_VOLUME_FRACTION(dom));

                                    if (nbr_nbr_vf > nbr_vf)
                                    {
                                        nbr_vf = nbr_nbr_vf;
                                        BF_i = n_i;
                                        BF_j = n_j;
                                        BF_k = n_k;
                                    }
                                }
                            }

                            if ((BF_i == nbr_i) && (BF_j == nbr_j) && (BF_k == nbr_k))
                            {
                                BF_i = i;
                                BF_j = j;
                                BF_k = k;
                            }
                        }

                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                        )
                        Add2S(S, xc, nbr_xc);
                    }
                }
                // ----------------------------------------------------

                // DIAGONALIZE ----------------------------------------
#if (AMREX_SPACEDIM == 2)
                linalg::dsy2ev(S, LAM, EV);
#endif
#if (AMREX_SPACEDIM == 3)
                linalg::dsy3ev(S, LAM, EV);
#endif
                //linalg::eye(AMREX_SPACEDIM, EV);
                // ----------------------------------------------------

                // EVAL THE LEFT AND RIGHT DIFFERENCES ----------------
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

                // MIRRORED IMAGE OF THE ELEMENT CENTROID
                if (elm_is_large_or_extended)
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 2.0*nxc[0]-xc[0];,
                        nbr_xc[1] = 2.0*nxc[1]-xc[1];,
                        nbr_xc[2] = 2.0*nxc[2]-xc[2];
                    )
                    ApplyR(EV, xc, nbr_xc, y);

                    // MIRRORED POINT SOLUTION
                    phi.eval_un(nxc, un);
                    IBVP.F_PHI_BCS(t, nxc, un, SOL, nbr_SOL);

                    IBVP.F_U2P(t, nxc, nbr_SOL, nbr_P);
                    
                    // ADD CONTRIBUTION
                    Add2dU(DG_N_SOL, P, y, nbr_P, y2L, y2R, dUL, dUR);
                }

                // NEIGHBORS
                for (int n = 0; n < STENCIL_N_NBR; ++n)
                {
                    const int dir = n/2;
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                    // NEIGHBOR ELEMENT CENTROID
                    if (ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        int BF_i, BF_j, BF_k;
                        
                        BF_i = nbr_i;
                        BF_j = nbr_j;
                        BF_k = nbr_k;

                        if (ELM_IS_SMALL(nbr_etype))
                        {
                            Real nbr_vf, nbr_nbr_vf;
                            
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME_FRACTION(dom));

                            for (int nn = 0; nn < STENCIL_N_NBR; ++nn)
                            {
                                if ((nn/2) != dir)
                                {
                                    int n_i = nbr_i+aux_table_i[nn];
                                    int n_j = nbr_j+aux_table_j[nn];
                                    int n_k = nbr_k+aux_table_k[nn];
                                    nbr_nbr_vf = eInfo_fab(n_i,n_j,n_k,ELM_VOLUME_FRACTION(dom));

                                    if (nbr_nbr_vf > nbr_vf)
                                    {
                                        nbr_vf = nbr_nbr_vf;
                                        BF_i = n_i;
                                        BF_j = n_j;
                                        BF_k = n_k;
                                    }
                                }
                            }

                            if ((BF_i == nbr_i) && (BF_j == nbr_j) && (BF_k == nbr_k))
                            {
                                BF_i = i;
                                BF_j = j;
                                BF_k = k;
                            }
                        }

                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                        )

                        ApplyR(EV, xc, nbr_xc, y);

                        // NEIGHBOR ELEMENT SOLUTION
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            nbr_SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                        }

                        IBVP.F_U2P(t, nbr_xc, nbr_SOL, nbr_P);

                        // ADD CONTRIBUTION
                        Add2dU(DG_N_SOL, P, y, nbr_P, y2L, y2R, dUL, dUR);
                    }
                }
                // ----------------------------------------------------

                // EVAL DIFFERENCES -----------------------------------
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
                // ----------------------------------------------------

                // LIMITED SLOPES (VIA CHARACTERISTICS) ---------------
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                {
                    AMREX_D_TERM
                    (
                        un[0] = EV[dim+0*AMREX_SPACEDIM];,
                        un[1] = EV[dim+1*AMREX_SPACEDIM];,
                        un[2] = EV[dim+2*AMREX_SPACEDIM];
                    )

                    // We use nbr_SOL as an auxiliary variable
                    IBVP.F_DP2DU(t, xc, SOL, &dUL[dim*DG_N_SOL], nbr_SOL);
                    IBVP.F_DU2DC(t, xc, un, SOL, nbr_SOL, &dCL[dim*DG_N_SOL]);

                    IBVP.F_DP2DU(t, xc, SOL, &dUR[dim*DG_N_SOL], nbr_SOL);
                    IBVP.F_DU2DC(t, xc, un, SOL, nbr_SOL, &dCR[dim*DG_N_SOL]);

                    LimitSlopes(N_SOL, &dCL[dim*DG_N_SOL], &dCR[dim*DG_N_SOL], &dC[dim*DG_N_SOL]);

                    IBVP.F_DC2DU(t, xc, un, SOL, &dC[dim*DG_N_SOL], &dU[dim*DG_N_SOL]);
                }
                // ----------------------------------------------------

                // BACK TO THE MULTIFAB -------------------------------
                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dX_fab(i,j,k,d1+AMREX_SPACEDIM*ru) += EV[d2+d1*AMREX_SPACEDIM]*dU[ru+d2*DG_N_SOL];
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();

    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // MAKE SURE THAT OVER AND UNDERSHOOTS ARE AVOIDED ================
    /*
    // COMPUTE THE BOUNDS
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const bool elm_is_small = ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], aux_P[DG_N_SOL];

                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CENTROID OF THE CURRENT ELEMENT
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )

                // SOLUTION AT THE CURRENT ELEMENT
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                }
                IBVP.F_U2P(t, xc, SOL, aux_P);

                // INIT BOUNDS ----------------------------------------
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    b_fab(i,j,k,ru) = aux_P[ru];
                    b_fab(i,j,k,ru+DG_N_SOL) = aux_P[ru];
                }
                // ----------------------------------------------------

                // FIND THE BOUNDS ------------------------------------
                // CYCLE OVER THE BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;

                    // NEIGHBOR CELL INFO
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_small = ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // Conditions for NOT considering the boundary
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

                    // CENTROID OF THE NEIGHBOR ELEMENT
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                    )

                    // SOLUTION AT THE NEIGHBOR ELEMENT
                    for (int ru = 0; ru < DG_N_SOL; ++ru)
                    {
                        nbr_SOL[ru] = X_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ru);
                    }
                    IBVP.F_U2P(t, nbr_xc, nbr_SOL, aux_P);

                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), aux_P[ru]);
                        b_fab(i,j,k,ru+DG_N_SOL) = std::max(b_fab(i,j,k,ru+DG_N_SOL), aux_P[ru]);
                    }
                }

                // ZERO LEVEL OF THE LEVEL SET
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0];,
                            x[1] = xptr[x_pos+1];,
                            x[2] = xptr[x_pos+2];
                        )

                        // UNIT NORMAL
                        AMREX_D_TERM
                        (
                            un[0] = xptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = xptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = xptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        IBVP.F_PHI_BCS(t, x, un, SOL, nbr_SOL);
                        IBVP.F_U2P(t, x, nbr_SOL, aux_P);

                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), aux_P[ru]);
                            b_fab(i,j,k,ru+DG_N_SOL) = std::max(b_fab(i,j,k,ru+DG_N_SOL), aux_P[ru]);
                        }
                    }
                }
                // ---------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    bounds.FillBoundary(mesh.geom.periodicity());

    // HANDLE EXTENDED CELLS
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        
        Array4<Real> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS EXTENDED
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_EXTENDED(etype))
            {
                int nbr_i, nbr_j, nbr_k, nbr_b;

                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    // NEIGHBOR CELL INFO
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_small = ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), b_fab(nbr_i,nbr_j,nbr_k,ru));
                            b_fab(i,j,k,ru+DG_N_SOL) = std::max(b_fab(i,j,k,ru+DG_N_SOL), b_fab(nbr_i,nbr_j,nbr_k,ru+DG_N_SOL));
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    bounds.FillBoundary(mesh.geom.periodicity());

    // APPLY BARTH-JESPERSEN LIMITER
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

        Array4<Real> const & a_fab = alpha.array(mfi);
        Array4<Real const> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], P[DG_N_SOL];
                Real min_[DG_N_SOL], max_[DG_N_SOL];
                Real u0, u_lo, u_hi, a;

                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CENTROID OF THE CURRENT ELEMENT
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )

                // SOLUTION AT THE CURRENT ELEMENT
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    SOL[cu] = X_fab(BF_i,BF_j,BF_k,cu);
                }
                IBVP.F_U2P(t, xc, SOL, P);

                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    min_[cu] = P[cu];
                    max_[cu] = P[cu];
                }

                // CYCLE OVER THE BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    const int dir = b/2;

                    int fi, fj, fk;
                    int bou_Nq;
                    long pos;

                    // WHICH FACE
                    GRID_FACE(i, j, k, b, fi, fj, fk);

                    if (dir == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_0(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_1(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_2(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    }
#endif
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0]+cell_lo[0];,
                            x[1] = xptr[x_pos+1]+cell_lo[1];,
                            x[2] = xptr[x_pos+2]+cell_lo[2];
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
                        IBVP.F_U2P(t, x, SOL, P);
                        
                        for (int cu = 0; cu < DG_N_SOL; ++cu)
                        {
                            min_[cu] = std::min(min_[cu], P[cu]);
                            max_[cu] = std::max(max_[cu], P[cu]);
                        }
                    }
                }

                // BOUNDARY CONDITIONS: ZERO LEVEL OF THE LEVELSET
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0];,
                            x[1] = xptr[x_pos+1];,
                            x[2] = xptr[x_pos+2];
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
                        IBVP.F_U2P(t, x, SOL, P);

                        for (int cu = 0; cu < DG_N_SOL; ++cu)
                        {
                            min_[cu] = std::min(min_[cu], P[cu]);
                            max_[cu] = std::max(max_[cu], P[cu]);
                        }
                    }
                }
                
                // EVAL THE COEFFICIENTS USED TO SCALE THE SLOPES -----
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = X_fab(BF_i,BF_j,BF_k,ru);
                }
                IBVP.F_U2P(t, xc, SOL, P);
                
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    a = 1.0;

                    u0 = P[cu];
                    u_lo = b_fab(BF_i,BF_j,BF_k,cu);
                    u_hi = b_fab(BF_i,BF_j,BF_k,cu+DG_N_SOL);

                    if ((min_[cu] < u_lo) && (min_[cu] != u0))
                    {
                        a = std::min(a, ((u_lo-u0)/(min_[cu]-u0)));
                    }

                    if ((max_[cu] > u_hi) && (max_[cu] != u0))
                    {
                        a = std::min(a, ((u_hi-u0)/(max_[cu]-u0)));
                    }

                    a_fab(i,j,k,cu) = a;
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();

    }
    bounds.FillBoundary(mesh.geom.periodicity());

    // EVENTUALLY FIX THE SLOPES
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real const> const & a_fab = alpha.array(mfi);
        
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_valid = ELM_IS_VALID(etype);
            const bool elm_is_extended = ELM_IS_EXTENDED(etype);

            // LOCAL VARIABLES
            int nbr_i, nbr_j, nbr_k, nbr_b;
            Real xc[AMREX_SPACEDIM], SOL[DG_N_SOL];
            Real a[DG_N_SOL], dU[DG_N_SOL], dP[DG_N_SOL];

            if (elm_is_valid)
            {
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    a[cu] = a_fab(i,j,k,cu);
                }
            }

            // MAKE SURE EXTENDED ELEMENTS ACCOUNT FOR SMALL ELEMENTS
            if (elm_is_extended)
            {
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    // NEIGHBOR CELL INFO
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int cu = 0; cu < DG_N_SOL; ++cu)
                        {
                            a[cu] = std::min(a[cu], a_fab(nbr_i,nbr_j,nbr_k,cu));
                        }
                    }
                }
            }

            if (elm_is_valid)
            {
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
                )
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    SOL[cu] = X_fab(i,j,k,cu);
                }

                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        dU[cu] = dX_fab(i,j,k,d+AMREX_SPACEDIM*cu);
                    }

                    IBVP.F_DU2DP(t, xc, SOL, dU, dP);
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        dP[cu] *= a[cu];
                    }
                    IBVP.F_DP2DU(t, xc, SOL, dP, dU);

                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        dX_fab(i,j,k,d+AMREX_SPACEDIM*cu) = dU[cu];
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    */
    // ================================================================

    // CHECK ==========================================================
    if (dX.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopesByLeastSquares\n";
        msg += "| dX contains nans.\n";
        Abort(msg);
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
            // CHECK WHETHER ELEMENT IS NOT EMPTY
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
                
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )

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
                    
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                    )

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
                                AMREX_D_TERM
                                (
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    SOL[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )

                                nbr_SOL[cu] = X_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,cu);
                                AMREX_D_TERM
                                (
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_SOL[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
                            }
                            // NUMERICAL FLUX FOR INTRAPHASE
                            IBVP.F_NF_ICS(t, x, un, SOL, nbr_SOL, NFn);

/**/
if (/**/linalg::contains_nan(DG_N_SOL, NFn) /*||*/ /*((i == 72 && j == 0) && (b == 0 || b == 1) && (q == 0) && (ru == 0))*/)
{
const int r = ParallelDescriptor::MyProc();
Print(r) << std::endl;
Print(r) << std::endl << "etype(" << i << "," << j << "," << k << "): " << ELM_TYPE_DESCRIPTION(etype) << ", b: " << b << std::endl;
Print(r) << "    SOL("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL,     SOL);
Print(r) << "nbr_SOL("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, nbr_SOL);
Real P[DG_N_SOL];
IBVP.F_U2P(t, x, SOL, P);
Print(r) << "    P("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, P);
IBVP.F_U2P(t, x, nbr_SOL, P);
Print(r) << "nbr_P("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, P);

Real dU[DG_N_SOL*AMREX_SPACEDIM];

for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
for (int rru = 0; rru < N_SOL; ++rru)
{
    dU[rru+d1*DG_N_SOL] = dX_fab(BF_i,BF_j,BF_k,d1+AMREX_SPACEDIM*rru);
}
Print(r) << "dU: " << std::endl; IO::PrintRealArray2D(r, DG_N_SOL, AMREX_SPACEDIM, dU);


Real URP[DG_N_SOL];
IBVP.F_RIEMANN(t, x, un, SOL, nbr_SOL, URP);
Print(r) << "URP("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, URP);
IBVP.F_U2P(t, x, URP, P);
Print(r) << "PRP("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, P);

Print(r) << "NFn("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, NFn);

exit(-1);
}
/**/

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

                        // NUMERICAL FLUX FOR INTERPHASE
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
        Abort(msg);
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
    EvalLimitedSlopesByLeastSquares(t, mesh, N_SOL, X, dX, IBVP);
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

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, mesh, matfactory, N_SOL, X_old, dXdt, IBVP);

    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillBoundaryCells(t, mesh, N_SOL, X1, IBVP);

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
if (i == 35)
{
const int r = ParallelDescriptor::MyProc();


Print(r) << std::endl << "etype(" << i << "," << j << "," << k << "): " << ELM_TYPE_DESCRIPTION(etype) << std::endl;

Print(r) << "SOL("; IO::PrintReals(r, AMREX_SPACEDIM, xc); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, SOL);
Print(r) << "  P("; IO::PrintReals(r, AMREX_SPACEDIM, xc); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, P);

Real min_P[DG_N_SOL], max_P[DG_N_SOL];
for (int ru = 0; ru < N_SOL; ++ru)
{
    min_P[ru] = bP_fab(i,j,k,ru);
    max_P[ru] = bP_fab(i,j,k,ru+DG_N_SOL);
}
Print(r) << "    min_P: "; IO::PrintRealArray2D(r, 1, DG_N_SOL, min_P);
Print(r) << "    max_P: "; IO::PrintRealArray2D(r, 1, DG_N_SOL, max_P);

Print(r) << "HERE WE ARE " << std::endl;
exit(-1);
}
*/


/*
if (i == (428-1) && j == 83 && t > 8.81e-06)
{
const int r = ParallelDescriptor::MyProc();

Real dU[DG_N_SOL*AMREX_SPACEDIM];

for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
for (int rru = 0; rru < N_SOL; ++rru)
{
    dU[rru+d1*DG_N_SOL] = dX_fab(i,j,k,d1+AMREX_SPACEDIM*rru);
}
Print(r) << "dU: " << std::endl; IO::PrintRealArray2D(r, DG_N_SOL, AMREX_SPACEDIM, dU);

Real SOL[DG_N_SOL], P[DG_N_SOL];
Real xc[AMREX_SPACEDIM] = {+6.25987924e+00, +1.22499932e+00};
Real x[AMREX_SPACEDIM] = {+6.26960926e+00, +1.23044470e+00};
for (int cu = 0; cu < DG_N_SOL; ++cu)
{
    SOL[cu] = X_fab(i,j,k,cu);
    AMREX_D_TERM
    (
        SOL[cu] += dX_fab(i,j,k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
        SOL[cu] += dX_fab(i,j,k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
        SOL[cu] += dX_fab(i,j,k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
    )   
}
IBVP.F_U2P(t, x, SOL, P);
Print(r) << "SOL("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, SOL);
Print(r) << "  P("; IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; IO::PrintRealArray2D(r, 1, DG_N_SOL, P);

//exit(-1);
}
*/