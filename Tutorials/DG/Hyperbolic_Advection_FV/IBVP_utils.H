//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define MCD_THETA 2.0
#define USE_LEAST_SQUARES
//#define APPLY_LIMITER
//#define APPLY_BJ_LIMITER

namespace amrex
{
namespace FV
{

/**
 * \brief Set the initial conditions over the implicitly-defined mesh.
 *
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions for the projected initial conditions. Must be a valid MultiFab.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_SOL0(const Real * x, Real * SOL) const;
 *
*/
template <typename IBVP_CLASS>
void SetInitialConditions(const DG::ImplicitMesh & mesh,
                          const DG::MatrixFactory & matfactory,
                          MultiFab & X,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectInitialConditions(const ImplicitMesh &, const MatrixFactory &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // DOMAINS
    const int dom = 0;
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: IBVP_utils.H - SetInitialConditions\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // INITIALIZATION =================================================
    X = 0.0;
    // ================================================================

    // EVAL THE PROJECTION ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_VALID(etype))
            {
                // LOCAL VARIABLES
                Real SOL[DG_N_SOL];
                Real xc[AMREX_SPACEDIM];

                // ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )

                IBVP.F_SOL0(xc, SOL);

                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    X_fab(i,j,k,ru) = SOL[ru];
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] X: a MultiFab object that contains the solution at the elements' centroids.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const DG::ImplicitMesh & mesh,
                const DG::MatrixFactory & /*matfactory*/,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Compute_dt(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // NUMBER OF DOMAINS
    const int N_DOM = 1;

    // MAX REAL
    const Real max_real = std::numeric_limits<Real>::max();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: IBVP_utils.H - Compute_dt\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dt(X.boxarray, X.distributionMap, N_DOM, 0);
    dt = max_real;

    Real dt_min;
    // ================================================================

    // EVAL THE MINIMUM TIME STEP =====================================
    for (MFIter mfi(dt); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        //Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG::ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,DG::ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];
                
                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL];
                Real elm_dt, x_dt;

                // SUPPORT OF THE SOLUTION
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // LOOP OVER THE QUADRATURE POINTS
                elm_dt = max_real;
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // QUADRATURE POINT
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[x_pos+0]+cell_lo[0];,
                        x[1] = xptr[x_pos+1]+cell_lo[1];,
                        x[2] = xptr[x_pos+2]+cell_lo[2];
                    )

                    // EVAL SOLUTION
                    SOL[RHO] = X_fab(BF_i,BF_j,BF_k,RHO);

                    x_dt = IBVP.F_DT(dom, &dx[0], t, x, SOL);
                    elm_dt = std::min(elm_dt, x_dt);
                }

                // STORE TIME STEP
                dt_fab(i,j,k,dom) = elm_dt;
            }
        });
        Gpu::synchronize();
    }

    dt_min = dt.min(0);
    for (int dom = 1; dom < N_DOM; ++dom)
    {
        dt_min = std::min(dt_min, dt.min(dom));
    }
    // ================================================================

    return dt_min;
}

/**
 * \brief Extract the primary variables state vector from a fab. 
 *
 * \param[in] X_fab: fab containing the data.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] P: pointer to memory that will contain the primary variables state vector.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Fab2P(Array4<Real const> const & X_fab,
           const int i, const int j, const int k,
           Real * P)
{
    P[RHO] = X_fab(i,j,k,RHO);;
}

/**
 * \brief Fill in the fab using the information from the primary variables.
 *
 * \param[in] P: pointer to memory containing the primary variables state vector.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] X_fab: fab that will be updated with the data.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void P2Fab(const Real * P,
           const int i, const int j, const int k,
           Array4<Real> const & X_fab)
{
    X_fab(i,j,k,RHO) = P[RHO];
}

/**
 * \brief Fill boundary cells with meaningful data.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[inout] X: a MultiFab object that contains the AVERAGE of the unknown solution fields. On
 *                  exit, the boundary (where possible) cells will have meaningful values. 
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_BCS(const int dom, const Real t, const Real * x, const Real * un,
 *                             const Real * SOL, Real * bcs_SOL) const;
 *
*/
template <typename IBVP_CLASS>
void FillBoundaryCells(const Real t,
                       const DG::ImplicitMesh & mesh,
                       const DG::MatrixFactory & /*matfactory*/,
                       MultiFab & X,
                       const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("FillBoundaryCells(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillBoundaryCells\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // FILL BOUNDARY CELLS ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            int BF_i, BF_j, BF_k;
            Real P[DG_N_SOL], bcs_P[DG_N_SOL];

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // CURRENT CELL SOLUTION
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                Fab2P(X_fab, BF_i, BF_j, BF_k, P);

                // UPDATE GRID BOUNDARY VALUES
                if ((i == domain.smallEnd(0)) && (is_periodic[0] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i-1,j,k,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i-1,j,k,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i-1,j,k,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] = -1.0;,
                        un[1] =  0.0;,
                        un[2] =  0.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);

                    P2Fab(bcs_P, i-1, j, k, X_fab);
                }
                if ((i == domain.bigEnd(0)) && (is_periodic[0] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i+1,j,k,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i+1,j,k,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i+1,j,k,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] = +1.0;,
                        un[1] =  0.0;,
                        un[2] =  0.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);
                    
                    P2Fab(bcs_P, i+1, j, k, X_fab);
                }
#if (AMREX_SPACEDIM > 1)
                if ((j == domain.smallEnd(1)) && (is_periodic[1] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j-1,k,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j-1,k,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j-1,k,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] = -1.0;,
                        un[2] =  0.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);

                    P2Fab(bcs_P, i, j-1, k, X_fab);
                }
                if ((j == domain.bigEnd(1)) && (is_periodic[1] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j+1,k,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j+1,k,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j+1,k,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] = +1.0;,
                        un[2] =  0.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);

                    P2Fab(bcs_P, i, j+1, k, X_fab);
                }
#endif
#if (AMREX_SPACEDIM > 2)
                if ((k == domain.smallEnd(2)) && (is_periodic[2] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j,k-1,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j,k-1,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j,k-1,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] =  0.0;,
                        un[2] = -1.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);

                    P2Fab(bcs_P, i, j, k-1, X_fab);
                }
                if ((k == domain.bigEnd(2)) && (is_periodic[2] == 0))
                {
                    AMREX_D_TERM
                    (
                        x[0] = eInfo_fab(i,j,k+1,DG::ELM_CENTROID_X1(dom));,
                        x[1] = eInfo_fab(i,j,k+1,DG::ELM_CENTROID_X2(dom));,
                        x[2] = eInfo_fab(i,j,k+1,DG::ELM_CENTROID_X3(dom));
                    )
                    AMREX_D_TERM
                    (
                        un[0] =  0.0;,
                        un[1] =  0.0;,
                        un[2] = +1.0;
                    )
                    IBVP.F_BCS(dom, t, x, un, P, bcs_P);
                    
                    P2Fab(bcs_P, i, j, k+1, X_fab);
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
        Abort(msg);
    }
    // ================================================================
}


/** \brief Multiply by rotation matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyR(const Real * R, const Real * xc, const Real * nbr_xc, Real * y)
{
#if (AMREX_SPACEDIM == 2)
    y[0] = R[0+0*2]*(nbr_xc[0]-xc[0])+R[0+1*2]*(nbr_xc[1]-xc[1]);
    y[1] = R[1+0*2]*(nbr_xc[0]-xc[0])+R[1+1*2]*(nbr_xc[1]-xc[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    y[0] = R[0+0*3]*(nbr_xc[0]-xc[0])+R[0+1*3]*(nbr_xc[1]-xc[1])+R[0+2*3]*(nbr_xc[2]-xc[2]);
    y[1] = R[1+0*3]*(nbr_xc[0]-xc[0])+R[1+1*3]*(nbr_xc[1]-xc[1])+R[1+2*3]*(nbr_xc[2]-xc[2]);
    y[2] = R[2+0*3]*(nbr_xc[0]-xc[0])+R[2+1*3]*(nbr_xc[1]-xc[1])+R[2+2*3]*(nbr_xc[2]-xc[2]);
#endif
}

/** \brief Add contribution for the least-square reconstruction.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2dU(const Real * U, const Real * y, const Real * nbr_U, Real * y2, Real * dU)
{
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        y2[d] += y[d]*y[d];

        for (int ru = 0; ru < DG_N_SOL; ++ru)
        {
            dU[ru+d*DG_N_SOL] += (nbr_U[ru]-U[ru])*y[d];
        }
    }
}

/**
 * \brief Limit slopes.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void LimitSlopes(const Real * slopes, const Real * tmp_slopes, Real * lim_slopes)
{
    for (int ru = 0; ru < DG_N_SOL; ++ru)
    {
        const Real ls_slp = slopes[ru];
        const Real sgn = (ls_slp > 0.0) ? +1.0 : -1.0;
        const Real tmp_slp = tmp_slopes[ru];

        if (tmp_slp*ls_slp > 0.0)
        {
            lim_slopes[ru] = sgn*std::min(std::abs(ls_slp), std::min(std::abs(MCD_THETA*tmp_slp), std::abs(lim_slopes[ru])));
        }
        else
        {
            lim_slopes[ru] = 0.0;
        }
    }
}

/**
 * \brief Eval limited slopes and store them. They'll be used to reconstruct the PRIMARY variables.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXP: a MultiFab object that will contain the limited slopes for the PRIMARY variables.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
*/
template <typename IBVP_CLASS>
void EvalSlopesLS_P(const Real t,
                    const DG::ImplicitMesh & mesh,
                    const MultiFab & X,
                    MultiFab & dXP,
                    const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalSlopesLS_P(const Real, const DG::ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
#ifdef AMREX_USE_CUDA
    const Real * eFVInfo_mem_ptr = mesh.eFVInfo_device_mem.data();
#else
    const Real * eFVInfo_mem_ptr = mesh.eFVInfo_host_mem.data();
#endif

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    // ================================================================

    // INITIALIZATION
    dXP = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<long const> const & eFVInfo_pos_fab = mesh.eFVInfo_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            const bool elm_is_valid = DG::ELM_IS_VALID(etype);

            if (elm_is_valid)
            {
                // LOCAL PARAMETERS
                const long pos = eFVInfo_pos_fab(i,j,k,DG::ELM_FV_INFO_POS(dom));
                const Real * xm = &eFVInfo_mem_ptr[pos];
                const Real * R = &eFVInfo_mem_ptr[pos+AMREX_SPACEDIM];

                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM], y[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_BF_i, nbr_BF_j, nbr_BF_k;
                short nbr_etype;
                Real y2[AMREX_SPACEDIM];
                Real dP[DG_N_SOL*AMREX_SPACEDIM], dC[DG_N_SOL*AMREX_SPACEDIM];
#ifdef APPLY_LIMITER
                int dir;
                Real tmp_dP[DG_N_SOL];
                Real lim_slopes[DG_N_SOL*AMREX_SPACEDIM];
#endif

                // INITIALIZATION -------------------------------------
                AMREX_D_TERM
                (
                    y2[0] = 0.0;,
                    y2[1] = 0.0;,
                    y2[2] = 0.0;
                )
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru+d*DG_N_SOL] = 0.0;
                }

                // ELEMENT INFO
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                Fab2P(X_fab, i, j, k, P);
                // ----------------------------------------------------

                // EVAL LEAST-SQUARE RECONSTRUCTION -------------------
                // LOOP OVER THE ELEMENT IN THE STENCIL
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    nbr_i = i+DG::base_stencil_table_i[n];
                    nbr_j = j+DG::base_stencil_table_j[n];
                    nbr_k = k+DG::base_stencil_table_k[n];
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

                        // ADD LS CONTRIBUTION
                        ApplyR(R, xc, nbr_xc, y);
                        Add2dU(P, y, nbr_P, y2, dP);
                    }
                }

                // ADD CONTRIBUTION FROM MIRRORED POINT
                if (DG::ELM_IS_LARGE_OR_EXTENDED(etype))
                {
                    // UNIT NORMAL
                    // We use nbr_xc[0] as a temporary variable
                    AMREX_D_TERM
                    (
                        un[0] = (xm[0]-xc[0]);,
                        un[1] = (xm[1]-xc[1]);,
                        un[2] = (xm[2]-xc[2]);
                    )
                    nbr_xc[0] = 1.0/std::sqrt(AMREX_D_TERM(un[0]*un[0],+un[1]*un[1],+un[2]*un[2]));
                    AMREX_D_TERM
                    (
                        un[0] *= nbr_xc[0];,
                        un[1] *= nbr_xc[0];,
                        un[2] *= nbr_xc[0];
                    )

                    // MIRRORED ELEMENT INFO
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = xm[0];,
                        nbr_xc[1] = xm[1];,
                        nbr_xc[2] = xm[2];
                    )
                    IBVP.F_PHI_BCS(dom, t, nbr_xc, un, P, nbr_P);
                    
                    // ADD LS CONTRIBUTION
                    ApplyR(R, xc, nbr_xc, y);
                    Add2dU(P, y, nbr_P, y2, dP);
                }

                // LEAST-SQUARE SOLUTION
                // We use dC as a temporary variable
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    y2[d] = 1.0/y2[d];

                    for (int ru = 0; ru < DG_N_SOL; ++ru)
                    {
                        dC[ru+d*DG_N_SOL] = dP[ru+d*DG_N_SOL]*y2[d];
                        dP[ru+d*DG_N_SOL] = 0.0;
                    }
                }

                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru+d1*DG_N_SOL] += R[d2+d1*AMREX_SPACEDIM]*dC[ru+d2*DG_N_SOL];
                }
                // ----------------------------------------------------

#ifdef APPLY_LIMITER
                // INITIALIZATION
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    lim_slopes[ru+d*DG_N_SOL] = dP[ru+d*DG_N_SOL];
                }

                // X1 DIRECTION =======================================
                dir = 0;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                for (int ii = -1; ii < 2; ii += 2)
                {
                    nbr_i = i+ii;
                    nbr_j = j;
                    nbr_k = k;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_VALID(nbr_etype))
                    {
                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                        // RECONSTRUCTED DIFFERENCE
                        for (int ru = 0; ru < DG_N_SOL; ++ru)
                        {
                            tmp_dP[ru] = -P[ru];
                            AMREX_D_TERM
                            (
                                ,
                                tmp_dP[ru] -= dP[ru+1*DG_N_SOL]*(nbr_xc[1]-xc[1]);,
                                tmp_dP[ru] -= dP[ru+2*DG_N_SOL]*(nbr_xc[2]-xc[2]);
                            )
                            tmp_dP[ru] += nbr_P[ru];
                            tmp_dP[ru] /= (nbr_xc[0]-xc[0]);
                        }
                        
                        // LIMIT SLOPES
                        LimitSlopes(&dP[dir*DG_N_SOL], tmp_dP, &lim_slopes[dir*DG_N_SOL]);
                    }
                }
                // ====================================================
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION =======================================
                dir = 1;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                for (int jj = -1; jj < 2; jj += 2)
                {
                    nbr_i = i;
                    nbr_j = j+jj;
                    nbr_k = k;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_VALID(nbr_etype))
                    {
                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                        // RECONSTRUCTED DIFFERENCE
                        for (int ru = 0; ru < DG_N_SOL; ++ru)
                        {
                            tmp_dP[ru] = -P[ru];
                            AMREX_D_TERM
                            (
                                tmp_dP[ru] -= dP[ru+0*DG_N_SOL]*(nbr_xc[0]-xc[0]);,
                                ,
                                tmp_dP[ru] -= dP[ru+2*DG_N_SOL]*(nbr_xc[2]-xc[2]);
                            )
                            tmp_dP[ru] += nbr_P[ru];
                            tmp_dP[ru] /= (nbr_xc[1]-xc[1]);
                        }

                        // LIMIT SLOPES
                        LimitSlopes(&dP[dir*DG_N_SOL], tmp_dP, &lim_slopes[dir*DG_N_SOL]);
                    }
                }
                // ====================================================
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION =======================================
                dir = 2;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                for (int kk = -1; kk < 2; kk += 2)
                {
                    nbr_i = i;
                    nbr_j = j;
                    nbr_k = k+kk;
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_VALID(nbr_etype))
                    {
                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                        // RECONSTRUCTED DIFFERENCE
                        for (int ru = 0; ru < DG_N_SOL; ++ru)
                        {
                            tmp_dP[ru] = -P[ru];
                            AMREX_D_TERM
                            (
                                tmp_dP[ru] -= dP[ru+0*DG_N_SOL]*(nbr_xc[0]-xc[0]);,
                                tmp_dP[ru] -= dP[ru+1*DG_N_SOL]*(nbr_xc[1]-xc[1]);,
                                
                            )
                            tmp_dP[ru] += nbr_P[ru];
                            tmp_dP[ru] /= (nbr_xc[2]-xc[2]);
                        }
                        
                        // LIMIT SLOPES
                        LimitSlopes(&dP[dir*DG_N_SOL], tmp_dP, &lim_slopes[dir*DG_N_SOL]);
                    }
                }
                // ====================================================
#endif
#endif

                // STORE IN THE MULTIFAB
#ifdef APPLY_LIMITER
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    AMREX_D_TERM
                    (
                        dXP_fab(i,j,k,0+AMREX_SPACEDIM*ru) = lim_slopes[ru+0*DG_N_SOL];,
                        dXP_fab(i,j,k,1+AMREX_SPACEDIM*ru) = lim_slopes[ru+1*DG_N_SOL];,
                        dXP_fab(i,j,k,2+AMREX_SPACEDIM*ru) = lim_slopes[ru+2*DG_N_SOL];
                    )
                }
#else
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    AMREX_D_TERM
                    (
                        dXP_fab(i,j,k,0+AMREX_SPACEDIM*ru) = dP[ru+0*DG_N_SOL];,
                        dXP_fab(i,j,k,1+AMREX_SPACEDIM*ru) = dP[ru+1*DG_N_SOL];,
                        dXP_fab(i,j,k,2+AMREX_SPACEDIM*ru) = dP[ru+2*DG_N_SOL];
                    )
                }
#endif
            }
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(mesh.geom.periodicity());
    
    // CHECK ==========================================================
    if (dXP.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalSlopesLS_P\n";
        msg += "| dXP contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Apply Barth-Jespersen-like limiters.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXP: a MultiFab object that will contain the limited slopes for the PRIMARY variables.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
*/
template <typename IBVP_CLASS>
void ApplyBarthJespersenLimiter_P(const Real /*t*/,
                                  const DG::ImplicitMesh & mesh,
                                  const MultiFab & X,
                                  MultiFab & dXP,
                                  const IBVP_CLASS & /*IBVP*/)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplyBarthJespersenLimiter_P(const Real, const DG::ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
#endif

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    MultiFab alpha(X.boxarray, X.distributionMap, DG_N_SOL, X.n_grow);
    // ================================================================

    // INITIALIZATION
    alpha = 1.0;
    // ==============

    // EVAL THE SOLUTION BOUNDS =======================================
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

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dXP_fab = dXP.array(mfi);

        Array4<Real> const & alpha_fab = alpha.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const bool elm_is_small = DG::ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL], Pr[DG_N_SOL], nbr_Pr[DG_N_SOL];
                Real min_P[DG_N_SOL], max_P[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_b;
                short nbr_etype;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real x[AMREX_SPACEDIM];
                Real alphas[DG_N_SOL];

                // SUPPORT OF THE SOLUTION (CURRENT ELEMENT)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                // CURRENT ELEMENT SOLUTION
                Fab2P(X_fab, BF_i, BF_j, BF_k, P);

                // MIN/MAX INITIALIZATION
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    min_P[cu] = P[cu];
                    max_P[cu] = P[cu];
                }

                // LOOP OVER THE BOUNDARIES ---------------------------
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    bool bou_is_not_valid, intraphase;
                    int bou_Nq;
                    const Real * xptr;

                    // NEIGHBOR CELL INFO
                    bou_is_not_valid = false;
                    {
                        DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                        const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        // Conditions for NOT considering the boundary:
                        // i) the current cell extends to the
                        //    neighboring cell;
                        // ii) the current cell is small and the
                        //     neighboring cell extends into it;
                        // iii) the neighboring cell is empty (it means
                        //      that the cell must be separated by the
                        //      zero level set of a function; this case
                        //      is handled by the INTERPHASE case.
                        //
                        const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                        const bool nbr_is_empty = DG::ELM_IS_EMPTY(nbr_etype);
                        if (cells_are_merged || nbr_is_empty) bou_is_not_valid = true;
                    }

                    // FACE QUADRATURE INFO
                    intraphase = true;
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        long pos;
                        DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

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
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#endif
                        xptr = &eBouQuad_mem_ptr[pos];
                    }

                    if (bou_is_not_valid) continue;

                    if (intraphase)
                    {
                        // SUPPORT OF THE SOLUTION (NEIGHBOR ELEMENT)
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // NEIGHBOR ELEMENT CENTROID
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )

                        // NEIGHBOR ELEMENT SOLUTION
                        Fab2P(X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

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

                            // EVAL SOLUTION
                            Pr[RHO] = P[RHO];
                            nbr_Pr[RHO] = nbr_P[RHO];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )

                                AMREX_D_TERM
                                (
                                    nbr_Pr[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_Pr[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_Pr[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
                            }

                            // UPDATE MIN/MAX
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                min_P[cu] = std::min(min_P[cu], nbr_P[cu]);
                                max_P[cu] = std::max(max_P[cu], nbr_P[cu]);
                            }
                        }
                    }
                }
                // ----------------------------------------------------

                // LOOP OVER THE BOUNDARIES (AGAIN) -------------------
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    alphas[cu] = 1.0;
                }

                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    bool bou_is_not_valid, intraphase;
                    int bou_Nq;
                    const Real * xptr;

                    // NEIGHBOR CELL INFO
                    bou_is_not_valid = false;
                    {
                        DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                        const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        // Conditions for NOT considering the boundary:
                        // i) the current cell extends to the
                        //    neighboring cell;
                        // ii) the current cell is small and the
                        //     neighboring cell extends into it;
                        // iii) the neighboring cell is empty (it means
                        //      that the cell must be separated by the
                        //      zero level set of a function; this case
                        //      is handled by the INTERPHASE case.
                        //
                        const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                        const bool nbr_is_empty = DG::ELM_IS_EMPTY(nbr_etype);
                        if (cells_are_merged || nbr_is_empty) bou_is_not_valid = true;
                    }

                    // FACE QUADRATURE INFO
                    intraphase = true;
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        long pos;
                        DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

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
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#endif
                        xptr = &eBouQuad_mem_ptr[pos];
                    }

                    if (bou_is_not_valid) continue;

                    if (intraphase)
                    {
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

                            // EVAL SOLUTION
                            Pr[RHO] = P[RHO];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

/*
if (i == __i__ && j == __j__)
{
Print() << "Pr("; DG::IO::PrintReals(AMREX_SPACEDIM, x); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, Pr);
}
*/

                            // UPDATE SLOPE COEFFICIENT
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                if (Pr[cu] > max_P[cu])
                                {
                                    alphas[cu] = std::min(alphas[cu], (max_P[cu]-P[cu])/(Pr[cu]-P[cu]));
                                }
                                else if (Pr[cu] < min_P[cu])
                                {
                                    alphas[cu] = std::min(alphas[cu], (min_P[cu]-P[cu])/(Pr[cu]-P[cu]));
                                }
                            }
                        }
                    }
                }
                // ----------------------------------------------------
                
                // STORE THE VALUES -----------------------------------
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    alpha_fab(i,j,k,cu) = alphas[cu];
                }
                // ----------------------------------------------------


/*
if (i == __i__ && j == __j__)
{
Print() << "min_P("; DG::IO::PrintReals(AMREX_SPACEDIM, xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, min_P);
Print() << "max_P("; DG::IO::PrintReals(AMREX_SPACEDIM, xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, max_P);
Print() << "alphas: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, alphas);
//exit(-1);
}
*/

            }
        });
        Gpu::synchronize();
    }
    alpha.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // HANDLE EXTENDED ELEMENTS =======================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & alpha_fab = alpha.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_EXTENDED(etype))
            {
                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b;
                Real alphas[DG_N_SOL];

                // INITIALIZATION
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    alphas[cu] = alpha_fab(i,j,k,cu);
                }

                // LOOP OVER THE BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int cu = 0; cu < DG_N_SOL; ++cu)
                        {
                            alphas[cu] = std::min(alphas[cu], alpha_fab(nbr_i,nbr_j,nbr_k,cu));
                        }
                    }
                }

                // STORE THE VALUES
                for (int cu = 0; cu < DG_N_SOL; ++cu)
                {
                    alpha_fab(i,j,k,cu) = alphas[cu];
                }
/*
if (i == __i__ && j == __j__)
{
Print() << "alphas: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, alphas);
exit(-1);
}
*/
            }
        });
        Gpu::synchronize();
    }
    alpha.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // LIMIT THE SLOPES ===============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real const> const & alpha_fab = alpha.array(mfi);

        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_VALID(etype))
            {
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    AMREX_D_TERM
                    (
                        dXP_fab(i,j,k,0+AMREX_SPACEDIM*ru) *= alpha_fab(i,j,k,ru);,
                        dXP_fab(i,j,k,1+AMREX_SPACEDIM*ru) *= alpha_fab(i,j,k,ru);,
                        dXP_fab(i,j,k,2+AMREX_SPACEDIM*ru) *= alpha_fab(i,j,k,ru);
                    )
                }
            }
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}


/**
 * \brief Eval the time derivative of the dG coefficients: Boundary contribution.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V_{,i}^T NFn
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] dXP: a MultiFab object that contains the limited slopes.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const DG::ImplicitMesh & mesh,
                          const DG::MatrixFactory & /*matfactory*/,
                          const MultiFab & X,
                          const MultiFab & dX,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const DG::ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
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

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
#ifdef AMREX_DEBUG
            const Real vf = eInfo_fab(i,j,k,DG::ELM_VOLUME_FRACTION(dom));
#endif

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
#ifdef AMREX_DEBUG
                const Real cell_v = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
#endif
                const bool elm_is_small = DG::ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL], Pr[DG_N_SOL], nbr_Pr[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_b;
                short nbr_etype;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real NFn[DG_N_SOL];
                Real integrand;
#ifdef AMREX_DEBUG
                Real elm_v;
#endif

                // SUPPORT OF THE SOLUTION (CURRENT ELEMENT)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                // CURRENT ELEMENT SOLUTION
                Fab2P(X_fab, BF_i, BF_j, BF_k, P);

//Print() << "(" << i << "," << j << "," << k << ") -> (" << BF_i << "," << BF_j << "," << BF_k << ")" << std::endl;

                // LOOP OVER THE BOUNDARIES
#ifdef AMREX_DEBUG
                elm_v = 0.0;
#endif
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    bool bou_is_not_valid, intraphase;
                    int bou_Nq;
                    const Real * xptr;

                    // NEIGHBOR CELL INFO
                    bou_is_not_valid = false;
                    {
                        DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                        const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        // Conditions for NOT computing the flux
                        // contribution:
                        // i) the current cell extends to the
                        //    neighboring cell;
                        // ii) the current cell is small and the
                        //     neighboring cell extends into it;
                        // iii) the neighboring cell is empty (it means
                        //      that the cell must be separated by the
                        //      zero level set of a function; this case
                        //      is handled by the INTERPHASE case.
                        //
                        const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                        const bool nbr_is_empty = DG::ELM_IS_EMPTY(nbr_etype);
                        if (cells_are_merged || nbr_is_empty) bou_is_not_valid = true;
                    }

                    // FACE QUADRATURE INFO
                    intraphase = true;
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        long pos;
                        DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

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
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                            if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#endif
                        xptr = &eBouQuad_mem_ptr[pos];

                        // UNIT NORMAL
                        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                        un[dir] = 2.0*(b%2)-1.0;
                    }

#ifdef AMREX_DEBUG
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

                        // INTEGRATION TEST
                        elm_v += x[0]*un[0]*w;
                    }
#endif
                    if (bou_is_not_valid) continue;

                    if (intraphase)
                    {
                        // SUPPORT OF THE SOLUTION (NEIGHBOR ELEMENT)
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

//Print() << " - nbr(" << nbr_i << "," << nbr_j << "," << nbr_k << ") -> (" << nbr_BF_i << "," << nbr_BF_j << "," << nbr_BF_k << ")" << std::endl;

                        // NEIGHBOR ELEMENT CENTROID
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )

                        // NEIGHBOR ELEMENT SOLUTION
                        Fab2P(X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

                        // COMPUTE THE INTEGRAL
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

                            // EVAL SOLUTION
                            Pr[RHO] = P[RHO];
                            nbr_Pr[RHO] = nbr_P[RHO];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )

                                AMREX_D_TERM
                                (
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_ICS(dom, t, x, un, Pr, nbr_Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
                            }
                        }
                    }
                    else
                    {
                        // COMPUTE THE INTEGRAL
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

                            // EVAL SOLUTION
                            Pr[RHO] = P[RHO];
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_BCS(dom, t, x, un, Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
                            }
                        }
                    }
                }

                // ZERO-LEVEL OF THE LEVEL SET (IF ANY)
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

#ifdef AMREX_DEBUG
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

                        // INTEGRATION TEST
                        elm_v += x[0]*un[0]*w;
                    }
#endif
                    {
                        // COMPUTE THE INTEGRAL
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

                            // EVAL SOLUTION
                            Pr[RHO] = P[RHO];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_PHI_BCS(dom, t, x, un, Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
                            }
                        }
                    }
                }

                // INTEGRATION TEST
#ifdef AMREX_DEBUG
                if (std::abs(elm_v/cell_v-vf)*100.0 > 1.0e-2)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
                    msg += "| Error in the boundary quadrature.\n";
                    msg += "| The number of quadrature points might not be enough.\n";
                    msg += "| Cell: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                    msg += "| elm_v          : "+std::to_string(elm_v)+".\n";
                    msg += "| vf             : "+std::to_string(vf)+".\n";
                    msg += "| vf*cell_v      : "+std::to_string(vf*cell_v)+".\n";
                    msg += "| elm_v-vf*cell_v: "+std::to_string(elm_v-vf*cell_v)+".\n";
                    msg += "| err (%)        : "+std::to_string(std::abs(elm_v/cell_v-vf)*100.0)+".\n";
                    Abort(msg);
                }
#endif

/*
if (i == __i__ && j == __j__)
{
Print() << "dUdt: " << std::endl;
for (int ru = 0; ru < DG_N_SOL; ++ru)
{
    Print() << dXdt_fab(i,j,k,ru) << std::endl;
}
Print() << std::endl;
exit(-1);
}
*/

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
 * \brief Eval the time derivative of the solution.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] X: a MultiFab object that contains the solution at the elements' centroids.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the solution states.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const DG::ImplicitMesh & mesh,
               const DG::MatrixFactory & matfactory,
               MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    const Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, 0);
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dX(ba, dm, AMREX_SPACEDIM*DG_N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

#ifdef USE_LEAST_SQUARES
    // FILL THE INVALID CELLS =========================================
    FillBoundaryCells(t, mesh, matfactory, X, IBVP);
    // ================================================================

    // COMPUTE THE SLOPES =============================================
    EvalSlopesLS_P(t, mesh, X, dX, IBVP);
    // ================================================================
#endif

#ifdef APPLY_BJ_LIMITER
    // BARTH-JESPERSEN LIMITER ========================================
    ApplyBarthJespersenLimiter_P(t, mesh, X, dX, IBVP);
    // ================================================================
#endif

    // EVAL THE TIME DERIVATIVES: DOMAIN CONTRIBUTION =================
    // ================================================================

    // EVAL THE TIME DERIVATIVES: BOUNDARY CONTRIBUTION ===============
    Eval_dXdt_BouContrib(t, mesh, matfactory, X, dX, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, DG_N_SOL, Sol2Dom, dXdt);
    MultiplyByInverseMassMatrix(mesh, matfactory, DG_N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using 2nd-order Runge-Kutta schemes.
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[inout] X_new: a MultiFab object that contains the solution at the elements' centroids; On
 *                      exit, the solution will be updated according to the Runge-Kutta algorithm.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const Real dt, const Real t,
                            const DG::ImplicitMesh & mesh,
                            const DG::MatrixFactory & matfactory,
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

    // VARIABLES ======================================================
    MultiFab X_old(ba, dm, n_comp, ngr);

    // AUXILIARY NEEDED MULTIFABS
    MultiFab dXdt(ba, dm, n_comp, ngr);
    MultiFab X1(ba, dm, n_comp, ngr);
    // ================================================================

    // COPY NEW TO OLD ================================================
    MultiFab::Copy(X_old, X_new, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 ============================================================
    {
        // RK2 STEP 1 -------------------------------------------------
        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, mesh, matfactory, X_old, dXdt, IBVP);

        // X1 := X_old+dt*dXdt
        MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, mesh, matfactory, X1, dXdt, IBVP);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);
        // ------------------------------------------------------------
    }
    // ================================================================

    // CHECK SOLUTION =================================================
    if (X_new.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - TakeRungeKuttaTimeStep\n";
        msg += "| Solution contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Export solution contained in a MultiFab and defined over an implicit mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] tags: a charMultiFab object that contains a two-value data for each cell.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const std::string & filename_root,
                const Real t,
                const DG::ImplicitMesh & mesh,
                const DG::MatrixFactory & matfactory,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_dom_Ng = std::max(mesh.ls_std_elem.dom_Ng, mesh.im_std_elem.dom_Ng);
    const int cell_e = std::max(mesh.ls_std_elem.g, mesh.im_std_elem.g);
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = (mesh.ls_std_elem.g > mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI1g.data() : mesh.im_std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = (mesh.ls_std_elem.g > mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI2g.data() : mesh.im_std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = (mesh.ls_std_elem.g > mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI3g.data() : mesh.im_std_elem.dom_XI3g.data();
    )

    // DG
    const int phi_sp = mesh.ls_std_elem.p;

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(VTK_LINE_ID, VTK_QUAD_ID, VTK_HEXAHEDRON_ID);
    const int VTK_cell_conn_len = AMREX_D_PICK(VTK_LINE_CONN_LEN, VTK_QUAD_CONN_LEN, VTK_HEXAHEDRON_CONN_LEN);

    // LEVEL SET FUNCTIONS/SOLUTION FIELDS
    const int N_PHI = 1;
    // ================================================================

    // VARIABLES ======================================================
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    VTK::Cell_conn_t n_VTK_nodes;
    VTK::Cell_offs_t n_VTK_cells;

    Vector<VTK::Float_t> VTK_nodes;
    Vector<VTK::Cell_conn_t> VTK_cell_conn;
    Vector<VTK::Cell_offs_t> VTK_cell_offset;
    Vector<VTK::Cell_type_t> VTK_cell_type;
    Vector<Vector<VTK::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<VTK::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_cell_conn_pos, VTK_cell_offset_pos, VTK_cell_type_pos;
    long VTK_nodal_field_pos;
    long VTK_cell_field_pos;

    long VTK_cell_conn_offset;

    std::string filepath, folderpath;

    int n_IBVP_fields;
    Vector<int> nodal_fields_domains;
    // ================================================================

    // SLOPES =========================================================
    MultiFab dX(X.boxarray, X.distributionMap, AMREX_SPACEDIM*DG_N_SOL, 1);

    {
        MultiFab Xr(X.boxarray, X.distributionMap, DG_N_SOL, 1);

        amrex::MultiFab::Copy(Xr, X, 0, 0, DG_N_SOL, 1);
#ifdef USE_LEAST_SQUARES
        FillBoundaryCells(t, mesh, matfactory, Xr, IBVP);
        EvalSlopesLS_P(t, mesh, Xr, dX, IBVP);
#endif
#ifdef APPLY_BJ_LIMITER
        ApplyBarthJespersenLimiter_P(t, mesh, Xr, dX, IBVP);
#endif
    }
    // ================================================================
    
    // GET INFO FROM THE IBVP CLASS ===================================
    // NODAL DATA
    IBVP.F_FIELDS_NAMES(VTK_nodal_fields_names);
    n_IBVP_fields = VTK_nodal_fields_names.size();

    for (int ph = 0; ph < N_PHI; ++ph)
    {
        VTK_nodal_fields_names.push_back("PHI_"+std::to_string(ph));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // DOMAINS OF DEFINITION FOR THE NODAL FIELDS
    IBVP.F_FIELDS_DOMAINS(nodal_fields_domains);

    // CELL DATA
    VTK_cell_fields_names = {"TAG"};

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            VTK_cell_conn_size += cell_dom_Ne*VTK_cell_conn_len;

            n_VTK_nodes += cell_dom_Ng;
            n_VTK_cells += cell_dom_Ne;
        }
    }
    // ================================================================

    // ALLOCATE MEMORY ================================================
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ================================================================
    
    // EVAL OUTPUT FIELDS =============================================
    VTK::Float_t * VTK_nodes_ptr = VTK_nodes.data();
    
    VTK::Cell_conn_t * VTK_cell_conn_ptr = VTK_cell_conn.data();
    
    VTK::Cell_offs_t * VTK_cell_offset_ptr = VTK_cell_offset.data();
    VTK_cell_offset_ptr[0] = 0;
    
    VTK::Cell_type_t * VTK_cell_type_ptr = VTK_cell_type.data();

    Vector<VTK::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    const int * nodal_fields_domains_ptr = nodal_fields_domains.data();

    Vector<VTK::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;
    VTK_cell_conn_pos = 0L;
    VTK_cell_offset_pos = 1L;
    VTK_cell_type_pos = 0L;

    VTK_nodal_field_pos = 0L;
    VTK_cell_field_pos = 0L;

    VTK_cell_conn_offset = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], xc[AMREX_SPACEDIM];
            Real PHI[__DG_MAX_N_PHI__];
            Real P[DG_N_SOL], Pr[DG_N_SOL], SOL[DG_N_SOL];
            Real FIELDS[__DG_MAX_N_FIELDS__];

            // LEVELSETS
            DG::DG_PHI_space_BFX phi(phi_sp, cell_lo, cell_hi, PHI_fab, i, j, k);

            // LOOP OVER THE GRID POINTS
            for (int g = 0; g < cell_dom_Ng; ++g)
            {
                // GRID POINT
                AMREX_D_TERM
                (
                    x[0] = dom_XI1g_ptr[g]+cell_lo[0];,
                    x[1] = dom_XI2g_ptr[g]+cell_lo[1];,
                    x[2] = dom_XI3g_ptr[g]+cell_lo[2];
                )

                // VTK NODES ------------------------------------------
#if (AMREX_SPACEDIM == 1)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif
                // ----------------------------------------------------

                // EVAL LEVELSETS -------------------------------------
                phi.eval(x, 0, N_PHI, PHI);
                // ----------------------------------------------------

                // VTK NODAL FIELDS -----------------------------------
                for (int f = 0; f < n_IBVP_fields; ++f)
                {
                    const int dom = nodal_fields_domains_ptr[f];
                    const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

                    int BF_i, BF_j, BF_k;
                    
                    DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                    // CURRENT ELEMENT CENTROID
                    AMREX_D_TERM
                    (
                        xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                        xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                        xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                    )

                    // EVAL SOLUTION
                    Fab2P(X_fab, BF_i, BF_j, BF_k, P);
                    Pr[RHO] = P[RHO];
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        AMREX_D_TERM
                        (
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                        )
                    }

                    // EVAL IBVP FIELDS
                    SOL[RHO] = Pr[RHO];
                    IBVP.F_FIELDS(t, x, SOL, FIELDS);

                    VTK_nodal_fields_ptr[f][VTK_nodal_field_pos] = FIELDS[f];
                }

                for (int ph = 0; ph < N_PHI; ++ph)
                {
                    VTK_nodal_fields_ptr[n_IBVP_fields+ph][VTK_nodal_field_pos] = PHI[ph];
                }
                // ----------------------------------------------------

                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;
            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY ------------------------------
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            VTK_cell_conn_offset += cell_dom_Ng;
        }
    }
    // ================================================================

    // DUMP HEADER FILE ===============================================
    DG::IO::MakeStepFolder(folderpath_root, n, n_steps);

    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = DG::IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

        VTK::PrintUnstructuredGridData_VTU(filepath,
                                           n_VTK_nodes,
                                           n_VTK_cells,
                                           VTK_nodes,
                                           VTK_cell_conn,
                                           VTK_cell_offset,
                                           VTK_cell_type,
                                           VTK_nodal_fields,
                                           VTK_nodal_fields_names,
                                           VTK_cell_fields,
                                           VTK_cell_fields_names);
    }
    // ================================================================
}

/**
 * \brief Eval maximum difference between two solutions.
*/
template <typename IBVP_CLASS>
Real EvalMaxDiff(const DG::ImplicitMesh & mesh,
                 const DG::MatrixFactory & /*matfactory*/,
                 const MultiFab & X1,
                 const MultiFab & X2,
                 const IBVP_CLASS & /*IBVP*/)
{
    // PARAMETERS =====================================================
    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    Real max_diff = 0.0;
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X1); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real const> const & X1_fab = X1.array(mfi);
        Array4<Real const> const & X2_fab = X2.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            
            if (DG::ELM_IS_VALID(etype))
            {
                max_diff = std::max(std::abs(X1_fab(i,j,k,RHO)-X2_fab(i,j,k,RHO)), max_diff);
            }
        }
    }
    ParallelDescriptor::ReduceRealMax(max_diff);
    // ================================================================

    return max_diff;
}

/**
 * \brief Eval domain and boundary error.
*/
template <typename IBVP_CLASS>
void EvalError(const Real t,
               const DG::ImplicitMesh & mesh,
               const DG::MatrixFactory & matfactory,
               const MultiFab & X,
               const IBVP_CLASS & IBVP,
               Real & dom_err, Real & bou_err)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    MultiFab err(X.boxarray, X.distributionMap, 2, 0);
    // ================================================================

    // SLOPES =========================================================
    MultiFab dX(X.boxarray, X.distributionMap, AMREX_SPACEDIM*DG_N_SOL, 1);

    {
        MultiFab Xr(X.boxarray, X.distributionMap, DG_N_SOL, 1);

        amrex::MultiFab::Copy(Xr, X, 0, 0, DG_N_SOL, 1);
        FillBoundaryCells(t, mesh, matfactory, Xr, IBVP);
        EvalSlopesLS_P(t, mesh, Xr, dX, IBVP);
    }
    // ================================================================

    // EVAL THE INTEGRALS: DOMAIN ERROR ===============================
    err = 0.0;

    for (MFIter mfi(err); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);
        Array4<Real> const & err_fab = err.array(mfi);
        
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG::ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,DG::ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM], w;
                int BF_i, BF_j, BF_k;
                Real P[DG_N_SOL], Pr[DG_N_SOL], Ue[DG_N_SOL];
                Real err_x, norm_x;

                // SUPPORT OF THE ELEMENT
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                for (int q = 0; q < dom_Nq; ++q)
                {
                    // QUADRATURE POINT
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[x_pos+0]+cell_lo[0];,
                        x[1] = xptr[x_pos+1]+cell_lo[1];,
                        x[2] = xptr[x_pos+2]+cell_lo[2];
                    )
                    w = xptr[x_pos+AMREX_SPACEDIM];

                    // EVAL SOLUTION
                    Fab2P(X_fab, BF_i, BF_j, BF_k, P);
                    Pr[RHO] = P[RHO];
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        AMREX_D_TERM
                        (
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                        )
                    }

                    // EVAL EXACT SOLUTION
                    IBVP.F_EXACT(t, x, Ue);

                    // EVAL ERROR
                    err_x = std::abs(Pr[RHO]-Ue[RHO]);
                    norm_x = 1.0;

                    err_fab(i,j,k,2*dom) += norm_x*w;
                    err_fab(i,j,k,2*dom+1) += err_x*w;
                }
            }
        });
        Gpu::synchronize();
    }

//Print() << "dom_err.sum(1): " << err.sum(1) << std::endl;
//Print() << "dom_err.sum(0): " << err.sum(0) << std::endl;

    dom_err = err.sum(1)/err.sum(0);
    // ================================================================

    // EVAL THE INTEGRALS: BOUNDARY ERROR =============================
    err = 0.0;

    for (MFIter mfi(err); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);
        Array4<Real> const & err_fab = err.array(mfi);
        
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_SMALL_OR_LARGE(etype))
            {
                // LOCAL PARAMETERS
                const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
                const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM], w;
                int BF_i, BF_j, BF_k;
                Real P[DG_N_SOL], Pr[DG_N_SOL], Ue[DG_N_SOL];
                Real err_x, norm_x;

                // SUPPORT OF THE ELEMENT
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

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
                    w = xptr[x_pos+AMREX_SPACEDIM];

                    // EVAL SOLUTION
                    Fab2P(X_fab, BF_i, BF_j, BF_k, P);
                    Pr[RHO] = P[RHO];
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        AMREX_D_TERM
                        (
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                            Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                        )
                    }

                    // EVAL EXACT SOLUTION
                    IBVP.F_EXACT(t, x, Ue);

                    // EVAL ERROR
                    err_x = std::abs(Pr[RHO]-Ue[RHO]);
                    norm_x = 1.0;

                    err_fab(i,j,k,2*dom) += norm_x*w;
                    err_fab(i,j,k,2*dom+1) += err_x*w;
                }
            }
        });
        Gpu::synchronize();
    }

//Print() << "bou_err.sum(1): " << err.sum(1) << std::endl;
//Print() << "bou_err.sum(0): " << err.sum(0) << std::endl;

    bou_err = err.sum(1)/err.sum(0);
    // ================================================================
}

} // namespace FV
} // namespace amrex