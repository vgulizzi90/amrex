//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define MCD_THETA 2.0

namespace amrex
{
namespace FV
{
namespace IdealGas
{

/**
 * \brief Extract the conserved variables state vector from a fab. 
 *
 * \param[in] X_fab: fab containing the data.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] U: pointer to memory that will contain the conserved variables state vector.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Fab2U(Array4<Real const> const & X_fab,
           const int i, const int j, const int k,
           Real * U)
{
    U[RHO] = X_fab(i,j,k,RHO);
    AMREX_D_TERM
    (
        U[RHOV1] = X_fab(i,j,k,RHOV1);,
        U[RHOV2] = X_fab(i,j,k,RHOV2);,
        U[RHOV3] = X_fab(i,j,k,RHOV3);
    )
    U[TE] = X_fab(i,j,k,TE);
}

/**
 * \brief Fill fab from a conserved variables state vector. 
 *
 * \param[in] U: pointer to memory containing the conserved variables state vector.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] X_fab: fab that will contain the updated data.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void U2Fab(const Real * U,
             const int i, const int j, const int k,
             Array4<Real> const & X_fab)
{
    X_fab(i,j,k,RHO) = U[RHO];
    AMREX_D_TERM
    (
        X_fab(i,j,k,RHOV1) = U[RHOV1];,
        X_fab(i,j,k,RHOV2) = U[RHOV2];,
        X_fab(i,j,k,RHOV3) = U[RHOV3];
    )
    X_fab(i,j,k,TE) = U[TE];
}

/**
 * \brief Extract the primary variables state vector from a fab. 
 *
 * \param[in] X_fab: fab containing the data.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] P: pointer to memory that will contain the primary variables state vector.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Fab2P(const Real gamma,
           Array4<Real const> const & X_fab,
           const int i, const int j, const int k,
           Real * P)
{
    const Real rho = X_fab(i,j,k,RHO);
    AMREX_D_TERM
    (
        const Real u1 = X_fab(i,j,k,RHOV1)/rho;,
        const Real u2 = X_fab(i,j,k,RHOV2)/rho;,
        const Real u3 = X_fab(i,j,k,RHOV3)/rho;
    )
    const Real p = (gamma-1.0)*(X_fab(i,j,k,TE)-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));

    P[RHO] = rho;
    AMREX_D_TERM
    (
        P[RHOV1] = u1;,
        P[RHOV2] = u2;,
        P[RHOV3] = u3;
    )
    P[TE] = p;
}

/**
 * \brief Compute the maximum time step 
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const Real * dx, const Real t, const Real * x, const Real * U) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const Geometry & geom,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
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
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_x[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};

            // LOCAL VARIABLES
            Real U[FV_N_SOL];

            // ELEMENT SOLUTION
            Fab2U(X_fab, i, j, k, U);

            // EVAL DT
            dt_fab(i,j,k,0) = IBVP.F_DT(&dx[0], t, cell_x, U);
        });
        Gpu::synchronize();
    }

    dt_min = dt.min(0);
    // ================================================================

    return dt_min;
}

/**
 * \brief Fill ghost rows using boundary conditions.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[inout] X: a MultiFab object that contains unknown solution fields at the cell centers. On
 *                  exit, the cells in the ghost rows will have their values updated.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * U, const Real * bcs_U) const;
*/
template <typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const Geometry & geom,
                      MultiFab & X,
                      const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();
    // ================================================================

    // FILL INVALID CELLS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            Real U[FV_N_SOL], bcs_U[FV_N_SOL];

            // CURRENT CELL SOLUTION
            Fab2U(X_fab, i, j, k, U);

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
                IBVP.F_BCS(t, x, un, U, bcs_U);

                U2Fab(bcs_U, i-1, j, k, X_fab);
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
                IBVP.F_BCS(t, x, un, U, bcs_U);

                U2Fab(bcs_U, i+1, j, k, X_fab);
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
                IBVP.F_BCS(t, x, un, U, bcs_U);

                U2Fab(bcs_U, i, j-1, k, X_fab);
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
                IBVP.F_BCS(t, x, un, U, bcs_U);
                
                U2Fab(bcs_U, i, j+1, k, X_fab);
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
                IBVP.F_BCS(t, x, un, U, bcs_U);
                
                U2Fab(bcs_U, i, j, k-1, X_fab);
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
                IBVP.F_BCS(t, x, un, U, bcs_U);
                
                U2Fab(bcs_U, i, j, k+1, X_fab);
            }
#endif
        });
        Gpu::synchronize();
    }
    X.FillBoundary(geom.periodicity());
    // ================================================================
}

/**
 * \brief Monotonized Central Difference (MCD) limiter.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void MCDLimiter(const Real * dCL, const Real * dCR, Real * dC)
{
    for (int ru = 0; ru < FV_N_SOL; ++ru)
    {
        const Real dc = 0.5*(dCL[ru]+dCR[ru]);
        const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
        const Real slope = MCD_THETA*std::min(std::abs(dCL[ru]), std::abs(dCR[ru]));
        const Real lim = (dCL[ru]*dCR[ru] >= 0.0) ? slope : 0.0;
        dC[ru] = sgn*std::min(lim, std::abs(dc));
    }
}

/**
 * \brief Eval limited slopes and store them. They will be used to reconstruct the solution.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the state vector for each cell.
 * \param[out] dXP: a MultiFab object that contains the limited slopes for PRIMARY variables.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                 void F_DP2DC(const Real t, const Real * x, const Real * un,
 *                              const Real * P, const Real * DP, Real * DC) const;
 *                 void F_DC2DP(const Real t, const Real * x, const Real * un,
 *                              const Real * P, const Real * DC, Real * DP) const;
*/
template <typename IBVP_CLASS>
void EvalLimitedSlopes(const Real t,
                       const Geometry & geom,
                       const MultiFab & X,
                       MultiFab & dXP,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> idx = geom.InvCellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    // ================================================================

    // INITIALIZATION
    dXP = 0.0;
    // ==============
    
    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            Real P[FV_N_SOL], nbr_P[FV_N_SOL], dP[FV_N_SOL];
            Real dCL[FV_N_SOL], dCR[FV_N_SOL], dC[FV_N_SOL];

            // CURRENT CELL CENTROID
            AMREX_D_TERM
            (
                x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                x[2] = prob_lo[2]+(k+0.5)*dx[2];
            )

            // CURRENT CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i, j, k, P);

            // X1 DIRECTION -------------------------------------------
            // UNIT NORMAL
            AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
            un[0] = +1.0;

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i-1, j, k, nbr_P);

            // LEFT DIFFERENCES
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = P[ru]-nbr_P[ru];
            }

if (i == 32)
{
    const int r = ParallelDescriptor::MyProc();
    Print(r) << std::endl;
    Print(r) << std::endl << "(" << i << "," << j << "," << k << "): " << std::endl;
    Print(r) << "P("; DG::IO::PrintReals(r, AMREX_SPACEDIM, x); Print(r) << "): "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, P);

    Real dP_[FV_N_SOL];
    
    for (int ru = 0; ru < FV_N_SOL; ++ru)
    {
        dP_[ru] = dP[ru]*idx[0];
    }
    Print(r) << "dPL: " << std::endl; DG::IO::PrintRealArray2D(r, FV_N_SOL, AMREX_SPACEDIM, dP_);
}

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCL);

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i+1, j, k, nbr_P);

            // RIGHT DIFFERENCE
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = nbr_P[ru]-P[ru];
            }

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCR);

            // LIMITED SLOPES (OF THE CHARACTERISTICS)
            MCDLimiter(dCL, dCR, dC);

            // BACK TO THE PRIMARY VARIABLES
            IBVP.F_DC2DP(t, x, un, P, dC, dP);

            // STORE IN THE MULTIFAB
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dXP_fab(i,j,k,0+AMREX_SPACEDIM*ru) = dP[ru]*idx[0];
            }

            // --------------------------------------------------------
#if (AMREX_SPACEDIM > 1)
            // X2 DIRECTION -------------------------------------------
            // UNIT NORMAL
            AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
            un[1] = +1.0;

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i, j-1, k, nbr_P);

            // LEFT DIFFERENCES
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = P[ru]-nbr_P[ru];
            }

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCL);

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i, j+1, k, nbr_P);

            // RIGHT DIFFERENCE
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = nbr_P[ru]-P[ru];
            }

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCR);

            // LIMITED SLOPES (OF THE CHARACTERISTICS)
            MCDLimiter(dCL, dCR, dC);

            // BACK TO THE PRIMARY VARIABLES
            IBVP.F_DC2DP(t, x, un, P, dC, dP);

            // STORE IN THE MULTIFAB
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dXP_fab(i,j,k,1+AMREX_SPACEDIM*ru) = dP[ru]*idx[1];
            }
            // --------------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
            // X3 DIRECTION -------------------------------------------
            // UNIT NORMAL
            AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
            un[2] = +1.0;

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i, j, k-1, nbr_P);

            // LEFT DIFFERENCES
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = P[ru]-nbr_P[ru];
            }

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCL);

            // NEIGHBOR CELL SOLUTION (PRIMARY VARIABLES)
            Fab2P(IBVP.gamma, X_fab, i, j, k+1, nbr_P);

            // RIGHT DIFFERENCE
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dP[ru] = nbr_P[ru]-P[ru];
            }

            // CHARACTERISTICS
            IBVP.F_DP2DC(t, x, un, P, dP, dCR);

            // LIMITED SLOPES (OF THE CHARACTERISTICS)
            MCDLimiter(dCL, dCR, dC);

            // BACK TO THE PRIMARY VARIABLES
            IBVP.F_DC2DP(t, x, un, P, dC, dP);

            // STORE IN THE MULTIFAB
            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dXP_fab(i,j,k,2+AMREX_SPACEDIM*ru) = dP[ru]*idx[2];
            }
            // --------------------------------------------------------
#endif
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (dXP.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopes\n";
        msg += "| dXP contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Eval time derivatives of the state vectors.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] X: a MultiFab object that contains the state vector for each cell.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of state vectors.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                 void F_P2U(const Real t, const Real * x, const Real * P, Real * U) const;
 *                 void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                               const Real * U, const Real * nbr_U, Real * NFn) const;
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const Geometry & geom,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> idx = geom.InvCellSizeArray();
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dXP(ba, dm, AMREX_SPACEDIM*FV_N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL LIMITED SLOPES ============================================
    EvalLimitedSlopes(t, geom, X, dXP, IBVP);
    // ================================================================

    // EVAL THE TIME DERIVATIVES ======================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dXP_fab = dXP.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, FV_N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS

            // LOCAL VARIABLES
            Real un[AMREX_SPACEDIM];
            int nbr_i, nbr_j, nbr_k, nbr_b;
            Real P[FV_N_SOL], nbr_P[FV_N_SOL];
            Real U[FV_N_SOL], nbr_U[FV_N_SOL], NFn[FV_N_SOL];

            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            {
                const int dir = b/2;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = 2.0*(b%2)-1.0;

                // RECONSTRUCTED SOLUTION AT CURRENT CELL
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);
                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    P[cu] += un[dir]*0.5*dXP_fab(i,j,k,dir+AMREX_SPACEDIM*cu)*dx[dir];
                }
                IBVP.F_P2U(t, nullptr, P, U);

                // NEIGHBOR CELL INDICES
                DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                // RECONSTRUCTED SOLUTION AT NEIGHBOR CELL
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);
                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    nbr_P[cu] -= un[dir]*0.5*dXP_fab(nbr_i,nbr_j,nbr_k,dir+AMREX_SPACEDIM*cu)*dx[dir];
                }
                IBVP.F_P2U(t, nullptr, nbr_P, nbr_U);

/*
if (i == 33)
{
const int r = ParallelDescriptor::MyProc();
Print(r) << "b " << b << "     P: "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, P);
Print(r) << "b " << b << " nbr_P: "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, nbr_P);
Print(r) << "b " << b << "     U: "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, U);
Print(r) << "b " << b << " nbr_U: "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, nbr_U);
Print(r) << "b " << b << " NFn: "; DG::IO::PrintRealArray2D(r, 1, FV_N_SOL, NFn);
}
*/
                // NUMERICAL FLUX
                IBVP.F_NF_ICS(t, nullptr, un, U, nbr_U, NFn);

                dXdt_fab(i,j,k,ru) -= NFn[ru]*idx[dir];
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(geom.periodicity());
    // ================================================================
}

/**
 * \brief Advance in time using 2nd-order RungeKutta scheme
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[inout] X: a MultiFab object that contains the state vector for each cell. On exit, the
 *                  state vector will be updated according to a 2nd-order RungeKutta algorithm.
 * \param[in] IBVP: the Initial Boundary Value Problem class.
*/
template <typename IBVP_CLASS>
void TakeTimeStep(const Real dt, const Real t,
                  const Geometry & geom,
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
    //       number of unknown solution fields.
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
    FillInvalidCells(t, geom, X_old, IBVP);

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, geom, X_old, dXdt, IBVP);
    
    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillInvalidCells(t, geom, X1, IBVP);
    
    // dXdt := dXdt(t+dt, X1)
    Eval_dXdt(t+dt, geom, X1, dXdt, IBVP);

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

} // namespace IdealGas
} // namespace FV
} // namespace amrex