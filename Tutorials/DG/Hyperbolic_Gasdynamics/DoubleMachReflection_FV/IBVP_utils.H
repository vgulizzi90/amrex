//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define DG_N_SOL (1+AMREX_SPACEDIM+1)

#define DG_LIMIT_CP

#define DG_SL_THETA 1.0

namespace amrex
{
namespace DG
{

/**
 * \brief Compute the maximum time step 
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
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
                const Geometry & geom,
                const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int dom_Ng = std_elem.dom_Ng;
    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = std_elem.dom_XI3g.data();
    )

    // DG
    const int sp = std_elem.p;
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
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};

            // LOCAL VARIABLES
            Real elm_dt, x_dt;
            Real x[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL];

            // SOLUTION
            DG_SOL_space_BFX sol(sp, cell_lo, cell_hi, X_fab, i, j, k);

            // LOOP OVER THE GRID POINTS
            elm_dt = std::numeric_limits<Real>::max();
            for (int g = 0; g < dom_Ng; ++g)
            {
                // GRID POINT
                AMREX_D_TERM
                (
                    x[0] = dom_XI1g_ptr[g]+cell_lo[0];,
                    x[1] = dom_XI2g_ptr[g]+cell_lo[1];,
                    x[2] = dom_XI3g_ptr[g]+cell_lo[2];
                )

                // EVAL SOLUTION
                sol.eval_BF_only_table(x);
                sol.eval_unsafe(0, N_SOL, SOL);

                // EVAL DT
                x_dt = IBVP.F_DT(&dx[0], t, x, SOL);
                elm_dt = std::min(elm_dt, x_dt);
            }

            dt_fab(i,j,k,0) = std::min(dt_fab(i,j,k,0), elm_dt);
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
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains unknown solution fields at the cell centers. On
 *                  exit, the cells in the ghost rows will have their values updated.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * SOL, const Real * bcs_SOL) const;
*/
template <typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const Geometry & geom,
                      const int N_SOL,
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
            Real SOL[DG_N_SOL], bcs_SOL[DG_N_SOL];

            // CURRENT CELL SOLUTION
            for (int ru = 0; ru < N_SOL; ++ru)
            {
                SOL[ru] = X_fab(i,j,k,ru);
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
                    X_fab(i,j-1,k,ru) = SOL[ru];
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
        });
        Gpu::synchronize();
    }
    X.FillBoundary(geom.periodicity());
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
 * \param[in] geom: a Geometry object that contains geometry information.
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
                       const Geometry & geom,
                       const int N_SOL,
                       const MultiFab & X,
                       MultiFab & dX,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    // ================================================================

    // INITIALIZATION
    dX = 0.0;
    // ==============
    
    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
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

            // X1 DIRECTION -------------------------------------------
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
            // --------------------------------------------------------
#if (AMREX_SPACEDIM > 1)
            // X2 DIRECTION -------------------------------------------
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
            // --------------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
            // X3 DIRECTION -------------------------------------------
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
            // --------------------------------------------------------
#endif
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(geom.periodicity());
    // ================================================================
}

/**
 * \brief Eval dXdt.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
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
void Eval_dXdt(const Real t,
               const Geometry & geom,
               const int N_SOL,
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
    MultiFab dX(ba, dm, AMREX_SPACEDIM*N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL LIMITED SLOPES ============================================
    EvalLimitedSlopes(t, geom, N_SOL, X, dX, IBVP);
    // ================================================================

    // EVAL THE TIME DERIVATIVES ======================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS

            // LOCAL VARIABLES
            Real un[AMREX_SPACEDIM];
            int nbr_i, nbr_j, nbr_k, nbr_b;
            Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], NFn[DG_N_SOL];

            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            {
                const int dir = b/2;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = 2.0*(b%2)-1.0;

                // RECONSTRUCTED SOLUTION AT CURRENT CELL
                for (int cu = 0; cu < N_SOL; ++cu)
                {
                    SOL[cu] = X_fab(i,j,k,cu);
                    SOL[cu] += un[dir]*0.5*dX_fab(i,j,k,dir+AMREX_SPACEDIM*cu)*dx[dir];
                }

                // NEIGHBOR CELL INDICES
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                // RECONSTRUCTED SOLUTION AT NEIGHBOR CELL
                for (int cu = 0; cu < N_SOL; ++cu)
                {
                    nbr_SOL[cu] = X_fab(nbr_i,nbr_j,nbr_k,cu);
                    nbr_SOL[cu] -= un[dir]*0.5*dX_fab(nbr_i,nbr_j,nbr_k,dir+AMREX_SPACEDIM*cu)*dx[dir];
                }

                // NUMERICAL FLUX
                IBVP.F_NF_ICS(t, nullptr, un, SOL, nbr_SOL, NFn);

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
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients will be updated according to
 *                  a 2nd-order RungeKutta algorithm.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * SOL, const Real * bcs_SOL) const;
 *                  void F_DU2DC(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DU,
 *                               Real * DC) const;
 *                  void F_DC2DU(const Real t, const Real * x, const Real * un,
 *                               const Real * SOL, const Real * DC, 
 *                               Real * DU) const;
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, const Real * nbr_SOL, Real * NFn) const;
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
void TakeTimeStep(const Real dt, const Real t,
                  const Geometry & geom,
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
    FillInvalidCells(t, geom, N_SOL, X_old, IBVP);

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, geom, N_SOL, X_old, dXdt, IBVP);

    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillInvalidCells(t, geom, N_SOL, X1, IBVP);

    // dXdt := dXdt(t+dt, X1)
    Eval_dXdt(t+dt, geom, N_SOL, X1, dXdt, IBVP);

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