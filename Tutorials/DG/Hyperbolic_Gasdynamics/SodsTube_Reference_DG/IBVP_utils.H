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

#define DG_SOUND_CELL 0
#define DG_TROUBLED_CELL 1

namespace amrex
{
namespace DG
{

/**
 * \brief Compute the maximum time step according to the CFL conditions. 
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
 * \param[inout] X: a MultiFab object that contains the averages of unknown solution fields. On
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
                    X_fab(i,j+1,k,ru) = SOL[ru];
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
                    X_fab(i,j,k-1,ru) = SOL[ru];
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
                    X_fab(i,j,k+1,ru) = SOL[ru];
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
 * \brief Determine which cells need to be limited.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] tags: a charMultiFab object that, on exit, will contain the information regarding
 *                   which cells need to be limited.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  bool F_TROUBLED(const Real t, const Real * x, const Real * SOL) const;
*/
template <typename IBVP_CLASS>
void IdentifyTroubledCells(const Real t,
                           const Geometry & geom,
                           const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                           const int N_SOL,
                           const MultiFab & X,
                           charMultiFab & tags,
                           const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int dom_Nq = std_elem.dom_Nq;
    AMREX_D_TERM
    (
        Real const * dom_XI1q_ptr = std_elem.dom_XI1q.data();,
        Real const * dom_XI2q_ptr = std_elem.dom_XI2q.data();,
        Real const * dom_XI3q_ptr = std_elem.dom_XI3q.data();
    )
    Real const * dom_Wq_ptr = std_elem.dom_Wq.data();

    // DG
    const int sp = std_elem.p;
    // ================================================================

    // VARIABLES ======================================================
    // ================================================================

    // INITIALIZATION =================================================
    tags = DG_SOUND_CELL;
    // ================================================================

    // IDENTIFY THE TROUBLED CELLS ====================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<char> const & tags_fab = tags.array(mfi);

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
            Real x[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL];

            // SOLUTION
            DG_SOL_space_BFX sol(sp, cell_lo, cell_hi, X_fab, i, j, k);

            // TEST #1: IS THE SOLUTION PHYSICALLY SOUND? -------------
            {
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // GRID POINT
                    AMREX_D_TERM
                    (
                        x[0] = dom_XI1q_ptr[q]+cell_lo[0];,
                        x[1] = dom_XI2q_ptr[q]+cell_lo[1];,
                        x[2] = dom_XI3q_ptr[q]+cell_lo[2];
                    )

                    // EVAL SOLUTION
                    sol.eval_BF_only_table(x);
                    sol.eval_unsafe(0, N_SOL, SOL);

                    // CHECK WHETHER SOLUTION FIELDS ARE SOUND
                    if (IBVP.F_TROUBLED(t, x, SOL))
                    {
                        tags_fab(i,j,k,0) = DG_TROUBLED_CELL;
                    }
                }
            }
            // --------------------------------------------------------

            // TEST #2: PERSSON AND PERAIRE SHOCK SENSOR --------------
            {
                // FURTHER LOCAL VARIABLES
                Real SOL_trunc[DG_N_SOL];
                Real err, norm;
                Real int_err, int_norm;

                // EVAL THE INTEGRAL
                err = norm = 0.0;

                for (int q = 0; q < dom_Nq; ++q)
                {
                    // QUADRATURE POINT AND WEIGHT
                    AMREX_D_TERM
                    (
                        x[0] = dom_XI1q_ptr[q]+cell_lo[0];,
                        x[1] = dom_XI2q_ptr[q]+cell_lo[1];,
                        x[2] = dom_XI3q_ptr[q]+cell_lo[2];
                    )

                    // EVAL SOLUTION
                    sol.eval_BF_only_table(x);
                    sol.eval_unsafe(0, N_SOL, SOL);
                    sol.eval_unsafe_trunc(sp-1, 0, N_SOL, SOL_trunc);

                    // ADD INTEGRAL CONTRIBUTIONS
                    int_err = int_norm = 0.0;
                    for (int cu = 0; cu < DG_N_SOL; ++cu)
                    {
                        int_err += (SOL[cu]-SOL_trunc[cu])*(SOL[cu]-SOL_trunc[cu]);
                        int_norm += SOL[cu]*SOL[cu];
                    }

                    err += int_err*dom_Wq_ptr[q];
                    norm += int_norm*dom_Wq_ptr[q];
                }

                err = err/norm;

                if (err > 0.0)
                {
                    const Real s = std::log10(err);
                    const Real s0 = -4.0*std::log10(sp);
                    const Real ds = -6.0;
                    
                    if (s > (s0+ds))
                    {
                        tags_fab(i,j,k,0) = DG_TROUBLED_CELL;
                    }
                }
            }
            // --------------------------------------------------------

        });
        Gpu::synchronize();
    }
    tags.FillBoundary(geom.periodicity());
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
 * \brief Modify the dG coefficients by accounting for the limited slopes.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[out] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                unknown solution fields. On exit, the coefficients of the troubled cells (i.e.
 *                those cells that need to be limited) will be modified by the slope limiter.
 * \param[in] tags: a charMultiFab object that contains information about which cells will be
 *                  limited.
 * \param[in] Xa: a MultiFab object that contains the cell averages of the solution fields. The cell
 *                averages will be used as slope limiter.
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
void LimitDGSolution(const Real t,
                     const Geometry & geom,
                     const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                     const int N_SOL,
                     MultiFab & X,
                     const charMultiFab & tags,
                     const MultiFab & Xa,
                     const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int sNp = std_elem.Np;
    Real const * SLOpe_ptr = std_elem.SLOpe.data();
    // ================================================================

    // VARIABLES ======================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<char const> const & tags_fab = tags.array(mfi);
        Array4<Real const> const & Xa_fab = Xa.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (tags_fab(i,j,k,0) == DG_TROUBLED_CELL)
            {
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL], dSOL[DG_N_SOL];
                Real dCL[DG_N_SOL], dCR[DG_N_SOL], dC[DG_N_SOL];
#ifdef DG_LIMIT_CP
                Real P[DG_N_SOL], nbr_P[DG_N_SOL], dP[DG_N_SOL];
#endif
                Real AMREX_D_DECL(dSOLdx1[DG_N_SOL], dSOLdx2[DG_N_SOL], dSOLdx3[DG_N_SOL]);
                Real Yl[1+AMREX_SPACEDIM], Xl[__DG_SOL_MAX_SPACE_Np__];

                // CURRENT CELL SPACE LOCATION
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )
                
                // CURRENT CELL SOLUTION AVERAGE
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    SOL[ru] = Xa_fab(i,j,k,ru);
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
                    nbr_SOL[ru] = Xa_fab(i-1,j,k,ru);
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
                    nbr_SOL[ru] = Xa_fab(i+1,j,k,ru);
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

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOLdx1[ru] = dSOL[ru]/dx[0];
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
                    nbr_SOL[ru] = Xa_fab(i,j-1,k,ru);
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
                    nbr_SOL[ru] = Xa_fab(i,j+1,k,ru);
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

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOLdx2[ru] = dSOL[ru]/dx[1];
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
                    nbr_SOL[ru] = Xa_fab(i,j,k-1,ru);
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
                    nbr_SOL[ru] = Xa_fab(i,j,k+1,ru);
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

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    dSOLdx3[ru] = dSOL[ru]/dx[2];
                }
                // ----------------------------------------------------
#endif

                // UPDATE THE DG COEFFICIENTS -------------------------
                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    // Average
                    Yl[0] = SOL[ru];

                    // Slopes
                    AMREX_D_TERM
                    (
                        Yl[1] = dSOLdx1[ru];,
                        Yl[2] = dSOLdx2[ru];,
                        Yl[3] = dSOLdx3[ru];
                    )

                    linalg::matmul(sNp, 1+AMREX_SPACEDIM, 1, SLOpe_ptr, Yl, Xl);

                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        X_fab(i,j,k,rs+ru*sNp) = Xl[rs];
                    }
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(geom.periodicity());
    // ================================================================
}


/**
 * \brief Apply slope limiter to dG solution.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients of the troubled cells (i.e.
 *                  those cells that need to be limited) will be modified by the slope limiter.
 * \param[out] tags: a charMultiFab object that contains information about which cells are tagged
 *                   for limiting.
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
void ApplySlopeLimiter(const Real t,
                       const Geometry & geom,
                       const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                       const int N_SOL,
                       MultiFab & X,
                       charMultiFab & tags,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    // NUMBER OF GHOST ROWS
    const IntVect ngr = X.n_grow;

    // DG
    const int Np = std_elem.Np;
    // ================================================================

    // VARIABLES ======================================================
    // AUXILIARY MULTIFABS
    // Cell averages
    MultiFab Xa(ba, dm, N_SOL, ngr);
    // ================================================================

    // INITIALIZATION
    Xa = 0.0;
    // ==============
    
    // STEP #1: COMPUTE CELLS' AVERAGE OF THE SOLUTION FIELDS =========
    // NOTE: In this case, the cell averages of the solution fields is
    //       nothing but the first of the dG coefficients for each
    //       solution field.
    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & Xa_fab = Xa.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            Xa_fab(i,j,k,ru) = X_fab(i,j,k,ru*Np);
        });
        Gpu::synchronize();
    }
    Xa.FillBoundary(geom.periodicity());
    // ================================================================

    // STEP #2: FILL INVALID CELLS USING BOUNDARY CONDITIONS ==========
    FillInvalidCells(t, geom, N_SOL, Xa, IBVP);
    // ================================================================

    // STEP #3: IDENTIFY CELLS THAT NEED TO BE LIMITED ================
    IdentifyTroubledCells(t, geom, std_elem, N_SOL, X, tags, IBVP);
    // ================================================================

    // STEP #4: MODIFY DG SOLUTION FOR THE TROUBLED CELLS =============
    LimitDGSolution(t, geom, std_elem, N_SOL, X, tags, Xa, IBVP);
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients: Domain contribution.
 *
 * The following integral is evaluated: 
 *
 * int_{Vh} V_{,i}^T F_i
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_F(const Real t, const Real * x,
 *                           const Real * SOL, Real * F1, Real * F2, Real * F3) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_DomContrib(const Real t,
                          const Geometry & geom,
                          const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                          const int N_SOL,
                          const MultiFab & X,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_DomContrib(const Real, const Geometry &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int sp = std_elem.p;
    const int sNp = std_elem.Np;
    const int dom_Nq = std_elem.dom_Nq;
    AMREX_D_TERM
    (
        Real const * dom_XI1q_ptr = std_elem.dom_XI1q.data();,
        Real const * dom_XI2q_ptr = std_elem.dom_XI2q.data();,
        Real const * dom_XI3q_ptr = std_elem.dom_XI3q.data();
    )
    Real const * dom_Wq_ptr = std_elem.dom_Wq.data();
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], w;
            Real SOL[DG_N_SOL];
            Real AMREX_D_DECL(F1[DG_N_SOL], F2[DG_N_SOL], F3[DG_N_SOL]);
            Real integrand;
            /*Real integral_test;*/

            // SOLUTION
            DG_SOL_space_BFX sol(sp, cell_lo, cell_hi, X_fab, i, j, k);

            // LOOP OVER THE QUADRATURE POINTS ------------------------
            /*integral_test = 0.0;*/
            for (int q = 0; q < dom_Nq; ++q)
            {
                // QUADRATURE POINT AND WEIGHT
                AMREX_D_TERM
                (
                    x[0] = dom_XI1q_ptr[q]+cell_lo[0];,
                    x[1] = dom_XI2q_ptr[q]+cell_lo[1];,
                    x[2] = dom_XI3q_ptr[q]+cell_lo[2];
                )
                w = dom_Wq_ptr[q];

                // SOLUTION
                sol.eval(x, 0, N_SOL, SOL);

                // FLUXES
                IBVP.F_F(t, x, SOL, AMREX_D_DECL(F1, F2, F3));

                // INTEGRAL CONTRIBUTION
                for (int rs = 0; rs < sNp; ++rs)
                {
                    AMREX_D_TERM
                    (
                        integrand  = sol.dBFdx1[rs]*F1[ru];,
                        integrand += sol.dBFdx2[rs]*F2[ru];,
                        integrand += sol.dBFdx3[rs]*F3[ru];
                    )
                    
                    dXdt_fab(i,j,k,rs+ru*sNp) += integrand*w;
                }

                // TEST
                /*integral_test += w;*/
            }

            // INTEGRATION TEST
            /*
#if (AMREX_SPACEDIM == 1)
            Print() << "integral_test: " << integral_test-dx[0] << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
            Print() << "integral_test: " << integral_test-dx[0]*dx[1] << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
            Print() << "integral_test: " << integral_test-dx[0]*dx[1]*dx[2] << std::endl;
#endif
            exit(-1);
            */
            // --------------------------------------------------------
        });
        Gpu::synchronize();
    }
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
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
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
                          const Geometry & geom,
                          const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                          const int N_SOL,
                          const MultiFab & X,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const Geometry &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // STANDARD ELEMENT
    const int sp = std_elem.p;
    const int sNp = std_elem.Np;
    const int bou_Nq = std_elem.bou_Nq;
    AMREX_D_TERM
    (
        Real const * bou_XI1q_ptr = std_elem.bou_XI1q.data();,
        Real const * bou_XI2q_ptr = std_elem.bou_XI2q.data();,
        Real const * bou_XI3q_ptr = std_elem.bou_XI3q.data();
    )
    Real const * bou_Wq_ptr = std_elem.bou_Wq.data();
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};

            // LOCAL VARIABLES
            int nbr_i, nbr_j, nbr_k, nbr_b;
            int fi, fj, fk;
            bool intraphase;
            Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
            Real SOL[DG_N_SOL], nbr_SOL[DG_N_SOL];
            Real NFn[DG_N_SOL];
            Real integrand;
            /*Real integral_test;*/

            // SOLUTION (AT CURRENT CELL)
            DG_SOL_space_BFX sol(sp, cell_lo, cell_hi, X_fab, i, j, k);
            
            // LOOP OVER THE BOUNDARIES
            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            {
                // LOCAL PARAMETERS
                const int dir = b/2;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = 2.0*(b%2)-1.0;

                // NEIGHBOR CELL INDICES AND SUPPORT
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                const Real nbr_cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+nbr_i*dx[0],
                                                                       prob_lo[1]+nbr_j*dx[1],
                                                                       prob_lo[2]+nbr_k*dx[2])};
                const Real nbr_cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(nbr_i+1)*dx[0],
                                                                       prob_lo[1]+(nbr_j+1)*dx[1],
                                                                       prob_lo[2]+(nbr_k+1)*dx[2])};

                // SOLUTION (AT NEIGHBOR CELL)
                DG_SOL_space_BFX nbr_sol(sp, nbr_cell_lo, nbr_cell_hi, X_fab, nbr_i, nbr_j, nbr_k);

                // BOUNDARY INDICES
                GRID_FACE(i, j, k, b, fi, fj, fk);

                // CHECK WHETHER WE ARE AT BOUNDARY OR INTRAPHASE
                intraphase = true;
                if (dir == 0)
                {
                    if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                        (is_periodic[dir] == 0))
                    {
                        intraphase = false;
                    }
                }
#if (AMREX_SPACEDIM > 1)
                else if (dir == 1)
                {
                    if ((fj == domain.smallEnd(dir) || fj == (domain.bigEnd(dir)+1)) &&
                        (is_periodic[dir] == 0))
                    {
                        intraphase = false;
                    }
                }
#endif
#if (AMREX_SPACEDIM > 2)
                else if (dir == 2)
                {
                    if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                        (is_periodic[dir] == 0))
                    {
                        intraphase = false;
                    }
                }
#endif
                // GRID'S INTRAPHASE
                if (intraphase)
                {
                    // LOOP OVER THE QUADRATURE POINTS ----------------
                    /*integral_test = 0.0;*/
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT AND WEIGHT
                        AMREX_D_TERM
                        (
                            x[0] = bou_XI1q_ptr[q+b*bou_Nq]+cell_lo[0];,
                            x[1] = bou_XI2q_ptr[q+b*bou_Nq]+cell_lo[1];,
                            x[2] = bou_XI3q_ptr[q+b*bou_Nq]+cell_lo[2];
                        )
                        w = bou_Wq_ptr[q+b*bou_Nq];

                        // SOLUTION
                        sol.eval(x, 0, N_SOL, SOL);
                        nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

                        // NUMBERICAL FLUX
                        IBVP.F_NF_ICS(t, x, un, SOL, nbr_SOL, NFn);

                        // INTEGRAL CONTRIBUTION
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = sol.BF[rs]*NFn[ru];
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }

                        // TEST
                        /*integral_test += w;*/
                    }

                    // INTEGRATION TEST
                    /*
#if (AMREX_SPACEDIM == 1)
                    Print() << "integral_test: " << integral_test-1.0 << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
                    Print() << "integral_test: " << integral_test-((dir == 0) ? dx[1] : dx[0]) << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
                    Print() << "integral_test: " << integral_test-((dir == 0) ? dx[1]*dx[2] : ((dir == 1) dx[0]*dx[2] ? : dx[0]*dx[1])) << std::endl;
#endif
                    exit(-1);
                    */
                    // ------------------------------------------------
                }
                // GRID'S BOUNDARY
                else
                {
                    // LOOP OVER THE QUADRATURE POINTS ----------------
                    /*integral_test = 0.0;*/
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT AND WEIGHT
                        AMREX_D_TERM
                        (
                            x[0] = bou_XI1q_ptr[q+b*bou_Nq]+cell_lo[0];,
                            x[1] = bou_XI2q_ptr[q+b*bou_Nq]+cell_lo[1];,
                            x[2] = bou_XI3q_ptr[q+b*bou_Nq]+cell_lo[2];
                        )
                        w = bou_Wq_ptr[q+b*bou_Nq];

                        // SOLUTION
                        sol.eval(x, 0, N_SOL, SOL);

                        // NUMBERICAL FLUX
                        IBVP.F_NF_BCS(t, x, un, SOL, NFn);

                        // INTEGRAL CONTRIBUTION
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = sol.BF[rs]*NFn[ru];
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }

                        // TEST
                        /*integral_test += w;*/
                    }

                    // INTEGRATION TEST
                    /*
#if (AMREX_SPACEDIM == 1)
                    Print() << "integral_test: " << integral_test-1.0 << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
                    Print() << "integral_test: " << integral_test-((dir == 0) ? dx[1] : dx[0]) << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
                    Print() << "integral_test: " << integral_test-((dir == 0) ? dx[1]*dx[2] : ((dir == 1) dx[0]*dx[2] ? : dx[0]*dx[1])) << std::endl;
#endif
                    exit(-1);
                    */
                    // ------------------------------------------------
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_F(const Real t, const Real * x,
 *                           const Real * SOL, Real * F1, Real * F2, Real * F3) const;
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, const Real * nbr_SOL, Real * NFn) const;
 *                  void F_NF_BCS(const Real t, const Real * x, const Real * un,
 *                                const Real * SOL, Real * NFn) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const Geometry & geom,
               const StandardRectangle<AMREX_SPACEDIM> & std_elem,
               const int N_SOL,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL THE TIME DERIVATIVES: DOMAIN CONTRIBUTION =================
    Eval_dXdt_DomContrib(t, geom, std_elem, N_SOL, X, dXdt, IBVP);
    // ================================================================

    // EVAL THE TIME DERIVATIVES: BOUNDARY CONTRIBUTION ===============
    Eval_dXdt_BouContrib(t, geom, std_elem, N_SOL, X, dXdt, IBVP);
    // ================================================================

    // MULTIPLY BY INVERSE MASS MATRIX ================================
    MultiplyByInverseMassMatrix(geom, std_elem, N_SOL, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using high-order RungeKutta schemes.
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients will be updated according to
 *                  the selected RungeKutta algorithm.
 * \param[out] tags: a charMultiFab object that contains information about which cells are tagged
 *                   for limiting.
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
void TakeTimeStep(const int RK_order,
                  const Real dt, const Real t,
                  const Geometry & geom,
                  const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                  const int N_SOL,
                  MultiFab & X_new,
                  charMultiFab & tags,
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
    // ================================================================

    // COPY NEW TO OLD ================================================
    MultiFab::Copy(X_old, X_new, 0, 0, n_comp, ngr);
    // ================================================================

    // RK1 ============================================================
    // NOTE: For RK1 we assume that the order of the basis functions is
    //       p = 0. Therefore, we are not applying the slope limiter.
    // ================================================================
    if (RK_order == 1)
    {
        // NEEDED MULTIFABS
        MultiFab dXdt(ba, dm, n_comp, ngr);

        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, geom, std_elem, N_SOL, X_old, dXdt, IBVP);

        // X_new := X_old+dt*dXdt
        MultiFab::Saxpy(X_new, dt, dXdt, 0, 0, n_comp, ngr);
    }
    // ================================================================
    // RK2 ============================================================
    else if (RK_order == 2)
    {
        // NEEDED MULTIFABS
        MultiFab dXdt(ba, dm, n_comp, ngr);
        MultiFab X1(ba, dm, n_comp, ngr);

        // RK2 STEP 1 -------------------------------------------------
        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, geom, std_elem, N_SOL, X_old, dXdt, IBVP);

        // X1 := X_old+dt*dXdt
        MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, X1, tags, IBVP);
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, geom, std_elem, N_SOL, X1, dXdt, IBVP);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, X_new, tags, IBVP);
        // ------------------------------------------------------------
    }
    // ================================================================
    // RK3 ============================================================
    else if (RK_order == 3)
    {
        // NEEDED MULTIFABS
        MultiFab dXdt1(ba, dm, n_comp, ngr);
        MultiFab dXdt2(ba, dm, n_comp, ngr);
        MultiFab dXdt3(ba, dm, n_comp, ngr);
        MultiFab Xs(ba, dm, n_comp, ngr);

        // RK3 STEP 1 -------------------------------------------------
        // dXdt1 := dXdt(t, X_old)
        Eval_dXdt(t, geom, std_elem, N_SOL, X_old, dXdt1, IBVP);

        // Xs := X_old+dt*dXdt1
        MultiFab::LinComb(Xs, 1.0, X_old, 0, dt, dXdt1, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, Xs, tags, IBVP);
        // ------------------------------------------------------------

        // RK3 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt, Xs)
        Eval_dXdt(t+dt, geom, std_elem, N_SOL, Xs, dXdt2, IBVP);

        // Xs := X_old+(dt/4)*dXdt1+(dt/4)*dXdt2
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.25*dt, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(Xs,                  0.25*dt, dXdt2, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, Xs, tags, IBVP);
        // ------------------------------------------------------------

        // RK3 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t+0.5*dt, geom, std_elem, N_SOL, Xs, dXdt3, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/6)*dXdt2+(2*dt/3)*dXdt3
        MultiFab::Saxpy(X_new,     dt/6.0, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new,     dt/6.0, dXdt2, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 2.0*dt/3.0, dXdt3, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, X_new, tags, IBVP);
        // ------------------------------------------------------------
    }
    // ================================================================
    // RK4 ============================================================
    else if (RK_order == 4)
    {
        // NEEDED MULTIFABS
        MultiFab dXdt1(ba, dm, n_comp, ngr);
        MultiFab dXdt2(ba, dm, n_comp, ngr);
        MultiFab dXdt3(ba, dm, n_comp, ngr);
        MultiFab dXdt4(ba, dm, n_comp, ngr);
        MultiFab Xs(ba, dm, n_comp, ngr);

        // RK4 STEP 1 -------------------------------------------------
        // dXdt1 := dXdt(t, X_old)
        Eval_dXdt(t, geom, std_elem, N_SOL, X_old, dXdt1, IBVP);

        // Xs := X_old+(dt/2)*dXdt1
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.5*dt, dXdt1, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, Xs, tags, IBVP);
        // ------------------------------------------------------------

        // RK4 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t, geom, std_elem, N_SOL, Xs, dXdt2, IBVP);

        // Xs := X_old+(dt/2)*dXdt2
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.5*dt, dXdt2, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, Xs, tags, IBVP);
        // ------------------------------------------------------------

        // RK4 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t, geom, std_elem, N_SOL, Xs, dXdt3, IBVP);

        // Xs := X_old+dt*dXdt3
        MultiFab::LinComb(Xs, 1.0, X_old, 0, dt, dXdt3, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, Xs, tags, IBVP);
        // ------------------------------------------------------------

        // RK4 STEP 4 -------------------------------------------------
        // dXdt4 := dXdt(t+dt, Xs)
        Eval_dXdt(t, geom, std_elem, N_SOL, Xs, dXdt4, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/3)*dXdt2+(dt/3)*dXdt3+(dt/6)*dXdt4
        MultiFab::Saxpy(X_new, dt/6.0, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/3.0, dXdt2, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/3.0, dXdt3, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/6.0, dXdt4, 0, 0, n_comp, ngr);

        // Apply slope limiter
        ApplySlopeLimiter(t, geom, std_elem, N_SOL, X_new, tags, IBVP);
        // ------------------------------------------------------------
    }
    // ================================================================
    // ================================================================
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - TakeTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
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