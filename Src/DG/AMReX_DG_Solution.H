//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution.H
 * \brief Contains functions for implicit-mesh discontinuous Galerkin methods.
*/

#ifndef AMREX_DG_SOLUTION_H_
#define AMREX_DG_SOLUTION_H_

namespace amrex
{
namespace DG
{

/**
 * \brief Project initial conditions over a standard structured grid.
 *
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains quadrature
 *                      abscissae and weights and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions for the projected initial conditions. Must be a valid MultiFab.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_SOL0(const Real * x, Real * SOL) const;
 *
 *                  where x is a pointer to memory containing the space location and SOL is a
 *                  pointer to memory that will be filled with the value of the unknown solution
 *                  fields at x.
 *
*/
template <typename IBVP_CLASS>
void ProjectInitialConditionsOverGrid(const Geometry & geom,
                                      const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                                      const int N_SOL,
                                      MultiFab & X,
                                      const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectInitialConditionsOverGrid(....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int sNp = std_elem.Np;
    const int dom_Nq = std_elem.dom_Nq;
    AMREX_D_TERM
    (
        Real const * dom_XI1q_ptr = std_elem.dom_XI1q.data();,
        Real const * dom_XI2q_ptr = std_elem.dom_XI2q.data();,
        Real const * dom_XI3q_ptr = std_elem.dom_XI3q.data();
    )
    Real const * dom_Wq_ptr = std_elem.dom_Wq.data();
    Real const * dom_BF_ptr = std_elem.dom_BF.data();
    Real const * iMM_ptr = std_elem.iMM.data();
    // ================================================================

    // INITIALIZATION =================================================
    X = 0.0;
    // ================================================================

    // EVAL THE PROJECTION ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM];
            Real integrand;
            Real SOL[__DG_MAX_N_SOL__];

            // PROJECTION
            for (int q = 0; q < dom_Nq; ++q)
            {
                // Quadrature point
                AMREX_D_TERM
                (
                    x[0] = dom_XI1q_ptr[q]+cell_lo[0];,
                    x[1] = dom_XI2q_ptr[q]+cell_lo[1];,
                    x[2] = dom_XI3q_ptr[q]+cell_lo[2];
                )

                IBVP.F_SOL0(x, SOL);

                // Eval the integral
                for (int ru = 0; ru < N_SOL; ++ru)
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = 0.0;
                    for (int cs = 0; cs < sNp; ++cs)
                    {
                        integrand += iMM_ptr[rs+cs*sNp]*dom_BF_ptr[q+cs*dom_Nq];
                    }
                    integrand *= SOL[ru];

                    // Sum integral contribution
                    X_fab(i,j,k,rs+ru*sNp) += integrand*dom_Wq_ptr[q];
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(geom.periodicity());
    // ================================================================
}

} // namespace DG
} // namespace amrex

#endif