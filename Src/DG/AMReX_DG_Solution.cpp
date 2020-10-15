//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution.cpp
 * \brief Contains implementations of functions for implicit-mesh discontinuous Galerkin methods.
*/

#include <AMReX_DG_Solution.H>

namespace amrex
{
namespace DG
{

/**
 * \brief Multiply dG coefficients by the inverse of the mass matrix of the standard element.
 *
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that must contain Cholesky
 *                      decomposition of the mass matrix.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions multiplied by the inverse of the standard element's mass matrix.
 *
*/
void MultiplyByInverseMassMatrix(const Geometry & geom,
                                 const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                                 const int N_SOL,
                                 MultiFab & X)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("MultiplyByInverseMassMatrix(const Geometry &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // STANDARD ELEMENT
    const int sNp = std_elem.Np;
    const char MM_uplo = std_elem.MM_uplo;
    Real const * MMCh_ptr = std_elem.MMCh.data();
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL VARIABLES
            int info;
            Real x[__DG_SOL_MAX_SPACE_Np__];

            // COPY FROM MEMORY
            for (int rs = 0; rs < sNp; ++rs)
            {
                x[rs] = X_fab(i,j,k,rs+ru*sNp);
            }

            // USE CHOLESKY DECOMPOSITION
            linalg::dpotrs(MM_uplo, sNp, 1, MMCh_ptr, sNp, x, sNp, info);

            // COPY BACK TO MEMORY
            for (int rs = 0; rs < sNp; ++rs)
            {
                X_fab(i,j,k,rs+ru*sNp) = x[rs];
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(geom.periodicity());
    // ================================================================
}


} // namespace DG
} // namespace amrex