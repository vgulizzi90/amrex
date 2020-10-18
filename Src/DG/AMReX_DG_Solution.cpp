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

/**
 * \brief Add the contribution of the small elements to the corresponding merging extended elements.
 *
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[inout] X: a MultiFab object whose extended elements' coefficients will be updated with the
 *                  contribution from the small elements.
 *
*/
void AddSmallElementsContribution(const ImplicitMesh & mesh,
                                  const MatrixFactory & matfactory,
                                  const int N_SOL,
                                  const Gpu::ManagedVector<int> & Sol2Dom,
                                  MultiFab & X)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("AddSmallElementsContribution(const ImplicitMesh &, const MatrixFactory &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // DG
    const int sNp = matfactory.std_elem.Np;

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    const int * Sol2Dom_ptr = Sol2Dom.data();
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = Sol2Dom_ptr[ru];
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
            const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);
            
            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;

            // EXTENDED ELEMENTS
            if (elm_is_extended)
            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            {
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                if (nbr_merged_b == nbr_b)
                {
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        X_fab(i,j,k,rs+ru*sNp) += X_fab(nbr_i,nbr_j,nbr_k,rs+ru*sNp);
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Multiply dG coefficients by the inverse of the mass matrices of the implicit mesh.
 *
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions multiplied by the inverse of the implicit-mesh elements' mass
 *                  matrices.
 *
*/
void MultiplyByInverseMassMatrix(const ImplicitMesh & mesh,
                                 const MatrixFactory & matfactory,
                                 const int N_SOL,
                                 const Gpu::ManagedVector<int> & Sol2Dom,
                                 MultiFab & X)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("MultiplyByInverseMassMatrix(const ImplicitMesh &, const MatrixFactory &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eMMCh_mem_ptr = matfactory.eMMCh_device_mem.data();
#else
    const Real * eMMCh_mem_ptr = matfactory.eMMCh_host_mem.data();
#endif

    // DG
    const char MM_uplo = matfactory.std_elem.MM_uplo;
    const int sNp = matfactory.std_elem.Np;

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    const int * Sol2Dom_ptr = Sol2Dom.data();
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<long const> const & eMMCh_pos_fab = matfactory.eMMCh_pos.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = Sol2Dom_ptr[ru];
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
            const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
            const bool elm_is_valid = (elm_is_entire || elm_is_large);

            // MASS MATRIX
            const long pos = eMMCh_pos_fab(i,j,k,dom);
            const Real * eMMCh_ptr = &eMMCh_mem_ptr[pos];
            
            // LOCAL VARIABLES
            int info;
            Real x[__DG_SOL_MAX_SPACE_Np__];

            // IF VALID, MULTIPLY BY THE INVERSE OF THE MASS MATRIX
            if (elm_is_valid)
            {
                // COPY FROM MEMORY
                for (int rs = 0; rs < sNp; ++rs)
                {
                    x[rs] = X_fab(i,j,k,rs+ru*sNp);
                }

                // USE CHOLESKY DECOMPOSITION
                linalg::dpotrs(MM_uplo, sNp, 1, eMMCh_ptr, sNp, x, sNp, info);

                // COPY BACK TO MEMORY
                for (int rs = 0; rs < sNp; ++rs)
                {
                    X_fab(i,j,k,rs+ru*sNp) = x[rs];
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}


} // namespace DG
} // namespace amrex