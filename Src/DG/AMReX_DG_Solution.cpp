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
 * \brief Copy values from src to dst.
 *
 * \param[in] src: Source MultiFab.
 * \param[inout] dst: Destination MultiFab.
 * \param[in] mask: a iMultiFab containing the information regarding which cells of dst will be
 *                  updated with the values contained in src.
 * \param[in] n_comp: number of components that will be replaced.
 * \param[in] per: periodicity of the problem.
 *
*/
void CopyWithMask(const MultiFab & src, MultiFab & dst, const iMultiFab & mask, const int n_comp, const Periodicity & per)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("CopyWithMask(const MultiFab &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // ================================================================
    for (MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<Real const> const & src_fab = src.array(mfi);
        Array4<Real> const & dst_fab = dst.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        ParallelFor(bx, n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int c) noexcept
        {
            if (CELL_IS_MASKED(mask_fab(i,j,k)))
            {
                dst_fab(i,j,k,c) = src_fab(i,j,k,c);
            }
        });
        Gpu::synchronize();
    }
    dst.FillBoundary(per);
    // ================================================================
}

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
 * \brief Multiply dG coefficients by the mass matrices of the implicit mesh.
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
 * \param[in] include_ghost_cells: if true, the operations will include the ghost cells.
 *
*/
void MultiplyByMassMatrix(const ImplicitMesh & mesh,
                          const MatrixFactory & matfactory,
                          const int N_SOL,
                          const Gpu::ManagedVector<int> & Sol2Dom,
                          MultiFab & X,
                          const bool include_ghost_cells)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("MultiplyByMassMatrix(const ImplicitMesh &, const MatrixFactory &, ....)");
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

#ifdef AMREX_DEBUG
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution.cpp - MultiplyByMassMatrix\n";
        msg += "| X contains nans (on entry).\n";
        Abort(msg);
    }
#endif

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

            // MASS MATRIX
            const long pos = eMMCh_pos_fab(i,j,k,dom);
            const Real * eMMCh_ptr = &eMMCh_mem_ptr[pos];
            
            // LOCAL VARIABLES
            Real x[__DG_SOL_MAX_SPACE_Np__];

            // MULTIPLY BY THE INVERSE OF THE MASS MATRIX
            if (elm_is_entire || elm_is_large)
            {
                // COPY FROM MEMORY
                for (int rs = 0; rs < sNp; ++rs)
                {
                    x[rs] = X_fab(i,j,k,rs+ru*sNp);
                }
                
                // USE CHOLESKY DECOMPOSITION
                // x := MMCh^T*MMCh*x
                linalg::dtrmv(MM_uplo, 'N', 'N', sNp, eMMCh_ptr, sNp, x, 1);
                linalg::dtrmv(MM_uplo, 'T', 'N', sNp, eMMCh_ptr, sNp, x, 1);

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

    // HANDLE GHOST CELLS
    if (include_ghost_cells)
    {
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

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
                const bool elm_is_ghost = ELM_IS_GHOST(etype);

                // MASS MATRIX
                const long pos = eMMCh_pos_fab(i,j,k,dom);
                const Real * eMMCh_ptr = &eMMCh_mem_ptr[pos];
                
                // LOCAL VARIABLES
                Real x[__DG_SOL_MAX_SPACE_Np__];

                // MULTIPLY BY THE INVERSE OF THE MASS MATRIX
                if ((elm_is_entire || elm_is_large) && elm_is_ghost)
                {
                    // COPY FROM MEMORY
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        x[rs] = X_fab(i,j,k,rs+ru*sNp);
                    }
                    
                    // USE CHOLESKY DECOMPOSITION
                    // x := MMCh^T*MMCh*x
                    linalg::dtrmv(MM_uplo, 'N', 'N', sNp, eMMCh_ptr, sNp, x, 1);
                    linalg::dtrmv(MM_uplo, 'T', 'N', sNp, eMMCh_ptr, sNp, x, 1);

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
    }
    // ================================================================

#ifdef AMREX_DEBUG
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution.cpp - MultiplyByMassMatrix\n";
        msg += "| X contains nans.\n";
        Abort(msg);
    }
#endif
}
/*
{
    Real tmp_MM[9] = {1.0, 0.0, 0.0,
                      0.0, 4.0, 0.5,
                      0.0, 0.5, 9.0};
    Real tmp_x[3] = {3.0, 2.0, 1.0}, tmp_y[3], tmp_y2[3];
    Real tmp_MMCh[9];

    linalg::matmul(3, 3, 1, tmp_MM, tmp_x, tmp_y);

    std::copy(tmp_MM, tmp_MM+9, tmp_MMCh);
    linalg::dpotf2('U', 3, tmp_MMCh, 3, info);
    
    std::copy(tmp_x, tmp_x+3, tmp_y2);
    linalg::dtrmv('U', 'N', 'N', 3, tmp_MMCh, 3, tmp_y2, 1);
    linalg::dtrmv('U', 'T', 'N', 3, tmp_MMCh, 3, tmp_y2, 1);

Print() << "tmp_MM: " << std::endl;
IO::PrintRealArray2D(3, 3, tmp_MM);
Print() << "tmp_x: "; IO::PrintRealArray2D(1, 3, tmp_x);
Print() << "tmp_y: "; IO::PrintRealArray2D(1, 3, tmp_y);
Print() << "tmp_MMCh: " << std::endl;
IO::PrintRealArray2D(3, 3, tmp_MMCh);
Print() << "tmp_y2: "; IO::PrintRealArray2D(1, 3, tmp_y2);
}
*/

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
 * \param[in] include_ghost_cells: if true, the operations will include the ghost cells.
 *
*/
void MultiplyByInverseMassMatrix(const ImplicitMesh & mesh,
                                 const MatrixFactory & matfactory,
                                 const int N_SOL,
                                 const Gpu::ManagedVector<int> & Sol2Dom,
                                 MultiFab & X,
                                 const bool include_ghost_cells)
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

#ifdef AMREX_DEBUG
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution.cpp - MultiplyByInverseMassMatrix\n";
        msg += "| X contains nans on entry.\n";
        amrex::Abort(msg);
    }
#endif

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

            // MASS MATRIX
            const long pos = eMMCh_pos_fab(i,j,k,dom);
            const Real * eMMCh_ptr = &eMMCh_mem_ptr[pos];
            
            // LOCAL VARIABLES
            int info;
            Real x[__DG_SOL_MAX_SPACE_Np__];

            // MULTIPLY BY THE INVERSE OF THE MASS MATRIX
            if (elm_is_entire || elm_is_large)
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

#ifdef AMREX_DEBUG
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution.cpp - MultiplyByInverseMassMatrix\n";
        msg += "| X contains nans prior to including the ghost cells.\n";
        amrex::Abort(msg);
    }
#endif

    // HANDLE GHOST CELLS
    if (include_ghost_cells)
    {
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

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
                const bool elm_is_ghost = ELM_IS_GHOST(etype);

                // MASS MATRIX
                const long pos = eMMCh_pos_fab(i,j,k,dom);
                const Real * eMMCh_ptr = &eMMCh_mem_ptr[pos];
                
                // LOCAL VARIABLES
                int info;
                Real x[__DG_SOL_MAX_SPACE_Np__];

                // MULTIPLY BY THE INVERSE OF THE MASS MATRIX
                if ((elm_is_entire || elm_is_large) && elm_is_ghost)
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
        
#ifdef AMREX_DEBUG
        if (X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution.cpp - MultiplyByInverseMassMatrix\n";
            msg += "| X contains nans after including the ghost cells.\n";
            amrex::Abort(msg);
        }
#endif
    }
    // ================================================================
}

/**
 * \brief Apply Barth-Jespersen-like limiters (typically to finite-volume schemes).
 *
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the solution state for the unknown solution fields.
 * \param[out] dX: a MultiFab object that will contain the limited slopes for the states.
*/
void ApplyBarthJespersenLimiter(const ImplicitMesh & /*mesh*/,
                                const MultiFab & /*X*/,
                                MultiFab & /*dX*/)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplyBarthJespersenLimiter(const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

Print() << "ApplyBarthJespersenLimiter" << std::endl;
exit(-1);

}


} // namespace DG
} // namespace amrex