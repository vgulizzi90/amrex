//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution.H
 * \brief Contains functions for implicit-mesh discontinuous Galerkin methods.
*/

#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <AMReX_DG_Constants.H>
#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

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
    BL_PROFILE("ProjectInitialConditionsOverGrid(const Geometry &, ....)");
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

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution.H - ProjectInitialConditionsOverGrid\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
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

/**
 * \brief Project N_PHI level sets over a standard structured grid.
 *
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains quadrature
 *                      abscissae and weights and basis functions data.
 * \param[in] N_PHI: Number of level sets.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions for the projected level sets. Must be a valid MultiFab.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_PHI(const Real t, const Real * x, Real * PHI) const;
 *
 *                  where t is time, x is a pointer to memory containing the space location and PHI
 *                  is a pointer to memory that will be filled with the value of the level sets at
 *                  (t,x).
 *
*/
template <typename IBVP_CLASS>
void ProjectLevelsetsOverGrid(const Real t,
                              const Geometry & geom,
                              const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                              const int N_PHI,
                              MultiFab & X,
                              const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectLevelsetsOverGrid(const Real t, const Geometry &, ....)");
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
        const Box & bx = mfi.fabbox();
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
            Real PHI[__DG_MAX_N_PHI__];

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

                IBVP.F_PHI(t, x, PHI);

                // Eval the integral
                for (int ru = 0; ru < N_PHI; ++ru)
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = 0.0;
                    for (int cs = 0; cs < sNp; ++cs)
                    {
                        integrand += iMM_ptr[rs+cs*sNp]*dom_BF_ptr[q+cs*dom_Nq];
                    }
                    integrand *= PHI[ru];

                    // Sum integral contribution
                    X_fab(i,j,k,rs+ru*sNp) += integrand*dom_Wq_ptr[q];
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================
}

void MultiplyByInverseMassMatrix(const Geometry & geom,
                                 const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                                 const int N_SOL,
                                 MultiFab & X);

void AddSmallElementsContribution(const ImplicitMesh & mesh,
                                  const MatrixFactory & matfactory,
                                  const int N_SOL,
                                  const Gpu::ManagedVector<int> & Sol2Dom,
                                  MultiFab & X);
void MultiplyByInverseMassMatrix(const ImplicitMesh & mesh,
                                 const MatrixFactory & matfactory,
                                 const int N_SOL,
                                 const Gpu::ManagedVector<int> & Sol2Dom,
                                 MultiFab & X);

/**
 * \brief Project initial conditions over the implicitly-defined mesh.
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
 *                  functions for the projected initial conditions. Must be a valid MultiFab.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_SOL0(const Real * x, Real * SOL) const;
 *
*/
template <typename IBVP_CLASS>
void ProjectInitialConditions(const ImplicitMesh & mesh,
                              const MatrixFactory & matfactory,
                              const int N_SOL,
                              const Gpu::ManagedVector<int> & Sol2Dom,
                              MultiFab & X,
                              const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectInitialConditions(const ImplicitMesh &, const MatrixFactory &, ....)");
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

    // DG
    const int sp = matfactory.std_elem.p;
    const int sNp = matfactory.std_elem.Np;

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    const int * Sol2Dom_ptr = Sol2Dom.data();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution.H - ProjectInitialConditions\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
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
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const int dom = Sol2Dom_ptr[ru];
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            // QUADRATURE INFO
            const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
            const long eDomQuad_pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
            const Real * xptr = &eDomQuad_mem_ptr[eDomQuad_pos];

            // LOCAL VARIABLES
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
            Real x[AMREX_SPACEDIM], w;
            Real integrand;
            Real SOL[__DG_MAX_N_SOL__];

            // SUPPORT OF THE BASIS FUNCTIONS
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

            // BASIS FUNCTIONS
            DG_SOL_space_BF bf(sp, BF_lo, BF_hi);

            // PROJECTION
            for (int q = 0; q < dom_Nq; ++q)
            {
                const long x_pos = (AMREX_SPACEDIM+1)*q;
                AMREX_D_TERM
                (
                    x[0] = xptr[x_pos+0]+cell_lo[0];,
                    x[1] = xptr[x_pos+1]+cell_lo[1];,
                    x[2] = xptr[x_pos+2]+cell_lo[2];
                )
                w = xptr[x_pos+AMREX_SPACEDIM];

                IBVP.F_SOL0(x, SOL);

                bf.eval_BF_only_table(x);
                
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = bf.BF[rs]*SOL[ru];
                    X_fab(i,j,k,rs+ru*sNp) += integrand*w;
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, N_SOL, Sol2Dom, X);
    MultiplyByInverseMassMatrix(mesh, matfactory, N_SOL, Sol2Dom, X);
    // ================================================================
}

/**
 * \brief Project initial conditions over the implicitly-defined mesh.
 *
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that, on exit, will contain the coefficients of the basis
 *                  functions for the projected initial conditions. Must be a valid MultiFab.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  void F_SOL0(const Real * x, Real * SOL) const;
 *
*/
template <typename IBVP_CLASS>
void ProjectInitialConditions(const ImplicitMesh & mesh,
                              const MatrixFactory & matfactory,
                              const int N_SOL,
                              MultiFab & X,
                              const IBVP_CLASS & IBVP)
{
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);
    ProjectInitialConditions(mesh, matfactory, N_SOL, Sol2Dom, X, IBVP);
}

/**
 * \brief Compute the error between the exact solution and the dG solution.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename IBVP_CLASS>
Real EvalError(const Real t,
               const ImplicitMesh & mesh,
               const MatrixFactory & matfactory,
               const int N_DOM,
               const int N_SOL,
               const MultiFab & X,
               const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalError(const Real, const ImplicitMesh &, ....)");
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

    // DG
    const int sp = matfactory.std_elem.p;
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution.H - EvalError\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // VARIABLES ======================================================
    MultiFab err(X.boxarray, X.distributionMap, 2*N_DOM, 0);
    err = 0.0;

    Real total_error = 0.0;
    // ================================================================

    // EVAL THE MINIMUM TIME STEP =====================================
    for (MFIter mfi(err); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & err_fab = err.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];
                
                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w;
                Real SOL[__DG_MAX_N_SOL__];
                Real err_x, norm_x;

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                // LOOP OVER THE QUADRATURE POINTS
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
                    sol.eval_BF_only_table(x);
                    sol.eval_unsafe(0, N_SOL, SOL);

                    // EVAL ERROR
                    IBVP.F_ERROR(dom, t, x, SOL, err_x, norm_x);

                    err_fab(i,j,k,2*dom) += norm_x*w;
                    err_fab(i,j,k,2*dom+1) += err_x*w;
                }
            }
        });
        Gpu::synchronize();
    }

Print() << "err.sum(1): " << err.sum(1) << std::endl;
Print() << "err.sum(0): " << err.sum(0) << std::endl;

    total_error = err.sum(1)/err.sum(0);
    for (int dom = 1; dom < N_DOM; ++dom)
    {
        total_error += err.sum(2*dom+1)/err.sum(2*dom);
    }
    // ================================================================

    return total_error;
}

/**
 * \brief Compute the error between the exact solution and the dG solution.
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
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename IBVP_CLASS>
Real EvalError(const Real t,
               const ImplicitMesh & mesh,
               const MatrixFactory & matfactory,
               const int N_SOL,
               const MultiFab & X,
               const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    return EvalError(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
}

} // namespace DG
} // namespace amrex

#endif