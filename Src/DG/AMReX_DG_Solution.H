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
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
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
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
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
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
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
        msg += "ERROR: AMReX_DG_Solution.H - EvalError\n";
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

//Print() << "err.sum(1): " << err.sum(1) << std::endl;
//Print() << "err.sum(0): " << err.sum(0) << std::endl;

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
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
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

/**
 * \brief Fill boundary cells with meaningful data.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
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
                       const ImplicitMesh & mesh,
                       const MatrixFactory & /*matfactory*/,
                       const int N_DOM,
                       const int N_SOL,
                       MultiFab & X,
                       const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("FillBoundaryCells(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution.H - FillBoundaryCells\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // FILL BOUNDARY CELLS ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            int BF_i, BF_j, BF_k;
            Real SOL[__DG_MAX_N_SOL__], bcs_SOL[__DG_MAX_N_SOL__];

            if (ELM_IS_NOT_EMPTY(etype))
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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);

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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);
                    
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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);

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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);
                    
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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);
                    
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
                    IBVP.F_BCS(dom, t, x, un, SOL, bcs_SOL);
                    
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
        msg += "ERROR: AMReX_DG_Solution.H - FillBoundaryCells\n";
        msg += "| X contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Fill boundary cells with meaningful data.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
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
                       const ImplicitMesh & mesh,
                       const MatrixFactory & matfactory,
                       const int N_SOL,
                       MultiFab & X,
                       const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    FillBoundaryCells(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
}

/**
 * \brief Fill invalid (boundary+small/empty) cells with meaningful data.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the AVERAGE of the unknown solution fields. On
 *                  exit, the boundary, small and empty (where possible) cells will have meaningful
 *                  values. 
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const ImplicitMesh & mesh,
                      const MatrixFactory & matfactory,
                      const int N_DOM,
                      const int N_SOL,
                      MultiFab & X,
                      const IBVP_CLASS & IBVP)
{
    // FILL THE BOUNDARY CELLS ========================================
    FillBoundaryCells(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // NEIGHBORS INFO IN STENCIL
    const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> aux_table_i =
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
    const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> aux_table_j =
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
    const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> aux_table_k = 
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
    const int sp = mesh.ls_std_elem.p;

    // NEAREST POINT SEARCH
    const int it_max = 10;
    const Real atol = 1.0e-3;
    const Real rtol = atol*AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);
    // ================================================================

    // FILL BOUNDARY CELLS ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
            const bool elm_is_not_valid = !ELM_IS_VALID(etype);

            // LOCAL VARIABLES
            bool elm_is_not_isolated;
            Real cc[AMREX_SPACEDIM], ncc[AMREX_SPACEDIM], nbr_x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            int nbr_i, nbr_j, nbr_k;
            short nbr_etype;
            int nbr_BF_i, nbr_BF_j, nbr_BF_k;
            Real nbr_cell_lo[AMREX_SPACEDIM], nbr_cell_hi[AMREX_SPACEDIM];
            Real nbr_SOL[__DG_MAX_N_SOL__], bcs_SOL[__DG_MAX_N_SOL__];

            // LEVEL SET
            DG_PHI_space_BFX phi(sp, PHI_fab);
            phi.set_u(Dom2Phi_ptr[2*dom]);

            // CHECK WHETHER THE ELEMENT HAS VALID NEIGHBORS ----------
            elm_is_not_isolated = false;

            if (elm_is_not_valid)
            {
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    nbr_i = i+aux_table_i[n];
                    nbr_j = j+aux_table_j[n];
                    nbr_k = k+aux_table_k[n];
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    if (ELM_IS_VALID(nbr_etype)) elm_is_not_isolated = true;
                }   
            }
            // --------------------------------------------------------

            if (elm_is_not_valid && elm_is_not_isolated)
            {
                // CELL'S CENTROID
                AMREX_D_TERM
                (
                    cc[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    cc[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    cc[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

//Print() << std::endl << "(" << i << "," << j << "," << k << ")" << std::endl;
//Print() << "cc: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, cc);

                // MIRRORED IMAGE OF THE CELL'S CENTROID
                nbr_i = i;
                nbr_j = j;
                nbr_k = k;
                NEAREST_POINT(&prob_lo[0], &dx[0], phi, cc,
                              nbr_i, nbr_j, nbr_k, nbr_cell_lo, nbr_cell_hi, ncc,
                              rtol, it_max);

//Print() << "ncc: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, ncc);
                
                // UNIT NORMAL
                phi.eval_un(ncc, un);
                AMREX_D_TERM
                (
                    un[0] *= -Dom2Phi_ptr[2*dom+1];,
                    un[1] *= -Dom2Phi_ptr[2*dom+1];,
                    un[2] *= -Dom2Phi_ptr[2*dom+1];
                )
                
                // SOLUTION AT THE NEIGHBOR ELEMENT
                AMREX_D_TERM
                (
                    nbr_x[0] = 2.0*ncc[0]-cc[0];,
                    nbr_x[1] = 2.0*ncc[1]-cc[1];,
                    nbr_x[2] = 2.0*ncc[2]-cc[2];
                )
                AMREX_D_TERM
                (
                    nbr_i = (int) std::floor((nbr_x[0]-prob_lo[0])/dx[0]);,
                    nbr_j = (int) std::floor((nbr_x[1]-prob_lo[1])/dx[1]);,
                    nbr_k = (int) std::floor((nbr_x[2]-prob_lo[2])/dx[2]);
                )

//Print() << "un: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
//Print() << "nbr_x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, nbr_x);
//Print() << "nbr(" << nbr_i << "," << nbr_j << "," << nbr_k << ")" << std::endl;

                // SUPPORT OF THE BASIS FUNCTIONS (NEIGHBOR CELL)
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    nbr_SOL[ru] = X_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ru);
                }

                IBVP.F_PHI_BCS(t, ncc, un, nbr_SOL, bcs_SOL);

                for (int ru = 0; ru < N_SOL; ++ru)
                {
                    X_fab(i,j,k,ru) = bcs_SOL[ru];
                }

//Print() << "bcs_SOL("; IO::PrintReals(AMREX_SPACEDIM, ncc); Print() << "): "; IO::PrintRealArray2D(1, N_SOL, bcs_SOL);
//if (linalg::contains_nan(N_SOL, bcs_SOL)) exit(-1);
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
        msg += "ERROR: AMReX_DG_Solution.H - FillInvalidCells\n";
        msg += "| X contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Fill invalid (boundary+small/empty) cells with meaningful data.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the AVERAGE of the unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const ImplicitMesh & mesh,
                      const MatrixFactory & matfactory,
                      const int N_SOL,
                      MultiFab & X,
                      const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    FillInvalidCells(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
}

} // namespace DG
} // namespace amrex

#endif