//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution_Hyperbolic.H
 * \brief Contains functions for implicit-mesh discontinuous Galerkin methods for hyperbolic PDEs.
*/

#ifndef AMREX_DG_SOLUTION_HYPERBOLIC_H_
#define AMREX_DG_SOLUTION_HYPERBOLIC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_DG_Constants.H>
#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

namespace amrex
{
namespace DG
{
namespace Hyperbolic
{

namespace Explicit
{
// ####################################################################
// SINGLE-LEVEL ROUTINES ##############################################
// ####################################################################
/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] tags: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const iMultiFab & mask,
                const int N_DOM,
                const int N_SOL,
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

    // DG
    const int sp = matfactory.std_elem.p;

    // MAX REAL
    const Real max_real = std::numeric_limits<Real>::max();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Compute_dt\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
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

        Array4<int const> const & mask_fab = mask.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype) && CELL_IS_NOT_MASKED(mask_fab(i,j,k)))
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
                Real x[AMREX_SPACEDIM];
                Real SOL[__DG_MAX_N_SOL__];
                Real elm_dt, x_dt;

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

/*
if (ELM_IS_SMALL(etype))
{
const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                   prob_lo[1]+(j+1)*dx[1],
                                                   prob_lo[2]+(k+1)*dx[2])};
Print() << "HELLO (" << i << "," << j << "," << k << ") -> (" << BF_i << "," << BF_j << "," << BF_k << ") " << std::endl;
Print() << "cell_lo: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, cell_lo);
Print() << "cell_hi: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, cell_hi);
Print() << "BF_lo: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, BF_lo);
Print() << "BF_hi: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, BF_hi);
exit(-1);
}
*/
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
                    sol.eval_BF_only_table(x);
                    sol.eval_unsafe(0, N_SOL, SOL);

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
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const int N_DOM,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    iMultiFab mask(X.boxarray, X.distributionMap, 1, X.n_grow);
    mask = __DG_CELL_UNMASKED__;

    return Compute_dt(t, mesh, matfactory, mask, N_DOM, N_SOL, X, IBVP);
}

/**
 * \brief Compute the maximum time step.
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
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    return Compute_dt(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
}

/**
 * \brief Eval the time derivative of the dG coefficients: Domain contribution.
 *
 * The following integral is evaluated: 
 *
 * int_{Vh} V_{,i}^T F_i
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_F(const int dom,
 *                           const Real t, const Real * x,
 *                           const Real * SOL, Real * F1, Real * F2, Real * F3) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_DomContrib(const Real t,
                          const ImplicitMesh & mesh,
                          const MatrixFactory & matfactory,
                          const int N_SOL,
                          const Gpu::ManagedVector<int> & Sol2Dom,
                          const MultiFab & X,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_DomContrib(const Real, const ImplicitMesh &, ....)");
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
        msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt_DomContrib\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
#ifdef AMREX_DEBUG
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
#endif
        
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // ELEMENT TYPE
            const int dom = Sol2Dom_ptr[ru];
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
#ifdef AMREX_DEBUG
            const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));
#endif

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
#ifdef AMREX_DEBUG
                const Real cell_v = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
#endif
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w;
                Real SOL[__DG_MAX_N_SOL__];
                Real AMREX_D_DECL(F1[__DG_MAX_N_SOL__], F2[__DG_MAX_N_SOL__], F3[__DG_MAX_N_SOL__]);
                Real integrand;
#ifdef AMREX_DEBUG
                Real elm_v;
#endif

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                // COMPUTE THE INTEGRAL
#ifdef AMREX_DEBUG
                elm_v = 0.0;
#endif
                for (int q = 0; q < dom_Nq; ++q)
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
                    sol.eval(x, 0, N_SOL, SOL);

                    // FLUXES
                    IBVP.F_F(dom, t, x, SOL, AMREX_D_DECL(F1, F2, F3));

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
#ifdef AMREX_DEBUG
                    // INTEGRATION TEST
                    elm_v += w;
#endif
                }
                
                // INTEGRATION TEST
#ifdef AMREX_DEBUG
                if (std::abs(elm_v/cell_v-vf)*100.0 > 1.0e-2)
                {
#ifdef AMREX_USE_GPU
                    Abort("ERROR: AMReX_DG_Solution_Hyperbolic.H - The number of quadrature points might not be enough");
#else
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt_DomContrib\n";
                    msg += "| Error in the domain quadrature.\n";
                    msg += "| The number of quadrature points might not be enough.\n";
                    msg += "| Cell: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                    msg += "| elm_v          : "+std::to_string(elm_v)+".\n";
                    msg += "| vf*cell_v      : "+std::to_string(vf*cell_v)+".\n";
                    msg += "| elm_v-vf*cell_v: "+std::to_string(elm_v-vf*cell_v)+".\n";
                    msg += "| err (%)        : "+std::to_string(std::abs(elm_v/cell_v-vf)*100.0)+".\n";
                    Abort(msg);
#endif
                }
#endif
            }
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
 * int_{dVh} V^T NFn
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const ImplicitMesh & mesh,
                          const MatrixFactory & matfactory,
                          const int N_SOL,
                          const Gpu::ManagedVector<int> & Sol2Dom,
                          const MultiFab & X,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const Geometry &, ....)");
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
        msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt_BouContrib\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
#ifdef AMREX_DEBUG
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
#endif
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

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // ELEMENT TYPE
            const int dom = Sol2Dom_ptr[ru];
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
#ifdef AMREX_DEBUG
            const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));
#endif

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
#ifdef AMREX_DEBUG
                const Real cell_v = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
#endif
                const bool elm_is_small = ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                int nbr_i, nbr_j, nbr_k, nbr_b;
                short nbr_etype;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real SOL[__DG_MAX_N_SOL__], nbr_SOL[__DG_MAX_N_SOL__];
                Real NFn[__DG_MAX_N_SOL__];
                Real integrand;
#ifdef AMREX_DEBUG
                Real elm_v;
#endif

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

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
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        const bool nbr_is_small = ELM_IS_SMALL(nbr_etype);
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
                        const bool nbr_is_empty = ELM_IS_EMPTY(nbr_etype);
                        if (cells_are_merged || nbr_is_empty) bou_is_not_valid = true;
                    }

                    // FACE QUADRATURE INFO
                    intraphase = true;
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        long pos;
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

                            if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                                (is_periodic[dir] == 0))
                            {
                                intraphase = false;
                            }
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

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
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));

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
                        // SUPPORT OF THE BASIS FUNCTIONS (NBR CELL)
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);

                        // SOLUTION (NBR CELL)
                        DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

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
                            sol.eval(x, 0, N_SOL, SOL);
                            nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

                            // NUMERICAL FLUX
                            IBVP.F_NF_ICS(dom, t, x, un, SOL, nbr_SOL, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                integrand = sol.BF[rs]*NFn[ru];
                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
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
                            sol.eval(x, 0, N_SOL, SOL);

                            // NUMERICAL FLUX
                            IBVP.F_NF_BCS(dom, t, x, un, SOL, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                integrand = sol.BF[rs]*NFn[ru];
                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                }

                // ZERO-LEVEL OF THE LEVEL SET (IF ANY)
                {
                    // NEIGHBOR DOMAIN (IF ANY)
                    const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
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
                    if (nbr_dom == -1)
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
                            sol.eval(x, 0, N_SOL, SOL);

                            // NUMERICAL FLUX
                            IBVP.F_NF_PHI_BCS(dom, t, x, un, SOL, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                integrand = sol.BF[rs]*NFn[ru];
                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                    else
                    {
Print() << "HELLO - Eval_dXdt_BouContrib - NUMERICAL FLUX - nbr_dom != -1" << std::endl;
exit(-1);
                    }
                }

                // INTEGRATION TEST
#ifdef AMREX_DEBUG
                if (std::abs(elm_v/cell_v-vf)*100.0 > 1.0e-2)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt_BouContrib\n";
                    msg += "| Error in the boundary quadrature.\n";
                    msg += "| The number of quadrature points might not be enough.\n";
                    msg += "| Cell: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                    msg += "| elm_v          : "+std::to_string(elm_v)+".\n";
                    msg += "| vf*cell_v      : "+std::to_string(vf*cell_v)+".\n";
                    msg += "| elm_v-vf*cell_v: "+std::to_string(elm_v-vf*cell_v)+".\n";
                    msg += "| err (%)        : "+std::to_string(std::abs(elm_v/cell_v-vf)*100.0)+".\n";
                    Warning(msg);
                }
#endif
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const ImplicitMesh & mesh,
               const MatrixFactory & matfactory,
               const int N_SOL,
               const Gpu::ManagedVector<int> & Sol2Dom,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL THE TIME DERIVATIVES: DOMAIN CONTRIBUTION =================
    Eval_dXdt_DomContrib(t, mesh, matfactory, N_SOL, Sol2Dom, X, dXdt, IBVP);
    // ================================================================

    // EVAL THE TIME DERIVATIVES: BOUNDARY CONTRIBUTION ===============
    Eval_dXdt_BouContrib(t, mesh, matfactory, N_SOL, Sol2Dom, X, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    MultiplyByInverseMassMatrix(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients will be updated according to
 *                  the selected Runge-Kutta algorithm.
 * \param[out] tags: a TagBoxArray object that contains information about which cells are tagged
 *                   for limiting.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
 * \param[in] apply_slope_limiter: flag that, if true, enables the slope limiter.
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const ImplicitMesh & mesh,
                            const MatrixFactory & matfactory,
                            const int N_DOM,
                            const int N_SOL,
                            const Gpu::ManagedVector<int> & Sol2Dom,
                            MultiFab & X_new,
                            TagBoxArray & tags,
                            const IBVP_CLASS & IBVP,
                            const bool apply_slope_limiter = false)
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
        Eval_dXdt(t, mesh, matfactory, N_SOL, Sol2Dom, X_old, dXdt, IBVP);

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
        Eval_dXdt(t, mesh, matfactory, N_SOL, Sol2Dom, X_old, dXdt, IBVP);

        // X1 := X_old+dt*dXdt
        MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK2 - apply_slope_limiter" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, X1, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, mesh, matfactory, N_SOL, Sol2Dom, X1, dXdt, IBVP);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK2 - apply_slope_limiter - 2" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, X_new, tags, IBVP);
        }
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
        Eval_dXdt(t, mesh, matfactory, N_SOL, Sol2Dom, X_old, dXdt1, IBVP);

        // Xs := X_old+dt*dXdt1
        MultiFab::LinComb(Xs, 1.0, X_old, 0, dt, dXdt1, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK3 - apply_slope_limiter - 1" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, Xs, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK3 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt, Xs)
        Eval_dXdt(t+dt, mesh, matfactory, N_SOL, Sol2Dom, Xs, dXdt2, IBVP);

        // Xs := X_old+(dt/4)*dXdt1+(dt/4)*dXdt2
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.25*dt, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(Xs,                  0.25*dt, dXdt2, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK3 - apply_slope_limiter - 2" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, Xs, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK3 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t+0.5*dt, mesh, matfactory, N_SOL, Sol2Dom, Xs, dXdt3, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/6)*dXdt2+(2*dt/3)*dXdt3
        MultiFab::Saxpy(X_new,     dt/6.0, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new,     dt/6.0, dXdt2, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 2.0*dt/3.0, dXdt3, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK3 - apply_slope_limiter - 3" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, X_new, tags, IBVP);
        }
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
        Eval_dXdt(t, mesh, matfactory, N_SOL, Sol2Dom, X_old, dXdt1, IBVP);

        // Xs := X_old+(dt/2)*dXdt1
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.5*dt, dXdt1, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK4 - apply_slope_limiter - 1" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, Xs, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK4 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t+0.5*dt, mesh, matfactory, N_SOL, Sol2Dom, Xs, dXdt2, IBVP);

        // Xs := X_old+(dt/2)*dXdt2
        MultiFab::LinComb(Xs, 1.0, X_old, 0, 0.5*dt, dXdt2, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK4 - apply_slope_limiter - 2" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, Xs, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK4 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, Xs)
        Eval_dXdt(t+0.5*dt, mesh, matfactory, N_SOL, Sol2Dom, Xs, dXdt3, IBVP);

        // Xs := X_old+dt*dXdt3
        MultiFab::LinComb(Xs, 1.0, X_old, 0, dt, dXdt3, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK4 - apply_slope_limiter - 3" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, Xs, tags, IBVP);
        }
        // ------------------------------------------------------------

        // RK4 STEP 4 -------------------------------------------------
        // dXdt4 := dXdt(t+dt, Xs)
        Eval_dXdt(t+dt, mesh, matfactory, N_SOL, Sol2Dom, Xs, dXdt4, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/3)*dXdt2+(dt/3)*dXdt3+(dt/6)*dXdt4
        MultiFab::Saxpy(X_new, dt/6.0, dXdt1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/3.0, dXdt2, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/3.0, dXdt3, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, dt/6.0, dXdt4, 0, 0, n_comp, ngr);

        // Apply slope limiter
        if (apply_slope_limiter)
        {
Print() << "HELLO RK4 - apply_slope_limiter - 4" << std::endl;
exit(-1);
            //ApplySlopeLimiter(t, mesh, matfactory, N_SOL, Sol2Dom, X_new, tags, IBVP);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
    // ================================================================
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ================================================================

    // CHECK SOLUTION =================================================
    if (X_new.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Solution contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] X: a MultiFab object that contains the coefficients of the basis functions for the
 *                  unknown solution fields. On exit, the coefficients will be updated according to
 *                  the selected Runge-Kutta algorithm.
 * \param[out] tags: a TagBoxArray object that contains information about which cells are tagged
 *                   for limiting.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const ImplicitMesh & mesh,
                            const MatrixFactory & matfactory,
                            const int N_SOL,
                            MultiFab & X_new,
                            const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);
    TagBoxArray tags(X_new.boxarray, X_new.distributionMap, X_new.n_grow);
    tags.setVal(TagBox::CLEAR);

    TakeRungeKuttaTimeStep(RK_order, dt, t,
                           mesh, matfactory,
                           N_DOM, N_SOL, Sol2Dom,
                           X_new,
                           tags,
                           IBVP);
}
// ####################################################################
// ####################################################################



// ####################################################################
// ADAPTIVE MESH REFINEMENT ROUTINES ##################################
// ####################################################################
/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const int N_DOM,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    const int finest_level = AMR.maxLevel();
    // ================================================================

    // VARIABLES ======================================================
    Real dt_min, dt_level;
    // ================================================================

    // CYCLE OVER THE LEVELS ==========================================
    dt_min = std::numeric_limits<Real>::max();

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (AMR.LevelIsValid(lev))
        {
            const Real CFL = AMR.inputs.grids[lev].CFL;
            const int p = AMR.inputs.dG[lev].space_p;

            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MatrixFactory & matfactory = *AMR.matfactories[lev];
            const iMultiFab & mask = AMR.masks[lev];
            const MultiFab & X = AMR.Xs[lev];

            dt_level = Compute_dt(t, mesh, matfactory, mask, N_DOM, N_SOL, X, IBVP);
            dt_level *= CFL/(1.0+2.0*p);

            dt_min = std::min(dt_min, dt_level);
        }
    }
    // ================================================================

    return dt_min;
}

/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    return Compute_dt(t, N_DOM, N_SOL, AMR, IBVP);
}

/**
 * \brief Eval the time derivative of the dG coefficients: Domain contribution.
 *
 * The following integral is evaluated: 
 *
 * int_{Vh} V_{,i}^T F_i
 *
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object.
 * \param[in] matfactory: the MatrixFactory object.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_F(const int dom,
 *                           const Real t, const Real * x,
 *                           const Real * SOL, Real * F1, Real * F2, Real * F3) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_DomContrib(const Real t,
                          const ImplicitMesh & mesh,
                          const MatrixFactory & matfactory,
                          const int N_DOM,
                          const int N_SOL,
                          const MultiFab & X,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_DomContrib(const Real, const ImplicitMesh &, ....)");
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
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_DomContrib", N_SOL);
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
#ifdef AMREX_DEBUG
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
#endif
        
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
#ifdef AMREX_DEBUG
            const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));
#endif

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
#ifdef AMREX_DEBUG
                const Real cell_v = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
#endif
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w;
                Real SOL[__DG_MAX_N_SOL__];
                Real AMREX_D_DECL(F1[__DG_MAX_N_SOL__], F2[__DG_MAX_N_SOL__], F3[__DG_MAX_N_SOL__]);
                Real integrand;
#ifdef AMREX_DEBUG
                Real elm_v;
#endif

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                // COMPUTE THE INTEGRAL
#ifdef AMREX_DEBUG
                elm_v = 0.0;
#endif
                for (int q = 0; q < dom_Nq; ++q)
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
                    sol.eval(x, 0, N_SOL, SOL);

                    // FLUXES
                    IBVP.F_F(dom, t, x, SOL, AMREX_D_DECL(F1, F2, F3));

                    // INTEGRAL CONTRIBUTION
                    for (int ru = 0; ru < N_SOL; ++ru)
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
#ifdef AMREX_DEBUG
                    // INTEGRATION TEST
                    elm_v += w;
#endif
                }
                
                // INTEGRATION TEST
#ifdef AMREX_DEBUG
                if (std::abs(elm_v/cell_v-vf)*100.0 > 1.0e-2)
                {
#ifdef AMREX_USE_GPU
                    Abort("ERROR: AMReX_DG_Solution_Hyperbolic.H - The number of quadrature points might not be enough");
#else
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt_DomContrib\n";
                    msg += "| Error in the domain quadrature.\n";
                    msg += "| The number of quadrature points might not be enough.\n";
                    msg += "| Cell: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                    msg += "| elm_v          : "+std::to_string(elm_v)+".\n";
                    msg += "| vf*cell_v      : "+std::to_string(vf*cell_v)+".\n";
                    msg += "| elm_v-vf*cell_v: "+std::to_string(elm_v-vf*cell_v)+".\n";
                    msg += "| err (%)        : "+std::to_string(std::abs(elm_v/cell_v-vf)*100.0)+".\n";
                    Abort(msg);
#endif
                }
#endif
            }
        });
        Gpu::synchronize();
    }
    // ================================================================
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_WALL(const Real t,
                 const int dom,
                 const Real * cell_lo, const Real * un,
                 const int bou_Nq, const Real * x_ptr,
                 DG_SOL_space_RX & sol,
                 const int N_SOL,
                 const int fi, const int fj, const int fk,
                 Array4<Real> const & FX_fab, const int offset,
                 const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_BCS(dom, t, x, un, SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,offset+ru) += integrand*w;
        }
    }
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_ICS(const Real t,
                const int dom,
                const Real * cell_lo, const Real * un,
                const int bou_Nq, const Real * x_ptr,
                DG_SOL_space_RX & sol, DG_SOL_space_RX & nbr_sol,
                const int N_SOL,
                const int fi, const int fj, const int fk,
                Array4<Real> const & FX_fab, const int offset,
                const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], nbr_SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);
        nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_ICS(dom, t, x, un, SOL, nbr_SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,offset+ru) -= integrand*w;
        }
    }
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_ICS(const Real t,
                const int dom,
                const Real * cell_lo, const Real * un,
                const int bou_Nq, const Real * x_ptr,
                DG_SOL_space_BFX & sol, DG_SOL_space_RX & nbr_sol,
                const int N_SOL, const int sNp,
                const int fi, const int fj, const int fk,
                Array4<Real> const & FX_fab, const int offset,
                const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], nbr_SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);
        nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_ICS(dom, t, x, un, SOL, nbr_SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        {
            for (int rs = 0; rs < sNp; ++rs)
            {
                integrand = sol.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*sNp) += integrand*w;
            }
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,offset+ru) -= integrand*w;
        }
    }
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_ICS(const Real t,
                const int dom,
                const Real * cell_lo, const Real * un,
                const int bou_Nq, const Real * x_ptr,
                DG_SOL_space_RX & sol, DG_SOL_space_BFX & nbr_sol,
                const int N_SOL, const int nbr_sNp,
                const int fi, const int fj, const int fk,
                Array4<Real> const & FX_fab, const int offset,
                const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], nbr_SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);
        nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_ICS(dom, t, x, un, SOL, nbr_SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        {
            integrand = NFn[ru];
            FX_fab(fi,fj,fk,ru) += integrand*w;
            for (int rs = 0; rs < nbr_sNp; ++rs)
            {
                integrand = nbr_sol.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,offset+rs+ru*nbr_sNp) -= integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the cell interfaces.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V^T NFn
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the state vector of the unknown solution fields.
 * \param[in] dX: a MultiFab object that contains the slopes of the unknown solution fields.
 * \param[out] FX: a MultiFab object that will contain the FX.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouFluxes_FV(const Real t,
                            const ImplicitMesh & mesh,
                            const int N_DOM,
                            const int N_SOL,
                            const MultiFab & X,
                            const MultiFab & dX,
                            Array<MultiFab, AMREX_SPACEDIM> & FX,
                            const int offset,
                            const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouFluxes_FV(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouFluxes_FV", N_SOL);
    // ================================================================

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
    // ================================================================

    // INITIALIZATION =================================================
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        FX[dim] = 0.0;
    }
    // ================================================================

    // EVAL THE INTEGRALS: CELL BOUNDARIES ============================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int const> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

            Array4<Real const> const & X_fab = X.array(mfi);
            Array4<Real const> const & dX_fab = dX.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short bou_type = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(bou_type))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};
                    const int bou_Nq = eBouQuad_Nq_fab(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                    const long pos = eBouQuad_pos_fab(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int i, j, k, nbr_i, nbr_j, nbr_k;
                    
                    int BF_i, BF_j, BF_k;
                    Real xc[AMREX_SPACEDIM];
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    Real nbr_xc[AMREX_SPACEDIM];

                    // FACE INDICES TO ADJACENT NEIGHBORS INDICES
                    FACE2NBRS(fi, fj, fk, dir, i, j, k, nbr_i, nbr_j, nbr_k);

                    // ELEMENT INFO
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                    // SELECT TYPE OF CELL BOUNDARY
                    // WALL
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // SOLUTION
                        BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX nbr_sol(X_fab, dX_fab, nbr_xc, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, nbr_sol, N_SOL, fi, fj, fk, FX_fab, offset, IBVP);
                    }
                    // WALL
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                        AMREX_D_TERM
                        (
                            xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                            xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                            xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, sol, N_SOL, fi, fj, fk, FX_fab, 0, IBVP);
                    }
                    // INTRAPHASE
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                        AMREX_D_TERM
                        (
                            xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                            xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                            xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);
                        BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX nbr_sol(X_fab, dX_fab, nbr_xc, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, fi, fj, fk, FX_fab, offset, IBVP);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================
}

/**
 * \brief Eval the boundary fluxes at the cell interfaces.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V^T NFn
 *
 * \param[in] t: time.
 * \param[in] c_rr: refinement ratio of the coarser level.
 * \param[in] c_eType: a shortMultiFab object containing the elementy type information at coarse
 *                     level.
 * \param[in] c_sp: space order of the coarse level.
 * \param[in] c_X: a MultiFab object containing the coefficients of the basis functions at the
 *                 coarse level.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the state vector of the unknown solution fields.
 * \param[in] dX: a MultiFab object that contains the slopes of the unknown solution fields.
 * \param[out] FX: a MultiFab object that will contain the FX.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouFluxes_FV(const Real t,
                            const IntVect c_rr,
                            const shortMultiFab & c_eType,
                            const int c_sp,
                            const MultiFab & c_X,
                            const ImplicitMesh & mesh,
                            const int N_DOM,
                            const int N_SOL,
                            const MultiFab & X,
                            const MultiFab & dX,
                            Array<MultiFab, AMREX_SPACEDIM> & FX,
                            const int offset,
                            const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouFluxes_FV(const Real, const IntVect, const shortMultiFab &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouFluxes_FV", N_SOL);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    const GpuArray<Real, AMREX_SPACEDIM> c_dx = {AMREX_D_DECL(dx[0]*c_rr[0], dx[1]*c_rr[1], dx[2]*c_rr[2])};

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
#endif
    
    // DG
    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));

    // PARALLEL COPY?
    const bool parallel_copy_is_needed = (!isMFIterSafe(X, c_X));
    // ================================================================

    // VARIABLES ======================================================
    shortMultiFab safe_c_eType;
    const shortMultiFab * safe_c_eType_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ================================================================

    // SAFE COARSE DATA ===============================================
    if (parallel_copy_is_needed)
    {
        const BoxArray safe_c_ba = coarsen(mesh.cc_ba, c_rr);
        const DistributionMapping & f_dm = mesh.dm;
        const int eType_n_comp = c_eType.n_comp;
        const IntVect & eType_n_grow = c_eType.n_grow;
        const int X_n_comp = c_X.n_comp;
        const IntVect & X_n_grow = c_X.n_grow;
        
        safe_c_eType.define(safe_c_ba, f_dm, eType_n_comp, eType_n_grow);
        safe_c_eType.ParallelCopy(c_eType, 0, 0, eType_n_comp, eType_n_grow, eType_n_grow, mesh.geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, X_n_comp, X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, X_n_comp, X_n_grow, X_n_grow, mesh.geom.periodicity());

        safe_c_eType_ptr = &safe_c_eType;
        safe_c_X_ptr = &safe_c_X;
    }
    else
    {
        safe_c_eType_ptr = &c_eType;
        safe_c_X_ptr = &c_X;
    }
    // ================================================================

    // INITIALIZATION =================================================
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        FX[dim] = 0.0;
    }
    // ================================================================

    // EVAL THE INTEGRALS: CELL BOUNDARIES ============================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(c_rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(c_rr[d1]*c_rr[d2]);
#endif
        
        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & c_eType_fab = safe_c_eType_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int const> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

            Array4<Real const> const & X_fab = X.array(mfi);
            Array4<Real const> const & dX_fab = dX.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short bou_type = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(bou_type))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};
                    const int bou_Nq = eBouQuad_Nq_fab(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                    const long pos = eBouQuad_pos_fab(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int i, j, k, nbr_i, nbr_j, nbr_k;
                    
                    int BF_i, BF_j, BF_k;
                    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    Real xc[AMREX_SPACEDIM];
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
                    Real nbr_xc[AMREX_SPACEDIM];

                    // FACE INDICES TO ADJACENT NEIGHBORS INDICES
                    FACE2NBRS(fi, fj, fk, dir, i, j, k, nbr_i, nbr_j, nbr_k);

                    // ELEMENT INFO
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool elm_is_ghost = ELM_IS_GHOST(etype);
                    const bool nbr_is_ghost = ELM_IS_GHOST(nbr_etype);

                    // SELECT TYPE OF CELL BOUNDARY
                    // WALL
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // SOLUTION
                        BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX nbr_sol(X_fab, dX_fab, nbr_xc, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, nbr_sol, N_SOL, fi, fj, fk, FX_fab, offset, IBVP);
                    }
                    // WALL
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                        AMREX_D_TERM
                        (
                            xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                            xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                            xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                        )
                        DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, sol, N_SOL, fi, fj, fk, FX_fab, 0, IBVP);
                    }
                    // INTRAPHASE
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (elm_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(i, j, k, c_rr, c_i, c_j, c_k);
                            const short c_etype = c_eType_fab(c_i,c_j,c_k,ELM_TYPE(dom));

                            // SOLUTION
                            BF_CELL(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                            DG_SOL_space_BFX sol(c_sp, BF_lo, BF_hi, c_X_fab, BF_i, BF_j, BF_k);
                            BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                            AMREX_D_TERM
                            (
                                nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                                nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                                nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                            )
                            DG_SOL_space_RX nbr_sol(X_fab, dX_fab, nbr_xc, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, c_sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (nbr_is_ghost)
                        {
                            // BASIS FUNCTIONS OF THE COARSER LEVEL
                            int c_nbr_i, c_nbr_j, c_nbr_k;
                            FINE_TO_COARSE(nbr_i, nbr_j, nbr_k, c_rr, c_nbr_i, c_nbr_j, c_nbr_k);
                            const short c_nbr_etype = c_eType_fab(c_nbr_i,c_nbr_j,c_nbr_k,ELM_TYPE(dom));

                            // SOLUTION
                            BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                            AMREX_D_TERM
                            (
                                xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                                xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                                xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                            )
                            DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);
                            BF_CELL(&prob_lo[0], &c_dx[0], c_nbr_i, c_nbr_j, c_nbr_k, c_nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            DG_SOL_space_BFX nbr_sol(c_sp, nbr_BF_lo, nbr_BF_hi, c_X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, c_sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // SOLUTION
                            BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                            AMREX_D_TERM
                            (
                                xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                                xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                                xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                            )
                            DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);
                            BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                            AMREX_D_TERM
                            (
                                nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X1(dom));,
                                nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X2(dom));,
                                nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,ELM_CENTROID_X3(dom));
                            )
                            DG_SOL_space_RX nbr_sol(X_fab, dX_fab, nbr_xc, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, fi, fj, fk, FX_fab, offset, IBVP);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================
}

/**
 * \brief Eval the fluxes at the internal boundaries and add contribution to the time derivatives.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the state vector of the unknown solution fields.
 * \param[in] dX: a MultiFab object that contains the slopes of the unknown solution fields.
 * \param[in] FX: a MultiFab object that contains the fluxes.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the the state vector
 *                   of the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib_FV(const Real t,
                             const ImplicitMesh & mesh,
                             const int N_DOM,
                             const int N_SOL,
                             const MultiFab & X,
                             const MultiFab & dX,
                             const Array<MultiFab, AMREX_SPACEDIM> & FX,
                             const int offset,
                             MultiFab & dXdt,
                             const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib_FV(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouContrib_FV", N_SOL);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // ADD FLUX CONTRIBUTION: INTERNAL BOUNDARIES =====================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                // NEIGHBOR DOMAIN (IF ANY)
                const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real xc[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real SOL[__DG_MAX_N_SOL__];
                Real NFn[__DG_MAX_N_SOL__];
                Real integrand;
                
                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                )
                DG_SOL_space_RX sol(X_fab, dX_fab, xc, BF_i, BF_j, BF_k);

                if (nbr_dom == -1)
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
                        sol.eval(x, 0, N_SOL, SOL);

                        // NUMERICAL FLUX
                        IBVP.F_NF_PHI_BCS(dom, t, x, un, SOL, NFn);

                        // INTEGRAL CONTRIBUTION
                        for (int ru = 0; ru < N_SOL; ++ru)
                        {
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                }
                else
                {
Print() << "Eval_dXdt_BouContrib_FV - NUMERICAL FLUX - nbr_dom != -1" << std::endl;
exit(-1);
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================

    // ADD FLUX CONTRIBUTION: CELL BOUNDARIES =========================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;

        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = 0; ru < N_SOL; ++ru)
                    {
                        dXdt_fab(i,j,k,ru) -= (FX_fab(fi,fj,fk,ru)+FX_fab(i,j,k,offset+ru));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================

    // SHARE INFO =====================================================
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_WALL(const Real t,
                 const int dom,
                 const Real * cell_lo, const Real * un,
                 const int bou_Nq, const Real * x_ptr,
                 DG_SOL_space_BFX & sol,
                 const int N_SOL, const int sNp,
                 const int fi, const int fj, const int fk,
                 Array4<Real> const & FX_fab, const int offset,
                 const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_BCS(dom, t, x, un, SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        for (int rs = 0; rs < sNp; ++rs)
        {
            integrand = sol.BF[rs]*NFn[ru];
            FX_fab(fi,fj,fk,offset+rs+ru*sNp) += integrand*w;
        }
    }
}

/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
template <typename IBVP_CLASS>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void BOU_FX_ICS(const Real t,
                const int dom,
                const Real * cell_lo, const Real * un,
                const int bou_Nq, const Real * x_ptr,
                DG_SOL_space_BFX & sol, DG_SOL_space_BFX & nbr_sol,
                const int N_SOL, const int sNp, const int nbr_sNp,
                const int fi, const int fj, const int fk,
                Array4<Real> const & FX_fab, const int offset,
                const IBVP_CLASS & IBVP)
{
    // VARIABLES
    Real integrand;
    Real SOL[__DG_MAX_N_SOL__], nbr_SOL[__DG_MAX_N_SOL__], NFn[__DG_MAX_N_SOL__];

    // COMPUTE THE INTEGRAL
    for (int q = 0; q < bou_Nq; ++q)
    {
        // QUADRATURE POINT AND WEIGHT
        const long x_pos = (AMREX_SPACEDIM+1)*q;
        const Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                     x_ptr[x_pos+1]+cell_lo[1],
                                                     x_ptr[x_pos+2]+cell_lo[2])};
        const Real w = x_ptr[x_pos+AMREX_SPACEDIM];

        // EVAL SOLUTION
        sol.eval(x, 0, N_SOL, SOL);
        nbr_sol.eval(x, 0, N_SOL, nbr_SOL);

        // NUMERICAL FLUX
        IBVP.F_NF_ICS(dom, t, x, un, SOL, nbr_SOL, NFn);

        // INTEGRAL CONTRIBUTION
        for (int ru = 0; ru < N_SOL; ++ru)
        {
            for (int rs = 0; rs < sNp; ++rs)
            {
                integrand = sol.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,rs+ru*sNp) += integrand*w;
            }
            for (int rs = 0; rs < nbr_sNp; ++rs)
            {
                integrand = nbr_sol.BF[rs]*NFn[ru];
                FX_fab(fi,fj,fk,offset+rs+ru*nbr_sNp) -= integrand*w;
            }
        }
    }
}

/**
 * \brief Eval the boundary fluxes at the cell interfaces.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V_{,i}^T NFn
 *
 * This routine is intended to be used for single-level applications
 * or for the coarsest level in a multi-level applications.
 *
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object.
 * \param[in] matfactory: the MatrixFactory object.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouFluxes(const Real t,
                         const ImplicitMesh & mesh,
                         const MatrixFactory & matfactory,
                         const int N_DOM,
                         const int N_SOL,
                         const MultiFab & X,
                         Array<MultiFab, AMREX_SPACEDIM> & FX,
                         const int offset,
                         const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouFluxes(const Real, const ImplicitMesh &, const MatrixFactory &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouFluxes", N_SOL);
    // ================================================================

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
    
    // DG
    const int sp = matfactory.std_elem.p;
    const int sNp = matfactory.std_elem.Np;
    // ================================================================

    // VARIABLES ======================================================
    // ================================================================

    // INITIALIZATION =================================================
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        FX[dim] = 0.0;
    }
    // ================================================================

    // EVAL THE INTEGRALS: CELL BOUNDARIES ============================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        
        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int const> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

            Array4<Real const> const & X_fab = X.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short bou_type = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(bou_type))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};
                    const int bou_Nq = eBouQuad_Nq_fab(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                    const long pos = eBouQuad_pos_fab(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int i, j, k, nbr_i, nbr_j, nbr_k;
                    
                    int BF_i, BF_j, BF_k;
                    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];

                    // FACE INDICES TO ADJACENT NEIGHBORS INDICES
                    FACE2NBRS(fi, fj, fk, dir, i, j, k, nbr_i, nbr_j, nbr_k);

                    // ELEMENT INFO
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                    // SELECT TYPE OF CELL BOUNDARY
                    // WALL
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // SOLUTION
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, nbr_sol, N_SOL, sNp, fi, fj, fk, FX_fab, offset, IBVP);

                    }
                    // WALL
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                        DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, sol, N_SOL, sNp, fi, fj, fk, FX_fab, 0, IBVP);
                    }
                    // INTRAPHASE
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                        DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, sNp, sNp, fi, fj, fk, FX_fab, offset, IBVP);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================
}

/**
 * \brief Eval the boundary fluxes at the cell interfaces.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V_{,i}^T NFn
 *
 * \param[in] t: time.
 * \param[in] c_rr: refinement ratio of the coarser level.
 * \param[in] c_eType: a shortMultiFab object containing the elementy type information at coarser
 *                     level.
 * \param[in] c_X: a MultiFab object containing the coefficients of the basis functions at the
 *                 coarser level.
 * \param[in] mesh: the ImplicitMesh object.
 * \param[in] matfactory: the MatrixFactory object.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] FX: an array of MultiFab objects that will contain the fluxes.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouFluxes(const Real t,
                         const IntVect c_rr,
                         const shortMultiFab & c_eType,
                         const int c_sp,
                         const MultiFab & c_X,
                         const ImplicitMesh & mesh,
                         const MatrixFactory & matfactory,
                         const int N_DOM,
                         const int N_SOL,
                         const MultiFab & X,
                         Array<MultiFab, AMREX_SPACEDIM> & FX,
                         const int offset,
                         const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouFluxes(const Real, const IntVect, const shortMultiFab &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouFluxes", N_SOL);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    const GpuArray<Real, AMREX_SPACEDIM> c_dx = {AMREX_D_DECL(dx[0]*c_rr[0], dx[1]*c_rr[1], dx[2]*c_rr[2])};

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
#endif
    
    // DG
    const int sp = matfactory.std_elem.p;
    const int sNp = matfactory.std_elem.Np;

    const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));

    // PARALLEL COPY?
    const bool parallel_copy_is_needed = (!isMFIterSafe(X, c_X));
    // ================================================================

    // VARIABLES ======================================================
    shortMultiFab safe_c_eType;
    const shortMultiFab * safe_c_eType_ptr;
    MultiFab safe_c_X;
    const MultiFab * safe_c_X_ptr;
    // ================================================================

    // SAFE COARSE DATA ===============================================
    if (parallel_copy_is_needed)
    {
        const BoxArray safe_c_ba = coarsen(mesh.cc_ba, c_rr);
        const DistributionMapping & f_dm = mesh.dm;
        const int eType_n_comp = c_eType.n_comp;
        const IntVect & eType_n_grow = c_eType.n_grow;
        const int X_n_comp = c_X.n_comp;
        const IntVect & X_n_grow = c_X.n_grow;
        
        safe_c_eType.define(safe_c_ba, f_dm, eType_n_comp, eType_n_grow);
        safe_c_eType.ParallelCopy(c_eType, 0, 0, eType_n_comp, eType_n_grow, eType_n_grow, mesh.geom.periodicity());
        
        safe_c_X.define(safe_c_ba, f_dm, X_n_comp, X_n_grow, MFInfo(), FArrayBoxFactory());
        safe_c_X.ParallelCopy(c_X, 0, 0, X_n_comp, X_n_grow, X_n_grow, mesh.geom.periodicity());

        safe_c_eType_ptr = &safe_c_eType;
        safe_c_X_ptr = &safe_c_X;
    }
    else
    {
        safe_c_eType_ptr = &c_eType;
        safe_c_X_ptr = &c_X;
    }
    // ================================================================

    // INITIALIZATION =================================================
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        FX[dim] = 0.0;
    }
    // ================================================================

    // EVAL THE INTEGRALS: CELL BOUNDARIES ============================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
#if (AMREX_SPACEDIM == 1)
        const Real fact = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
        const Real fact = Real(c_rr[1-dir]);
#endif
#if (AMREX_SPACEDIM == 3)
        const int d1 = (dir*(dir-1))/2;
        const int d2 = 1+((3-dir)*dir)/2;
        const Real fact = Real(c_rr[d1]*c_rr[d2]);
#endif
        
        for (MFIter mfi(FX[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & c_eType_fab = safe_c_eType_ptr->array(mfi);
            Array4<Real const> const & c_X_fab = safe_c_X_ptr->array(mfi);

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int const> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
            Array4<long const> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

            Array4<Real const> const & X_fab = X.array(mfi);

            Array4<Real> const & FX_fab = FX[dir].array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
            {
                // ELEMENT BOUNDARY TYPE
                const short bou_type = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(bou_type))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};
                    const int bou_Nq = eBouQuad_Nq_fab(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                    const long pos = eBouQuad_pos_fab(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int i, j, k, nbr_i, nbr_j, nbr_k;
                    
                    int BF_i, BF_j, BF_k;
                    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];

                    // FACE INDICES TO ADJACENT NEIGHBORS INDICES
                    FACE2NBRS(fi, fj, fk, dir, i, j, k, nbr_i, nbr_j, nbr_k);

                    // ELEMENT INFO
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    const bool elm_is_ghost = ELM_IS_GHOST(etype);
                    const bool nbr_is_ghost = ELM_IS_GHOST(nbr_etype);

                    // SELECT TYPE OF CELL BOUNDARY
                    // WALL
                    if ((ff == domain.smallEnd(dir)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // SOLUTION
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, nbr_sol, N_SOL, sNp, fi, fj, fk, FX_fab, offset, IBVP);
                    }
                    // WALL
                    else if ((ff == (domain.bigEnd(dir)+1)) && (is_periodic[dir] == 0))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // SOLUTION
                        BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                        DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                        // COMPUTE THE INTEGRAL
                        BOU_FX_WALL(t, dom, cell_lo, un, bou_Nq, xptr, sol, N_SOL, sNp, fi, fj, fk, FX_fab, 0, IBVP);
                    }
                    // INTRAPHASE
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        if (elm_is_ghost)
                        {
                            // CELL OF THE COARSER LEVEL
                            int c_i, c_j, c_k;
                            FINE_TO_COARSE(i, j, k, c_rr, c_i, c_j, c_k);
                            const short c_etype = c_eType_fab(c_i,c_j,c_k,ELM_TYPE(dom));

                            // SOLUTION
                            BF_CELL(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                            DG_SOL_space_BFX sol(c_sp, BF_lo, BF_hi, c_X_fab, BF_i, BF_j, BF_k);
                            BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, c_sNp, sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE BETWEEN LEVEL AND COARSER LEVEL
                        else if (nbr_is_ghost)
                        {
                            // BASIS FUNCTIONS OF THE COARSER LEVEL
                            int c_nbr_i, c_nbr_j, c_nbr_k;
                            FINE_TO_COARSE(nbr_i, nbr_j, nbr_k, c_rr, c_nbr_i, c_nbr_j, c_nbr_k);
                            const short c_nbr_etype = c_eType_fab(c_nbr_i,c_nbr_j,c_nbr_k,ELM_TYPE(dom));

                            // SOLUTION
                            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                            DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);
                            BF_CELL(&prob_lo[0], &c_dx[0], c_nbr_i, c_nbr_j, c_nbr_k, c_nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            DG_SOL_space_BFX nbr_sol(c_sp, nbr_BF_lo, nbr_BF_hi, c_X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, sNp, c_sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,offset+rs+ru*c_sNp) *= fact;
                            }
                        }
                        // INTRAPHASE WITHIN LEVEL
                        else
                        {
                            // SOLUTION
                            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                            DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);
                            BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, sNp, sNp, fi, fj, fk, FX_fab, offset, IBVP);
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================
}

/**
 * \brief Eval the fluxes at the internal boundaries and add contribution to the time derivatives.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] FX: a MultiFab object that contains the fluxes.
 * \param[in] offset: an integer used for accessing the FX components.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the the state vector
 *                   of the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const ImplicitMesh & mesh,
                          const MatrixFactory & matfactory,
                          const int N_DOM,
                          const int N_SOL,
                          const MultiFab & X,
                          const Array<MultiFab, AMREX_SPACEDIM> & FX,
                          const int offset,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Hyperbolic.H", "Eval_dXdt_BouContrib", N_SOL);
    // ================================================================

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.std_elem.p;
    const int sNp = matfactory.std_elem.Np;
    // ================================================================

    // ADD FLUX CONTRIBUTION: INTERNAL BOUNDARIES =====================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

            if (ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                // NEIGHBOR DOMAIN (IF ANY)
                const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real SOL[__DG_MAX_N_SOL__];
                Real NFn[__DG_MAX_N_SOL__];
                Real integrand;
                
                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // SOLUTION
                DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                if (nbr_dom == -1)
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
                        sol.eval(x, 0, N_SOL, SOL);

                        // NUMERICAL FLUX
                        IBVP.F_NF_PHI_BCS(dom, t, x, un, SOL, NFn);

                        // INTEGRAL CONTRIBUTION
                        for (int ru = 0; ru < N_SOL; ++ru)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            integrand = sol.BF[rs]*NFn[ru];
                            dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                        }
                    }
                }
                else
                {
Print() << "Eval_dXdt_BouContrib - NUMERICAL FLUX - nbr_dom != -1" << std::endl;
exit(-1);
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================

    // ADD FLUX CONTRIBUTION: CELL BOUNDARIES =========================
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;

        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            Array4<Real const> const & FX_fab = FX[dir].array(mfi);

            Array4<Real> const & dXdt_fab = dXdt.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    int fi, fj, fk;

                    GRID_FACE(i, j, k, b+1, fi, fj, fk);

                    for (int ru = 0; ru < N_SOL; ++ru)
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        dXdt_fab(i,j,k,rs+ru*sNp) -= (FX_fab(fi,fj,fk,rs+ru*sNp)+FX_fab(i,j,k,offset+rs+ru*sNp));
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================

    // SHARE INFO =====================================================
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] n_valid_levels: Number of valid levels to be considered.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] Xs: a vector of MultiFab objects that contain the coefficients of the basis functions
 *                for the unknown solution fields.
 * \param[out] dXdt: a vector of MultiFab objects that will contain the time derivative of the
 *                   coefficients of the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const int n_valid_levels,
               const Vector<IntVect> & ref_ratios,
               const Vector<ImplicitMesh *> & meshes,
               const Vector<MatrixFactory *> & matfactories,
               const int N_DOM,
               const int N_SOL,
               const Gpu::ManagedVector<int> & Sol2Dom,
               const Vector<MultiFab> & Xs,
               Vector<MultiFab> & dXdt,
               const IBVP_CLASS & IBVP)
{
    // VARIABLES ======================================================
    int offset;
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_valid_levels);
    Vector<MultiFab> dXs(n_valid_levels);
    // ================================================================

    // INITIALIZATION =================================================
    {
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            dXdt[lev] = 0.0;
        }
            
        // HIGHEST ORDER AMONG LEVELS
        int h_sp;

        h_sp = 0;
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            h_sp = std::max(h_sp, matfactories[lev]->std_elem.p);
        }
        const int h_sNp = AMREX_D_TERM((1+h_sp),*(1+h_sp),*(1+h_sp));

        // OFFSET USED FOR ACCESSING FX
        offset = h_sNp*N_SOL;
        
        // As the number of components for the fluxes at each level, we
        // use the maximum among all the levels. This will facilitate
        // the use of the average_down_faces function.
        const int FX_n_comp = 2*offset;

        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                FX[lev][dim] = MultiFab(meshes[lev]->fc_ba[dim], meshes[lev]->dm, FX_n_comp, 0);
                FX[lev][dim] = 0.0;
            }
        }

        // MULTIFAB STORING SLOPES
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            if (matfactories[lev]->std_elem.p == 0)
            {
                dXs[lev].define(Xs[lev].boxarray, Xs[lev].distributionMap, AMREX_SPACEDIM*N_SOL, 1);
                dXs[lev] = 0.0;
            }
        }
    }
    // ================================================================

    // DOMAIN CONTRIBUTION TO THE TIME DERIVATIVES ====================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        // FINITE-VOLUME APPROACH
        if (matfactories[lev]->std_elem.p == 0)
        {
            // Nothing (for now)
        }
        // DISCONTINUOUS GALERKIN APPROACH
        else
        {
            Eval_dXdt_DomContrib(t, *meshes[lev], *matfactories[lev], N_DOM, N_SOL, Xs[lev], dXdt[lev], IBVP);
        }
    }
    // ================================================================

    // SLOPE RECONSTRUCTION FOR FINITE-VOLUME APPROACHES ==============
    {
        const int lev = n_valid_levels-1;

        if (matfactories[lev]->std_elem.p == 0)
        {
            IBVP.F_R_SLOPES(t, *meshes[lev], N_DOM, N_SOL, Xs[lev], dXs[lev]);

#ifdef AMREX_DEBUG
        if (dXs[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt\n";
            msg += "| The reconstructed slopes contain nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
#endif
        }
    }
    // ================================================================

    // CELL BOUNDARY FLUXES ===========================================
    // LEVEL 0 --------------------------------------------------------
    {
        const int lev = 0;
        
        // FINITE-VOLUME APPROACH (USING RECONSTRUCTED SLOPES)
        if (matfactories[lev]->std_elem.p == 0)
        {
            Eval_dXdt_BouFluxes_FV(t,
                                   *meshes[lev],
                                   N_DOM, N_SOL,
                                   Xs[lev], dXs[lev], FX[lev], offset,
                                   IBVP);
        }
        // DISCONTINUOUS GALERKIN APPROACH
        else
        {
            Eval_dXdt_BouFluxes(t,
                                *meshes[lev], *matfactories[lev],
                                N_DOM, N_SOL,
                                Xs[lev], FX[lev], offset,
                                IBVP);
        }
    }
    // ----------------------------------------------------------------

    // LEVEL > 0 ------------------------------------------------------
    for (int lev = 1; lev < n_valid_levels; ++lev)
    {
        const IntVect c_rr = ref_ratios[lev-1];
        const ImplicitMesh & c_mesh = *meshes[lev-1];
        const shortMultiFab & c_eType = c_mesh.eType;
        const MatrixFactory & c_matfactory = *matfactories[lev-1];
        const int c_sp = c_matfactory.std_elem.p;
        const MultiFab & c_X = Xs[lev-1];

        // FINITE-VOLUME APPROACH (USING RECONSTRUCTED SLOPES)
        if (matfactories[lev]->std_elem.p == 0)
        {
            Eval_dXdt_BouFluxes_FV(t,
                                    c_rr, c_eType, c_sp, c_X,
                                    *meshes[lev],
                                    N_DOM, N_SOL,
                                    Xs[lev], dXs[lev], FX[lev], offset,
                                    IBVP);
        }
        // DISCONTINUOUS GALERKIN APPROACH
        else
        {
            Eval_dXdt_BouFluxes(t,
                                c_rr, c_eType, c_sp, c_X,
                                *meshes[lev], *matfactories[lev],
                                N_DOM, N_SOL,
                                Xs[lev], FX[lev], offset,
                                IBVP);
        }
    }
    // ----------------------------------------------------------------
    // ================================================================

    // AVERAGE DOWN THE FLUXES ========================================
    for (int lev = (n_valid_levels-1); lev > 0; --lev)
    {
        average_down_faces(GetArrOfConstPtrs(FX[lev]), GetArrOfPtrs(FX[lev-1]), ref_ratios[lev-1], meshes[lev-1]->geom);
    }
    // ================================================================

    // BOUNDARY CONTRIBUTION TO THE TIME DERIVATIVES ==================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        // FINITE-VOLUME APPROACH (USING RECONSTRUCTED SLOPES)
        if (matfactories[lev]->std_elem.p == 0)
        {
            Eval_dXdt_BouContrib_FV(t, *meshes[lev], N_DOM, N_SOL, Xs[lev], dXs[lev], FX[lev], offset, dXdt[lev], IBVP);
        }
        // DISCONTINUOUS GALERKIN APPROACH
        else
        {
            Eval_dXdt_BouContrib(t, *meshes[lev], *matfactories[lev], N_DOM, N_SOL, Xs[lev], FX[lev], offset, dXdt[lev], IBVP);
        }
    }
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        AddSmallElementsContribution(*meshes[lev], *matfactories[lev], N_SOL, Sol2Dom, dXdt[lev]);
        MultiplyByInverseMassMatrix(*meshes[lev], *matfactories[lev], N_SOL, Sol2Dom, dXdt[lev]);
    }
    // ================================================================

#ifdef AMREX_DEBUG
    // CHECK COMPUTED TIME DERIVATIVES ================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        if (dXdt[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt\n";
            msg += "| dXdt contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    // ================================================================
#endif
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[inout] Xs: a vector of MultiFab objects that contain the coefficients of the basis
 *                   functions for the unknown solution fields. On exit, the coefficients will be
 *                   updated according to the selected Runge-Kutta algorithm.
 * \param[in] masks: a vector containing the iMultiFab tags objects at different levels.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const Vector<IntVect> & ref_ratios,
                            const Vector<ImplicitMesh *> & meshes,
                            const Vector<MatrixFactory *> & matfactories,
                            const int N_DOM,
                            const int N_SOL,
                            const Gpu::ManagedVector<int> & Sol2Dom,
                            Vector<MultiFab> & Xs,
                            Vector<iMultiFab> & masks,
                            const IBVP_CLASS & IBVP)
{
    
    // PARAMETERS =====================================================
    const int n_levels = meshes.size();
    // ================================================================

    // VARIABLES ======================================================
    int n_valid_levels;
    Vector<MultiFab> Xs_old(n_levels);
    // ================================================================

    // FIND THE NUMBER OF VALID LEVELS ================================
    n_valid_levels = 0;
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if ((meshes[lev] != nullptr) && (matfactories[lev] != nullptr))
        {
            n_valid_levels += 1;
        }
    }
    // ================================================================

    // SWAP NEW AND OLD ===============================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        Xs_old[lev].define(Xs[lev].boxarray, Xs[lev].distributionMap, Xs[lev].n_comp, Xs[lev].n_grow);
        std::swap(Xs_old[lev], Xs[lev]);
    }
    // ================================================================

    // RK1 ============================================================
    if (RK_order == 1)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ================================================================
    // RK2 ============================================================
    else if (RK_order == 2)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_valid_levels);
        Vector<MultiFab> X1(n_valid_levels);
        
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            const MultiFab & X = Xs[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK2 STEP 1 -------------------------------------------------
        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, Xs_old, dXdt, IBVP);

        // X1 := X_old+dt*dXdt
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, dt, dXdt[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt, IBVP);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(Xs[lev], 0.5, Xs_old[lev], 0, 0.5, X1[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev], 0.5*dt, dXdt[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
    // RK3 ============================================================
    else if (RK_order == 3)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_valid_levels);
        Vector<MultiFab> dXdt2(n_valid_levels);
        Vector<MultiFab> dXdt3(n_valid_levels);
        Vector<MultiFab> X1(n_valid_levels);
        
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            const MultiFab & X = Xs[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK3 STEP 1 -------------------------------------------------
        // dXdt1 := dXdt(t, X_old)
        Eval_dXdt(t, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, Xs_old, dXdt1, IBVP);

        // X1 := X_old+dt*dXdt1
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK3 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt2, IBVP);

        // X1 := X_old+(dt/4)*dXdt1+(dt/4)*dXdt2
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, 0.25*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            MultiFab::Saxpy(X1[lev],                        0.25*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK3 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, X1)
        Eval_dXdt(t+0.5*dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt3, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/6)*dXdt2+(2*dt/3)*dXdt3
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(Xs[lev], 1.0, Xs_old[lev], 0,     dt/6.0, dXdt1[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev],                            dt/6.0, dXdt2[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev],                        2.0*dt/3.0, dXdt3[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
    // RK4 ============================================================
    else if (RK_order == 4)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_valid_levels);
        Vector<MultiFab> dXdt2(n_valid_levels);
        Vector<MultiFab> dXdt3(n_valid_levels);
        Vector<MultiFab> dXdt4(n_valid_levels);
        Vector<MultiFab> X1(n_valid_levels);
        
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            const MultiFab & X = Xs[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt4[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK4 STEP 1 -------------------------------------------------
        // dXdt1 := dXdt(t, X_old)
        Eval_dXdt(t, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, Xs_old, dXdt1, IBVP);

        // X1 := X_old+(dt/2)*dXdt1
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, 0.5*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK4 STEP 2 -------------------------------------------------
        // dXdt2 := dXdt(t+dt/2, X1)
        Eval_dXdt(t+0.5*dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt2, IBVP);

        // X1 := X_old+(dt/2)*dXdt2
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, 0.5*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK4 STEP 3 -------------------------------------------------
        // dXdt3 := dXdt(t+dt/2, X1)
        Eval_dXdt(t+0.5*dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt3, IBVP);
        
        // X1 := X_old+dt*dXdt3
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, dt, dXdt3[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        // ------------------------------------------------------------

        // RK4 STEP 4 -------------------------------------------------
        // dXdt4 := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt4, IBVP);

        // X_new := X_old+(dt/6)*dXdt1+(dt/3)*dXdt2+(dt/3)*dXdt3+(dt/6)*dXdt4
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(Xs[lev], 1.0, Xs_old[lev], 0, dt/6.0, dXdt1[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev],                        dt/3.0, dXdt2[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev],                        dt/3.0, dXdt3[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev],                        dt/6.0, dXdt4[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
    // ================================================================
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ================================================================

#ifdef AMREX_DEBUG
    // CHECK SOLUTION =================================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        const MultiFab & X = Xs[lev];

        if (X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    // ================================================================
#endif

    // RESTRICT: PROJECT FINE SOLUTION ONTO THE COARSE LEVEL ==========
    for (int lev = (n_valid_levels-1); lev > 0; --lev)
    {
        AMR::Restrict(N_SOL, Sol2Dom,
                     *meshes[lev], *matfactories[lev], Xs[lev],
                     ref_ratios[lev-1],
                     *meshes[lev-1], *matfactories[lev-1], masks[lev-1], Xs[lev-1]);

        if (Xs[lev-1].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
            msg += "| Solution contains nans at level "+std::to_string(lev-1)+" after restriction.\n";
            Abort(msg);
        }
    }
    // ================================================================

    // CHECK ==========================================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        AMR::CheckMergeLeaking(*meshes[lev], N_DOM, masks[lev]);
    }
    // ================================================================
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] Xs: a vector of MultiFab objects that contain the coefficients of the basis
 *                   functions for the unknown solution fields. On exit, the coefficients will be
 *                   updated according to the selected Runge-Kutta algorithm.
 * \param[in] masks: a vector containing the iMultiFab tags objects at different levels.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const Vector<IntVect> & ref_ratios,
                            const Vector<ImplicitMesh *> & meshes,
                            const Vector<MatrixFactory *> & matfactories,
                            const int N_SOL,
                            Vector<MultiFab> & Xs,
                            Vector<iMultiFab> & masks,
                            const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);

    TakeRungeKuttaTimeStep(RK_order, dt, t,
                           ref_ratios, meshes, matfactories,
                           N_DOM, N_SOL, Sol2Dom,
                           Xs, masks,
                           IBVP);
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object.
 * \param[in] matfactory: the MatrixFactory object.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXdt: the MultiFab objects that will contain the time derivative of the coefficients
 *                   of the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void Eval_Tentative_dXdt_SingleLevel(const Real t,
                                     const ImplicitMesh & mesh,
                                     const MatrixFactory & matfactory,
                                     const int N_DOM,
                                     const int N_SOL,
                                     const Gpu::ManagedVector<int> & Sol2Dom,
                                     const MultiFab & X,
                                     MultiFab & dXdt,
                                     const IBVP_CLASS & IBVP,
                                     const SLOPE_LIMITER & LIMITER)
{
    // VARIABLES ======================================================
    Array<MultiFab, AMREX_SPACEDIM> LFX;
    Array<MultiFab, AMREX_SPACEDIM> FX;
    // ================================================================

    // PARAMETERS =====================================================
    const int sp = matfactory.std_elem.p;
    const int offset = matfactory.std_elem.Np*N_SOL;
    const int FX_n_comp = 2*offset;
    // ================================================================

    // INITIALIZATION =================================================
    dXdt = 0.0;

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        LFX[dim] = MultiFab(LIMITER.mesh.fc_ba[dim], LIMITER.mesh.dm, FX_n_comp, 0);
        LFX[dim] = 0.0;

        FX[dim] = MultiFab(mesh.fc_ba[dim], mesh.dm, FX_n_comp, 0);
        FX[dim] = 0.0;
    }
    // ================================================================

    // DOMAIN CONTRIBUTION TO THE TIME DERIVATIVES ====================
    Eval_dXdt_DomContrib(t, mesh, matfactory, N_DOM, N_SOL, X, dXdt, IBVP);
    // ================================================================

    // CELL BOUNDARY FLUXES ===========================================
    Eval_dXdt_BouFluxes(t, mesh, matfactory, N_DOM, N_SOL, X, FX, offset, IBVP);
    Eval_dXdt_BouFluxes(t, LIMITER.rr, mesh.eType, sp, X,
                        LIMITER.mesh, LIMITER.matfactory,
                        N_DOM, N_SOL,
                        LIMITER.X, LFX, offset,
                        IBVP);
    // ================================================================

    // AVERAGE DOWN THE FLUXES ========================================
    average_down_faces(GetArrOfConstPtrs(LFX), GetArrOfPtrs(FX), LIMITER.rr, mesh.geom);
    // ================================================================

    // BOUNDARY CONTRIBUTION TO THE TIME DERIVATIVES ==================
    Eval_dXdt_BouContrib(t, mesh, matfactory, N_DOM, N_SOL, X, FX, offset, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    MultiplyByInverseMassMatrix(mesh, matfactory, N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] n_valid_levels: Number of valid levels to be considered.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] Xs: a vector of MultiFab objects that contain the coefficients of the basis functions
 *                for the unknown solution fields.
 * \param[out] dXdt: a vector of MultiFab objects that will contain the time derivative of the
 *                   coefficients of the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void Eval_Tentative_dXdt_MultiLevel(const Real /*t*/,
                                    const int /*n_valid_levels*/,
                                    const Vector<IntVect> & /*ref_ratios*/,
                                    const Vector<ImplicitMesh *> & /*meshes*/,
                                    const Vector<MatrixFactory *> & /*matfactories*/,
                                    const int /*N_DOM*/,
                                    const int /*N_SOL*/,
                                    const Gpu::ManagedVector<int> & /*Sol2Dom*/,
                                    const Vector<MultiFab> & /*Xs*/,
                                    Vector<MultiFab> & /*dXdt*/,
                                    const IBVP_CLASS & /*IBVP*/,
                                    const SLOPE_LIMITER & /*LIMITER*/)
{
Print() << "Eval_Tentative_dXdt_MultiLevel" << std::endl;
exit(-1);
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] n_valid_levels: Number of valid levels to be considered.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] Xs: a vector of MultiFab objects that contain the coefficients of the basis functions
 *                for the unknown solution fields.
 * \param[out] dXdt: a vector of MultiFab objects that will contain the time derivative of the
 *                   coefficients of the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void Eval_Tentative_dXdt(const Real t,
                         const int n_valid_levels,
                         const Vector<IntVect> & ref_ratios,
                         const Vector<ImplicitMesh *> & meshes,
                         const Vector<MatrixFactory *> & matfactories,
                         const int N_DOM,
                         const int N_SOL,
                         const Gpu::ManagedVector<int> & Sol2Dom,
                         const Vector<MultiFab> & Xs,
                         Vector<MultiFab> & dXdt,
                         const IBVP_CLASS & IBVP,
                         const SLOPE_LIMITER & LIMITER)
{
    if (n_valid_levels == 1)
    {
        Eval_Tentative_dXdt_SingleLevel(t, *meshes[0], *matfactories[0],
                                        N_DOM, N_SOL, Sol2Dom,
                                        Xs[0], dXdt[0],
                                        IBVP, LIMITER);
    }
    else
    {
        Eval_Tentative_dXdt_MultiLevel(t, n_valid_levels, ref_ratios, meshes, matfactories,
                                       N_DOM, N_SOL, Sol2Dom,
                                       Xs, dXdt,
                                       IBVP, LIMITER);
    }
}

/**
 * \brief Eval the time derivative of the dG coefficients.
 *
 * \param[in] t: time.
 * \param[in] n_valid_levels: Number of valid levels to be considered.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: Number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] Xs: a vector of MultiFab objects that contain the coefficients of the basis functions
 *                for the unknown solution fields.
 * \param[out] dXdt: a vector of MultiFab objects that will contain the time derivative of the
 *                   coefficients of the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void Eval_dXdt(const Real t,
               const int n_valid_levels,
               const Vector<IntVect> & ref_ratios,
               const Vector<ImplicitMesh *> & meshes,
               const Vector<MatrixFactory *> & matfactories,
               const int N_DOM,
               const int N_SOL,
               const Gpu::ManagedVector<int> & Sol2Dom,
               const Vector<MultiFab> & Xs,
               Vector<MultiFab> & dXdt,
               MultiFab & dLXdt,
               const IBVP_CLASS & IBVP,
               const SLOPE_LIMITER & LIMITER)
{
    // VARIABLES ======================================================
    int offset;
    Array<MultiFab, AMREX_SPACEDIM> LFX;
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_valid_levels);
    // ================================================================

    // INITIALIZATION =================================================
    {
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            dXdt[lev] = 0.0;
        }
        dLXdt = 0.0;
            
        // HIGHEST ORDER AMONG LEVELS
        int h_sp;

        h_sp = 0;
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            h_sp = std::max(h_sp, matfactories[lev]->std_elem.p);
        }
        const int h_sNp = AMREX_D_TERM((1+h_sp),*(1+h_sp),*(1+h_sp));

        // OFFSET USED FOR ACCESSING FX
        offset = h_sNp*N_SOL;
        
        // As the number of components for the fluxes at each level, we
        // use the maximum among all the levels. This will facilitate
        // the use of the average_down_faces function.
        const int FX_n_comp = 2*offset;

        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                FX[lev][dim] = MultiFab(meshes[lev]->fc_ba[dim], meshes[lev]->dm, FX_n_comp, 0);
                FX[lev][dim] = 0.0;
            }
        }

        {
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                LFX[dim] = MultiFab(LIMITER.mesh.fc_ba[dim], LIMITER.mesh.dm, FX_n_comp, 0);
                LFX[dim] = 0.0;
            }
        }
    }
    // ================================================================

    // DOMAIN CONTRIBUTION TO THE TIME DERIVATIVES ====================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        Eval_dXdt_DomContrib(t, *meshes[lev], *matfactories[lev], N_DOM, N_SOL, Xs[lev], dXdt[lev], IBVP);
    }

    // LIMITER GRID
    {
        // Nothing
    }
    // ================================================================

    // CELL BOUNDARY FLUXES ===========================================
    // LEVEL 0 --------------------------------------------------------
    {
        const int lev = 0;
        Eval_dXdt_BouFluxes(t, *meshes[lev], *matfactories[lev], N_DOM, N_SOL, Xs[lev], FX[lev], offset, IBVP);
    }
    // ----------------------------------------------------------------
    
    // LEVEL > 0 ------------------------------------------------------
    for (int lev = 1; lev < n_valid_levels; ++lev)
    {
Print() << "Eval_dXdt - More levels to be added" << std::endl;
exit(-1);
    }
    // ----------------------------------------------------------------

    // LIMITER GRID ---------------------------------------------------
    {
        const int lev = n_valid_levels-1;
        const int sp = matfactories[lev]->std_elem.p;

        Eval_dXdt_BouFluxes(t, LIMITER.rr, meshes[lev]->eType, sp, Xs[lev],
                            LIMITER.mesh, LIMITER.matfactory,
                            N_DOM, N_SOL,
                            LIMITER.X, LFX, offset,
                            IBVP);
    }
    // ----------------------------------------------------------------
    // ================================================================

    // AVERAGE DOWN THE FLUXES ========================================
    for (int lev = (n_valid_levels-1); lev > 0; --lev)
    {
        average_down_faces(GetArrOfConstPtrs(FX[lev]), GetArrOfPtrs(FX[lev-1]), ref_ratios[lev-1], meshes[lev-1]->geom);
    }

    // LIMITER GRID
    {
        const int lev = n_valid_levels-1;

        average_down_faces(GetArrOfConstPtrs(LFX), GetArrOfPtrs(FX[lev]), LIMITER.rr, meshes[lev]->geom);
    }
    // ================================================================

    // BOUNDARY CONTRIBUTION TO THE TIME DERIVATIVES ==================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        Eval_dXdt_BouContrib(t, *meshes[lev], *matfactories[lev], N_DOM, N_SOL, Xs[lev], FX[lev], offset, dXdt[lev], IBVP);
    }

    // LIMITER GRID
    {
        Eval_dXdt_BouContrib(t, LIMITER.mesh, LIMITER.matfactory, N_DOM, N_SOL, LIMITER.X, LFX, offset, dLXdt, IBVP);
    }
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        AddSmallElementsContribution(*meshes[lev], *matfactories[lev], N_SOL, Sol2Dom, dXdt[lev]);
        MultiplyByInverseMassMatrix(*meshes[lev], *matfactories[lev], N_SOL, Sol2Dom, dXdt[lev]);
    }

    // LIMITER GRID
    {
        AddSmallElementsContribution(LIMITER.mesh, LIMITER.matfactory, N_SOL, Sol2Dom, dLXdt);
        MultiplyByInverseMassMatrix(LIMITER.mesh, LIMITER.matfactory, N_SOL, Sol2Dom, dLXdt);
    }
    // ================================================================

#ifdef AMREX_DEBUG
    // CHECK COMPUTED TIME DERIVATIVES ================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        if (dXdt[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt\n";
            msg += "| dXdt contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    {
        if (dLXdt.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt\n";
            msg += "| dLXdt contains nans.\n";
            Abort(msg);
        }
    }
    // ================================================================
#endif
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[inout] Xs: a vector of MultiFab objects that contain the coefficients of the basis
 *                   functions for the unknown solution fields. On exit, the coefficients will be
 *                   updated according to the selected Runge-Kutta algorithm.
 * \param[in] masks: a vector containing the iMultiFab tags objects at different levels.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const Vector<IntVect> & ref_ratios,
                            const Vector<ImplicitMesh *> & meshes,
                            const Vector<MatrixFactory *> & matfactories,
                            const int N_DOM,
                            const int N_SOL,
                            const Gpu::ManagedVector<int> & Sol2Dom,
                            Vector<MultiFab> & Xs,
                            Vector<iMultiFab> & masks,
                            const IBVP_CLASS & IBVP,
                            SLOPE_LIMITER & LIMITER)
{
    
    // PARAMETERS =====================================================
    const int n_levels = meshes.size();
    // ================================================================

    // VARIABLES ======================================================
    int n_valid_levels;
    Vector<MultiFab> Xs_old(n_levels);
    MultiFab LX_old;
    // ================================================================

    // FIND THE NUMBER OF VALID LEVELS ================================
    n_valid_levels = 0;
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if ((meshes[lev] != nullptr) && (matfactories[lev] != nullptr))
        {
            n_valid_levels += 1;
        }
    }
    // ================================================================

    // SWAP NEW AND OLD ===============================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        Xs_old[lev].define(Xs[lev].boxarray, Xs[lev].distributionMap, Xs[lev].n_comp, Xs[lev].n_grow);
        std::swap(Xs_old[lev], Xs[lev]);
    }
    {
        LX_old.define(LIMITER.X.boxarray, LIMITER.X.distributionMap, LIMITER.X.n_comp, LIMITER.X.n_grow);
        MultiFab::Copy(LX_old, LIMITER.X, 0, 0, LIMITER.X.n_comp, LIMITER.X.n_grow);
    }
    // ================================================================

    // RK1 ============================================================
    if (RK_order == 1)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ================================================================
    // RK2 ============================================================
    else if (RK_order == 2)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_valid_levels);
        Vector<MultiFab> X1(n_valid_levels);
        MultiFab dLXdt, LX1;

        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            const MultiFab & X = Xs[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }
        {
            const MultiFab & X = LIMITER.X;

            dLXdt.define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            LX1.define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK2 STEP 1 -------------------------------------------------
        // TENTATIVE STEP
        {
            Eval_Tentative_dXdt(t, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, Xs_old, dXdt, IBVP, LIMITER);

            const int lev = n_valid_levels-1;
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, dt, dXdt[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);

            IBVP.F_UPDATE_TROUBLED(t, *meshes[lev], *matfactories[lev], X1[lev], masks[lev]);
            LIMITER.UpdateMask(masks[lev]);
        }

        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, Xs_old, dXdt, dLXdt, IBVP, LIMITER);

        // X1 := X_old+dt*dXdt
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, Xs_old[lev], 0, dt, dXdt[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
        }
        {
            MultiFab::LinComb(LX1, 1.0, LIMITER.X, 0, dt, dLXdt, 0, 0, LIMITER.X.n_comp, LIMITER.X.n_grow);
            std::swap(LX1, LIMITER.X);
        }
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // TENTATIVE STEP
        {   
            Eval_Tentative_dXdt(t+dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt, IBVP, LIMITER);

            const int lev = n_valid_levels-1;
            MultiFab::LinComb(Xs[lev], 0.5, Xs_old[lev], 0, 0.5, X1[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev], 0.5*dt, dXdt[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);

            IBVP.F_UPDATE_TROUBLED(t, *meshes[lev], *matfactories[lev], Xs[lev], masks[lev]);
            LIMITER.UpdateMask(masks[lev]);
        }

        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, n_valid_levels, ref_ratios, meshes, matfactories, N_DOM, N_SOL, Sol2Dom, X1, dXdt, dLXdt, IBVP, LIMITER);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        for (int lev = 0; lev < n_valid_levels; ++lev)
        {
            MultiFab::LinComb(Xs[lev], 0.5, Xs_old[lev], 0, 0.5, X1[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
            MultiFab::Saxpy(Xs[lev], 0.5*dt, dXdt[lev], 0, 0, Xs[lev].n_comp, Xs[lev].n_grow);
        }
        {
            std::swap(LX1, LIMITER.X);
            MultiFab::LinComb(LIMITER.X, 0.5, LX_old, 0, 0.5, LX1, 0, 0, LIMITER.X.n_comp, LIMITER.X.n_grow);
            MultiFab::Saxpy(LIMITER.X, 0.5*dt, dLXdt, 0, 0, LIMITER.X.n_comp, LIMITER.X.n_grow);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
    // ================================================================
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ================================================================

#ifdef AMREX_DEBUG
    // CHECK SOLUTION =================================================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        const MultiFab & X = Xs[lev];

        if (X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    // ================================================================
#endif

    // RESTRICT: PROJECT FINE SOLUTION ONTO THE COARSE LEVEL ==========
    {
        const int lev = n_valid_levels-1;

        AMR::Restrict(N_SOL, Sol2Dom,
                      LIMITER.mesh, LIMITER.matfactory, LIMITER.X,
                      LIMITER.rr,
                      *meshes[lev], *matfactories[lev], masks[lev], Xs[lev]);
    }

    for (int lev = (n_valid_levels-1); lev > 0; --lev)
    {
        AMR::Restrict(N_SOL, Sol2Dom,
                     *meshes[lev], *matfactories[lev], Xs[lev],
                     ref_ratios[lev-1],
                     *meshes[lev-1], *matfactories[lev-1], masks[lev-1], Xs[lev-1]);

#ifdef AMREX_DEBUG
        if (Xs[lev-1].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - TakeRungeKuttaTimeStep\n";
            msg += "| Solution contains nans at level "+std::to_string(lev-1)+" after restriction.\n";
            Abort(msg);
        }
#endif
    }
    // ================================================================

    // RE-INITIATE LIMITER MASK =======================================
    {
        const int lev = n_valid_levels-1;

        IBVP.F_EVAL_TROUBLED(t, *meshes[lev], *matfactories[lev], Xs[lev], masks[lev]);

        LIMITER.Remake(*meshes[lev], *matfactories[lev], Xs[lev], masks[lev]);

        AMR::Interpolate(N_SOL, Sol2Dom, *meshes[lev], *matfactories[lev], Xs[lev],
                         LIMITER.rr,
                         LIMITER.mesh, LIMITER.matfactory, LIMITER.X);
    }
    // ================================================================
}

/**
 * \brief Advance in time using high-order Runge-Kutta schemes.
 *
 * \param[in] RK_order: order of the Runge-Kutta scheme.
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] ref_ratios: a vector containing refinement rations between consecutive levels.
 * \param[in] meshes: a vector containing the ImplicitMesh objects at different levels.
 * \param[in] matfactories: a vector containing the MatrixFactory objects at different levels.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[inout] Xs: a vector of MultiFab objects that contain the coefficients of the basis
 *                   functions for the unknown solution fields. On exit, the coefficients will be
 *                   updated according to the selected Runge-Kutta algorithm.
 * \param[in] masks: a vector containing the iMultiFab tags objects at different levels.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS, typename SLOPE_LIMITER>
void TakeRungeKuttaTimeStep(const int RK_order,
                            const Real dt, const Real t,
                            const Vector<IntVect> & ref_ratios,
                            const Vector<ImplicitMesh *> & meshes,
                            const Vector<MatrixFactory *> & matfactories,
                            const int N_SOL,
                            Vector<MultiFab> & Xs,
                            Vector<iMultiFab> & masks,
                            const IBVP_CLASS & IBVP,
                            SLOPE_LIMITER & LIMITER)
{
    const int N_DOM = 1;
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);

    TakeRungeKuttaTimeStep(RK_order, dt, t,
                           ref_ratios, meshes, matfactories,
                           N_DOM, N_SOL, Sol2Dom,
                           Xs, masks,
                           IBVP,
                           LIMITER);
}
// ####################################################################
// ####################################################################


} // namespace Explicit

} // namespace Hyperbolic
} // namespace DG
} // namespace amrex

#endif