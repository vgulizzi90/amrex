//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution_Hyperbolic.H
 * \brief Contains functions for implicit-mesh discontinuous Galerkin methods for hyperbolic PDEs.
*/

#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <AMReX_DG_Constants.H>
#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

#ifndef AMREX_DG_SOLUTION_HYPERBOLIC_H_
#define AMREX_DG_SOLUTION_HYPERBOLIC_H_

namespace amrex
{
namespace DG
{
namespace Hyperbolic
{

namespace Explicit
{
/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_DOM: number of domains to be considered.
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
                const Gpu::ManagedVector<int> & /*Sol2Dom*/,
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

    // SOLUTIONS-TO-DOMAIN CORRESPONDENCE
    //const int * Sol2Dom_ptr = Sol2Dom.data();

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
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

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
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);
    return Compute_dt(t, mesh, matfactory, N_DOM, N_SOL, Sol2Dom, X, IBVP);
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
 * int_{dVh} V_{,i}^T NFn
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

/*
Print() << "dXdt.sum(0): " << dXdt.sum(0) << std::endl;
Print() << "dXdt.sum(1): " << dXdt.sum(1) << std::endl;
Print() << "dXdt.sum(2): " << dXdt.sum(2) << std::endl;
Print() << "dXdt.sum(3): " << dXdt.sum(3) << std::endl;

Print() << "dXdt.sum(0+4): " << dXdt.sum(0+4) << std::endl;
Print() << "dXdt.sum(1+4): " << dXdt.sum(1+4) << std::endl;
Print() << "dXdt.sum(2+4): " << dXdt.sum(2+4) << std::endl;
Print() << "dXdt.sum(3+4): " << dXdt.sum(3+4) << std::endl;

Print() << "dXdt.sum(0+2*4): " << dXdt.sum(0+2*4) << std::endl;
Print() << "dXdt.sum(1+2*4): " << dXdt.sum(1+2*4) << std::endl;
Print() << "dXdt.sum(2+2*4): " << dXdt.sum(2+2*4) << std::endl;
Print() << "dXdt.sum(3+2*4): " << dXdt.sum(3+2*4) << std::endl;

Print() << "dXdt.sum(0+3*4): " << dXdt.sum(0+3*4) << std::endl;
Print() << "dXdt.sum(1+3*4): " << dXdt.sum(1+3*4) << std::endl;
Print() << "dXdt.sum(2+3*4): " << dXdt.sum(2+3*4) << std::endl;
Print() << "dXdt.sum(3+3*4): " << dXdt.sum(3+3*4) << std::endl;

Print() << "dXdt.sum(0+4*4): " << dXdt.sum(0+4*4) << std::endl;
Print() << "dXdt.sum(1+4*4): " << dXdt.sum(1+4*4) << std::endl;
Print() << "dXdt.sum(2+4*4): " << dXdt.sum(2+4*4) << std::endl;
Print() << "dXdt.sum(3+4*4): " << dXdt.sum(3+4*4) << std::endl;

Print() << "HELLO - Eval_dXdt" << std::endl;
exit(-1);
*/

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
 * \param[out] tags: a charMultiFab object that contains information about which cells are tagged
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
                            charMultiFab & tags,
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
 * \param[out] tags: a charMultiFab object that contains information about which cells are tagged
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
                            const int N_SOL,
                            MultiFab & X_new,
                            const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    const Gpu::ManagedVector<int> Sol2Dom(N_SOL, 0);
    charMultiFab tags(X_new.boxarray, X_new.distributionMap, 1, X_new.n_grow);
    tags = 0;

    TakeRungeKuttaTimeStep(RK_order, dt, t,
                           mesh, matfactory,
                           N_DOM, N_SOL, Sol2Dom,
                           X_new,
                           tags,
                           IBVP);
}

} // namespace Explicit

} // namespace Hyperbolic
} // namespace DG
} // namespace amrex

#endif