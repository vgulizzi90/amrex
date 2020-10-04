// AMReX_DG_RungeKutta.H

#ifndef BL_DG_RK_H_
#define BL_DG_RK_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_SlopeLimiter.H>

namespace amrex
{
namespace DG
{
// RUNGE-KUTTA EXPLICIT: DOMAIN CONTRIB - HYPERBOLIC ##################
// FINITE-VOLUME SCHEMES ##############################################
AMREX_GPU_HOST_DEVICE
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ELEM_RKE_DOM_H_FV(const Real t,
                       const Real * prob_lo, const Real * dx, const Real * J, const Real aJ,
                       const int ru, const int i, const int j, const int k, const short etype,
                       Array4<Real const> const & /*X_fab*/,
                       Array4<int const> const & eDomQuad_Nq_fab,
                       Array4<long const> const & eDomQuad_pos_fab,
                       const Real * eDomQuad_mem_ptr,
                       Array4<Real> const & dXdt_fab,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // DOMAIN
    const int dom = IBVP.F_U2DOM(ru);
    
    // GRID
    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                      prob_lo[1]+(j+0.5)*dx[1],
                                                      prob_lo[2]+(k+0.5)*dx[2])};

    // QUADRATURE INFO
    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
    const long eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
    const Real * xptr = &eDomQuad_mem_ptr[eDomQuad_pos];
    // ================================================================

    // VARIABLES ======================================================
    // BASIS FUNCTIONS
    int BF_i, BF_j, BF_k;
    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
    
    // QUADRATURE
    long qi_pos;
    Real x[AMREX_SPACEDIM], w;
    
    // DG SOLUTION
    Real B[N_U];
    Real integrand;
    // ================================================================

    // GET THE SUPPORT OF THE BASIS FUNCTIONS =========================
    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL =================================
    //
    // int_{Vh} V.T B
    //
    // ================================================================
    for (int qi = 0; qi < dom_Nq; ++qi)
    {
        // QUADRATURE POINT AND WEIGHT
        qi_pos = (AMREX_SPACEDIM+1)*qi;
        AMREX_D_TERM
        (
            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
        )
        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;
        
        // BODY LOAD
        IBVP.F_B(t, x, B);
        
        // FV CONTRIBUTION
        integrand = B[ru];
        dXdt_fab(i,j,k,ru) += integrand*w;
    }
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_DOM_H_FV(const Real t,
                  const ImplicitMesh<N_PHI, N_DOM> & mesh,
                  const MatrixFactory<N_PHI, N_DOM> & /*matfactory*/,
                  const MultiFab & X,
                  MultiFab & dXdt,
                  const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_DOM_H_FV(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            if (elm_is_not_empty)
            {
                ELEM_RKE_DOM_H_FV<N_PHI, N_DOM, N_U>(t, &prob_lo[0], &dx[0], &J[0], aJ,
                                                     ru, i, j, k, etype,
                                                     X_fab,
                                                     eDomQuad_Nq_fab,
                                                     eDomQuad_pos_fab,
                                                     eDomQuad_mem_ptr,
                                                     dXdt_fab,
                                                     IBVP);
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// RUNGE-KUTTA EXPLICIT: BOUNDARY CONTRIB - HYPERBOLIC ################
// FINITE-VOLUME SCHEMES ##############################################
AMREX_GPU_HOST_DEVICE
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ELEM_RKE_BOU_H_FV(const Real t,
                       const Box & domain, const int * is_periodic,
                       const Real * prob_lo, const Real * dx, const Real * J, const Real * bou_aJ,
                       const int ru, const int i, const int j, const int k,
                       Array4<short const> const & eType_fab,
                       Array4<Real const> const & X_fab, Array4<Real const> const & dX_fab,
                       AMREX_D_DECL(Array4<int const> const & eBouQuad_Nq_fab_0,
                                    Array4<int const> const & eBouQuad_Nq_fab_1,
                                    Array4<int const> const & eBouQuad_Nq_fab_2),
                       Array4<int const> const & eIntBouQuad_Nq_fab,
                       AMREX_D_DECL(Array4<long const> const & eBouQuad_pos_fab_0,
                                    Array4<long const> const & eBouQuad_pos_fab_1,
                                    Array4<long const> const & eBouQuad_pos_fab_2),
                       Array4<long const> const & eIntBouQuad_pos_fab,
                       const Real * eBouQuad_mem_ptr,
                       const Real * eIntBouQuad_mem_ptr,
                       Array4<Real> const & dXdt_fab,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // DOMAIN
    const int dom = IBVP.F_U2DOM(ru);
    
    // GRID
    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                      prob_lo[1]+(j+0.5)*dx[1],
                                                      prob_lo[2]+(k+0.5)*dx[2])};

    // ELEMENT INFO
    const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
    const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
    const int merged_b = (elm_is_small) ? (etype/10) : -1;
    // ================================================================

    // VARIABLES ======================================================
    // BASIS FUNCTIONS
    int BF_i, BF_j, BF_k;
    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

    // QUADRATURE
    int bou_Nq;
    long eBouQuad_pos;
    const Real * xptr;
    long qi_pos;
    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];

    // FV SOLUTION
    Real Uh[N_U], nbr_Uh[N_U];
    Real NFn[N_U];
    Real integrand;
    // ================================================================

    // GET THE SUPPORT OF THE BASIS FUNCTIONS =========================
    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
    // ================================================================

    // BASIS FUNCTIONS AND FV SOLUTION ================================
    FV_sol_space_BFX sol(BF_lo, BF_hi, X_fab, dX_fab, BF_i, BF_j, BF_k, -1);
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL: CELL BOUNDARIES ================
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
    {
        // LOCAL PARAMETERS
        const int dir = b/2;

        // LOCAL VARIABLES
        int nbr_i, nbr_j, nbr_k, nbr_b;
        int nbr_BF_i, nbr_BF_j, nbr_BF_k;
        Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
        int fi, fj, fk;
        bool intraphase;

        // UNIT NORMAL
        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
        un[dir] = 2.0*(b%2)-1.0;

        // NEIGHBORING CELL INFO
        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
        const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

        // Conditions for NOT computing the flux contribution:
        // i) the current cell extends to the neighboring cell;
        // ii) the current cell is small and the neighboring cell
        //     extends into it;
        // iii) the neighboring cell is empty (it means that the cell
        //      must be separated by the zero level set of a function;
        //      this case is handled by the INTERPHASE case.
        //
        const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
        const bool nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
        if (cells_are_merged || nbr_is_empty) continue;

        // GET THE SUPPORT OF THE BASIS FUNCTIONS (NEIGHBORING CELL)
        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
        
        // BASIS FUNCTIONS AND FV SOLUTION (NEIGHBORING CELL)
        FV_sol_space_BFX nbr_sol(nbr_BF_lo, nbr_BF_hi, X_fab, dX_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, -1);

        // QUADRATURE AND BASIS FUNCTIONS INFO
        GRID_FACE(i, j, k, b, fi, fj, fk);
        if (dir == 0)
        {
            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fi == domain.smallEnd(dir) || fi == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#if (AMREX_SPACEDIM > 1)
        else if (dir == 1)
        {
            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fj == domain.smallEnd(dir) || fj == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#endif
#if (AMREX_SPACEDIM > 2)
        else
        {
            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fk == domain.smallEnd(dir) || fk == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#endif
        xptr = &eBouQuad_mem_ptr[eBouQuad_pos];

        // INTER-ELEMENT BOUNDARY OR PERIODIC GRID BOUNDARY -----------
        // INTRAPHASE CONDITIONS TO BE ASSIGNED
        if (intraphase)
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                // SOLUTION
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol.eval(x);

                    nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
                    
                    nbr_Uh[cu] = nbr_sol.eval(x);
                }
                
                // NUMERICAL FLUX FOR INTRAPHASE
                IBVP.F_NF_ICS(dom, t, x, un, Uh, nbr_Uh, NFn);

                // FV CONTRIBUTION
                integrand = NFn[ru];
                dXdt_fab(i,j,k,ru) -= integrand*w;
            }
        }
        // ------------------------------------------------------------
        // GRID BOUNDARY ----------------------------------------------
        // BOUNDARY CONDITIONS TO BE ASSIGNED
        else
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                // SOLUTION
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol.eval(x);
                }

                // NUMERICAL FLUX FOR GRID BOUNDARY CONDITIONS
                IBVP.F_NF_BCS(dom, t, x, un, Uh, NFn);

                // FV CONTRIBUTION
                integrand = NFn[ru];
                dXdt_fab(i,j,k,ru) -= integrand*w;
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL: CELL INTERNAL BOUNDARY =========
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    {
        // LOCAL PARAMETERS
        const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

        // QUADRATURE INFO
        bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
        eBouQuad_pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
        xptr = &eIntBouQuad_mem_ptr[eBouQuad_pos];

        // NO NEIGHBORING DOMAIN --------------------------------------
        // BOUNDARY CONDITIONS TO BE ASSIGNED
        if (nbr_dom == -1)
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0];,
                    x[1] = xptr[qi_pos+1];,
                    x[2] = xptr[qi_pos+2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM];

                // UNIT NORMAL
                AMREX_D_TERM
                (
                    un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                    un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                    un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                )

                // SOLUTION
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol.eval(x);
                }
                
                // NUMERICAL FLUX FOR INTERNAL BOUNDARY
                IBVP.F_NF_PHI_BCS(dom, t, x, un, Uh, NFn);

                // FV CONTRIBUTION
                integrand = NFn[ru];
                dXdt_fab(i,j,k,ru) -= integrand*w;
            }
        }
        // ------------------------------------------------------------
        // VALID NEIGHBORING DOMAIN -----------------------------------
        // INTERFACE CONDITIONS TO BE ASSIGNED
        else
        {
            // LOCAL VARIABLES
            int nbr_BF_i, nbr_BF_j, nbr_BF_k;
            Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];

            // NEIGHBORING CELL INFO
            const short nbr_etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);
            
            // GET THE SUPPORT OF THE BASIS FUNCTIONS (NEIGHBORING CELL)
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
        
            // BASIS FUNCTIONS AND FV SOLUTION (NEIGHBORING CELL)
            FV_sol_space_BFX nbr_sol(nbr_BF_lo, nbr_BF_hi, X_fab, dX_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, -1);

            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0];,
                    x[1] = xptr[qi_pos+1];,
                    x[2] = xptr[qi_pos+2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM];

                // UNIT NORMAL
                AMREX_D_TERM
                (
                    un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                    un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                    un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                )

                // SOLUTION
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol.eval(x);

                    nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);

                    nbr_Uh[cu] = nbr_sol.eval(x);
                }

                // NUMERICAL FLUX FOR INTERNAL INTERFACE
                IBVP.F_NF_PHI_ICS(dom, nbr_dom, t, x, un, Uh, nbr_Uh, NFn);

                // FV CONTRIBUTION
                integrand = NFn[ru];
                dXdt_fab(i,j,k,ru) -= integrand*w;
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_BOU_H_FV(const Real t,
                  const ImplicitMesh<N_PHI, N_DOM> & mesh,
                  const MatrixFactory<N_PHI, N_DOM> & /*matfactory*/,
                  const MultiFab & X,
                  const MultiFab & dX,
                  MultiFab & dXdt,
                  const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_BOU_H_FV(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box & domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.is_periodic;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = mesh.bou_aJ;
    
    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // ================================================================
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
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
        Array4<Real const> const & dX_fab = dX.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);
        
        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            if (elm_is_not_empty)
            {
                ELEM_RKE_BOU_H_FV<N_PHI, N_DOM, N_U>(t,
                                                     domain, &is_periodic[0],
                                                     &prob_lo[0], &dx[0], &J[0], &bou_aJ[0],
                                                     ru, i, j, k, eType_fab,
                                                     X_fab, dX_fab,
                                                     AMREX_D_DECL(eBouQuad_Nq_fab_0, eBouQuad_Nq_fab_1, eBouQuad_Nq_fab_2),
                                                     eIntBouQuad_Nq_fab,
                                                     AMREX_D_DECL(eBouQuad_pos_fab_0, eBouQuad_pos_fab_1, eBouQuad_pos_fab_2),
                                                     eIntBouQuad_pos_fab,
                                                     eBouQuad_mem_ptr,
                                                     eIntBouQuad_mem_ptr,
                                                     dXdt_fab,
                                                     IBVP);
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################


// RUNGE-KUTTA EXPLICIT: DOMAIN CONTRIB - HYPERBOLIC ##################
AMREX_GPU_HOST_DEVICE
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ELEM_RKE_DOM_H(const Real t,
                    const Real * prob_lo, const Real * dx, const Real * J, const Real aJ,
                    const int ru, const int i, const int j, const int k, const short etype,
                    Array4<Real const> const & X_fab, const int sp,
                    Array4<int const> const & eDomQuad_Nq_fab,
                    Array4<long const> const & eDomQuad_pos_fab,
                    const Real * eDomQuad_mem_ptr,
                    Array4<Real> const & dXdt_fab,
                    const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // DOMAIN
    const int dom = IBVP.F_U2DOM(ru);
    
    // GRID
    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                      prob_lo[1]+(j+0.5)*dx[1],
                                                      prob_lo[2]+(k+0.5)*dx[2])};

    // QUADRATURE INFO
    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
    const long eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
    const Real * xptr = &eDomQuad_mem_ptr[eDomQuad_pos];

    // DG
    const int sNp = AMREX_D_PICK(1+sp, (1+sp)*(1+sp), (1+sp)*(1+sp)*(1+sp));
    // ================================================================

    // VARIABLES ======================================================
    // BASIS FUNCTIONS
    int BF_i, BF_j, BF_k;
    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
    
    // QUADRATURE
    long qi_pos;
    Real x[AMREX_SPACEDIM], w;
    
    // DG SOLUTION
    Real Uh[N_U];
    Real AMREX_D_DECL(F1h[N_U], F2h[N_U], F3h[N_U]), B[N_U];
    Real integrand;
    // ================================================================

    // GET THE SUPPORT OF THE BASIS FUNCTIONS =========================
    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
    // ================================================================

    // BASIS FUNCTIONS AND DG SOLUTION ================================
    DG_sol_space_BFX sol(BF_lo, BF_hi, X_fab, sp, BF_i, BF_j, BF_k, -1);
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL =================================
    //
    // int_{Vh} V_{,i}.T F_i + int_{Vh} V.T B
    //
    // ================================================================
    for (int qi = 0; qi < dom_Nq; ++qi)
    {
        // QUADRATURE POINT AND WEIGHT
        qi_pos = (AMREX_SPACEDIM+1)*qi;
        AMREX_D_TERM
        (
            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
        )
        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

        // SOLUTION
        sol._eval_BF_table_(x);
        for (int cu = 0; cu < N_U; ++cu)
        {
            sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
            
            Uh[cu] = sol._eval_unsafe_();
        }

        // FLUX
        IBVP.F_F(t, x, Uh, AMREX_D_DECL(F1h, F2h, F3h));
        
        // BODY LOAD
        IBVP.F_B(t, x, B);
        
        // DG CONTRIBUTION
        for (int rs = 0; rs < sNp; ++rs)
        {
            AMREX_D_TERM
            (
                integrand  = sol.dBFdx1[rs]*F1h[ru];,
                integrand += sol.dBFdx2[rs]*F2h[ru];,
                integrand += sol.dBFdx3[rs]*F3h[ru];
            )

            integrand += sol.BF[rs]*B[ru];

            dXdt_fab(i,j,k,rs+ru*sNp) += integrand*w;
        }
    }
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_DOM_H(const Real t,
               const ImplicitMesh<N_PHI, N_DOM> & mesh,
               const MatrixFactory<N_PHI, N_DOM> & matfactory,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_DOM_H(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            if (elm_is_not_empty)
            {
                ELEM_RKE_DOM_H<N_PHI, N_DOM, N_U>(t, &prob_lo[0], &dx[0], &J[0], aJ,
                                                  ru, i, j, k, etype,
                                                  X_fab, sp,
                                                  eDomQuad_Nq_fab,
                                                  eDomQuad_pos_fab,
                                                  eDomQuad_mem_ptr,
                                                  dXdt_fab,
                                                  IBVP);
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

// IN CASE OF TAGGED CELLS
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_DOM_H_TAG(const Real t,
                   const ImplicitMesh<N_PHI, N_DOM> & mesh,
                   const MatrixFactory<N_PHI, N_DOM> & matfactory,
                   const MultiFab & X,
                   const charMultiFab & tags,
                   MultiFab & dXdt,
                   const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_DOM_H_TAG(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<char const> const & tags_fab = tags.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);
            const bool elm_is_not_tagged = (tags_fab(i,j,k,dom) != TAGGED_ELEMENT);

            if (elm_is_not_empty && elm_is_not_tagged)
            {
                ELEM_RKE_DOM_H<N_PHI, N_DOM, N_U>(t, &prob_lo[0], &dx[0], &J[0], aJ,
                                                  ru, i, j, k, etype,
                                                  X_fab, sp,
                                                  eDomQuad_Nq_fab,
                                                  eDomQuad_pos_fab,
                                                  eDomQuad_mem_ptr,
                                                  dXdt_fab,
                                                  IBVP);
            }
        });
        Gpu::synchronize();
    }
    
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// RUNGE-KUTTA EXPLICIT: BOUNDARY CONTRIB - HYPERBOLIC ################
AMREX_GPU_HOST_DEVICE
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ELEM_RKE_BOU_H(const Real t,
                    const Box & domain, const int * is_periodic,
                    const Real * prob_lo, const Real * dx, const Real * J, const Real * bou_aJ,
                    const int ru, const int i, const int j, const int k,
                    Array4<short const> const & eType_fab,
                    Array4<Real const> const & X_fab, const int sp,
                    AMREX_D_DECL(Array4<int const> const & eBouQuad_Nq_fab_0,
                                 Array4<int const> const & eBouQuad_Nq_fab_1,
                                 Array4<int const> const & eBouQuad_Nq_fab_2),
                    Array4<int const> const & eIntBouQuad_Nq_fab,
                    AMREX_D_DECL(Array4<long const> const & eBouQuad_pos_fab_0,
                                 Array4<long const> const & eBouQuad_pos_fab_1,
                                 Array4<long const> const & eBouQuad_pos_fab_2),
                    Array4<long const> const & eIntBouQuad_pos_fab,
                    const Real * eBouQuad_mem_ptr,
                    const Real * eIntBouQuad_mem_ptr,
                    Array4<Real> const & dXdt_fab,
                    const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // DOMAIN
    const int dom = IBVP.F_U2DOM(ru);
    
    // GRID
    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                      prob_lo[1]+(j+0.5)*dx[1],
                                                      prob_lo[2]+(k+0.5)*dx[2])};

    // ELEMENT INFO
    const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
    const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
    const int merged_b = (elm_is_small) ? (etype/10) : -1;

    // DG
    const int sNp = AMREX_D_PICK(1+sp, (1+sp)*(1+sp), (1+sp)*(1+sp)*(1+sp));
    // ================================================================

    // VARIABLES ======================================================
    // BASIS FUNCTIONS
    int BF_i, BF_j, BF_k;
    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

    // QUADRATURE
    int bou_Nq;
    long eBouQuad_pos;
    const Real * xptr;
    long qi_pos;
    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];

    // DG SOLUTION
    Real Uh[N_U], nbr_Uh[N_U];
    Real NFn[N_U];
    Real integrand;
    // ================================================================

    // GET THE SUPPORT OF THE BASIS FUNCTIONS =========================
    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
    // ================================================================

    // BASIS FUNCTIONS AND DG SOLUTION ================================
    DG_sol_space_BFX sol(BF_lo, BF_hi, X_fab, sp, BF_i, BF_j, BF_k, -1);
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL: CELL BOUNDARIES ================
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
    {
        // LOCAL PARAMETERS
        const int dir = b/2;

        // LOCAL VARIABLES
        int nbr_i, nbr_j, nbr_k, nbr_b;
        int nbr_BF_i, nbr_BF_j, nbr_BF_k;
        Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
        int fi, fj, fk;
        bool intraphase;

        // UNIT NORMAL
        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
        un[dir] = 2.0*(b%2)-1.0;

        // NEIGHBORING CELL INFO
        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
        const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

        // Conditions for NOT computing the flux contribution:
        // i) the current cell extends to the neighboring cell;
        // ii) the current cell is small and the neighboring cell
        //     extends into it;
        // iii) the neighboring cell is empty (it means that the cell
        //      must be separated by the zero level set of a function;
        //      this case is handled by the INTERPHASE case.
        //
        const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
        const bool nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
        if (cells_are_merged || nbr_is_empty) continue;

        // GET THE SUPPORT OF THE BASIS FUNCTIONS (NEIGHBORING CELL)
        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
        
        // BASIS FUNCTIONS AND DG SOLUTION (NEIGHBORING CELL)
        DG_sol_space_BFX nbr_sol(nbr_BF_lo, nbr_BF_hi, X_fab, sp, nbr_BF_i, nbr_BF_j, nbr_BF_k, -1);

        // QUADRATURE AND BASIS FUNCTIONS INFO
        GRID_FACE(i, j, k, b, fi, fj, fk);
        if (dir == 0)
        {
            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fi == domain.smallEnd(dir) || fi == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#if (AMREX_SPACEDIM > 1)
        else if (dir == 1)
        {
            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fj == domain.smallEnd(dir) || fj == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#endif
#if (AMREX_SPACEDIM > 2)
        else
        {
            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
            eBouQuad_pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

            if ((fk == domain.smallEnd(dir) || fk == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
            else intraphase = true;
        }
#endif
        xptr = &eBouQuad_mem_ptr[eBouQuad_pos];

        // INTER-ELEMENT BOUNDARY OR PERIODIC GRID BOUNDARY -----------
        // INTRAPHASE CONDITIONS TO BE ASSIGNED
        if (intraphase)
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                // SOLUTION
                sol._eval_BF_only_table_(x);
                nbr_sol._eval_BF_only_table_(x);
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol._eval_unsafe_();

                    nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
                    
                    nbr_Uh[cu] = nbr_sol._eval_unsafe_();
                }
                
                // NUMERICAL FLUX FOR INTRAPHASE
                IBVP.F_NF_ICS(dom, t, x, un, Uh, nbr_Uh, NFn);

                // DG CONTRIBUTION
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = sol.BF[rs]*NFn[ru];
                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                }
            }
        }
        // ------------------------------------------------------------
        // GRID BOUNDARY ----------------------------------------------
        // BOUNDARY CONDITIONS TO BE ASSIGNED
        else
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                // SOLUTION
                sol._eval_BF_only_table_(x);
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol._eval_unsafe_();
                }

                // NUMERICAL FLUX FOR GRID BOUNDARY CONDITIONS
                IBVP.F_NF_BCS(dom, t, x, un, Uh, NFn);

                // DG CONTRIBUTION
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = sol.BF[rs]*NFn[ru];
                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                }
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // COMPUTE THE FOLLOWING INTEGRAL: CELL INTERNAL BOUNDARY =========
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    {
        // LOCAL PARAMETERS
        const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

        // QUADRATURE INFO
        bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
        eBouQuad_pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
        xptr = &eIntBouQuad_mem_ptr[eBouQuad_pos];

        // NO NEIGHBORING DOMAIN --------------------------------------
        // BOUNDARY CONDITIONS TO BE ASSIGNED
        if (nbr_dom == -1)
        {
            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0];,
                    x[1] = xptr[qi_pos+1];,
                    x[2] = xptr[qi_pos+2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM];

                // UNIT NORMAL
                AMREX_D_TERM
                (
                    un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                    un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                    un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                )

                // SOLUTION
                sol._eval_BF_only_table_(x);
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol._eval_unsafe_();
                }
                
                // NUMERICAL FLUX FOR INTERNAL BOUNDARY
                IBVP.F_NF_PHI_BCS(dom, t, x, un, Uh, NFn);

                // DG CONTRIBUTION
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = sol.BF[rs]*NFn[ru];
                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                }
            }
        }
        // ------------------------------------------------------------
        // VALID NEIGHBORING DOMAIN -----------------------------------
        // INTERFACE CONDITIONS TO BE ASSIGNED
        else
        {
            // LOCAL VARIABLES
            int nbr_BF_i, nbr_BF_j, nbr_BF_k;
            Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];

            // NEIGHBORING CELL INFO
            const short nbr_etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);
            
            // GET THE SUPPORT OF THE BASIS FUNCTIONS (NEIGHBORING CELL)
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
        
            // BASIS FUNCTIONS AND DG SOLUTION (NEIGHBORING CELL)
            DG_sol_space_BFX nbr_sol(nbr_BF_lo, nbr_BF_hi, X_fab, sp, nbr_BF_i, nbr_BF_j, nbr_BF_k, -1);

            // COMPUTE THE INTEGRAL
            for (int qi = 0; qi < bou_Nq; ++qi)
            {
                // QUADRATURE POINT AND WEIGHT
                qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0];,
                    x[1] = xptr[qi_pos+1];,
                    x[2] = xptr[qi_pos+2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM];

                // UNIT NORMAL
                AMREX_D_TERM
                (
                    un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                    un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                    un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                )

                // SOLUTION
                sol._eval_BF_only_table_(x);
                nbr_sol._eval_BF_only_table_(x);
                for (int cu = 0; cu < N_U; ++cu)
                {
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                    Uh[cu] = sol._eval_unsafe_();

                    nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);

                    nbr_Uh[cu] = nbr_sol._eval_unsafe_();
                }

                // NUMERICAL FLUX FOR INTERNAL INTERFACE
                IBVP.F_NF_PHI_ICS(dom, nbr_dom, t, x, un, Uh, nbr_Uh, NFn);

                // DG CONTRIBUTION
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = sol.BF[rs]*NFn[ru];
                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                }
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_BOU_H(const Real t,
               const ImplicitMesh<N_PHI, N_DOM> & mesh,
               const MatrixFactory<N_PHI, N_DOM> & matfactory,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_BOU_H(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box & domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.is_periodic;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = mesh.bou_aJ;
    
    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    // ================================================================

    // ================================================================
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
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
        
        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            if (elm_is_not_empty)
            {
                ELEM_RKE_BOU_H<N_PHI, N_DOM, N_U>(t,
                                                  domain, &is_periodic[0],
                                                  &prob_lo[0], &dx[0], &J[0], &bou_aJ[0],
                                                  ru, i, j, k, eType_fab,
                                                  X_fab, sp,
                                                  AMREX_D_DECL(eBouQuad_Nq_fab_0, eBouQuad_Nq_fab_1, eBouQuad_Nq_fab_2),
                                                  eIntBouQuad_Nq_fab,
                                                  AMREX_D_DECL(eBouQuad_pos_fab_0, eBouQuad_pos_fab_1, eBouQuad_pos_fab_2),
                                                  eIntBouQuad_pos_fab,
                                                  eBouQuad_mem_ptr,
                                                  eIntBouQuad_mem_ptr,
                                                  dXdt_fab,
                                                  IBVP);
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################


// ####################################################################
// RUNGE-KUTTA CLASS ##################################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct RungeKutta
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // RUNGE-KUTTA MULTISTEPS
    Array<MultiFab, DG_MAX_RK_ORDER> X, dXdt;
    // ################################################################

    // CONSTRUCTOR ####################################################
    RungeKutta()
    {}
    // ################################################################

    // DESTRUCTOR #####################################################
    ~RungeKutta()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void Define(const BoxArray & cc_ba, const DistributionMapping & dm, const int n_comp, const int n_steps)
    {
        // RUNGE-KUTTA MULTISTEPS =====================================
        for (int k = 0; k < n_steps; ++k)
        {
            this->X[k].define(cc_ba, dm, n_comp, this->dom_data_ngr);
            this->X[k] = 0.0;

            this->dXdt[k].define(cc_ba, dm, n_comp, this->dom_data_ngr);
            this->dXdt[k] = 0.0;
        }
        // ============================================================
    }
    // ################################################################

    // TAKE TIME STEP #################################################
    template<typename IBVP_CLASS>
    void TakeTimeStep(const int RK_order,
                      const Real dt, const Real t,
                      const ImplicitMesh<N_PHI, N_DOM> & mesh,
                      const MatrixFactory<N_PHI, N_DOM> & matfactory,
                      const MultiFab & X_old,
                      MultiFab & X_new,
                      const std::string & PDEs_info,
                      const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int ngr = 1;
        const int sNp = matfactory.cube.Np;
        const int X_n_comp = sNp*N_U;
        // ============================================================

        // UPDATE ACCORDING TO SELECTED RUNGE-KUTTA ORDER =============
        // RK1 --------------------------------------------------------
        if (RK_order == 1)
        {
            // Step 1
            this->Eval_dXdt(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new, dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
        }
        // ------------------------------------------------------------
        // RK2 --------------------------------------------------------
        else if (RK_order == 2)
        {
            // Step 1
            this->Eval_dXdt(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            // Step 2
            this->Eval_dXdt(t+dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);
        }
        // ------------------------------------------------------------
        // RK3 --------------------------------------------------------
        else if (RK_order == 3)
        {
            // Step 1
            this->Eval_dXdt(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            // Step 2
            this->Eval_dXdt(t+dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            // Step 3
            this->Eval_dXdt(t+0.5*dt, mesh, matfactory, this->X[2], this->dXdt[2], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 2.0*dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
        }
        // ------------------------------------------------------------
        // RK4 --------------------------------------------------------
        else if (RK_order == 4)
        {
            // Step 1
            this->Eval_dXdt(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            // Step 2
            this->Eval_dXdt(t+0.5*dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            // Step 3
            this->Eval_dXdt(t+0.5*dt, mesh, matfactory, this->X[2], this->dXdt[2], PDEs_info, IBVP);

            MultiFab::Copy(this->X[3], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[3], dt, this->dXdt[2], 0, 0, X_n_comp, ngr);

            // Step 4
            this->Eval_dXdt(t+dt, mesh, matfactory, this->X[3], this->dXdt[3], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[3], 0, 0, X_n_comp, ngr);
        }
        // ------------------------------------------------------------
        // ------------------------------------------------------------
        else
        {
            Print() << "ERROR: AMReX_DG_RK.H - RungeKutta::TakeTimeStep" << std::endl;
            Print() << "| Unexpected order: " << RK_order << std::endl;
            exit(-1);
        }
        // ============================================================
    }
    // ################################################################

    // TIME-DERIVATIVE EVALUATION #####################################
    template<typename IBVP_CLASS>
    void Eval_dXdt(const Real t,
                   const ImplicitMesh<N_PHI, N_DOM> & mesh,
                   const MatrixFactory<N_PHI, N_DOM> & matfactory,
                   const MultiFab & _X_,
                   MultiFab & _dXdt_,
                   const std::string & PDEs_info,
                   const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("RungeKutta::Eval_dXdt(const Real, ....)");
        // ------------------------------------------------------------

        // INITIALIZATION =============================================
        _dXdt_ = 0.0;
        // ============================================================

        // DOMAIN AND BOUNDARY CONTRIBUTION ===========================
        if (PDEs_info.compare("Hyperbolic") == 0)
        {
            RKE_DOM_H<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, _dXdt_, IBVP);
            RKE_BOU_H<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, _dXdt_, IBVP);
        }
        else
        {
            Print() << "ERROR: AMReX_DG_RungeKutta.H - RungeKutta::Eval_dXdt" << std::endl;
            Print() << "| Unexpected type of PDEs: " << PDEs_info << std::endl;
            exit(-1);
        }
        // ============================================================

        // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX 
        AddSmallElementsContribution<N_PHI, N_DOM, N_U>(mesh, matfactory, _dXdt_, IBVP);
        MultiplyByInverseMassMatrix<N_PHI, N_DOM, N_U>(mesh, matfactory, _dXdt_, IBVP);
        // ============================================================
    }
    // ################################################################

    // TAKE TIME STEP WITH SLOPE LIMITER ##############################
    template<typename IBVP_CLASS>
    void SlopeLimiterSteps_DG(const Real t,
                              const ImplicitMesh<N_PHI, N_DOM> & mesh,
                              const MatrixFactory<N_PHI, N_DOM> & matfactory,
                              MultiFab & _X_,
                              charMultiFab & tags,
                              const IBVP_CLASS & IBVP)
    {
         // PARAMETERS =================================================
        const int ngr = 1;
        // ============================================================

        // VARIABLES ==================================================
        MultiFab Xa(mesh.cc_ba, mesh.dm, N_U, ngr);
        // ============================================================

        // EVAL THE AVERAGES THAT WILL BE USED AS LIMITERS ============
        EvalAverages<N_PHI, N_DOM, N_U>(mesh,
                                        matfactory,
                                        _X_,
                                        Xa,
                                        IBVP);
        FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, Xa, IBVP);
        // ============================================================

        // IDENTIFY THE TROUBLED CELLS ================================
        IdentifyTroubledElements<N_PHI, N_DOM, N_U>(t,
                                                    mesh,
                                                    matfactory,
                                                    _X_,
                                                    tags,
                                                    IBVP);
        // ============================================================

        // APPLY THE LIMITER ==========================================
        ApplySlopeLimiter_DG<N_PHI, N_DOM, N_U>(t,
                                                mesh,
                                                matfactory,
                                                Xa,
                                                tags,
                                                _X_,
                                                IBVP);
        // ============================================================
    }
    
    template<typename IBVP_CLASS>
    void TakeTimeStepWithSlopeLimiter(const int RK_order,
                                      const Real dt, const Real t,
                                      const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                      const MatrixFactory<N_PHI, N_DOM> & matfactory,
                                      const MultiFab & X_old,
                                      charMultiFab & tags,
                                      MultiFab & X_new,
                                      const std::string & PDEs_info,
                                      const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int ngr = 1;
        const int sNp = matfactory.cube.Np;
        const int X_n_comp = sNp*N_U;
        // ============================================================

        // VARIABLES ==================================================
        MultiFab Xa;
        // ============================================================

        // INIT MEMORY ================================================
        if (sNp != 1)
        {
            Xa.define(mesh.cc_ba, mesh.dm, N_U, ngr);
        }
        // ============================================================

        // UPDATE ACCORDING TO SELECTED RUNGE-KUTTA ORDER =============
        // RK2 --------------------------------------------------------
        if (RK_order == 2)
        {
            // Step 1
            this->Eval_dXdt_SL(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[1], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     this->X[1],
                                     tags,
                                     IBVP);
            }

            // Step 2
            this->Eval_dXdt_SL(t+dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t+dt, mesh, X_new, IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t+dt,
                                     mesh,
                                     matfactory,
                                     X_new,
                                     tags,
                                     IBVP);
            }
        }
        // ------------------------------------------------------------
        // RK3 --------------------------------------------------------
        else if (RK_order == 3)
        {
            // Step 1
            this->Eval_dXdt_SL(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[1], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     this->X[1],
                                     tags,
                                     IBVP);
            }

            // Step 2
            this->Eval_dXdt_SL(t+dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.25*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[2], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t+dt,
                                     mesh,
                                     matfactory,
                                     this->X[2],
                                     tags,
                                     IBVP);
            }

            // Step 3
            this->Eval_dXdt_SL(t+0.5*dt, mesh, matfactory, this->X[2], this->dXdt[2], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new,     dt/6.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, 2.0*dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, X_new, IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t+0.5*dt,
                                     mesh,
                                     matfactory,
                                     X_new,
                                     tags,
                                     IBVP);
            }
        }
        // ------------------------------------------------------------
        // RK4 --------------------------------------------------------
        else if (RK_order == 4)
        {
            // Step 1
            this->Eval_dXdt(t, mesh, matfactory, X_old, this->dXdt[0], PDEs_info, IBVP);

            MultiFab::Copy(this->X[1], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[1], 0.5*dt, this->dXdt[0], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[1], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     this->X[1],
                                     tags,
                                     IBVP);
            }

            // Step 2
            this->Eval_dXdt(t+0.5*dt, mesh, matfactory, this->X[1], this->dXdt[1], PDEs_info, IBVP);

            MultiFab::Copy(this->X[2], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[2], 0.5*dt, this->dXdt[1], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[2], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     this->X[2],
                                     tags,
                                     IBVP);
            }

            // Step 3
            this->Eval_dXdt(t+0.5*dt, mesh, matfactory, this->X[2], this->dXdt[2], PDEs_info, IBVP);

            MultiFab::Copy(this->X[3], X_old, 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(this->X[3], dt, this->dXdt[2], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, this->X[3], IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     this->X[3],
                                     tags,
                                     IBVP);
            }

            // Step 4
            this->Eval_dXdt(t+dt, mesh, matfactory, this->X[3], this->dXdt[3], PDEs_info, IBVP);

            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[0], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[1], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/3.0, this->dXdt[2], 0, 0, X_n_comp, ngr);
            MultiFab::Saxpy(X_new, dt/6.0, this->dXdt[3], 0, 0, X_n_comp, ngr);

            if (sNp == 1)
            {
                FillInvalidCells<N_PHI, N_DOM, N_U>(t, mesh, X_new, IBVP);
            }
            else
            {
                SlopeLimiterSteps_DG(t,
                                     mesh,
                                     matfactory,
                                     X_new,
                                     tags,
                                     IBVP);
            }
        }
        // ------------------------------------------------------------
        else
        {
            Print() << "ERROR: AMReX_DG_RK.H - RungeKutta::TakeTimeStepWithSlopeLimiter" << std::endl;
            Print() << "| Unexpected order: " << RK_order << std::endl;
            exit(-1);
        }
        // ============================================================
    }
    // ################################################################

    // TIME-DERIVATIVE EVALUATION WITH SLOPE LIMITER ##################
    template<typename IBVP_CLASS>
    void Eval_dXdt_SL(const Real t,
                      const ImplicitMesh<N_PHI, N_DOM> & mesh,
                      const MatrixFactory<N_PHI, N_DOM> & matfactory,
                      const MultiFab & _X_,
                      MultiFab & _dXdt_,
                      const std::string & PDEs_info,
                      const IBVP_CLASS & IBVP)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("RungeKutta::Eval_dXdt_SL(const Real, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        const int sNp = matfactory.cube.Np;
        // ============================================================

        // INITIALIZATION =============================================
        _dXdt_ = 0.0;
        // ============================================================

        // FV SCHEMES =================================================
        if (sNp == 1)
        {
            // EVAL LIMITED SLOPES
            MultiFab dX(mesh.cc_ba, mesh.dm, AMREX_SPACEDIM*N_U, this->dom_data_ngr);
            ApplySlopeLimiter_FV<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, dX, IBVP);

            if (PDEs_info.compare("Hyperbolic") == 0)
            {
                RKE_DOM_H_FV<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, _dXdt_, IBVP);
                RKE_BOU_H_FV<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, dX, _dXdt_, IBVP);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_RungeKutta.H - RungeKutta::Eval_dXdt_SL\n";
                msg += "| Unexpected type of PDEs: "+PDEs_info+".\n";
                amrex::Abort(msg);
            }
        }
        // DG =========================================================
        else
        {
            if (PDEs_info.compare("Hyperbolic") == 0)
            {
                RKE_DOM_H<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, _dXdt_, IBVP);
                RKE_BOU_H<N_PHI, N_DOM, N_U>(t, mesh, matfactory, _X_, _dXdt_, IBVP);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_RungeKutta.H - RungeKutta::Eval_dXdt_SL\n";
                msg += "| Unexpected type of PDEs: "+PDEs_info+".\n";
                amrex::Abort(msg);
            }
        }
        // ============================================================

        // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX
        AddSmallElementsContribution<N_PHI, N_DOM, N_U>(mesh, matfactory, _dXdt_, IBVP);
        MultiplyByInverseMassMatrix<N_PHI, N_DOM, N_U>(mesh, matfactory, _dXdt_, IBVP);
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif

/*
ExportSolution_VTK<N_PHI, N_DOM, N_U>("./IBVP_2d/PROBLEM_SodsTube_64x64_p1/", "Aux",
                                              0, 2, t,
                                              mesh,
                                              matfactory,
                                              this->X[1],
                                              tags,
                                              {0, AMREX_D_DECL(0, 0, 0), 0, 0},
                                              {"rho", AMREX_D_DECL("ux", "uy", "uz"), "M", "p"},
                                              IBVP);

ExportSolution_VTK<N_PHI, N_DOM, N_U>("./IBVP_2d/PROBLEM_SodsTube_64x64_p1/", "AuxLimited",
                                              0, 2, t,
                                              limiter.mesh,
                                              limiter.matfactory,
                                              limiter.X_new,
                                              limiter.tags,
                                              {0, AMREX_D_DECL(0, 0, 0), 0, 0},
                                              {"rho", AMREX_D_DECL("ux", "uy", "uz"), "M", "p"},
                                              IBVP);


if ((i == 10 && (j == 7 || j == 9) && k ==  7 && ru == 0) ||
    (i ==  7 && (j == 7 || j == 9) && k == 10 && ru == 0))
{
    for (int cu = 0; cu < N_U; ++cu)
    {
        sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
        Uh[cu] = sol._eval_unsafe_();
    }

Print() << "ELEM_RKE_BOU_H (" << i << "," << j << "," << k << "), Uh: "; DG_utils::PrintRealArray2D(1, N_U, Uh);
}

if (((i == 10 && (j == 7 || j == 9) && k ==  7 && ru == 0) ||
     (i ==  7 && (j == 7 || j == 9) && k == 10 && ru == 0)) && (qi == 0))
{
    for (int cu = 0; cu < N_U; ++cu)
    {
        nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
        nbr_Uh[cu] = nbr_sol._eval_unsafe_();
    }

Print() << " - nbr (" << nbr_i << "," << nbr_j << "," << nbr_k << ") nbr_etype: " << nbr_etype << " (" << (nbr_is_small ? "small" : "not small") << "), Uh: "; DG_utils::PrintRealArray2D(1, N_U, nbr_Uh);
}


//if ((i == 10 && (j == 7 || j == 9) && k ==  7 && ru == 0) ||
//    (i ==  7 && (j == 7 || j == 9) && k == 10 && ru == 0))
if (((i == 9 && k == 7) || (i == 7 && k == 9)) && (j == 9 || j == 7) && (ru == 0))
{
    for (int cu = 0; cu < N_U; ++cu)
    {
        sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
        Uh[cu] = sol._eval_unsafe_();
    }

Print() << "ELEM_RKE_DOM_H (" << i << "," << j << "," << k << "), Uh: "; DG_utils::PrintRealArray2D(1, N_U, Uh);
}

*/


/*
// IN CASE OF A (S)LOPE (L)IMITER DEFINED ON A FINER GRID
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void RKE_BOU_H_SL_AMR(const Real t,
                  const ImplicitMesh<N_PHI, N_DOM> & mesh,
                  const MatrixFactory<N_PHI, N_DOM> & matfactory,
                  const MultiFab & X,
                  const charMultiFab & tags,
                  const IntVect rr,
                  const ImplicitMesh<N_PHI, N_DOM> & fine_mesh,
                  const MatrixFactory<N_PHI, N_DOM> & fine_matfactory,
                  const MultiFab & fine_X,
                  MultiFab & dXdt,
                  const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("RKE_BOU_H_SL_AMR(const Real &, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box & domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.is_periodic;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = mesh.bou_aJ;

    const GpuArray<Real, AMREX_SPACEDIM> fine_dx = fine_mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> fine_prob_lo = fine_mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> fine_J = fine_mesh.J;
    const GpuArray<Real, AMREX_SPACEDIM> fine_bou_aJ = fine_mesh.bou_aJ;
    
    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();

    const Real * fine_eBouQuad_mem_ptr = fine_mesh.eBouQuad_device_mem.data();
#else
    const Real * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    const Real * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();

    const Real * fine_eBouQuad_mem_ptr = fine_mesh.eBouQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    const int sNp = matfactory.cube.Np;

    const int fine_sp = fine_matfactory.cube.p;
    // ================================================================

    // ================================================================
    //
    // int_{dVh} [[V.T]]_{i} NF_i
    //
    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        // COARSE MESH
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<char const> const & tags_fab = tags.array(mfi);
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
        
        // FINE MESH
        Array4<short const> const & fine_eType_fab = fine_mesh.eType.array(mfi);
        AMREX_D_TERM
        (
            Array4<int const> const & fine_eBouQuad_Nq_fab_0 = fine_mesh.eBouQuad_Nq[0].array(mfi);
            Array4<long const> const & fine_eBouQuad_pos_fab_0 = fine_mesh.eBouQuad_pos[0].array(mfi);,
            Array4<int const> const & fine_eBouQuad_Nq_fab_1 = fine_mesh.eBouQuad_Nq[1].array(mfi);
            Array4<long const> const & fine_eBouQuad_pos_fab_1 = fine_mesh.eBouQuad_pos[1].array(mfi);,
            Array4<int const> const & fine_eBouQuad_Nq_fab_2 = fine_mesh.eBouQuad_Nq[2].array(mfi);
            Array4<long const> const & fine_eBouQuad_pos_fab_2 = fine_mesh.eBouQuad_pos[2].array(mfi);
        )
        
        Array4<Real const> const & fine_X_fab = fine_X.array(mfi);
        
        Array4<Real> const & dXdt_fab = dXdt.array(mfi);
        
        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);
            const bool elm_is_not_tagged = (tags_fab(i,j,k,dom) != TAGGED_ELEMENT);

            if (elm_is_not_empty && elm_is_not_tagged)
            {
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                  prob_lo[1]+(j+0.5)*dx[1],
                                                                  prob_lo[2]+(k+0.5)*dx[2])};
                const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real Uh[N_U];
                Real NFn[N_U];
                Real integrand;

                // LOCAL VARIABLES (NEIGHBORING CELL)
                bool intraphase;
                short nbr_etype;
                bool nbr_is_small, nbr_is_not_tagged, cells_are_merged, nbr_is_empty;
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
                Real nbr_Uh[N_U];

                // QUADRATURE INFO
                int fi, fj, fk;
                int bou_Nq;
                long eBouQuad_pos, qi_pos;
                const Real * xptr;

                // Get the support of the basis functions (for the
                // current cell)
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // Data structures used to compute the basis functions
                // and the unknown fields for the current cell
                DG_sol_space_BFX sol(BF_lo, BF_hi, X_fab, sp, BF_i, BF_j, BF_k, -1);

                // CELL BOUNDARIES
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    const int dir = b/2;

                    // UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = 2.0*(b%2)-1.0;

                    // Neighboring cell info
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    nbr_is_not_tagged = (tags_fab(nbr_i, nbr_j, nbr_k,dom) != TAGGED_ELEMENT);
                    
                    // Conditions for NOT computing the flux
                    // contribution:
                    // i) the current cell extends to the neighboring
                    //    cell;
                    // ii) the current cell is small and the
                    //     neighboring cell extends into it;
                    // iii) the neighboring cell is empty (it means
                    //      that the cells must be separated by the
                    //      zero level set of a function; this case is
                    //      handled in the INTERPHASE case
                    //
                    cells_are_merged = (nbr_merged_b == nbr_b);
                    cells_are_merged = cells_are_merged || (merged_b == b);
                    nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                    if (cells_are_merged || nbr_is_empty) continue;

                    // INTERFACE BETWEEN ELEMENTS OF SAME GRID --------
                    if (nbr_is_not_tagged)
                    {
                        // Data structure used to compute the basis
                        // functions and the unknown fields for the
                        // neighboring cell
                        DG_sol_space_BFX nbr_sol(X_fab, sp);

                        // Get the support of the basis functions (for
                        // the neighboring cell)
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        nbr_sol.set_support(nbr_BF_lo, nbr_BF_hi);

                        // QUADRATURE AND BASIS FUNCTIONS INFO
                        GRID_FACE(i, j, k, b, fi, fj, fk);
                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            eBouQuad_pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                            if ((fi == domain.smallEnd(dir) || fi == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                            else intraphase = true;
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            eBouQuad_pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                            if ((fj == domain.smallEnd(dir) || fj == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                            else intraphase = true;
                        }
#endif
#if (AMREX_SPACEDIM > 2)
                        else
                        {
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                            eBouQuad_pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));

                            if ((fk == domain.smallEnd(dir) || fk == domain.bigEnd(dir)+1) && is_periodic[dir] == 0) intraphase = false;
                            else intraphase = true;
                        }
#endif
                        xptr = &eBouQuad_mem_ptr[eBouQuad_pos];

                        // INTER-ELEMENT OR PERIODIC GRID BOUNDARY
                        // INTRAPHASE CONDITIONS TO BE ASSIGNED
                        if (intraphase)
                        {
                            // COMPUTE THE INTEGRAL
                            for (int qi = 0; qi < bou_Nq; ++qi)
                            {
                                // QUADRATURE POINT AND WEIGHT
                                qi_pos = (AMREX_SPACEDIM+1)*qi;
                                AMREX_D_TERM
                                (
                                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                                )
                                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                                // SOLUTION
                                sol._eval_BF_only_table_(x);
                                nbr_sol._eval_BF_only_table_(x);
                                for (int cu = 0; cu < N_U; ++cu)
                                {
                                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                    Uh[cu] = sol._eval_unsafe_();

                                    nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
                                    
                                    nbr_Uh[cu] = nbr_sol._eval_unsafe_();
                                }
                                
                                // NUMERICAL FLUX FOR INTRAPHASE
                                IBVP.F_NF_ICS(dom, t, x, un, Uh, nbr_Uh, NFn);

                                // DG CONTRIBUTION
                                for (int rs = 0; rs < sNp; ++rs)
                                {
                                    integrand = sol.BF[rs]*NFn[ru];
                                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                                }
                            }
                        }
                        // GRID BOUNDARY
                        // BOUNDARY CONDITIONS TO BE ASSIGNED
                        else
                        {
                            // COMPUTE THE INTEGRAL
                            for (int qi = 0; qi < bou_Nq; ++qi)
                            {
                                // QUADRATURE POINT AND WEIGHT
                                qi_pos = (AMREX_SPACEDIM+1)*qi;
                                AMREX_D_TERM
                                (
                                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                                )
                                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[dir];

                                // SOLUTION
                                sol._eval_BF_only_table_(x);
                                for (int cu = 0; cu < N_U; ++cu)
                                {
                                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                    Uh[cu] = sol._eval_unsafe_();
                                }

                                // NUMERICAL FLUX FOR GRID BOUNDARY CONDITIONS
                                IBVP.F_NF_BCS(dom, t, x, un, Uh, NFn);

                                // DG CONTRIBUTION
                                for (int rs = 0; rs < sNp; ++rs)
                                {
                                    integrand = sol.BF[rs]*NFn[ru];
                                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                                }
                            }
                        }
                    }
                    // THE NEIGHBORING ELEMENT BELONGS TO A FINER GRID
                    else
                    {
                        // Bounds for the cells to loop over
                        Dim3 fine_lo, fine_hi;
                        COARSE_TO_FINE_GRID_BOUNDS(i, j, k, b, rr, fine_lo, fine_hi);

                        // Data structure used to compute the basis
                        // functions and the unknown fields for the
                        // neighboring cell
                        DG_sol_space_BFX fine_sol(fine_X_fab, fine_sp);

                        for (int fine_k = fine_lo.z; fine_k <= fine_hi.z; ++fine_k)
                        for (int fine_j = fine_lo.y; fine_j <= fine_hi.y; ++fine_j)
                        for (int fine_i = fine_lo.x; fine_i <= fine_hi.x; ++fine_i)
                        {
                            // Neighboring cell info
                            const Real fine_cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(fine_prob_lo[0]+(fine_i+0.5)*fine_dx[0],
                                                                                   fine_prob_lo[1]+(fine_j+0.5)*fine_dx[1],
                                                                                   fine_prob_lo[2]+(fine_k+0.5)*fine_dx[2])};

                            nbr_etype = fine_eType_fab(fine_i,fine_j,fine_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                            nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);

                            if (nbr_is_empty) continue;

                            // Get the support of the basis functions
                            // (for the fine neighboring cell)
                            BF_CELL(&fine_prob_lo[0], &fine_dx[0], fine_i, fine_j, fine_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            fine_sol.set_support(nbr_BF_lo, nbr_BF_hi);
                            
                            // QUADRATURE AND BASIS FUNCTIONS INFO
                            GRID_FACE(fine_i, fine_j, fine_k, nbr_b, fi, fj, fk);
                            if (dir == 0)
                            {
                                bou_Nq = fine_eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                                eBouQuad_pos = fine_eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                            }
#if (AMREX_SPACEDIM > 1)
                            else if (dir == 1)
                            {
                                bou_Nq = fine_eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                                eBouQuad_pos = fine_eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                            }
#endif
#if (AMREX_SPACEDIM > 2)
                            else
                            {
                                bou_Nq = fine_eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                                eBouQuad_pos = fine_eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(nbr_b%2));
                            }
#endif
                            xptr = &fine_eBouQuad_mem_ptr[eBouQuad_pos];

                            // COMPUTE THE INTEGRAL
                            for (int qi = 0; qi < bou_Nq; ++qi)
                            {
                                // QUADRATURE POINT AND WEIGHT
                                qi_pos = (AMREX_SPACEDIM+1)*qi;
                                AMREX_D_TERM
                                (
                                    x[0] = xptr[qi_pos+0]; x[0] = fine_J[0]*x[0]+fine_cell_m[0];,
                                    x[1] = xptr[qi_pos+1]; x[1] = fine_J[1]*x[1]+fine_cell_m[1];,
                                    x[2] = xptr[qi_pos+2]; x[2] = fine_J[2]*x[2]+fine_cell_m[2];
                                )
                                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*fine_bou_aJ[dir];

                                // SOLUTION
                                sol._eval_BF_only_table_(x);
                                fine_sol._eval_BF_only_table_(x);
                                for (int cu = 0; cu < N_U; ++cu)
                                {
                                    sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                    Uh[cu] = sol._eval_unsafe_();

                                    fine_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);
                                    
                                    nbr_Uh[cu] = fine_sol._eval_unsafe_();
                                }
                                
                                // NUMERICAL FLUX FOR INTRAPHASE
                                IBVP.F_NF_ICS(dom, t, x, un, Uh, nbr_Uh, NFn);

                                // DG CONTRIBUTION
                                for (int rs = 0; rs < sNp; ++rs)
                                {
                                    integrand = sol.BF[rs]*NFn[ru];
                                    dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                                }
                            }
                        }
                    }
                    // ------------------------------------------------
                }

                // INTERNAL BOUNDARY
                {
                    // Neighboring domain info
                    const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                    // QUADRATURE INFO
                    bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    eBouQuad_pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    xptr = &eIntBouQuad_mem_ptr[eBouQuad_pos];

                    // NO NEIGHBORING DOMAIN
                    // BOUNDARY CONDITIONS TO BE ASSIGNED
                    if (nbr_dom == -1)
                    {
                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0];,
                                x[1] = xptr[qi_pos+1];,
                                x[2] = xptr[qi_pos+2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM];

                            // UNIT NORMAL
                            AMREX_D_TERM
                            (
                                un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                            )

                            // SOLUTION
                            sol._eval_BF_only_table_(x);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();
                            }
                            
                            // NUMERICAL FLUX FOR INTERNAL BOUNDARY
                            IBVP.F_NF_PHI_BCS(dom, t, x, un, Uh, NFn);

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                integrand = sol.BF[rs]*NFn[ru];
                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                    // VALID NEIGHBORING DOMAIN
                    // INTERFACE CONDITIONS TO BE ASSIGNED
                    else
                    {
                        // Neighboring cell info
                        nbr_etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);

                        // Data structure used to compute the basis
                        // functions and the unknown fields for the
                        // neighboring cell
                        DG_sol_space_BFX nbr_sol(X_fab, sp);

                        // Get the support of the basis functions (for the
                        // neighboring cell)
                        BF_CELL(&prob_lo[0], &dx[0], i, j, k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        nbr_sol.set_support(nbr_BF_lo, nbr_BF_hi);

                        // COMPUTE THE INTEGRAL
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[qi_pos+0];,
                                x[1] = xptr[qi_pos+1];,
                                x[2] = xptr[qi_pos+2];
                            )
                            w = xptr[qi_pos+AMREX_SPACEDIM];

                            // UNIT NORMAL
                            AMREX_D_TERM
                            (
                                un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                            )

                            // SOLUTION
                            sol._eval_BF_only_table_(x);
                            nbr_sol._eval_BF_only_table_(x);
                            for (int cu = 0; cu < N_U; ++cu)
                            {
                                sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);

                                Uh[cu] = sol._eval_unsafe_();

                                nbr_sol.set_fab_ptrs(nbr_BF_i, nbr_BF_j, nbr_BF_k, cu);

                                nbr_Uh[cu] = nbr_sol._eval_unsafe_();
                            }
                            
                            // NUMERICAL FLUX FOR INTERNAL INTERFACE
                            IBVP.F_NF_PHI_ICS(dom, nbr_dom, t, x, un, Uh, nbr_Uh, NFn);

                            // DG CONTRIBUTION
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                integrand = sol.BF[rs]*NFn[ru];
                                dXdt_fab(i,j,k,rs+ru*sNp) -= integrand*w;
                            }
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
*/