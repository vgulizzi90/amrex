// AMReX_DG_SlopeLimiter.H

#ifndef BL_DG_SLOPE_LIMITER_H_
#define BL_DG_SLOPE_LIMITER_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

#define DG_SL_THETA 2.0

namespace amrex
{
namespace DG
{
// ####################################################################
// DG SLOPE LIMITER AUXILIARY OPERATOR ################################
// ####################################################################
template<int N_PHI, int N_DOM>
void EvalAuxiliarySlopeLimiterOperator(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                       MatrixFactory<N_PHI, N_DOM> & matfactory)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalAuxiliarySlopeLimiterOperator(const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;
    const Real iaJ = mesh.iaJ;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    UnitCube<AMREX_SPACEDIM> & cube = matfactory.cube;

    // IMPLICIT MESH
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();

    // DG
    const int sp = cube.p;
    const int sNp = cube.Np;
    // ================================================================

    // VARIABLES ======================================================
    long space_SLOpe_mem_pos;
    // ================================================================

    // INITIALIZATION =================================================
    space_SLOpe_mem_pos = 0L;
    // ================================================================

    // INIT THE OPERATOR FOR THE STANDARD ELEMENT =====================
    cube.InitSLOpe(&dx[0]);
    // ================================================================

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM INFO ===========================
    {
        // Resize memory
        matfactory.space_SLOpe_host_mem.resize(sNp*(1+AMREX_SPACEDIM));

        // Store data
        std::copy(matfactory.cube.SLOpe.begin(), matfactory.cube.SLOpe.end(), matfactory.space_SLOpe_host_mem.data());

        // Update memory
        space_SLOpe_mem_pos += sNp*(1+AMREX_SPACEDIM);
    }
    // ================================================================

    // STORE THE OPERATOR INFO AS DOMAIN (CELL-CENTERED) DATA =========
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

        Array4<long> const & space_SLOpe_pos_fab = matfactory.space_SLOpe_pos.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
            const bool elm_is_partial = (elm_is_large || elm_is_small);
            const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], w;
            Real SLOpe[DG_SOL_MAX_SPACE_Np*(1+AMREX_SPACEDIM)];
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

            // Get the support of the basis functions.
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

            // Data structures used to compute the basis functions
            DG_sol_space_BF bf(BF_lo, BF_hi, sp);

            // EMPTY ELEMENTS -----------------------------------------
            if (elm_is_empty)
            {
                space_SLOpe_pos_fab(i,j,k,dom) = -1L;
            }
            // --------------------------------------------------------
            // ENTIRE ELEMENTS ----------------------------------------
            else if (elm_is_entire)
            {
                // At the moment we store the slope limiter auxiliary
                // operator of the [-1,1]^AMREX_SPACEDIM cube which
                // will be subsequently combined with the operator of
                // the small element the current element extends to.
                if (elm_is_entire_extended)
                {
                    // Store memory offset
                    space_SLOpe_pos_fab(i,j,k,dom) = space_SLOpe_mem_pos;

                    // Expand memory
                    matfactory.space_SLOpe_host_mem.resize(space_SLOpe_mem_pos+sNp*(1+AMREX_SPACEDIM));

                    // Store data
                    for (int cs = 0; cs < (sNp*(1+AMREX_SPACEDIM)); ++cs)
                    {
                        SLOpe[cs] = matfactory.cube.SLOpe[cs]*aJ;
                    }

                    // Multiply by mass matrix
                    DG_utils::matmul(sNp, sNp, 1+AMREX_SPACEDIM, matfactory.cube.MM.data(), SLOpe, &matfactory.space_SLOpe_host_mem[space_SLOpe_mem_pos]);

                    // Update memory
                    space_SLOpe_mem_pos += sNp*(1+AMREX_SPACEDIM);
                }
                else
                {
                    space_SLOpe_pos_fab(i,j,k,dom) = 0L;
                }
            }
            // --------------------------------------------------------
            // PARTIAL ELEMENTS ---------------------------------------
            else if (elm_is_partial)
            {
                // At the moment we compute and store the average
                // operator of the element WITHOUT multiplying by the
                // inverse mass matrix.
                {
                    // QUADRATURE INFO
                    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    const long pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    const Real * xptr = &eDomQuad_mem_ptr[pos];
                    long qi_pos;

                    // Store memory offset
                    space_SLOpe_pos_fab(i,j,k,dom) = space_SLOpe_mem_pos;

                    // Expand memory
                    matfactory.space_SLOpe_host_mem.resize(space_SLOpe_mem_pos+sNp*(1+AMREX_SPACEDIM));

                    // Store data
                    std::fill(SLOpe, SLOpe+sNp*(1+AMREX_SPACEDIM), 0.0);

                    for (int qi = 0; qi < dom_Nq; ++qi)
                    {
                        qi_pos = (AMREX_SPACEDIM+1)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                        )
                        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

                        bf._eval_BF_only_table_(x);

                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            SLOpe[rs] += bf.BF[rs]*w;
                            AMREX_D_TERM
                            (
                                SLOpe[rs+1*sNp] += bf.BF[rs]*(x[0]-cell_m[0])*w;,
                                SLOpe[rs+2*sNp] += bf.BF[rs]*(x[1]-cell_m[1])*w;,
                                SLOpe[rs+3*sNp] += bf.BF[rs]*(x[2]-cell_m[2])*w;
                            )
                        }
                    }

                    for (int cs = 0; cs < (sNp*(1+AMREX_SPACEDIM)); ++cs)
                    {
                        matfactory.space_SLOpe_host_mem[space_SLOpe_mem_pos+cs] = SLOpe[cs];
                    }

                    // Update memory
                    space_SLOpe_mem_pos += sNp*(1+AMREX_SPACEDIM);
                }
            }
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // STORE MEMORY INFO ==============================================
    matfactory.mem_for_space_SLOpe = space_SLOpe_mem_pos;
    // ================================================================

    // COPY MEMORY TO THE DEVICE ======================================
#ifdef AMREX_USE_CUDA
    matfactory.space_SLOpe_device_mem.resize(matfactory.space_SLOpe_host_mem.size());

    Gpu::copy(Gpu::hostToDevice, matfactory.space_SLOpe_host_mem.begin(), matfactory.space_SLOpe_host_mem.end(), matfactory.space_SLOpe_device_mem.begin());
#endif
    // ================================================================

    // ADD THE CONTRIBUTION FOR THE EXTENDED ELEMENTS =================
#ifdef AMREX_USE_CUDA
    const Real * space_MMCh_mem_ptr = matfactory.space_MMCh_device_mem.data();
    Real * space_SLOpe_mem_ptr = matfactory.space_SLOpe_device_mem.data();
#else
    const Real * space_MMCh_mem_ptr = matfactory.space_MMCh_host_mem.data();
    Real * space_SLOpe_mem_ptr = matfactory.space_SLOpe_host_mem.data();
#endif

    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<long const> const & space_MMCh_pos_fab = matfactory.space_MMCh_pos.array(mfi);
        Array4<long const> const & space_SLOpe_pos_fab = matfactory.space_SLOpe_pos.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);
            const bool elm_is_special = elm_is_large || elm_is_entire_extended;
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

            // MASS MATRIX
            const long MMCh_pos = space_MMCh_pos_fab(i,j,k,dom);
            const Real * space_MMCh_ptr = &space_MMCh_mem_ptr[MMCh_pos];

            // AUXILIARY OPERATOR
            const long pos = space_SLOpe_pos_fab(i,j,k,dom);

            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;
            long nbr_pos;
            Real SLOpe[DG_SOL_MAX_SPACE_Np*(1+AMREX_SPACEDIM)];

            // RETRIEVE INFO FOR SPECIAL ELEMENTS
            if (elm_is_special)
            {
                // Read from memory
                for (int rs = 0; rs < (sNp*(1+AMREX_SPACEDIM)); ++rs)
                {
                    SLOpe[rs] = space_SLOpe_mem_ptr[pos+rs];
                }

                // Cycle over the boundaries, check whether the element
                // is extended and add the contribution
                if (elm_is_extended)
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        nbr_pos = space_SLOpe_pos_fab(nbr_i,nbr_j,nbr_k,dom);

                        // i) AVERAGE OPERATOR
                        for (int rs = 0; rs < (sNp*(1+AMREX_SPACEDIM)); ++rs)
                        {
                            SLOpe[rs] += space_SLOpe_mem_ptr[nbr_pos+rs];
                        }
                    }
                }

                // Multiply by inverse mass matrix
                DG_utils::Cholesky_solve_overwrite(sNp, 1+AMREX_SPACEDIM, space_MMCh_ptr, SLOpe);

                // Copy back to memory
                for (int rs = 0; rs < (sNp*(1+AMREX_SPACEDIM)); ++rs)
                {
                    space_SLOpe_mem_ptr[pos+rs] = SLOpe[rs]*iaJ;
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================
}
// ####################################################################



// ####################################################################
// DG SLOPE LIMITER ROUTINES ##########################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_FV(const Real t,
                          const ImplicitMesh<N_PHI, N_DOM> & mesh,
                          const MatrixFactory<N_PHI, N_DOM> & /*matfactory*/,
                          const MultiFab & X,
                          MultiFab & dX,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_FV(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    // ================================================================

    // INITIALIZATION
    dX = 0.0;
    // ==============

    // EVAL THE PRIMARY VARIABLES AND THEIR GRADIENTS =================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real Uh[N_U], nbr_Uh[N_U], URP[N_U], dUh[N_U];
                Real dCh_left[N_U], dCh_right[N_U], dCh[N_U];

                // Current cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    Uh[ru] = X_fab(i,j,k,ru);
                }

                // X1 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[0] = +1.0;

                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i    )*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i-1,j,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+1.0)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i+1,j,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,0+AMREX_SPACEDIM*ru) = dUh[ru]/dx[0];
                }
                // ----------------------------------------------------
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[1] = +1.0;
                
                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j    )*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i,j-1,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+1.0)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i,j+1,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,1+AMREX_SPACEDIM*ru) = dUh[ru]/dx[1];
                }
                // ----------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[2] = +1.0;
                
                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k    )*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i,j,k-1,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+1.0)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = X_fab(i,j,k+1,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE MULTIFAB
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dX_fab(i,j,k,2+AMREX_SPACEDIM*ru) = dUh[ru]/dx[2];
                }
                // ----------------------------------------------------
#endif
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_FV_LS(const Real t,
                             const ImplicitMesh<N_PHI, N_DOM> & mesh,
                             const MatrixFactory<N_PHI, N_DOM> & /*matfactory*/,
                             const MultiFab & X,
                             MultiFab & dX,
                             const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_FV_LS(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    const int ngr = 1;

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    // ================================================================

    // EVAL THE PRIMARY VARIABLES AND THEIR GRADIENTS =================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dX_fab = dX.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            if (elm_is_valid)
            {


Print() << "HERE WE ARE - ApplySlopeLimiter_FV_LS" << std::endl;
exit(-1);
            }
        });
        Gpu::synchronize();
    }
    dX.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}


template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ApplySlopeLimiter_DG(const Real t,
                          const ImplicitMesh<N_PHI, N_DOM> & mesh,
                          const MatrixFactory<N_PHI, N_DOM> & matfactory,
                          const MultiFab & Xa,
                          const charMultiFab & tags,
                          MultiFab & X,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ApplySlopeLimiter_DG(const Real t, const ImplicitMesh<N_PHI, N_DOM> &, MultiFab &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;

#ifdef AMREX_USE_CUDA
    const Real * space_SLOpe_mem_ptr = matfactory.space_SLOpe_device_mem.data();
#else
    const Real * space_SLOpe_mem_ptr = matfactory.space_SLOpe_host_mem.data();
#endif

    // DG
    const int sNp = matfactory.cube.Np;
    // ================================================================

    // EVAL THE PRIMARY VARIABLES AND THEIR GRADIENTS =================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<char const> const & tags_fab = tags.array(mfi);
        Array4<long const> const & space_SLOpe_pos_fab = matfactory.space_SLOpe_pos.array(mfi);

        Array4<Real const> const & Xa_fab = Xa.array(mfi);
        
        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_troubled = (tags_fab(i,j,k,dom) == TROUBLED_ELEMENT);

            // IF THE ELEMENT IS TROUBLED, APPLY THE LIMITER
            if (elm_is_valid && elm_is_troubled)
            {
                // LOCAL PARAMETERS
                const long pos = space_SLOpe_pos_fab(i,j,k,dom);
                const Real * SLOpe_ptr = &space_SLOpe_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real Uh[N_U], nbr_Uh[N_U], URP[N_U], dUh[N_U];
                Real dCh_left[N_U], dCh_right[N_U], dCh[N_U];
                Real AMREX_D_DECL(dUhdx1[N_U], dUhdx2[N_U], dUhdx3[N_U]);
                Real Yl[1+AMREX_SPACEDIM], Xl[DG_SOL_MAX_SPACE_Np];

                // Current cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    Uh[ru] = Xa_fab(i,j,k,ru);
                }

                // X1 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[0] = +1.0;

                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i    )*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i-1,j,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+1.0)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i+1,j,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUhdx1[ru] = dUh[ru]/dx[0];
                }
                // ----------------------------------------------------
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[1] = +1.0;
                
                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j    )*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i,j-1,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+1.0)*dx[1];,
                    x[2] = prob_lo[2]+(k+0.5)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i,j+1,k,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUhdx2[ru] = dUh[ru]/dx[1];
                }
                // ----------------------------------------------------
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION ---------------------------------------
                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[2] = +1.0;
                
                // LEFT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k    )*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i,j,k-1,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, nbr_Uh, Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = Uh[ru]-nbr_Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_left);

                // RIGHT DIFFERENCES
                // Space location
                AMREX_D_TERM
                (
                    x[0] = prob_lo[0]+(i+0.5)*dx[0];,
                    x[1] = prob_lo[1]+(j+0.5)*dx[1];,
                    x[2] = prob_lo[2]+(k+1.0)*dx[2];
                )

                // Neighbor cell info
                for (int ru = 0; ru < N_U; ++ru)
                {
                    nbr_Uh[ru] = Xa_fab(i,j,k+1,ru);
                }

                // Riemann problem
                IBVP.F_RIEMANN(dom, t, x, un, Uh, nbr_Uh, URP);

                // Characteristics
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUh[ru] = nbr_Uh[ru]-Uh[ru];
                }
                IBVP.F_DU2DC(dom, t, x, un, URP, dUh, dCh_right);

                // LIMITED SLOPES (OF THE CHARACTERISTICS)
                for (int ru = 0; ru < N_U; ++ru)
                {
                    const Real dc = 0.5*(dCh_left[ru]+dCh_right[ru]);
                    const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
                    const Real slope = DG_SL_THETA*fmin(fabs(dCh_left[ru]), fabs(dCh_right[ru]));
                    const Real lim = (dCh_left[ru]*dCh_right[ru] >= 0.0) ? slope : 0.0;
                    dCh[ru] = sgn*fmin(lim, fabs(dc));
                }

                // BACK TO THE CONSERVED VARIABLES
                IBVP.F_DC2DU(dom, t, x, un, URP, dCh, dUh);

                // AND TO THE SLOPES
                for (int ru = 0; ru < N_U; ++ru)
                {
                    dUhdx3[ru] = dUh[ru]/dx[2];
                }
                // ----------------------------------------------------
#endif
                // UPDATE THE DG COEFFICIENTS -------------------------
                for (int ru = 0; ru < N_U; ++ru)
                {
                    Yl[0] = Uh[ru];
                    AMREX_D_TERM
                    (
                        Yl[1] = dUhdx1[ru];,
                        Yl[2] = dUhdx2[ru];,
                        Yl[3] = dUhdx3[ru];
                    )

                    DG_utils::matmul(sNp, 1+AMREX_SPACEDIM, 1, SLOpe_ptr, Yl, Xl);

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
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################



// ####################################################################
// DG SLOPE LIMITER CLASS #############################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct SlopeLimiter
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // REFINEMENT RATIO
    IntVect ref_ratio;

    // IMPLICIT MESH
    ImplicitMesh<N_PHI, N_DOM> mesh;

    // MATRIX FACTORY
    MatrixFactory<N_PHI, N_DOM> matfactory;

    // SOLUTION MULTIFABS
    MultiFab X_old, X_new, dXdt;
    charMultiFab tags;
    // ################################################################

    // INITIALIZATION #################################################
    void Define(const ImplicitMesh<N_PHI, N_DOM> & coarse_mesh,
                const MatrixFactory<N_PHI, N_DOM> & coarse_matfactory,
                const InputReader & inputs, const int d)
    {
        // REFINEMENT RATIO ===========================================
        const int p = inputs.dG[d].space_p;
        
        AMREX_D_TERM
        (
            this->ref_ratio[0] = (inputs.dG[d].limiter.freeze_refinement[0] != 0) ? 1 : (2*p+1);,
            this->ref_ratio[1] = (inputs.dG[d].limiter.freeze_refinement[1] != 0) ? 1 : (2*p+1);,
            this->ref_ratio[2] = (inputs.dG[d].limiter.freeze_refinement[2] != 0) ? 1 : (2*p+1);
        )
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->mesh.DefineForSlopeLimiter(coarse_mesh, this->ref_ratio, inputs, d);
        // ============================================================

        // MATRIX FACTORY =============================================
        this->matfactory.DefineForSlopeLimiter(coarse_matfactory, this->ref_ratio,
                                               this->mesh.cc_ba, this->mesh.dm, inputs, d);
        // ============================================================

        // SOLUTION MULTIFABS =========================================
        {
            const int n_comp = (this->matfactory.cube.Np)*N_U;

            this->X_old.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_old = 0.0;

            this->X_new.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_new = 0.0;

            this->dXdt.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->dXdt = 0.0;
        }

        this->tags.define(this->mesh.cc_ba, this->mesh.dm, N_DOM, this->dom_data_ngr);
        this->tags = SOUND_ELEMENT;
        // ============================================================
    }
    // ################################################################

    // SET/UPDATE TAGS FROM COARSE MESH ###############################
    void SetTags(const charMultiFab & coarse_tags)
    {
        // PARAMETERS =================================================
        const IntVect rr = this->ref_ratio;
        // ============================================================

        // ============================================================
        for (MFIter mfi(this->tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<char const> const & coarse_tags_fab = coarse_tags.array(mfi);

            Array4<char> const & tags_fab = this->tags.array(mfi);
                
            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // COARSE GRID'S ELEMENT
#if (AMREX_SPACEDIM == 1)
#endif
#if (AMREX_SPACEDIM == 2)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = k;
#endif
#if (AMREX_SPACEDIM == 3)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = (k-pk)/rr[2];
#endif
    
                if (coarse_tags_fab(ci,cj,ck,dom) == TROUBLED_ELEMENT)
                {
                    tags_fab(i,j,k,dom) = SOUND_ELEMENT;
                }
                else
                {
                    tags_fab(i,j,k,dom) = TROUBLED_ELEMENT;
                }
            });
            Gpu::synchronize();
        }
        this->tags.FillBoundary(this->mesh.geom.periodicity());
        // ============================================================
    }

    void UpdateTags(const charMultiFab & coarse_tags)
    {
        // PARAMETERS =================================================
        const IntVect rr = this->ref_ratio;
        // ============================================================

        // ============================================================
        for (MFIter mfi(this->tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<char const> const & coarse_tags_fab = coarse_tags.array(mfi);

            Array4<char> const & tags_fab = this->tags.array(mfi);
                
            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // COARSE GRID'S ELEMENT
#if (AMREX_SPACEDIM == 1)
#endif
#if (AMREX_SPACEDIM == 2)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = k;
#endif
#if (AMREX_SPACEDIM == 3)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
                const int ci = (i-pi)/rr[0];
                const int cj = (j-pj)/rr[1];
                const int ck = (k-pk)/rr[2];
#endif
                const bool elm_is_tagged = (tags_fab(i,j,k,dom) == TROUBLED_ELEMENT);
                const bool coarse_elm_is_troubled = (coarse_tags_fab(ci,cj,ck,dom) == TROUBLED_ELEMENT);

                if (elm_is_tagged && coarse_elm_is_troubled)
                {
                    tags_fab(i,j,k,dom) = SOUND_ELEMENT;
                }
            });
            Gpu::synchronize();
        }
        this->tags.FillBoundary(this->mesh.geom.periodicity());
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif