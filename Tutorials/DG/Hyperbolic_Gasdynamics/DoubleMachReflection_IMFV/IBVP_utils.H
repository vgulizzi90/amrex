//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define MCD_THETA 2.0

#define STENCIL_N_NBR AMREX_D_PICK(2, 4, 6)

namespace amrex
{
namespace FV
{
namespace IdealGas
{

/**
 * \brief Extract the conserved variables state vector from a fab. 
 *
 * \param[in] X_fab: fab containing the data.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] U: pointer to memory that will contain the conserved variables state vector.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Fab2U(Array4<Real const> const & X_fab,
           const int i, const int j, const int k,
           Real * U)
{
    U[RHO] = X_fab(i,j,k,RHO);
    AMREX_D_TERM
    (
        U[RHOV1] = X_fab(i,j,k,RHOV1);,
        U[RHOV2] = X_fab(i,j,k,RHOV2);,
        U[RHOV3] = X_fab(i,j,k,RHOV3);
    )
    U[TE] = X_fab(i,j,k,TE);
}

/**
 * \brief Fill fab from a conserved variables state vector. 
 *
 * \param[in] U: pointer to memory containing the conserved variables state vector.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] X_fab: fab that will contain the updated data.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void U2Fab(const Real * U,
             const int i, const int j, const int k,
             Array4<Real> const & X_fab)
{
    X_fab(i,j,k,RHO) = U[RHO];
    AMREX_D_TERM
    (
        X_fab(i,j,k,RHOV1) = U[RHOV1];,
        X_fab(i,j,k,RHOV2) = U[RHOV2];,
        X_fab(i,j,k,RHOV3) = U[RHOV3];
    )
    X_fab(i,j,k,TE) = U[TE];
}

/**
 * \brief Extract the primary variables state vector from a fab. 
 *
 * \param[in] X_fab: fab containing the data.
 * \param[in] (i,j,k): indices of the cell.
 * \param[out] P: pointer to memory that will contain the primary variables state vector.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Fab2P(const Real gamma,
           Array4<Real const> const & X_fab,
           const int i, const int j, const int k,
           Real * P)
{
    const Real rho = X_fab(i,j,k,RHO);
    AMREX_D_TERM
    (
        const Real u1 = X_fab(i,j,k,RHOV1)/rho;,
        const Real u2 = X_fab(i,j,k,RHOV2)/rho;,
        const Real u3 = X_fab(i,j,k,RHOV3)/rho;
    )
    const Real p = (gamma-1.0)*(X_fab(i,j,k,TE)-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));

    P[RHO] = rho;
    AMREX_D_TERM
    (
        P[RHOV1] = u1;,
        P[RHOV2] = u2;,
        P[RHOV3] = u3;
    )
    P[TE] = p;
}

/**
 * \brief Compute the maximum time step 
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const Real * dx, const Real t, const Real * x, const Real * U) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const DG::ImplicitMesh & mesh,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dt(X.boxarray, X.distributionMap, 1, 0);
    dt = std::numeric_limits<Real>::max();

    Real dt_min;
    // ================================================================

    // EVAL THE MINIMUM TIME STEP =====================================
    for (MFIter mfi(dt); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_VALID(etype))
            {
                // LOCAL PARAMTERS
                const Real vf = eInfo_fab(i,j,k,DG::ELM_VOLUME_FRACTION(dom));

                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM];
                Real elm_dt;
                Real U[FV_N_SOL];

                // ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )

                // ELEMENT SOLUTION
                Fab2U(X_fab, i, j, k, U);

                // EVAL DT
                elm_dt = IBVP.F_DT(&dx[0], t, xc, U);

                if (DG::ELM_IS_LARGE(etype))
                {
                    elm_dt *= vf;
                }

                dt_fab(i,j,k,0) = elm_dt;
            }
        });
        Gpu::synchronize();
    }

    dt_min = dt.min(0);
    // ================================================================

    return dt_min;
}

/**
 * \brief Fill cells at the grid's boundaries using boundary conditions.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[inout] X: a MultiFab object that contains unknown solution fields at the cell centers. On
 *                  exit, the cells in the ghost rows will have their values updated.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_BCS(const Real t, const Real * x, const Real * un,
 *                             const Real * U, const Real * bcs_U) const;
*/
template <typename IBVP_CLASS>
void FillBoundaryCells(const Real t,
                       const DG::ImplicitMesh & mesh,
                       MultiFab & X,
                       const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();
    
    // DOMAINS
    const int dom = 0;
    // ================================================================

    // CHECK ==========================================================
    if (X.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - FillBoundaryCells\n";
        msg += "| X contains nans (On entry).\n";
        Warning(msg);
    }
    // ================================================================

    // FILL BOUNDARY CELLS ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            int BF_i, BF_j, BF_k;
            Real U[FV_N_SOL], bcs_U[FV_N_SOL];

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // CURRENT CELL SOLUTION
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                Fab2U(X_fab, BF_i, BF_j, BF_k, U);

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
                    IBVP.F_BCS(t, x, un, U, bcs_U);

                    U2Fab(bcs_U, i-1, j, k, X_fab);
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
                    IBVP.F_BCS(t, x, un, U, bcs_U);
                    
                    U2Fab(bcs_U, i+1, j, k, X_fab);
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
                    IBVP.F_BCS(t, x, un, U, bcs_U);

                    U2Fab(bcs_U, i, j-1, k, X_fab);
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
                    IBVP.F_BCS(t, x, un, U, bcs_U);
                    
                    U2Fab(bcs_U, i, j+1, k, X_fab);
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
                    IBVP.F_BCS(t, x, un, U, bcs_U);
                    
                    U2Fab(bcs_U, i, j, k-1, X_fab);
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
                    IBVP.F_BCS(t, x, un, U, bcs_U);
                    
                    U2Fab(bcs_U, i, j, k+1, X_fab);
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
        msg += "ERROR: IBVP_utils.H - FillBoundaryCells\n";
        msg += "| X contains nans.\n";
        Warning(msg);
    }
    // ================================================================
}

/** \brief Apply Barth-Jespersen Limiter to avoid under- and overshoots.
 *
*/
template <typename IBVP_CLASS>
void ApplyBarthJespersenLimiter(const Real t,
                                const DG::ImplicitMesh & mesh,
                                const MultiFab & X,
                                MultiFab & dXP,
                                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // AUXILIARY MULTIFABS FOR LIMITING
    MultiFab alpha(X.boxarray, X.distributionMap, FV_N_SOL, X.n_grow);
    MultiFab bounds(X.boxarray, X.distributionMap, 2*FV_N_SOL, X.n_grow);

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // INITIALIZATION
    alpha = 0.0;
    bounds = 0.0;
    // ==============

    // COMPUTE THE BOUNDS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);
        
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<Real> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const bool elm_is_small = DG::ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real P[FV_N_SOL], nbr_P[FV_N_SOL];

                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // SOLUTION AT THE CURRENT ELEMENT
                Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);

                // INIT BOUNDS ----------------------------------------
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    b_fab(i,j,k,ru) = P[ru];
                    b_fab(i,j,k,ru+FV_N_SOL) = P[ru];
                }
                // ----------------------------------------------------

                // FIND THE BOUNDS ------------------------------------
                // CYCLE OVER THE BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;

                    // NEIGHBOR CELL INFO
                    DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // Conditions for NOT considering the boundary
                    // i) the current cell extends to the neighboring
                    //    cell;
                    // ii) the current cell is small and the
                    //     neighboring cell extends into it;
                    // iii) the neighboring cell is empty (it means
                    //      that the cell must be separated by the zero
                    //      level set of a function; this case is
                    //      handled by the INTERPHASE case.
                    //
                    const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                    const bool nbr_is_empty = (nbr_etype == __DG_ELM_TYPE_EMPTY__);
                    if (cells_are_merged || nbr_is_empty) continue;

                    // SUPPORT OF THE BASIS FUNCTIONS (NEIGHBOR CELL)
                    DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                    // SOLUTION AT THE NEIGHBOR ELEMENT
                    Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

                    for (int ru = 0; ru < FV_N_SOL; ++ru)
                    {
                        b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), nbr_P[ru]);
                        b_fab(i,j,k,ru+FV_N_SOL) = std::max(b_fab(i,j,k,ru+FV_N_SOL), nbr_P[ru]);
                    }
                }

                // ZERO LEVEL OF THE LEVEL SET
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0];,
                            x[1] = xptr[x_pos+1];,
                            x[2] = xptr[x_pos+2];
                        )

                        // UNIT NORMAL
                        AMREX_D_TERM
                        (
                            un[0] = xptr[x_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = xptr[x_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = xptr[x_pos+AMREX_SPACEDIM+1+2];
                        )

                        IBVP.F_PHI_BCS(t, x, un, P, nbr_P);

                        for (int ru = 0; ru < FV_N_SOL; ++ru)
                        {
                            b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), nbr_P[ru]);
                            b_fab(i,j,k,ru+FV_N_SOL) = std::max(b_fab(i,j,k,ru+FV_N_SOL), nbr_P[ru]);
                        }
                    }
                }
                // ---------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    bounds.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // HANDLE EXTENDED ELEMENTS =======================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        
        Array4<Real> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS EXTENDED
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_EXTENDED(etype))
            {
                int nbr_i, nbr_j, nbr_k, nbr_b;

                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    // NEIGHBOR CELL INFO
                    DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int ru = 0; ru < FV_N_SOL; ++ru)
                        {
                            b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), b_fab(nbr_i,nbr_j,nbr_k,ru));
                            b_fab(i,j,k,ru+FV_N_SOL) = std::max(b_fab(i,j,k,ru+FV_N_SOL), b_fab(nbr_i,nbr_j,nbr_k,ru+FV_N_SOL));
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    bounds.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // APPLY BARTH-JESPERSEN LIMITER ==================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
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
        Array4<Real const> const & dXP_fab = dXP.array(mfi);

        Array4<Real> const & a_fab = alpha.array(mfi);
        Array4<Real const> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // CHECK WHETHER ELEMENT IS NOT EMPTY
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM];
                Real P[FV_N_SOL], Px[FV_N_SOL];
                Real min_[FV_N_SOL], max_[FV_N_SOL];
                Real u0, u_lo, u_hi, a;

                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CENTROID OF THE CURRENT ELEMENT
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                // SOLUTION AT THE CURRENT ELEMENT
                Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);

                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    min_[cu] = P[cu];
                    max_[cu] = P[cu];
                }

                // CYCLE OVER THE BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    const int dir = b/2;

                    int fi, fj, fk;
                    int bou_Nq;
                    long pos;

                    // WHICH FACE
                    DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                    if (dir == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));
                    }
#endif
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0]+cell_lo[0];,
                            x[1] = xptr[x_pos+1]+cell_lo[1];,
                            x[2] = xptr[x_pos+2]+cell_lo[2];
                        )
                        
                        // SOLUTION
                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            Px[cu] = P[cu];
                            AMREX_D_TERM
                            (
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                            )
                        }
                        
                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            min_[cu] = std::min(min_[cu], Px[cu]);
                            max_[cu] = std::max(max_[cu], Px[cu]);
                        }
                    }
                }

                // BOUNDARY CONDITIONS: ZERO LEVEL OF THE LEVELSET
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // CYCLE OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0];,
                            x[1] = xptr[x_pos+1];,
                            x[2] = xptr[x_pos+2];
                        )

                        // SOLUTION
                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            Px[cu] = P[cu];
                            AMREX_D_TERM
                            (
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                Px[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                            )
                        }

                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            min_[cu] = std::min(min_[cu], Px[cu]);
                            max_[cu] = std::max(max_[cu], Px[cu]);
                        }
                    }
                }
                
                // EVAL THE COEFFICIENTS USED TO SCALE THE SLOPES -----
                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    a = 1.0;

                    u0 = P[cu];
                    u_lo = b_fab(BF_i,BF_j,BF_k,cu);
                    u_hi = b_fab(BF_i,BF_j,BF_k,cu+FV_N_SOL);

                    if ((min_[cu] < u_lo) && (min_[cu] != u0))
                    {
                        a = std::min(a, ((u_lo-u0)/(min_[cu]-u0)));
                    }

                    if ((max_[cu] > u_hi) && (max_[cu] != u0))
                    {
                        a = std::min(a, ((u_hi-u0)/(max_[cu]-u0)));
                    }

                    a_fab(i,j,k,cu) = a;
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();

    }
    bounds.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // EVENTUALLY FIX THE SLOPES ======================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real const> const & a_fab = alpha.array(mfi);
        
        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            const bool elm_is_valid = DG::ELM_IS_VALID(etype);
            const bool elm_is_extended = DG::ELM_IS_EXTENDED(etype);

            // LOCAL VARIABLES
            int nbr_i, nbr_j, nbr_k, nbr_b;
            Real a[FV_N_SOL];

            if (elm_is_valid)
            {
                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    a[cu] = a_fab(i,j,k,cu);
                }
            }

            // MAKE SURE EXTENDED ELEMENTS ACCOUNT FOR SMALL ELEMENTS
            if (elm_is_extended)
            {
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    // NEIGHBOR CELL INFO
                    DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    const bool nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);

                    if (cells_are_merged)
                    {
                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            a[cu] = std::min(a[cu], a_fab(nbr_i,nbr_j,nbr_k,cu));
                        }
                    }
                }
            }

            if (elm_is_valid)
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    for (int cu = 0; cu < FV_N_SOL; ++cu)
                    {
                        dXP_fab(i,j,k,d+AMREX_SPACEDIM*cu) *= a[cu];
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}

/** \brief Add contribution to the Least-square matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2S(Real * S, const Real * g, const Real * x)
{
#if (AMREX_SPACEDIM == 2)
    S[0+0*2] += (x[0]-g[0])*(x[0]-g[0]);
    S[0+1*2] += (x[0]-g[0])*(x[1]-g[1]);
    S[1+0*2]  = S[0+1*2];
    S[1+1*2] += (x[1]-g[1])*(x[1]-g[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    S[0+0*3] += (x[0]-g[0])*(x[0]-g[0]);
    S[0+1*3] += (x[0]-g[0])*(x[1]-g[1]);
    S[0+2*3] += (x[0]-g[0])*(x[2]-g[2]);
    S[1+0*3]  = S[0+1*3];
    S[2+0*3]  = S[0+2*3];
    S[1+1*3] += (x[1]-g[1])*(x[1]-g[1]);
    S[1+2*3] += (x[2]-g[2])*(x[2]-g[2]);
    S[2+1*3]  = S[1+2*3];
    S[2+2*3] += (x[2]-g[2])*(x[2]-g[2]);
#endif
}

/** \brief Multiply by rotation matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyR(const Real * R, const Real * g, const Real * x, Real * y)
{
#if (AMREX_SPACEDIM == 2)
    y[0] = R[0+0*2]*(x[0]-g[0])+R[0+1*2]*(x[1]-g[1]);
    y[1] = R[1+0*2]*(x[0]-g[0])+R[1+1*2]*(x[1]-g[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    y[0] = R[0+0*3]*(x[0]-g[0])+R[0+1*3]*(x[1]-g[1])+R[0+2*3]*(x[2]-g[2]);
    y[1] = R[1+0*3]*(x[0]-g[0])+R[1+1*3]*(x[1]-g[1])+R[1+2*3]*(x[2]-g[2]);
    y[2] = R[2+0*3]*(x[0]-g[0])+R[2+1*3]*(x[1]-g[1])+R[2+2*3]*(x[2]-g[2]);
#endif
}

/** \brief Add contribution to the left and right differences.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2dP(const Real * P0, const Real * y, const Real * P, Real * y2L, Real * y2R, Real * dPL, Real * dPR)
{
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        if (y[dim] < 0.0)
        {
            y2L[dim] += y[dim]*y[dim];

            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dPL[ru+dim*FV_N_SOL] += (P[ru]-P0[ru])*y[dim];
            }
        }
        else
        {
            y2R[dim] += y[dim]*y[dim];

            for (int ru = 0; ru < FV_N_SOL; ++ru)
            {
                dPR[ru+dim*FV_N_SOL] += (P[ru]-P0[ru])*y[dim];
            }
        }
    }
}

/**
 * \brief Monotonized Central Difference (MCD) limiter.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void MCDLimiter(const Real * dCL, const Real * dCR, Real * dC)
{
    for (int ru = 0; ru < FV_N_SOL; ++ru)
    {
        const Real dc = 0.5*(dCL[ru]+dCR[ru]);
        const Real sgn = (dc >= 0.0) ? +1.0 : -1.0;
        const Real slope = MCD_THETA*std::min(std::abs(dCL[ru]), std::abs(dCR[ru]));
        const Real lim = (dCL[ru]*dCR[ru] >= 0.0) ? slope : 0.0;
        dC[ru] = sgn*std::min(lim, std::abs(dc));
    }
}

/**
 * \brief Eval limited slopes and store them. They'll be used to reconstruct the PRIMARY variables.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[out] dXP: a MultiFab object that will contain the limited slopes for the PRIMARY variables.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
*/
template <typename IBVP_CLASS>
void EvalLimitedSlopesByLeastSquares(const Real t,
                                     const DG::ImplicitMesh & mesh,
                                     const MultiFab & X,
                                     MultiFab & dXP,
                                     const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    
    // DOMAINS
    const int dom = 0;

    // NEIGHBORS INFO FOR LEAST-SQUARE SLOPE LIMITING
    const GpuArray<int, STENCIL_N_NBR> aux_table_i =
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
    const GpuArray<int, STENCIL_N_NBR> aux_table_j =
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
    const GpuArray<int, STENCIL_N_NBR> aux_table_k = 
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

    // AUXILIARY MULTIFABS FOR LIMITING
    MultiFab bounds(X.boxarray, X.distributionMap, 2*FV_N_SOL, X.n_grow);
    // ================================================================

    // INITIALIZATION
    dXP = 0.0;

    bounds = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES =========================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXP_fab = dXP.array(mfi);
        
        Array4<Real> const & b_fab = bounds.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            const bool elm_is_valid = DG::ELM_IS_VALID(etype);
            const bool elm_is_large_or_extended = DG::ELM_IS_LARGE_OR_EXTENDED(etype);

            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], nxc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
                Real P[FV_N_SOL], nbr_P[FV_N_SOL];
                Real S[AMREX_SPACEDIM*AMREX_SPACEDIM], EV[AMREX_SPACEDIM*AMREX_SPACEDIM];
                Real y[AMREX_SPACEDIM], y2L[AMREX_SPACEDIM], y2R[AMREX_SPACEDIM];
                Real dPL[FV_N_SOL*AMREX_SPACEDIM], dPR[FV_N_SOL*AMREX_SPACEDIM], dP[FV_N_SOL*AMREX_SPACEDIM];
                Real dCL[FV_N_SOL], dCR[FV_N_SOL], dC[FV_N_SOL];

                // LEVEL SET
                DG::DG_PHI_space_BFX phi(sp, PHI_fab);
                phi.set_u(Dom2Phi_ptr[2*dom]);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                // CURRENT ELEMENT SOLUTION
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);

                // INIT THE BOUNDS
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    b_fab(i,j,k,ru) = P[ru];
                    b_fab(i,j,k,ru+FV_N_SOL) = P[ru];
                }

                // FILL THE MATRIX S ----------------------------------
                DG::linalg::zeros(AMREX_SPACEDIM*AMREX_SPACEDIM, S);

                // MIRRORED IMAGE OF THE ELEMENT CENTROID
                if (elm_is_large_or_extended)
                {
                    int nbr_i = i;
                    int nbr_j = j;
                    int nbr_k = k;
                    DG::NEAREST_POINT(&prob_lo[0], &dx[0], phi, xc, nbr_i, nbr_j, nbr_k, nxc, rtol, it_max);
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 2.0*nxc[0]-xc[0];,
                        nbr_xc[1] = 2.0*nxc[1]-xc[1];,
                        nbr_xc[2] = 2.0*nxc[2]-xc[2];
                    )
                    Add2S(S, xc, nbr_xc);
                }

                // NEIGHBORS
                for (int n = 0; n < STENCIL_N_NBR; ++n)
                {
                    const int dir = n/2;
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    // NEIGHBOR ELEMENT CENTROID
                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        int BF_i, BF_j, BF_k;

                        BF_i = nbr_i;
                        BF_j = nbr_j;
                        BF_k = nbr_k;

                        if (DG::ELM_IS_SMALL(nbr_etype))
                        {
                            Real nbr_vf, nbr_nbr_vf;
                            
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_VOLUME_FRACTION(dom));

                            for (int nn = 0; nn < STENCIL_N_NBR; ++nn)
                            {
                                if ((nn/2) != dir)
                                {
                                    int n_i = nbr_i+aux_table_i[nn];
                                    int n_j = nbr_j+aux_table_j[nn];
                                    int n_k = nbr_k+aux_table_k[nn];
                                    nbr_nbr_vf = eInfo_fab(n_i,n_j,n_k,DG::ELM_VOLUME_FRACTION(dom));

                                    if (nbr_nbr_vf > nbr_vf)
                                    {
                                        nbr_vf = nbr_nbr_vf;
                                        BF_i = n_i;
                                        BF_j = n_j;
                                        BF_k = n_k;
                                    }
                                }
                            }

                            if ((BF_i == nbr_i) && (BF_j == nbr_j) && (BF_k == nbr_k))
                            {
                                BF_i = i;
                                BF_j = j;
                                BF_k = k;
                            }
                        }

                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Add2S(S, xc, nbr_xc);
                    }
                }
                // ----------------------------------------------------

                // DIAGONALIZE (WE DON'T CARE ABOUT THE EIGENVALUES) --
#if (AMREX_SPACEDIM == 2)
                DG::linalg::dsy2ev(S, un, EV);
#endif
#if (AMREX_SPACEDIM == 3)
                DG::linalg::dsy3ev(S, un, EV);
#endif
                //DG::linalg::eye(AMREX_SPACEDIM, EV);
                // ----------------------------------------------------

                // EVAL THE LEFT AND RIGHT DIFFERENCES ----------------
                AMREX_D_TERM
                (
                    y2L[0] = 0.0; y2R[0] = 0.0;,
                    y2L[1] = 0.0; y2R[1] = 0.0;,
                    y2L[2] = 0.0; y2R[2] = 0.0;
                )
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    dPL[ru+dim*FV_N_SOL] = 0.0;
                    dPR[ru+dim*FV_N_SOL] = 0.0;
                }

                // MIRRORED IMAGE OF THE ELEMENT CENTROID
                if (elm_is_large_or_extended)
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 2.0*nxc[0]-xc[0];,
                        nbr_xc[1] = 2.0*nxc[1]-xc[1];,
                        nbr_xc[2] = 2.0*nxc[2]-xc[2];
                    )
                    ApplyR(EV, xc, nbr_xc, y);

                    // MIRRORED POINT SOLUTION
                    phi.eval_un(nxc, un);
                    IBVP.F_PHI_BCS(t, nxc, un, P, nbr_P);
                    
                    // ADD CONTRIBUTION
                    Add2dP(P, y, nbr_P, y2L, y2R, dPL, dPR);

                    // UPDATE THE BOUNDS
                    for (int ru = 0; ru < FV_N_SOL; ++ru)
                    {
                        b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), nbr_P[ru]);
                        b_fab(i,j,k,ru+FV_N_SOL) = std::max(b_fab(i,j,k,ru+FV_N_SOL), nbr_P[ru]);
                    }
                }

                // NEIGHBORS
                for (int n = 0; n < STENCIL_N_NBR; ++n)
                {
                    const int dir = n/2;
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    // NEIGHBOR ELEMENT CENTROID
                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        int BF_i, BF_j, BF_k;
                        
                        BF_i = nbr_i;
                        BF_j = nbr_j;
                        BF_k = nbr_k;

                        if (DG::ELM_IS_SMALL(nbr_etype))
                        {
                            Real nbr_vf, nbr_nbr_vf;
                            
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_VOLUME_FRACTION(dom));

                            for (int nn = 0; nn < STENCIL_N_NBR; ++nn)
                            {
                                if ((nn/2) != dir)
                                {
                                    int n_i = nbr_i+aux_table_i[nn];
                                    int n_j = nbr_j+aux_table_j[nn];
                                    int n_k = nbr_k+aux_table_k[nn];
                                    nbr_nbr_vf = eInfo_fab(n_i,n_j,n_k,DG::ELM_VOLUME_FRACTION(dom));

                                    if (nbr_nbr_vf > nbr_vf)
                                    {
                                        nbr_vf = nbr_nbr_vf;
                                        BF_i = n_i;
                                        BF_j = n_j;
                                        BF_k = n_k;
                                    }
                                }
                            }

                            if ((BF_i == nbr_i) && (BF_j == nbr_j) && (BF_k == nbr_k))
                            {
                                BF_i = i;
                                BF_j = j;
                                BF_k = k;
                            }
                        }

                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                        )

                        ApplyR(EV, xc, nbr_xc, y);

                        // NEIGHBOR ELEMENT SOLUTION
                        Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, nbr_P);

                        // ADD CONTRIBUTION
                        Add2dP(P, y, nbr_P, y2L, y2R, dPL, dPR);

                        // UPDATE THE BOUNDS
                        for (int ru = 0; ru < FV_N_SOL; ++ru)
                        {
                            b_fab(i,j,k,ru) = std::min(b_fab(i,j,k,ru), nbr_P[ru]);
                            b_fab(i,j,k,ru+FV_N_SOL) = std::max(b_fab(i,j,k,ru+FV_N_SOL), nbr_P[ru]);
                        }
                    }
                }
                // ----------------------------------------------------

                // EVAL DIFFERENCES -----------------------------------
                // We use un as a temporary variable
                AMREX_D_TERM
                (
                    un[0] = 1.0/y2L[0];,
                    un[1] = 1.0/y2L[1];,
                    un[2] = 1.0/y2L[2];
                )
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    dPL[ru+dim*FV_N_SOL] *= un[dim];
                }
                // We use un as a temporary variable
                AMREX_D_TERM
                (
                    un[0] = 1.0/y2R[0];,
                    un[1] = 1.0/y2R[1];,
                    un[2] = 1.0/y2R[2];
                )
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    dPR[ru+dim*FV_N_SOL] *= un[dim];
                }
                // ----------------------------------------------------

                // LIMITED SLOPES (VIA CHARACTERISTICS) ---------------
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                {
                    AMREX_D_TERM
                    (
                        un[0] = EV[dim+0*AMREX_SPACEDIM];,
                        un[1] = EV[dim+1*AMREX_SPACEDIM];,
                        un[2] = EV[dim+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DP2DC(t, xc, un, P, &dPL[dim*FV_N_SOL], dCL);
                    IBVP.F_DP2DC(t, xc, un, P, &dPR[dim*FV_N_SOL], dCR);

                    MCDLimiter(dCL, dCR, dC);

                    IBVP.F_DC2DP(t, xc, un, P, dC, &dP[dim*FV_N_SOL]);
                }
                // ----------------------------------------------------

                // STORE IN THE MULTIFAB ------------------------------
                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                for (int ru = 0; ru < FV_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,d1+AMREX_SPACEDIM*ru) += EV[d2+d1*AMREX_SPACEDIM]*dP[ru+d2*FV_N_SOL];
                }
                // ----------------------------------------------------

                // FIND THE RECONSTRUCTED MINIMUM AND MAXIMUM ---------
                // We use dPL, dPR and dP as auxiliary variables.
                // ----------------------------------------------------

                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    dPL[cu] = P[cu];
                    dPR[cu] = P[cu];
                }

                // MIRRORED IMAGE OF THE ELEMENT CENTROID
                if (elm_is_large_or_extended)
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 2.0*nxc[0]-xc[0];,
                        nbr_xc[1] = 2.0*nxc[1]-xc[1];,
                        nbr_xc[2] = 2.0*nxc[2]-xc[2];
                    )

                    for (int cu = 0; cu < FV_N_SOL; ++cu)
                    {
                        dP[cu] = P[cu];
                        AMREX_D_TERM
                        (
                            dP[cu] += dXP_fab(i,j,k,0+AMREX_SPACEDIM*cu)*(nbr_xc[0]-xc[0]);,
                            dP[cu] += dXP_fab(i,j,k,1+AMREX_SPACEDIM*cu)*(nbr_xc[1]-xc[1]);,
                            dP[cu] += dXP_fab(i,j,k,2+AMREX_SPACEDIM*cu)*(nbr_xc[2]-xc[2]);
                        )

                        dPL[cu] = std::min(dPL[cu], dP[cu]);
                        dPR[cu] = std::max(dPR[cu], dP[cu]);
                    }
                }

                // NEIGHBORS
                for (int n = 0; n < STENCIL_N_NBR; ++n)
                {
                    const int dir = n/2;
                    const int nbr_i = i+aux_table_i[n];
                    const int nbr_j = j+aux_table_j[n];
                    const int nbr_k = k+aux_table_k[n];
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    // NEIGHBOR ELEMENT CENTROID
                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        int BF_i, BF_j, BF_k;
                        
                        BF_i = nbr_i;
                        BF_j = nbr_j;
                        BF_k = nbr_k;

                        if (DG::ELM_IS_SMALL(nbr_etype))
                        {
                            Real nbr_vf, nbr_nbr_vf;
                            
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_VOLUME_FRACTION(dom));

                            for (int nn = 0; nn < STENCIL_N_NBR; ++nn)
                            {
                                if ((nn/2) != dir)
                                {
                                    int n_i = nbr_i+aux_table_i[nn];
                                    int n_j = nbr_j+aux_table_j[nn];
                                    int n_k = nbr_k+aux_table_k[nn];
                                    nbr_nbr_vf = eInfo_fab(n_i,n_j,n_k,DG::ELM_VOLUME_FRACTION(dom));

                                    if (nbr_nbr_vf > nbr_vf)
                                    {
                                        nbr_vf = nbr_nbr_vf;
                                        BF_i = n_i;
                                        BF_j = n_j;
                                        BF_k = n_k;
                                    }
                                }
                            }

                            if ((BF_i == nbr_i) && (BF_j == nbr_j) && (BF_k == nbr_k))
                            {
                                BF_i = i;
                                BF_j = j;
                                BF_k = k;
                            }
                        }

                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                        )
                    }

                    for (int cu = 0; cu < FV_N_SOL; ++cu)
                    {
                        dP[cu] = P[cu];
                        AMREX_D_TERM
                        (
                            dP[cu] += dXP_fab(i,j,k,0+AMREX_SPACEDIM*cu)*(nbr_xc[0]-xc[0]);,
                            dP[cu] += dXP_fab(i,j,k,1+AMREX_SPACEDIM*cu)*(nbr_xc[1]-xc[1]);,
                            dP[cu] += dXP_fab(i,j,k,2+AMREX_SPACEDIM*cu)*(nbr_xc[2]-xc[2]);
                        )

                        dPL[cu] = std::min(dPL[cu], dP[cu]);
                        dPR[cu] = std::max(dPR[cu], dP[cu]);
                    }
                }
                // ----------------------------------------------------

                // APPLY BARTH-JESPERSEN LIMITER ----------------------
                for (int cu = 0; cu < FV_N_SOL; ++cu)
                {
                    Real a, u0, u_lo, u_hi, b_lo, b_hi;
                    
                    a = 1.0;
                    b_lo = b_fab(i,j,k,cu);
                    b_hi = b_fab(i,j,k,cu+FV_N_SOL);

                    u0 = P[cu];
                    u_lo = dPL[cu];
                    u_hi = dPR[cu];

                    if ((u_lo < b_lo) && (u_lo != u0))
                    {
                        a = std::min(a, ((b_lo-u0)/(u_lo-u0)));
                    }

                    if ((u_hi > b_hi) && (u_hi != u0))
                    {
                        a = std::min(a, ((b_hi-u0)/(u_hi-u0)));
                    }

                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        dXP_fab(i,j,k,d+AMREX_SPACEDIM*cu) *= a;
                    }
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();

    }
    dXP.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (dXP.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalLimitedSlopesByLeastSquares\n";
        msg += "| dXP contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Eval the time derivative of the state vectors.
 *
 * The following integral is evaluated: 
 *
 * int_{dVh} V_{,i}^T NFn
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] X: a MultiFab object that contains the state vector for each cell.
 * \param[in] dXP: a MultiFab object that contains the limited slopes for the PRIMARY variables.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of state vectors.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  void F_NF_ICS(const Real t, const Real * x, const Real * un,
 *                                const Real * U, const Real * nbr_U, Real * NFn) const;
 *                  void F_NF_BCS(const Real t, const Real * x, const Real * un,
 *                                const Real * U, Real * NFn) const;
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const DG::ImplicitMesh & mesh,
                          const MultiFab & X,
                          const MultiFab & dXP,
                          MultiFab & dXdt,
                          const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Eval_dXdt_BouContrib(const Real, const ImplicitMesh &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

    // DOMAIN
    const int dom = 0;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
    Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
    Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
    // ================================================================

    // EVAL THE INTEGRALS =============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
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
        Array4<Real const> const & dXP_fab = dXP.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx, FV_N_SOL,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const bool elm_is_small = DG::ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real P[FV_N_SOL], nbr_P[FV_N_SOL], NFn[FV_N_SOL];
                Real integrand;
                
                // SUPPORT OF THE BASIS FUNCTIONS (CURRENT CELL)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                // CELL BOUNDARIES
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    const int dir = b/2;

                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                    int fi, fj, fk;
                    int bou_Nq;
                    long pos;
                    bool intraphase;

                    // NEIGHBOR CELL INFO
                    DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    // Conditions for NOT computing the flux
                    // contribution:
                    // i) the current cell extends to the neighboring
                    //    cell;
                    // ii) the current cell is small and the
                    //     neighboring cell extends into it;
                    // iii) the neighboring cell is empty (it means
                    //      that the cell must be separated by the zero
                    //      level set of a function; this case is
                    //      handled by the INTERPHASE case.
                    //
                    const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                    const bool nbr_is_empty = (nbr_etype == __DG_ELM_TYPE_EMPTY__);
                    if (cells_are_merged || nbr_is_empty) continue;

                    // SUPPORT OF THE BASIS FUNCTIONS (NEIGHBOR CELL)
                    DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                    
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                    )

                    // WHICH FACE
                    DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                    // QUADRATURE INFO
                    intraphase = true;
                    if (dir == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_0(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                        if ((fi == domain.smallEnd(dir) || fi == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (dir == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_1(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

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
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_NQ(dom, b));
                        pos = eBouQuad_pos_fab_2(fi,fj,fk,DG::ELM_BOU_QUAD_POS(dom, b));

                        if ((fk == domain.smallEnd(dir) || fk == (domain.bigEnd(dir)+1)) &&
                            (is_periodic[dir] == 0))
                        {
                            intraphase = false;
                        }
                    }
#endif
                    const Real * xptr = &eBouQuad_mem_ptr[pos];

                    // UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = 2.0*(b%2)-1.0;

                    // INTRAPHASE -------------------------------------
                    if (intraphase)
                    {
                        // EVAL THE INTEGRAL
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
                            
                            // SOLUTION
                            Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);
                            Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);
                            for (int cu = 0; cu < FV_N_SOL; ++cu)
                            {
                                /**/
                                AMREX_D_TERM
                                (
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                                /**/
                                //P[cu] += dXP_fab(BF_i,BF_j,BF_k,dir+AMREX_SPACEDIM*cu)*(x[dir]-xc[dir]);

                                /**/
                                AMREX_D_TERM
                                (
                                    nbr_P[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_P[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_P[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
                                /**/
                                //nbr_P[cu] += dXP_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,dir+AMREX_SPACEDIM*cu)*(x[dir]-nbr_xc[dir]);
                            }

                            // NUMERICAL FLUX FOR INTRAPHASE
                            IBVP.F_NF_ICS_P(t, x, un, P, nbr_P, NFn);

                            // INTEGRAL CONTRIBUTION
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                    // ------------------------------------------------
                    // BOUNDARY CONDITIONS: GRID BOUNDARY -------------
                    else
                    {
                        // EVAL THE INTEGRAL
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
                            
                            // SOLUTION
                            Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);
                            for (int cu = 0; cu < FV_N_SOL; ++cu)
                            {
                                /**/
                                AMREX_D_TERM
                                (
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    P[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                                /**/
                                //P[cu] += dXP_fab(BF_i,BF_j,BF_k,dir+AMREX_SPACEDIM*cu)*(x[dir]-xc[dir]);
                            }

                            // NUMERICAL FLUX FOR GRID'S BOUNDARY
                            IBVP.F_NF_BCS_P(t, x, un, P, NFn);

                            // INTEGRAL CONTRIBUTION
                            integrand = NFn[ru];
                            dXdt_fab(i,j,k,ru) -= integrand*w;
                        }
                    }
                    // ------------------------------------------------
                }

                // BOUNDARY CONDITIONS: ZERO LEVEL OF THE LEVELSET ----
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
                    const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                    // EVAL THE INTEGRAL
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

                        // SOLUTION
                        Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);
                        for (int cu = 0; cu < FV_N_SOL; ++cu)
                        {
                            AMREX_D_TERM
                            (
                                P[cu] += dXP_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                P[cu] += dXP_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                P[cu] += dXP_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                            )
                        }

                        // NUMERICAL FLUX FOR INTERPHASE
                        IBVP.F_NF_PHI_BCS_P(t, x, un, P, NFn);

                        // INTEGRAL CONTRIBUTION
                        integrand = NFn[ru];
                        dXdt_fab(i,j,k,ru) -= integrand*w;
                    }
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    dXdt.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // CHECK ==========================================================
    if (dXdt.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
        msg += "| dXdt contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

/**
 * \brief Eval dXdt.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] X: a MultiFab object that contains the state vector for each cell.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of state vectors.
 * \param[in] IBVP: the Initial Boundary Value Problem class.
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const DG::ImplicitMesh & mesh,
               const DG::MatrixFactory & matfactory,
               const MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    // DOMAIN
    const int dom = 0;
    const Gpu::ManagedVector<int> Sol2Dom(FV_N_SOL, dom);
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dXP(ba, dm, AMREX_SPACEDIM*FV_N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

    // EVAL LIMITED SLOPES ============================================
    EvalLimitedSlopesByLeastSquares(t, mesh, X, dXP, IBVP);
    //ApplyBarthJespersenLimiter(t, mesh, X, dXP, IBVP);
    // ================================================================

    // EVAL THE TIME DERIVATIVES ======================================
    Eval_dXdt_BouContrib(t, mesh, X, dXP, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    DG::AddSmallElementsContribution(mesh, matfactory, FV_N_SOL, Sol2Dom, dXdt);
    DG::MultiplyByInverseMassMatrix(mesh, matfactory, FV_N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using 2nd-order RungeKutta scheme
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[inout] X: a MultiFab object that contains the state vector for each cell. On exit, the
 *                  state vector will be updated according to a 2nd-order RungeKutta algorithm.
 * \param[in] IBVP: the Initial Boundary Value Problem class.
*/
template <typename IBVP_CLASS>
void TakeTimeStep(const Real dt, const Real t,
                  const DG::ImplicitMesh & mesh,
                  const DG::MatrixFactory & matfactory,
                  MultiFab & X_new,
                  const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X_new.boxarray;
    const DistributionMapping & dm = X_new.distributionMap;

    // NUMBER OF COMPONENTS AND GHOST ROWS
    const int n_comp = X_new.n_comp;
    const IntVect ngr = X_new.n_grow;
    // ================================================================
    // NOTE: In this case, the number of components coincides with the
    //       number of unknown solution fields.
    // ================================================================

    // VARIABLES ======================================================
    MultiFab X_old(ba, dm, n_comp, ngr);
    MultiFab X1(ba, dm, n_comp, ngr);
    MultiFab dXdt(ba, dm, n_comp, ngr);
    // ================================================================

    // COPY NEW TO OLD ================================================
    MultiFab::Copy(X_old, X_new, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 1 =====================================================
    FillBoundaryCells(t, mesh, X_old, IBVP);

    // dXdt := dXdt(t, X_old)
    Eval_dXdt(t, mesh, matfactory, X_old, dXdt, IBVP);

    // X1 := X_old+dt*dXdt
    MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 STEP 2 =====================================================
    FillBoundaryCells(t, mesh, X1, IBVP);

    // dXdt := dXdt(t+dt, X1)
    Eval_dXdt(t+dt, mesh, matfactory, X1, dXdt, IBVP);

    // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
    MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
    MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);
    // ================================================================

    // CHECK SOLUTION =================================================
    if (X_new.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - TakeTimeStep\n";
        msg += "| Solution contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

} // namespace IdealGas
} // namespace FV
} // namespace amrex