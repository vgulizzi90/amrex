/**
 * \brief Auxiliary function used for evaluating the boundary flux at a wall.
 *
 */
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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

            Array4<short const> const & c_eType_fab = c_eType.array(mfi);
            Array4<Real const> const & c_X_fab = c_X.array(mfi);

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
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= c_rr[dir];
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
                                FX_fab(fi,fj,fk,offset+rs+ru*c_sNp) *= c_rr[dir];
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
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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
AMREX_GPU_HOST_DEVICE
template <typename IBVP_CLASS>
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
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
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
 * \param[in] c_eType: a shortMultiFab object containing the elementy type information at coarse
 *                     level.
 * \param[in] c_X: a MultiFab object containing the coefficients of the basis functions at the
 *                 coarse level.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
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

            Array4<short const> const & c_eType_fab = c_eType.array(mfi);
            Array4<Real const> const & c_X_fab = c_X.array(mfi);

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
                            DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, c_X_fab, BF_i, BF_j, BF_k);
                            BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                            DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, c_sNp, sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,rs+ru*c_sNp) *= c_rr[dir];
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
                            DG_SOL_space_BFX nbr_sol(sp, nbr_BF_lo, nbr_BF_hi, c_X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            // COMPUTE THE INTEGRAL
                            BOU_FX_ICS(t, dom, cell_lo, un, bou_Nq, xptr, sol, nbr_sol, N_SOL, sNp, c_sNp, fi, fj, fk, FX_fab, offset, IBVP);

                            // NEEDED TO USE average_down_faces
                            for (int ru = 0; ru < N_SOL; ++ru)
                            for (int rs = 0; rs < c_sNp; ++rs)
                            {
                                FX_fab(fi,fj,fk,offset+rs+ru*c_sNp) *= c_rr[dir];
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
 * \param[in] masks: a vector containing the iMultiFab mask objects at different levels.
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
               const Vector<iMultiFab> & masks,
               const int N_DOM,
               const int N_SOL,
               const Gpu::ManagedVector<int> & Sol2Dom,
               const Vector<MultiFab> & Xs,
               Vector<MultiFab> & dXdt,
               const IBVP_CLASS & IBVP)
{
    // VARIABLES ======================================================
    int offset;
    
    Vector<MultiFab> dXs(n_valid_levels);
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_valid_levels);
    // ================================================================
    // NOTE: dXs[lev] is used only if lev-th level uses p = 0, which
    //       activates a finite-volume scheme.
    // ================================================================

    // INITIALIZATION =================================================
    {
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
            dXdt[lev] = 0.0;

            // FINITE-VOLUME APPROACH (NEEDS RECONSTRUCTION)
            if (matfactories[lev]->std_elem.p == 0)
            {
                dXs[lev].define(Xs[lev].boxarray, Xs[lev].distributionMap, AMREX_SPACEDIM*N_SOL, 1);
                dXs[lev] = 0.0;
            }

            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                FX[lev][dim] = MultiFab(meshes[lev]->fc_ba[dim], meshes[lev]->dm, FX_n_comp, 0);
                FX[lev][dim] = 0.0;
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


    // PERFORM RECONSTRUCTION FOR 0-TH ORDER LEVELS ===================
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        if (matfactories[lev]->std_elem.p == 0)
        {
            const ImplicitMesh & mesh = *meshes[lev];
            const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
            const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
            const MultiFab & X = Xs[lev];

            for (MFIter mfi(dXs[lev]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                Array4<short const> const & eType_fab = mesh.eType.array(mfi);
                Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

                Array4<Real const> const & X_fab = X.array(mfi);
                Array4<Real> const & dX_fab = dXs[lev].array(mfi);

                ParallelFor(bx, N_DOM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                    if (ELM_IS_VALID(etype))
                    {
                        IBVP.F_R_SLOPES(t, &prob_lo[0], &dx[0], i, j, k, dom, eType_fab, eInfo_fab, X_fab, dX_fab);
                    }
                });
                Gpu::synchronize();
            }
            dXs[lev].FillBoundary(mesh.geom.periodicity());
        }
    }

#ifdef AMREX_DEBUG
    // CHECK SLOPES
    for (int lev = 0; lev < n_valid_levels; ++lev)
    {
        if (matfactories[lev]->std_elem.p == 0)
        {
            if (dXs[lev].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_Solution_Hyperbolic.H - Eval_dXdt\n";
                msg += "| The reconstructed slopes contain nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
#endif
    // ================================================================

    // CELL BOUNDARY FLUXES ===========================================
    // LEVEL 0 --------------------------------------------------------
    // FINITE-VOLUME APPROACH (USING RECONSTRUCTED SLOPES)
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

        if (isMFIterSafe(Xs[lev], c_X))
        {
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
        else
        {
            const BoxArray safe_c_ba = coarsen(meshes[lev]->cc_ba, c_rr);
            const DistributionMapping & f_dm = meshes[lev]->dm;
            
            const int X_n_comp = c_X.n_comp;
            const IntVect X_n_grow = c_X.n_grow;

            MultiFab safe_c_X(safe_c_ba, f_dm, X_n_comp, X_n_grow, MFInfo(), FArrayBoxFactory());
            safe_c_X.ParallelCopy(c_X, 0, 0, X_n_comp, X_n_grow, X_n_grow, c_mesh.geom.periodicity());

            const int eType_n_comp = c_eType.n_comp;
            const IntVect eType_n_grow = c_eType.n_grow;

            shortMultiFab safe_c_eType(safe_c_ba, f_dm, eType_n_comp, eType_n_grow);
            safe_c_eType.ParallelCopy(c_eType, 0, 0, eType_n_comp, eType_n_grow, eType_n_grow, c_mesh.geom.periodicity());
            
            // FINITE-VOLUME APPROACH (USING RECONSTRUCTED SLOPES)
            if (matfactories[lev]->std_elem.p == 0)
            {
                Eval_dXdt_BouFluxes_FV(t,
                                       c_rr, safe_c_eType, c_sp, safe_c_X,
                                       *meshes[lev],
                                       N_DOM, N_SOL,
                                       Xs[lev], dXs[lev], FX[lev], offset,
                                       IBVP);
            }
            // DISCONTINUOUS GALERKIN APPROACH
            else
            {
                Eval_dXdt_BouFluxes(t,
                                    c_rr, safe_c_eType, c_sp, safe_c_X,
                                    *meshes[lev], *matfactories[lev],
                                    N_DOM, N_SOL,
                                    Xs[lev], FX[lev], offset,
                                    IBVP);
            }
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
