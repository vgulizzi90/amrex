/**
 * \brief Recentered limiter.
*/
#ifdef LIMIT_C
template <typename IBVP_CLASS>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void RecenteredLimiter(const Real t, const Real * xc, const Real * P, const Real * dP, const Real * dC, const Real * R, const Real * y, const Real * nbr_P, const Real * ay_max, Real * lim_dC, const IBVP_CLASS & IBVP)
#else
void RecenteredLimiter(const Real * P, const Real * dP, const Real * y, const Real * nbr_P, const Real * ay_max, Real * lim_dP)
#endif
{
    Real tmp_dP[DG_N_SOL];
#ifdef LIMIT_C
    Real tmp_dC[DG_N_SOL], dir[AMREX_SPACEDIM];
#endif

    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (std::abs(y[d]) > 0.25*ay_max[d])
        {
            for (int ru = 0; ru < DG_N_SOL; ++ru)
            {
                tmp_dP[ru] = P[ru];
                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                {
                    if (d2 != d)
                    {
                        tmp_dP[ru] += dP[ru+d2*DG_N_SOL]*y[d2];
                    }
                }
                tmp_dP[ru] = nbr_P[ru]-tmp_dP[ru];
                tmp_dP[ru] /= y[d];
            }

#ifdef LIMIT_C
            AMREX_D_TERM
            (
                dir[0] = R[d+0*AMREX_SPACEDIM];,
                dir[1] = R[d+1*AMREX_SPACEDIM];,
                dir[2] = R[d+2*AMREX_SPACEDIM];
            )

            IBVP.F_DP2DC(t, xc, dir, P, tmp_dP, tmp_dC);

            for (int ru = 0; ru < DG_N_SOL; ++ru)
            {
                const Real sgn = (dC[ru+d*DG_N_SOL] > 0.0) ? +1.0 : -1.0;

                if (tmp_dC[ru]*dC[ru+d*DG_N_SOL] > 0.0)
                {
                    lim_dC[ru+d*DG_N_SOL] = sgn*std::min(std::abs(lim_dC[ru+d*DG_N_SOL]), MCD_THETA*std::abs(tmp_dC[ru]));
                }
                else
                {
                    lim_dC[ru+d*DG_N_SOL] = 0.0;
                }
            }

#else
            for (int ru = 0; ru < DG_N_SOL; ++ru)
            {
                const Real sgn = (dP[ru+d*DG_N_SOL] > 0.0) ? +1.0 : -1.0;

                if (tmp_dP[ru]*dP[ru+d*DG_N_SOL] > 0.0)
                {
                    lim_dP[ru+d*DG_N_SOL] = sgn*std::min(std::abs(lim_dP[ru+d*DG_N_SOL]), MCD_THETA*std::abs(tmp_dP[ru]));
                }
                else
                {
                    lim_dP[ru+d*DG_N_SOL] = 0.0;
                }
            }
#endif
        }
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
void EvalSlopesLS_P(const Real t,
                    const DG::ImplicitMesh & mesh,
                    const MultiFab & X,
                    MultiFab & dXP,
                    const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
#ifdef AMREX_USE_CUDA
    const Real * eFVInfo_mem_ptr = mesh.eFVInfo_device_mem.data();
#else
    const Real * eFVInfo_mem_ptr = mesh.eFVInfo_host_mem.data();
#endif

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // VARIABLES ======================================================
    // ================================================================

    // INITIALIZATION
    dXP = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<long const> const & eFVInfo_pos_fab = mesh.eFVInfo_pos.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            const bool elm_is_valid = DG::ELM_IS_VALID(etype);

            if (elm_is_valid)
            {
                // LOCAL PARAMETERS
                const long pos = eFVInfo_pos_fab(i,j,k,DG::ELM_FV_INFO_POS(dom));
                const Real * xm = &eFVInfo_mem_ptr[pos];
                const Real * R = &eFVInfo_mem_ptr[pos+AMREX_SPACEDIM];

                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM], y[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_BF_i, nbr_BF_j, nbr_BF_k;
                short nbr_etype;
                Real y2[AMREX_SPACEDIM], ay_max[AMREX_SPACEDIM];
                Real dP[DG_N_SOL*AMREX_SPACEDIM];
#ifdef APPLY_LIMITER
#ifdef LIMIT_C
                Real dC[DG_N_SOL*AMREX_SPACEDIM];
#endif
                Real lim_slopes[DG_N_SOL*AMREX_SPACEDIM];
#endif

                // INITIALIZATION -------------------------------------
                AMREX_D_TERM
                (
                    y2[0] = 0.0;,
                    y2[1] = 0.0;,
                    y2[2] = 0.0;
                )
                AMREX_D_TERM
                (
                    ay_max[0] = 0.0;,
                    ay_max[1] = 0.0;,
                    ay_max[2] = 0.0;
                )
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru+d*DG_N_SOL] = 0.0;
                }

                // ELEMENT INFO
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);

/**/
if (i == __i__ && j == __j__)
{
Print() << std::endl << "P("; DG::IO::PrintReals(AMREX_SPACEDIM, xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, P);
}
/**/
                // ----------------------------------------------------

                // EVAL LEFT AND RIGHT DIFFERENCES --------------------
                // LOOP OVER THE ELEMENT IN THE STENCIL
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    nbr_i = i+DG::base_stencil_table_i[n];
                    nbr_j = j+DG::base_stencil_table_j[n];
                    nbr_k = k+DG::base_stencil_table_k[n];
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

//nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                        // ADD LS CONTRIBUTION
                        ApplyR(R, xc, nbr_xc, y);
                        Add2dU(P, y, nbr_P, y2, dP);

                        AMREX_D_TERM
                        (
                            ay_max[0] = std::max(ay_max[0], std::abs(y[0]));,
                            ay_max[1] = std::max(ay_max[1], std::abs(y[1]));,
                            ay_max[2] = std::max(ay_max[2], std::abs(y[2]));
                        )

/**/
if (i == __i__ && j == __j__)
{
Print() << "nbr_P("; DG::IO::PrintReals(AMREX_SPACEDIM, nbr_xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);
}
/**/
                    }
                }

                // ADD CONTRIBUTION FROM MIRRORED POINT
                if (DG::ELM_IS_LARGE_OR_EXTENDED(etype))
                {
                    // UNIT NORMAL
                    // We use nbr_xc[0] as a temporary variable
                    AMREX_D_TERM
                    (
                        un[0] = (xm[0]-xc[0]);,
                        un[1] = (xm[1]-xc[1]);,
                        un[2] = (xm[2]-xc[2]);
                    )
                    nbr_xc[0] = 1.0/std::sqrt(AMREX_D_TERM(un[0]*un[0],+un[1]*un[1],+un[2]*un[2]));
                    AMREX_D_TERM
                    (
                        un[0] *= nbr_xc[0];,
                        un[1] *= nbr_xc[0];,
                        un[2] *= nbr_xc[0];
                    )

                    // MIRRORED ELEMENT INFO
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = xm[0];,
                        nbr_xc[1] = xm[1];,
                        nbr_xc[2] = xm[2];
                    )
                    IBVP.F_PHI_BCS_P(dom, t, nbr_xc, un, P, nbr_P);

//nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                    // ADD LS CONTRIBUTION
                    ApplyR(R, xc, nbr_xc, y);
                    Add2dU(P, y, nbr_P, y2, dP);

                    AMREX_D_TERM
                    (
                        ay_max[0] = std::max(ay_max[0], std::abs(y[0]));,
                        ay_max[1] = std::max(ay_max[1], std::abs(y[1]));,
                        ay_max[2] = std::max(ay_max[2], std::abs(y[2]));
                    )

/**/
if (i == __i__ && j == __j__)
{
Print() << "nbr_P("; DG::IO::PrintReals(AMREX_SPACEDIM, nbr_xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);
}
/**/
                }
                // ----------------------------------------------------

/**/
if (i == __i__ && j == __j__)
{
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);
}
/**/
                
                // EVAL LEAST-SQUARE RECONSTRUCTION -------------------
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    y2[d] = 1.0/y2[d];

                    for (int ru = 0; ru < DG_N_SOL; ++ru)
                    {
                        dP[ru+d*DG_N_SOL] *= y2[d];
                    }
                }
                // ----------------------------------------------------

#ifdef APPLY_LIMITER
                // APPLY LIMITER --------------------------------------
#ifdef LIMIT_C
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    AMREX_D_TERM
                    (
                        un[0] = R[d+0*AMREX_SPACEDIM];,
                        un[1] = R[d+1*AMREX_SPACEDIM];,
                        un[2] = R[d+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DP2DC(t, xc, un, P, &dP[d*DG_N_SOL], &dC[d*DG_N_SOL]);

                    for (int ru = 0; ru < DG_N_SOL; ++ru)
                    {
                        lim_slopes[ru+d*DG_N_SOL] = dC[ru+d*DG_N_SOL];
                    }
                }
#else
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    lim_slopes[ru+d*DG_N_SOL] = dP[ru+d*DG_N_SOL];
                }
#endif

                // LOOP OVER THE ELEMENT IN THE STENCIL
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    nbr_i = i+DG::base_stencil_table_i[n];
                    nbr_j = j+DG::base_stencil_table_j[n];
                    nbr_k = k+DG::base_stencil_table_k[n];
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                    if (DG::ELM_IS_NOT_EMPTY(nbr_etype))
                    {
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        // NEIGHBOR ELEMENT INFO
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )
                        Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

//nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                        ApplyR(R, xc, nbr_xc, y);

#ifdef LIMIT_C
                        RecenteredLimiter(t, xc, P, dP, dC, R, y, nbr_P, ay_max, lim_slopes, IBVP);
#else
                        RecenteredLimiter(P, dP, y, nbr_P, ay_max, lim_slopes);
#endif
                    }
                }

                // ADD CONTRIBUTION FROM MIRRORED POINT
                if (DG::ELM_IS_LARGE_OR_EXTENDED(etype))
                {
                    // UNIT NORMAL
                    // We use nbr_xc[0] as a temporary variable
                    AMREX_D_TERM
                    (
                        un[0] = (xm[0]-xc[0]);,
                        un[1] = (xm[1]-xc[1]);,
                        un[2] = (xm[2]-xc[2]);
                    )
                    nbr_xc[0] = 1.0/std::sqrt(AMREX_D_TERM(un[0]*un[0],+un[1]*un[1],+un[2]*un[2]));
                    AMREX_D_TERM
                    (
                        un[0] *= nbr_xc[0];,
                        un[1] *= nbr_xc[0];,
                        un[2] *= nbr_xc[0];
                    )

                    // MIRRORED ELEMENT INFO
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = xm[0];,
                        nbr_xc[1] = xm[1];,
                        nbr_xc[2] = xm[2];
                    )
                    IBVP.F_PHI_BCS_P(dom, t, nbr_xc, un, P, nbr_P);

//nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                    ApplyR(R, xc, nbr_xc, y);

#ifdef LIMIT_C
                    RecenteredLimiter(t, xc, P, dP, dC, R, y, nbr_P, ay_max, lim_slopes, IBVP);
#else
                    RecenteredLimiter(P, dP, y, nbr_P, ay_max, lim_slopes);
#endif
                }

#ifdef LIMIT_C
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    AMREX_D_TERM
                    (
                        un[0] = R[d+0*AMREX_SPACEDIM];,
                        un[1] = R[d+1*AMREX_SPACEDIM];,
                        un[2] = R[d+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DC2DP(t, xc, un, P, &lim_slopes[d*DG_N_SOL], &dP[d*DG_N_SOL]);
                }
#else
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru+d*DG_N_SOL] = lim_slopes[ru+d*DG_N_SOL];
                }
#endif
                // ----------------------------------------------------
#endif

                // STORE IN THE MULTIFAB ------------------------------
                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,d1+AMREX_SPACEDIM*ru) += R[d2+d1*AMREX_SPACEDIM]*dP[ru+d2*DG_N_SOL];
                }
                // ----------------------------------------------------
/**/
if (i == __i__ && j == __j__)
{
Print() << "ay: "; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, ay_max);
Print() << "R: " << std::endl;
DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, R);

for (int d = 0; d < AMREX_SPACEDIM; ++d)
for (int cu = 0; cu < DG_N_SOL; ++cu)
{
    dP[cu+d*DG_N_SOL] = dXP_fab(i,j,k,d+AMREX_SPACEDIM*cu);
}
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);

//exit(-1);
}
/**/
            }
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(mesh.geom.periodicity());
    
    // CHECK ==========================================================
    if (dXP.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalSlopesLS_P\n";
        msg += "| dXP contains nans.\n";
        Abort(msg);
    }
    // ================================================================
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
void EvalSlopes_P(const Real t,
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
    // ================================================================

    // VARIABLES ======================================================
    // ================================================================

    // INITIALIZATION
    dXP = 0.0;
    // ==============

    // COMPUTE LIMITED SLOPES
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dXP_fab = dXP.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
            const bool elm_is_valid = DG::ELM_IS_VALID(etype);

            if (elm_is_valid)
            {
                // LOCAL VARIABLES
                Real xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM], inv_dx;
                Real P[DG_N_SOL];
                int dir, nbr_i, nbr_j, nbr_k;
                short nbr_etype;
                Real nbr_P[DG_N_SOL], dP[DG_N_SOL];
                Real dCL[DG_N_SOL], dCR[DG_N_SOL], dC[DG_N_SOL];

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                // CURRENT ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);

                // X1 DIRECTION =======================================
                dir = 0;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                // LEFT DIFFERENCE ------------------------------------
                nbr_i = i-1;
                nbr_j = j;
                nbr_k = k;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL LEFT DIFFERENCES
                inv_dx = 1.0/(xc[dir]-nbr_xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (P[ru]-nbr_P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCL);
                // ----------------------------------------------------

                // RIGHT DIFFERENCE -----------------------------------
                nbr_i = i+1;
                nbr_j = j;
                nbr_k = k;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL RIGHT DIFFERENCES
                inv_dx = 1.0/(nbr_xc[dir]-xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (nbr_P[ru]-P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCR);
                // ----------------------------------------------------

                // LIMITED SLOPES -------------------------------------
                MCDLimiter(dCL, dCR, dC);

                // BACK TO THE PRIMARY VARIABLES
                IBVP.F_DC2DP(t, xc, un, P, dC, dP);

                // STORE IN THE MULTIFAB
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = dP[ru];
                }
                // ----------------------------------------------------
                // ====================================================
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION =======================================
                dir = 1;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                // LEFT DIFFERENCE ------------------------------------
                nbr_i = i;
                nbr_j = j-1;
                nbr_k = k;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL LEFT DIFFERENCES
                inv_dx = 1.0/(xc[dir]-nbr_xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (P[ru]-nbr_P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCL);
                // ----------------------------------------------------

                // RIGHT DIFFERENCE -----------------------------------
                nbr_i = i;
                nbr_j = j+1;
                nbr_k = k;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL RIGHT DIFFERENCES
                inv_dx = 1.0/(nbr_xc[dir]-xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (nbr_P[ru]-P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCR);
                // ----------------------------------------------------

                // LIMITED SLOPES -------------------------------------
                MCDLimiter(dCL, dCR, dC);

                // BACK TO THE PRIMARY VARIABLES
                IBVP.F_DC2DP(t, xc, un, P, dC, dP);

                // STORE IN THE MULTIFAB
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = dP[ru];
                }
                // ----------------------------------------------------
                // ====================================================
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION =======================================
                dir = 2;

                // UNIT NORMAL
                AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                un[dir] = +1.0;

                // LEFT DIFFERENCE ------------------------------------
                nbr_i = i;
                nbr_j = j;
                nbr_k = k-1;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL LEFT DIFFERENCES
                inv_dx = 1.0/(xc[dir]-nbr_xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (P[ru]-nbr_P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCL);
                // ----------------------------------------------------

                // RIGHT DIFFERENCE -----------------------------------
                nbr_i = i;
                nbr_j = j;
                nbr_k = k+1;
                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));

                // NEIGHBOR ELEMENT CENTROID
                if (DG::ELM_IS_VALID(nbr_etype))
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG::ELM_CENTROID_X3(dom));
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = prob_lo[0]+(nbr_i+0.5)*dx[0];,
                        nbr_xc[1] = prob_lo[1]+(nbr_j+0.5)*dx[1];,
                        nbr_xc[2] = prob_lo[2]+(nbr_k+0.5)*dx[2];
                    )
                }
                // NEIGHBOR ELEMENT SOLUTION (PRIMARY VARIABLES)
                Fab2P(IBVP.gamma, X_fab, nbr_i, nbr_j, nbr_k, nbr_P);

                // EVAL RIGHT DIFFERENCES
                inv_dx = 1.0/(nbr_xc[dir]-xc[dir]);
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru] = (nbr_P[ru]-P[ru])*inv_dx;
                }

                // CHARACTERISTICS
                IBVP.F_DP2DC(t, xc, un, P, dP, dCR);
                // ----------------------------------------------------

                // LIMITED SLOPES -------------------------------------
                MCDLimiter(dCL, dCR, dC);

                // BACK TO THE PRIMARY VARIABLES
                IBVP.F_DC2DP(t, xc, un, P, dC, dP);

                // STORE IN THE MULTIFAB
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,dir+AMREX_SPACEDIM*ru) = dP[ru];
                }
                // ----------------------------------------------------
                // ====================================================
#endif
            }
        });
        Gpu::synchronize();
    }
    dXP.FillBoundary(mesh.geom.periodicity());
    
    // CHECK ==========================================================
    if (dXP.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - EvalSlopes_P\n";
        msg += "| dXP contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}