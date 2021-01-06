AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2dULR(const Real * U, const Real * y, const Real * nbr_U, Real * y2L, Real * y2R, Real * dUL, Real * dUR)
{
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (y[d] < 0.0)
        {
            y2L[d] += y[d]*y[d];

            for (int ru = 0; ru < DG_N_SOL; ++ru)
            {
                dUL[ru+d*DG_N_SOL] += (nbr_U[ru]-U[ru])*y[d];
            }
        }
        else
        {
            y2R[d] += y[d]*y[d];

            for (int ru = 0; ru < DG_N_SOL; ++ru)
            {
                dUR[ru+d*DG_N_SOL] += (nbr_U[ru]-U[ru])*y[d];
            }
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
#ifdef APPLY_LIMITER
                Real y2L[AMREX_SPACEDIM], y2R[AMREX_SPACEDIM];
                Real dPL[DG_N_SOL*AMREX_SPACEDIM], dPR[DG_N_SOL*AMREX_SPACEDIM], dP[DG_N_SOL*AMREX_SPACEDIM];
                Real dCL[DG_N_SOL], dCR[DG_N_SOL], dC[DG_N_SOL];
#else
                Real y2[AMREX_SPACEDIM];
                Real dP[DG_N_SOL*AMREX_SPACEDIM], dC[DG_N_SOL];
#endif

#ifdef APPLY_LIMITER
                AMREX_D_TERM
                (
                    y2L[0] = 0.0; y2R[0] = 0.0;,
                    y2L[1] = 0.0; y2R[1] = 0.0;,
                    y2L[2] = 0.0; y2R[2] = 0.0;
                )
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dPL[ru+d*DG_N_SOL] = 0.0;
                    dPR[ru+d*DG_N_SOL] = 0.0;
                }
#else
                AMREX_D_TERM
                (
                    y2[0] = 0.0;,
                    y2[1] = 0.0;,
                    y2[2] = 0.0;
                )
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dP[ru+d*DG_N_SOL] = 0.0;
                }
#endif

                // ELEMENT INFO
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);
/**/
if (i == 3 && j == 18)
{
Print() << "P("; DG::IO::PrintReals(AMREX_SPACEDIM, xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, P);
}
/**/

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

nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                        // ADD LS CONTRIBUTION
                        ApplyR(R, xc, nbr_xc, y);
#ifdef APPLY_LIMITER
                        Add2dU(P, y, nbr_P, y2L, y2R, dPL, dPR);
#else
                        Add2dU(P, y, nbr_P, y2, dP);
#endif

/**/
if (i == 3 && j == 18)
{
Print() << "nbr_P("; DG::IO::PrintReals(AMREX_SPACEDIM, nbr_xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);
}
/**/
                    }

                }

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

nbr_P[0] = P[0]+1.0*(nbr_xc[0]-xc[0])+2.0*(nbr_xc[1]-xc[1]);

                    // ADD LS CONTRIBUTION
                    ApplyR(R, xc, nbr_xc, y);
#ifdef APPLY_LIMITER
                    Add2dU(P, y, nbr_P, y2L, y2R, dPL, dPR);
#else
                    Add2dU(P, y, nbr_P, y2, dP);
#endif

/**/
if (i == 3 && j == 18)
{
Print() << "nbr_P("; DG::IO::PrintReals(AMREX_SPACEDIM, nbr_xc); amrex::Print() << "): "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);
}
/**/
                }

/**/
if (i == 3 && j == 18)
{
#ifdef APPLY_LIMITER
Print() << "dPL: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPL);
Print() << "dPR: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPR);
#else
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);
#endif
}
/**/

                // EVAL DIFFERENCES -----------------------------------
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
#ifdef APPLY_LIMITER
                    if (y2L[d] > 0.0)
                    {
                        y2L[d] = 1.0/y2L[d];

                        for (int ru = 0; ru < DG_N_SOL; ++ru)
                        {
                            dPL[ru+d*DG_N_SOL] *= y2L[d];
                        }
                    }

                    if (y2R[d] > 0.0)
                    {
                        y2R[d] = 1.0/y2R[d];

                        for (int ru = 0; ru < DG_N_SOL; ++ru)
                        {
                            dPR[ru+d*DG_N_SOL] *= y2R[d];
                        }
                    }
#else
                    y2[d] = 1.0/y2[d];

                    for (int ru = 0; ru < DG_N_SOL; ++ru)
                    {
                        dP[ru+d*DG_N_SOL] *= y2[d];
                    }
#endif
                }
/**/
if (i == 3 && j == 18)
{
#ifdef APPLY_LIMITER
Print() << "y2L: " << std::endl; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2L);
Print() << "y2R: " << std::endl; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2R);
#else
Print() << "y2: " << std::endl; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2);
#endif
}
/**/
/**/
if (i == 3 && j == 18)
{
#ifdef APPLY_LIMITER
Print() << "dPL: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPL);
Print() << "dPR: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPR);
#else
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);
#endif
}
/**/
                // ----------------------------------------------------

#ifdef APPLY_LIMITER
                // LIMITED SLOPES (VIA CHARACTERISTICS) ---------------
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
#ifdef LIMIT_C
                    AMREX_D_TERM
                    (
                        un[0] = R[d+0*AMREX_SPACEDIM];,
                        un[1] = R[d+1*AMREX_SPACEDIM];,
                        un[2] = R[d+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DP2DC(t, xc, un, P, &dPL[d*DG_N_SOL], dCL);
                    IBVP.F_DP2DC(t, xc, un, P, &dPR[d*DG_N_SOL], dCR);

                    MCDLimiter(dCL, dCR, dC);
/**/
if (i == 3 && j == 18)
{
Print() << "dCL: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, dCL);
Print() << "dCR: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, dCR);
Print() << "dC: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, dC);
}
/**/
                    IBVP.F_DC2DP(t, xc, un, P, dC, &dP[d*DG_N_SOL]);
#else
                    MCDLimiter(&dPL[d*DG_N_SOL], &dPR[d*DG_N_SOL], &dP[d*DG_N_SOL]);
#endif
                }
                // ----------------------------------------------------
#else
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    AMREX_D_TERM
                    (
                        un[0] = R[d+0*AMREX_SPACEDIM];,
                        un[1] = R[d+1*AMREX_SPACEDIM];,
                        un[2] = R[d+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DP2DC(t, xc, un, P, &dP[d*DG_N_SOL], dC);
/**/
if (i == 3 && j == 18)
{
Print() << "dC: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, dC);
}
/**/
                    IBVP.F_DC2DP(t, xc, un, P, dC, &dP[d*DG_N_SOL]);
                }
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
if (i == 3 && j == 18)
{
Print() << "R: " << std::endl;
DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, R);

for (int d = 0; d < AMREX_SPACEDIM; ++d)
for (int cu = 0; cu < DG_N_SOL; ++cu)
{
    dP[cu+d*DG_N_SOL] = dXP_fab(i,j,k,d+AMREX_SPACEDIM*cu);
}
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);

exit(-1);
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