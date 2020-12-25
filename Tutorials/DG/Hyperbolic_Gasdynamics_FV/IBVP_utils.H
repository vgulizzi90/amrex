//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_utils.H
 * \brief Contains routines for the solution of the IBVP.
*/

#define MCD_THETA 2.0


namespace amrex
{
namespace FV
{

/**
 * \brief Compute the maximum time step.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] X: a MultiFab object that contains the solution at the elements' centroids.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_DT(const int dom, const Real * dx, const Real t, const Real * x, const Real * SOL) const;
 *
 * \return the maximum time step.
*/
template <typename IBVP_CLASS>
Real Compute_dt(const Real t,
                const DG::ImplicitMesh & mesh,
                const DG::MatrixFactory & /*matfactory*/,
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

    // NUMBER OF DOMAINS
    const int N_DOM = 1;

    // MAX REAL
    const Real max_real = std::numeric_limits<Real>::max();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: IBVP_utils.H - Compute_dt\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
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
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG::ELM_DOM_QUAD_NQ(dom));
                const long pos = eDomQuad_pos_fab(i,j,k,DG::ELM_DOM_QUAD_POS(dom));
                const Real * xptr = &eDomQuad_mem_ptr[pos];
                
                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real x[AMREX_SPACEDIM];
                Real SOL[DG_N_SOL];
                Real elm_dt, x_dt;

                // SUPPORT OF THE SOLUTION
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

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
                    SOL[RHO] = X_fab(BF_i,BF_j,BF_k,RHO);
                    AMREX_D_TERM
                    (
                        SOL[RHOV1] = X_fab(BF_i,BF_j,BF_k,RHOV1);,
                        SOL[RHOV2] = X_fab(BF_i,BF_j,BF_k,RHOV2);,
                        SOL[RHOV3] = X_fab(BF_i,BF_j,BF_k,RHOV3);
                    )
                    SOL[TE] = X_fab(BF_i,BF_j,BF_k,TE);

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
 * \brief Monotonized Central Difference (MCD) limiter.
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void MCDLimiter(const Real * dCL, const Real * dCR, Real * dC)
{
    for (int ru = 0; ru < DG_N_SOL; ++ru)
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

/** \brief Multiply by rotation matrix.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void ApplyR(const Real * R, const Real * xc, const Real * nbr_xc, Real * y)
{
#if (AMREX_SPACEDIM == 2)
    y[0] = R[0+0*2]*(nbr_xc[0]-xc[0])+R[0+1*2]*(nbr_xc[1]-xc[1]);
    y[1] = R[1+0*2]*(nbr_xc[0]-xc[0])+R[1+1*2]*(nbr_xc[1]-xc[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    y[0] = R[0+0*3]*(nbr_xc[0]-xc[0])+R[0+1*3]*(nbr_xc[1]-xc[1])+R[0+2*3]*(nbr_xc[2]-xc[2]);
    y[1] = R[1+0*3]*(nbr_xc[0]-xc[0])+R[1+1*3]*(nbr_xc[1]-xc[1])+R[1+2*3]*(nbr_xc[2]-xc[2]);
    y[2] = R[2+0*3]*(nbr_xc[0]-xc[0])+R[2+1*3]*(nbr_xc[1]-xc[1])+R[2+2*3]*(nbr_xc[2]-xc[2]);
#endif
}

/** \brief Add contribution to the left and right differences.
 *
*/
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Add2dU(const Real * U, const Real * y, const Real * nbr_U, Real * y2L, Real * y2R, Real * dUL, Real * dUR)
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
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> idx = mesh.geom.InvCellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

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
                Real xc[AMREX_SPACEDIM], un[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_BF_i, nbr_BF_j, nbr_BF_k;
                short nbr_etype;
                Real y[AMREX_SPACEDIM], y2L[AMREX_SPACEDIM], y2R[AMREX_SPACEDIM];
                Real dPL[DG_N_SOL*AMREX_SPACEDIM], dPR[DG_N_SOL*AMREX_SPACEDIM], dP[DG_N_SOL*AMREX_SPACEDIM];
                Real dCL[DG_N_SOL], dCR[DG_N_SOL], dC[DG_N_SOL];

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

                // ELEMENT INFO
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(i,j,k,DG::ELM_CENTROID_X3(dom));
                )
                Fab2P(IBVP.gamma, X_fab, i, j, k, P);

//Print() << "P: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, P);

                // LOOP OVER THE ELEMENT IN THE STENCIL
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    nbr_i = i+DG::base_stencil_table_i[n];
                    nbr_j = j+DG::base_stencil_table_j[n];
                    nbr_k = k+DG::base_stencil_table_k[n];
                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                    DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                    // NEIGHBOR ELEMENT INFO
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                        nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                        nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                    )
                    Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

                    // ADD LS CONTRIBUTION
                    ApplyR(R, xc, nbr_xc, y);
                    Add2dU(P, y, nbr_P, y2L, y2R, dPL, dPR);

//Print() << "nbr_P: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);

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

                    // POINT AND SOLUTION AT THE INTERFACE
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = 0.5*(xm[0]+xc[0]);,
                        nbr_xc[1] = 0.5*(xm[1]+xc[1]);,
                        nbr_xc[2] = 0.5*(xm[2]+xc[2]);
                    )
                    IBVP.F_PHI_BCS(t, nbr_xc, un, P, nbr_P);

                    // MIRRORED ELEMENT INFO
                    AMREX_D_TERM
                    (
                        nbr_xc[0] = xm[0];,
                        nbr_xc[1] = xm[1];,
                        nbr_xc[2] = xm[2];
                    )

                    // ADD LS CONTRIBUTION
                    ApplyR(R, xc, nbr_xc, y);
                    Add2dU(P, y, nbr_P, y2L, y2R, dPL, dPR);

//Print() << "nbr_P: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_P);

//Print() << "HERE WE ARE - EvalSlopesLS_P - large or extended" << std::endl;
//exit(-1);
                }

//Print() << "dPL: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPL);
//Print() << "dPR: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dPR);

                // EVAL DIFFERENCES -----------------------------------
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
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
                }
//Print() << "y2L: " << std::endl; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2L);
//Print() << "y2R: " << std::endl; DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, y2R);
                // ----------------------------------------------------

                // LIMITED SLOPES (VIA CHARACTERISTICS) ---------------
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    AMREX_D_TERM
                    (
                        un[0] = R[d+0*AMREX_SPACEDIM];,
                        un[1] = R[d+1*AMREX_SPACEDIM];,
                        un[2] = R[d+2*AMREX_SPACEDIM];
                    )

                    IBVP.F_DP2DC(t, xc, un, P, &dPL[d*DG_N_SOL], dCL);
                    IBVP.F_DP2DC(t, xc, un, P, &dPR[d*DG_N_SOL], dCR);

                    MCDLimiter(dCL, dCR, dC);

//Print() << "dC: "; DG::IO::PrintRealArray2D(1, DG_N_SOL, dC);

                    IBVP.F_DC2DP(t, xc, un, P, dC, &dP[d*DG_N_SOL]);

                }
                // ----------------------------------------------------

                // STORE IN THE MULTIFAB ------------------------------
                for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
                for (int d1 = 0; d1 < AMREX_SPACEDIM; ++d1)
                for (int ru = 0; ru < DG_N_SOL; ++ru)
                {
                    dXP_fab(i,j,k,d1+AMREX_SPACEDIM*ru) += R[d2+d1*AMREX_SPACEDIM]*dP[ru+d2*DG_N_SOL];
                }
                // ----------------------------------------------------
/*
{
Print() << "R: " << std::endl;
DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, R);

for (int d = 0; d < AMREX_SPACEDIM; ++d)
for (int cu = 0; cu < DG_N_SOL; ++cu)
{
    dP[cu+d*DG_N_SOL] = dXP_fab(i,j,k,d+AMREX_SPACEDIM*cu);
}
Print() << "dP: " << std::endl; DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);

Print() << "HERE WE ARE - EvalSlopesLS_P" << std::endl;
exit(-1);
}
*/
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
 * \param[in] Sol2Dom: Vector containing information regarding over which domain each unknown
 *                     solution field is defined; Sol2Dom[ru] = dom means that the ru-th unknown
 *                     solution field is defined over the dom-th domain.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] dXP: a MultiFab object that contains the limited slopes.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the coefficients of
 *                   the basis functions for the unknown solution fields.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt_BouContrib(const Real t,
                          const DG::ImplicitMesh & mesh,
                          const DG::MatrixFactory & /*matfactory*/,
                          const MultiFab & X,
                          const MultiFab & dX,
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

    // DOMAINS
    const int dom = 0;
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (DG_N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| DG_N_SOL        : "+std::to_string(DG_N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
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
        Array4<Real const> const & dX_fab = dX.array(mfi);

        Array4<Real> const & dXdt_fab = dXdt.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = eType_fab(i,j,k,DG::ELM_TYPE(dom));
#ifdef AMREX_DEBUG
            const Real vf = eInfo_fab(i,j,k,DG::ELM_VOLUME_FRACTION(dom));
#endif

            if (DG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
#ifdef AMREX_DEBUG
                const Real cell_v = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
#endif
                const bool elm_is_small = DG::ELM_IS_SMALL(etype);
                const int merged_b = (elm_is_small) ? (etype/10) : -1;

                // LOCAL VARIABLES
                int BF_i, BF_j, BF_k;
                Real xc[AMREX_SPACEDIM], nbr_xc[AMREX_SPACEDIM];
                Real P[DG_N_SOL], nbr_P[DG_N_SOL], Pr[DG_N_SOL], nbr_Pr[DG_N_SOL];
                int nbr_i, nbr_j, nbr_k, nbr_b;
                short nbr_etype;
                int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real NFn[DG_N_SOL];
                Real integrand;
#ifdef AMREX_DEBUG
                Real elm_v;
#endif

                // SUPPORT OF THE SOLUTION (CURRENT ELEMENT)
                DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                // CURRENT ELEMENT CENTROID
                AMREX_D_TERM
                (
                    xc[0] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X1(dom));,
                    xc[1] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X2(dom));,
                    xc[2] = eInfo_fab(BF_i,BF_j,BF_k,DG::ELM_CENTROID_X3(dom));
                )

                // CURRENT ELEMENT SOLUTION
                Fab2P(IBVP.gamma, X_fab, BF_i, BF_j, BF_k, P);

Print() << "(" << i << "," << j << "," << k << ") -> (" << BF_i << "," << BF_j << "," << BF_k << ")" << std::endl;

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
                        DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG::ELM_TYPE(dom));
                        const bool nbr_is_small = DG::ELM_IS_SMALL(nbr_etype);
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
                        const bool nbr_is_empty = DG::ELM_IS_EMPTY(nbr_etype);
                        if (cells_are_merged || nbr_is_empty) bou_is_not_valid = true;
                    }

                    // FACE QUADRATURE INFO
                    intraphase = true;
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        long pos;
                        DG::GRID_FACE(i, j, k, b, fi, fj, fk);

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
                        // SUPPORT OF THE SOLUTION (NEIGHBOR ELEMENT)
                        DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

Print() << " - nbr(" << nbr_i << "," << nbr_j << "," << nbr_k << ") -> (" << nbr_BF_i << "," << nbr_BF_j << "," << nbr_BF_k << ")" << std::endl;

                        // NEIGHBOR ELEMENT CENTROID
                        AMREX_D_TERM
                        (
                            nbr_xc[0] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X1(dom));,
                            nbr_xc[1] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X2(dom));,
                            nbr_xc[2] = eInfo_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG::ELM_CENTROID_X3(dom));
                        )

                        // NEIGHBOR ELEMENT SOLUTION
                        Fab2P(IBVP.gamma, X_fab, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_P);

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
                            Pr[RHO] = P[RHO];
                            AMREX_D_TERM
                            (
                                Pr[RHOV1] = P[RHOV1];,
                                Pr[RHOV2] = P[RHOV2];,
                                Pr[RHOV3] = P[RHOV3];
                            )
                            Pr[TE] = P[TE];
                            nbr_Pr[RHO] = nbr_P[RHO];
                            AMREX_D_TERM
                            (
                                nbr_Pr[RHOV1] = nbr_P[RHOV1];,
                                nbr_Pr[RHOV2] = nbr_P[RHOV2];,
                                nbr_Pr[RHOV3] = nbr_P[RHOV3];
                            )
                            nbr_Pr[TE] = nbr_P[TE];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )

                                AMREX_D_TERM
                                (
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-nbr_xc[0]);,
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-nbr_xc[1]);,
                                    nbr_Pr[cu] += dX_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-nbr_xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_ICS_P(dom, t, x, un, Pr, nbr_Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
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
                            Pr[RHO] = P[RHO];
                            AMREX_D_TERM
                            (
                                Pr[RHOV1] = P[RHOV1];,
                                Pr[RHOV2] = P[RHOV2];,
                                Pr[RHOV3] = P[RHOV3];
                            )
                            Pr[TE] = P[TE];
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_BCS_P(dom, t, x, un, Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
                            }
                        }
                    }
                }

                // ZERO-LEVEL OF THE LEVEL SET (IF ANY)
                {
                    const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG::ELM_INT_BOU_QUAD_NQ(dom));
                    const long pos = eIntBouQuad_pos_fab(i,j,k,DG::ELM_INT_BOU_QUAD_POS(dom));
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
                            Pr[RHO] = P[RHO];
                            AMREX_D_TERM
                            (
                                Pr[RHOV1] = P[RHOV1];,
                                Pr[RHOV2] = P[RHOV2];,
                                Pr[RHOV3] = P[RHOV3];
                            )
                            Pr[TE] = P[TE];

                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                AMREX_D_TERM
                                (
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*cu)*(x[0]-xc[0]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*cu)*(x[1]-xc[1]);,
                                    Pr[cu] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*cu)*(x[2]-xc[2]);
                                )
                            }

                            // NUMERICAL FLUX
                            IBVP.F_NF_PHI_BCS_P(dom, t, x, un, Pr, NFn);

                            // INTEGRAL CONTRIBUTION
                            for (int ru = 0; ru < DG_N_SOL; ++ru)
                            {
                                integrand = NFn[ru];
                                dXdt_fab(i,j,k,ru) -= integrand*w;
                            }
                        }
                    }
                }

                // INTEGRATION TEST
#ifdef AMREX_DEBUG
                if (std::abs(elm_v/cell_v-vf)*100.0 > 1.0e-2)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_utils.H - Eval_dXdt_BouContrib\n";
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

#define USE_LEAST_SQUARES

/**
 * \brief Eval the time derivative of the solution.
 *
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[in] X: a MultiFab object that contains the solution at the elements' centroids.
 * \param[out] dXdt: a MultiFab object that will contain the time derivative of the solution states.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void Eval_dXdt(const Real t,
               const DG::ImplicitMesh & mesh,
               const DG::MatrixFactory & matfactory,
               MultiFab & X,
               MultiFab & dXdt,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // BOX ARRAY AND DISTRIBUTION MAPPING
    const BoxArray & ba = X.boxarray;
    const DistributionMapping & dm = X.distributionMap;

    const Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, 0);
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dX(ba, dm, AMREX_SPACEDIM*DG_N_SOL, 1);
    // ================================================================

    // INITIALIZATION
    dXdt = 0.0;
    // ==============

#ifdef USE_LEAST_SQUARES
    // FILL THE INVALID CELLS =========================================
    DG::FillBoundaryCells(t, mesh, matfactory, DG_N_SOL, X, IBVP);
    // ================================================================

    // COMPUTE THE SLOPES =============================================
    EvalSlopesLS_P(t, mesh, X, dX, IBVP);
    // ================================================================
#else
    // FILL THE INVALID CELLS =========================================
    DG::FillInvalidCells(t, mesh, matfactory, DG_N_SOL, X, IBVP);
    // ================================================================

    // COMPUTE THE SLOPES =============================================
    EvalSlopes_P(t, mesh, X, dX, IBVP);
    // ================================================================
#endif

    // EVAL THE TIME DERIVATIVES: DOMAIN CONTRIBUTION =================
    // ================================================================

    // EVAL THE TIME DERIVATIVES: BOUNDARY CONTRIBUTION ===============
    Eval_dXdt_BouContrib(t, mesh, matfactory, X, dX, dXdt, IBVP);
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution(mesh, matfactory, DG_N_SOL, Sol2Dom, dXdt);
    MultiplyByInverseMassMatrix(mesh, matfactory, DG_N_SOL, Sol2Dom, dXdt);
    // ================================================================
}

/**
 * \brief Advance in time using 2nd-order Runge-Kutta schemes.
 *
 * \param[in] dt: time step.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operators for the implicitly-defined
 *                        mesh.
 * \param[inout] X_new: a MultiFab object that contains the solution at the elements' centroids; On
 *                      exit, the solution will be updated according to the Runge-Kutta algorithm.
 * \param[in] IBVP: a class that must have methods with the following prototypes:
 *
 *                  XXXX
 *
*/
template <typename IBVP_CLASS>
void TakeRungeKuttaTimeStep(const Real dt, const Real t,
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

    // VARIABLES ======================================================
    MultiFab X_old(ba, dm, n_comp, ngr);

    // AUXILIARY NEEDED MULTIFABS
    MultiFab dXdt(ba, dm, n_comp, ngr);
    MultiFab X1(ba, dm, n_comp, ngr);
    // ================================================================

    // COPY NEW TO OLD ================================================
    MultiFab::Copy(X_old, X_new, 0, 0, n_comp, ngr);
    // ================================================================

    // RK2 ============================================================
    {
        // RK2 STEP 1 -------------------------------------------------
        // dXdt := dXdt(t, X_old)
        Eval_dXdt(t, mesh, matfactory, X_old, dXdt, IBVP);

        // X1 := X_old+dt*dXdt
        MultiFab::LinComb(X1, 1.0, X_old, 0, dt, dXdt, 0, 0, n_comp, ngr);
        // ------------------------------------------------------------

        // RK2 STEP 2 -------------------------------------------------
        // dXdt := dXdt(t+dt, X1)
        Eval_dXdt(t+dt, mesh, matfactory, X1, dXdt, IBVP);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        MultiFab::LinComb(X_new, 0.5, X_old, 0, 0.5, X1, 0, 0, n_comp, ngr);
        MultiFab::Saxpy(X_new, 0.5*dt, dXdt, 0, 0, n_comp, ngr);
        // ------------------------------------------------------------
    }
    // ================================================================

    // CHECK SOLUTION =================================================
    if (X_new.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_utils.H - TakeRungeKuttaTimeStep\n";
        msg += "| Solution contains nans.\n";
        Abort(msg);
    }
    // ================================================================
}

} // namespace FV
} // namespace amrex