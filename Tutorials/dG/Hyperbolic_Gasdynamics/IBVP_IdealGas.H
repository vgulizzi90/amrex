//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_IdealGas.H
 * \brief Contains base constants and data structures for IBVP of ideal gases.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define IDEAL_GAS_NU (1+AMREX_SPACEDIM+1)

// SOLUTION FIELDS ORDER
#if (AMREX_SPACEDIM == 1)
#define RHO 0
#define RHOV1 1
#define TE 2
#endif
#if (AMREX_SPACEDIM == 2)
#define RHO 0
#define RHOV1 1
#define RHOV2 2
#define TE 3
#endif
#if (AMREX_SPACEDIM == 3)
#define RHO 0
#define RHOV1 1
#define RHOV2 2
#define RHOV3 3
#define TE 4
#endif

// AUXILIARY PARAMETERS
#define SMALL_p 1.0e-10
#define SMALL_rho 1.0e-19

#define APPLY_LIMITER
#define APPLY_BJ_V1
//#define APPLY_BJ_V2
#define BJ_USES_P
#define BJ_USES_ICS
// ####################################################################


namespace ideal_gas
{
// SMALL ELEMENT TREATMENT ============================================
/**
 * \brief Copy data from extended elements to their small elements.
 *
 * \param[in] geom: amrex Geometry object; contains the problem periodicity.
 * \param[in] solution: amrex dG Solution object; contains information about element types.
 * \param[inout] mf: the MultiFab object whose components are updated.
 * \param[in] n_comp: number of components of the multifab.
 *
*/
void copy_to_small_elements(const amrex::Geometry & geom, const amrex::dG::Solution & solution, amrex::MultiFab & mf, const int n_comp)
{
    // PARAMETERS -----------------------------------------------------
    const int dom = 0;
    // ----------------------------------------------------------------

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real> const & mf_fab = mf.array(mfi);

        ParallelFor(bx, n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int c) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));
            const bool elm_is_small = amrex::dG::ELM_IS_SMALL(etype);

            // LOCAL VARIABLES
            int b, nbr_i, nbr_j, nbr_k, nbr_b;

            if (elm_is_small)
            {
                b = etype/10;
                amrex::dG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                mf_fab(i,j,k,c) = mf_fab(nbr_i,nbr_j,nbr_k,c);
            }
        });
        amrex::Gpu::synchronize();
    }
    mf.FillBoundary(geom.periodicity());
}
// ====================================================================


// AUXILIARY ROUTINES =================================================
/**
 * \brief Eval primitive variables from conserved variables.
 * 
 * \param[in] g: specific heat ratio.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] rho: density.
 * \param[out] v1: velocity component in x1 direction.
 * \param[out] v2: velocity component in x2 direction (non present if AMREX_SPACEDIM < 2).
 * \param[out] v3: velocity component in x3 direction (non present if AMREX_SPACEDIM < 3).
 * \param[out] p: pressure.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_primitive_variables(const amrex::Real g, const amrex::Real * U,
                              amrex::Real & rho,
                              AMREX_D_DECL(amrex::Real & v1, amrex::Real & v2, amrex::Real & v3),
                              amrex::Real & p,
                              const bool avoid_negative_values = true)
{
    const amrex::Real e = U[TE];

    rho = U[RHO];
    AMREX_D_TERM
    (
        v1 = U[RHOV1]/rho;,
        v2 = U[RHOV2]/rho;,
        v3 = U[RHOV3]/rho;
    )
    p = (g-1.0)*(e-0.5*rho*(AMREX_D_TERM(v1*v1,+v2*v2,+v3*v3)));
    
    if (avoid_negative_values)
    {
        rho = amrex::max(rho, SMALL_rho);
        p = amrex::max(p, SMALL_p);
    }
}

/**
 * \brief Largest wave speed for an ideal gas.
 * 
 * \param[in] g: specific heat ratio.
 * \param[in] U: pointer to memory containing the conserved variables.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_largest_wave_speed(const amrex::Real g, const amrex::Real * U)
{
    amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p, v, a;
    
    eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);
    
    v = std::sqrt(AMREX_D_TERM(v1*v1,+v2*v2,+v3*v3));
    a = std::sqrt(g*p/rho);
    
    return v+a;
}
// ====================================================================


// GOVERNING EQUATIONS ================================================
/**
 * \brief Transform solution from global reference system to unit normal reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] U: pointer to memory containing the conserved variables in the global reference
 *               system.
 * \param[out] Un: pointer to memory that will contain the conserved variables in the unit normal
 *                 reference system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void U_to_Un(const amrex::Real * un, const amrex::Real * U, amrex::Real * Un)
{
#if (AMREX_SPACEDIM == 1)
    Un[RHO] = U[RHO];
    Un[RHOV1] = un[0]*U[RHOV1];
    Un[TE] = U[TE];
#endif
#if (AMREX_SPACEDIM == 2)
    Un[RHO] = U[RHO];
    Un[RHOV1] =  un[0]*U[RHOV1]+un[1]*U[RHOV2];
    Un[RHOV2] = -un[1]*U[RHOV1]+un[0]*U[RHOV2];
    Un[TE] = U[TE];
#endif
#if (AMREX_SPACEDIM == 3)
    // We assume un be written as
    // un = n1 = cos(phi)*sin(theta)
    //      n2   sin(phi)*sin(theta)
    //      n3   cos(theta)
    const amrex::Real theta = std::acos(un[2]);
    const amrex::Real phi = std::atan2(un[1], un[0]);
    const amrex::Real cth = std::cos(theta);
    const amrex::Real sth = std::sin(theta);
    const amrex::Real cph = std::cos(phi);
    const amrex::Real sph = std::sin(phi);

    Un[RHO] = U[RHO];
    Un[RHOV1] =  cph*sth*U[RHOV1]+sph*sth*U[RHOV2]+cth*U[RHOV3];
    Un[RHOV2] =     -sph*U[RHOV1]    +cph*U[RHOV2];
    Un[RHOV3] = -cph*cth*U[RHOV1]-cth*sph*U[RHOV2]+sth*U[RHOV3];
    Un[TE] = U[TE];
#endif
}

/**
 * \brief Transform solution from unit normal reference system to global reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] Un: pointer to memory that containing the conserved variables in the unit normal
 *                reference system.
 * \param[out] U: pointer to memory that will contain the conserved variables in the global
 *                reference system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Un_to_U(const amrex::Real * un, const amrex::Real * Un, amrex::Real * U)
{
#if (AMREX_SPACEDIM == 1)
    U[RHO] = Un[RHO];
    U[RHOV1] = un[0]*Un[RHOV1];
    U[TE] = Un[TE];
#endif
#if (AMREX_SPACEDIM == 2)
    U[RHO] = Un[RHO];
    U[RHOV1] = un[0]*Un[RHOV1]-un[1]*Un[RHOV2];
    U[RHOV2] = un[1]*Un[RHOV1]+un[0]*Un[RHOV2];
    U[TE] = Un[TE];
#endif
#if (AMREX_SPACEDIM == 3)
    // We assume un be written as
    // un = n1 = cos(phi)*sin(theta)
    //      n2   sin(phi)*sin(theta)
    //      n3   cos(theta)
    const amrex::Real theta = std::acos(un[2]);
    const amrex::Real phi = std::atan2(un[1], un[0]);
    const amrex::Real cth = std::cos(theta);
    const amrex::Real sth = std::sin(theta);
    const amrex::Real cph = std::cos(phi);
    const amrex::Real sph = std::sin(phi);

    U[RHO] = Un[RHO];
    U[RHOV1] = cph*sth*Un[RHOV1]-sph*Un[RHOV2]-cph*cth*Un[RHOV3];
    U[RHOV2] = sph*sth*Un[RHOV1]+cph*Un[RHOV2]-cth*sph*Un[RHOV3];
    U[RHOV3] =     cth*Un[RHOV1]                  +sth*Un[RHOV3];
    U[TE] = Un[TE];
#endif
}

/**
 * \brief Eval jumps in the characteristics along x1 from jumps in the primitive variables along x1.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] P: pointer to memory containing the primitive variables.
 * \param[in] dP: pointer to memory containing jumps in the primitive variables.
 * \param[out] dC: pointer to memory that will contain jumps in the characteristics (of the
 *                 primitive variables).
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dP_to_dC_1(const amrex::Real g, const amrex::Real * P,
                const amrex::Real * dP, amrex::Real * dC)
{
    // PARAMETERS
    const amrex::Real rho = amrex::max(P[RHO], SMALL_rho);
    const amrex::Real p = amrex::max(P[TE], SMALL_p);
    const amrex::Real a = std::sqrt(g*p/rho);
    const amrex::Real inv_a = 1.0/a;

#if (AMREX_SPACEDIM == 1)
    dC[RHO]   =        -0.5*inv_a*rho*dP[RHOV1]+0.5*inv_a*inv_a*dP[TE];
    dC[RHOV1] = dP[RHO]                            -inv_a*inv_a*dP[TE];
    dC[TE]    =        +0.5*inv_a*rho*dP[RHOV1]+0.5*inv_a*inv_a*dP[TE];
#endif
#if (AMREX_SPACEDIM == 2)
    dC[RHO]   =        -0.5*inv_a*rho*dP[RHOV1]              +0.5*inv_a*inv_a*dP[TE];
    dC[RHOV1] = dP[RHO]                                          -inv_a*inv_a*dP[TE];
    dC[RHOV2] =                                +rho*dP[RHOV2];
    dC[TE]    =        +0.5*inv_a*rho*dP[RHOV1]              +0.5*inv_a*inv_a*dP[TE];
#endif
#if (AMREX_SPACEDIM == 3)
    dC[RHO]   =        -0.5*inv_a*rho*dP[RHOV1]                            +0.5*inv_a*inv_a*dP[TE];
    dC[RHOV1] = dP[RHO]                                                        -inv_a*inv_a*dP[TE];
    dC[RHOV2] =                                +rho*dP[RHOV2];
    dC[RHOV3] =                                              +rho*dP[RHOV3];
    dC[TE]    =        +0.5*inv_a*rho*dP[RHOV1]                            +0.5*inv_a*inv_a*dP[TE];
#endif  
}
/**
 * \brief Eval jumps in the characteristics from jumps in the primitive variables.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
 * \param[in] P: pointer to memory containing the primitive variables.
 * \param[in] dP: pointer to memory containing jumps in the primitive variables.
 * \param[out] dC: pointer to memory that will contain jumps in the characteristics (of the
 *                 primitive variables).
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dP_to_dC(const amrex::Real g, const amrex::Real * un, const amrex::Real * P,
              const amrex::Real * dP, amrex::Real * dC)
{
    // VARIABLES
    amrex::Real Pn[IDEAL_GAS_NU], dPn[IDEAL_GAS_NU];
    
    U_to_Un(un, P, Pn);
    U_to_Un(un, dP, dPn);
    dP_to_dC_1(g, Pn, dPn, dC);
}

/**
 * \brief Eval jumps in the primitive variables along x1 from jumps in the characteristics along x1.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] P: pointer to memory containing the primitive variables.
 * \param[in] dC: pointer to memory containing jumps in the characteristics (of the primitive
 *                variables).
 * \param[out] dP: pointer to memory that will contain jumps in the primitive variables.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dC_to_dP_1(const amrex::Real g, const amrex::Real * P,
                const amrex::Real * dC, amrex::Real * dP)
{
    // PARAMETERS
    const amrex::Real rho = amrex::max(P[RHO], SMALL_rho);
    const amrex::Real p = amrex::max(P[TE], SMALL_p);
    const amrex::Real a = std::sqrt(g*p/rho);
    const amrex::Real inv_rho = 1.0/rho;
    
#if (AMREX_SPACEDIM == 1)
    dP[RHO]   =            dC[RHO]+dC[RHOV1]          +dC[TE];
    dP[RHOV1] = -inv_rho*a*dC[RHO]          +inv_rho*a*dC[TE];
    dP[TE]    =        a*a*dC[RHO]                +a*a*dC[TE];
#endif
#if (AMREX_SPACEDIM == 2)
    dP[RHO]   =            dC[RHO]+dC[RHOV1]                            +dC[TE];
    dP[RHOV1] = -inv_rho*a*dC[RHO]                            +inv_rho*a*dC[TE];
    dP[RHOV2] =                             +inv_rho*dC[RHOV2];
    dP[TE]    =        a*a*dC[RHO]                                  +a*a*dC[TE];
#endif
#if (AMREX_SPACEDIM == 3)
    dP[RHO]   =            dC[RHO]+dC[RHOV1]                                              +dC[TE];
    dP[RHOV1] = -inv_rho*a*dC[RHO]                                              +inv_rho*a*dC[TE];
    dP[RHOV2] =                             +inv_rho*dC[RHOV2];
    dP[RHOV3] =                                               +inv_rho*dC[RHOV3];
    dP[TE]    =        a*a*dC[RHO]                                                    +a*a*dC[TE];
#endif
}

/**
 * \brief Eval the jumps in the primitive variables from jumps in the characteristics.
 *
 * \param[in] t: time.
 * \param[in] x: pointer to memory containing the space location.
 * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
 * \param[in] P: pointer to memory containing the primitive variables.
 * \param[in] dC: pointer to memory containing jumps in the characteristics (of the primitive
 *                variables).
 * \param[out] dP: pointer to memory that will contain jumps in the primitive variables.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dC_to_dP(const amrex::Real g, const amrex::Real * un, const amrex::Real * P,
              const amrex::Real * dC, amrex::Real * dP)
{
    // VARIABLES
    amrex::Real Pn[IDEAL_GAS_NU], dPn[IDEAL_GAS_NU];

    U_to_Un(un, P, Pn);
    dC_to_dP_1(g, Pn, dC, dPn);
    Un_to_U(un, dPn, dP);
}

/**
 * \brief Eval jumps in the conserved variables from jumps in the primitive variables.
 *
 * \param[in] t: time.
 * \param[in] x: pointer to memory containing the space location.
 * \param[in] U: pointer to memory containing the unknown solution fields.
 * \param[in] dP: pointer to memory containing jumps in the primitive variables.
 * \param[out] dU: pointer to memory that will contain jumps in the conserved variables.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dP_to_dU(const amrex::Real g, const amrex::Real * U,
              const amrex::Real * dP, amrex::Real * dU)
{
    // PARAMETERS
    const amrex::Real ig1 = 1.0/(g-1.0);

    // VARIABLES
    amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
    eval_primitive_variables(g, U, rho, AMREX_D_DECL(u1, u2, u3), p);

#if (AMREX_SPACEDIM == 1)
    dU[RHO]   =           dP[RHO];
    dU[RHOV1] =        u1*dP[RHO]   +rho*dP[RHOV1];
    dU[TE]    = 0.5*u1*u1*dP[RHO]+rho*u1*dP[RHOV1]+ig1*dP[TE];
#endif
#if (AMREX_SPACEDIM == 2)
    dU[RHO]   =                   dP[RHO];
    dU[RHOV1] =                u1*dP[RHO]   +rho*dP[RHOV1];
    dU[RHOV2] =                u2*dP[RHO]                    +rho*dP[RHOV2];
    dU[TE]    = 0.5*(u1*u1+u2*u2)*dP[RHO]+rho*u1*dP[RHOV1]+rho*u2*dP[RHOV2]+ig1*dP[TE];
#endif
#if (AMREX_SPACEDIM == 3)
    dU[RHO]   =                         dP[RHO];
    dU[RHOV1] =                      u1*dP[RHO]   +rho*dP[RHOV1];
    dU[RHOV2] =                      u2*dP[RHO]                    +rho*dP[RHOV2];
    dU[RHOV3] =                      u3*dP[RHO]                                     +rho*dP[RHOV3];
    dU[TE]    = 0.5*(u1*u1+u2*u2+u3*u3)*dP[RHO]+rho*u1*dP[RHOV1]+rho*u2*dP[RHOV2]+rho*u3*dP[RHOV3]+ig1*dP[TE];
#endif
}

/**
 * \brief Eval the fluxes for the considered IBVP of an ideal gas.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_F(const amrex::Real g, const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3))
{
    amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p;
    eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);

    F1[RHO] = U[RHOV1];
    AMREX_D_TERM
    (
        F1[RHOV1] = U[RHOV1]*v1+p;,
        F1[RHOV2] = U[RHOV2]*v1;,
        F1[RHOV3] = U[RHOV3]*v1;
    )
    F1[TE] = (U[TE]+p)*v1;

#if (AMREX_SPACEDIM > 1)
    F2[RHO] = U[RHOV2];
    AMREX_D_TERM
    (
        F2[RHOV1] = U[RHOV1]*v2;,
        F2[RHOV2] = U[RHOV2]*v2+p;,
        F2[RHOV3] = U[RHOV3]*v2;
    )
    F2[TE] = (U[TE]+p)*v2;
#endif

#if (AMREX_SPACEDIM > 2)
    F3[RHO] = U[RHOV3];
    AMREX_D_TERM
    (
        F3[RHOV1] = U[RHOV1]*v3;,
        F3[RHOV2] = U[RHOV2]*v3;,
        F3[RHOV3] = U[RHOV3]*v3+p;
    )
    F3[TE] = (U[TE]+p)*v3;
#endif
}

/**
 * \brief Eval the flux F1 for the considered IBVP of an ideal gas.
 *
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] F1: pointer to memory that will contain the flux.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_F1(const amrex::Real g, const amrex::Real * U, amrex::Real * F1)
{
    amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p;
    eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);

    F1[RHO] = U[RHOV1];
    AMREX_D_TERM
    (
        F1[RHOV1] = U[RHOV1]*v1+p;,
        F1[RHOV2] = U[RHOV2]*v1;,
        F1[RHOV3] = U[RHOV3]*v1;
    )
    F1[TE] = (U[TE]+p)*v1;
}
// ====================================================================


// RIEMANN SOLVER =====================================================
/**
 * \brief Solves the Riemann problem between left and right states in unit normal reference system.
 *
 * \param[in] (rl, ul, ut1l, ut2l, pl): primitive variables at the left of the shock.
 * \param[in] (rr, ur, ut1r, ut2r, pr): primitive variables at the right of the shock.
 * \param[out] URP: pointer to memory that will contain the solution of the Riemann problem, i.e.
 *                  the state at the shock.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void two_shock_Riemann_solver(const amrex::Real gamma,
                              const amrex::Real rl, AMREX_D_DECL(const amrex::Real ul, const amrex::Real ut1l, const amrex::Real ut2l), const amrex::Real pl,
                              const amrex::Real rr, AMREX_D_DECL(const amrex::Real ur, const amrex::Real ut1r, const amrex::Real ut2r), const amrex::Real pr,
                              amrex::Real * URP)
{
    const amrex::Real weakwv = 1.e-03, small = 1.e-6;
    const int itno = 3;

    const amrex::Real clsql = gamma*pl*rl;
    const amrex::Real clsqr = gamma*pr*rr;
    amrex::Real wl = std::sqrt(clsql);
    amrex::Real wr = std::sqrt(clsqr);
    const amrex::Real cleft = wl/rl;
    const amrex::Real cright = wr/rr;
    
    amrex::Real pstar, pstnm1, wlsq, wrsq, ustarp, ustarm, ustnm1, ustnp1;
    amrex::Real dpditer, zp, zm, denom;
    amrex::Real ustar;

    amrex::Real ro, uo, po, sgnm AMREX_D_DECL(, utrans1, utrans2);
    amrex::Real wosq, co, wo, dpjmp, rstar, cstar, spout, spin, frac;
    amrex::Real rgdnv, ugdnv, pgdnv;

    pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar, SMALL_p);
    pstnm1 = pstar;

    wlsq = (0.5*(gamma-1.0)*(pstar+pl)+pstar)*rl;
    wrsq = (0.5*(gamma-1.0)*(pstar+pr)+pstar)*rr;

    wl = std::sqrt(wlsq);
    wr = std::sqrt(wrsq);

    ustarp = ul-(pstar-pl)/wl;
    ustarm = ur+(pstar-pr)/wr;

    pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
    pstar = amrex::max(pstar, SMALL_p);

    for (int it = 0; it < itno; ++it)
    {
        wlsq = (0.5*(gamma-1.0)*(pstar+pl)+pstar)*rl;
        wrsq = (0.5*(gamma-1.0)*(pstar+pr)+pstar)*rr;

        wl = 1.0/std::sqrt(wlsq);
        wr = 1.0/std::sqrt(wrsq);

        ustnm1 = ustarm;
        ustnp1 = ustarp;

        ustarm = ur - (pr - pstar)*wr;
        ustarp = ul + (pl - pstar)*wl;

        dpditer = std::abs(pstnm1-pstar);
        zp = std::abs(ustarp-ustnp1);
        if (zp-weakwv*cleft < 0.0 )
        {
            zp = dpditer*wl;
        }
        zm = std::abs(ustarm-ustnm1);
        if (zm-weakwv*cright < 0.0 )
        {
            zm = dpditer*wr;
        }

        denom = dpditer/amrex::max(zp+zm,small*(cleft+cright));
        pstnm1 = pstar;
        pstar = pstar - denom*(ustarm-ustarp);
        pstar = amrex::max(pstar,SMALL_p);
        ustar = 0.5*(ustarm+ustarp);
    }

    if (ustar > 0.0)
    {
        ro = rl;
        uo = ul;
        po = pl;
        sgnm = 1.0;
        AMREX_D_TERM
        (
            ,
            utrans1 = ut1l;,
            utrans2 = ut2l;
        )
    }
    else if (ustar < 0.0)
    {
        ro = rr;
        uo = ur;
        po = pr;
        sgnm = -1.0;
        AMREX_D_TERM
        (
            ,
            utrans1 = ut1r;,
            utrans2 = ut2r;
        )
    }
    else
    {
        uo = 0.50*(ur+ul);
        po = 0.50*(pr+pl);
        ro = 2.0*(rl*rr)/(rl+rr);
        sgnm = 1.0;
        AMREX_D_TERM
        (
            ,
            utrans1 = 0.50*(ut1l+ut1r);,
            utrans2 = 0.50*(ut2l+ut2r);
        )
    }
    
    wosq = (.5*(gamma-1.0)*(pstar+po)+pstar)*ro;
    co = std::sqrt(gamma * po / ro);
    wo = std::sqrt(wosq);
    dpjmp = pstar-po;
    rstar = ro/(1.0-ro*dpjmp/wosq);
    cstar = std::sqrt(gamma * pstar / rstar);
    spout = co-sgnm*uo;
    spin = cstar - sgnm*uo;

    if (pstar >= po)
    {
        spin = wo/ro-sgnm*uo;
        spout = spin;
    }
    frac = 0.50*(1.0+(spin+spout)/amrex::max(spout-spin,amrex::max(spin+spout,small*(cleft+cright))));

    if (spout < 0.0)
    {
        rgdnv = ro;
        ugdnv = uo;
        pgdnv = po;
    }
    else if (spin >= 0.0)
    {
        rgdnv = rstar;
        ugdnv = ustar;
        pgdnv = pstar;
    }
    else
    {
        rgdnv = frac*rstar + (1.0 - frac)* ro;
        ugdnv = frac*ustar + (1.0 - frac)* uo;
        pgdnv = frac*pstar + (1.0 - frac)* po;
    }

    URP[RHO] = rgdnv;
    AMREX_D_TERM
    (
        URP[RHOV1] = rgdnv*ugdnv;,
        URP[RHOV2] = rgdnv*utrans1;,
        URP[RHOV3] = rgdnv*utrans2;
    )
    URP[TE] = pgdnv/(gamma-1.0)+0.5*rgdnv*(AMREX_D_TERM(ugdnv*ugdnv,+utrans1*utrans1,+utrans2*utrans2));
}
// ====================================================================


// DG =================================================================
/**
 * \brief Eval the numerical flux at a grid's boundary.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
 *                 unit normal.
 * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
 *                 unit normal.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver(const amrex::Real g, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn)
{
    // VARIABLES ------------------------------------------------------
    amrex::Real tmp[IDEAL_GAS_NU];
    amrex::Real rl, AMREX_D_DECL(ul, ut1l, ut2l), pl;
    amrex::Real rr, AMREX_D_DECL(ur, ut1r, ut2r), pr;
    // ----------------------------------------------------------------

    // EVAL PRIMITIVE VARIABLES IN THE UNIT NORMAL REFERENCE SYSTEM ---
    U_to_Un(un, m_U, tmp);
    eval_primitive_variables(g, tmp, rl, AMREX_D_DECL(ul, ut1l, ut2l), pl);

    U_to_Un(un, p_U, tmp);
    eval_primitive_variables(g, tmp, rr, AMREX_D_DECL(ur, ut1r, ut2r), pr);
    // ----------------------------------------------------------------

    // SOLVE RIEMANN PROBLEM ------------------------------------------
    // We use NFn as a temporary variable
    two_shock_Riemann_solver(g, rl, AMREX_D_DECL(ul, ut1l, ut2l), pl, rr, AMREX_D_DECL(ur, ut1r, ut2r), pr, NFn);
    // ----------------------------------------------------------------

    // EVAL THE FLUX --------------------------------------------------
    // Unit normal reference system
    eval_F1(g, NFn, tmp);

    // Global reference system
    Un_to_U(un, tmp, NFn);
    // ----------------------------------------------------------------
}

/**
 * \brief Eval the numerical flux at a boundary to enforce wall bcs.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_wall(const amrex::Real g, const amrex::Real * un,
                   const amrex::Real * U, amrex::Real * NFn)
{
    amrex::Real U_ghost[IDEAL_GAS_NU];

    // We use NFn as a temporary variable
    U_to_Un(un, U, NFn);
    NFn[RHOV1] = -NFn[RHOV1];
    Un_to_U(un, NFn, U_ghost);

    eval_NFn_Riemann_solver(g, un, U, U_ghost, NFn);
}
// ====================================================================


// FV =================================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_centroid_and_conserved_variables(const int i, const int j, const int k, const int dom,
                                          const amrex::Array4<const amrex::Real> & elm_centroid_fab,
                                          const amrex::Array4<const amrex::Real> & X_fab,
                                          amrex::Real * xc, amrex::Real * U)
{
    // CENTROID
    AMREX_D_TERM
    (
        xc[0] = elm_centroid_fab(i,j,k,amrex::dG::ELM_CENTROID_X1(dom));,
        xc[1] = elm_centroid_fab(i,j,k,amrex::dG::ELM_CENTROID_X2(dom));,
        xc[2] = elm_centroid_fab(i,j,k,amrex::dG::ELM_CENTROID_X3(dom));
    )

    // CONSERVED VARIABLES
    U[RHO] = X_fab(i,j,k,RHO);
    AMREX_D_TERM
    (
        U[RHOV1] = X_fab(i,j,k,RHOV1);,
        U[RHOV2] = X_fab(i,j,k,RHOV2);,
        U[RHOV3] = X_fab(i,j,k,RHOV3);
    )
    U[TE] = X_fab(i,j,k,TE);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void get_centroid_and_primitive_variables(const amrex::Real g,
                                          const int i, const int j, const int k, const int dom,
                                          const amrex::Array4<const amrex::Real> & elm_centroid_fab,
                                          const amrex::Array4<const amrex::Real> & X_fab,
                                          amrex::Real * xc, amrex::Real * P)
{
    // VARIABLES
    amrex::Real U[IDEAL_GAS_NU];
    amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p;

    // CENTROID AND CONSERVED VARIABLES
    get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);

    // PRIMITIVE VARIABLES
    eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);
    P[RHO] = rho;
    AMREX_D_TERM
    (
        P[RHOV1] = v1;,
        P[RHOV2] = v2;,
        P[RHOV3] = v3;
    )
    P[TE] = p;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void update_least_square_system(const amrex::Real * xc, const amrex::Real * P,
                                const amrex::Real * nbr_xc, const amrex::Real * nbr_P,
                                amrex::Real * S, amrex::Real * rhs)
{
#if (AMREX_SPACEDIM == 1)
    S[0] += (nbr_xc[0]-xc[0])*(nbr_xc[0]-xc[0]);
#endif
#if (AMREX_SPACEDIM == 2)
    S[0+0*2] += (nbr_xc[0]-xc[0])*(nbr_xc[0]-xc[0]);
    S[0+1*2] += (nbr_xc[0]-xc[0])*(nbr_xc[1]-xc[1]);
    S[1+0*2]  = S[0+1*2];
    S[1+1*2] += (nbr_xc[1]-xc[1])*(nbr_xc[1]-xc[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    S[0+0*3] += (nbr_xc[0]-xc[0])*(nbr_xc[0]-xc[0]);
    S[0+1*3] += (nbr_xc[0]-xc[0])*(nbr_xc[1]-xc[1]);
    S[0+2*3] += (nbr_xc[0]-xc[0])*(nbr_xc[2]-xc[2]);
    S[1+0*3]  = S[0+1*3];
    S[2+0*3]  = S[0+2*3];
    S[1+1*3] += (nbr_xc[1]-xc[1])*(nbr_xc[1]-xc[1]);
    S[1+2*3] += (nbr_xc[1]-xc[1])*(nbr_xc[2]-xc[2]);
    S[2+1*3]  = S[1+2*3];
    S[2+2*3] += (nbr_xc[2]-xc[2])*(nbr_xc[2]-xc[2]);
#endif
    for (int u = 0; u < IDEAL_GAS_NU; ++u)
    {
        AMREX_D_TERM
        (
            rhs[0+u*AMREX_SPACEDIM] += (nbr_xc[0]-xc[0])*(nbr_P[u]-P[u]);,
            rhs[1+u*AMREX_SPACEDIM] += (nbr_xc[1]-xc[1])*(nbr_P[u]-P[u]);,
            rhs[2+u*AMREX_SPACEDIM] += (nbr_xc[2]-xc[2])*(nbr_P[u]-P[u]);
        )
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void limit_slopes(amrex::Real * slopes, const amrex::Real * tmp_slopes, const amrex::Real theta)
{
    for (int u = 0; u < IDEAL_GAS_NU; ++u)
    {
        const amrex::Real slp = slopes[u];
        const amrex::Real sgn = (slp > 0.0) ? +1.0 : -1.0;
        const amrex::Real tmp_slp = tmp_slopes[u];

        if (tmp_slp*slp > 0.0)
        {
            slopes[u] = sgn*amrex::min(std::abs(slp), theta*std::abs(tmp_slp));
        }
        else
        {
            slopes[u] = 0.0;
        }
    }
}

/**
 * \brief Reconstruct and limit the slopes for FV schemes.
 *
 * \param[in] g: specific heat ratio.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution states.
 * \param[out] dX: a MultiFabs that will contain the solution slopes.
 *
*/
void eval_FV_slopes(const amrex::Real g,
                    const amrex::Geometry & geom,
                    const amrex::dG::Mesh & mesh,
                    const amrex::dG::Solution & solution,
                    const amrex::MultiFab & X,
                    amrex::MultiFab & dX)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ideal_gas::eval_FV_slopes(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const amrex::Box domain = geom.Domain();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic = geom.isPeriodicArray();

    // DOMAINS
    const int dom = 0;
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    dX = 0.0;
    // ----------------------------------------------------------------

    // EVAL THE SLOPES
    for (amrex::MFIter mfi(dX); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
        
        amrex::Array4<amrex::Real> const & dX_fab = dX.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_VALID(etype))
            {
                // LOCAL PARAMETERS
#ifdef APPLY_LIMITER
                const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                          prob_lo[1]+j*dx[1],
                                                                          prob_lo[2]+k*dx[2])};
#endif

                // LOCAL VARIABLES
                amrex::Real xc[AMREX_SPACEDIM];
                amrex::Real U[IDEAL_GAS_NU], P[IDEAL_GAS_NU];
                int n_nbr;
                int nbr_is_valid[__DG_BASE_STENCIL_N_NBR__];
                amrex::Real nbr_Ps[IDEAL_GAS_NU*__DG_BASE_STENCIL_N_NBR__];
                amrex::Real nbr_xcs[AMREX_SPACEDIM*__DG_BASE_STENCIL_N_NBR__];
                amrex::Real S[AMREX_SPACEDIM*AMREX_SPACEDIM], rhs[AMREX_SPACEDIM*IDEAL_GAS_NU];
                amrex::Real dP[AMREX_SPACEDIM*IDEAL_GAS_NU];
#ifdef APPLY_LIMITER
                amrex::Real tmp_dP[IDEAL_GAS_NU], tmp_dC[IDEAL_GAS_NU];
                amrex::Real theta;
#endif

                // CONSERVED VARIABLES
#if defined(BJ_USES_P) && defined(APPLY_LIMITER)
#else
                U[RHO] = X_fab(i,j,k,RHO);
                AMREX_D_TERM
                (
                    U[RHOV1] = X_fab(i,j,k,RHOV1);,
                    U[RHOV2] = X_fab(i,j,k,RHOV2);,
                    U[RHOV3] = X_fab(i,j,k,RHOV3);
                )
                U[TE] = X_fab(i,j,k,TE);
#endif

                // ELEMENT'S CENTROID AND PRIMITIVE VARIABLES
                get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);

                // INITIALIZATION
                for (int u = 0; u < IDEAL_GAS_NU*__DG_BASE_STENCIL_N_NBR__; ++u)
                {
                    nbr_Ps[u] = 0.0;
                }
                for (int u = 0; u < AMREX_SPACEDIM*__DG_BASE_STENCIL_N_NBR__; ++u)
                {
                    nbr_xcs[u] = 0.0;
                }

                // COLLECT NEIGHBORING ELEMENTS INFORMATION -----------
                n_nbr = 0;
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::base_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::base_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::base_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    const bool nbr_is_outside_domain = !domain.contains(nbr_i, nbr_j, nbr_k);
                    const bool nbr_is_empty = amrex::dG::ELM_IS_EMPTY(nbr_etype);
                    const bool nbr_is_small = amrex::dG::ELM_IS_SMALL(nbr_etype);
                    const bool cells_are_merged = amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype);

                    // Good for nothing
                    if ((nbr_is_outside_domain && (is_periodic[n/2] == 0)) || nbr_is_empty || (nbr_is_small && cells_are_merged))
                    {
                        nbr_is_valid[n] = 0;
                    }
                    // Good for least-square reconstruction only
                    else if (nbr_is_small)
                    {
                        nbr_is_valid[n] = 1;
                    }
                    // Good for least-square reconstruction and slope
                    // limiting
                    else
                    {
                        nbr_is_valid[n] = 2;
                    }
                    
                    if (nbr_is_valid[n] > 0)
                    {
                        get_centroid_and_primitive_variables(g, nbr_i, nbr_j, nbr_k, dom, elm_centroid_fab, X_fab, &nbr_xcs[AMREX_SPACEDIM*n], &nbr_Ps[IDEAL_GAS_NU*n]);
                        n_nbr += 1;
                    }
                }

/*
if (i == __i__ && j == __j__ && k == __k__)
{
amrex::Real nbr_Us[IDEAL_GAS_NU*__DG_BASE_STENCIL_N_NBR__];

amrex::Print() << "xc: " << std::endl;
amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, 1, xc);
amrex::Print() << "U: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, 1, U);
amrex::Print() << "P: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, 1, P);
amrex::Print() << "nbr_is_valid: " << std::endl;
for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
{
    const int nbr_i = i+amrex::dG::tables::base_stencil_i[n];
    const int nbr_j = j+amrex::dG::tables::base_stencil_j[n];
    const int nbr_k = k+amrex::dG::tables::base_stencil_k[n];
    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
    amrex::Print() << nbr_is_valid[n] << std::endl;

    get_centroid_and_conserved_variables(nbr_i, nbr_j, nbr_k, dom, elm_centroid_fab, X_fab, &nbr_xcs[AMREX_SPACEDIM*n], &nbr_Us[IDEAL_GAS_NU*n]);
}
amrex::Print() << "nbr_xcs: " << std::endl;
amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, __DG_BASE_STENCIL_N_NBR__, nbr_xcs);
amrex::Print() << "nbr_Us: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, __DG_BASE_STENCIL_N_NBR__, nbr_Us);
amrex::Print() << "nbr_Ps: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, __DG_BASE_STENCIL_N_NBR__, nbr_Ps);
//exit(-1);
}
*/
                // ----------------------------------------------------

                // EVAL LEAST-SQUARE RECONSTRUCTION -------------------
                if (n_nbr >= AMREX_SPACEDIM)
                {
                    for (int u = 0; u < AMREX_SPACEDIM*AMREX_SPACEDIM; ++u) S[u] = 0.0;
                    for (int u = 0; u < AMREX_SPACEDIM*IDEAL_GAS_NU; ++u) rhs[u] = 0.0;

                    for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                    {
                        if (nbr_is_valid[n] > 0)
                        {
                            update_least_square_system(xc, P, &nbr_xcs[AMREX_SPACEDIM*n], &nbr_Ps[IDEAL_GAS_NU*n], S, rhs);
                        }
                    }

#if (AMREX_SPACEDIM == 1)
                    for (int u = 0; u < IDEAL_GAS_NU; ++u) dP[u] = rhs[u]/S[0];
#endif
#if (AMREX_SPACEDIM == 2)
                    amrex::dG::linalg::dpo2s(IDEAL_GAS_NU, S, rhs, dP);
#endif
#if (AMREX_SPACEDIM == 3)
                    amrex::dG::linalg::dpo3s(IDEAL_GAS_NU, S, rhs, dP);
#endif

                }
                else
                {
                    for (int u = 0; u < AMREX_SPACEDIM*IDEAL_GAS_NU; ++u)
                    {
                        dP[u] = 0.0;
                    }
                }

                amrex::dG::linalg::transpose(AMREX_SPACEDIM, IDEAL_GAS_NU, dP, rhs);
                for (int u = 0; u < IDEAL_GAS_NU*AMREX_SPACEDIM; ++u)
                {
                    dP[u] = rhs[u];
                }
                // ----------------------------------------------------

/*
if (i == __i__ && j == __j__ && k == __k__)
//if (amrex::dG::linalg::contains_nan(IDEAL_GAS_NU*AMREX_SPACEDIM, dP))
{
amrex::Print() << "dP(" << i << "," << j << "," << k << "): " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, AMREX_SPACEDIM, dP);
//exit(-1);
}
*/

#ifdef APPLY_LIMITER
                // LIMIT THE COMPUTED SLOPES --------------------------
                // STORE THE CHARACTERISTICS CENTRAL DIFFERENCE IN rhs
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    amrex::Real un[AMREX_SPACEDIM];
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = +1.0;
                    
                    dP_to_dC(g, un, P, &dP[dir*IDEAL_GAS_NU], &rhs[dir*IDEAL_GAS_NU]);
                }

                // X1 DIRECTION
                {
                    const int dir = 0;
                    const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(1.0, 0.0, 0.0)};

                    for (int n = 2*dir; n < 2*(dir+1); ++n)
                    {
                        if (nbr_is_valid[n] > 1)
                        {
                            // SLOPE COEFFICIENT
                            theta = (nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0])/(cell_lo[0]+(n%2)*dx[0]-xc[0]);

                            // RECONSTRUCTED DIFFERENCE
                            for (int u = 0; u < IDEAL_GAS_NU; ++u)
                            {
                                tmp_dP[u] = -P[u];
                                AMREX_D_TERM
                                (
                                    ,
                                    tmp_dP[u] -= dP[u+1*IDEAL_GAS_NU]*(nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);,
                                    tmp_dP[u] -= dP[u+2*IDEAL_GAS_NU]*(nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                                )
                                tmp_dP[u] += nbr_Ps[u+n*IDEAL_GAS_NU];
                                tmp_dP[u] /= (nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);
                            }

                            // CHARACTERISTICS
                            dP_to_dC(g, un, P, tmp_dP, tmp_dC);

                            // LIMIT SLOPES
                            limit_slopes(&rhs[dir*IDEAL_GAS_NU], tmp_dC, theta);
                        }
                    }
                }
#if (AMREX_SPACEDIM > 1)
                // X2 DIRECTION
                {
                    const int dir = 1;
                    const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 1.0, 0.0)};

                    for (int n = 2*dir; n < 2*(dir+1); ++n)
                    {
                        if (nbr_is_valid[n] > 1)
                        {
                            // SLOPE COEFFICIENT
                            theta = (nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1])/(cell_lo[1]+(n%2)*dx[1]-xc[1]);

                            // RECONSTRUCTED DIFFERENCE
                            for (int u = 0; u < IDEAL_GAS_NU; ++u)
                            {
                                tmp_dP[u] = -P[u];
                                AMREX_D_TERM
                                (
                                    tmp_dP[u] -= dP[u+0*IDEAL_GAS_NU]*(nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);,
                                    ,
                                    tmp_dP[u] -= dP[u+2*IDEAL_GAS_NU]*(nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                                )
                                tmp_dP[u] += nbr_Ps[u+n*IDEAL_GAS_NU];
                                tmp_dP[u] /= (nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);
                            }

                            // CHARACTERISTICS
                            dP_to_dC(g, un, P, tmp_dP, tmp_dC);

                            // LIMIT SLOPES
                            limit_slopes(&rhs[dir*IDEAL_GAS_NU], tmp_dC, theta);
                        }
                    }
                }
#endif
#if (AMREX_SPACEDIM > 2)
                // X3 DIRECTION
                {
                    const int dir = 2;
                    const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 1.0)};

                    for (int n = 2*dir; n < 2*(dir+1); ++n)
                    {
                        if (nbr_is_valid[n] > 1)
                        {
                            // SLOPE COEFFICIENT
                            theta = (nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2])/(cell_lo[2]+(n%2)*dx[2]-xc[2]);

                            // RECONSTRUCTED DIFFERENCE
                            for (int u = 0; u < IDEAL_GAS_NU; ++u)
                            {
                                tmp_dP[u] = -P[u];
                                AMREX_D_TERM
                                (
                                    tmp_dP[u] -= dP[u+0*IDEAL_GAS_NU]*(nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);,
                                    tmp_dP[u] -= dP[u+1*IDEAL_GAS_NU]*(nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);,
                                    
                                )
                                tmp_dP[u] += nbr_Ps[u+n*IDEAL_GAS_NU];
                                tmp_dP[u] /= (nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                            }

                            // CHARACTERISTICS
                            dP_to_dC(g, un, P, tmp_dP, tmp_dC);

                            // LIMIT SLOPES
                            limit_slopes(&rhs[dir*IDEAL_GAS_NU], tmp_dC, theta);
                        }
                    }
                }
#endif
                // BACK TO PRIMITIVE VARIABLES
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                {
                    amrex::Real un[AMREX_SPACEDIM];
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[dir] = +1.0;
                    
                    dC_to_dP(g, un, P, &rhs[dir*IDEAL_GAS_NU], &dP[dir*IDEAL_GAS_NU]);
                }
                // ----------------------------------------------------
#endif

                // STORE THE SLOPES -----------------------------------
#if defined(BJ_USES_P) && defined(APPLY_LIMITER)
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    AMREX_D_TERM
                    (
                        dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = dP[u+0*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = dP[u+1*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = dP[u+2*IDEAL_GAS_NU];
                    )
                }
#else

                dP_to_dU(g, U, &dP[0*IDEAL_GAS_NU], &rhs[0*IDEAL_GAS_NU]);
#if (AMREX_SPACEDIM > 1)
                dP_to_dU(g, U, &dP[1*IDEAL_GAS_NU], &rhs[1*IDEAL_GAS_NU]);
#endif
#if (AMREX_SPACEDIM > 2)
                dP_to_dU(g, U, &dP[2*IDEAL_GAS_NU], &rhs[2*IDEAL_GAS_NU]);
#endif

/*
if (i == __i__ && j == __j__ && k == __k__)
{
amrex::Print() << "dP: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, AMREX_SPACEDIM, dP);
amrex::Print() << "dU: " << std::endl;
amrex::dG::io::print_real_array_2d(IDEAL_GAS_NU, AMREX_SPACEDIM, rhs);
//exit(-1);
}
*/

                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    AMREX_D_TERM
                    (
                        dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = rhs[u+0*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = rhs[u+1*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = rhs[u+2*IDEAL_GAS_NU];
                    )
                }
#endif
                // ----------------------------------------------------
            }
        });
        amrex::Gpu::synchronize();
    }
    dX.FillBoundary(geom.periodicity());

    // COPY THE SLOPES OF THE EXTENDED ELEMENTS TO THEIR SMALL ELEMENTS
    copy_to_small_elements(geom, solution, dX, dX.n_comp);
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    if (dX.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_IdealGas.H - eval_FV_slopes\n";
        msg += "| Reconstructed slopes multifab dX contains nans prior to BJ limiter.\n";
        amrex::Abort(msg);
    }
    // ----------------------------------------------------------------
#endif

    // APPLY THE BARTH JESPERSEN LIMITER
#ifdef APPLY_LIMITER

    // NEEDED MULTIFABS
    amrex::MultiFab bounds(dX.boxarray, dX.distributionMap, 2*IDEAL_GAS_NU, dX.n_grow);
    amrex::MultiFab alpha(dX.boxarray, dX.distributionMap, IDEAL_GAS_NU, dX.n_grow);
    bounds = 0.0;
    alpha = 1.0;

    // SET BOUNDS -----------------------------------------------------
    for (amrex::MFIter mfi(bounds); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);

        amrex::Array4<amrex::Real> const & bounds_fab = bounds.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_VALID(etype))
            {
                // VARIABLES
                amrex::Real dummy_x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                amrex::Real tmp_P[IDEAL_GAS_NU];
                amrex::Real min_P[IDEAL_GAS_NU], max_P[IDEAL_GAS_NU];
#else
                amrex::Real tmp_U[IDEAL_GAS_NU];
                amrex::Real min_U[IDEAL_GAS_NU], max_U[IDEAL_GAS_NU];
#endif
#ifdef BJ_USES_P
                // PRIMITIVE VARIABLES
                get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, dummy_x, tmp_P);

                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    min_P[u] = tmp_P[u];
                    max_P[u] = tmp_P[u];
                }
#else
                // CONSERVED VARIABLES
                get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, dummy_x, tmp_U);
                
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    min_U[u] = tmp_U[u];
                    max_U[u] = tmp_U[u];
                }
#endif
                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::base_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::base_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::base_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    const bool nbr_is_outside_domain = !domain.contains(nbr_i, nbr_j, nbr_k);
                    const bool nbr_is_empty = amrex::dG::ELM_IS_EMPTY(nbr_etype);
                    //const bool nbr_is_small = amrex::dG::ELM_IS_SMALL(nbr_etype);
                    //const bool cells_are_merged = amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype);
                    const bool bad_nbr = ((nbr_is_outside_domain && (is_periodic[n/2] == 0)) || nbr_is_empty);

                    if (!bad_nbr)
                    {
#ifdef BJ_USES_P
                        get_centroid_and_primitive_variables(g, nbr_i, nbr_j, nbr_k, dom, elm_centroid_fab, X_fab, dummy_x, tmp_P);

                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
                            min_P[u] = amrex::min(min_P[u], tmp_P[u]);
                            max_P[u] = amrex::max(max_P[u], tmp_P[u]);
                        }
#else
                        get_centroid_and_conserved_variables(nbr_i, nbr_j, nbr_k, dom, elm_centroid_fab, X_fab, dummy_x, tmp_U);

                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
                            min_U[u] = amrex::min(min_U[u], tmp_U[u]);
                            max_U[u] = amrex::max(max_U[u], tmp_U[u]);
                        }
#endif
                    }
                }

                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
#ifdef BJ_USES_P
                    bounds_fab(i,j,k,u) = min_P[u];
                    bounds_fab(i,j,k,u+IDEAL_GAS_NU) = max_P[u];
#else
                    bounds_fab(i,j,k,u) = min_U[u];
                    bounds_fab(i,j,k,u+IDEAL_GAS_NU) = max_U[u];
#endif
                }
            }
        });
        amrex::Gpu::synchronize();
    }
    bounds.FillBoundary(geom.periodicity());
    copy_to_small_elements(geom, solution, bounds, bounds.n_comp);
    // ----------------------------------------------------------------

#ifdef APPLY_BJ_V1

    // INTRAPHASE BOUNDARIES ------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        amrex::Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        amrex::Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif
        for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            amrex::Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);

            amrex::Array4<amrex::Real const> const & bounds_fab = bounds.array(mfi);
            amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                if (amrex::dG::ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    amrex::Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                    amrex::Real P[IDEAL_GAS_NU], min_P[IDEAL_GAS_NU], max_P[IDEAL_GAS_NU], Pr[IDEAL_GAS_NU];
#else
                    amrex::Real U[IDEAL_GAS_NU], min_U[IDEAL_GAS_NU], max_U[IDEAL_GAS_NU], Ur[IDEAL_GAS_NU];
#endif
                    amrex::Real alphas[IDEAL_GAS_NU];

#ifdef BJ_USES_P
                    get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        min_P[u] = bounds_fab(i,j,k,u);
                        max_P[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                        alphas[u] = alpha_fab(i,j,k,u);
                    }
#else
                    get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        min_U[u] = bounds_fab(i,j,k,u);
                        max_U[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                        alphas[u] = alpha_fab(i,j,k,u);
                    }
#endif

                    for (int b = 2*dir; b < 2*(dir+1); ++b)
                    {
                        int fi, fj, fk;
                        amrex::dG::GRID_FACE(i, j, k, b, fi, fj, fk);

                        // ELEMENT BOUNDARY TYPE
                        const short ebtype = elm_bou_type_fab(fi,fj,fk,amrex::dG::ELM_BOU_TYPE(dom));

                        if (amrex::dG::ELM_BOU_IS_VALID(ebtype))
                        {
                            const amrex::Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                                      prob_lo[1]+fj*dx[1],
                                                                                      prob_lo[2]+fk*dx[2])};
                            const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,amrex::dG::CELL_BOU_QUAD_NQ(dom));
                            const long pos = cell_bou_quad_info_fab(fi,fj,fk,amrex::dG::CELL_BOU_QUAD_POS(dom));
                            const amrex::Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                            for (int q = 0; q < bou_Nq; ++q)
                            {
                                // QUADRATURE POINT AND WEIGHT
                                const long x_pos = (AMREX_SPACEDIM+1)*q;
                                AMREX_D_TERM
                                (
                                    x[0] = x_ptr[x_pos+0]+face_lo[0];,
                                    x[1] = x_ptr[x_pos+1]+face_lo[1];,
                                    x[2] = x_ptr[x_pos+2]+face_lo[2];
                                )

                                // EVAL RECONSTRUCTED STATES
                                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                                {
#ifdef BJ_USES_P
                                    Pr[u] = P[u];
                                    AMREX_D_TERM
                                    (
                                        Pr[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                        Pr[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                        Pr[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                    )
#else
                                    Ur[u] = U[u];
                                    AMREX_D_TERM
                                    (
                                        Ur[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                        Ur[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                        Ur[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                    )
#endif
                                }

                                // UPDATE LIMITING COEFFICIENTS
                                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                                {
#ifdef BJ_USES_P
                                    if (Pr[u] > max_P[u])
                                    {
                                        alphas[u] = amrex::min(alphas[u], (max_P[u]-P[u])/(Pr[u]-P[u]));
                                    }
                                    else if (Pr[u] < min_P[u])
                                    {
                                        alphas[u] = amrex::min(alphas[u], (min_P[u]-P[u])/(Pr[u]-P[u]));
                                    }
#else
                                    if (Ur[u] > max_U[u])
                                    {
                                        alphas[u] = amrex::min(alphas[u], (max_U[u]-U[u])/(Ur[u]-U[u]));
                                    }
                                    else if (Ur[u] < min_U[u])
                                    {
                                        alphas[u] = amrex::min(alphas[u], (min_U[u]-U[u])/(Ur[u]-U[u]));
                                    }
#endif
                                }
                            }
                        }
                    }

                    // STORE UPDATED LIMITING COEFFICIENTS
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alpha_fab(i,j,k,u) = alphas[u];
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------

#ifdef BJ_USES_ICS
    // INTERFACE BOUNDARIES -------------------------------------------
    {
#ifdef AMREX_USE_GPU
        amrex::Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        amrex::Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif
        for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
            amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);

            amrex::Array4<amrex::Real const> const & bounds_fab = bounds.array(mfi);
            amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                if (amrex::dG::ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,amrex::dG::CELL_INT_BOU_QUAD_NQ(dom));
                    const long pos = cell_int_bou_quad_info_fab(i,j,k,amrex::dG::CELL_INT_BOU_QUAD_POS(dom));
                    const amrex::Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    amrex::Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                    amrex::Real P[IDEAL_GAS_NU], min_P[IDEAL_GAS_NU], max_P[IDEAL_GAS_NU], Pr[IDEAL_GAS_NU];
#else
                    amrex::Real U[IDEAL_GAS_NU], min_U[IDEAL_GAS_NU], max_U[IDEAL_GAS_NU], Ur[IDEAL_GAS_NU];
#endif
                    amrex::Real alphas[IDEAL_GAS_NU];

#ifdef BJ_USES_P
                    get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        min_P[u] = bounds_fab(i,j,k,u);
                        max_P[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                        alphas[u] = alpha_fab(i,j,k,u);
                    }
#else
                    get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        min_U[u] = bounds_fab(i,j,k,u);
                        max_U[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                        alphas[u] = alpha_fab(i,j,k,u);
                    }
#endif

                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT AND WEIGHT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )

                        // EVAL RECONSTRUCTED STATES
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
#ifdef BJ_USES_P
                            Pr[u] = P[u];
                            AMREX_D_TERM
                            (
                                Pr[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Pr[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Pr[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#else
                            Ur[u] = U[u];
                            AMREX_D_TERM
                            (
                                Ur[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Ur[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Ur[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#endif
                        }

                        // UPDATE LIMITING COEFFICIENTS
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
#ifdef BJ_USES_P
                            if (Pr[u] > max_P[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (max_P[u]-P[u])/(Pr[u]-P[u]));
                            }
                            else if (Pr[u] < min_P[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (min_P[u]-P[u])/(Pr[u]-P[u]));
                            }
#else
                            if (Ur[u] > max_U[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (max_U[u]-U[u])/(Ur[u]-U[u]));
                            }
                            else if (Ur[u] < min_U[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (min_U[u]-U[u])/(Ur[u]-U[u]));
                            }
#endif
                        }
                    }

                    // STORE UPDATED LIMITING COEFFICIENTS
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alpha_fab(i,j,k,u) = alphas[u];
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
#endif // BJ_USES_ICS

#endif // APPLY_BJ_V1

#ifdef APPLY_BJ_V2

    // EVAL LIMITING COEFFICIENTS -------------------------------------
    for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
        amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);

        amrex::Array4<amrex::Real const> const & bounds_fab = bounds.array(mfi);
        amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_VALID(etype))
            {
                // LOCAL VARIABLES
                amrex::Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                amrex::Real P[IDEAL_GAS_NU], min_P[IDEAL_GAS_NU], max_P[IDEAL_GAS_NU], Pr[IDEAL_GAS_NU];
#else
                amrex::Real U[IDEAL_GAS_NU], min_U[IDEAL_GAS_NU], max_U[IDEAL_GAS_NU], Ur[IDEAL_GAS_NU];
#endif
                amrex::Real alphas[IDEAL_GAS_NU];

#ifdef BJ_USES_P
                get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    min_P[u] = bounds_fab(i,j,k,u);
                    max_P[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                    alphas[u] = alpha_fab(i,j,k,u);
                }
#else
                get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    min_U[u] = bounds_fab(i,j,k,u);
                    max_U[u] = bounds_fab(i,j,k,u+IDEAL_GAS_NU);
                    alphas[u] = alpha_fab(i,j,k,u);
                }
#endif

/*
if (i == __i__ && j == __j__ && k == __k__)
{
amrex::Print() << "xc: "; amrex::dG::io::print_reals(AMREX_SPACEDIM, xc); amrex::Print() << std::endl;
amrex::Print() << "P: " << std::endl;
amrex::dG::io::print_real_array_2d(1, IDEAL_GAS_NU, P);
amrex::Print() << "min_P: " << std::endl;
amrex::dG::io::print_real_array_2d(1, IDEAL_GAS_NU, min_P);
amrex::Print() << "max_P: " << std::endl;
amrex::dG::io::print_real_array_2d(1, IDEAL_GAS_NU, max_P);
amrex::Print() << "alphas: " << std::endl;
amrex::dG::io::print_real_array_2d(1, IDEAL_GAS_NU, alphas);
//exit(-1);
}
*/

                for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::base_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::base_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::base_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    const bool nbr_is_outside_domain = !domain.contains(nbr_i, nbr_j, nbr_k);
                    const bool nbr_is_empty = amrex::dG::ELM_IS_EMPTY(nbr_etype);
                    //const bool nbr_is_small = amrex::dG::ELM_IS_SMALL(nbr_etype);
                    //const bool cells_are_merged = amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype);
                    const bool bad_nbr = ((nbr_is_outside_domain && (is_periodic[n/2] == 0)) || nbr_is_empty);

                    if (!bad_nbr)
                    {
                        // We reconstruct the solution states at the
                        // neighbor centroid.
                        AMREX_D_TERM
                        (
                            x[0] = elm_centroid_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_CENTROID_X1(dom));,
                            x[1] = elm_centroid_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_CENTROID_X2(dom));,
                            x[2] = elm_centroid_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_CENTROID_X3(dom));
                        )

                        // EVAL RECONSTRUCTED STATES
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
#ifdef BJ_USES_P
                            Pr[u] = P[u];
                            AMREX_D_TERM
                            (
                                Pr[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Pr[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Pr[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#else
                            Ur[u] = U[u];
                            AMREX_D_TERM
                            (
                                Ur[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Ur[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Ur[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#endif
                        }
/*
if (i == __i__ && j == __j__ && k == __k__)
{
amrex::Print() << "xc: "; amrex::dG::io::print_reals(AMREX_SPACEDIM, x);
amrex::Print() << " Pr(" << nbr_i << "," << nbr_j << "," << nbr_k << "): "; amrex::dG::io::print_reals(IDEAL_GAS_NU, Pr); amrex::Print() << std::endl;
//exit(-1);
}
*/

                        // UPDATE LIMITING COEFFICIENTS
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
#ifdef BJ_USES_P
                            if (Pr[u] > max_P[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (max_P[u]-P[u])/(Pr[u]-P[u]));
                            }
                            else if (Pr[u] < min_P[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (min_P[u]-P[u])/(Pr[u]-P[u]));
                            }
#else
                            if (Ur[u] > max_U[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (max_U[u]-U[u])/(Ur[u]-U[u]));
                            }
                            else if (Ur[u] < min_U[u])
                            {
                                alphas[u] = amrex::min(alphas[u], (min_U[u]-U[u])/(Ur[u]-U[u]));
                            }
#endif
                        }
                    }
                }

/*
if (i == __i__ && j == __j__ && k == __k__)
{
amrex::Print() << "alphas: " << std::endl;
amrex::dG::io::print_real_array_2d(1, IDEAL_GAS_NU, alphas);
exit(-1);
}
*/

                // STORE UPDATED LIMITING COEFFICIENTS
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    alpha_fab(i,j,k,u) = alphas[u];
                }
            }
        });
        amrex::Gpu::synchronize();
    }
    alpha.FillBoundary(geom.periodicity());
    copy_to_small_elements(geom, solution, alpha, alpha.n_comp);
    // ----------------------------------------------------------------
    
    // INTRAPHASE BOUNDARIES ------------------------------------------
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        amrex::Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        amrex::Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif
        for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            amrex::Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);

            amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                if (amrex::dG::ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL VARIABLES
                    amrex::Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                    amrex::Real P[IDEAL_GAS_NU], Pr[IDEAL_GAS_NU];
#else
                    amrex::Real U[IDEAL_GAS_NU], Ur[IDEAL_GAS_NU];
#endif
                    amrex::Real alphas[IDEAL_GAS_NU];

#ifdef BJ_USES_P
                    get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);
#else
                    get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);
#endif
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alphas[u] = alpha_fab(i,j,k,u);
                    }

                    for (int b = 2*dir; b < 2*(dir+1); ++b)
                    {
                        int fi, fj, fk;
                        amrex::dG::GRID_FACE(i, j, k, b, fi, fj, fk);

                        // ELEMENT BOUNDARY TYPE
                        const short ebtype = elm_bou_type_fab(fi,fj,fk,amrex::dG::ELM_BOU_TYPE(dom));

                        if (amrex::dG::ELM_BOU_IS_VALID(ebtype))
                        {
                            const amrex::Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                                      prob_lo[1]+fj*dx[1],
                                                                                      prob_lo[2]+fk*dx[2])};
                            const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,amrex::dG::CELL_BOU_QUAD_NQ(dom));
                            const long pos = cell_bou_quad_info_fab(fi,fj,fk,amrex::dG::CELL_BOU_QUAD_POS(dom));
                            const amrex::Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                            for (int q = 0; q < bou_Nq; ++q)
                            {
                                // QUADRATURE POINT AND WEIGHT
                                const long x_pos = (AMREX_SPACEDIM+1)*q;
                                AMREX_D_TERM
                                (
                                    x[0] = x_ptr[x_pos+0]+face_lo[0];,
                                    x[1] = x_ptr[x_pos+1]+face_lo[1];,
                                    x[2] = x_ptr[x_pos+2]+face_lo[2];
                                )

                                // EVAL RECONSTRUCTED STATES
                                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                                {
#ifdef BJ_USES_P
                                    Pr[u] = P[u];
                                    AMREX_D_TERM
                                    (
                                        Pr[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                        Pr[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                        Pr[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                    )
#else
                                    Ur[u] = U[u];
                                    AMREX_D_TERM
                                    (
                                        Ur[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                        Ur[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                        Ur[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                    )
#endif
                                }

                                // UPDATE LIMITING COEFFICIENTS
                                {
#ifdef BJ_USES_P
                                    amrex::Real rho, p;
                                    rho = Pr[RHO];
                                    p = Pr[TE];
                                    if (rho < SMALL_rho)
                                    {
                                        alphas[RHO] = amrex::min(alphas[RHO], (SMALL_rho-P[RHO])/(Pr[RHO]-P[RHO]));
                                    }
                                    if (p < SMALL_p)
                                    {
                                        alphas[TE] = amrex::min(alphas[TE], (SMALL_p-P[TE])/(Pr[TE]-P[TE]));
                                    }
#else
                                    const amrex::Real SMALL_e = SMALL_p/(g-1.0);
                                    amrex::Real rho, e;
                                    rho = Ur[RHO];
                                    e = Ur[TE];
                                    if (rho < SMALL_rho)
                                    {
                                        alphas[RHO] = amrex::min(alphas[RHO], (SMALL_rho-U[RHO])/(Ur[RHO]-U[RHO]));
                                    }
                                    if (e < SMALL_e)
                                    {
                                        alphas[TE] = amrex::min(alphas[TE], (SMALL_e-U[TE])/(Ur[TE]-U[TE]));
                                    }
#endif
                                }
                            }
                        }
                    }

                    // STORE UPDATED LIMITING COEFFICIENTS
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alpha_fab(i,j,k,u) = alphas[u];
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------

#ifdef BJ_USES_ICS
    // INTERFACE BOUNDARIES -------------------------------------------
    {
#ifdef AMREX_USE_GPU
        amrex::Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
        amrex::Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif
        for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
            amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            amrex::Array4<amrex::Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);

            amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                if (amrex::dG::ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,amrex::dG::CELL_INT_BOU_QUAD_NQ(dom));
                    const long pos = cell_int_bou_quad_info_fab(i,j,k,amrex::dG::CELL_INT_BOU_QUAD_POS(dom));
                    const amrex::Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    amrex::Real xc[AMREX_SPACEDIM], x[AMREX_SPACEDIM];
#ifdef BJ_USES_P
                    amrex::Real P[IDEAL_GAS_NU], Pr[IDEAL_GAS_NU];
#else
                    amrex::Real U[IDEAL_GAS_NU], Ur[IDEAL_GAS_NU];
#endif
                    amrex::Real alphas[IDEAL_GAS_NU];

#ifdef BJ_USES_P
                    get_centroid_and_primitive_variables(g, i, j, k, dom, elm_centroid_fab, X_fab, xc, P);
#else
                    get_centroid_and_conserved_variables(i, j, k, dom, elm_centroid_fab, X_fab, xc, U);
#endif
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alphas[u] = alpha_fab(i,j,k,u);
                    }

                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT AND WEIGHT
                        const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0];,
                            x[1] = x_ptr[x_pos+1];,
                            x[2] = x_ptr[x_pos+2];
                        )

                        // EVAL RECONSTRUCTED STATES
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
#ifdef BJ_USES_P
                            Pr[u] = P[u];
                            AMREX_D_TERM
                            (
                                Pr[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Pr[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Pr[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#else
                            Ur[u] = U[u];
                            AMREX_D_TERM
                            (
                                Ur[u] += dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                Ur[u] += dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                Ur[u] += dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                            )
#endif
                        }

                        // UPDATE LIMITING COEFFICIENTS
                        {
#ifdef BJ_USES_P
                            amrex::Real rho, p;
                            rho = Pr[RHO];
                            p = Pr[TE];
                            if (rho < SMALL_rho)
                            {
                                alphas[RHO] = amrex::min(alphas[RHO], (SMALL_rho-P[RHO])/(Pr[RHO]-P[RHO]));
                            }
                            if (p < SMALL_p)
                            {
                                alphas[TE] = amrex::min(alphas[TE], (SMALL_p-P[TE])/(Pr[TE]-P[TE]));
                            }
#else
                            const amrex::Real SMALL_e = SMALL_p/(g-1.0);
                            amrex::Real rho, e;
                            rho = Ur[RHO];
                            e = Ur[TE];
                            if (rho < SMALL_rho)
                            {
                                alphas[RHO] = amrex::min(alphas[RHO], (SMALL_rho-U[RHO])/(Ur[RHO]-U[RHO]));
                            }
                            if (e < SMALL_e)
                            {
                                alphas[TE] = amrex::min(alphas[TE], (SMALL_e-U[TE])/(Ur[TE]-U[TE]));
                            }
#endif
                        }
                    }

                    // STORE UPDATED LIMITING COEFFICIENTS
                    for (int u = 0; u < IDEAL_GAS_NU; ++u)
                    {
                        alpha_fab(i,j,k,u) = alphas[u];
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
    // ----------------------------------------------------------------
#endif // BJ_USES_ICS

#endif // APPLY_BJ_V2

    alpha.FillBoundary(geom.periodicity());

    // HANDLE EXTENDED ELEMENTS ---------------------------------------
    for (amrex::MFIter mfi(alpha); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);

        amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));
            
            if (amrex::dG::ELM_IS_EXTENDED(etype))
            {
                // LOCAL VARIABLES
                amrex::Real alphas[IDEAL_GAS_NU];

                // INITIALIZATION
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    alphas[u] = alpha_fab(i,j,k,u);
                }

                // LOOP OVER THE NEIGHBORS
                for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::extended_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::extended_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::extended_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    
                    if (amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                    {
                        for (int u = 0; u < IDEAL_GAS_NU; ++u)
                        {
                            alphas[u] = amrex::min(alphas[u], alpha_fab(nbr_i,nbr_j,nbr_k,u));
                        }
                    }
                }

                // STORE THE VALUES
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    alpha_fab(i,j,k,u) = alphas[u];
                }
            }
        });
        amrex::Gpu::synchronize();
    }
    alpha.FillBoundary(geom.periodicity());
    copy_to_small_elements(geom, solution, alpha, alpha.n_comp);
    // ----------------------------------------------------------------

    // EVENTUALLY STORE THE LIMITED SLOPES ----------------------------
    for (amrex::MFIter mfi(dX); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & alpha_fab = alpha.array(mfi);
#ifdef BJ_USES_P
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
#endif

        amrex::Array4<amrex::Real> const & dX_fab = dX.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            // Note: the small elements modify the slopes of their
            //       parent extended elements.
            if (amrex::dG::ELM_IS_NOT_EMPTY(etype))
            {
                // LOCAL VARIABLES
#ifdef BJ_USES_P
                amrex::Real U[IDEAL_GAS_NU];
                amrex::Real dP[IDEAL_GAS_NU*AMREX_SPACEDIM];
#endif
                amrex::Real dU[IDEAL_GAS_NU*AMREX_SPACEDIM];
                amrex::Real tmp;

#ifdef BJ_USES_P
                // CONSERVED VARIABLES
                U[RHO] = X_fab(i,j,k,RHO);
                AMREX_D_TERM
                (
                    U[RHOV1] = X_fab(i,j,k,RHOV1);,
                    U[RHOV2] = X_fab(i,j,k,RHOV2);,
                    U[RHOV3] = X_fab(i,j,k,RHOV3);
                )
                U[TE] = X_fab(i,j,k,TE);
#endif

#ifdef BJ_USES_P
                // LIMITED SLOPES OF THE PRIMITIVE VARIABLES
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    tmp = alpha_fab(i,j,k,u);

                    AMREX_D_TERM
                    (
                        dP[u+0*IDEAL_GAS_NU] = dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*tmp;,
                        dP[u+1*IDEAL_GAS_NU] = dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*tmp;,
                        dP[u+2*IDEAL_GAS_NU] = dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*tmp;
                    )
                }
#else
                // LIMITED SLOPES OF THE CONSERVED VARIABLES
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    tmp = alpha_fab(i,j,k,u);
                    AMREX_D_TERM
                    (
                        dU[u+0*IDEAL_GAS_NU] = dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*tmp;,
                        dU[u+1*IDEAL_GAS_NU] = dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*tmp;,
                        dU[u+2*IDEAL_GAS_NU] = dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*tmp;
                    )
                }
#endif

#ifdef BJ_USES_P
                // LIMITED SLOPES OF THE CONSERVED VARIABLES
                dP_to_dU(g, U, &dP[0*IDEAL_GAS_NU], &dU[0*IDEAL_GAS_NU]);
#if (AMREX_SPACEDIM > 1)
                dP_to_dU(g, U, &dP[1*IDEAL_GAS_NU], &dU[1*IDEAL_GAS_NU]);
#endif
#if (AMREX_SPACEDIM > 2)
                dP_to_dU(g, U, &dP[2*IDEAL_GAS_NU], &dU[2*IDEAL_GAS_NU]);
#endif

#endif
                for (int u = 0; u < IDEAL_GAS_NU; ++u)
                {
                    AMREX_D_TERM
                    (
                        dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = dU[u+0*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = dU[u+1*IDEAL_GAS_NU];,
                        dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = dU[u+2*IDEAL_GAS_NU];
                    )
                }
            }
        });
        amrex::Gpu::synchronize();
    }
    dX.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
#endif // APPLY_LIMITER

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    if (dX.contains_nan())
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_IdealGas.H - eval_FV_slopes\n";
        msg += "| Reconstructed slopes multifab dX contains nans prior to exiting.\n";
        amrex::Abort(msg);
    }
    // ----------------------------------------------------------------
#endif

}
// ====================================================================


// SHOCK SENSOR =======================================================
/**
 * \brief Tag those cells where the presence of the shock is sensed.
 *
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution dG coefficients.
 * \param[in] X_low: a MultiFabs containing the projection of the dG coefficients in the space of
 *                   lower order polynomials.
 * \param[in] thr: threshold parameter that triggers the tagging.
 * \param[inout] mask: the TagBoxArray containing the tags of the cells.
 * \param[in] masked: the value to be assigned to the tagged cells.
 *
*/
void shock_sensor(const amrex::Geometry & geom, const amrex::dG::Mesh & mesh, const amrex::dG::Solution & solution,
                  const amrex::MultiFab & X, const amrex::MultiFab & X_low, const amrex::Real thr,
                  amrex::TagBoxArray & mask, const char masked)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ideal_gas::shock_sensor(...)");
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // GRID
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // ORDER
    const int sp = solution.params.space_p;
    const int sp_low = sp-1;

    // DOMAINS
    const int dom = 0;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::MultiFab sensor(X.boxarray, X.distributionMap, 2, 1);
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    sensor = 0.0;
    // ----------------------------------------------------------------

    // EVAL THE SENSOR ------------------------------------------------
#ifdef AMREX_USE_GPU
    const amrex::Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
    const amrex::Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif

    for (amrex::MFIter mfi(sensor); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
        amrex::Array4<amrex::Real const> const & X_low_fab = X_low.array(mfi);

        amrex::Array4<char const> const & mask_fab = mask.array(mfi);
        
        amrex::Array4<amrex::Real> const & sensor_fab = sensor.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_NOT_EMPTY(etype) && (mask_fab(i,j,k) != masked))
            {
                // LOCAL PARAMETERS
                const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                          prob_lo[1]+j*dx[1],
                                                                          prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,amrex::dG::CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,amrex::dG::CELL_DOM_QUAD_POS(dom));
                const amrex::Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                amrex::Real x[AMREX_SPACEDIM], w;
                amrex::Real U[IDEAL_GAS_NU], U_low[IDEAL_GAS_NU];
                amrex::Real err, norm;
                amrex::Real integrand_err, integrand_norm;

                // UNKNOWN FIELDS
                amrex::dG::space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);
                amrex::dG::space_elm_bfx bfu_low(&prob_lo[0], &dx[0], i, j, k, etype, sp_low, X_low_fab);

                // EVAL THE INTEGRAL
                err = norm = 0.0;
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // Solution at (t, x)
                    bfu.eval(x, 0, IDEAL_GAS_NU, U);
                    bfu_low.eval(x, 0, IDEAL_GAS_NU, U_low);

                    integrand_err = (U[RHO]-U_low[RHO])*(U[RHO]-U_low[RHO]);
                    integrand_norm = U[RHO]*U[RHO];

                    err += integrand_err*w;
                    norm += integrand_norm*w;
                }

                sensor_fab(i,j,k,0) = norm;
                sensor_fab(i,j,k,1) = err;
            }
        });
        amrex::Gpu::synchronize();
    }
    sensor.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------

    // TAG THE CELLS --------------------------------------------------
    for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & sensor_fab = sensor.array(mfi);

        amrex::Array4<char> const & mask_fab = mask.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_VALID(etype) && (mask_fab(i,j,k) != masked))
            {
                // LOCAL VARIABLES
                amrex::Real err, norm;

                norm = sensor_fab(i,j,k,0);
                err = sensor_fab(i,j,k,1);

                // Handle extended elements
                for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::extended_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::extended_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::extended_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    
                    if (amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                    {
                        norm += sensor_fab(nbr_i,nbr_j,nbr_k,0);
                        err += sensor_fab(nbr_i,nbr_j,nbr_k,1);
                    }
                }

                // Tag cells
                err = err/norm;
                if (err > 0.0)
                {
                    const amrex::Real s = std::log10(err);
                    const amrex::Real s0 = -4.0*std::log10(sp);
                    
                    if (s > (s0+thr))
                    {
                        mask_fab(i,j,k) = masked;
                    }
                }
            }
        });
        amrex::Gpu::synchronize();
        
    }
    mask.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ====================================================================

} // namespace ideal_gas
