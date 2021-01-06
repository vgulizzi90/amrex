//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Base.H
 * \brief Contains base constants and data structures for IBVP of ideal gases.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define DG_N_SOL (1+AMREX_SPACEDIM+1)

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
// ####################################################################
int global_cnt = 0;
/**
 * \brief This struct contains constants and methods for the considered IBVP. 
 *
*/
struct IDEAL_GAS_BASE
{
    // DATA MEMBERS ###################################################
    // VARIABLES
    const int eb_flag;
    const amrex::Real gamma;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS_BASE(const amrex::Vector<int> & int_params, const amrex::Vector<amrex::Real> & params)
    :
    eb_flag(int_params[0]),
    gamma{params[0]}
    {}
    // ################################################################

    // AUXILIARY FUNCTIONS ############################################
    /**
     * \brief Eval primary variables from conserved variables.
     * 
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] rho: density.
     * \param[out] u1: velocity component in x1 direction.
     * \param[out] u2: velocity component in x2 direction (non present if AMREX_SPACEDIM < 2).
     * \param[out] u3: velocity component in x3 direction (non present if AMREX_SPACEDIM < 3).
     * \param[out] p: pressure.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_primary_variables(const amrex::Real * U, 
                                amrex::Real & rho,
                                AMREX_D_DECL(amrex::Real & u1, amrex::Real & u2, amrex::Real & u3),
                                amrex::Real & p) const
    {
        const amrex::Real g = this->gamma;

        rho = U[RHO]; rho = std::max(rho, SMALL_rho);
        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1.0)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))); p = std::max(p, SMALL_p);
    }

    /**
     * \brief Eval enthalpy variables from conserved variables.
     * 
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] u1: velocity component in x1 direction.
     * \param[out] u2: velocity component in x2 direction (non present if AMREX_SPACEDIM < 2).
     * \param[out] u3: velocity component in x3 direction (non present if AMREX_SPACEDIM < 3).
     * \param[out] p: pressure.
     * \param[out] a: sound speed.
     * \param[out] h: enthalpy.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_enthalpy_variables(const amrex::Real * U, 
                                 AMREX_D_DECL(amrex::Real & u1, amrex::Real & u2, amrex::Real & u3),
                                 amrex::Real & p, amrex::Real & a, amrex::Real & h) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real rho = std::max(U[RHO], SMALL_rho);

        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))); p = std::max(p, SMALL_p);
        a = std::sqrt(g*p/rho);
        h = 0.5*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))+a*a/(g-1.0);
    }

    /**
     * \brief Transform solution from global reference system to unit normal reference system.
     * 
     * \param[in] un: pointer to memory containing the unit normal.
     * \param[in] U: pointer to memory containing the unknown solution fields in the global reference
     *               system.
     * \param[out] Un: pointer to memory containing the unknown solution fields in the unit normal
     *                 reference system.
    */
    AMREX_GPU_HOST_DEVICE
    void eval_U2Un(const amrex::Real * un, const amrex::Real * U, amrex::Real * Un) const
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
     * \param[in] Un: pointer to memory containing the unknown solution fields in the unit normal
     *                reference system.
     * \param[out] U: pointer to memory containing the unknown solution fields in the global reference
     *                system.
    */
    AMREX_GPU_HOST_DEVICE
    void eval_Un2U(const amrex::Real * un, const amrex::Real * Un, amrex::Real * U) const
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
    // ################################################################

    // PRIMARY VARIABLES VS CHARACTERISTICS VARIABLES #################
    /**
     * \brief Eval jumps in the characteristics along x1 from jumps in the primary variables along x1.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[in] DP: pointer to memory containing jumps in the conserved variables.
     * \param[out] DC: pointer to memory that will contain jumps in the characteristics (of the
     *                 primary variables).
    */
    AMREX_GPU_HOST_DEVICE
    void F_DP2DC_1(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                   const amrex::Real * P,
                   const amrex::Real * DP, amrex::Real * DC) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;

        const amrex::Real rho = std::max(P[RHO], SMALL_rho);
        const amrex::Real p = std::max(P[TE], SMALL_p);
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real inv_a = 1.0/a;

#if (AMREX_SPACEDIM == 1)
        DC[RHO]   =        -0.5*inv_a*rho*DP[RHOV1]+0.5*inv_a*inv_a*DP[TE];
        DC[RHOV1] = DP[RHO]                            -inv_a*inv_a*DP[TE];
        DC[TE]    =        +0.5*inv_a*rho*DP[RHOV1]+0.5*inv_a*inv_a*DP[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DC[RHO]   =        -0.5*inv_a*rho*DP[RHOV1]              +0.5*inv_a*inv_a*DP[TE];
        DC[RHOV1] = DP[RHO]                                          -inv_a*inv_a*DP[TE];
        DC[RHOV2] =                                +rho*DP[RHOV2];
        DC[TE]    =        +0.5*inv_a*rho*DP[RHOV1]              +0.5*inv_a*inv_a*DP[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DC[RHO]   =        -0.5*inv_a*rho*DP[RHOV1]                            +0.5*inv_a*inv_a*DP[TE];
        DC[RHOV1] = DP[RHO]                                                        -inv_a*inv_a*DP[TE];
        DC[RHOV2] =                                +rho*DP[RHOV2];
        DC[RHOV3] =                                              +rho*DP[RHOV3];
        DC[TE]    =        +0.5*inv_a*rho*DP[RHOV1]                            +0.5*inv_a*inv_a*DP[TE];
#endif  
    }

    /**
     * \brief Eval jumps in the characteristics from jumps in the primary variables.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[in] DP: pointer to memory containing jumps in the primary variables.
     * \param[out] DC: pointer to memory that will contain jumps in the characteristics (of the
     *                 primary variables).
    */
    AMREX_GPU_HOST_DEVICE
    void F_DP2DC(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                 const amrex::Real * P,
                 const amrex::Real * DP, amrex::Real * DC) const
    {
        // VARIABLES
        amrex::Real Pn[DG_N_SOL], DPn[DG_N_SOL];
        
        this->eval_U2Un(un, P, Pn);
        this->eval_U2Un(un, DP, DPn);
        
        this->F_DP2DC_1(t, x, Pn, DPn, DC);
    }

    /**
     * \brief Eval jumps in the primary variables along x1 from jumps in the characteristics along x1.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[in] DC: pointer to memory containing jumps in the characteristics (of the primary
     *                variables).
     * \param[out] DP: pointer to memory that will contain jumps in the primary variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DC2DP_1(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                   const amrex::Real * P,
                   const amrex::Real * DC, amrex::Real * DP) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;

        const amrex::Real rho = std::max(P[RHO], SMALL_rho);
        const amrex::Real p = std::max(P[TE], SMALL_p);
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real inv_rho = 1.0/rho;
        
#if (AMREX_SPACEDIM == 1)
        DP[RHO]   =            DC[RHO]+DC[RHOV1]          +DC[TE];
        DP[RHOV1] = -inv_rho*a*DC[RHO]          +inv_rho*a*DC[TE];
        DP[TE]    =        a*a*DC[RHO]                +a*a*DC[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DP[RHO]   =            DC[RHO]+DC[RHOV1]                            +DC[TE];
        DP[RHOV1] = -inv_rho*a*DC[RHO]                            +inv_rho*a*DC[TE];
        DP[RHOV2] =                             +inv_rho*DC[RHOV2];
        DP[TE]    =        a*a*DC[RHO]                                  +a*a*DC[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DP[RHO]   =            DC[RHO]+DC[RHOV1]                                              +DC[TE];
        DP[RHOV1] = -inv_rho*a*DC[RHO]                                              +inv_rho*a*DC[TE];
        DP[RHOV2] =                             +inv_rho*DC[RHOV2];
        DP[RHOV3] =                                               +inv_rho*DC[RHOV3];
        DP[TE]    =        a*a*DC[RHO]                                                    +a*a*DC[TE];
#endif
    }

    /**
     * \brief Eval the jumps in the primary variables from jumps in the characteristics.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[in] DC: pointer to memory containing jumps in the characteristics (of the primary
     *                variables).
     * \param[out] DP: pointer to memory that will contain jumps in the primary variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DC2DP(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                 const amrex::Real * P,
                 const amrex::Real * DC, amrex::Real * DP) const
    {
        // VARIABLES
        amrex::Real Pn[DG_N_SOL], DPn[DG_N_SOL];

        this->eval_U2Un(un, P, Pn);
        this->F_DC2DP_1(t, x, Pn, DC, DPn);
        this->eval_Un2U(un, DPn, DP);
    }
    // ################################################################

    // TIME STEP ######################################################
    /**
     * \brief Eval the maximum time step.
     *
     * \param[in] dom: index of the considered domain.
     * \param[in] dx: pointer to memory containing the sizes of the grid's element.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables at (t,x).
     *
     * \return the maximum allowed time step.
    */
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const int /*dom*/,
                     const amrex::Real * dx,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/,
                     const amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;

        // VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        amrex::Real dt;

        // EVAL THE PRIMARY VARIABLES
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        // AND THE LARGEST WAVE SPEED
        const amrex::Real u = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real lam = u+a;

        // SMALLEST ELEMENT SIDE
#if (AMREX_SPACEDIM == 1)
        const amrex::Real h = dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
#endif
        dt = h/lam;

        return dt;
    }
    // ################################################################

    // FLUXES #########################################################
    /**
     * \brief Eval the flux F1 for the IBVP.
     *
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F1(const amrex::Real * U, amrex::Real * F1) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F1[RHO] = U[RHOV1];
        AMREX_D_TERM
        (
            F1[RHOV1] = U[RHOV1]*u1+p;,
            F1[RHOV2] = U[RHOV2]*u1;,
            F1[RHOV3] = U[RHOV3]*u1;
        )
        F1[TE] = (U[TE]+p)*u1;
    }

#if (AMREX_SPACEDIM > 1)
    /**
     * \brief Eval the flux F2 for the IBVP.
     *
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F2: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F2(const amrex::Real * U, amrex::Real * F2) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F2[RHO] = U[RHOV2];
        AMREX_D_TERM
        (
            F2[RHOV1] = U[RHOV1]*u2;,
            F2[RHOV2] = U[RHOV2]*u2+p;,
            F2[RHOV3] = U[RHOV3]*u2;
        )
        F2[TE] = (U[TE]+p)*u2;
    }
#endif

#if (AMREX_SPACEDIM > 2)
    /**
     * \brief Eval the flux F3 for the IBVP.
     *
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F3: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F3(const amrex::Real * U, amrex::Real * F3) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F3[RHO] = U[RHOV3];
        AMREX_D_TERM
        (
            F3[RHOV1] = U[RHOV1]*u3;,
            F3[RHOV2] = U[RHOV2]*u3;,
            F3[RHOV3] = U[RHOV3]*u3+p;
        )
        F3[TE] = (U[TE]+p)*u3;
    }
#endif

    /**
     * \brief Eval the fluxes for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE
    void F_F(const int /*dom*/,
             const amrex::Real /*t*/, const amrex::Real * /*x*/,
             const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        AMREX_D_TERM
        (
            this->eval_F1(U, F1);,
            this->eval_F2(U, F2);,
            this->eval_F3(U, F3);
        )
    }
    // ################################################################

    // RIEMANN PROBLEM ################################################
    /**
     * \brief Solves the Riemann problem between left and right states in unit normal reference system.
     *
     * \param[in] (rl, ul, ut1l, ut2l, pl): primary variables at the left of the shock.
     * \param[in] (rr, ur, ut1r, ut2r, pr): primary variables at the right of the shock.
     * \param[out] URP: pointer to memory that will contain the solution of the Riemann problem, i.e.
     *                  the state at the shock.
    */
    AMREX_GPU_HOST_DEVICE
    void RiemannSolver_P(const amrex::Real rl, AMREX_D_DECL(const amrex::Real ul, const amrex::Real ut1l, const amrex::Real ut2l), const amrex::Real pl,
                         const amrex::Real rr, AMREX_D_DECL(const amrex::Real ur, const amrex::Real ut1r, const amrex::Real ut2r), const amrex::Real pr,
                         amrex::Real * URP) const
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
        pstar = std::max(pstar, SMALL_p);
        pstnm1 = pstar;

        wlsq = (0.5*(gamma-1.0)*(pstar+pl)+pstar)*rl;
        wrsq = (0.5*(gamma-1.0)*(pstar+pr)+pstar)*rr;

        wl = std::sqrt(wlsq);
        wr = std::sqrt(wrsq);

        ustarp = ul-(pstar-pl)/wl;
        ustarm = ur+(pstar-pr)/wr;

        pstar = (wl*pr + wr*pl - wr*wl*(ur-ul))/(wl+wr);
        pstar = std::max(pstar, SMALL_p);

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

            denom = dpditer/std::max(zp+zm,small*(cleft+cright));
            pstnm1 = pstar;
            pstar = pstar - denom*(ustarm-ustarp);
            pstar = std::max(pstar,SMALL_p);
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
        frac = 0.50*(1.0+(spin+spout)/std::max(spout-spin,std::max(spin+spout,small*(cleft+cright))));

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
    
    /**
     * \brief Solves the Riemann problem between left and right states in unit normal reference system.
     *
     * \param[in] Un_left: pointer to memory containing the conserved variables at the left of the shock
     *                     expressed in the unit normal reference system.
     * \param[in] Un_right: pointer to memory containing the conserved variables at the right of the
     *                      shock expressed in the unit normal reference system.
     * \param[out] URP: pointer to memory that will contain the solution of the Riemann problem, i.e.
     *                  the state at the shock.
    */
    AMREX_GPU_HOST_DEVICE
    void RiemannSolver(const amrex::Real * Un_left, const amrex::Real * Un_right,
                       amrex::Real * URP) const
    {
        amrex::Real rl, AMREX_D_DECL(ul, ut1l, ut2l), pl;
        this->eval_primary_variables(Un_left, rl, AMREX_D_DECL(ul, ut1l, ut2l), pl);
        
        amrex::Real rr, AMREX_D_DECL(ur, ut1r, ut2r), pr;
        this->eval_primary_variables(Un_right, rr, AMREX_D_DECL(ur, ut1r, ut2r), pr);

        this->RiemannSolver_P(rl, AMREX_D_DECL(ul, ut1l, ut2l), pl,
                              rr, AMREX_D_DECL(ur, ut1r, ut2r), pr,
                              URP);
    }
    // ################################################################

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_U memory to enforce wall boundary conditions.
     *
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void bcs_wall(const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * bcs_U) const
    {
        amrex::Real Un[DG_N_SOL];

        this->eval_U2Un(un, U, Un);
        Un[RHOV1] = -Un[RHOV1];
        this->eval_Un2U(un, Un, bcs_U);
    }

    /**
     * \brief Fill bcs_U memory to enforce prescribed velocity boundary conditions.
     *
     * \param[in] p_rho: prescribed density.
     * \param[in] (p_u1, p_u2, p_u3): components of the prescribed velocity.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void bcs_prescribed_velocity(const amrex::Real p_rho, AMREX_D_DECL(const amrex::Real p_u1, const amrex::Real p_u2, const amrex::Real p_u3),
                                 const amrex::Real * un,
                                 const amrex::Real * U,
                                 amrex::Real * bcs_U) const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->gamma;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real rho, AMREX_D_DECL(u1n, u2t, u3t), p, a;
        amrex::Real AMREX_D_DECL(p_u1n, p_u2t, p_u3t);
        amrex::Real AL, BL, du, ps;
        amrex::Real tmp, tmp2;
        // ------------------------------------------------------------

        // FIND THE COMPONENTS REFERRED TO THE UNIT NORMAL ------------
        {
            amrex::Real Un[DG_N_SOL];

            this->eval_U2Un(un, U, Un);
            this->eval_primary_variables(Un, rho, AMREX_D_DECL(u1n, u2t, u3t), p);
            a = std::sqrt(g*p/rho);
        }

        {
            const amrex::Real aux_U[DG_N_SOL] = {0.0, AMREX_D_DECL(p_u1, p_u2, p_u3), 0.0};
            amrex::Real aux_Un[DG_N_SOL];

            this->eval_U2Un(un, aux_U, aux_Un);

            AMREX_D_TERM
            (
                p_u1n = aux_Un[RHOV1];,
                p_u2t = aux_Un[RHOV2];,
                p_u3t = aux_Un[RHOV3];
            )
        }
        // ------------------------------------------------------------

        // PRESSURE IN THE STAR REGION AND GHOST STATE ----------------
        du = p_u1n-u1n;

        // RAREFACTION
        if (du > 0.0)
        {
            tmp2 = 1.0-0.5*(g-1.0)*du/a;

            if (tmp2 < 0.0)
            {
                amrex::Print() << "un: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
                amrex::Print() << "p_u1n: " << p_u1n << std::endl;
                amrex::Print() << "u1n: " << u1n << std::endl;
                amrex::Print() << "a: " << a << std::endl;
                amrex::Print() << "p: " << p << std::endl;
                amrex::Print() << "tmp2: " << tmp2 << std::endl;
                exit(-1);
            }

            ps = p*std::pow(tmp2, 2.0*g/(g-1.0));

            // Speeds of the head and tail of the rarefaction
            tmp = u1n-a;
            tmp2 = p_u1n-a*tmp2;

            // OUTLET
            if (tmp > 0.0)
            {
                // Do nothing
            }
            else if (tmp2 < 0.0)
            {
                // OUTLET: us > 0 (STAR REGION - BEFORE CONTACT)
                if (p_u1n >= 0.0)
                {
                    rho *= std::pow(ps/p, 1.0/g);
                    u1n = p_u1n;
                    p = ps;
                }
                // INLET: us < 0 (STAR REGION - PAST CONTACT)
                else if (p_u1n < 0.0)
                {
                    rho = p_rho;
                    AMREX_D_TERM
                    (
                        u1n = p_u1n;,
                        u2t = p_u2t;,
                        u3t = p_u3t;
                    )
                    p = ps;
                }
            } 
            // INLET: WITHIN THE RAREFACTION
            else
            {
amrex::Print() << "RAREFACTION - INLET WITHIN RAREFACTION" << std::endl;
exit(-1);
                tmp = std::pow(2.0/(g+1.0)+(g-1.0)/(g+1.0)*(u1n/a), 2.0/(g-1.0));

                rho *= tmp;
                u1n = 2.0/(g+1.0)*(a+0.5*(g-1.0)*u1n);
                p *= std::pow(tmp, g);
            }

        }
        // SHOCK
        else
        {
            AL = 2.0/((g+1.0)*rho);
            BL = (g-1.0)/(g+1.0)*p;
            tmp = std::sqrt(4.0*AL*(BL+p)+du*du);
            ps = p+0.5*(du*du/AL-tmp*du/AL);

            // Speed of the shock
            tmp2 = u1n-a*std::sqrt(2.0*(g+1.0)/g*ps/p+2.0*(g-1.0)/g);

            // OUTLET
            if (tmp2 >= 0.0)
            {
                // Do nothing
            }
            else
            {
                // OUTLET: us > 0
                if (p_u1n >= 0.0)
                {
                    tmp = (g-1.0)/(g+1.0)*(p/ps)+1.0;
                    tmp2 = (g-1.0)/(g+1.0)+p/ps;

                    rho *= tmp/tmp2;
                    u1n = p_u1n;
                    p = ps;
                }
                // INLET: us < 0
                else if (p_u1n < 0.0)
                {
                    rho = p_rho;
                    AMREX_D_TERM
                    (
                        u1n = p_u1n;,
                        u2t = p_u2t;,
                        u3t = p_u3t;
                    )
                    p = ps;
                }
            }
        }
        // ------------------------------------------------------------

        // GHOST STATE (CONSERVED VARIABLES) --------------------------
        {
            amrex::Real bcs_Un[DG_N_SOL];

            bcs_Un[RHO] = rho;
            AMREX_D_TERM
            (
                bcs_Un[RHOV1] = rho*u1n;,
                bcs_Un[RHOV2] = rho*u2t;,
                bcs_Un[RHOV3] = rho*u3t;
            )
            bcs_Un[TE] = p/(g-1.0)+0.5*rho*(AMREX_D_TERM(u1n*u1n,+u2t*u2t,+u3t*u3t));

            this->eval_Un2U(un, bcs_Un, bcs_U);   
        }
        // ------------------------------------------------------------



/*
{
    const amrex::Real aux_U[DG_N_SOL] = {0.0, AMREX_D_DECL(p_u1, p_u2, p_u3), 0.0};
    amrex::Real aux_Un[DG_N_SOL];

    this->eval_U2Un(un, aux_U, aux_Un);

    AMREX_D_TERM
    (
        p_u1n = aux_Un[RHOV1];,
        p_u2t = aux_Un[RHOV2];,
        p_u3t = aux_Un[RHOV3];
    )
}
amrex::Print() << "p_u1n: " << p_u1n << std::endl;
amrex::Print() << "p_u2t: " << p_u2t << std::endl;

{
amrex::Real Un[DG_N_SOL], bcs_Un[DG_N_SOL], U_RP[DG_N_SOL];
this->eval_U2Un(un, U, Un);
this->eval_U2Un(un, bcs_U, bcs_Un);

amrex::Print() << "RIEMANN PROBLEM STATES" << std::endl;
this->eval_primary_variables(Un, rho, AMREX_D_DECL(u1n, u2t, u3t), p);
amrex::Print() << "rho: " << rho << std::endl;
amrex::Print() << "u1n: " << u1n << std::endl;
amrex::Print() << "u2t: " << u2t << std::endl;
amrex::Print() << "p: " << p << std::endl;
this->eval_primary_variables(bcs_Un, tmp, AMREX_D_DECL(p_u1n, p_u2t, p_u3t), tmp2);
amrex::Print() << "bcs_rho: " << tmp << std::endl;
amrex::Print() << "bcs_u1n: " << p_u1n << std::endl;
amrex::Print() << "bcs_u2t: " << p_u2t << std::endl;
amrex::Print() << "bcs_p: " << tmp2 << std::endl;

this->RiemannSolver_P(rho, u1n, u2t, p,
                      tmp, p_u1n, p_u2t, tmp2,
                      U_RP);
this->eval_primary_variables(U_RP, rho, AMREX_D_DECL(u1n, u2t, u3t), p);
amrex::Print() << "RIEMANN PROBLEM SOLUTION" << std::endl;
amrex::Print() << "u1n: " << u1n << std::endl;
amrex::Print() << "u2t: " << u2t << std::endl << std::endl;
}

global_cnt += 1;

if (global_cnt == 20) exit(-1);
*/
    }

    /**
     * \brief Fill bcs_P memory to enforce prescribed pressure boundary conditions.
     *
     * \param[in] rhoB: prescribed density.
     * \param[in] pB: prescribed pressure.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void bcs_prescribed_pressure_P(AMREX_D_DECL(const amrex::Real rhoB, const amrex::Real u2tB, const amrex::Real u3tB), const amrex::Real pB,
                                   const amrex::Real * un,
                                   const amrex::Real * P,
                                   amrex::Real * bcs_P) const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->gamma;
        const amrex::Real ig = 1.0/g;
        const amrex::Real gg = (g-1.0)/(g+1.0);
        const amrex::Real gpg = 0.5*(g+1.0)/g;
        const amrex::Real gmg = 0.5*(g-1.0)/g;
        const amrex::Real igp = 2.0/(g+1.0);
        const amrex::Real igm = 2.0/(g-1.0);

        // WALL VELOCITY
        const amrex::Real sw = 0.0;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real rhoL, AMREX_D_DECL(u1L, u2L, u3L), pL, aL;
        amrex::Real ps, us, rhosL;
        amrex::Real rhoR, AMREX_D_DECL(u1R, u2R, u3R), pR;
        // ------------------------------------------------------------

        // FIND THE COMPONENTS REFERRED TO THE UNIT NORMAL ------------
        {
            amrex::Real Pn[DG_N_SOL];

            this->eval_U2Un(un, P, Pn);
            rhoL = Pn[RHO];
            AMREX_D_TERM
            (
                u1L = Pn[RHOV1];,
                u2L = Pn[RHOV2];,
                u3L = Pn[RHOV3];
            )
            pL = Pn[TE];
            
            aL = std::sqrt(g*pL/rhoL);

            ps = pB;
        }
        // ------------------------------------------------------------

        // STAR REGION ------------------------------------------------
        // SHOCK
        if (ps > pL)
        {
            us = u1L-(ps-pL)*std::sqrt(igp/(rhoL*(ps+gg*pL)));
            rhosL = (gg*(pL/ps)+1.0)/(pL/ps+gg)*rhoL;
        }
        // RAREFACTION
        else
        {
            us = u1L+igm*aL*(1.0-std::pow(ps/pL, gmg));
            rhosL = rhoL*std::pow(ps/pL, ig);
        }
        // ------------------------------------------------------------

        // RIGHT STATE: PRIMARY VARIABLES -----------------------------
        // SHOCK
        if (ps > pL)
        {
            const amrex::Real sL = u1L-aL*std::sqrt(gpg*(ps/pL)+gmg);

            if (sL > sw)
            {
                pR = pL;
                u1R = u1L;
                AMREX_D_TERM
                (
                    rhoR = rhoL;,
                    u2R = u2L;,
                    u3R = u3L;
                )
            }
            else
            {
                pR = ps;
                u1R = us;
                if (us < sw)
                {
                    AMREX_D_TERM
                    (
                        rhoR = rhoB;,
                        u2R = u2tB;,
                        u3R = u3tB;
                    )
                    
                }
                else
                {
                    AMREX_D_TERM
                    (
                        rhoR = rhosL;,
                        u2R = u2L;,
                        u3R = u3L;
                    )
                }
            }
        }
        // RAREFACTION
        else
        {
            const amrex::Real sTL = us-aL*std::pow(ps/pL, gmg);
            const amrex::Real sHL = u1L-aL;

            if (sHL >= sw)
            {
                pR = pL;
                u1R = u1L;
                AMREX_D_TERM
                (
                    rhoR = rhoL;,
                    u2R = u2L;,
                    u3R = u3L;
                )
            }   
            else
            {
                if (sTL > sw)
                {
                    rhoR = rhoL*std::pow(igp+(1.0/aL)*gg*(u1L-sw), igm);
                    pR = pL*std::pow(rhoR/rhoL, g);
                    u1R = igp*(aL+0.5*(g-1.0)*u1L+sw);
                    u1R = us; // not sure
amrex::Print() << "#4" << std::endl;
exit(-1);

                }
                else
                {
                    pR = ps;
                    u1R = us;
                    if (us < sw)
                    {
                        AMREX_D_TERM
                        (
                            rhoR = rhoB;,
                            u2R = u2tB;,
                            u3R = u3tB;
                        )
                        
                    }
                    else
                    {
                        AMREX_D_TERM
                        (
                            rhoR = rhosL;,
                            u2R = u2L;,
                            u3R = u3L;
                        )
                    }
                }
            }
        }

        {
            amrex::Real Pn[DG_N_SOL];
            Pn[RHO] = rhoR;
            AMREX_D_TERM
            (
                Pn[RHOV1] = u1R;,
                Pn[RHOV2] = u2R;,
                Pn[RHOV3] = u3R;
            )
            Pn[TE] = pR;

            this->eval_Un2U(un, Pn, bcs_P);
        }
        // ------------------------------------------------------------
    }
    
    /**
     * \brief Fill bcs_P memory to enforce prescribed velocity boundary conditions.
     *
     * \param[in] rhoB: prescribed density.
     * \param[in] (u1B, u2B, u3B): components of the prescribed velocity.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void bcs_prescribed_velocity_P(const amrex::Real rhoB, AMREX_D_DECL(const amrex::Real u1B, const amrex::Real u2B, const amrex::Real u3B),
                                   const amrex::Real * un,
                                   const amrex::Real * P,
                                   amrex::Real * bcs_P) const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->gamma;
        const amrex::Real gp = 0.5*(g+1.0);
        const amrex::Real igm = 2.0/(g-1.0);
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real AMREX_D_DECL(u1nB, u2tB, u3tB);
        amrex::Real rhoL, u1L, pL, aL;
        amrex::Real us, du, ps;
        // ------------------------------------------------------------

        // FIND THE COMPONENTS REFERRED TO THE UNIT NORMAL ------------
        {
            const amrex::Real aux_P[DG_N_SOL] = {0.0, AMREX_D_DECL(u1B, u2B, u3B), 0.0};
            amrex::Real aux_Pn[DG_N_SOL];

            this->eval_U2Un(un, aux_P, aux_Pn);

            AMREX_D_TERM
            (
                u1nB = aux_Pn[RHOV1];,
                u2tB = aux_Pn[RHOV2];,
                u3tB = aux_Pn[RHOV3];
            )
        }

        {
            amrex::Real Pn[DG_N_SOL];

            this->eval_U2Un(un, P, Pn);
            rhoL = Pn[RHO];
            u1L = Pn[RHOV1];
            pL = Pn[TE];
            
            aL = std::sqrt(g*pL/rhoL);

            us = u1nB;
            du = u1L-us;
        }
        // ------------------------------------------------------------

        // STAR REGION ------------------------------------------------
        // SHOCK
        if (us < u1L)
        {
            const amrex::Real delta = std::sqrt(4.0*g*rhoL*pL+rhoL*rhoL*gp*gp*du*du);
            ps = pL+0.5*gp*rhoL*du*du+0.5*du*delta;
        }
        // RAREFACTION
        else if ((u1L <= us) && (us < (u1L+igm*aL)))
        {
            ps = pL*std::pow((du+igm*aL)/(igm*aL), igm*g);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: IBVP_Base.H - bcs_prescribed_velocity_P\n";
            msg += "| Bad value of prescribed velocity.\n";
            amrex::Abort(msg);
        }
        // ------------------------------------------------------------

        // RIGHT STATE ------------------------------------------------
        this->bcs_prescribed_pressure_P(AMREX_D_DECL(rhoB, u2tB, u3tB), ps, un, P, bcs_P);
        // ------------------------------------------------------------
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    /**
     * \brief Eval the numerical flux at the interface between two adjacent states.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] P: pointer to memory containing the PRIMARY variables.
     * \param[in] nbr_P: pointer to memory containing the neighbor PRIMARY variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS_P(const int /*dom*/,
                    const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                    const amrex::Real * P, const amrex::Real * nbr_P,
                    amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Pn[DG_N_SOL], nbr_Pn[DG_N_SOL];
        amrex::Real U_RP[DG_N_SOL], tmp[DG_N_SOL];
        
        // EXPRESS THE STATE IN THE UNIT NORMAL REFERENCE SYSTEM
        this->eval_U2Un(un, P, Pn);
        this->eval_U2Un(un, nbr_P, nbr_Pn);

        // CALL THE RIEMANN SOLVER AND THE EVALUATE THE FLUX
        this->RiemannSolver_P(std::max(Pn[RHO], SMALL_rho),
                              AMREX_D_DECL(Pn[RHOV1], Pn[RHOV2], Pn[RHOV3]),
                              std::max(Pn[TE], SMALL_p),
                              std::max(nbr_Pn[RHO], SMALL_rho),
                              AMREX_D_DECL(nbr_Pn[RHOV1], nbr_Pn[RHOV2], nbr_Pn[RHOV3]),
                              std::max(nbr_Pn[TE], SMALL_p),
                              U_RP);
        this->eval_F1(U_RP, tmp);
        this->eval_Un2U(un, tmp, NFn);
    }
    // ################################################################

};