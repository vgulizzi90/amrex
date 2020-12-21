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
        p = (g-1)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))); p = std::max(p, SMALL_p);
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

        amrex::Real ro, uo, po, sgnm, utrans1, utrans2;
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