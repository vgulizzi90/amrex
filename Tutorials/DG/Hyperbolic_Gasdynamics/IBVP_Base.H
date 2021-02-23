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



// ####################################################################
// IDEAL GAS: IBVP CLASS ##############################################
// ####################################################################
/**
 * \brief This struct contains constants and methods for the considered IBVP. 
 *
*/
struct IDEAL_GAS_BASE
{
    // DATA MEMBERS ==================================================
    const int eb_flag;
    const amrex::Real gamma;
    // ===============================================================


    // CONSTRUCTOR ===================================================
    /**
     * \brief This method initializes the IBVP.
    */
    AMREX_GPU_HOST_DEVICE
    IDEAL_GAS_BASE(const amrex::Vector<int> & input_int_params, const amrex::Vector<amrex::Real> & input_params)
    :
    eb_flag(input_int_params[0]),
    gamma{input_params[0]}
    {}
    // ===============================================================


    // AUXILIARY FUNCTIONS ===========================================
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
                                amrex::Real & p,
                                const bool avoid_negative_values = true) const
    {
        const amrex::Real g = this->gamma;

        rho = U[RHO];
        if (avoid_negative_values) rho = amrex::max(rho, SMALL_rho);
        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1.0)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));
        if (avoid_negative_values) p = amrex::max(p, SMALL_p);
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
                                 amrex::Real & p, amrex::Real & a, amrex::Real & h,
                                 const bool avoid_negative_values = true) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real rho = avoid_negative_values ? amrex::max(U[RHO], SMALL_rho) : U[RHO];

        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1.0)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));
        if (avoid_negative_values) p = amrex::max(p, SMALL_p);
        a = std::sqrt(g*p/rho);
        h = 0.5*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))+a*a/(g-1.0);
    }

    /**
     * \brief Eval decomposition of the state vector using left eigenvalues.
     * 
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] L: pointer to memory that will contain the coefficients of the left eigenvectors.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_eigen_decomposition_1(const amrex::Real * U, 
                                    amrex::Real * L) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p;
        amrex::Real a, h, id;

        this->eval_enthalpy_variables(U, AMREX_D_DECL(u1, u2, u3), p, a, h);
        id = AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)-2.0*h; id = 1.0/id;

#if (AMREX_SPACEDIM == 1)
amrex::Abort("eval_eigen_decomposition_1 - DIM = 1");
#endif
#if (AMREX_SPACEDIM == 2)
        L[RHO]   =  0.5*(u1/a-2.0*h*id-1.0)*U[RHO]+(u1*id-0.5/a)*U[RHOV1]    +u2*id*U[RHOV2]    -id*U[TE];
        L[RHOV1] =           2.0*(h*id+1.0)*U[RHO]    -2.0*u1*id*U[RHOV1]-2.0*u2*id*U[RHOV2]+2.0*id*U[TE];
        L[RHOV2] =                      -u2*U[RHO]                                 +U[RHOV2];
        L[TE]    = -0.5*(u1/a+2.0*h*id+1.0)*U[RHO]+(u1*id+0.5/a)*U[RHOV1]    +u2*id*U[RHOV2]    -id*U[TE];
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Abort("eval_eigen_decomposition_1 - DIM = 3");
#endif
    }

    /**
     * \brief Eval reconstruction of the state vector using left eigenvalues.
     * 
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] L: pointer to memory containing the coefficients of the left eigenvectors.
     * \param[out] Ur: pointer to memory that will contain the reconstructed conserved variables.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_eigen_reconstruction_1(const amrex::Real * U,
                                     const amrex::Real * L,
                                     amrex::Real * Ur) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p;
        amrex::Real a, h, k;

        this->eval_enthalpy_variables(U, AMREX_D_DECL(u1, u2, u3), p, a, h);
        k = 0.5*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));

#if (AMREX_SPACEDIM == 1)
amrex::Abort("eval_eigen_reconstruction_1 - DIM = 1");
#endif
#if (AMREX_SPACEDIM == 2)
        Ur[RHO]   =          L[RHO]   +L[RHOV1]                     +L[TE];
        Ur[RHOV1] =   (u1-a)*L[RHO]+u1*L[RHOV1]              +(u1+a)*L[TE];
        Ur[RHOV2] =       u2*L[RHO]+u2*L[RHOV1]   +L[RHOV2]      +u2*L[TE];
        Ur[TE]    = (h-a*u1)*L[RHO] +k*L[RHOV1]+u2*L[RHOV2]+(h+a*u1)*L[TE];
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Abort("eval_eigen_reconstruction_1 - DIM = 3");
#endif
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

    /**
    * \brief Compute the primary variables from the conserved variables.
    *
    * \param[in] U: pointer to memory containing the conserved variables state vector.
    * \param[out] P: pointer to memory that will contain the primary variables state vector.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_U2P(const amrex::Real * U, amrex::Real * P) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);
        
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
    * \brief Compute the conserved variables from the primary variables.
    *
    * \param[in] P: pointer to memory containing the primary variables state vector.
    * \param[out] U: pointer to memory that will contain the conserved variables state vector.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_P2U(const amrex::Real * P, amrex::Real * U) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real rho = amrex::max(P[RHO], SMALL_rho);
        AMREX_D_TERM
        (
            const amrex::Real u1 = P[RHOV1];,
            const amrex::Real u2 = P[RHOV2];,
            const amrex::Real u3 = P[RHOV3];
        )
        const amrex::Real p = amrex::max(P[TE], SMALL_p);
        
        U[RHO] = rho;
        AMREX_D_TERM
        (
            U[RHOV1] = rho*u1;,
            U[RHOV2] = rho*u2;,
            U[RHOV3] = rho*u3;
        )
        U[TE] = p/(g-1.0)+0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
    }

    /**
     * \brief Eval jumps in the conserved variables from jumps in the primary variables.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DP: pointer to memory containing jumps in the primary variables.
     * \param[out] DU: pointer to memory that will contain jumps in the conserved variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DP2DU(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                 const amrex::Real * U,
                 const amrex::Real * DP, amrex::Real * DU) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;
        const amrex::Real ig1 = 1.0/(g-1.0);

        // VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

#if (AMREX_SPACEDIM == 1)
        DU[RHO]   =           DP[RHO];
        DU[RHOV1] =        u1*DP[RHO]   +rho*DP[RHOV1];
        DU[TE]    = 0.5*u1*u1*DP[RHO]+rho*u1*DP[RHOV1]+ig1*DP[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DU[RHO]   =                   DP[RHO];
        DU[RHOV1] =                u1*DP[RHO]   +rho*DP[RHOV1];
        DU[RHOV2] =                u2*DP[RHO]                    +rho*DP[RHOV2];
        DU[TE]    = 0.5*(u1*u1+u2*u2)*DP[RHO]+rho*u1*DP[RHOV1]+rho*u2*DP[RHOV2]+ig1*DP[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DU[RHO]   =                         DP[RHO];
        DU[RHOV1] =                      u1*DP[RHO]   +rho*DP[RHOV1];
        DU[RHOV2] =                      u2*DP[RHO]                    +rho*DP[RHOV2];
        DU[RHOV3] =                      u3*DP[RHO]                                     +rho*DP[RHOV3];
        DU[TE]    = 0.5*(u1*u1+u2*u2+u3*u3)*DP[RHO]+rho*u1*DP[RHOV1]+rho*u2*DP[RHOV2]+rho*u3*DP[RHOV3]+ig1*DP[TE];
#endif
    }

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

        const amrex::Real rho = amrex::max(P[RHO], SMALL_rho);
        const amrex::Real p = amrex::max(P[TE], SMALL_p);
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

        const amrex::Real rho = amrex::max(P[RHO], SMALL_rho);
        const amrex::Real p = amrex::max(P[TE], SMALL_p);
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
    // ===============================================================


    // TIME STEP =====================================================
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

        // PRIMARY VARIABLES
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        // LARGEST WAVE SPEED
        const amrex::Real u = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real lam = u+a;

        // SMALLEST ELEMENT SIDE
#if (AMREX_SPACEDIM == 1)
        const amrex::Real h = dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = amrex::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
#endif
        dt = h/lam;

        return dt;
    }
    // ===============================================================


    // GOVERNING EQUATIONS ===========================================
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

    /**
     * \brief Eval the fluxes for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_F(const int /*dom*/,
             const amrex::Real /*t*/, const amrex::Real * /*x*/,
             const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
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

#if (AMREX_SPACEDIM > 1)
        F2[RHO] = U[RHOV2];
        AMREX_D_TERM
        (
            F2[RHOV1] = U[RHOV1]*u2;,
            F2[RHOV2] = U[RHOV2]*u2+p;,
            F2[RHOV3] = U[RHOV3]*u2;
        )
        F2[TE] = (U[TE]+p)*u2;
#endif

#if (AMREX_SPACEDIM > 2)
        F3[RHO] = U[RHOV3];
        AMREX_D_TERM
        (
            F3[RHOV1] = U[RHOV1]*u3;,
            F3[RHOV2] = U[RHOV2]*u3;,
            F3[RHOV3] = U[RHOV3]*u3+p;
        )
        F3[TE] = (U[TE]+p)*u3;
#endif
    }
    // ===============================================================


    // RIEMANN PROBLEM ===============================================
    /**
     * \brief Solves the Riemann problem between left and right states in unit normal reference system.
     *
     * \param[in] (rl, ul, ut1l, ut2l, pl): primary variables at the left of the shock.
     * \param[in] (rr, ur, ut1r, ut2r, pr): primary variables at the right of the shock.
     * \param[out] URP: pointer to memory that will contain the solution of the Riemann problem, i.e.
     *                  the state at the shock.
    */
    AMREX_GPU_HOST_DEVICE
    void ExactRiemannSolver(const amrex::Real rl, AMREX_D_DECL(const amrex::Real ul, const amrex::Real ut1l, const amrex::Real ut2l), const amrex::Real pl,
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

    /**
     * \brief Compute the flux of the Riemann problem using the modified Osher solver.
     *
     * \param[in] Un_left: pointer to memory containing the conserved variables at the left of the shock
     *                     expressed in the unit normal reference system.
     * \param[in] Un_right: pointer to memory containing the conserved variables at the right of the
     *                      shock expressed in the unit normal reference system.
     * \param[out] NFn: pointer to memory that will contain the flux at the interface.
    */
    AMREX_GPU_HOST_DEVICE
    void OsherSolver(const amrex::Real * /*left_Un*/, const amrex::Real * /*right_Un*/,
                     amrex::Real * /*NFn*/) const
    {
        // PARAMETERS ------------------------------------------------
        //const amrex::Real g = this->gamma;
        //const int qo = 3;
        //const amrex::Real xG[3] = {0.5-0.1*std::sqrt(15.0), 0.5, 0.5+0.1*std::sqrt(15.0)};
        //const amrex::Real wG[3] = {5.0/18.0, 4.0/9.0, 5.0/18.0};
        // -----------------------------------------------------------

amrex::Abort("IDEAL_GAS_BASE.OsherSolver");
    }
    // ===============================================================


    // NUMERICAL FLUXES ==============================================
    /**
     * \brief Add artificial viscosity to the computed numerical flux.
     *
     * \param[in] Un_left: pointer to memory containing the CONSERVED variables.
     * \param[in] Un_right: pointer to memory containing the neighbor CONSERVED variables.
     * \param[inout] NFn: pointer to memory that will contain the updated numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void AddArtificialViscosity(const amrex::Real * Un_left, const amrex::Real * Un_right,
                                amrex::Real * NFn) const
    {
        const amrex::Real g = this->gamma;
        amrex::Real rl, AMREX_D_DECL(ul, ut1l, ut2l), pl, al;
        amrex::Real rr, AMREX_D_DECL(ur, ut1r, ut2r), pr, ar;
        amrex::Real dr, AMREX_D_DECL(dm1, dm2, dm3), de;
        amrex::Real K;

        this->eval_primary_variables(Un_left, rl, AMREX_D_DECL(ul, ut1l, ut2l), pl);
        this->eval_primary_variables(Un_right, rr, AMREX_D_DECL(ur, ut1r, ut2r), pr);

        al = std::sqrt(g*pl/rl);
        ar = std::sqrt(g*pr/rr);
        
        K = amrex::max(std::abs(al+ul), std::abs(al-ul));
        K = amrex::max(K, amrex::max(std::abs(ar+ur), std::abs(ar-ur)));
        K *= 0.1;

        dr = Un_right[RHO]-Un_left[RHO];
        AMREX_D_TERM
        (
            dm1 = Un_right[RHOV1]-Un_left[RHOV1];,
            dm2 = Un_right[RHOV2]-Un_left[RHOV2];,
            dm3 = Un_right[RHOV3]-Un_left[RHOV3];
        )
        de = Un_right[TE]-Un_left[TE];
        
        NFn[RHO] -= K*dr;
        AMREX_D_TERM
        (
            NFn[RHOV1] -= K*dm1;,
            NFn[RHOV2] -= K*dm2;,
            NFn[RHOV3] -= K*dm3;
        )
        NFn[TE] -= K*de;
    }
    
    /**
     * \brief Eval the numerical flux at the interface between two adjacent states.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] nbr_U: pointer to memory containing the neighbor conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int /*dom*/,
                  const amrex::Real /*t*/, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real tmp[DG_N_SOL];
        amrex::Real rl, AMREX_D_DECL(ul, ut1l, ut2l), pl;
        amrex::Real rr, AMREX_D_DECL(ur, ut1r, ut2r), pr;

        // EVAL PRIMARY VARIABLES IN THE UNIT NORMAL REFERENCE SYSTEM -
        this->eval_U2Un(un, U, tmp);
        this->eval_primary_variables(tmp, rl, AMREX_D_DECL(ul, ut1l, ut2l), pl);

        this->eval_U2Un(un, nbr_U, tmp);
        this->eval_primary_variables(tmp, rr, AMREX_D_DECL(ur, ut1r, ut2r), pr);
        // ------------------------------------------------------------

        // SOLVE RIEMANN PROBLEM --------------------------------------
        // We NFn as a temporary variable
        this->ExactRiemannSolver(rl, AMREX_D_DECL(ul, ut1l, ut2l), pl,
                                 rr, AMREX_D_DECL(ur, ut1r, ut2r), pr,
                                 NFn);

        // EVAL THE FLUX
        this->eval_F1(NFn, tmp);
        
        // ADD ARTIFICIAL VISCOSITY
        {
            amrex::Real Un[DG_N_SOL];
            amrex::Real nbr_Un[DG_N_SOL];

            this->eval_U2Un(un, U, Un);
            this->eval_U2Un(un, nbr_U, nbr_Un);
            this->AddArtificialViscosity(Un, nbr_Un, tmp);
        }
        
        this->eval_Un2U(un, tmp, NFn);
        // ------------------------------------------------------------

        // PRINT TO SCREEN --------------------------------------------
        /*
        amrex::Print() << "    U: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, U);
        amrex::Print() << "nbr_U: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, nbr_U);
        amrex::Print() << "NFn: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, NFn);
        */
        // ------------------------------------------------------------
    }

    /**
     * \brief Eval the numerical flux at a boundary to enforce wall bcs.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_WALL(const int dom,
                   const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                   const amrex::Real * U,
                   amrex::Real * NFn) const
    {
        amrex::Real ghost_U[DG_N_SOL];

        // We use NFn as a temporary variables
        this->eval_U2Un(un, U, NFn);
        NFn[RHOV1] = -NFn[RHOV1];
        this->eval_Un2U(un, NFn, ghost_U);

        this->F_NF_ICS(dom, t, x, un, U, ghost_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a boundary to enforce inflow bcs.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] bcs_U: pointer to memory containing the conserved variables with desired boundary 
     *                   conditions.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_INFLOW(const int dom,
                     const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                     const amrex::Real * U, const amrex::Real * bcs_U,
                     amrex::Real * NFn) const
    {
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a boundary to enforce outflow bcs.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] out_U: pointer to memory containing the conserved variables at the outlet. 
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_OUTFLOW(const int /*dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                     const amrex::Real * U, const amrex::Real * /*out_U*/,
                     amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real tmp[DG_N_SOL];
        amrex::Real /*rl,*/ AMREX_D_DECL(ul, ut1l, ut2l), pl;
        amrex::Real a, h;

        // We use NFn as a temporary variables
        this->eval_U2Un(un, U, NFn);
        this->eval_enthalpy_variables(NFn, AMREX_D_DECL(ul, ut1l, ut2l), pl, a, h);

        // SUPERSONIC OUTLET
        if ((ul-a) > 0.0)
        {
            this->eval_F1(NFn, tmp);
            this->eval_Un2U(un, tmp, NFn);
        }
        else
        {
amrex::Abort("IDEAL_GAS_BASE.F_NF_OUTFLOW");
        }

        // PRINT TO SCREEN --------------------------------------------
        /*
        amrex::Print() << "ul-a: " << (ul-a) << std::endl;
        amrex::Print() << "ul: " << ul << std::endl;
        amrex::Print() << "ul+a: " << (ul+a) << std::endl;
        */
        // ------------------------------------------------------------
    }

    /**
     * \brief Eval the numerical flux at a boundary to enforce bcs.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] bcs_U: pointer to memory containing the conserved variables with desired boundary 
     *                   conditions.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PRESCRIBED_VELOCITY_1(const int dom,
                                    const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                                    const amrex::Real * Un, const amrex::Real * bcs_Un,
                                    amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), pl;
        amrex::Real a, h;

        this->eval_enthalpy_variables(Un, AMREX_D_DECL(u1, u2, u3), pl, a, h);

        // OUTLET
        if (u1 > 0.0)
        {
            // SUPERSONIC OUTLET
            if ((u1-a) > 0.0)
            {
                this->eval_F1(Un, NFn);
            }
            // SUBSONIC OUTLET
            else
            {
                amrex::Real L[DG_N_SOL], bcs_L[DG_N_SOL];
                this->eval_eigen_decomposition_1(Un, L);
                this->eval_eigen_decomposition_1(bcs_Un, bcs_L);

                L[RHO] = bcs_L[RHO];

                // We use bcs_L as a temporary variable
                this->eval_eigen_reconstruction_1(Un, L, bcs_L);
                this->eval_F1(bcs_L, NFn);
            }
        }
        // INLET
        else if (u1 < 0.0)
        {
            // SUPERSONIC INLET
            if ((u1+a) < 0.0)
            {
                this->eval_F1(bcs_Un, NFn);
            }
            // SUBSONIC INLET
            else
            {
                amrex::Real L[DG_N_SOL], bcs_L[DG_N_SOL];
                this->eval_eigen_decomposition_1(Un, L);
                this->eval_eigen_decomposition_1(bcs_Un, bcs_L);

                bcs_L[TE] = L[TE];

                // We use L as a temporary variable
                this->eval_eigen_reconstruction_1(Un, bcs_L, L);
                this->eval_F1(L, NFn);
            }
        }
        // SLIP WALL
        else
        {
amrex::Abort("F_NF_PRESCRIBED_VELOCITY_1 - SLIP WALL");
        }
    }
    
    /**
     * \brief Eval the numerical flux at a boundary to enforce bcs.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] bcs_U: pointer to memory containing the conserved variables with desired boundary 
     *                   conditions.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PRESCRIBED_VELOCITY(const int dom,
                                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                                  const amrex::Real * U, const amrex::Real * bcs_U,
                                  amrex::Real * NFn) const
    {
        amrex::Real Un[DG_N_SOL], bcs_Un[DG_N_SOL], tmp[DG_N_SOL];

        this->eval_U2Un(un, U, Un);
        this->eval_U2Un(un, bcs_U, bcs_Un);

        this->F_NF_PRESCRIBED_VELOCITY_1(dom, t, x, un, Un, bcs_Un, tmp);

        this->eval_Un2U(un, tmp, NFn);
    }
    // ===============================================================


    // SLOPE RECONSTRUCTION FOR LEVELS USING FINITE-VOLUME SCHEMES ===
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void UpdateLeastSquareSystem(const amrex::Real * xc, const amrex::Real * P,
                                 const amrex::Real * nbr_xc, const amrex::Real * nbr_P,
                                 amrex::Real * S, amrex::Real * rhs) const
    {
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
        S[1+2*3] += (nbr_xc[2]-xc[2])*(nbr_xc[2]-xc[2]);
        S[2+1*3]  = S[1+2*3];
        S[2+2*3] += (nbr_xc[2]-xc[2])*(nbr_xc[2]-xc[2]);
#endif
        for (int u = 0; u < DG_N_SOL; ++u)
        {
            AMREX_D_TERM
            (
                rhs[0+u*AMREX_SPACEDIM] += (nbr_xc[0]-xc[0])*(nbr_P[u]-P[u]);,
                rhs[1+u*AMREX_SPACEDIM] += (nbr_xc[1]-xc[1])*(nbr_P[u]-P[u]);,
                rhs[2+u*AMREX_SPACEDIM] += (nbr_xc[2]-xc[2])*(nbr_P[u]-P[u]);
            )
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void Fab2xcP(const int i, const int j, const int k, const int dom,
                 const amrex::Array4<const amrex::Real> & eInfo_fab,
                 const amrex::Array4<const amrex::Real> & X_fab,
                 amrex::Real * xc, amrex::Real * P) const
    {
        // VARIABLES
        amrex::Real U[DG_N_SOL];

        // CENTROID
        AMREX_D_TERM
        (
            xc[0] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X1(dom));,
            xc[1] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X2(dom));,
            xc[2] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X3(dom));
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

        // PRIMARY VARIABLES
        this->F_U2P(U, P);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void LimitSlopes(amrex::Real * slopes, const amrex::Real * tmp_slopes, const amrex::Real theta = 1.0) const
    {
        for (int u = 0; u < DG_N_SOL; ++u)
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
    
    void F_R_SLOPES(const amrex::Real t,
                    const amrex::DG::ImplicitMesh & mesh,
                    const int /*N_DOM*/,
                    const int /*N_SOL*/,
                    const amrex::MultiFab & X,
                    amrex::MultiFab & dX) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const amrex::Box domain = mesh.geom.Domain();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();
        const amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.geom.isPeriodicArray();

        // DOMAINS
        const int dom = 0;
        // ------------------------------------------------------------

        // INITIALIZATION
        dX = 0.0;
        // --------------

        // EVAL THE SLOPES
        for (amrex::MFIter mfi(dX); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<amrex::Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real> const & dX_fab = dX.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_VALID(etype))
                {
                    // PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};

                    // VARIABLES
                    amrex::Real xc[AMREX_SPACEDIM];
                    amrex::Real U[DG_N_SOL], P[DG_N_SOL];
                    int n_nbr;
                    bool nbr_is_valid[__DG_BASE_STENCIL_N_NBR__];
                    amrex::Real nbr_Ps[DG_N_SOL*__DG_BASE_STENCIL_N_NBR__];
                    amrex::Real nbr_xcs[AMREX_SPACEDIM*__DG_BASE_STENCIL_N_NBR__];
                    amrex::Real S[AMREX_SPACEDIM*AMREX_SPACEDIM], rhs[AMREX_SPACEDIM*DG_N_SOL];
                    amrex::Real dP[AMREX_SPACEDIM*DG_N_SOL];
                    amrex::Real tmp_dP[DG_N_SOL], tmp_dC[DG_N_SOL];
                    amrex::Real theta;

                    // CONSERVED VARIABLES
                    U[RHO] = X_fab(i,j,k,RHO);
                    AMREX_D_TERM
                    (
                        U[RHOV1] = X_fab(i,j,k,RHOV1);,
                        U[RHOV2] = X_fab(i,j,k,RHOV2);,
                        U[RHOV3] = X_fab(i,j,k,RHOV3);
                    )
                    U[TE] = X_fab(i,j,k,TE);
                    
                    // ELEMENT'S CENTROID AND PRIMARY VARIABLES
                    this->Fab2xcP(i, j, k, dom, eInfo_fab, X_fab, xc, P);

                    // INITIALIZATION
                    for (int u = 0; u < DG_N_SOL*__DG_BASE_STENCIL_N_NBR__; ++u)
                    {
                        nbr_Ps[u] = 0.0;
                    }
                    for (int u = 0; u < AMREX_SPACEDIM*__DG_BASE_STENCIL_N_NBR__; ++u)
                    {
                        nbr_xcs[u] = 0.0;
                    }

                    // COLLECT NEIGHBORING ELEMENTS INFORMATION -------
                    n_nbr = 0;
                    for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                    {
                        const int nbr_i = i+amrex::DG::base_stencil_table_i[n];
                        const int nbr_j = j+amrex::DG::base_stencil_table_j[n];
                        const int nbr_k = k+amrex::DG::base_stencil_table_k[n];
                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
                        const bool nbr_is_outside_domain = !domain.contains(nbr_i, nbr_j, nbr_k);

                        if (amrex::DG::ELM_IS_NOT_EMPTY(nbr_etype))
                        {
                            // NEIGHBOR'S CENTROID AND PRIMARY VARIABLES
                            int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                            amrex::DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            bool bad_nbr;
                            bad_nbr = ((nbr_BF_i == i) && (nbr_BF_j == j) && (nbr_BF_k == k));
                            bad_nbr = bad_nbr || (nbr_is_outside_domain && (is_periodic[n/2] == 0));
                            
                            if (bad_nbr)
                            {
                                nbr_is_valid[n] = false;
                            }
                            else
                            {
                                nbr_is_valid[n] = true;
                                this->Fab2xcP(nbr_BF_i, nbr_BF_j, nbr_BF_k, dom, eInfo_fab, X_fab, &nbr_xcs[AMREX_SPACEDIM*n], &nbr_Ps[DG_N_SOL*n]);
                                n_nbr += 1;
                            }
                        }
                        else
                        {
                            nbr_is_valid[n] = false;
                        }
                    }
                    // ------------------------------------------------

/*
if (i == __i__ && j == __j__)
{
amrex::Print() << "xc: " << std::endl;
amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, 1, xc);
amrex::Print() << "P: " << std::endl;
amrex::DG::IO::PrintRealArray2D(DG_N_SOL, 1, P);
amrex::Print() << "nbr_is_valid: " << std::endl;
for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
{
    const int nbr_i = i+amrex::DG::base_stencil_table_i[n];
    const int nbr_j = j+amrex::DG::base_stencil_table_j[n];
    const int nbr_k = k+amrex::DG::base_stencil_table_k[n];
    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
    amrex::Print() << ((nbr_is_valid[n]) ? "yes - " : "no - ") << amrex::DG::ELM_TYPE_DESCRIPTION(nbr_etype) << std::endl;
}
amrex::Print() << "nbr_xcs: " << std::endl;
amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, __DG_BASE_STENCIL_N_NBR__, nbr_xcs);
amrex::Print() << "nbr_Ps: " << std::endl;
amrex::DG::IO::PrintRealArray2D(DG_N_SOL, __DG_BASE_STENCIL_N_NBR__, nbr_Ps);
exit(-1);
}
*/

                    // EVAL LEAST-SQUARE RECONSTRUCTION ---------------
                    if (n_nbr >= AMREX_SPACEDIM)
                    {
                        for (int u = 0; u < AMREX_SPACEDIM*AMREX_SPACEDIM; ++u) S[u] = 0.0;
                        for (int u = 0; u < AMREX_SPACEDIM*DG_N_SOL; ++u) rhs[u] = 0.0;

                        for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                        {
                            if (nbr_is_valid[n])
                            {
                                this->UpdateLeastSquareSystem(xc, P, &nbr_xcs[AMREX_SPACEDIM*n], &nbr_Ps[DG_N_SOL*n], S, rhs);
                            }
                        }

#if (AMREX_SPACEDIM == 1)
                        for (int u = 0; u < DG_N_SOL; ++u) dP[u] = rhs[u]/S[0];
#endif
#if (AMREX_SPACEDIM == 2)
                        amrex::DG::linalg::dpo2s(DG_N_SOL, S, rhs, dP);
#endif
#if (AMREX_SPACEDIM == 3)
                        amrex::DG::linalg::dpo3s(DG_N_SOL, S, rhs, dP);
#endif

                    }
                    else
                    {
                        for (int u = 0; u < AMREX_SPACEDIM*DG_N_SOL; ++u)
                        {
                            dP[u] = 0.0;
                        }
                    }

                    amrex::DG::linalg::transpose(AMREX_SPACEDIM, DG_N_SOL, dP, rhs);
                    for (int u = 0; u < DG_N_SOL*AMREX_SPACEDIM; ++u)
                    {
                        dP[u] = rhs[u];
                    }
                    // ------------------------------------------------

/*
if (i == __i__ && j == __j__)
{
amrex::Print() << "dP: " << std::endl;
amrex::DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dP);
}
*/


                    // LIMIT THE COMPUTED SLOPES ----------------------
                    /**/
                    // STORE THE CHARACTERISTICS CENTRAL DIFFERENCE IN rhs
                    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                    {
                        amrex::Real un[AMREX_SPACEDIM];
                        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                        un[dir] = +1.0;
                        
                        this->F_DP2DC(t, xc, un, P, &dP[dir*DG_N_SOL], &rhs[dir*DG_N_SOL]);
                    }

                    // X1 DIRECTION
                    {
                        const int dir = 0;
                        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(1.0, 0.0, 0.0)};

                        for (int n = 2*dir; n < 2*(dir+1); ++n)
                        {
                            if (nbr_is_valid[n])
                            {
                                // SLOPE COEFFICIENT
                                theta = (nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0])/(cell_lo[0]+(n%2)*dx[0]-xc[0]);

                                // RECONSTRUCTED DIFFERENCE
                                for (int u = 0; u < DG_N_SOL; ++u)
                                {
                                    tmp_dP[u] = -P[u];
                                    AMREX_D_TERM
                                    (
                                        ,
                                        tmp_dP[u] -= dP[u+1*DG_N_SOL]*(nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);,
                                        tmp_dP[u] -= dP[u+2*DG_N_SOL]*(nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                                    )
                                    tmp_dP[u] += nbr_Ps[u+n*DG_N_SOL];
                                    tmp_dP[u] /= (nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);
                                }

                                // CHARACTERISTICS
                                this->F_DP2DC(t, xc, un, P, tmp_dP, tmp_dC);

                                // LIMIT SLOPES
                                this->LimitSlopes(&rhs[dir*DG_N_SOL], tmp_dC, theta);
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
                            if (nbr_is_valid[n])
                            {
                                // SLOPE COEFFICIENT
                                theta = (nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1])/(cell_lo[1]+(n%2)*dx[1]-xc[1]);

                                // RECONSTRUCTED DIFFERENCE
                                for (int u = 0; u < DG_N_SOL; ++u)
                                {
                                    tmp_dP[u] = -P[u];
                                    AMREX_D_TERM
                                    (
                                        tmp_dP[u] -= dP[u+0*DG_N_SOL]*(nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);,
                                        ,
                                        tmp_dP[u] -= dP[u+2*DG_N_SOL]*(nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                                    )
                                    tmp_dP[u] += nbr_Ps[u+n*DG_N_SOL];
                                    tmp_dP[u] /= (nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);
                                }

                                // CHARACTERISTICS
                                this->F_DP2DC(t, xc, un, P, tmp_dP, tmp_dC);

                                // LIMIT SLOPES
                                this->LimitSlopes(&rhs[dir*DG_N_SOL], tmp_dC, theta);
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
                            if (nbr_is_valid[n])
                            {
                                // SLOPE COEFFICIENT
                                theta = (nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2])/(cell_lo[2]+(n%2)*dx[2]-xc[2]);

                                // RECONSTRUCTED DIFFERENCE
                                for (int u = 0; u < DG_N_SOL; ++u)
                                {
                                    tmp_dP[u] = -P[u];
                                    AMREX_D_TERM
                                    (
                                        tmp_dP[u] -= dP[u+0*DG_N_SOL]*(nbr_xcs[0+n*AMREX_SPACEDIM]-xc[0]);,
                                        tmp_dP[u] -= dP[u+1*DG_N_SOL]*(nbr_xcs[1+n*AMREX_SPACEDIM]-xc[1]);,
                                        
                                    )
                                    tmp_dP[u] += nbr_Ps[u+n*DG_N_SOL];
                                    tmp_dP[u] /= (nbr_xcs[2+n*AMREX_SPACEDIM]-xc[2]);
                                }

                                // CHARACTERISTICS
                                this->F_DP2DC(t, xc, un, P, tmp_dP, tmp_dC);

                                // LIMIT SLOPES
                                this->LimitSlopes(&rhs[dir*DG_N_SOL], tmp_dC, theta);
                            }
                        }
                    }
#endif
                    // BACK TO PRIMARY VARIABLES
                    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                    {
                        amrex::Real un[AMREX_SPACEDIM];
                        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                        un[dir] = +1.0;
                        
                        this->F_DC2DP(t, xc, un, P, &rhs[dir*DG_N_SOL], &dP[dir*DG_N_SOL]);
                    }
                    /**/
                    // ------------------------------------------------

                    // STORE THE SLOPES -------------------------------
                    /*
                    this->F_DP2DU(t, xc, U, &dP[0*DG_N_SOL], &rhs[0*DG_N_SOL]);
#if (AMREX_SPACEDIM > 1)
                    this->F_DP2DU(t, xc, U, &dP[1*DG_N_SOL], &rhs[1*DG_N_SOL]);
#endif
#if (AMREX_SPACEDIM > 2)
                    this->F_DP2DU(t, xc, U, &dP[2*DG_N_SOL], &rhs[2*DG_N_SOL]);
#endif

                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        AMREX_D_TERM
                        (
                            dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = rhs[u+0*DG_N_SOL];,
                            dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = rhs[u+1*DG_N_SOL];,
                            dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = rhs[u+2*DG_N_SOL];
                        )
                    }
                    */

/*
if (i == __i__ && j == __j__)
{
amrex::Print() << "dP: " << std::endl;
amrex::DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, rhs);
amrex::Print() << std::endl;
exit(-1);
}
*/
                    /**/
                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        AMREX_D_TERM
                        (
                            dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = dP[u+0*DG_N_SOL];,
                            dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = dP[u+1*DG_N_SOL];,
                            dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = dP[u+2*DG_N_SOL];
                        )
                    }
                    /**/
                    // ------------------------------------------------------------
                }
            });
            amrex::Gpu::synchronize();
        }
        dX.FillBoundary(mesh.geom.periodicity());


        // APPLY THE BARTH JESPERSEN LIMITER
        /**/
#ifdef AMREX_USE_CUDA
        amrex::Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
        amrex::Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
        amrex::Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
        amrex::Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif
        amrex::MultiFab bounds(X.boxarray, X.distributionMap, 2*DG_N_SOL, X.n_grow);
        amrex::MultiFab alpha(X.boxarray, X.distributionMap, DG_N_SOL, X.n_grow);
        bounds = 0.0;
        alpha = 1.0;

        // SET BOUNDS
        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<amrex::Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);

            amrex::Array4<amrex::Real> const & bounds_fab = bounds.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_VALID(etype))
                {
                    // VARIABLES
                    amrex::Real dummy_x[AMREX_SPACEDIM];
                    amrex::Real tmp_P[DG_N_SOL];
                    amrex::Real min_P[DG_N_SOL], max_P[DG_N_SOL];
                    
                    // ELEMENT'S CENTROID AND PRIMARY VARIABLES
                    this->Fab2xcP(i, j, k, dom, eInfo_fab, X_fab, dummy_x, tmp_P);

                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        min_P[u] = tmp_P[u];
                        max_P[u] = tmp_P[u];
                    }

                    for (int n = 0; n < __DG_BASE_STENCIL_N_NBR__; ++n)
                    {
                        const int nbr_i = i+amrex::DG::base_stencil_table_i[n];
                        const int nbr_j = j+amrex::DG::base_stencil_table_j[n];
                        const int nbr_k = k+amrex::DG::base_stencil_table_k[n];
                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
                        const bool nbr_is_outside_domain = !domain.contains(nbr_i, nbr_j, nbr_k);

                        if (amrex::DG::ELM_IS_NOT_EMPTY(nbr_etype))
                        {
                            // NEIGHBOR'S CENTROID AND PRIMARY VARIABLES
                            int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                            amrex::DG::BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                            bool bad_nbr;
                            bad_nbr = ((nbr_BF_i == i) && (nbr_BF_j == j) && (nbr_BF_k == k));
                            bad_nbr = bad_nbr || (nbr_is_outside_domain && (is_periodic[n/2] == 0));
                            
                            if (!bad_nbr)
                            {
                                this->Fab2xcP(nbr_BF_i, nbr_BF_j, nbr_BF_k, dom, eInfo_fab, X_fab, dummy_x, tmp_P);

                                for (int u = 0; u < DG_N_SOL; ++u)
                                {
                                    min_P[u] = amrex::min(min_P[u], tmp_P[u]);
                                    max_P[u] = amrex::max(max_P[u], tmp_P[u]);
                                }
                            }
                        }
                    }

                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        bounds_fab(i,j,k,u) = min_P[u];
                        bounds_fab(i,j,k,u+DG_N_SOL) = max_P[u];
                    }
                }

            });
            amrex::Gpu::synchronize();
        }
        bounds.FillBoundary(mesh.geom.periodicity());

        // CELL BOUNDARIES
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();
                amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
                amrex::Array4<amrex::Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
                amrex::Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
                amrex::Array4<int const> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
                amrex::Array4<long const> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

                amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
                amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);
                
                amrex::Array4<amrex::Real const> const & bounds_fab = bounds.array(mfi);
                amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                    if (amrex::DG::ELM_IS_NOT_EMPTY(etype))
                    {
                        // LOCAL PARAMETERS
                        const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                                  prob_lo[1]+j*dx[1],
                                                                                  prob_lo[2]+k*dx[2])};

                        // LOCAL VARIABLES
                        int BF_i, BF_j, BF_k;
                        amrex::Real xc[AMREX_SPACEDIM], P[DG_N_SOL], min_P[DG_N_SOL], max_P[DG_N_SOL], Pr[DG_N_SOL];
                        int fi, fj, fk;
                        amrex::Real alphas[DG_N_SOL];

                        amrex::DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                        this->Fab2xcP(BF_i, BF_j, BF_k, dom, eInfo_fab, X_fab, xc, P);
                        for (int u = 0; u < DG_N_SOL; ++u)
                        {
                            min_P[u] = bounds_fab(BF_i,BF_j,BF_k,u);
                            max_P[u] = bounds_fab(BF_i,BF_j,BF_k,u+DG_N_SOL);
                            alphas[u] = alpha_fab(i,j,k,u);
                        }

                        for (int b = 2*dir; b < 2*(dir+1); ++b)
                        {
                            amrex::DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                            // ELEMENT BOUNDARY TYPE
                            const short bou_type = eBouType_fab(fi,fj,fk,amrex::DG::ELM_BOU_TYPE(dom));

                            if (amrex::DG::ELM_BOU_IS_VALID(bou_type))
                            {
                                const int bou_Nq = eBouQuad_Nq_fab(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_NQ(dom, b));
                                const long pos = eBouQuad_pos_fab(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_POS(dom, b));
                                const amrex::Real * x_ptr = &eBouQuad_mem_ptr[pos];

                                for (int q = 0; q < bou_Nq; ++q)
                                {
                                    // QUADRATURE POINT AND WEIGHT
                                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                                    const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0]+cell_lo[0],
                                                                                        x_ptr[x_pos+1]+cell_lo[1],
                                                                                        x_ptr[x_pos+2]+cell_lo[2])};

                                    // EVAL RECONSTRUCTED SOLUTION
                                    for (int u = 0; u < DG_N_SOL; ++u)
                                    {
                                        Pr[u] = P[u];
                                        AMREX_D_TERM
                                        (
                                            Pr[u] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                            Pr[u] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                            Pr[u] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                        )
                                    }

                                    // UPDATE LIMITING COEFFICIENTS
                                    for (int u = 0; u < DG_N_SOL; ++u)
                                    {
                                        if (Pr[u] > max_P[u])
                                        {
                                            alphas[u] = amrex::min(alphas[u], (max_P[u]-P[u])/(Pr[u]-P[u]));
                                        }
                                        else if (Pr[u] < min_P[u])
                                        {
                                            alphas[u] = amrex::min(alphas[u], (min_P[u]-P[u])/(Pr[u]-P[u]));
                                        }
                                    }
                                }
                            }
                        }

                        // STORE UPDATED LIMITING COEFFICIENTS
                        for (int u = 0; u < DG_N_SOL; ++u)
                        {
                            alpha_fab(i,j,k,u) = alphas[u];
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
        }

        // INTERNAL BOUNDARIES
        {
            for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();
                amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
                amrex::Array4<amrex::Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
                amrex::Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
                amrex::Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

                amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
                amrex::Array4<amrex::Real const> const & dX_fab = dX.array(mfi);
                
                amrex::Array4<amrex::Real const> const & bounds_fab = bounds.array(mfi);
                amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                    if (amrex::DG::ELM_IS_NOT_EMPTY(etype))
                    {
                        // LOCAL PARAMETERS
                        const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,amrex::DG::ELM_INT_BOU_QUAD_NQ(dom));
                        const long pos = eIntBouQuad_pos_fab(i,j,k,amrex::DG::ELM_INT_BOU_QUAD_POS(dom));
                        const amrex::Real * x_ptr = &eIntBouQuad_mem_ptr[pos];
                        
                        // LOCAL VARIABLES
                        int BF_i, BF_j, BF_k;
                        amrex::Real xc[AMREX_SPACEDIM], P[DG_N_SOL], min_P[DG_N_SOL], max_P[DG_N_SOL], Pr[DG_N_SOL];
                        amrex::Real alphas[DG_N_SOL];

                        amrex::DG::BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                        this->Fab2xcP(BF_i, BF_j, BF_k, dom, eInfo_fab, X_fab, xc, P);
                        for (int u = 0; u < DG_N_SOL; ++u)
                        {
                            min_P[u] = bounds_fab(BF_i,BF_j,BF_k,u);
                            max_P[u] = bounds_fab(BF_i,BF_j,BF_k,u+DG_N_SOL);
                            alphas[u] = alpha_fab(i,j,k,u);
                        }

                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            // QUADRATURE POINT AND WEIGHT
                            const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                            const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(x_ptr[x_pos+0],
                                                                                x_ptr[x_pos+1],
                                                                                x_ptr[x_pos+2])};

                            // EVAL RECONSTRUCTED SOLUTION
                            for (int u = 0; u < DG_N_SOL; ++u)
                            {
                                Pr[u] = P[u];
                                AMREX_D_TERM
                                (
                                    Pr[u] += dX_fab(BF_i,BF_j,BF_k,0+AMREX_SPACEDIM*u)*(x[0]-xc[0]);,
                                    Pr[u] += dX_fab(BF_i,BF_j,BF_k,1+AMREX_SPACEDIM*u)*(x[1]-xc[1]);,
                                    Pr[u] += dX_fab(BF_i,BF_j,BF_k,2+AMREX_SPACEDIM*u)*(x[2]-xc[2]);
                                )
                            }

                            // UPDATE LIMITING COEFFICIENTS
                            for (int u = 0; u < DG_N_SOL; ++u)
                            {
                                if (Pr[u] > max_P[u])
                                {
                                    alphas[u] = amrex::min(alphas[u], (max_P[u]-P[u])/(Pr[u]-P[u]));
                                }
                                else if (Pr[u] < min_P[u])
                                {
                                    alphas[u] = amrex::min(alphas[u], (min_P[u]-P[u])/(Pr[u]-P[u]));
                                }
                            }
                        }

                        // STORE UPDATED LIMITING COEFFICIENTS
                        for (int u = 0; u < DG_N_SOL; ++u)
                        {
                            alpha_fab(i,j,k,u) = alphas[u];
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
            alpha.FillBoundary(mesh.geom.periodicity());
        }

        // HANDLE EXTENDED ELEMENTS
        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            amrex::Array4<amrex::Real> const & alpha_fab = alpha.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_EXTENDED(etype))
                {
                    // LOCAL VARIABLES
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    amrex::Real alphas[DG_N_SOL];

                    // INITIALIZATION
                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        alphas[u] = alpha_fab(i,j,k,u);
                    }

                    // LOOP OVER THE BOUNDARIES
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        amrex::DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
                        const bool nbr_is_small = amrex::DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                        const bool cells_are_merged = (nbr_merged_b == nbr_b);

                        if (cells_are_merged)
                        {
                            for (int u = 0; u < DG_N_SOL; ++u)
                            {
                                alphas[u] = amrex::min(alphas[u], alpha_fab(nbr_i,nbr_j,nbr_k,u));
                            }
                        }
                    }

                    // STORE THE VALUES
                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        alpha_fab(i,j,k,u) = alphas[u];
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
        alpha.FillBoundary(mesh.geom.periodicity());

        // EVENTUALLY STORE THE LIMITED SLOPES
        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<amrex::Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

            amrex::Array4<amrex::Real const> const & alpha_fab = alpha.array(mfi);

            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real> const & dX_fab = dX.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_VALID(etype))
                {
                    amrex::Real xc[AMREX_SPACEDIM];
                    amrex::Real U[DG_N_SOL];
                    amrex::Real dP[DG_N_SOL*AMREX_SPACEDIM], dU[DG_N_SOL*AMREX_SPACEDIM];

                    // CENTROID
                    AMREX_D_TERM
                    (
                        xc[0] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X1(dom));,
                        xc[1] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X2(dom));,
                        xc[2] = eInfo_fab(i,j,k,amrex::DG::ELM_CENTROID_X3(dom));
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

                    // LIMITED SLOPES OF THE PRIMARY VARIABLES
                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        AMREX_D_TERM
                        (
                            dP[u+0*DG_N_SOL] = dX_fab(i,j,k,0+AMREX_SPACEDIM*u)*alpha_fab(i,j,k,u);,
                            dP[u+1*DG_N_SOL] = dX_fab(i,j,k,1+AMREX_SPACEDIM*u)*alpha_fab(i,j,k,u);,
                            dP[u+2*DG_N_SOL] = dX_fab(i,j,k,2+AMREX_SPACEDIM*u)*alpha_fab(i,j,k,u);
                        )
                    }

                    // LIMITED SLOPES OF THE CONSERVED VARIABLES
                    this->F_DP2DU(t, xc, U, &dP[0*DG_N_SOL], &dU[0*DG_N_SOL]);
#if (AMREX_SPACEDIM > 1)
                    this->F_DP2DU(t, xc, U, &dP[1*DG_N_SOL], &dU[1*DG_N_SOL]);
#endif
#if (AMREX_SPACEDIM > 2)
                    this->F_DP2DU(t, xc, U, &dP[2*DG_N_SOL], &dU[2*DG_N_SOL]);
#endif

/*
if (i == __i__ && j == __j__)
{
amrex::Print() << "limited dU: " << std::endl;
amrex::DG::IO::PrintRealArray2D(DG_N_SOL, AMREX_SPACEDIM, dU);
amrex::Print() << std::endl;
}
*/

                    for (int u = 0; u < DG_N_SOL; ++u)
                    {
                        AMREX_D_TERM
                        (
                            dX_fab(i,j,k,0+AMREX_SPACEDIM*u) = dU[u+0*DG_N_SOL];,
                            dX_fab(i,j,k,1+AMREX_SPACEDIM*u) = dU[u+1*DG_N_SOL];,
                            dX_fab(i,j,k,2+AMREX_SPACEDIM*u) = dU[u+2*DG_N_SOL];
                        )
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
        dX.FillBoundary(mesh.geom.periodicity());
        /**/
    }
    // ================================================================

    // TAG THE TROUBLED CELLS =========================================
    template <typename MASK_MF, typename MASK_T>
    void F_UPDATE_TROUBLED(const amrex::Real /*t*/,
                           const amrex::DG::ImplicitMesh & mesh,
                           const amrex::DG::MatrixFactory & matfactory,
                           const amrex::DG::MatrixFactory & mf,
                           const amrex::MultiFab & X,
                           MASK_MF & mask,
                           const MASK_T masked) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const amrex::Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
        const amrex::Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

        // DG
        const int sp = matfactory.std_elem.p;

        const int sp_low = matfactory.std_elem.p-1;
        const int sNp_low = AMREX_D_TERM((1+sp_low),*(1+sp_low),*(1+sp_low));
        const int n_comp_low = DG_N_SOL*sNp_low;
        const amrex::Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, 0);

        // DOMAIN
        const int dom = 0;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::MultiFab X_low(X.boxarray, X.distributionMap, n_comp_low, 1);
        amrex::MultiFab sensor(X.boxarray, X.distributionMap, 2, 1);
        // ------------------------------------------------------------

        // INTERPOLATE THE SOLUTION USING A LOWER ORDER BASIS ---------
        X_low = 0.0;
        amrex::DG::AMR::Interpolate(DG_N_SOL, Sol2Dom,
                                    mesh, matfactory, X,
                                    amrex::IntVect::TheUnitVector(),
                                    mesh, mf, X_low);
        // ------------------------------------------------------------

        // UPDATE THE MASK
        sensor = 0.0;

        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
            amrex::Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);
            amrex::Array4<amrex::Real const> const & X_low_fab = X_low.array(mfi);

            auto mask_fab = mask.array(mfi);

            amrex::Array4<amrex::Real> const & sensor_fab = sensor.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_NOT_EMPTY(etype) && (mask_fab(i,j,k) != masked))
                {
                    // LOCAL PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};

                    // LOCAL VARIABLES
                    int BF_i, BF_j, BF_k;
                    amrex::Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    amrex::Real x[AMREX_SPACEDIM], w;
                    amrex::Real U[DG_N_SOL], U_trunc[DG_N_SOL];
                    amrex::Real P[DG_N_SOL], P_trunc[DG_N_SOL];
                    amrex::Real err, norm;
                    amrex::Real int_err, int_norm;

                    // SUPPORT OF THE BASIS FUNCTIONS
                    amrex::DG::BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                    // SOLUTION
                    amrex::DG::DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);
                    amrex::DG::DG_SOL_space_BFX sol_low(sp_low, BF_lo, BF_hi, X_low_fab, BF_i, BF_j, BF_k);

                    // EVAL THE INTEGRAL
                    err = norm = 0.0;

                    // SELF CONTRIBUTION
                    {
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,amrex::DG::ELM_DOM_QUAD_NQ(dom));
                        const long pos = eDomQuad_pos_fab(i,j,k,amrex::DG::ELM_DOM_QUAD_POS(dom));
                        const amrex::Real * xptr = &eDomQuad_mem_ptr[pos];

                        for (int q = 0; q < dom_Nq; ++q)
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
                            sol.eval_BF_only_table(x);
                            sol.eval_unsafe(0, DG_N_SOL, U);
                            sol_low.eval_BF_only_table(x);
                            sol_low.eval_unsafe(0, DG_N_SOL, U_trunc);
                            //sol.eval_unsafe_trunc(sp-1, 0, DG_N_SOL, U_trunc);

                            // ADD INTEGRAL CONTRIBUTIONS
                            int_err = int_norm = 0.0;
                            /*
                            for (int cu = 0; cu < DG_N_SOL; ++cu)
                            {
                                int_err += (U[cu]-U_trunc[cu])*(U[cu]-U_trunc[cu]);
                                int_norm += U[cu]*U[cu];
                            }
                            */

                            /**/
                            //this->F_U2P(U, P);
                            //this->F_U2P(U_trunc, P_trunc);
                            int_err += (U[RHO]-U_trunc[RHO])*(U[RHO]-U_trunc[RHO]);
                            int_norm += U[RHO]*U[RHO];
                            /**/

                            err += int_err*w;
                            norm += int_norm*w;
                        }
                    }

                    sensor_fab(i,j,k,0) = norm;
                    sensor_fab(i,j,k,1) = err;
                }
            });
            amrex::Gpu::synchronize();
        }
        sensor.FillBoundary(mesh.geom.periodicity());

        amrex::Real ds;
        if (sp == 1)
        {
            ds = AMREX_D_PICK(-1.5, -2.0, -2.425);
        }
        else if (sp == 2)
        {
            ds = AMREX_D_PICK(-1.5, -2.0, -2.425);
        }
        else if (sp == 3)
        {
            ds = AMREX_D_PICK(-2.5, -3.0, -3.425);
        }
        
        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<amrex::Real const> const & sensor_fab = sensor.array(mfi);

            auto mask_fab = mask.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_VALID(etype) && (mask_fab(i,j,k) != masked))
                {
                    // LOCAL VARIABLES
                    amrex::Real err, norm;

                    norm = sensor_fab(i,j,k,0);
                    err = sensor_fab(i,j,k,1);

                    // ADD CONTRIBUTION FROM SMALL ELEMENTS
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        amrex::DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
                        const bool nbr_is_small = amrex::DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        if (nbr_merged_b == nbr_b)
                        {
                            norm += sensor_fab(nbr_i,nbr_j,nbr_k,0);
                            err += sensor_fab(nbr_i,nbr_j,nbr_k,1);
                        }
                    }

                    err = err/norm;
                    if (err > 0.0)
                    {
                        const amrex::Real s = std::log10(err);
                        const amrex::Real s0 = -4.0*std::log10(sp);
                        
                        if (s > (s0+ds))
                        {
                            mask_fab(i,j,k) = masked;
                        }
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
        mask.FillBoundary(mesh.geom.periodicity());
    }

    template <typename MASK_MF, typename MASK_T>
    void F_EVAL_TROUBLED(const amrex::Real t,
                         const amrex::DG::ImplicitMesh & mesh,
                         const amrex::DG::MatrixFactory & matfactory,
                         const amrex::DG::MatrixFactory & mf,
                         const amrex::MultiFab & X,
                         MASK_MF & mask,
                         const MASK_T masked,
                         const MASK_T unmasked) const
    {
        mask.setVal(unmasked);
        this->F_UPDATE_TROUBLED(t, mesh, matfactory, mf, X, mask, masked);
    }

    template <typename MASK_MF, typename MASK_T>
    void F_UPDATE_TAGS(const amrex::Real /*t*/,
                       const amrex::DG::ImplicitMesh & mesh,
                       const amrex::DG::MatrixFactory & matfactory,
                       const amrex::DG::MatrixFactory & /*mf*/,
                       const amrex::MultiFab & X,
                       MASK_MF & mask,
                       const MASK_T masked) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        const amrex::Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
        const amrex::Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

        // DG
        const int sp = matfactory.std_elem.p;

        // DOMAIN
        const int dom = 0;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::MultiFab sensor(X.boxarray, X.distributionMap, 2, 1);
        // ------------------------------------------------------------

        // UPDATE THE MASK
        sensor = 0.0;

        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
            amrex::Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);

            auto mask_fab = mask.array(mfi);

            amrex::Array4<amrex::Real> const & sensor_fab = sensor.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_NOT_EMPTY(etype) && (mask_fab(i,j,k) != masked))
                {
                    // LOCAL PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};

                    // LOCAL VARIABLES
                    int BF_i, BF_j, BF_k;
                    amrex::Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    amrex::Real x[AMREX_SPACEDIM], w;
                    amrex::Real grad_rho[AMREX_SPACEDIM], grad_rho_mag;
                    amrex::Real err, norm;
                    amrex::Real int_err, int_norm;

                    // SUPPORT OF THE BASIS FUNCTIONS
                    amrex::DG::BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                    // SOLUTION
                    amrex::DG::DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                    // EVAL THE INTEGRAL
                    err = norm = 0.0;

                    // SELF CONTRIBUTION
                    {
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,amrex::DG::ELM_DOM_QUAD_NQ(dom));
                        const long pos = eDomQuad_pos_fab(i,j,k,amrex::DG::ELM_DOM_QUAD_POS(dom));
                        const amrex::Real * xptr = &eDomQuad_mem_ptr[pos];

                        for (int q = 0; q < dom_Nq; ++q)
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
                            sol.eval_BF_table(x);
                            sol.set_u(RHO);
                            sol.eval_grad_unsafe(grad_rho);
                            grad_rho_mag = std::sqrt(AMREX_D_PICK(grad_rho[0]*grad_rho[0],+grad_rho[1]*grad_rho[1],+grad_rho[2]*grad_rho[2]));

                            // ADD INTEGRAL CONTRIBUTIONS
                            int_err = int_norm = 0.0;
                            
                            int_err = grad_rho_mag;
                            int_norm = 1.0;

                            err += int_err*w;
                            norm += int_norm*w;
                        }
                    }

                    sensor_fab(i,j,k,0) = norm;
                    sensor_fab(i,j,k,1) = err;
                }
            });
            amrex::Gpu::synchronize();
        }
        sensor.FillBoundary(mesh.geom.periodicity());

        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            amrex::Array4<amrex::Real const> const & sensor_fab = sensor.array(mfi);

            auto mask_fab = mask.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_VALID(etype) && (mask_fab(i,j,k) != masked))
                {
                    // LOCAL VARIABLES
                    amrex::Real err, norm;

                    norm = sensor_fab(i,j,k,0);
                    err = sensor_fab(i,j,k,1);

                    // ADD CONTRIBUTION FROM SMALL ELEMENTS
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        amrex::DG::NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,amrex::DG::ELM_TYPE(dom));
                        const bool nbr_is_small = amrex::DG::ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        if (nbr_merged_b == nbr_b)
                        {
                            norm += sensor_fab(nbr_i,nbr_j,nbr_k,0);
                            err += sensor_fab(nbr_i,nbr_j,nbr_k,1);
                        }
                    }

                    err = err/norm;
                    if (err > 100.0)
                    {
                        mask_fab(i,j,k) = masked;
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
        mask.FillBoundary(mesh.geom.periodicity());
    }

    template <typename MASK_MF, typename MASK_T>
    void F_EVAL_TAGS(const amrex::Real t,
                     const amrex::DG::ImplicitMesh & mesh,
                     const amrex::DG::MatrixFactory & matfactory,
                     const amrex::DG::MatrixFactory & mf,
                     const amrex::MultiFab & X,
                     MASK_MF & mask,
                     const MASK_T masked,
                     const MASK_T unmasked) const
    {
        mask.setVal(unmasked);
        this->F_UPDATE_TAGS(t, mesh, matfactory, mf, X, mask, masked);
    }
    // ================================================================
};
// ####################################################################
// ####################################################################