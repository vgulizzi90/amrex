//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Base.H
 * \brief Contains base constants and data structures for IBVP of ideal gases.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define FV_N_SOL (1+AMREX_SPACEDIM+1)

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
    amrex::Real gamma;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
     * 
     * \param[in] gamma_: ratio of specific heats.
    */
    IDEAL_GAS_BASE(const amrex::Real gamma_)
    :
    gamma{gamma_}
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

    // SOUNDNESS OF THE SOLUTION ######################################
    /**
     * \brief Check whether the solution at (t,x) is physically acceptable.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables fields at (t,x).
     *
     * \return true if the solution is physically acceptable, false otherwide.
    */
    AMREX_GPU_HOST_DEVICE
    bool F_TROUBLED(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                    const amrex::Real * U) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        return ((rho < 0.0) || (p < 0.0));
    }
    // ################################################################

    // TIME STEP ######################################################
    /**
     * \brief Eval the maximum time step.
     *
     * \param[in] dx: pointer to memory containing the sizes of the grid's element.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const amrex::Real * dx,
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

    // RIEMANN PROBLEM ################################################
    /**
     * \brief Auxiliary function for Riemann solver.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real _aux_gK_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;

        return std::sqrt(A/(ps+B));
    }

    /**
     * \brief Auxiliary function for Riemann solver.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real _aux_fK_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;
        amrex::Real res;

        if (ps > p)
        {
            res = (ps-p)*std::sqrt(A/(ps+B));
        }
        else
        {
            res = 2.0*a/(g-1.0)*(std::pow(ps/p, 0.5*(g-1.0)/g)-1.0);
        }

        return res;
    }

    /**
     * \brief Auxiliary function for Riemann solver.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real _aux_dfK_(const amrex::Real g,
                          const amrex::Real ps,
                          const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;
        amrex::Real res;

        if (ps > p)
        {
            res = (1.0-0.5*(ps-p)/(ps+B))*std::sqrt(A/(ps+B));
        }
        else
        {
            res = (1.0/(a*rho))*std::pow(ps/p, -0.5*(g+1.0)/g);
        }

        return res;
    }

    /**
     * \brief Auxiliary function for Riemann solver.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real _aux_f_(const amrex::Real g,
                        const amrex::Real ps,
                        const amrex::Real rhoL, const amrex::Real pL,
                        const amrex::Real rhoR, const amrex::Real pR,
                        const amrex::Real dun) const
    {
        const amrex::Real res = (this->_aux_fK_(g, ps, rhoL, pL)+this->_aux_fK_(g, ps, rhoR, pR)+dun);
        return res;
    }

    /**
     * \brief Auxiliary function for Riemann solver.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real _aux_df_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rhoL, const amrex::Real pL,
                         const amrex::Real rhoR, const amrex::Real pR) const
    {
        const amrex::Real res = (this->_aux_dfK_(g, ps, rhoL, pL)+this->_aux_dfK_(g, ps, rhoR, pR));
        return res;
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
    

    /**
     * \brief Solves the Riemann problem between left and right states in unit normal reference system.
     *
     * \param[in] (rhoL, u1L, u2L, u3L, pL): primary variables at the left of the shock.
     * \param[in] (rhoR, u1R, u2R, u3R, pR): primary variables at the right of the shock.
     * \param[out] URP: pointer to memory that will contain the solution of the Riemann problem, i.e.
     *                  the state at the shock.
    */
    AMREX_GPU_HOST_DEVICE
    void RiemannSolver_P_old(const amrex::Real rhoL, AMREX_D_DECL(const amrex::Real u1L, const amrex::Real u2L, const amrex::Real u3L), const amrex::Real pL,
                         const amrex::Real rhoR, AMREX_D_DECL(const amrex::Real u1R, const amrex::Real u2R, const amrex::Real u3R), const amrex::Real pR,
                         amrex::Real * URP) const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->gamma;
        const amrex::Real gg = (g-1.0)/(g+1.0);
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // LEFT STATE
        amrex::Real aL;

        // RIGHT STATE
        amrex::Real aR;

        // INTERFACE STATE
        amrex::Real rhoI, AMREX_D_DECL(u1I, u2I, u3I), pI;

        // AUXILIARY QUANTITIES
        amrex::Real dun, us;
        amrex::Real fs, dfs;
        amrex::Real pPV, pTS, ps, dps, aux_ps;
        amrex::Real gL, gR;
        
        const int n_max_it = 10;
        int it;
        // ------------------------------------------------------------

        // EVAL THE LEFT AND RIGHT STATES -----------------------------
        aL = std::sqrt(g*pL/rhoL);

        aR = std::sqrt(g*pR/rhoR);

        // SOME AUXILIARY QUANTITIES
        dun = u1R-u1L;
        // ------------------------------------------------------------

        // FIND PRESSURE AND VELOCITY IN STAR REGION ------------------
        pPV = 0.5*(pL+pR)-0.125*dun*(rhoL+rhoR)*(aL+aR);
        pPV = std::max(1.0e-10, pPV);
        gL = this->_aux_gK_(g, pPV, rhoL, pL);
        gR = this->_aux_gK_(g, pPV, rhoR, pR);
        pTS = (gL*pL+gR*pR-dun)/(gL+gR);

        // START ITERATION (TO COMPUTE PRESSURE)
        ps = pTS;
        it = 0;
        do
        {
            aux_ps = 0.5*ps;

            fs = this->_aux_f_(g, ps, rhoL, pL, rhoR, pR, dun);
            dfs = this->_aux_df_(g, ps, rhoL, pL, rhoR, pR);
            dps = -fs/dfs;
            ps += dps;

            aux_ps += 0.5*ps;

            it += 1;
        }
        while ((it < n_max_it) && (std::abs(dps) > aux_ps*1.0e-8));

        // EVAL VELOCITY
        us = 0.5*(u1L+u1R)+0.5*(this->_aux_fK_(g, ps, rhoR, pR)-this->_aux_fK_(g, ps, rhoL, pL));
        // ------------------------------------------------------------

        // FIND SOLUTION AT INTERFACE ---------------------------------
        if (us > 0.0)
        {
            // LEFT SHOCK
            if (ps > pL)
            {
                const amrex::Real rhoLs = rhoL*(ps/pL+gg)/(gg*ps/pL+1.0);
                const amrex::Real SL = u1L-aL*std::sqrt((0.5/g)*((g+1.0)*(ps/pL)+g-1.0));

                if (SL < 0.0)
                {
                    rhoI = rhoLs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = ps;
                }
                else
                {
                    rhoI = rhoL;
                    AMREX_D_TERM
                    (
                        u1I = u1L;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL;
                }
            }
            // LEFT FAN
            else
            {
                const amrex::Real rhoLs = rhoL*std::pow(ps/pL, 1.0/g);
                const amrex::Real aLs = std::sqrt(g*ps/rhoLs);

                if ((us-aLs) < 0.0)
                {
                    rhoI = rhoLs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = ps;
                }
                else if ((u1L-aL) > 0.0)
                {
                    rhoI = rhoL;
                    AMREX_D_TERM
                    (
                        u1I = u1L;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL;
                }
                else
                {
                    rhoI = rhoL*std::pow(2.0/(g+1)+(g-1.0)/(g+1.0)*u1L/aL, 2.0/(g-1.0));
                    AMREX_D_TERM
                    (
                        u1I = 2.0/(g+1.0)*(aL+0.5*(g-1.0)*u1L);,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL*std::pow(2.0/(g+1)+(g-1.0)/(g+1.0)*u1L/aL, 2.0*g/(g-1.0));
                }
            }
        }
        else if (us < 0.0)
        {   
            // RIGHT SHOCK
            if (ps > pR)
            {
                const amrex::Real rhoRs = rhoR*(ps/pR+gg)/(gg*ps/pR+1.0);
                const amrex::Real SR = u1R+aR*std::sqrt((0.5/g)*((g+1.0)*(ps/pR)+g-1.0));

                if (SR > 0.0)
                {
                    rhoI = rhoRs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = ps;
                }
                else
                {
                    rhoI = rhoR;
                    AMREX_D_TERM
                    (
                        u1I = u1R;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR;
                }
            }
            // RIGHT FAN
            else
            {
                const amrex::Real rhoRs = rhoR*std::pow(ps/pR, 1.0/g);
                const amrex::Real aRs = std::sqrt(g*ps/rhoRs);

                if ((us+aRs) > 0.0)
                {
                    rhoI = rhoRs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = ps;
                }
                else if ((u1R+aR) < 0.0)
                {
                    rhoI = rhoR;
                    AMREX_D_TERM
                    (
                        u1I = u1R;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR;
                }
                else
                {
                    rhoI = rhoR*std::pow(2.0/(g+1)-(g-1.0)/(g+1.0)*u1R/aR, 2.0/(g-1.0));
                    AMREX_D_TERM
                    (
                        u1I = 2.0/(g+1.0)*(-aR+0.5*(g-1.0)*u1R);,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR*std::pow(2.0/(g+1)-(g-1.0)/(g+1.0)*u1R/aR, 2.0*g/(g-1.0));
                }
            }
        }
        else
        {
            amrex::Real rhoLs, rhoRs;

            // LEFT SHOCK
            if (ps > pL)
            {
                rhoLs = rhoL*(ps/pL+gg)/(gg*ps/pL+1.0);
            }
            // LEFT FAN
            else
            {
                rhoLs = rhoL*std::pow(ps/pL, 1.0/g);
            }
            
            // RIGHT SHOCK
            if (ps > pR)
            {
                rhoRs = rhoR*(ps/pR+gg)/(gg*ps/pR+1.0);
            }
            // RIGHT FAN
            else
            {
                rhoRs = rhoR*std::pow(ps/pR, 1.0/g);
            }

            //rhoI = 0.5*(rhoLs+rhoRs);
            rhoI = 2.0*rhoLs*rhoRs/(rhoLs+rhoRs);
            AMREX_D_TERM
            (
                u1I = us;,
                u2I = 0.5*(u2L+u2R);,
                u3I = 0.5*(u3L+u3R);
            )
            pI = ps;
        }

        URP[RHO] = rhoI;
        AMREX_D_TERM
        (
            URP[RHOV1] = rhoI*u1I;,
            URP[RHOV2] = rhoI*u2I;,
            URP[RHOV3] = rhoI*u3I;
        )
        URP[TE] = pI/(g-1.0)+0.5*rhoI*(AMREX_D_TERM(u1I*u1I,+u2I*u2I,+u3I*u3I));
        // ------------------------------------------------------------
    }

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
    // ################################################################

    // CONSERVED VARIABLES VS CHARACTERISTICS VARIABLES ###############
    /**
     * \brief Eval jumps in the characteristics along x1 from jumps in the conserved variables along x1.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DU: pointer to memory containing jumps in the conserved variables.
     * \param[out] DC: pointer to memory that will contain jumps in the characteristics.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_1(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                   const amrex::Real * U,
                   const amrex::Real * DU, amrex::Real * DC) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;
        const amrex::Real g1 = g-1.0;

        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h;
        this->eval_enthalpy_variables(U, AMREX_D_DECL(u1, u2, u3), p, a, h);

#if (AMREX_SPACEDIM == 1)
        DC[RHO]   =     (h+a*(u1-a)/g1)*DU[RHO]+(-a/g1-u1)*DU[RHOV1]    +DU[TE];
        DC[RHOV1] = (-2.0*h+4.0*a*a/g1)*DU[RHO]    +2.0*u1*DU[RHOV1]-2.0*DU[TE];
        DC[TE]    =     (h-a*(u1+a)/g1)*DU[RHO]+(+a/g1-u1)*DU[RHOV1]    +DU[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DC[RHO]   =     (h+a*(u1-a)/g1)*DU[RHO]+(-a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]    +DU[TE];
        DC[RHOV1] = (-2.0*h+4.0*a*a/g1)*DU[RHO]    +2.0*u1*DU[RHOV1]      +2.0*u2*DU[RHOV2]-2.0*DU[TE];
        DC[RHOV2] =    (-2.0*a*a*u2/g1)*DU[RHO]                     +(2.0*a*a/g1)*DU[RHOV2];
        DC[TE]    =     (h-a*(u1+a)/g1)*DU[RHO]+(+a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]    +DU[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DC[RHO]   =     (h+a*(u1-a)/g1)*DU[RHO]+(-a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]          -u3*DU[RHOV3]    +DU[TE];
        DC[RHOV1] = (-2.0*h+4.0*a*a/g1)*DU[RHO]    +2.0*u1*DU[RHOV1]      +2.0*u2*DU[RHOV2]      +2.0*u3*DU[RHOV3]-2.0*DU[TE];
        DC[RHOV2] =    (-2.0*a*a*u2/g1)*DU[RHO]                     +(2.0*a*a/g1)*DU[RHOV2];
        DC[RHOV3] =    (-2.0*a*a*u3/g1)*DU[RHO]                                            +(2.0*a*a/g1)*DU[RHOV3];
        DC[TE]    =     (h-a*(u1+a)/g1)*DU[RHO]+(+a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]          -u3*DU[RHOV3]    +DU[TE];
#endif
        const amrex::Real delta = 0.5*g1/(a*a);
        DC[RHO] *= delta;
        AMREX_D_TERM
        (
            DC[RHOV1] *= delta;,
            DC[RHOV2] *= delta;,
            DC[RHOV3] *= delta;
        )
        DC[TE] *= delta;
    }
    
    /**
     * \brief Eval jumps in the characteristics from jumps in the conserved variables.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DU: pointer to memory containing jumps in the conserved variables.
     * \param[out] DC: pointer to memory that will contain jumps in the characteristics.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                 const amrex::Real * U,
                 const amrex::Real * DU, amrex::Real * DC) const
    {
        // VARIABLES
        amrex::Real Un[FV_N_SOL], DUn[FV_N_SOL];
        
        this->eval_U2Un(un, U, Un);
        this->eval_U2Un(un, DU, DUn);
        
        this->F_DU2DC_1(t, x, Un, DUn, DC);
    }

    /**
     * \brief Eval jumps in the conserved variables along x1 from jumps in the characteristics along x1.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DC: pointer to memory containing jumps in the characteristics.
     * \param[out] DU: pointer to memory that will contain jumps in the conserved variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_1(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                   const amrex::Real * U,
                   const amrex::Real * DC, amrex::Real * DU) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h;
        this->eval_enthalpy_variables(U, AMREX_D_DECL(u1, u2, u3), p, a, h);

#if (AMREX_SPACEDIM == 1)
        DU[RHO]   =          DC[RHO]          +DC[RHOV1]         +DC[TE];
        DU[RHOV1] =  (-a+u1)*DC[RHO]       +u1*DC[RHOV1]  +(a+u1)*DC[TE];
        DU[TE]    = (h-a*u1)*DC[RHO]+0.5*u1*u1*DC[RHOV1]+(h+a*u1)*DC[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DU[RHO]   =          DC[RHO]                  +DC[RHOV1]                      +DC[TE];
        DU[RHOV1] =  (-a+u1)*DC[RHO]               +u1*DC[RHOV1]               +(a+u1)*DC[TE];
        DU[RHOV2] =       u2*DC[RHO]               +u2*DC[RHOV1]   +DC[RHOV2]      +u2*DC[TE];
        DU[TE]    = (h-a*u1)*DC[RHO]+0.5*(u1*u1+u2*u2)*DC[RHOV1]+u2*DC[RHOV2]+(h+a*u1)*DC[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DU[RHO]   =          DC[RHO]                        +DC[RHOV1]                                   +DC[TE];
        DU[RHOV1] =  (-a+u1)*DC[RHO]                     +u1*DC[RHOV1]                            +(a+u1)*DC[TE];
        DU[RHOV2] =       u2*DC[RHO]                     +u2*DC[RHOV1]   +DC[RHOV2]                   +u2*DC[TE];
        DU[RHOV3] =       u3*DC[RHO]                     +u3*DC[RHOV1]                +DC[RHOV3]      +u3*DC[TE];
        DU[TE]    = (h-a*u1)*DC[RHO]+0.5*(u1*u1+u2*u2+u3*u3)*DC[RHOV1]+u2*DC[RHOV2]+u3*DC[RHOV3]+(h+a*u1)*DC[TE];
#endif
    }

    /**
     * \brief Eval the jumps in the conserved variables from jumps in the characteristics.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal along which the jump is defined.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DC: pointer to memory containing jumps in the characteristics.
     * \param[out] DU: pointer to memory that will contain jumps in the conserved variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                 const amrex::Real * U,
                 const amrex::Real * DC, amrex::Real * DU) const
    {
        // VARIABLES
        amrex::Real Un[FV_N_SOL], DUn[FV_N_SOL];

        this->eval_U2Un(un, U, Un);
        this->F_DC2DU_1(t, x, Un, DC, DUn);
        this->eval_Un2U(un, DUn, DU);
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

        const amrex::Real rho = P[RHO];
        const amrex::Real p = P[TE];
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
        amrex::Real Pn[FV_N_SOL], DPn[FV_N_SOL];
        
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

        const amrex::Real rho = P[RHO];
        const amrex::Real p = P[TE];
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
        amrex::Real Pn[FV_N_SOL], DPn[FV_N_SOL];

        this->eval_U2Un(un, P, Pn);
        this->F_DC2DP_1(t, x, Pn, DC, DPn);
        this->eval_Un2U(un, DPn, DP);
    }
    // ################################################################

    // CONSERVED VARIABLES VS PRIMARY VARIABLES #######################
    /**
     * \brief Eval primary variables from conserved variables.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] P: pointer to memory that, on exit, will contain the primary variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_U2P(const amrex::Real /*t*/, const amrex::Real * /*x*/, 
               const amrex::Real * U, amrex::Real * P) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        P[RHO] = U[RHO];
        AMREX_D_TERM
        (
            P[RHOV1] = u1;,
            P[RHOV2] = u2;,
            P[RHOV3] = u3;
        )
        P[TE] = p;
    }

    /**
     * \brief Eval conserved variables from primary variables.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[out] U: pointer to memory that, on exit, will contain the conserved variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_P2U(const amrex::Real /*t*/, const amrex::Real * /*x*/, 
               const amrex::Real * P, amrex::Real * U) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real ig1 = 1.0/(g-1.0);
        const amrex::Real rho = std::max(P[RHO], SMALL_rho);
        AMREX_D_TERM
        (
            const amrex::Real u1 = P[RHOV1];,
            const amrex::Real u2 = P[RHOV2];,
            const amrex::Real u3 = P[RHOV3];
        )
        const amrex::Real p = std::max(P[TE], SMALL_p);

        U[RHO] = rho;
        AMREX_D_TERM
        (
            U[RHOV1] = rho*u1;,
            U[RHOV2] = rho*u2;,
            U[RHOV3] = rho*u3;
        )
        U[TE] = ig1*p+0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
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
     * \brief Eval jumps in the primary variables from jumps in the conserved variables.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the unknown solution fields.
     * \param[in] DU: pointer to memory containing jumps in the conserved variables.
     * \param[out] DP: pointer to memory that will contain jumps in the primary variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_DU2DP(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                 const amrex::Real * U,
                 const amrex::Real * DU, amrex::Real * DP) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;
        const amrex::Real g1 = (g-1.0);

        // VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);
        const amrex::Real inv_rho = 1.0/rho;

#if (AMREX_SPACEDIM == 1)
        DP[RHO]   =              DU[RHO];
        DP[RHOV1] =  -u1*inv_rho*DU[RHO]+inv_rho*DU[RHOV1];
        DP[TE]    = 0.5*g1*u1*u1*DU[RHO]  -g1*u1*DU[RHOV1]+g1*DU[TE];
#endif
#if (AMREX_SPACEDIM == 2)
        DP[RHO]   =                      DU[RHO];
        DP[RHOV1] =          -u1*inv_rho*DU[RHO]+inv_rho*DU[RHOV1];
        DP[RHOV2] =          -u2*inv_rho*DU[RHO]                  +inv_rho*DU[RHOV2];
        DP[TE]    = 0.5*g1*(u1*u1+u2*u2)*DU[RHO]  -g1*u1*DU[RHOV1]  -g1*u2*DU[RHOV2]+g1*DU[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DP[RHO]   =                            DU[RHO];
        DP[RHOV1] =                -u1*inv_rho*DU[RHO]+inv_rho*DU[RHOV1];
        DP[RHOV2] =                -u2*inv_rho*DU[RHO]                  +inv_rho*DU[RHOV2];
        DP[RHOV3] =                -u3*inv_rho*DU[RHO]                                    +inv_rho*DU[RHOV3];
        DP[TE]    = 0.5*g1*(u1*u1+u2*u2+u3*u3)*DU[RHO]  -g1*u1*DU[RHOV1]  -g1*u2*DU[RHOV2]  -g1*u3*DU[RHOV3]+g1*DU[TE];
#endif
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
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real /*t*/, const amrex::Real * /*x*/,
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

    // NUMERICAL FLUXES ###############################################
    /**
     * \brief Eval the numerical flux at the interface between two adjacent states.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] nbr_U: pointer to memory containing the neighbor conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Un[FV_N_SOL], nbr_Un[FV_N_SOL];
        amrex::Real U_RP[FV_N_SOL], tmp[FV_N_SOL];
        
        // EXPRESS THE STATE IN THE UNIT NORMAL REFERENCE SYSTEM
        this->eval_U2Un(un, U, Un);
        this->eval_U2Un(un, nbr_U, nbr_Un);

        // CALL THE RIEMANN SOLVER AND THE EVALUATE THE FLUX
        this->RiemannSolver(Un, nbr_Un, U_RP);
        this->eval_F1(U_RP, tmp);
        this->eval_Un2U(un, tmp, NFn);
    }

    /**
     * \brief Eval the numerical flux at the interface between two adjacent states.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] P: pointer to memory containing the primary variables.
     * \param[in] nbr_P: pointer to memory containing the neighbor primary variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS_P(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                    const amrex::Real * P, const amrex::Real * nbr_P,
                    amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Pn[FV_N_SOL], nbr_Pn[FV_N_SOL];
        amrex::Real U_RP[FV_N_SOL], tmp[FV_N_SOL];
        
        // EXPRESS THE STATE IN THE UNIT NORMAL REFERENCE SYSTEM
        this->eval_U2Un(un, P, Pn);
        this->eval_U2Un(un, nbr_P, nbr_Pn);

        // CALL THE RIEMANN SOLVER AND THE EVALUATE THE FLUX
        this->RiemannSolver_P(std::max(    Pn[RHO], SMALL_rho), AMREX_D_DECL(    Pn[RHOV1],     Pn[RHOV2],     Pn[RHOV3]), std::max(    Pn[TE], SMALL_p),
                              std::max(nbr_Pn[RHO], SMALL_rho), AMREX_D_DECL(nbr_Pn[RHOV1], nbr_Pn[RHOV2], nbr_Pn[RHOV3]), std::max(nbr_Pn[TE], SMALL_p),
                              U_RP);
        this->eval_F1(U_RP, tmp);
        this->eval_Un2U(un, tmp, NFn);
    }
    // ################################################################
};