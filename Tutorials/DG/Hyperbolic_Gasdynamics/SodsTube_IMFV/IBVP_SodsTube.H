//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SodsTube.H
 * \brief Contains constants and data structures for the Sod's tube problem with ideal gases.
*/

#include "../IBVP_Base.H"

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct IDEAL_GAS
:
public IDEAL_GAS_BASE
{
    // DATA MEMBERS ###################################################
    amrex::Real params[10];
    // ################################################################
    // NOTE: params must store:
    //       - params[0]: rhoL, i.e. density at the left of the shock;
    //       - params[1]: uL, i.e. velocity at the left of the shock;
    //       - params[2]: pL, i.e. pressure at the left of the shock;
    //       - params[3]: shock location;
    //       - params[4]: rhoR, i.e. density at the right of the shock;
    //       - params[5]: uR, i.e. velocity at the right of the shock;
    //       - params[6]: pR, i.e. pressure at the right of the shock;
    //       - params[7]: diameter of the tube;
    //       - params[8]: inclination of the tube;
    //       - params[9]: elevation of the tube;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
     * 
     * \param[in] gamma_: ratio of specific heats.
    */
    IDEAL_GAS(const amrex::Real gamma_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(gamma_)
    {
        std::copy(params_.begin(), params_.end(), this->params);
    }
    // ################################################################

    // AUXILIARY FUNCTIONS ############################################
    // ################################################################

    // LEVEL SET FUNCTIONS ############################################
    /**
     * \brief Fill PHI with values of the level sets at (t,x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] PHI: pointer to memory that will be filled with the value of the level sets at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_PHI(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * PHI) const
    {
        const amrex::Real diam = this->params[7];

#if (AMREX_SPACEDIM == 2)
        const amrex::Real theta = (this->params[8])*M_PI/180.0;
        const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
        const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
        const amrex::Real y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
        const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real theta = (this->params[8])*M_PI/180.0;
        const amrex::Real phi = (this->params[9])*M_PI/180.0;
        const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
        const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
        const amrex::Real y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
        const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]);
#endif
        const amrex::Real res = d2-0.25*diam*diam;
        
        PHI[0] = res;
    }

    /**
     * \brief Fill phi_info with information about which domain is associated to which levelset.
     * 
     * \param[in] dom: domain index.
     * \param[out] phi_info: pointer to memory that will be filled domain info. More specifically:
     *
     *             phi_info[0] = ph means that the dom-th domain is defined by the ph-th levelset;
     *             phi_info[1] = -1 means that the dom-th domain is defined where the ph-th levelset is negative;
     *             phi_info[1] = +1 means that the dom-th domain is defined where the ph-th levelset is positive;
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_DOM2PHI(const int /*dom*/, int * phi_info) const
    {
        phi_info[0] = 0;
        phi_info[1] = -1;
    }

    /**
     * \brief Returns the domain index dom where the unknown solution field ru is defined.
     * 
     * \param[in] ru: index of the unknown solution field.
     *
     * \return the index of the domain where the unknown solution field ru is defined.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int F_SOL2DOM(const int /*ru*/) const
    {
        return 0;
    }
    // ################################################################
    
    // INITIAL CONDITIONS #############################################
    /**
     * \brief Fill SOL with initial conditions of the unknown solution fields.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] SOL: pointer to memory that will be filled with the value of the unknown solution
     *                  fields at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * x, amrex::Real * SOL) const
    {
        // SPACE LOCATION
        const amrex::Real theta = (this->params[8])*M_PI/180.0;
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
#if (AMREX_SPACEDIM == 3)
        const amrex::Real phi = (this->params[9])*M_PI/180.0;
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
        const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-0.5, x[1]-0.5, x[2]-0.5)};

        const amrex::Real xi = AMREX_D_TERM(xi_dir[0]*y[0],+xi_dir[1]*y[1],+xi_dir[2]*y[2]);

        // STATES
        const amrex::Real g = this->gamma;
        const amrex::Real rhoL = this->params[0];
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xis = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        amrex::Real SOLxi[N_SOL];

        if (xi > xis)
        {
            SOLxi[RHO] = rhoR;
            AMREX_D_TERM
            (
                SOLxi[RHOV1] = rhoR*uR;,
                SOLxi[RHOV2] = 0.0;,
                SOLxi[RHOV3] = 0.0;
            )
            SOLxi[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
        }
        else
        {
            SOLxi[RHO] = rhoL;
            AMREX_D_TERM
            (
                SOLxi[RHOV1] = rhoL*uL;,
                SOLxi[RHOV2] = 0.0;,
                SOLxi[RHOV3] = 0.0;
            )
            SOLxi[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
        }

        this->eval_Un2U(xi_dir, SOLxi, SOL);
    }
    // ################################################################

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_SOL memory to enforce boundary conditions.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] bcs_SOL: pointer to memory that will be filled with values suitably chosen to
     *                      enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
               const amrex::Real * SOL,
               amrex::Real * bcs_SOL) const
    {
        bcs_SOL[RHO] = SOL[RHO];
        AMREX_D_TERM
        (
            bcs_SOL[RHOV1] = SOL[RHOV1];,
            bcs_SOL[RHOV2] = SOL[RHOV2];,
            bcs_SOL[RHOV3] = SOL[RHOV3];
        )
        bcs_SOL[TE] = SOL[TE];
    }
    
    /**
     * \brief Fill bcs_SOL memory to enforce boundary conditions on boundaries of the levelsets.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] bcs_SOL: pointer to memory that will be filled with values suitably chosen to
     *                      enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                   const amrex::Real * SOL,
                   amrex::Real * bcs_SOL) const
    {
        amrex::Real SOLn[N_SOL];

        this->eval_U2Un(un, SOL, SOLn);
        SOLn[RHOV1] = -SOLn[RHOV1];
        this->eval_Un2U(un, SOLn, bcs_SOL);
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    // NOTE: The function void F_NF_ICS is found in "../IBVP_Base.H".
    // ################################################################
    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] SOL: pointer to memory containing the unknown solution fields.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * SOL,
                  amrex::Real * NFn) const
    {
        amrex::Real bcs_SOL[N_SOL];
        this->F_BCS(t, x, un, SOL, bcs_SOL);
        this->F_NF_ICS(t, x, un, SOL, bcs_SOL, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] SOL: pointer to memory containing the unknown solution fields.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * SOL,
                      amrex::Real * NFn) const
    {
        amrex::Real bcs_SOL[N_SOL];
        this->F_PHI_BCS(t, x, un, SOL, bcs_SOL);
        this->F_NF_ICS(t, x, un, SOL, bcs_SOL, NFn);
    }
    // ################################################################

    // OUTPUT FIELDS ##################################################
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void F_FIELDS_NAMES(amrex::Vector<std::string> & fields_names) const
    {
        fields_names = {"rho", AMREX_D_DECL("u_x", "u_y", "u_z"), "p", "mach", "e", AMREX_D_DECL("u_xi", "u_eta", "u_zeta")};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, AMREX_D_DECL(0, 0, 0)};
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_FIELDS(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                  const amrex::Real * SOL,
                  amrex::Real * F) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real theta = (this->params[8])*M_PI/180.0;
#if (AMREX_SPACEDIM == 3)
        const amrex::Real phi = (this->params[9])*M_PI/180.0;
#endif

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(SOL, rho, AMREX_D_DECL(u1, u2, u3), p);

        // SOUND SPEED AND MACH NUMBER
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real M = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))/a;

        // VELOCITY COMPONENTS ALONG THE TUBE'S CENTERLINE
#if (AMREX_SPACEDIM == 2)
        const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
#endif
        amrex::Real SOLn[N_SOL];
        this->eval_U2Un(xi, SOL, SOLn);

        // OUTPUT FIELDS
        F[RHO] = rho;
        AMREX_D_TERM
        (
            F[RHOV1] = u1;,
            F[RHOV2] = u2;,
            F[RHOV3] = u3;
        )
        F[TE] = p;
        F[TE+1] = M;
        F[TE+2] = SOL[TE];
        AMREX_D_TERM
        (
            F[TE+3] = SOLn[RHOV1]/rho;,
            F[TE+4] = SOLn[RHOV2]/rho;,
            F[TE+5] = SOLn[RHOV3]/rho;
        )
    }
    // ################################################################
};