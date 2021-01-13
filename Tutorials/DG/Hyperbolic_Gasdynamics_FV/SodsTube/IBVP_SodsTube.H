//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SodsTube.H
 * \brief Contains constants and data structures for the double Mach reflection problem.
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
    amrex::Real params[11];
    // ################################################################
    // NOTE: params must store:
    //       - params[0]: rhoL, i.e. density at the left of the shock;
    //       - params[1]: uL, i.e. velocity at the left of the shock;
    //       - params[2]: pL, i.e. pressure at the left of the shock;
    //       - params[3]: shock location;
    //       - params[4]: rhoR, i.e. density at the right of the shock;
    //       - params[5]: uR, i.e. velocity at the right of the shock;
    //       - params[6]: pR, i.e. pressure at the right of the shock;
    //
    //       if (this->eb_flag == 0)
    //       - params[7-10]: not used;
    //
    //       if (this->eb_flag == 1)
    //       - params[7]: diameter of the embedded Sod's tube;
    //       - params[8]: inclination (or azimuth in 3D) of the
    //                    embedded Sod's tube;
    //       - params[9]: elevation of the embedded Sod's tube (used in
    //                    3D only);
    //       - params[10]: not used;
    //
    //       if (this->eb_flag == 2)
    //       - params[7]: radius of the embedded circle (in 2D) or
    //                    sphere (in 3D);
    //       - params[8-9(10)]: (x1,x2(,x3)) coordinates of the center
    //                          of the circle/sphere.
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS(const amrex::Vector<int> & int_params_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(int_params_, params_)
    {
        std::fill(this->params, this->params+11, 0.0);
        std::copy(params_.begin()+1, params_.end(), this->params);
    }
    // ################################################################
    // NOTE: params_[0] must contain the specific heats ratio and is
    //       stored when the constructor IDEAL_GAS_BASE is called.
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
        if (this->eb_flag == 0)
        {
            PHI[0] = -1.0;
        }
        else if (this->eb_flag == 1)
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
        else
        {
            const amrex::Real r = this->params[7];
            const amrex::Real * xc = &this->params[8];

            PHI[0] = r*r-(AMREX_D_TERM((x[0]-xc[0])*(x[0]-xc[0]),+(x[1]-xc[1])*(x[1]-xc[1]),+(x[2]-xc[2])*(x[2]-xc[2])));
        }
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

    /**
     * \brief Returns the index nbr_dom of the domain that is neighbor with the current domain.
     * 
     * \param[in] dom: index of the current domain.
     *
     * \return the index nbr_dom of the domain that is neighbor with domain dom.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int F_DOM2NBRDOM(const int /*dom*/) const
    {
        return -1;
    }
    // ################################################################

    // INITIAL CONDITIONS #############################################
    /**
     * \brief Fill U with initial conditions of the unknown solution fields.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * x, amrex::Real * U) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real rhoL = this->params[0];
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xis = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        if ((this->eb_flag == 0) || (this->eb_flag == 2))
        {
            if (x[0] > xis)
            {
                U[RHO] = rhoR;
                AMREX_D_TERM
                (
                    U[RHOV1] = rhoR*uR;,
                    U[RHOV2] = 0.0;,
                    U[RHOV3] = 0.0;
                )
                U[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
            }
            else
            {
                U[RHO] = rhoL;
                AMREX_D_TERM
                (
                    U[RHOV1] = rhoL*uL;,
                    U[RHOV2] = 0.0;,
                    U[RHOV3] = 0.0;
                )
                U[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
            }
        }
        else
        {
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

            amrex::Real Uxi[DG_N_SOL];

            if (xi > 0.0)
            {
                Uxi[RHO] = rhoR;
                AMREX_D_TERM
                (
                    Uxi[RHOV1] = rhoR*uR;,
                    Uxi[RHOV2] = 0.0;,
                    Uxi[RHOV3] = 0.0;
                )
                Uxi[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
            }
            else
            {
                Uxi[RHO] = rhoL;
                AMREX_D_TERM
                (
                    Uxi[RHOV1] = rhoL*uL;,
                    Uxi[RHOV2] = 0.0;,
                    Uxi[RHOV3] = 0.0;
                )
                Uxi[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
            }

            this->eval_Un2U(xi_dir, Uxi, U);
        }
        
    }
    // ################################################################

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_U memory to enforce boundary conditions.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const int /*dom*/,
               const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        bcs_U[RHO] = U[RHO];
        AMREX_D_TERM
        (
            bcs_U[RHOV1] = U[RHOV1];,
            bcs_U[RHOV2] = U[RHOV2];,
            bcs_U[RHOV3] = U[RHOV3];
        )
        bcs_U[TE] = U[TE];
    }
    
    /**
     * \brief Fill bcs_U memory to enforce boundary conditions on boundaries of the levelsets.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                   const amrex::Real * U,
                   amrex::Real * bcs_U) const
    {
        amrex::Real Un[DG_N_SOL];

        this->eval_U2Un(un, U, Un);
        Un[RHOV1] = -Un[RHOV1];
        this->eval_Un2U(un, Un, bcs_U);
    }

    /**
     * \brief Fill bcs_P memory to enforce boundary conditions.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables at (t,x).
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS_P(const int /*dom*/,
                 const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                 const amrex::Real * P,
                 amrex::Real * bcs_P) const
    {
        bcs_P[RHO] = P[RHO];
        AMREX_D_TERM
        (
            bcs_P[RHOV1] = P[RHOV1];,
            bcs_P[RHOV2] = P[RHOV2];,
            bcs_P[RHOV3] = P[RHOV3];
        )
        bcs_P[TE] = P[TE];
    }

    /**
     * \brief Fill bcs_P memory to enforce boundary conditions on boundaries of the levelsets.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables at (t,x).
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS_P(const int /*dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                     const amrex::Real * P,
                     amrex::Real * bcs_P) const
    {
        this->bcs_wall(un, P, bcs_P);
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    // NOTE: The function void F_NF_ICS_P is found in "../IBVP_Base.H".
    // ################################################################
    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS_P(const int dom,
                    const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                    const amrex::Real * P,
                    amrex::Real * NFn) const
    {
        amrex::Real bcs_P[DG_N_SOL];
        this->F_BCS(dom, t, x, un, P, bcs_P);
        this->F_NF_ICS_P(dom, t, x, un, P, bcs_P, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS_P(const int dom,
                        const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                        const amrex::Real * P,
                        amrex::Real * NFn) const
    {
        amrex::Real bcs_P[DG_N_SOL];
        this->F_PHI_BCS(t, x, un, P, bcs_P);
        this->F_NF_ICS_P(dom, t, x, un, P, bcs_P, NFn);
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
        if (this->eb_flag == 0)
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy"};
        }
        else if (this->eb_flag == 1)
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy",
                             AMREX_D_DECL("velocity_xi", "velocity_eta", "velocity_zeta")};
        }
        else if (this->eb_flag == 2)
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy"};
        }
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        if (this->eb_flag == 0)
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0};
        }
        else if (this->eb_flag == 1)
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0, AMREX_D_DECL(0, 0, 0)};
        }
        else if (this->eb_flag == 2)
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0};
        }
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_FIELDS(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                  const amrex::Real * U,
                  amrex::Real * F) const
    {
        const amrex::Real g = this->gamma;

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        // SOUND SPEED AND MACH NUMBER
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real M = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))/a;

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
        F[TE+2] = U[TE];
        F[TE+3] = p/(rho*(g-1.0));

        // VELOCITY COMPONENTS ALONG THE TUBE'S CENTERLINE
        if (this->eb_flag == 1)
        {
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = (this->params[9])*M_PI/180.0;
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
#endif
            amrex::Real Un[DG_N_SOL];
            this->eval_U2Un(xi, U, Un);

            AMREX_D_TERM
            (
                F[TE+4] = Un[RHOV1]/rho;,
                F[TE+5] = Un[RHOV2]/rho;,
                F[TE+6] = Un[RHOV3]/rho;
            )
        }
    }
    // ################################################################
};