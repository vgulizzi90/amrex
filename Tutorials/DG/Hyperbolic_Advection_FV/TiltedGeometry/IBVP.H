//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SodsTube.H
 * \brief Contains constants and data structures for the linear advection problem.
*/

#include "../IBVP_Base.H"

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct ADVECTION
:
public ADVECTION_BASE
{
    // DATA MEMBERS ###################################################
    amrex::Real params[3];
    // ################################################################
    // NOTE: params must store:
    //       
    //       if (this->eb_flag == 0)
    //       - params[0-2]: not used;
    //
    //       if (this->eb_flag == 1)
    //       - params[0]: diameter of the embedded Sod's tube;
    //       - params[1]: inclination (or azimuth in 3D) of the
    //                    embedded Sod's tube;
    //       - params[2]: elevation of the embedded Sod's tube (used in
    //                    3D only);
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    ADVECTION(const amrex::Vector<int> & int_params_, const amrex::Vector<amrex::Real> & params_)
    :
    ADVECTION_BASE(int_params_)
    {
        std::fill(this->params, this->params+3, 0.0);
        std::copy(params_.begin(), params_.end(), this->params);
    }
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
            const amrex::Real diam = this->params[0];

#if (AMREX_SPACEDIM == 2)
            const amrex::Real theta = (this->params[1])*M_PI/180.0;
            const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
            const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
            const amrex::Real y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
            const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]);
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real theta = (this->params[1])*M_PI/180.0;
            const amrex::Real phi = (this->params[2])*M_PI/180.0;
            const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
            const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
            const amrex::Real y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
            const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]);
#endif
            const amrex::Real res = d2-0.25*diam*diam;
        
            PHI[0] = res;
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
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real t, const amrex::Real * x, amrex::Real * U) const
    {
        amrex::Real V[AMREX_SPACEDIM];
        this->F_V(t, x, V);
        const amrex::Real xVt[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-V[0]*t, x[1]-V[1]*t, x[2]-V[2]*t)};

        if (this->eb_flag == 0)
        {
            if (xVt[0] > 0.5)
            {
                U[RHO] = 0.0;
            }
            else
            {
                U[RHO] = 1.0;
            }
        }
        else if (this->eb_flag == 1)
        {
            //const amrex::Real theta = (this->params[1])*M_PI/180.0;
            const amrex::Real theta = 0.0;
            const amrex::Real cth = std::cos(theta);
            const amrex::Real sth = std::sin(theta);
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = (this->params[2])*M_PI/180.0;
            const amrex::Real cph = std::cos(phi);
            const amrex::Real sph = std::sin(phi);
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
            const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(xVt[0]-0.5, xVt[1]-0.5, xVt[2]-0.5)};
            const amrex::Real xi = AMREX_D_TERM(xi_dir[0]*y[0],+xi_dir[1]*y[1],+xi_dir[2]*y[2]);

            if (xi > 0.0)
            {
                U[RHO] = 0.0;
            }
            else
            {
                U[RHO] = 1.0;
            }
        }
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
        this->F_EXACT(0.0, x, U);
    }
    // ################################################################

    // VELOCITY FIELD #################################################
    /**
     * \brief Fill V with the velocity field at (t,x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] V: pointer to memory that will be filled with the value of the velocity components.
    */
    AMREX_GPU_HOST_DEVICE
    void F_V(const amrex::Real /*t*/, const amrex::Real * /*x*/, amrex::Real * V) const override
    {
        if (this->eb_flag == 0)
        {
            AMREX_D_TERM
            (
                V[0] = 1.0;,
                V[1] = 1.0;,
                V[2] = 1.0;
            )
        }
        else if (this->eb_flag == 1)
        {
            const amrex::Real theta = (this->params[1])*M_PI/180.0;
            const amrex::Real cth = std::cos(theta);
            const amrex::Real sth = std::sin(theta);
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = (this->params[2])*M_PI/180.0;
            const amrex::Real cph = std::cos(phi);
            const amrex::Real sph = std::sin(phi);
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif

            AMREX_D_TERM
            (
                V[0] = xi_dir[0];,
                V[1] = xi_dir[1];,
                V[2] = xi_dir[2];
            )
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
               const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        amrex::Real V[AMREX_SPACEDIM], V_dot_un;
        this->F_V(t, x, V);
        
        V_dot_un = AMREX_D_TERM(V[0]*un[0],+V[1]*un[1],+V[2]*un[2]);

        if (V_dot_un > 0.0)
        {
            bcs_U[RHO] = U[RHO];
        }
        else
        {
            this->F_EXACT(t, x, bcs_U);
        }
    }

    /**
     * \brief Fill bcs_U memory to enforce boundary conditions for the level set function.
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
    void F_PHI_BCS(const int /*dom*/,
                   const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                   const amrex::Real * U,
                   amrex::Real * bcs_U) const
    {
        amrex::Real V[AMREX_SPACEDIM], V_dot_un;
        this->F_V(t, x, V);
        
        V_dot_un = AMREX_D_TERM(V[0]*un[0],+V[1]*un[1],+V[2]*un[2]);

        if (V_dot_un > 0.0)
        {
            bcs_U[RHO] = U[RHO];
        }
        else
        {
            this->F_EXACT(t, x, bcs_U);
        }
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    // NOTE: The function void F_NF_ICS is found in "../IBVP_Base.H".
    // ################################################################
    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the advected field.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int dom,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        amrex::Real bcs_U[DG_N_SOL];
        this->F_BCS(dom, t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the advected field.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        amrex::Real bcs_U[DG_N_SOL];
        this->F_PHI_BCS(dom, t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
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
        fields_names = {"density",
                        AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z")};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0, AMREX_D_DECL(0, 0, 0)};
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
    void F_FIELDS(const amrex::Real t, const amrex::Real * x,
                  const amrex::Real * U,
                  amrex::Real * F) const
    {
        amrex::Real V[AMREX_SPACEDIM];
        this->F_V(t, x, V);

        F[RHO] = U[RHO];
        AMREX_D_TERM
        (
            F[RHO+1] = V[0];,
            F[RHO+2] = V[1];,
            F[RHO+3] = V[2];
        )
    }
    // ################################################################
};