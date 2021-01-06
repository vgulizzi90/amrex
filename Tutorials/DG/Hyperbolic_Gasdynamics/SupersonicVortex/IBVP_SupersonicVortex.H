//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_DoubleMachReflection.H
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
    amrex::Real params[4];
    // ################################################################
    // NOTE: params must store:
    //       - params[0]: rho_inner, i.e. density at the inner radius
    //                    of the circular annulus;
    //       - params[1]: Mach_inner, i.e. Mach number at the inner
    //                    radius of the circular annulus;
    //       - params[2-3]: r_inner, r_outer, i.e. inner and outer
    //                      radii of the circular annulus;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS(const amrex::Vector<int> & int_params_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(int_params_, params_)
    {
        std::fill(this->params, this->params+4, 0.0);
        std::copy(params_.begin()+1, params_.end(), this->params);
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
        const amrex::Real ri2 = this->params[2]*this->params[2];
        const amrex::Real ro2 = this->params[3]*this->params[3];
        const amrex::Real ra2 = 0.25*(ri2+ro2+2.0*this->params[2]*this->params[3]);

        const amrex::Real x2 = x[0]*x[0];
        const amrex::Real y2 = x[1]*x[1];
        const amrex::Real r2 = x2+y2;

        if (r2 < ra2)
        {
            PHI[0] = ri2-r2;
        }
        else
        {
            PHI[0] = r2-ro2;
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

    // EXACT SOLUTION #################################################
    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;
        const amrex::Real g2 = 0.5*(g-1.0);
        const amrex::Real ig = 1.0/(g-1.0);
        const amrex::Real rho_i = this->params[0];
        const amrex::Real M_i = this->params[1];

        const amrex::Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]);
        const amrex::Real cth = x[0]/r;
        const amrex::Real sth = x[1]/r;
        const amrex::Real ri = this->params[2];
        const amrex::Real rr2 = (ri/r)*(ri/r);

        const amrex::Real rho = rho_i*std::pow(1.0+g2*M_i*M_i*(1.0-rr2), ig);
        const amrex::Real a_i = 1.0;
        const amrex::Real uth = a_i*M_i*(ri/r);
        const amrex::Real u = -uth*sth;
        const amrex::Real v = uth*cth;
        const amrex::Real p = std::pow(rho, g)/g;

        U[RHO] = rho;
        U[RHOV1] = rho*u;
        U[RHOV2] = rho*v;
        U[TE] = ig*p+0.5*rho*(u*u+v*v);
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

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_U memory to enforce boundary conditions.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const amrex::Real t, const amrex::Real * x, const amrex::Real * /*un*/,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        this->F_EXACT(t, x, bcs_U);
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
    void F_PHI_BCS(const amrex::Real t, const amrex::Real * x, const amrex::Real * /*un*/,
                   const amrex::Real * U,
                   amrex::Real * bcs_U) const
    {
        this->F_EXACT(t, x, bcs_U);
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
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int dom,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        amrex::Real bcs_U[DG_N_SOL];
        this->F_BCS(t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        amrex::Real bcs_U[DG_N_SOL];
        this->F_PHI_BCS(t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }
    // ################################################################

    // ERROR ##########################################################
    /**
     * \brief Compute the error and a normalization value at (t,x).
     * 
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] err_x: error at (t,x).
     * \param[out] norm_x: normalization value at (t,x).
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_ERROR(const int /*dom*/,
                 const amrex::Real t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x, amrex::Real & norm_x) const
    {
        // VARIABLES
        amrex::Real Ue[DG_N_SOL];
        
        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

        norm_x = 1.0;

        err_x = std::abs(Ue[RHO]-U[RHO]);
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
                        "velocity_x", "velocity_y",
                        "pressure",
                        "mach",
                        "total_energy",
                        "internal_energy",
                        "velocity_xi", "velocity_eta"};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        const amrex::Real theta = (this->params[8])*M_PI/180.0;

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        // SOUND SPEED AND MACH NUMBER
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real M = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))/a;

        // VELOCITY COMPONENTS ALONG THE PLANE DIRECTION
        const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
        amrex::Real Un[DG_N_SOL];
        this->eval_U2Un(xi, U, Un);

        // OUTPUT FIELDS
        F[RHO] = rho;
        F[RHOV1] = u1;
        F[RHOV2] = u2;
        F[TE] = p;
        F[TE+1] = M;
        F[TE+2] = U[TE];
        F[TE+3] = p/(rho*(g-1.0));
        F[TE+4] = Un[RHOV1]/rho;
        F[TE+5] = Un[RHOV2]/rho;
    }
    // ################################################################
};