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
    amrex::Real params[9];
    // ################################################################
    // NOTE: params must store:
    //       - params[0]: rhoL, i.e. density at the left of the shock;
    //       - params[1]: uL, i.e. velocity at the left of the shock;
    //       - params[2]: pL, i.e. pressure at the left of the shock;
    //       - params[3]: shock location;
    //       - params[4]: rhoR, i.e. density at the right of the shock;
    //       - params[5]: uR, i.e. velocity at the right of the shock;
    //       - params[6]: pR, i.e. pressure at the right of the shock;
    //       - params[7]: x coordinate of the point P(Px,0) on the
    //                    plane;
    //       - params[8]: inclination of the plane;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS(const amrex::Vector<int> & int_params_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(int_params_, params_)
    {
        std::fill(this->params, this->params+9, 0.0);
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
        else
        {
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
            const amrex::Real eta[2] = {-std::sin(theta), +std::cos(theta)};
            const amrex::Real P[2] = {this->params[7], 0.0};
            const amrex::Real s = (x[0]-P[0])*eta[0]+(x[1]-P[1])*eta[1];

            PHI[0] = -s;
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
        // STATES
        const amrex::Real g = this->gamma;
        const amrex::Real rhoL = this->params[0];
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real Sx = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        if (x[0] > Sx)
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
               const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        if (un[1] < -0.5)
        {
            bcs_U[RHO] = U[RHO];
            AMREX_D_TERM
            (
                bcs_U[RHOV1] = U[RHOV1];,
                bcs_U[RHOV2] = -U[RHOV2];,
                bcs_U[RHOV3] = U[RHOV3];
            )
            bcs_U[TE] = U[TE];
        }
        else
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