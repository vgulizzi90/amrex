//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Vortex.H
 * \brief Contains constants and data structures for the vortex advection problem.
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
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    ADVECTION()
    {}
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
    void F_PHI(const amrex::Real /*t*/, const amrex::Real * /*x*/, amrex::Real * PHI) const
    {
        PHI[0] = -1.0;
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

    // EXACT SOLUTION/INITIAL CONDITIONS ##############################
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
        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(0.5, 0.75, 0.5)};
        const amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
        U[0] = std::exp(-r2);
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
    void F_V(const amrex::Real t, const amrex::Real * x, amrex::Real * V) const override
    {
        const amrex::Real ct = -std::cos(0.5*M_PI*t);
        const amrex::Real sx = std::sin(M_PI*x[0]);
        const amrex::Real sy = std::sin(M_PI*x[1]);
        const amrex::Real s2x = std::sin(2.0*M_PI*x[0]);
        const amrex::Real s2y = std::sin(2.0*M_PI*x[1]);
        AMREX_D_TERM
        (
            V[0] = -ct*sx*sx*s2y;,
            V[1] = +ct*s2x*sy*sy;,
            V[2] = 1.0;
        )
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
               const amrex::Real * /*U*/,
               amrex::Real * /*bcs_U*/) const
    {
amrex::Print() << "ADVECTION.F_BCS: We should have not ended up in here" << std::endl;
exit(-1);
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
                   const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                   const amrex::Real * /*U*/,
                   amrex::Real * /*bcs_U*/) const
    {
amrex::Print() << "ADVECTION.F_PHI_BCS: We should have not ended up in here" << std::endl;
exit(-1);
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
    void F_NF_BCS(const int /*dom*/,
                  const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                  const amrex::Real * /*U*/,
                  amrex::Real * /*NFn*/) const
    {
amrex::Print() << "ADVECTION.F_NF_BCS: We should have not ended up in here" << std::endl;
exit(-1);
    }

    /**
     * \brief Eval the numerical flux at the boundary of the level set function.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int /*dom*/,
                      const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                      const amrex::Real * /*U*/,
                      amrex::Real * /*NFn*/) const
    {
amrex::Print() << "ADVECTION.F_NF_PHI_BCS: We should have not ended up in here" << std::endl;
exit(-1);
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
                 const amrex::Real /*t*/, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x, amrex::Real & norm_x) const
    {
        // VARIABLES
        amrex::Real Ue[DG_N_SOL];

        this->F_SOL0(x, Ue);

        norm_x = Ue[0];
        err_x = std::abs(Ue[0]-U[0]);
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
        fields_names = {"phi", AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z")};
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
        // VARIABLES
        amrex::Real V[AMREX_SPACEDIM];
        this->F_V(t, x, V);

        F[0] = U[0];
        AMREX_D_TERM
        (
            F[1] = V[0];,
            F[2] = V[1];,
            F[3] = V[2];
        )
    }
    // ################################################################
};