//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Base.H
 * \brief Contains base constants and data structures.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define DG_N_SOL 1

#define RHO 0
// ####################################################################

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
 *
*/
struct ADVECTION_BASE
{
    // DATA MEMBERS ###################################################
    // VARIABLES
    const int eb_flag;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    ADVECTION_BASE(const amrex::Vector<int> & int_params)
    :
    eb_flag(int_params[0])
    {}

    virtual
    ~ADVECTION_BASE()
    {};
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
    virtual
    void F_V(const amrex::Real /*t*/, const amrex::Real * /*x*/, amrex::Real * /*V*/) const = 0;
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real F_DT(const int /*dom*/,
                     const amrex::Real * dx,
                     const amrex::Real t, const amrex::Real * x,
                     const amrex::Real * /*U*/) const
    {
        // VARIABLES
        amrex::Real V[AMREX_SPACEDIM], dt;

        this->F_V(t, x, V);
        
        AMREX_D_TERM
        (
            dt = dx[0]/std::abs(V[0]);,
            dt = std::min(dt, dx[1]/std::abs(V[1]));,
            dt = std::min(dt, dx[2]/std::abs(V[2]));
        )

        return dt;
    }
    // ################################################################

    // FLUXES #########################################################
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
             const amrex::Real t, const amrex::Real * x,
             const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        // VARIABLES
        amrex::Real V[AMREX_SPACEDIM];

        this->F_V(t, x, V);

        AMREX_D_TERM
        (
            F1[0] = V[0]*U[0];,
            F2[0] = V[1]*U[0];,
            F3[0] = V[2]*U[0];
        )
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
    void bcs_wall(const amrex::Real * /*un*/,
                  const amrex::Real * U,
                  amrex::Real * bcs_U) const
    {
        bcs_U[0] = U[0];
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
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] nbr_U: pointer to memory containing the neighbor conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int /*dom*/,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real V[AMREX_SPACEDIM], V_dot_un;

        this->F_V(t, x, V);
        V_dot_un = AMREX_D_TERM(V[0]*un[0],+V[1]*un[1],+V[2]*un[2]);

        if (V_dot_un > 0.0)
        {
            NFn[0] = V_dot_un*U[0];
        }
        else
        {
            NFn[0] = V_dot_un*nbr_U[0];
        }
    }
    // ################################################################

};