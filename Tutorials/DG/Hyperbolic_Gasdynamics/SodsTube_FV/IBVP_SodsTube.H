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
    amrex::Real params[7];
    int ori;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
     * 
     * \param[in] gamma_: ratio of specific heats.
    */
    IDEAL_GAS(const amrex::Real gamma_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(gamma_),
    ori{-1}
    {
        std::copy(params_.begin(), params_.end(), this->params);
    }
    // ################################################################

    // AUXILIARY METHODS ##############################################
    /**
     * \brief Sets the orientation of the Sod's tube.
     * 
     * \param[in] ori_: input orientation.
    */
    void set_ori(const int ori_)
    {
        this->ori = ori_;
    }
    // ################################################################

    // INITIAL CONDITIONS #############################################
    /**
     * \brief Fill U with initial conditions of the conserved variables.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * x, amrex::Real * U) const
    {
        const amrex::Real g = this->gamma;

        const amrex::Real rhoL = this->params[0];
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xs = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        const amrex::Real y = x[this->ori];
        
        int AMREX_D_DECL(RHOVn, RHOVt1, RHOVt2);
        if (this->ori == 0) {AMREX_D_TERM(RHOVn = 1;, RHOVt1 = 2;,RHOVt2 = 3;)}
        else if (this->ori == 1) {AMREX_D_TERM(RHOVn = 2;, RHOVt1 = 1;, RHOVt2 = 3;)}
        else if (this->ori == 2) {AMREX_D_TERM(RHOVn = 3;, RHOVt1 = 1;, RHOVt2 = 2;)}

        if (y > xs)
        {
            U[RHO] = rhoR;
            AMREX_D_TERM
            (
                U[RHOVn] = rhoR*uR;,
                U[RHOVt1] = 0.0;,
                U[RHOVt2] = 0.0;
            )
            U[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
        }
        else
        {
            U[RHO] = rhoL;
            AMREX_D_TERM
            (
                U[RHOVn] = rhoL*uL;,
                U[RHOVt1] = 0.0;,
                U[RHOVt2] = 0.0;
            )
            U[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
        }
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
     *                      enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
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
                        AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                        "pressure",
                        "mach",
                        "total_energy",
                        "internal_energy"};
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] U: pointer to memory that stores the conserved variables fields at (t,x).
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
    }
    // ################################################################
};