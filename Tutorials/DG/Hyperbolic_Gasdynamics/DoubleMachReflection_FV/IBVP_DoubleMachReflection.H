//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_DoubleMachReflection.H
 * \brief Contains constants and data structures for the considered IBVP.
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
    amrex::Real params[8];
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

        // CHECK
        const amrex::Real theta = this->params[4];

        if (theta <= 0.0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: IBVP_DoubleMachReflection.H - IDEAL_GAS::IDEAL_GAS\n";
            msg += "| The orientation of the shock must be positive.\n";
            amrex::Abort(msg);
        }
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
        const amrex::Real g = this->gamma;

        const amrex::Real rhoL = this->params[0];
        //const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xs = this->params[3];
        const amrex::Real theta = this->params[4]*M_PI/180.0;
        const amrex::Real rhoR = this->params[5];
        //const amrex::Real uR = this->params[6];
        const amrex::Real pR = this->params[7];

        const amrex::Real xi = -(x[0]-xs)*std::sin(theta)+x[1]*std::cos(theta);

        if (xi > 0.0)
        {
            SOL[RHO] = rhoL;
            SOL[RHOV1] = 0.0;
            SOL[RHOV2] = 0.0;
            SOL[TE] = pL/(g-1.0);
        }
        else
        {
            SOL[RHO] = rhoR;
            SOL[RHOV1] = 0.0;
            SOL[RHOV2] = 0.0;
            SOL[TE] = pR/(g-1.0);
        }
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
        if (std::abs(un[0]) > 0.5)
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
        else if (un[1] > 0.5)
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
        else if (un[1] < -0.5)
        {
            bcs_SOL[RHO] = SOL[RHO];
            AMREX_D_TERM
            (
                bcs_SOL[RHOV1] = SOL[RHOV1];,
                bcs_SOL[RHOV2] = -SOL[RHOV2];,
                bcs_SOL[RHOV3] = SOL[RHOV3];
            )
            bcs_SOL[TE] = SOL[TE];
        }
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
        fields_names = {"rho", AMREX_D_DECL("u_x", "u_y", "u_z"), "p", "mach", "e"};
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

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(SOL, rho, AMREX_D_DECL(u1, u2, u3), p);

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
        F[TE+2] = SOL[TE];
    }
    // ################################################################
};