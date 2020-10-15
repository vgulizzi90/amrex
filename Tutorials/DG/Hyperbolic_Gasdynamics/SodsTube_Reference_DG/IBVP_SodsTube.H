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

    // AUXILIARY METHODS ##############################################
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
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xs = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        const amrex::Real y = x[0];
        
        const int AMREX_D_DECL(RHOVn = 1, RHOVt1 = 2, RHOVt2 = 3);

        if (y > xs)
        {
            SOL[RHO] = rhoR;
            AMREX_D_TERM
            (
                SOL[RHOVn] = rhoR*uR;,
                SOL[RHOVt1] = 0.0;,
                SOL[RHOVt2] = 0.0;
            )
            SOL[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
        }
        else
        {
            SOL[RHO] = rhoL;
            AMREX_D_TERM
            (
                SOL[RHOVn] = rhoL*uL;,
                SOL[RHOVt1] = 0.0;,
                SOL[RHOVt2] = 0.0;
            )
            SOL[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
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
        // OUTFLOW
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
        // SYMMETRY
#if (AMREX_SPACEDIM > 1)
        else if (std::abs(un[1]) > 0.5)
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
#endif
#if (AMREX_SPACEDIM > 2)
        else if (std::abs(un[2]) > 0.5)
        {
            bcs_SOL[RHO] = SOL[RHO];
            AMREX_D_TERM
            (
                bcs_SOL[RHOV1] = SOL[RHOV1];,
                bcs_SOL[RHOV2] = SOL[RHOV2];,
                bcs_SOL[RHOV3] = -SOL[RHOV3];
            )
            bcs_SOL[TE] = SOL[TE];
        }
#endif
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