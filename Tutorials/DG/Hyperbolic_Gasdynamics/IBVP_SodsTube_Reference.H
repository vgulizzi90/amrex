/* GASDYNAMICS: SODS TUBE TEST */

#include <IBVP_Base.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF LEVEL SET FUNCTIONS/DOMAINS/UNKNOWN FIELDS ---------------
#define N_PHI 1
#define N_DOM 1
#define N_U (N_DOM*N_U_PER_DOM)
// --------------------------------------------------------------------

// SOME PARAMETERS ----------------------------------------------------
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
class GASDYNAMICS
:
public GASDYNAMICS_BASE<N_PHI, N_DOM, N_U>
{
public:
    // DATA MEMBERS ===================================================
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    GASDYNAMICS(const amrex::Vector<std::string> & material_type,
                const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    GASDYNAMICS_BASE(material_type, material_properties)
    {}
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
#define xs 0.5
#define pR 0.1
#define rhoR 0.125
#define vR 0.0
#define pL 1.0
#define rhoL 1.0
#define vL 0.0

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        const amrex::Real g = this->gamma[0];

        amrex::Real U0[N_U];
        if (x[0] > xs)
        {
            U0[RHO] = rhoR;
            AMREX_D_TERM
            (
                U0[RHOV1] = rhoR*vR;,
                U0[RHOV2] = 0.0;,
                U0[RHOV3] = 0.0;
            )
            U0[TE] = pR/(g-1.0)+0.5*rhoR*vR*vR;
        }
        else
        {
            U0[RHO] = rhoL;
            AMREX_D_TERM
            (
                U0[RHOV1] = rhoL*vL;,
                U0[RHOV2] = 0.0;,
                U0[RHOV3] = 0.0;
            )
            U0[TE] = pL/(g-1.0)+0.5*rhoL*vL*vL;
        }

        return U0[u];
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    // ================================================================

    // NUMERICAL FLUX =================================================
    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_(const int dom, const amrex::Real * un, const amrex::Real * U, const amrex::Real * nbr_U,
                    amrex::Real * NFn) const
    {
        this->_eval_NFn_ExactRiemann_(dom, un, U, nbr_U, NFn);
        //this->_eval_NFn_mOsher_(dom, un, U, nbr_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTRAPHASE ===================================
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        this->_eval_NFn_(dom, un, U, nbr_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: GRID BOUNDARIES ==============================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int dom,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real bcs_U[N_U];

        this->F_BCS(dom, t, x, un, U, bcs_U);

        this->_eval_NFn_(dom, un, U, bcs_U, NFn);
    }

    AMREX_GPU_HOST_DEVICE
    void F_BCS(const int dom,
               const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        // OUTFLOW BOUNDARY CONDITIONS
        if (std::abs(un[0]) > 0.5)
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
        // SYMMETRY BOUNDARY CONDITIONS
        else if (std::abs(un[1]) > 0.5)
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
#if (AMREX_SPACEDIM > 2)
        else if (std::abs(un[2]) > 0.5)
        {
            bcs_U[RHO] = U[RHO];
            bcs_U[RHOV1] = U[RHOV1];
            bcs_U[RHOV2] = U[RHOV2];
            bcs_U[RHOV3] = -U[RHOV3];
            bcs_U[TE] = U[TE];
        }
#endif
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
        fields_domain = {0, AMREX_D_DECL(0, 0, 0), 0, 0};
        fields_name = {"rho", AMREX_D_DECL("ux", "uy", "uz"), "M", "p"};
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // PARAMETERS
        const int dom = 0;
        const amrex::Real g = this->gamma[dom];

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);
    
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real M = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))/a;

        F[RHO] = rho;
        AMREX_D_TERM
        (
            F[RHOV1] = u1;,
            F[RHOV2] = u2;,
            F[RHOV3] = u3;
        )
        F[TE] = M;
        F[TE+1] = p;
    }
    // ================================================================
};
// ####################################################################