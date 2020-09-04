/* ELASTIC WAVES IN SOLIDS: VELOCITY-STRESS FORMULATION */
/* HP CONVERGENCE ANALYSIS */

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

    // LEVEL SET FUNCTION =============================================
#define TH (-49.0*M_PI/180.0)
#define P1 5.2
#define P2 0.0
#define N1 (std::cos(TH))
#define N2 (std::sin(TH))
#define VF 0.1
#define DELTA ((1.0-VF)*3.75/32.0)
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        //return (N1*x[0]+N2*x[1])-(N1*P1+N2*P2);

        return (DELTA-x[1]);
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1
        };

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }

    amrex::Real F_EXACT_VOLUME(const int & dom) const
    {
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            7.5*(3.75-DELTA)
        };
        return EXACT_VOLUME[dom];
    }

    amrex::Real F_EXACT_SURFACE(const int & dom) const
    {
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            7.5
        };
        return EXACT_SURFACE[dom];
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
#if (AMREX_SPACEDIM == 2)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0, 0, 0
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0, 0, 0, 0
        };
#endif
        return DOM[u];
    }
    // ================================================================

    // RELATIONSHIP BETWEEN NEIGHBORING DOMAINS =======================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {
        const int NBRDOM[N_DOM] =
        {
            // dom = 0
            -1
        };
        return NBRDOM[dom];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
#define xs 4.2
#define pR 1.95e4
#define rhoR 3.29e-5
#define vR 0.0
#define pL 7.42e6
#define rhoL 3.61e-4
#define vL 0.0

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        const amrex::Real g = this->gamma[0];

        amrex::Real U0[N_U];
        if (x[0] > xs)
        {
            U0[RHO] = rhoR;
            U0[RHOV1] = rhoR*vR;
            U0[RHOV2] = 0.0;
            U0[TE] = pR/(g-1.0)+0.5*rhoR*vR*vR;
        }
        else
        {
            U0[RHO] = rhoL;
            U0[RHOV1] = rhoL*vL;
            U0[RHOV2] = 0.0;
            U0[TE] = pL/(g-1.0)+0.5*rhoL*vL*vL;
        }

        return U0[u];
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    // ================================================================

    // NUMERICAL FLUXES: INTRAPHASE ===================================
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        this->_eval_NFn_LF_(dom, un, U, nbr_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: GRID BOUNDARIES ==============================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int & dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real bcs_U[N_U];

        // OUTFLOW BOUNDARY CONDITIONS
        if (un[0] < -0.5)
        {
            bcs_U[RHO] = 2.0*rhoL-U[RHO];
            bcs_U[RHOV1] = U[RHOV1];
            bcs_U[RHOV2] = U[RHOV2];
            bcs_U[TE] = U[TE];
        }
        else if (un[0] > 0.5)
        {
            bcs_U[RHO] = 2.0*rhoR-U[RHO];
            bcs_U[RHOV1] = U[RHOV1];
            bcs_U[RHOV2] = U[RHOV2];
            bcs_U[TE] = U[TE];
        }
        else if (un[1] < -0.5)
        {
            bcs_U[RHO] = U[RHO];
            bcs_U[RHOV1] = U[RHOV1];
            bcs_U[RHOV2] = -U[RHOV2];
            bcs_U[TE] = U[TE];
        }
        else if (un[1] > 0.5)
        {
            bcs_U[RHO] = U[RHO];
            bcs_U[RHOV1] = U[RHOV1];
            bcs_U[RHOV2] = -U[RHOV2];
            bcs_U[TE] = U[TE];
        }

        this->_eval_NFn_LF_(dom, un, U, bcs_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL BOUNDARIES ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real bcs_U[N_U];

        bcs_U[RHO] = U[RHO];
        bcs_U[RHOV1] = U[RHOV1];
        bcs_U[RHOV2] = -U[RHOV2];
        bcs_U[TE] = U[TE];

        this->_eval_NFn_LF_(dom, un, U, bcs_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL INTERFACE ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int dom, const int nbr_dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
amrex::Abort("GASDYNAMICS.F_NF_PHI_ICS");
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        fields_domain = {0, 0, 0, 0};
        fields_name = {"rho", "ux", "uy", "M"};
#endif

#if (AMREX_SPACEDIM == 3)
        fields_domain = {0, 0, 0, 0, 0};
        fields_name = {"rho", "ux", "uy", "uz", "M"};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
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
        
#if (AMREX_SPACEDIM == 2)
        // DOMAIN 0
        F[0] = rho;
        F[1] = u1;
        F[2] = u2;
        F[3] = M;
#endif

#if (AMREX_SPACEDIM == 3)
        // DOMAIN 0
        F[0] = rho;
        F[1] = u1;
        F[2] = u2;
        F[3] = u3;
        F[4] = M;
#endif
    }
    // ================================================================
};
// ####################################################################