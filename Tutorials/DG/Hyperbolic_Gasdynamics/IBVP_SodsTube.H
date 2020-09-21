/* GASDYNAMICS: INCLINED SODS TUBE TEST */

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
    AMREX_GPU_HOST_DEVICE
    amrex::Real norm2_3D(const amrex::Real * v) const
    {
        amrex::Real res;
        res =  v[0]*v[0];
        res += v[1]*v[1];
        res += v[2]*v[2];

        return std::sqrt(res);
    }
    AMREX_GPU_HOST_DEVICE
    amrex::Real dot_3D(const amrex::Real * v1, const amrex::Real * v2) const
    {
        amrex::Real res;
        res =  v1[0]*v2[0];
        res += v1[1]*v2[1];
        res += v1[2]*v2[2];

        return res;
    }
    AMREX_GPU_HOST_DEVICE
    void cross(const amrex::Real * v1, const amrex::Real * v2, amrex::Real * res) const
    {
        res[0] = v1[1]*v2[2]-v1[2]*v2[1];
        res[1] = v1[2]*v2[0]-v1[0]*v2[2];
        res[2] = v1[0]*v2[1]-v1[1]*v2[0];
    }

#define THETA (0.0*M_PI/6.0)
#define A1 0.0
#define B1 1.0
#if (AMREX_SPACEDIM == 2)
#define A2 (0.5-0.5*std::tan(THETA))
#define A3 0.0
#define B2 (0.5+0.5*std::tan(THETA))
#define B3 0.0
#endif
#if (AMREX_SPACEDIM == 3)
#define A2 0.5
#define A3 (0.5-0.5*std::tan(THETA))
#define B2 0.5
#define B3 (0.5+0.5*std::tan(THETA))
#endif
#define AB1 (B1-A1)
#define AB2 (B2-A2)
#define AB3 (B3-A3)
#define aAB (std::sqrt(AB1*AB1+AB2*AB2+AB3*AB3))
#define uAB1 (AB1/aAB)
#define uAB2 (AB2/aAB)
#define uAB3 (AB3/aAB)
#define DIAM AMREX_D_PICK(0.2, 0.2, 0.25)

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int ph, const amrex::Real t, const amrex::Real * x) const
    {
        const amrex::Real A[3] = {A1, A2, A3};
        const amrex::Real B[3] = {B1, B2, B3};
        const amrex::Real AB[3] = {AB1, AB2, AB3};
#if (AMREX_SPACEDIM == 2)
        const amrex::Real AX[3] = {x[0]-A[0], x[1]-A[1], 0.0};
        const amrex::Real BX[3] = {x[0]-B[0], x[1]-B[1], 0.0};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real AX[3] = {x[0]-A[0], x[1]-A[1], x[2]-A[2]};
        const amrex::Real BX[3] = {x[0]-B[0], x[1]-B[1], x[2]-B[2]};
#endif
        amrex::Real tmp[3];
        this->cross(BX, AX, tmp);

        const amrex::Real res = this->norm2_3D(tmp)/this->norm2_3D(AB)-0.5*DIAM;

        return res;
//return -x[1]+x[0]*x[0]-0.1;
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
    amrex::Real F_EXACT_VOLUME(const int dom) const
    {
        return DIAM/std::cos(THETA);
    }

    amrex::Real F_EXACT_SURFACE(const int dom) const
    {
        return 2.0/std::cos(THETA);
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
#define xis 0.5
#define pR 0.1
#define rhoR 0.125
#define vR 0.0
#define pL 1.0
#define rhoL 1.0
#define vL 0.0

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        const amrex::Real A[3] = {A1, A2, A3};
        const amrex::Real B[3] = {B1, B2, B3};
        const amrex::Real AB[3] = {AB1, AB2, AB3};
#if (AMREX_SPACEDIM == 2)
        const amrex::Real AX[3] = {x[0]-A[0], x[1]-A[1], 0.0};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real AX[3] = {x[0]-A[0], x[1]-A[1], x[2]-A[2]};
#endif
        const amrex::Real tmp_x = this->dot_3D(AX, AB)/this->dot_3D(AB, AB);
        const amrex::Real Px[AMREX_SPACEDIM] = {AMREX_D_DECL(A[0]+AB[0]*tmp_x, A[1]+AB[1]*tmp_x, A[2]+AB[2]*tmp_x)};
        
        const amrex::Real xi = (Px[0]-A[0])/(B[0]-A[0]);

        const amrex::Real g = this->gamma[0];

        amrex::Real U0[N_U];
        if (xi > xis)
//if ((x[0] + x[1] - 0.8) > 0.0)
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
        //this->_eval_NFn_mOsher_(dom, un, U, nbr_U, NFn);
        this->_eval_NFn_ExactRiemann_(dom, un, U, nbr_U, NFn);
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

        bcs_U[RHO] = U[RHO];
        AMREX_D_TERM
        (
            bcs_U[RHOV1] = U[RHOV1];,
            bcs_U[RHOV2] = U[RHOV2];,
            bcs_U[RHOV3] = U[RHOV3];
        )
        bcs_U[TE] = U[TE];

        this->_eval_NFn_(dom, un, U, bcs_U, NFn);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL BOUNDARIES ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Un[N_U], bcs_U[N_U];

        this->_eval_U2Un_(un, U, Un);
        Un[RHOV1] = -Un[RHOV1];
        this->_eval_Un2U_(un, Un, bcs_U);

        this->_eval_NFn_(dom, un, U, bcs_U, NFn);
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
        fields_domain = {0, AMREX_D_DECL(0, 0, 0), 0, 0, AMREX_D_DECL(0, 0, 0)};
        fields_name = {"rho", AMREX_D_DECL("u_x", "u_y", "u_z"), "M", "p", AMREX_D_DECL("u_xi", "u_eta", "u_zeta")};
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

        // VARIABLES
        amrex::Real Un[N_U], un[AMREX_SPACEDIM];
        AMREX_D_TERM
        (
            un[0] = uAB1;,
            un[1] = uAB2;,
            un[2] = uAB3;
        )
        this->_eval_U2Un_(un, U, Un);

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        
        rho = U[RHO];
        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));
        
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
        AMREX_D_TERM
        (
            F[TE+2] = Un[RHOV1]/rho;,
            F[TE+3] = Un[RHOV2]/rho;,
            F[TE+4] = Un[RHOV3]/rho;
        )
    }
    // ================================================================
};
// ####################################################################