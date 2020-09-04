/* ELASTIC WAVES IN SOLIDS */

#include <AMReX_DG_IBVP.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF VELOCITY COMPONENTS AND STRESS COMPONENTS ----------------
#define N_V AMREX_SPACEDIM

#define N_U_PER_DOM (1+N_V+1)

// ORDERING OF THE PRIMARY VARIABLES
#if (AMREX_SPACEDIM == 1)
#define RHO 0
#define RHOV1 1
#define TE 2
#endif
#if (AMREX_SPACEDIM == 2)
#define RHO 0
#define RHOV1 1
#define RHOV2 2
#define TE 3
#endif
#if (AMREX_SPACEDIM == 3)
#define RHO 0
#define RHOV1 1
#define RHOV2 2
#define RHOV3 3
#define TE 4
#endif
// --------------------------------------------------------------------

// PDES PARAMETERS AND CONSTANT OPERATORS -----------------------------
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
template<int N_PHI, int N_DOM, int N_U>
class GASDYNAMICS_BASE
:
public amrex::DG::IBVP_HYPERBOLIC_BASE<N_PHI, N_DOM, N_U>
{
public:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    amrex::Real gamma[N_DOM];

    // LARGEST WAVE SPEED
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    GASDYNAMICS_BASE(const amrex::Vector<std::string> & material_type,
                     const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    {
#if (AMREX_SPACEDIM == 1)
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_Base.H - GASDYNAMICS_BASE::GASDYNAMICS_BASE\n";
        msg += "| GASDYNAMICS_BASE must be double-checked step-by-step when dimension is "+std::to_string(AMREX_SPACEDIM)+".\n";
        amrex::Abort(msg);
#endif
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------

        // STORE THE MATERIAL PROPERTIES ------------------------------
        for (int d = 0; d < N_DOM; ++d)
        {
            // PARAMETERS
            const std::string & m_type = material_type[d];
            const amrex::Vector<amrex::Real> & m_prop = material_properties[d];

            // STIFFNESS MATRIX
            if (m_type.compare("Ideal gas") == 0)
            {
                // ADIABATIC INDEX
                this->gamma[d] = m_prop[0];
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: IBVP_Base.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE\n";
                msg += "| Unexpected material type: "+m_type+" with dimension: "+std::to_string(AMREX_SPACEDIM)+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // TIME STEP ======================================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const amrex::Real * dx,
                     const amrex::Real t, const amrex::Real * x,
                     const amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma[0];

        // VARIABLES
        amrex::Real dt;

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(0, U, rho, AMREX_D_DECL(u1, u2, u3), p);
        const amrex::Real u = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
        const amrex::Real a = std::sqrt(g*p/rho);

        // LARGEST EIGENVALUE
        const amrex::Real lam = u+a;

#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
#endif
        dt = h/lam;

        return dt;
    }
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    // ================================================================

    // BODY LOAD ======================================================
    AMREX_GPU_HOST_DEVICE
    void F_B(const amrex::Real t, const amrex::Real * x,
             amrex::Real * B) const
    {
        for (int u = 0; u < (N_DOM*N_U_PER_DOM); ++u)
        {
            B[u] = 0.0;
        }
    }
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void _eval_U2Un_(const amrex::Real * un, const amrex::Real * U, amrex::Real * Un) const
    {
#if (AMREX_SPACEDIM == 2)
        Un[RHO] = U[RHO];
        Un[RHOV1] =  un[0]*U[RHOV1]+un[1]*U[RHOV2];
        Un[RHOV2] = -un[1]*U[RHOV1]+un[0]*U[RHOV2];
        Un[TE] = U[TE];
#else
        amrex::Abort("GASDYNAMICS_BASE._eval_U2Un_");
#endif
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_Un2U_(const amrex::Real * un, const amrex::Real * Un, amrex::Real * U) const
    {
#if (AMREX_SPACEDIM == 2)
        Un[RHO] = U[RHO];
        Un[RHOV1] = un[0]*U[RHOV1]-un[1]*U[RHOV2];
        Un[RHOV2] = un[1]*U[RHOV1]+un[0]*U[RHOV2];
        Un[TE] = U[TE];
#else
        amrex::Abort("GASDYNAMICS_BASE._eval_Un2U_");
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void _eval_primary_variables_(const int dom, const amrex::Real * U,
                                  amrex::Real & rho,
                                  AMREX_D_DECL(amrex::Real & u1, amrex::Real & u2, amrex::Real & u3),
                                  amrex::Real & p) const
    {
        const amrex::Real g = this->gamma[dom];

        rho = U[RHO];
        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));

if ((p < 0.0) || (rho < 0.0))
{
amrex::Print() << "We might need a limiter" << std::endl;
amrex::Print() << "U: ";
amrex::DG_utils::PrintRealArray2D(1, N_U_PER_DOM, U);
amrex::Print() << "rho: " << rho << std::endl;
amrex::Print() << "p: " << p << std::endl;
exit(-1);
}
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_enthalpy_variables_(const int dom, const amrex::Real * U,
                                   AMREX_D_DECL(amrex::Real & u1, amrex::Real & u2, amrex::Real & u3),
                                   amrex::Real & p, amrex::Real & a, amrex::Real & h) const
    {
        const amrex::Real g = this->gamma[dom];
        const amrex::Real rho = U[RHO];

        AMREX_D_TERM
        (
            u1 = U[RHOV1]/rho;,
            u2 = U[RHOV2]/rho;,
            u3 = U[RHOV3]/rho;
        )
        const amrex::Real e = U[TE];
        p = (g-1)*(e-0.5*rho*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3)));
        a = std::sqrt(g*p/rho);
        h = 0.5*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))+a*a/(g-1.0);
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real _eval_max_wave_speed_(const int dom, const amrex::Real * un, const amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma[dom];

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);
        const amrex::Real a = std::sqrt(g*p/rho);

#if (AMREX_SPACEDIM == 2)
        return std::max(-a+u1*un[0]+u2*un[1], +a+u1*un[0]+u2*un[1]);
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_dFndU_eigen_system_(const int dom, const amrex::Real * un, const amrex::Real * U,
                                   amrex::Real * w,
                                   amrex::Real * v) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma[dom];

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);
        const amrex::Real a = std::sqrt(g*p/rho);

#if (AMREX_SPACEDIM == 2)
        w[0] = -a+u1*un[0]+u2*un[1];
        w[1] = u1*un[0]+u2*un[1];
        w[2] = u1*un[0]+u2*un[1];
        w[3] = +a+u1*un[0]+u2*un[1];
#else
        amrex::Abort("GASDYNAMICS_BASE._eval_dFndU_eigen_system_");
#endif


amrex::Print() << "w: ";
amrex::DG_utils::PrintRealArray2D(1, 4, w);

amrex::Print() << "HERE WE ARE " << std::endl;
exit(-1);
    }

    // LAX-FRIEDRICHS FLUX (ALSO KNOWN AS RUSANOV FLUX)
    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_LF_(const int dom, const amrex::Real * un, const amrex::Real * U, const amrex::Real * nbr_U,
                       amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Fn[N_U_PER_DOM], nbr_Fn[N_U_PER_DOM];

        // EVAL MAXIMUM WAVE SPEED
        const amrex::Real s = std::max(this->_eval_max_wave_speed_(dom, un, U), this->_eval_max_wave_speed_(dom, un, nbr_U));

        // EVAL LEFT AND RIGHT FLUXES
        this->_eval_Fn_(dom, un, U, Fn);
        this->_eval_Fn_(dom, un, nbr_U, nbr_Fn);

        // EVAL NUMERICAL FLUX
        NFn[RHO] = 0.5*(Fn[RHO]+nbr_Fn[RHO])+0.5*s*(U[RHO]-nbr_U[RHO]);
        AMREX_D_TERM
        (
            NFn[RHOV1] = 0.5*(Fn[RHOV1]+nbr_Fn[RHOV1])+0.5*s*(U[RHOV1]-nbr_U[RHOV1]);,
            NFn[RHOV2] = 0.5*(Fn[RHOV2]+nbr_Fn[RHOV2])+0.5*s*(U[RHOV2]-nbr_U[RHOV2]);,
            NFn[RHOV3] = 0.5*(Fn[RHOV3]+nbr_Fn[RHOV3])+0.5*s*(U[RHOV3]-nbr_U[RHOV3]);
        )
        NFn[TE] = 0.5*(Fn[TE]+nbr_Fn[TE])+0.5*s*(U[TE]-nbr_U[TE]);
    }

    // MODIFIED OSHER FLUX
    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_mOsher_(const int dom, const amrex::Real * un, const amrex::Real * U, const amrex::Real * nbr_U,
                           amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Fn[N_U_PER_DOM], nbr_Fn[N_U_PER_DOM];
        amrex::Real Uxi[N_U_PER_DOM];
        amrex::Real An[N_U_PER_DOM*N_U_PER_DOM], IntAn[N_U_PER_DOM*N_U_PER_DOM];

        // EVAL LEFT AND RIGHT FLUXES
        this->_eval_Fn_(dom, un, U, Fn);
        this->_eval_Fn_(dom, un, nbr_U, nbr_Fn);

        // EVAL INTEGRAL CONTRIBUTION
        const int qo = 3;

        std::fill(IntAn, IntAn+N_U_PER_DOM*N_U_PER_DOM, 0.0);

        for (int i = 0; i < qo; ++i)
        {
            const amrex::Real xi = Algoim::GaussQuad::x(qo, i);
            const amrex::Real w = Algoim::GaussQuad::w(qo, i);

            Uxi[RHO] = U[RHO]+xi*(nbr_U[RHO]-U[RHO]);
            AMREX_D_TERM
            (
                Uxi[RHOV1] = U[RHOV1]+xi*(nbr_U[RHOV1]-U[RHOV1]);,
                Uxi[RHOV2] = U[RHOV2]+xi*(nbr_U[RHOV2]-U[RHOV2]);,
                Uxi[RHOV3] = U[RHOV3]+xi*(nbr_U[RHOV3]-U[RHOV3]);
            )
            Uxi[TE] = U[TE]+xi*(nbr_U[TE]-U[TE]);

            this->_eval_An_(dom, un, Uxi, An);

            for (int k = 0; k < N_U_PER_DOM*N_U_PER_DOM; ++k)
            {
                IntAn[k] += std::abs(An[k])*w;
            }
        }

        // EVAL NUMERICAL FLUX
        NFn[RHO] = 0.5*(Fn[RHO]+nbr_Fn[RHO]);
        AMREX_D_TERM
        (
            NFn[RHOV1] = 0.5*(Fn[RHOV1]+nbr_Fn[RHOV1]);,
            NFn[RHOV2] = 0.5*(Fn[RHOV2]+nbr_Fn[RHOV2]);,
            NFn[RHOV3] = 0.5*(Fn[RHOV3]+nbr_Fn[RHOV3]);
        )
        NFn[TE] = 0.5*(Fn[TE]+nbr_Fn[TE]);

        for (int r = 0; r < N_U_PER_DOM; ++r)
        for (int c = 0; c < N_U_PER_DOM; ++c)
        {
            NFn[r] += 0.5*IntAn[r+c*N_U_PER_DOM]*(U[c]-nbr_U[c]);
        }
    }
    // ================================================================

    // EVAL FLUX FUNCTIONS ============================================
    // EVAL F1 --------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_F1_(const int dom, const amrex::Real * U, amrex::Real * F1) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F1[RHO] = U[RHOV1];
        AMREX_D_TERM
        (
            F1[RHOV1] = U[RHOV1]*u1+p;,
            F1[RHOV2] = U[RHOV2]*u1;,
            F1[RHOV3] = U[RHOV3]*u1;
        )
        F1[TE] = (U[TE]+p)*u1;
    }
    // ----------------------------------------------------------------

    // EVAL F2 --------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_F2_(const int dom, const amrex::Real * U, amrex::Real * F2) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F2[RHO] = U[RHOV2];
        AMREX_D_TERM
        (
            F2[RHOV1] = U[RHOV1]*u2;,
            F2[RHOV2] = U[RHOV2]*u2+p;,
            F2[RHOV3] = U[RHOV3]*u2;
        )
        F2[TE] = (U[TE]+p)*u2;
    }
    // ----------------------------------------------------------------

    // EVAL F3 --------------------------------------------------------
#if (AMREX_SPACEDIM > 2)
    AMREX_GPU_HOST_DEVICE
    void _eval_F3_(const int dom, const amrex::Real * U, amrex::Real * F3) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);

        F3[RHO] = U[RHOV3];
        AMREX_D_TERM
        (
            F3[RHOV1] = U[RHOV1]*u3;,
            F3[RHOV2] = U[RHOV2]*u3;,
            F3[RHOV3] = U[RHOV3]*u3+p;
        )
        F3[TE] = (U[TE]+p)*u3;
    }
#endif
    // ----------------------------------------------------------------

    // EVAL Fn = Fi*ni ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_Fn_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * Fn) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);

#if (AMREX_SPACEDIM == 2)
        Fn[RHO] = U[RHOV1]*un[0]+U[RHOV2]*un[1];
        Fn[RHOV1] = (U[RHOV1]*u1+p)*un[0]+(U[RHOV1]*u2  )*un[1];
        Fn[RHOV2] = (U[RHOV2]*u1  )*un[0]+(U[RHOV2]*u2+p)*un[1];
        Fn[TE] = (U[TE]+p)*u1*un[0]+(U[TE]+p)*u2*un[1];
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "HERE WE ARE - _eval_Fn_" << std::endl;
exit(-1);
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_Fn_via_Un_(const int dom, const amrex::Real * un, const amrex::Real * Un, amrex::Real * Fn) const
    {
        // VARIABLES
        amrex::Real F1[N_U];

        this->_eval_F1_(dom, Un, F1);

#if (AMREX_SPACEDIM == 2)
        Fn[RHO] = F1[RHO];
        Fn[RHOV1] = F1[RHOV1]*un[0]-F1[RHOV2]*un[1];
        Fn[RHOV2] = F1[RHOV1]*un[1]+F1[RHOV2]*un[0];
        Fn[TE] = F1[TE];
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "HERE WE ARE - _eval_Fn_" << std::endl;
exit(-1);
#endif
    }
    // ----------------------------------------------------------------

    // EVAL An = dFndU = dFidU*ni -------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_An_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * An) const
    {
        // PARAMETERS
        const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);
        const amrex::Real g = this->gamma[dom];
        const amrex::Real g1 = g-1.0;
        const amrex::Real g3 = g-3.0;

        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h;
        this->_eval_enthalpy_variables_(dom, U, AMREX_D_DECL(u1, u2, u3), p, a, h);

#if (AMREX_SPACEDIM == 2)
        An[0+0*N_U_PER_DOM] = 0.0;
        An[1+0*N_U_PER_DOM] = 0.5*(g3*u1*u1+g1*u2*u2)*n1-u1*u2*n2;
        An[2+0*N_U_PER_DOM] = 0.5*(g1*u1*u1+g3*u2*u2)*n2-u1*u2*n1;
        An[3+0*N_U_PER_DOM] = 0.5*u1*(-2.0*h+g1*(u1*u1+u2*u2))*n1+0.5*u2*(-2.0*h+g1*(u1*u1+u2*u2))*n2;

        An[0+1*N_U_PER_DOM] = n1;
        An[1+1*N_U_PER_DOM] = -g3*u1*n1+u2*n2;
        An[2+1*N_U_PER_DOM] = u2*n1-g1*u1*n2;
        An[3+1*N_U_PER_DOM] = (h-g1*u1*u1)*n1-g1*u1*u2*n2;

        An[0+2*N_U_PER_DOM] = n2;
        An[1+2*N_U_PER_DOM] = u1*n2-g1*u2*n1;
        An[2+2*N_U_PER_DOM] = -g3*u2*n2+u1*n1;
        An[3+2*N_U_PER_DOM] = (h-g1*u2*u2)*n2-g1*u1*u2*n1;

        An[0+3*N_U_PER_DOM] = 0.0;
        An[1+3*N_U_PER_DOM] = g1*n1;
        An[2+3*N_U_PER_DOM] = g1*n2;
        An[3+3*N_U_PER_DOM] = g*u1*n1+g*u2*n2;
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "HERE WE ARE - _eval_An_" << std::endl;
exit(-1);
#endif
    }
    // ----------------------------------------------------------------
    // ================================================================

    // PDES SYSTEM ====================================================
    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        for (int d = 0; d < N_DOM; ++d)
        {
            const amrex::Real * Ud = &U[d*N_U_PER_DOM];
            AMREX_D_TERM
            (
                amrex::Real * F1d = &F1[d*N_U_PER_DOM];,
                amrex::Real * F2d = &F2[d*N_U_PER_DOM];,
                amrex::Real * F3d = &F3[d*N_U_PER_DOM];
            )

            AMREX_D_TERM
            (
                this->_eval_F1_(d, Ud, F1d);,
                this->_eval_F2_(d, Ud, F2d);,
                this->_eval_F3_(d, Ud, F3d);
            )
        }
    }
    // ================================================================

    // TROUBLED CELLS INDICATOR =======================================
    AMREX_GPU_HOST_DEVICE
    int F_TROUBLED(const int dom, const amrex::Real t, const amrex::Real * x, const amrex::Real * U) const
    {
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->_eval_primary_variables_(dom, U, rho, AMREX_D_DECL(u1, u2, u3), p);
        
        const int res = ((rho < 0.0) || (p < 0.0)) ? 1 : 0;

        return res;
    }
    // ================================================================
};
// ####################################################################