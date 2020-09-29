/* GASDYNAMICS */

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
#define SMALL_P 1.0e-10
#define SMALL_RHO 1.0e-19
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

    // CONSERVED-CHARACTERISTICS VARIABLES RELATIONSHIPS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_1(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DU, amrex::Real * DC) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma[dom];
        const amrex::Real g1 = g-1.0;
        
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h;
        this->_eval_enthalpy_variables_(dom, U, AMREX_D_DECL(u1, u2, u3), p, a, h);
        
#if (AMREX_SPACEDIM == 2)
        DC[RHO]   =     (h+a*(u1-a)/g1)*DU[RHO]+(-a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]    +DU[TE];
        DC[RHOV1] = (-2.0*h+4.0*a*a/g1)*DU[RHO]    +2.0*u1*DU[RHOV1]      +2.0*u2*DU[RHOV2]-2.0*DU[TE];
        DC[RHOV2] =    (-2.0*a*a*u2/g1)*DU[RHO]                     +(2.0*a*a/g1)*DU[RHOV2];
        DC[TE]    =     (h-a*(u1+a)/g1)*DU[RHO]+(+a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]    +DU[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DC[RHO]   =     (h+a*(u1-a)/g1)*DU[RHO]+(-a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]          -u3*DU[RHOV3]    +DU[TE];
        DC[RHOV1] = (-2.0*h+4.0*a*a/g1)*DU[RHO]    +2.0*u1*DU[RHOV1]      +2.0*u2*DU[RHOV2]      +2.0*u3*DU[RHOV3]-2.0*DU[TE];
        DC[RHOV2] =    (-2.0*a*a*u2/g1)*DU[RHO]                     +(2.0*a*a/g1)*DU[RHOV2];
        DC[RHOV3] =    (-2.0*a*a*u3/g1)*DU[RHO]                                            +(2.0*a*a/g1)*DU[RHOV3];
        DC[TE]    =     (h-a*(u1+a)/g1)*DU[RHO]+(+a/g1-u1)*DU[RHOV1]          -u2*DU[RHOV2]          -u3*DU[RHOV3]    +DU[TE];
#endif
        const amrex::Real delta = 0.5*g1/(a*a);
        DC[RHO] *= delta;
        AMREX_D_TERM
        (
            DC[RHOV1] *= delta;,
            DC[RHOV2] *= delta;,
            DC[RHOV3] *= delta;
        )
        DC[TE] *= delta;
    }
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_2(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DU, amrex::Real * DC) const
    {   
#if (AMREX_SPACEDIM == 2)
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV2], -U[RHOV1], U[TE]};
        const amrex::Real tmp[N_U] = {DU[RHO], DU[RHOV2], -DU[RHOV1], DU[TE]};
#endif
#if (AMREX_SPACEDIM == 3)
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV2], -U[RHOV1], U[RHOV3], U[TE]};
        const amrex::Real tmp[N_U] = {DU[RHO], DU[RHOV2], -DU[RHOV1], DU[RHOV3], DU[TE]};
#endif
        this->F_DU2DC_1(dom, t, x, Un, tmp, DC);
    }
#if (AMREX_SPACEDIM == 3)
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_3(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DU, amrex::Real * DC) const
    {
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV3], U[RHOV2], -U[RHOV1], U[TE]};
        const amrex::Real tmp[N_U] = {DU[RHO], DU[RHOV3], DU[RHOV2], -DU[RHOV1], DU[TE]};
        
        this->F_DU2DC_1(dom, t, x, Un, tmp, DC);
    }
#endif


    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_1(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DC, amrex::Real * DU) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h;
        this->_eval_enthalpy_variables_(dom, U, AMREX_D_DECL(u1, u2, u3), p, a, h);
#if (AMREX_SPACEDIM == 2)
        DU[RHO]   =          DC[RHO]                  +DC[RHOV1]                      +DC[TE];
        DU[RHOV1] =  (-a+u1)*DC[RHO]               +u1*DC[RHOV1]               +(a+u1)*DC[TE];
        DU[RHOV2] =       u2*DC[RHO]               +u2*DC[RHOV1]   +DC[RHOV2]      +u2*DC[TE];
        DU[TE]    = (h-a*u1)*DC[RHO]+0.5*(u1*u1+u2*u2)*DC[RHOV1]+u2*DC[RHOV2]+(h+a*u1)*DC[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        DU[RHO]   =          DC[RHO]                        +DC[RHOV1]                                   +DC[TE];
        DU[RHOV1] =  (-a+u1)*DC[RHO]                     +u1*DC[RHOV1]                            +(a+u1)*DC[TE];
        DU[RHOV2] =       u2*DC[RHO]                     +u2*DC[RHOV1]   +DC[RHOV2]                   +u2*DC[TE];
        DU[RHOV3] =       u3*DC[RHO]                     +u3*DC[RHOV1]                +DC[RHOV3]      +u3*DC[TE];
        DU[TE]    = (h-a*u1)*DC[RHO]+0.5*(u1*u1+u2*u2+u3*u3)*DC[RHOV1]+u2*DC[RHOV2]+u3*DC[RHOV3]+(h+a*u1)*DC[TE];
#endif
    }
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_2(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DC, amrex::Real * DU) const
    {
#if (AMREX_SPACEDIM == 2)
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV2], -U[RHOV1], U[TE]};
        amrex::Real tmp[N_U];

        this->F_DC2DU_1(dom, t, x, Un, DC, tmp);
        
        DU[RHO] = tmp[RHO];
        DU[RHOV1] = -tmp[RHOV2];
        DU[RHOV2] = tmp[RHOV1];
        DU[TE] = tmp[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV2], -U[RHOV1], U[RHOV3], U[TE]};
        amrex::Real tmp[N_U];

        this->F_DC2DU_1(dom, t, x, Un, DC, tmp);

        DU[RHO] = tmp[RHO];
        DU[RHOV1] = -tmp[RHOV2];
        DU[RHOV2] = tmp[RHOV1];
        DU[RHOV3] = tmp[RHOV3];
        DU[TE] = tmp[TE];
#endif
    }
#if (AMREX_SPACEDIM == 3)
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_3(const int dom,
                   const amrex::Real t, const amrex::Real * x,
                   const amrex::Real * U,
                   const amrex::Real * DC, amrex::Real * DU) const
    {
        // VARIABLES
        const amrex::Real Un[N_U] = {U[RHO], U[RHOV3], U[RHOV2], -U[RHOV1], U[TE]};
        amrex::Real tmp[N_U];

        this->F_DC2DU_1(dom, t, x, Un, DC, tmp);

        DU[RHO] = tmp[RHO];
        DU[RHOV1] = -tmp[RHOV3];
        DU[RHOV2] = tmp[RHOV2];
        DU[RHOV3] = tmp[RHOV1];
        DU[TE] = tmp[TE];
    }
#endif
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void _eval_Tn_(const amrex::Real * un, amrex::Real * Tn) const
    {
#if (AMREX_SPACEDIM == 2)
        Tn[0+0*N_U_PER_DOM] = 1.0; Tn[0+1*N_U_PER_DOM] =    0.0; Tn[0+2*N_U_PER_DOM] =   0.0; Tn[0+3*N_U_PER_DOM] = 0.0;
        Tn[1+0*N_U_PER_DOM] = 0.0; Tn[1+1*N_U_PER_DOM] =  un[0]; Tn[1+2*N_U_PER_DOM] = un[1]; Tn[1+3*N_U_PER_DOM] = 0.0;
        Tn[2+0*N_U_PER_DOM] = 0.0; Tn[2+1*N_U_PER_DOM] = -un[1]; Tn[2+2*N_U_PER_DOM] = un[0]; Tn[2+3*N_U_PER_DOM] = 0.0;
        Tn[3+0*N_U_PER_DOM] = 0.0; Tn[3+1*N_U_PER_DOM] =    0.0; Tn[3+2*N_U_PER_DOM] =   0.0; Tn[3+3*N_U_PER_DOM] = 1.0;
#else
        amrex::Abort("GASDYNAMICS_BASE._eval_Tn_");
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_U2Un_(const amrex::Real * un, const amrex::Real * U, amrex::Real * Un) const
    {
#if (AMREX_SPACEDIM == 2)
        Un[RHO] = U[RHO];
        Un[RHOV1] =  un[0]*U[RHOV1]+un[1]*U[RHOV2];
        Un[RHOV2] = -un[1]*U[RHOV1]+un[0]*U[RHOV2];
        Un[TE] = U[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        Un[RHO] = U[RHO];
        Un[RHOV1] =  cph*sth*U[RHOV1]+sph*sth*U[RHOV2]+cth*U[RHOV3];
        Un[RHOV2] =     -sph*U[RHOV1]    +cph*U[RHOV2];
        Un[RHOV3] = -cph*cth*U[RHOV1]-cth*sph*U[RHOV2]+sth*U[RHOV3];
        Un[TE] = U[TE];
#endif
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_Un2U_(const amrex::Real * un, const amrex::Real * Un, amrex::Real * U) const
    {
#if (AMREX_SPACEDIM == 2)
        U[RHO] = Un[RHO];
        U[RHOV1] = un[0]*Un[RHOV1]-un[1]*Un[RHOV2];
        U[RHOV2] = un[1]*Un[RHOV1]+un[0]*Un[RHOV2];
        U[TE] = Un[TE];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        U[RHO] = Un[RHO];
        U[RHOV1] = cph*sth*Un[RHOV1]-sph*Un[RHOV2]-cph*cth*Un[RHOV3];
        U[RHOV2] = sph*sth*Un[RHOV1]+cph*Un[RHOV2]-cth*sph*Un[RHOV3];
        U[RHOV3] =     cth*Un[RHOV1]                  +sth*Un[RHOV3];
        U[TE] = Un[TE];
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
    std::string msg;
    msg  = "\n";
    msg += "ERROR: IBVP_Base.H - GASDYNAMICS_BASE::_eval_primary_variables_\n";
    msg += "| We might need a limiter.\n";
    msg += "| U: "+std::to_string(U[RHO]);
    AMREX_D_TERM
    (
        msg += " "+std::to_string(U[RHOV1]);,
        msg += " "+std::to_string(U[RHOV2]);,
        msg += " "+std::to_string(U[RHOV3]);
    )
    msg += " "+std::to_string(U[TE]);
    msg += "\n";
    msg += "| rho: "+std::to_string(rho)+"\n";
    msg += "| p: "+std::to_string(p)+"\n";
    amrex::Abort(msg);
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

if ((p < 0.0) || (rho < 0.0))
{
    std::string msg;
    msg  = "\n";
    msg += "ERROR: IBVP_Base.H - GASDYNAMICS_BASE::_eval_enthalpy_variables_\n";
    msg += "| We might need a limiter.\n";
    msg += "| U: "+std::to_string(U[RHO]);
    AMREX_D_TERM
    (
        msg += " "+std::to_string(U[RHOV1]);,
        msg += " "+std::to_string(U[RHOV2]);,
        msg += " "+std::to_string(U[RHOV3]);
    )
    msg += " "+std::to_string(U[TE]);
    msg += "\n";
    msg += "| rho: "+std::to_string(rho)+"\n";
    msg += "| p: "+std::to_string(p)+"\n";
    amrex::Abort(msg);
}

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
#if (AMREX_SPACEDIM == 3)
        amrex::Abort("GASDYNAMICS_BASE._eval_max_wave_speed_");
        return 0.0;
#endif
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

    // EXACT RIEMANN SOLVER
    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_ExactRiemann_(const int dom, const amrex::Real * un, const amrex::Real * U, const amrex::Real * nbr_U,
                                 amrex::Real * NFn) const
    {
        // VARIABLES --------------------------------------------------
        amrex::Real Un[N_U_PER_DOM], nbr_Un[N_U_PER_DOM];
        amrex::Real tmp[N_U_PER_DOM];
        // ------------------------------------------------------------

        // EXPRESS THE STATE IN LOCAL REFERENCE SYSTEM ----------------
        this->_eval_U2Un_(un, U, Un);
        this->_eval_U2Un_(un, nbr_U, nbr_Un);
        // ------------------------------------------------------------

        // CALL THE ACTUAL RIEMANN SOLVER -----------------------------
        this->_eval_NFn_ExactRiemann_Un_(dom, Un, nbr_Un, tmp);
        this->_eval_Un2U_(un, tmp, NFn);
        // ------------------------------------------------------------
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real _aux_gK_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;

        return std::sqrt(A/(ps+B));
    }
    AMREX_GPU_HOST_DEVICE
    amrex::Real _aux_fK_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;
        amrex::Real res;

        if (ps > p)
        {
            res = (ps-p)*std::sqrt(A/(ps+B));
        }
        else
        {
            res = 2.0*a/(g-1.0)*(std::pow(ps/p, 0.5*(g-1.0)/g)-1.0);
        }

        return res;
    }
    AMREX_GPU_HOST_DEVICE
    amrex::Real _aux_dfK_(const amrex::Real g,
                          const amrex::Real ps,
                          const amrex::Real rho, const amrex::Real p) const
    {
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real A = 2.0/((g+1.0)*rho);
        const amrex::Real B = (g-1.0)/(g+1.0)*p;
        amrex::Real res;

        if (ps > p)
        {
            res = (1.0-0.5*(ps-p)/(ps+B))*std::sqrt(A/(ps+B));
        }
        else
        {
            res = (1.0/(a*rho))*std::pow(ps/p, -0.5*(g+1.0)/g);
        }

        return res;
    }
    AMREX_GPU_HOST_DEVICE
    amrex::Real _aux_f_(const amrex::Real g,
                        const amrex::Real ps,
                        const amrex::Real rhoL, const amrex::Real pL,
                        const amrex::Real rhoR, const amrex::Real pR,
                        const amrex::Real dun) const
    {
        const amrex::Real res = (this->_aux_fK_(g, ps, rhoL, pL)+this->_aux_fK_(g, ps, rhoR, pR)+dun);
        return res;
    }
    AMREX_GPU_HOST_DEVICE
    amrex::Real _aux_df_(const amrex::Real g,
                         const amrex::Real ps,
                         const amrex::Real rhoL, const amrex::Real pL,
                         const amrex::Real rhoR, const amrex::Real pR,
                         const amrex::Real dun) const
    {
        const amrex::Real res = (this->_aux_dfK_(g, ps, rhoL, pL)+this->_aux_dfK_(g, ps, rhoR, pR));
        return res;
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_ExactRiemann_Un_(const int dom, const amrex::Real * Un, const amrex::Real * nbr_Un,
                                    amrex::Real * NFn) const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->gamma[dom];
        const amrex::Real gg = (g-1.0)/(g+1.0);
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // LEFT STATE
        amrex::Real rhoL, AMREX_D_DECL(u1L, u2L, u3L), pL, aL;

        // RIGHT STATE
        amrex::Real rhoR, AMREX_D_DECL(u1R, u2R, u3R), pR, aR;

        // INTERFACE STATE
        amrex::Real rhoI, AMREX_D_DECL(u1I, u2I, u3I), pI;
        amrex::Real UI[N_U_PER_DOM];

        // AUXILIARY QUANTITIES
        amrex::Real dun, us;
        amrex::Real pm, pM, fm, fM, fs, dfs;
        amrex::Real pPV, pTS, ps, dps, aux_ps;
        amrex::Real gL, gR;
        
        const int n_max_it = 20;
        int it;
        // ------------------------------------------------------------

        // EVAL THE LEFT AND RIGHT STATES -----------------------------
        this->_eval_primary_variables_(dom,     Un, rhoL, AMREX_D_DECL(u1L, u2L, u3L), pL);
        aL = std::sqrt(g*pL/rhoL);

        this->_eval_primary_variables_(dom, nbr_Un, rhoR, AMREX_D_DECL(u1R, u2R, u3R), pR);
        aR = std::sqrt(g*pR/rhoR);

        // SOME AUXILIARY QUANTITIES
        dun = u1R-u1L;

        pm = std::min(pL, pR);
        pM = std::max(pL, pR);

        fm = this->_aux_f_(g, pm, rhoL, pL, rhoR, pR, dun);
        fM = this->_aux_f_(g, pM, rhoL, pL, rhoR, pR, dun);
        // ------------------------------------------------------------

        // FIND PRESSURE AND VELOCITY IN STAR REGION ------------------
        pPV = 0.5*(pL+pR)-0.125*dun*(rhoL+rhoR)*(aL+aR);
        pPV = std::max(1.0e-10, pPV);
        gL = this->_aux_gK_(g, pPV, rhoL, pL);
        gR = this->_aux_gK_(g, pPV, rhoR, pR);
        pTS = (gL*pL+gR*pR-dun)/(gL+gR);

        // START ITERATION (TO COMPUTE PRESSURE)
        ps = pTS;
        it = 0;
        do
        {
            aux_ps = 0.5*ps;

            fs = this->_aux_f_(g, ps, rhoL, pL, rhoR, pR, dun);
            dfs = this->_aux_df_(g, ps, rhoL, pL, rhoR, pR, dun);
            dps = -fs/dfs;
            ps += dps;

            aux_ps += 0.5*ps;

            it += 1;
        }
        while ((it < n_max_it) && (std::abs(dps) > aux_ps*1.0e-8));

        // EVAL VELOCITY
        us = 0.5*(u1L+u1R)+0.5*(this->_aux_fK_(g, ps, rhoR, pR)-this->_aux_fK_(g, ps, rhoL, pL));
        // ------------------------------------------------------------

        // FIND SOLUTION AT INTERFACE ---------------------------------
        if (us > 0.0)
        {
            // LEFT SHOCK
            if (ps > pL)
            {
                const amrex::Real rhoLs = rhoL*(ps/pL+gg)/(gg*ps/pL+1.0);
                const amrex::Real SL = u1L-aL*std::sqrt((0.5/g)*((g+1.0)*(ps/pL)+g-1.0));

                if (SL < 0.0)
                {
                    rhoI = rhoLs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = ps;
                }
                else
                {
                    rhoI = rhoL;
                    AMREX_D_TERM
                    (
                        u1I = u1L;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL;
                }
            }
            // LEFT FAN
            else
            {
                const amrex::Real rhoLs = rhoL*std::pow(ps/pL, 1.0/g);
                const amrex::Real aLs = std::sqrt(g*ps/rhoLs);

                if ((us-aLs) < 0.0)
                {
                    rhoI = rhoLs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = ps;
                }
                else if ((u1L-aL) > 0.0)
                {
                    rhoI = rhoL;
                    AMREX_D_TERM
                    (
                        u1I = u1L;,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL;
                }
                else
                {
                    rhoI = rhoL*std::pow(2.0/(g+1)+(g-1.0)/(g+1.0)*u1L/aL, 2.0/(g-1.0));
                    AMREX_D_TERM
                    (
                        u1I = 2.0/(g+1.0)*(aL+0.5*(g-1.0)*u1L);,
                        u2I = u2L;,
                        u3I = u3L;
                    )
                    pI = pL*std::pow(2.0/(g+1)+(g-1.0)/(g+1.0)*u1L/aL, 2.0*g/(g-1.0));
                }
            }
        }
        else if (us < 0.0)
        {   
            // RIGHT SHOCK
            if (ps > pR)
            {
                const amrex::Real rhoRs = rhoR*(ps/pR+gg)/(gg*ps/pR+1.0);
                const amrex::Real SR = u1R+aR*std::sqrt((0.5/g)*((g+1.0)*(ps/pR)+g-1.0));

                if (SR > 0.0)
                {
                    rhoI = rhoRs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = ps;
                }
                else
                {
                    rhoI = rhoR;
                    AMREX_D_TERM
                    (
                        u1I = u1R;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR;
                }
            }
            // RIGHT FAN
            else
            {
                const amrex::Real rhoRs = rhoR*std::pow(ps/pR, 1.0/g);
                const amrex::Real aRs = std::sqrt(g*ps/rhoRs);

                if ((us+aRs) > 0.0)
                {
                    rhoI = rhoRs;
                    AMREX_D_TERM
                    (
                        u1I = us;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = ps;
                }
                else if ((u1R+aR) < 0.0)
                {
                    rhoI = rhoR;
                    AMREX_D_TERM
                    (
                        u1I = u1R;,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR;
                }
                else
                {
                    rhoI = rhoR*std::pow(2.0/(g+1)-(g-1.0)/(g+1.0)*u1R/aR, 2.0/(g-1.0));
                    AMREX_D_TERM
                    (
                        u1I = 2.0/(g+1.0)*(-aR+0.5*(g-1.0)*u1R);,
                        u2I = u2R;,
                        u3I = u3R;
                    )
                    pI = pR*std::pow(2.0/(g+1)-(g-1.0)/(g+1.0)*u1R/aR, 2.0*g/(g-1.0));
                }
            }
        }
        else
        {
            amrex::Real rhoLs, rhoRs;

            // LEFT SHOCK
            if (ps > pL)
            {
                rhoLs = rhoL*(ps/pL+gg)/(gg*ps/pL+1.0);
            }
            // LEFT FAN
            else
            {
                rhoLs = rhoL*std::pow(ps/pL, 1.0/g);
            }
            
            // RIGHT SHOCK
            if (ps > pR)
            {
                rhoRs = rhoR*(ps/pR+gg)/(gg*ps/pR+1.0);
            }
            // RIGHT FAN
            else
            {
                rhoRs = rhoR*std::pow(ps/pR, 1.0/g);
            }

            rhoI = 0.5*(rhoLs+rhoRs);
            AMREX_D_TERM
            (
                u1I = us;,
                u2I = 0.5*(u2L+u2R);,
                u3I = 0.5*(u3L+u3R);
            )
            pI = ps;
        }

        UI[RHO] = rhoI;
        AMREX_D_TERM
        (
            UI[RHOV1] = rhoI*u1I;,
            UI[RHOV2] = rhoI*u2I;,
            UI[RHOV3] = rhoI*u3I;
        )
        UI[TE] = pI/(g-1.0)+0.5*rhoI*(AMREX_D_TERM(u1I*u1I,+u2I*u2I,+u3I*u3I));
        // ------------------------------------------------------------

        // EVAL FLUX --------------------------------------------------
        this->_eval_F1_(dom, UI, NFn);
        // ------------------------------------------------------------
    }

    // MODIFIED OSHER FLUX
    AMREX_GPU_HOST_DEVICE
    void _eval_NFn_mOsher_(const int dom, const amrex::Real * un, const amrex::Real * U, const amrex::Real * nbr_U,
                           amrex::Real * NFn) const
    {
        // VARIABLES
        amrex::Real Fn[N_U_PER_DOM], nbr_Fn[N_U_PER_DOM];
        amrex::Real Uxi[N_U_PER_DOM];
        amrex::Real absAn[N_U_PER_DOM*N_U_PER_DOM], IntAn[N_U_PER_DOM*N_U_PER_DOM];

        // EVAL LEFT AND RIGHT FLUXES
        this->_eval_Fn_(dom, un, U, Fn);
        this->_eval_Fn_(dom, un, nbr_U, nbr_Fn);

        // EVAL INTEGRAL CONTRIBUTION
        const int qo = 5;

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

            this->_eval_absAn_(dom, un, Uxi, absAn);

            for (int k = 0; k < N_U_PER_DOM*N_U_PER_DOM; ++k)
            {
                IntAn[k] += absAn[k]*w;
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
    // ----------------------------------------------------------------

    // EVAL EIGENVALUES AND EIGENVECTORS OF A1 = dF1dU ----------------
    AMREX_GPU_HOST_DEVICE
    void _eval_A1_eigensystem_(const int dom, const amrex::Real * U, amrex::Real * A1w, amrex::Real * A1v, amrex::Real * iA1v) const
    {
        // VARIABLES
        amrex::Real AMREX_D_DECL(u1, u2, u3), p, a, h, delta;
        this->_eval_enthalpy_variables_(dom, U, AMREX_D_DECL(u1, u2, u3), p, a, h);

#if (AMREX_SPACEDIM == 2)
        delta = 1.0/(u1*u1+u2*u2-2.0*h);
        
        // EIGENVALUES
        A1w[0] = -a+u1;
        A1w[1] = u1;
        A1w[2] = u1;
        A1w[3] = +a+u1;

        // EIGENVECTORS MATRIX (ORDERED PER COLUMN)
        A1v[0+0*N_U_PER_DOM] = 1.0;
        A1v[1+0*N_U_PER_DOM] = -a+u1;
        A1v[2+0*N_U_PER_DOM] = u2;
        A1v[3+0*N_U_PER_DOM] = h-a*u1;

        A1v[0+1*N_U_PER_DOM] = 1.0;
        A1v[1+1*N_U_PER_DOM] = u1;
        A1v[2+1*N_U_PER_DOM] = u2;
        A1v[3+1*N_U_PER_DOM] = 0.5*(u1*u1+u2*u2);

        A1v[0+2*N_U_PER_DOM] = 0.0;
        A1v[1+2*N_U_PER_DOM] = 0.0;
        A1v[2+2*N_U_PER_DOM] = 1.0;
        A1v[3+2*N_U_PER_DOM] = u2;

        A1v[0+3*N_U_PER_DOM] = 1.0;
        A1v[1+3*N_U_PER_DOM] = a+u1;
        A1v[2+3*N_U_PER_DOM] = u2;
        A1v[3+3*N_U_PER_DOM] = h+a*u1;

        // INVERSE OF THE EIGENVECTORS MATRIX
        iA1v[0+0*N_U_PER_DOM] = 0.5*(-1.0-2.0*h*delta+u1/a);
        iA1v[1+0*N_U_PER_DOM] = 2.0+2.0*h*delta;
        iA1v[2+0*N_U_PER_DOM] = -u2;
        iA1v[3+0*N_U_PER_DOM] = -h*delta-0.5*(a+u1)/a;

        iA1v[0+1*N_U_PER_DOM] = -0.5/a+u1*delta;
        iA1v[1+1*N_U_PER_DOM] = -2.0*u1*delta;
        iA1v[2+1*N_U_PER_DOM] = 0.0;
        iA1v[3+1*N_U_PER_DOM] = 0.5/a+u1*delta;

        iA1v[0+2*N_U_PER_DOM] = u2*delta;
        iA1v[1+2*N_U_PER_DOM] = -2.0*u2*delta;
        iA1v[2+2*N_U_PER_DOM] = 1.0;
        iA1v[3+2*N_U_PER_DOM] = u2*delta;

        iA1v[0+3*N_U_PER_DOM] = -1.0*delta;
        iA1v[1+3*N_U_PER_DOM] = 2.0*delta;
        iA1v[2+3*N_U_PER_DOM] = 0.0;
        iA1v[3+3*N_U_PER_DOM] = -1.0*delta;
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "HERE WE ARE - _eval_A1_eigensystem_" << std::endl;
exit(-1);
#endif
    }
    // ----------------------------------------------------------------

    // EVAL EIGENVALUES AND EIGENVECTORS OF An = dFndU = dFidU*ni -----
    AMREX_GPU_HOST_DEVICE
    void _eval_An_eigensystem_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * Anw, amrex::Real * Anv, amrex::Real * iAnv) const
    {
        // PARAMETERS
        const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);

        // VARIABLES
        amrex::Real Un[N_U_PER_DOM];

        // SOLUTION VECTOR IN THE NORMAL REFERENCE SYSTEM
        this->_eval_U2Un_(un, U, Un);

        // EVAL An EIGEN-DECOMPOSITION USING A1(Un)
        this->_eval_A1_eigensystem_(dom, Un, Anw, Anv, iAnv);

#if (AMREX_SPACEDIM == 2)
        // PRE-MULTIPLY Anv TIMES iTn
        for (int c = 0; c < N_U_PER_DOM; ++c)
        {
            const amrex::Real r2 = Anv[1+c*N_U_PER_DOM];
            const amrex::Real r3 = Anv[2+c*N_U_PER_DOM];

            Anv[1+c*N_U_PER_DOM] = r2*n1-r3*n2;
            Anv[2+c*N_U_PER_DOM] = r3*n1+r2*n2;
        }
        // POST-MULTIPLY iAnv TIMES Tn
        for (int r = 0; r < N_U_PER_DOM; ++r)
        {
            const amrex::Real c2 = iAnv[r+1*N_U_PER_DOM];
            const amrex::Real c3 = iAnv[r+2*N_U_PER_DOM];

            iAnv[r+1*N_U_PER_DOM] = c2*n1-c3*n2; iAnv[r+2*N_U_PER_DOM] = c3*n1+c2*n2;
        }  
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "HERE WE ARE - _eval_An_eigensystem_" << std::endl;
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

    // EVAL |An| = Anv*diag(abs(Anw))*iAnv ----------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_absAn_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * absAn) const
    {
        // VARIABLES
        amrex::Real Anw[N_U_PER_DOM], Anv[N_U_PER_DOM*N_U_PER_DOM], iAnv[N_U_PER_DOM*N_U_PER_DOM];

        // EVAL THE An EIGEN-SYSTEM
        this->_eval_An_eigensystem_(dom, un, U, Anw, Anv, iAnv);
        
        for (int c = 0; c < N_U_PER_DOM; ++c)
        for (int r = 0; r < N_U_PER_DOM; ++r)
        {
            Anv[r+c*N_U_PER_DOM] *= std::abs(Anw[c]);
        }

        amrex::DG_utils::matmul(N_U_PER_DOM, N_U_PER_DOM, N_U_PER_DOM, Anv, iAnv, absAn);
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
    bool F_TROUBLED(const int dom, const amrex::Real t, const amrex::Real * x, const amrex::Real * U) const
    {   
        const amrex::Real g = this->gamma[dom];
        
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
        
        return ((rho < 0.0) || (p < 0.0));
    }
    // ================================================================
};
// ####################################################################