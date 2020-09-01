#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>

#include <AMReX_BC_TYPES.H>

#include <AMReX_Print.H>

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving the following two-dimensional
// hyperbolic system
//
// U_{,t}+A1*U_{,x1}+A2*U_{,x2} = g(t,x) for (t,x) in [0,1]x[0,1]^2
//
// where
//
// A1 = [ 1  1 -1], A2 = [-6  0 -8]
//      [ 1  1  2]       [ 1 -4  7]
//      [ 1  1  2]       [ 0  0  2]
//
// and g(t,x), ICS and BCS are chosen such that the exact solution is
//
// U(t,x) = sin(2*pi*(x1+x2-2*t))*[1]
//                                [1]
//                                [1]
//
// ####################################################################



// IBVP CLASS #########################################################
#define N_PHI 1
#define N_DOM 1
#define N_U 3

class LINEAR_SYSTEM
{
public:
    // DATA MEMBERS ===================================================
    const amrex::Real A1[N_U*N_U] = { 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0,
                                     -1.0, 2.0, 2.0};
    const amrex::Real A2[N_U*N_U] = {-6.0,  1.0, 0.0,
                                      0.0, -4.0, 0.0,
                                     -8.0,  7.0, 2.0};
    const amrex::Real max_wave_speed = 6.102135494996343;
    /*
    const amrex::Real A1[N_U*N_U] = { 1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0};
    const amrex::Real A2[N_U*N_U] = { 1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0};
    const amrex::Real max_wave_speed = std::sqrt(2.0);
    */

    amrex::Real A1_w[N_U], A1_v[N_U*N_U], iA1_v[N_U*N_U], A1p[N_U*N_U], A1m[N_U*N_U];
    amrex::Real A2_w[N_U], A2_v[N_U*N_U], iA2_v[N_U*N_U], A2p[N_U*N_U], A2m[N_U*N_U];
    // ================================================================

    // CONSTRUCTOR ====================================================
    LINEAR_SYSTEM()
    {
        amrex::Real wIm[N_U], diag_w[N_U*N_U], tmp[N_U*N_U];
        std::fill(diag_w, diag_w+N_U*N_U, 0.0);

        // A1
        amrex::DG_utils::eig(false, true, N_U, this->A1, this->A1_w, wIm, nullptr, this->A1_v);
        amrex::DG_utils::matinv(N_U, this->A1_v, this->iA1_v);

        for (int u = 0; u < N_U; ++u)
        {
            diag_w[u+u*N_U] = std::max(this->A1_w[u], 0.0);
        }
        amrex::DG_utils::matmul(N_U, N_U, N_U, this->A1_v, diag_w, tmp);
        amrex::DG_utils::matmul(N_U, N_U, N_U, tmp, this->iA1_v, this->A1p);

        for (int u = 0; u < N_U; ++u)
        {
            diag_w[u+u*N_U] = std::min(this->A1_w[u], 0.0);
        }
        amrex::DG_utils::matmul(N_U, N_U, N_U, this->A1_v, diag_w, tmp);
        amrex::DG_utils::matmul(N_U, N_U, N_U, tmp, this->iA1_v, this->A1m);

        // A2
        amrex::DG_utils::eig(false, true, N_U, this->A2, this->A2_w, wIm, nullptr, this->A2_v);
        amrex::DG_utils::matinv(N_U, this->A2_v, this->iA2_v);

        for (int u = 0; u < N_U; ++u)
        {
            diag_w[u+u*N_U] = std::max(this->A2_w[u], 0.0);
        }
        amrex::DG_utils::matmul(N_U, N_U, N_U, this->A2_v, diag_w, tmp);
        amrex::DG_utils::matmul(N_U, N_U, N_U, tmp, this->iA2_v, this->A2p);

        for (int u = 0; u < N_U; ++u)
        {
            diag_w[u+u*N_U] = std::min(this->A2_w[u], 0.0);
        }
        amrex::DG_utils::matmul(N_U, N_U, N_U, this->A2_v, diag_w, tmp);
        amrex::DG_utils::matmul(N_U, N_U, N_U, tmp, this->iA2_v, this->A2m);
    }
    // ================================================================

    // LEVEL SET FUNCTION =============================================
#define DELTA 0.05
#define R 0.25

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        //return -1.0;
        //return (DELTA-x[0])*(1.0-DELTA-x[0]);
        return R*R-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5);
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
        return 1.0;
    }

    amrex::Real F_EXACT_SURFACE(const int & dom) const
    {
        return 0.0;
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
        return 0;
    }
    // ================================================================

    // RELATIONSHIP BETWEEN NEIGHBORING DOMAINS =======================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {
        return -1;
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    AMREX_GPU_HOST_DEVICE
    void _eval_U_exact_(const amrex::Real & t, const amrex::Real * x,
                        amrex::Real * Ue) const
    {
        Ue[0] = std::sin(2.0*M_PI*(x[0]+x[1]-2.0*t));
        Ue[1] = std::sin(2.0*M_PI*(x[0]+x[1]-2.0*t));
        Ue[2] = std::sin(2.0*M_PI*(x[0]+x[1]-2.0*t));
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_dUdt_exact_(const amrex::Real & t, const amrex::Real * x,
                           amrex::Real * dUedt) const
    {
        dUedt[0] = -4.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedt[1] = -4.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedt[2] = -4.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_dUdx1_exact_(const amrex::Real & t, const amrex::Real * x,
                            amrex::Real * dUedx1) const
    {
        dUedx1[0] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedx1[1] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedx1[2] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_dUdx2_exact_(const amrex::Real & t, const amrex::Real * x,
                            amrex::Real * dUedx2) const
    {
        dUedx2[0] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedx2[1] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
        dUedx2[2] = 2.0*M_PI*std::cos(2.0*M_PI*(x[0]+x[1]-2.0*t));
    }

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const int dom,
                 const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
        // PARAMETERS
        const amrex::Real * Ud = &U[dom*N_U];

        // WE DON'T USE A NORMALIZATION
        norm_x = 1.0;

        // EXACT SOLUTION
        amrex::Real Ue[N_U];
        this->_eval_U_exact_(t, x, Ue);

        // ERROR
        Ue[0] -= Ud[0];
        Ue[1] -= Ud[1];
        Ue[2] -= Ud[2];
        err_x = Ue[0]*Ue[0];
    }
    // ================================================================

    // PDES SYSTEM ====================================================
    AMREX_GPU_HOST_DEVICE
    void _eval_F1_(const int dom, const amrex::Real * U, amrex::Real * F1) const
    {
        F1[0] = this->A1[0+0*3]*U[0]+this->A1[0+1*3]*U[1]+this->A1[0+2*3]*U[2];
        F1[1] = this->A1[1+0*3]*U[0]+this->A1[1+1*3]*U[1]+this->A1[1+2*3]*U[2];
        F1[2] = this->A1[2+0*3]*U[0]+this->A1[2+1*3]*U[1]+this->A1[2+2*3]*U[2];
    }
    AMREX_GPU_HOST_DEVICE
    void _eval_F2_(const int dom, const amrex::Real * U, amrex::Real * F2) const
    {
        F2[0] = this->A2[0+0*3]*U[0]+this->A2[0+1*3]*U[1]+this->A2[0+2*3]*U[2];
        F2[1] = this->A2[1+0*3]*U[0]+this->A2[1+1*3]*U[1]+this->A2[1+2*3]*U[2];
        F2[2] = this->A2[2+0*3]*U[0]+this->A2[2+1*3]*U[1]+this->A2[2+2*3]*U[2];
    }

    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        this->_eval_F1_(0, U, F1);
        this->_eval_F2_(0, U, F2);
    }
    // ================================================================

    // BODY LOAD ======================================================
    AMREX_GPU_HOST_DEVICE
    void F_B(const amrex::Real t, const amrex::Real * x,
             amrex::Real * B) const
    {
        // EXACT SOLUTION'S TIME/SPACE DERIVATIVES
        amrex::Real dUedt[N_U], dUedx1[N_U], dUedx2[N_U];
        this->_eval_dUdt_exact_(t, x, dUedt);
        this->_eval_dUdx1_exact_(t, x, dUedx1);
        this->_eval_dUdx2_exact_(t, x, dUedx2);

        B[0]  = dUedt[0];
        B[0] += this->A1[0+0*3]*dUedx1[0]+this->A1[0+1*3]*dUedx1[1]+this->A1[0+2*3]*dUedx1[2];
        B[0] += this->A2[0+0*3]*dUedx2[0]+this->A2[0+1*3]*dUedx2[1]+this->A2[0+2*3]*dUedx2[2];

        B[1]  = dUedt[1];
        B[1] += this->A1[1+0*3]*dUedx1[0]+this->A1[1+1*3]*dUedx1[1]+this->A1[1+2*3]*dUedx1[2];
        B[1] += this->A2[1+0*3]*dUedx2[0]+this->A2[1+1*3]*dUedx2[1]+this->A2[1+2*3]*dUedx2[2];

        B[2]  = dUedt[2];
        B[2] += this->A1[2+0*3]*dUedx1[0]+this->A1[2+1*3]*dUedx1[1]+this->A1[2+2*3]*dUedx1[2];
        B[2] += this->A2[2+0*3]*dUedx2[0]+this->A2[2+1*3]*dUedx2[1]+this->A2[2+2*3]*dUedx2[2];
    }
    // ================================================================

    // INITIAL CONDITIONS =============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        // EXACT SOLUTION AT t = 0
        amrex::Real Ue[N_U];
        this->_eval_U_exact_(0.0, x, Ue);

        return Ue[u];
    }
    // ================================================================

    // FOR POST-PROCESSING ============================================
    void F_POINT_SOL_DESCRIPTION(amrex::Vector<amrex::Array<amrex::Real, AMREX_SPACEDIM>> & point_fields_location,
                                 amrex::Vector<int> & point_fields_domain,
                                 amrex::Vector<amrex::Vector<std::string>> & point_fields_name) const
    {
        point_fields_location.clear();
    }

    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL(const int p,
                     const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
                     amrex::Real * F) const
    {
amrex::Abort("Hello! LINEAR_SYSTEM.F_POINT_SOL (We should not end up in here)");
    }

    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
        fields_domain = {0, 0, 0,
                         0, 0, 0};
        fields_name = {"U1", "U2", "U3",
                       "err_U1", "err_U2", "err_U3"};
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // VARIABLES
        amrex::Real Ue[N_U];
        
        // EXACT SOLUTION
        this->_eval_U_exact_(t, x, Ue);

        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];

        F[0+N_U] = U[0]-Ue[0];
        F[1+N_U] = U[1]-Ue[1];
        F[2+N_U] = U[2]-Ue[2];
    }
    // ================================================================

    // TIME STEP ======================================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const amrex::Real * dx,
                     const amrex::Real & t, const amrex::Real * x,
                     const amrex::Real * U) const
    {
        amrex::Real dt;

#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
        dt = h/(this->max_wave_speed);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
        dt = h/(this->max_wave_speed);
#endif
        return dt;
    }
    // ================================================================

    // NUMERICAL FLUXES: RIEMANN SOLVER/LAX-FRIEDRICHS FLUX ===========
    AMREX_GPU_HOST_DEVICE
    void _eval_AnU_(const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU) const
    {
        for (int r = 0; r < N_U; ++r)
        {
            AnU[r] = (this->A1[r+0*N_U]*un[0]+this->A2[r+0*N_U]*un[1])*U[0];
            for (int c = 1; c < N_U; ++c)
            {
                AnU[r] += (this->A1[r+c*N_U]*un[0]+this->A2[r+c*N_U]*un[1])*U[c];
            }
        }
    }
    
    AMREX_GPU_HOST_DEVICE
    void _NF_RiemannSolver_base_(const amrex::Real * un,
                                 const amrex::Real * Up, const amrex::Real * Um,
                                 amrex::Real * NFn) const
    {
        amrex::Real tmp[N_U];

        if (un[0] > 0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1p, Up, tmp);
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1m, Um, NFn);
            NFn[0] += tmp[0];
            NFn[1] += tmp[1];
            NFn[2] += tmp[2];
        }
        else if (un[0] < -0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1m, Up, tmp);
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1p, Um, NFn);
            NFn[0] = -NFn[0]-tmp[0];
            NFn[1] = -NFn[1]-tmp[1];
            NFn[2] = -NFn[2]-tmp[2];
        }
        else if (un[1] > 0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2p, Up, tmp);
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2m, Um, NFn);
            NFn[0] += tmp[0];
            NFn[1] += tmp[1];
            NFn[2] += tmp[2];
        }
        else if (un[1] < -0.5)
        {
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2m, Up, tmp);
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2p, Um, NFn);
            NFn[0] = -NFn[0]-tmp[0];
            NFn[1] = -NFn[1]-tmp[1];
            NFn[2] = -NFn[2]-tmp[2];
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _NF_LF_(const amrex::Real * un,
                 const amrex::Real * Up, const amrex::Real * Um,
                 amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->max_wave_speed;

        // VARIABLES
        amrex::Real avg_U[N_U];

        avg_U[0] = 0.5*(Up[0]+Um[0]);
        avg_U[1] = 0.5*(Up[1]+Um[1]);
        avg_U[2] = 0.5*(Up[2]+Um[2]);

        this->_eval_AnU_(un, avg_U, NFn);

        NFn[0] += 0.5*mu*(Up[0]-Um[0]);
        NFn[1] += 0.5*mu*(Up[1]-Um[1]);
        NFn[2] += 0.5*mu*(Up[2]-Um[2]);
    }
    // ================================================================
    
    // NUMERICAL FLUXES: INTRAPHASE ===================================
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real * Ud = &U[dom*N_U], * nbr_Ud = &nbr_U[dom*N_U];

        // VARIABLES
        amrex::Real * NFnd = &NFn[dom*N_U];

        this->_NF_LF_(un, Ud, nbr_Ud, NFnd);
    }
    // ================================================================

    // NUMERICAL FLUXES: GRID BOUNDARIES ==============================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int & dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real * Ud = &U[dom*N_U];

        // VARIABLES
        amrex::Real Ue[N_U];
        amrex::Real * NFnd = &NFn[dom*N_U];

        // EXACT SOLUTION
        this->_eval_U_exact_(t, x, Ue);

        this->_NF_LF_(un, Ud, Ue, NFnd);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL BOUNDARIES ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real * Ud = &U[dom*N_U];

        // VARIABLES
        amrex::Real Ue[N_U];
        amrex::Real * NFnd = &NFn[dom*N_U];

        // EXACT SOLUTION
        this->_eval_U_exact_(t, x, Ue);

        this->_NF_LF_(un, Ud, Ue, NFnd);
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL INTERFACE ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int dom, const int nbr_dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_PHI_ICS");
    }
    // ================================================================
};
// ####################################################################



// ACTUAL MAIN PROGRAM ################################################
void main_main()
{
    // HEADING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# AMREX & DG PROJECT                                                   " << std::endl;
amrex::Print() << "# Author: Vincenzo Gulizzi (vgulizzi@lbl.gov)                          " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# SUMMARY:                                                             " << std::endl;
amrex::Print() << "# In this tutorial, we are solving the following two-dimensional       " << std::endl;
amrex::Print() << "# hyperbolic system                                                    " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# U_{,t}+A1*U_{,x1}+A2*U_{,x2} = g(t,x) for (t,x) in [0,1]x[0,1]^2     " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    // ================================================================

    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;

    // INPUTS
    amrex::DG::InputReader inputs;
    // ================================================================

    // READ INPUTS FOR HP-CONVERGENCE ANALYSIS ========================
    amrex::Vector<int> hp_p;
    amrex::Vector<amrex::Vector<int>> hp_Ne;
    amrex::ParmParse pp_hp("hp");

    pp_hp.getarr("p", hp_p);
    
    for (size_t ip = 0; ip < hp_p.size(); ++ip)
    {
        const int p = hp_p[ip];
        const std::string s = "Ne["+std::to_string(p)+"]";
        amrex::Vector<int> Ne;

        pp_hp.getarr(s.c_str(), Ne);
        hp_Ne.push_back(Ne);
    }
    // ================================================================

    // TIC ============================================================
    start_time = amrex::second();
    // ================================================================

    // OPEN FILE WHERE TO WRITE THE HP-CONVERGENCE RESULTS ============
    std::ofstream fp;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        time_t date_and_time = time(0);
        char * date_and_time_ = ctime(&date_and_time);

        fp.open("hp-Convergence.txt", std::ofstream::app);
        fp << std::endl << "HP-CONVERGENCE ANALYSIS: " << date_and_time_ << "\n";
    }
    amrex::ParallelDescriptor::Barrier();
    // ================================================================

    // PERFORM THE HP-CONVERGENCE ANALYSIS ============================
    for (int ip = 0; ip < hp_p.size(); ++ip)
    {
        // PHYSICAL BOX -----------------------------------------------
        amrex::RealBox real_box(inputs.space[0].prob_lo.data(),
                                inputs.space[0].prob_hi.data());
        // ------------------------------------------------------------

        // MODIFY INPUTS ----------------------------------------------
        inputs.dG[0].phi_space_p = 3;
        
        inputs.dG[0].space_p = hp_p[ip];
        inputs.dG[0].space_q = std::max(inputs.dG[0].phi_space_p+2, inputs.dG[0].space_p+2);
        inputs.dG[0].time_p = inputs.dG[0].space_p;

        inputs.dG[0].space_q_im = 3;
        // ------------------------------------------------------------

        // WRITE TO FILE ----------------------------------------------
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            fp << "p = " << inputs.dG[0].space_p << "\n";
        }
        // ------------------------------------------------------------

        for (int iNe = 0; iNe < hp_Ne[ip].size(); ++iNe)
        {
            // MODIFY INPUTS ------------------------------------------
            const int Ne = hp_Ne[ip][iNe];
            
            AMREX_D_TERM
            (
                inputs.mesh[0].n_cells[0] = Ne;,
                inputs.mesh[0].n_cells[1] = Ne;,
                inputs.mesh[0].n_cells[2] = Ne;
            )
            // --------------------------------------------------------

            // PRINT TO SCREEN ----------------------------------------
            amrex::Print() << std::endl;
            inputs.PrintSummary();
            // --------------------------------------------------------

            // BOX ARRAY ----------------------------------------------
            amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, inputs.mesh[0].n_cells-1);

            amrex::BoxArray ba;
            ba.define(indices_box);
            ba.maxSize(inputs.mesh[0].max_grid_size);
            // --------------------------------------------------------

            // GEOMETRY -----------------------------------------------
            amrex::Geometry geom;
            geom.define(indices_box, &real_box, inputs.space[0].coord_sys, inputs.space[0].is_periodic.data());
            // --------------------------------------------------------

            // BOXES DISTRIBUTION AMONG MPI PROCESSES -----------------
            amrex::DistributionMapping dm(ba);
            // --------------------------------------------------------

            // INIT DG DATA STRUCTURES --------------------------------
            amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, inputs.dG[0]);
            amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, inputs.dG[0]);
            amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta", inputs);
            dG.InitData(iGeom, MatFactory);
            // --------------------------------------------------------

            // DESTINATION FOLDER AND OUTPUT DATA ---------------------
            std::string dst_folder;
    
            const std::string dG_order = "p"+std::to_string(inputs.dG[0].space_p);
            const std::string dG_mesh = AMREX_D_TERM(std::to_string(inputs.mesh[0].n_cells[0]),+"x"+
                                                     std::to_string(inputs.mesh[0].n_cells[1]),+"x"+
                                                     std::to_string(inputs.mesh[0].n_cells[2]));

            dst_folder = "./IBVP_"+std::to_string(AMREX_SPACEDIM)+"d/"+"m"+dG_mesh+"_"+dG_order+"/";

            if (amrex::ParallelDescriptor::IOProcessor())
            {
                if (!amrex::UtilCreateDirectory(dst_folder, 0755))
                {
                    amrex::CreateDirectoryFailed(dst_folder);
                }
            }
            amrex::ParallelDescriptor::Barrier();
            // --------------------------------------------------------

            // INIT IBVP DATA STRUCTURE -------------------------------
            LINEAR_SYSTEM LS;
            // --------------------------------------------------------

            // INIT OUTPUT DATA INFORMATION ---------------------------
            dG.SetOutput(dst_folder, "PointSolution", iGeom, LS);
            // --------------------------------------------------------

            // INIT FIELDS' DATA WITH INITIAL CONDITIONS --------------
            iGeom.ProjectLevelsetFunctions(LS);
            iGeom.EvalImplicitMesh(LS);
            MatFactory.Eval(iGeom);
            dG.SetICs(iGeom, MatFactory, LS);
            
            // WRITE TO OUTPUT
            dG.PrintPointSolution(0, 0.0, iGeom, MatFactory, LS);

            if (inputs.plot_int > 0)
            {
                int n = 0;
                amrex::Real time = 0.0;

                iGeom.Export_VTK_Mesh(dst_folder, "Mesh", n, inputs.time.n_steps);
                dG.Export_VTK(dst_folder, "Solution", n, inputs.time.n_steps, time, iGeom, MatFactory, LS);
            }
            // --------------------------------------------------------

            // START THE ANALYSIS (ADVANCE IN TIME) -------------------
            amrex::Print() << "# START OF THE ANALYSIS                                                " << std::endl;

            // VARIABLES
            int n;
            amrex::Real time, dt;
            amrex::Real start_clock_time_per_step, stop_clock_time_per_step, clock_time_per_time_step;

            // INIT CLOCK TIME PER STEP
            clock_time_per_time_step = 0.0;

            // ADVANCE IN TIME
            n = 0;
            time = 0.0;
            while ((time < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                start_clock_time_per_step = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, LS);
                dt = std::min(time+dt, inputs.time.T)-time;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1;
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << ", dt = " << dt << ", time = " << time+dt
                               << ", clock time per time step = " << clock_time_per_time_step << std::endl;

                // TIME STEP
                dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, LS);

                // UPDATE TIME AND STEP
                n += 1;
                time += dt;
                
                // COMPUTE ERROR
                if (std::abs(time/inputs.time.T-1.0) < 1.0e-12)
                {
                    amrex::Real err;
                    err = dG.EvalError(time, iGeom, MatFactory, LS);
                    amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << std::sqrt(err) << std::endl;

                    // WRITE TO FILE
                    if (amrex::ParallelDescriptor::IOProcessor())
                    {
                        fp << "Ne = " << Ne << ", error = " << std::scientific << std::setprecision(5) << std::setw(12) << std::sqrt(err);
                    }
                }

                // WRITE TO OUTPUT
                dG.PrintPointSolution(n, time, iGeom, MatFactory, LS);

                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(time/inputs.time.T-1.0) < 1.0e-12)))
                {
                    dG.Export_VTK(dst_folder, "Solution", n, inputs.time.n_steps, time, iGeom, MatFactory, LS);
                }

                // CLOCK TIME PER TIME STEP TOC
                stop_clock_time_per_step = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(stop_clock_time_per_step, IOProc);

                clock_time_per_time_step = (clock_time_per_time_step*n+(stop_clock_time_per_step-start_clock_time_per_step))/(n+1);

                // WRITE TO FILE
                if ((std::abs(time/inputs.time.T-1.0) < 1.0e-12) && (amrex::ParallelDescriptor::IOProcessor()))
                {
                    fp << ", clock time per time step = " << std::scientific << std::setprecision(5) << std::setw(12) << clock_time_per_time_step << "\n";
                }
            }

            // END OF ANALYSIS
            amrex::Print() << "# END OF THE ANALYSIS                                                  " << std::endl;
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // CLOSE FILE =====================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        fp.close();
    }
    // ================================================================

    // TOC ============================================================
    stop_time = amrex::second();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
    // ================================================================

    // CLOSING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "# Time = " << std::scientific << std::setprecision(5) << std::setw(12) << (stop_time-start_time) << " s" << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

}
// ####################################################################


// DUMMY MAIN #########################################################
int main(int argc, char* argv[])
{
    // INIT AMREX ================
    amrex::Initialize(argc, argv);
    // ===========================

    main_main();

    // END AMREX =====
    amrex::Finalize();
    // ===============

    return 0;
}
// ####################################################################
