/* ELASTIC WAVE EQUATION */

#include <IBVP_WavesInSolids_Base.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS ----------------
#define N_PHI 1
#define N_DOM 1
#define N_U N_VS
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
class ELASTIC_WAVES
:
public ELASTIC_WAVES_BASE<N_DOM>
{
private:
    // DATA MEMBERS ===================================================
    // EIGEN STATE INITIAL CONDITIONS
    const int ICs_dir, ICs_iv;
    amrex::Real ICs_un[AMREX_SPACEDIM];
    amrex::Real * ICs_An_w;
    amrex::Real * ICs_An_v;
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const amrex::Vector<std::string> & material_type,
                  const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    ELASTIC_WAVES_BASE(material_type, material_properties),
    ICs_dir{0},
    ICs_iv{AMREX_D_PICK(-1, 3, 4)}
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------

        // EIGEN STATE INITIAL CONDITIONS -----------------------------
        if (this->ICs_dir == 0)
        {
            AMREX_D_TERM
            (
                this->ICs_un[0] = 1.0;,
                this->ICs_un[1] = 0.0;,
                this->ICs_un[2] = 0.0;
            )
            this->ICs_An_w = this->A1_w[0];
            this->ICs_An_v = this->A1_v[0];
        }
        else if (this->ICs_dir == 1)
        {
            AMREX_D_TERM
            (
                this->ICs_un[0] = 0.0;,
                this->ICs_un[1] = 1.0;,
                this->ICs_un[2] = 0.0;
            )
            this->ICs_An_w = this->A2_w[0];
            this->ICs_An_v = this->A2_v[0];
        }
#if (AMREX_SPACEDIM == 3)
        else if (this->ICs_dir == 2)
        {
            AMREX_D_TERM
            (
                this->ICs_un[0] = 0.0;,
                this->ICs_un[1] = 0.0;,
                this->ICs_un[2] = 1.0;
            )
            this->ICs_An_w = this->A3_w[0];
            this->ICs_An_v = this->A3_v[0];
        }
#endif
        // ------------------------------------------------------------
    }
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        const amrex::Real f1 = std::abs(std::sin(M_PI*(x[0]-x[1])))-0.6;
        const amrex::Real f2 = std::abs(std::sin(M_PI*(x[0]+x[1])))-0.6;
        const amrex::Real f3 = (7.5-2.25-x[0])*(x[0]-7.5-2.25);
        const amrex::Real alpha = 2.0, beta = 0.2;

        amrex::Real phi;

        phi = 1.0;
        phi -= std::pow(std::max(0.0, 1.0-f1/beta), alpha);
        phi -= std::pow(std::max(0.0, 1.0-f2/beta), alpha);
        phi -= std::pow(std::max(0.0, 1.0-f3/beta), alpha);
        
        return phi;
    }
    // ================================================================

    // RELATIONSHIP AMONG DISTANCE FUNCTIONS AND DOMAINS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
        const int PHI_INFO[2*N_DOM] =
        {
            0, -1
        };

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
#if (AMREX_SPACEDIM == 2)
        const int DOM[N_U] =
        {
            0, 0, 0, 0, 0
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0
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
            -1
        };
        return NBRDOM[dom];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        const amrex::Real * un = this->ICs_un;
        const amrex::Real xi[AMREX_SPACEDIM] = {x[0]-2.0, x[1]};
        const amrex::Real xi_dot_un = AMREX_D_PICK(xi[0]*un[0], xi[0]*un[0]+xi[1]*un[1], xi[0]*un[0]+xi[1]*un[1]+xi[2]*un[2]);
        const amrex::Real f0 = std::exp(-16.0*xi_dot_un*xi_dot_un);
        const int pos = u+(this->ICs_iv)*N_VS;
        const amrex::Real res = this->ICs_An_v[pos]*f0;

        return res;
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUXES ===============
    // INTRAPHASE -----------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        amrex::Real avg_U[N_U];

        avg_U[0] = 0.5*(U[0]+nbr_U[0]);
        avg_U[1] = 0.5*(U[1]+nbr_U[1]);
        avg_U[2] = 0.5*(U[2]+nbr_U[2]);
        avg_U[3] = 0.5*(U[3]+nbr_U[3]);
        avg_U[4] = 0.5*(U[4]+nbr_U[4]);

        if (std::abs(un[0]) > 0.5)
        {
            // DOMAIN 0
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A1[0], avg_U, NFn);

            NFn[0] *= un[0];
            NFn[1] *= un[0];
            NFn[2] *= un[0];
            NFn[3] *= un[0];
            NFn[4] *= un[0];

            NFn[0] += 0.5*mu*(U[0]-nbr_U[0]);
            NFn[1] += 0.5*mu*(U[1]-nbr_U[1]);
            NFn[2] += 0.5*mu*(U[2]-nbr_U[2]);
            NFn[3] += 0.5*mu*(U[3]-nbr_U[3]);
            NFn[4] += 0.5*mu*(U[4]-nbr_U[4]);
        }
        else if (std::abs(un[1]) > 0.5)
        {
            // DOMAIN 0
            amrex::DG_utils::matmul(N_U, N_U, 1, this->A2[0], avg_U, NFn);

            NFn[0] *= un[1];
            NFn[1] *= un[1];
            NFn[2] *= un[1];
            NFn[3] *= un[1];
            NFn[4] *= un[1];

            NFn[0] += 0.5*mu*(U[0]-nbr_U[0]);
            NFn[1] += 0.5*mu*(U[1]-nbr_U[1]);
            NFn[2] += 0.5*mu*(U[2]-nbr_U[2]);
            NFn[3] += 0.5*mu*(U[3]-nbr_U[3]);
            NFn[4] += 0.5*mu*(U[4]-nbr_U[4]);
        }
    }
    // ----------------------------------------------------------------

    // GRID BOUNDARIES ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        amrex::Real aux_U[N_VS], bcs_U[N_VS];

        // ABSORBING BCS
        if (std::abs(un[0]) > +0.5)
        {
            const int dom = 0;

            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->iA1_v[dom], U, aux_U);
            for (int u = 0; u < N_VS; ++u)
            {
                if (this->A1_w[dom][u]*un[0] < 0.0) aux_U[u] = 0.0;
            }
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1_v[dom], aux_U, bcs_U);

            aux_U[0] = 0.5*(U[0]+bcs_U[0]);
            aux_U[1] = 0.5*(U[1]+bcs_U[1]);
            aux_U[2] = 0.5*(U[2]+bcs_U[2]);
            aux_U[3] = 0.5*(U[3]+bcs_U[3]);
            aux_U[4] = 0.5*(U[4]+bcs_U[4]);
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], aux_U, NFn);

            NFn[0] *= un[0];
            NFn[1] *= un[0];
            NFn[2] *= un[0];
            NFn[3] *= un[0];
            NFn[4] *= un[0];

            NFn[0] += 0.5*mu*(U[0]-bcs_U[0]);
            NFn[1] += 0.5*mu*(U[1]-bcs_U[1]);
            NFn[2] += 0.5*mu*(U[2]-bcs_U[2]);
            NFn[3] += 0.5*mu*(U[3]-bcs_U[3]);
            NFn[4] += 0.5*mu*(U[4]-bcs_U[4]);
        }
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "Hello! F_NF_BCS" << std::endl;
exit(-1);
#endif
    }
    // ----------------------------------------------------------------

    // INTERNAL BOUNDARIES --------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
const amrex::Real mu = this->max_wave_speed;

        amrex::Real T[N_VS*N_VS], iT[N_VS*N_VS];
        this->_eval_T_(un, T, iT);

        amrex::Real bcs_U[N_VS], Un[N_VS];
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(bcs_F1[N_U], bcs_F2[N_U], bcs_F3[N_U]);

        amrex::DG_utils::matmul(N_VS, N_VS, 1, T, U, Un);

        // PRESCRIBED ZERO TRACTIONS
#if (AMREX_SPACEDIM == 2)
        Un[2] = -Un[2]; // Snn
        Un[4] = -Un[4]; // Sns
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "Hello! F_NF_PHI_BCS" << std::endl;
exit(-1);
#endif

        amrex::DG_utils::matmul(N_VS, N_VS, 1, iT, Un, bcs_U);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, bcs_U, AMREX_D_DECL(bcs_F1, bcs_F2, bcs_F3));

        for (int u = 0; u < N_U; ++u)
        {
            AMREX_D_TERM
            (
                NF1[u] = 0.5*(F1[u]+bcs_F1[u])+0.5*mu*(U[u]-bcs_U[u])*un[0];,
                NF2[u] = 0.5*(F2[u]+bcs_F2[u])+0.5*mu*(U[u]-bcs_U[u])*un[1];,
                NF3[u] = 0.5*(F3[u]+bcs_F3[u])+0.5*mu*(U[u]-bcs_U[u])*un[2];
            )
        }
    }
    // ----------------------------------------------------------------

    // INTERNAL INTERFACE --------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
amrex::Print() << "Hello! F_NF_PHI_ICS (We should not end up here)" << std::endl;
exit(-1);
    }
    // ----------------------------------------------------------------
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL_DESCRIPTION(amrex::Vector<amrex::Array<amrex::Real, AMREX_SPACEDIM>> & point_fields_location,
                                 amrex::Vector<int> & point_fields_domain,
                                 amrex::Vector<amrex::Vector<std::string>> & point_fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        point_fields_location = 
        {
            {0.0, 0.5},
            {3.0, 0.5},
            {15.0, 0.5}
        };
        point_fields_domain =
        {
            0,
            0,
            0
        };
        point_fields_name =
        {
            {"Vx", "Vy", "Sxx", "Syy", "Sxy"},
            {"Vx", "Vy", "Sxx", "Syy", "Sxy"},
            {"Vx", "Vy", "Sxx", "Syy", "Sxy"}
        };
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL(const int p,
                     const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
                     amrex::Real * F) const
    {
#if (AMREX_SPACEDIM == 2)
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        fields_domain = {0, 0, 0, 0, 0};
        fields_name = {"Vx", "Vy", "Sxx", "Syy", "Sxy"};
#endif
#if (AMREX_SPACEDIM == 3)
        fields_domain = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        fields_name = {"Vx", "Vy", "Vz", "Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy"};
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
#if (AMREX_SPACEDIM == 2)
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
#endif
#if (AMREX_SPACEDIM == 3)
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
        F[5] = U[5];
        F[6] = U[6];
        F[7] = U[7];
        F[8] = U[8];
#endif
    }
    // ================================================================
};
// ####################################################################