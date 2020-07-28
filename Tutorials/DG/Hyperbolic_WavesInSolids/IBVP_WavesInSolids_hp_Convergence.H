/* ELASTIC WAVE EQUATION */

#include <IBVP_WavesInSolids_Base.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS ----------------
#define N_PHI 1
#define N_DOM 2
#define N_U (2*N_VS)
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
#if (AMREX_SPACEDIM == 2)
        const amrex::Real x1 = x[0];
        const amrex::Real x2 = x[1];
        const amrex::Real phi = 0.125-std::cos(2.0*M_PI*x1)*std::cos(2.0*M_PI*x2);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real x1 = x[0];
        const amrex::Real x2 = x[1];
        const amrex::Real x3 = x[2];
        const amrex::Real phi = 0.125+std::cos(2.0*M_PI*x1)*std::cos(2.0*M_PI*x2)*std::cos(2.0*M_PI*x3);
#endif
        return (-phi);
    }
    // ================================================================

    // RELATIONSHIP AMONG DISTANCE FUNCTIONS AND DOMAINS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1,

            // dom = 1
            0, +1
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
            // dom = 0
            0, 0,  0, 0, 0,

            // dom = 1
            1, 1,  1, 1, 1
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0, 0,  0, 0, 0, 0, 0, 0,

            // dom = 1
            1, 1, 1,  1, 1, 1, 1, 1, 1
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
            1,

            // dom = 1
            0
        };
        return NBRDOM[dom];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        const amrex::Real * un = this->ICs_un;
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real f0 = std::sin(2.0*M_PI*x_dot_un);
        const int uu = (u < N_VS) ? u : (u-N_VS);
        const int pos = uu+(this->ICs_iv)*N_VS;

        amrex::Real res;

        res = this->ICs_An_v[pos]*f0;

        return res;
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
        const amrex::Real * un = this->ICs_un;
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real w = this->ICs_An_w[this->ICs_iv];
        const amrex::Real f0 = std::sin(2.0*M_PI*(x_dot_un-w*t));

        // EXACT SOLUTION
        amrex::Real Ue[N_VS];

        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] = this->ICs_An_v[u+(this->ICs_iv)*N_VS]*f0;
        }

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY
        amrex::Real V[N_VS];
        AMREX_D_TERM
        (
            V[0] = this->density[0]*Ue[0];,
            V[1] = this->density[0]*Ue[1];,
            V[2] = this->density[0]*Ue[2];
        )
        amrex::DG_utils::matmul(N_S, N_S, 1, this->sES[0], &Ue[N_V], &V[N_V]);

        // NORMALIZATION QUANTITY
        norm_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            norm_x += Ue[u]*V[u];
        }

        // DETERMINE WHICH DOMAIN WE ARE IN
        const amrex::Real phi = this->F_PHI(0, t, x);

        // DG SOLUTION
        const int pos = (phi < 0.0) ? 0 : N_VS;

        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] -= U[u+pos];
        }

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY OF
        // THE DG SOLUTION
        AMREX_D_TERM
        (
            V[0] = this->density[0]*Ue[0];,
            V[1] = this->density[0]*Ue[1];,
            V[2] = this->density[0]*Ue[2];
        )
        amrex::DG_utils::matmul(N_S, N_S, 1, this->sES[0], &Ue[N_V], &V[N_V]);

        // ERROR
        err_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            err_x += Ue[u]*V[u];
        }
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

#if (AMREX_SPACEDIM == 2)
        avg_U[0] = 0.5*(U[0]+nbr_U[0]);
        avg_U[1] = 0.5*(U[1]+nbr_U[1]);
        avg_U[2] = 0.5*(U[2]+nbr_U[2]);
        avg_U[3] = 0.5*(U[3]+nbr_U[3]);
        avg_U[4] = 0.5*(U[4]+nbr_U[4]);
        
        avg_U[0+N_VS] = 0.5*(U[0+N_VS]+nbr_U[0+N_VS]);
        avg_U[1+N_VS] = 0.5*(U[1+N_VS]+nbr_U[1+N_VS]);
        avg_U[2+N_VS] = 0.5*(U[2+N_VS]+nbr_U[2+N_VS]);
        avg_U[3+N_VS] = 0.5*(U[3+N_VS]+nbr_U[3+N_VS]);
        avg_U[4+N_VS] = 0.5*(U[4+N_VS]+nbr_U[4+N_VS]);

        if (std::abs(un[0]) > 0.5)
        {
            // DOMAIN 0
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[0], avg_U, NFn);

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

            // DOMAIN 1
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[1], &avg_U[1*N_VS], &NFn[1*N_VS]);

            NFn[0+N_VS] *= un[0];
            NFn[1+N_VS] *= un[0];
            NFn[2+N_VS] *= un[0];
            NFn[3+N_VS] *= un[0];
            NFn[4+N_VS] *= un[0];

            NFn[0+N_VS] += 0.5*mu*(U[0+N_VS]-nbr_U[0+N_VS]);
            NFn[1+N_VS] += 0.5*mu*(U[1+N_VS]-nbr_U[1+N_VS]);
            NFn[2+N_VS] += 0.5*mu*(U[2+N_VS]-nbr_U[2+N_VS]);
            NFn[3+N_VS] += 0.5*mu*(U[3+N_VS]-nbr_U[3+N_VS]);
            NFn[4+N_VS] += 0.5*mu*(U[4+N_VS]-nbr_U[4+N_VS]);
        }
        else if (std::abs(un[1]) > 0.5)
        {
            // DOMAIN 0
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2[0], avg_U, NFn);

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

            // DOMAIN 1
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2[1], &avg_U[1*N_VS], &NFn[1*N_VS]);

            NFn[0+N_VS] *= un[1];
            NFn[1+N_VS] *= un[1];
            NFn[2+N_VS] *= un[1];
            NFn[3+N_VS] *= un[1];
            NFn[4+N_VS] *= un[1];

            NFn[0+N_VS] += 0.5*mu*(U[0+N_VS]-nbr_U[0+N_VS]);
            NFn[1+N_VS] += 0.5*mu*(U[1+N_VS]-nbr_U[1+N_VS]);
            NFn[2+N_VS] += 0.5*mu*(U[2+N_VS]-nbr_U[2+N_VS]);
            NFn[3+N_VS] += 0.5*mu*(U[3+N_VS]-nbr_U[3+N_VS]);
            NFn[4+N_VS] += 0.5*mu*(U[4+N_VS]-nbr_U[4+N_VS]);
        }
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "Hello! F_NF_ICS" << std::endl;
exit(-1);
#endif
    }
    // ----------------------------------------------------------------

    // GRID BOUNDARIES ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
amrex::Print() << "Hello! F_NF_BCS (We should not end up here)" << std::endl;
exit(-1);
    }
    // ----------------------------------------------------------------

    // INTERNAL BOUNDARIES --------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
amrex::Print() << "Hello! F_NF_PHI_BCS (We should not end up here)" << std::endl;
exit(-1);
    }
    // ----------------------------------------------------------------

    // INTERNAL INTERFACE ---------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        amrex::Real avg_U[N_U];

amrex::Print() << "Hello! F_NF_PHI_ICS" << std::endl;
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
        point_fields_location.clear();
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL(const int p,
                     const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
                     amrex::Real * F) const
    {
amrex::Print() << "Hello! F_POINT_SOL (We should not end up here)" << std::endl;
exit(-1);
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        fields_domain = {0, 0,  0, 0, 0,
                         1, 1,  1, 1, 1};
        fields_name = {"Vx_a", "Vy_a",  "Sxx_a", "Syy_a", "Sxy_a",
                       "Vx_b", "Vy_b",  "Sxx_b", "Syy_b", "Sxy_b"};
#endif
#if (AMREX_SPACEDIM == 3)
        fields_domain = {0, 0, 0,  0, 0, 0, 0, 0, 0,
                         1, 1, 1,  1, 1, 1, 1, 1, 1};
        fields_name = {"Vx_a", "Vy_a", "Vz_a",  "Sxx_a", "Syy_a", "Szz_a", "Syz_a", "Sxz_a", "Sxy_a",
                       "Vx_b", "Vy_b", "Vz_b",  "Sxx_b", "Syy_b", "Szz_b", "Syz_b", "Sxz_b", "Sxy_b"};
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

        F[0+N_VS] = U[0+N_VS];
        F[1+N_VS] = U[1+N_VS];
        F[2+N_VS] = U[2+N_VS];
        F[3+N_VS] = U[3+N_VS];
        F[4+N_VS] = U[4+N_VS];
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

        F[0+N_VS] = U[0+N_VS];
        F[1+N_VS] = U[1+N_VS];
        F[2+N_VS] = U[2+N_VS];
        F[3+N_VS] = U[3+N_VS];
        F[4+N_VS] = U[4+N_VS];
        F[5+N_VS] = U[5+N_VS];
        F[6+N_VS] = U[6+N_VS];
        F[7+N_VS] = U[7+N_VS];
        F[8+N_VS] = U[8+N_VS];
#endif
    }
    // ================================================================

};