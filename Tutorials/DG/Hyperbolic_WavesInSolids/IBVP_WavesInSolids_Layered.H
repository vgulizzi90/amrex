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
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const amrex::Vector<std::string> & material_type,
                  const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    ELASTIC_WAVES_BASE(material_type, material_properties)
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------
    }
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        amrex::Real res;
        res  = -(x[0]-5.0)*(x[0]-5.8);
        res *= +(x[0]-6.0)*(x[0]-6.8);
        res *= +(x[0]-7.0)*(x[0]-7.8);
        res *= +(x[0]-8.0)*(x[0]-8.8);
        res *= +(x[0]-9.0)*(x[0]-9.8);

        return res;
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
            0, 0, 0, 0, 0,

            // dom = 1
            1, 1, 1, 1, 1
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
        const amrex::Real res = 0.0;
        
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
        if (un[0] > +0.5)
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
        else if (un[0] < -0.5)
        {
            const int dom = 0;

            const amrex::Real fc = 1.4;
            const amrex::Real v1 = std::sin(2.0*M_PI*(0.5*fc*t*t))*std::exp(-t*t);
            const amrex::Real v2 = 0.0;

            bcs_U[0] = v1;
            bcs_U[1] = v2;
            bcs_U[2] = U[2];
            bcs_U[3] = U[3];
            bcs_U[4] = U[4];

            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], bcs_U, &NFn[dom*N_VS]);

            NFn[0] *= un[0];
            NFn[1] *= un[0];
            NFn[2] *= un[0];
            NFn[3] *= un[0];
            NFn[4] *= un[0];

            NFn[0] += 0.5*mu*(U[0]-bcs_U[0]);
            NFn[1] += 0.5*mu*(U[1]-bcs_U[1]);
        }
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
    // N.B.: We assume that the interfaces are straight lines with unit
    //       normal un = {+1,0} or {-1,0}.
    // ----------------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        amrex::Real Fn[N_VS], nbr_Fn[N_VS];

        if (dom == 0)
        {
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], &U[dom*N_VS], Fn);
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], &nbr_U[(1-dom)*N_VS], nbr_Fn);

            NFn[0] = 0.5*(Fn[0]+nbr_Fn[0])*un[0]+0.5*mu*(U[0]-nbr_U[0+N_VS]);
            NFn[1] = 0.5*(Fn[1]+nbr_Fn[1])*un[0]+0.5*mu*(U[1]-nbr_U[1+N_VS]);
            NFn[2] = 0.5*(Fn[2]+nbr_Fn[2])*un[0]+0.5*mu*(U[2]-nbr_U[2+N_VS]);
            NFn[3] = 0.5*(Fn[3]+nbr_Fn[3])*un[0];
            NFn[4] = 0.5*(Fn[4]+nbr_Fn[4])*un[0]+0.5*mu*(U[4]-nbr_U[4+N_VS]);
        }
        else if (dom == 1)
        {
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], &U[dom*N_VS], Fn);
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[dom], &nbr_U[(1-dom)*N_VS], nbr_Fn);

            NFn[0+N_VS] = 0.5*(Fn[0]+nbr_Fn[0])*un[0]+0.5*mu*(U[0+N_VS]-nbr_U[0]);
            NFn[1+N_VS] = 0.5*(Fn[1]+nbr_Fn[1])*un[0]+0.5*mu*(U[1+N_VS]-nbr_U[1]);
            NFn[2+N_VS] = 0.5*(Fn[2]+nbr_Fn[2])*un[0]+0.5*mu*(U[2+N_VS]-nbr_U[2]);
            NFn[3+N_VS] = 0.5*(Fn[3]+nbr_Fn[3])*un[0];
            NFn[4+N_VS] = 0.5*(Fn[4]+nbr_Fn[4])*un[0]+0.5*mu*(U[4+N_VS]-nbr_U[4]);
        }
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
            {0.0, 1.5},
            {15.0, 1.5}
        };
        point_fields_domain =
        {
            0,
            0
        };
        point_fields_name =
        {
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
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        fields_domain = {0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1};
        fields_name = {"Vx_a", "Vy_a", "Sxx_a", "Syy_a", "Sxy_a",
                       "Vx_b", "Vy_b", "Sxx_b", "Syy_b", "Sxy_b"};
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
    }
    // ================================================================

};