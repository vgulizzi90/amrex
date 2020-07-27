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
        return -1.0;
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

        amrex::Real avg_U[N_VS];

        avg_U[0] = 0.5*(U[0]+nbr_U[0]);
        avg_U[1] = 0.5*(U[1]+nbr_U[1]);
        avg_U[2] = 0.5*(U[2]+nbr_U[2]);
        avg_U[3] = 0.5*(U[3]+nbr_U[3]);
        avg_U[4] = 0.5*(U[4]+nbr_U[4]);

        if (std::abs(un[0]) > 0.5)
        {
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
        }
        else if (std::abs(un[1]) > 0.5)
        {
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
        }
#if (AMREX_SPACEDIM == 3)
        else if (std::abs(un[2]) > 0.5)
        {
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A3[0], avg_U, NFn);

            NFn[0] *= un[2];
            NFn[1] *= un[2];
            NFn[2] *= un[2];
            NFn[3] *= un[2];
            NFn[4] *= un[2];

            NFn[0] += 0.5*mu*(U[0]-nbr_U[0]);
            NFn[1] += 0.5*mu*(U[1]-nbr_U[1]);
            NFn[2] += 0.5*mu*(U[2]-nbr_U[2]);
            NFn[3] += 0.5*mu*(U[3]-nbr_U[3]);
            NFn[4] += 0.5*mu*(U[4]-nbr_U[4]);
        }
#endif
/*
amrex::Real T[N_VS*N_VS], iT[N_VS*N_VS];
this->_eval_T_(un, T, iT);

amrex::Real avg_U[N_VS], avg_Un[N_VS];

for (int u = 0; u < N_U; ++u)
{
    avg_U[u] = 0.5*(U[u]+nbr_U[u]);
}
amrex::DG_utils::matmul(N_VS, N_VS, 1, T, avg_U, avg_Un);

amrex::Real Fn[N_VS];
amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[0], avg_Un, Fn);

amrex::DG_utils::matmul(N_VS, N_VS, 1, iT, Fn, NFn);

for (int u = 0; u < N_U; ++u)
{
    NFn[u] += 0.5*mu*(U[u]-nbr_U[u]);
}
*/
    }
    // ----------------------------------------------------------------

    // GRID BOUNDARIES ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        amrex::Real AMREX_D_DECL(t1, t2, t3);
        amrex::Real bcs_U[N_VS];

        // PRESCRIBED ZERO TRACTIONS
        if (std::abs(un[0]) > 0.5)
        {
            t1 = 0.0;
            t2 = 0.0;

            bcs_U[0] = U[0];
            bcs_U[1] = U[1];
            bcs_U[2] = t1;
            bcs_U[3] = U[3];
            bcs_U[4] = t2;

            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[0], bcs_U, NFn);

            NFn[0] *= un[0];
            NFn[1] *= un[0];
            NFn[2] *= un[0];
            NFn[3] *= un[0];
            NFn[4] *= un[0];

            NFn[2] += mu*(U[2]-bcs_U[2]);
            NFn[4] += mu*(U[4]-bcs_U[4]);
        }
        // PRESCRIBED ZERO TRACTIONS
        else if (un[1] > 0.5)
        {
            t1 = 0.0;
            t2 = 0.0;

            bcs_U[0] = U[0];
            bcs_U[1] = U[1];
            bcs_U[2] = U[2];
            bcs_U[3] = t2;
            bcs_U[4] = t1;

            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2[0], bcs_U, NFn);

            NFn[0] *= un[1];
            NFn[1] *= un[1];
            NFn[2] *= un[1];
            NFn[3] *= un[1];
            NFn[4] *= un[1];

            NFn[3] += mu*(U[3]-bcs_U[3]);
            NFn[4] += mu*(U[4]-bcs_U[4]);
        }
        // PRESCRIBED TRACTIONS
        else if (un[1] < -0.5)
        {
            const amrex::Real f = 10.0;
            const amrex::Real tc = 0.1;
            const amrex::Real T = std::exp(-1000.0*(t-tc)*(t-tc))*std::sin(f*2.0*M_PI*t);
            t1 = 0.0;
            t2 = T;

            bcs_U[0] = U[0];
            bcs_U[1] = U[1];
            bcs_U[2] = U[2];
            bcs_U[3] = t2;
            bcs_U[4] = t1;

            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2[0], bcs_U, NFn);

            NFn[0] *= un[1];
            NFn[1] *= un[1];
            NFn[2] *= un[1];
            NFn[3] *= un[1];
            NFn[4] *= un[1];
            
            NFn[3] += mu*(U[3]-bcs_U[3]);
            NFn[4] += mu*(U[4]-bcs_U[4]);
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
amrex::Print() << "Hello! F_NF_PHI_BCS (We should not end up here)" << std::endl;
exit(-1);
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
            {0.5, 0.0},
            {0.5, 1.0}
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