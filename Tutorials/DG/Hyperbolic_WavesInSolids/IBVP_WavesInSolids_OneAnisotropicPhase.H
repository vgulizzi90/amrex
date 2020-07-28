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
            {0.0, 3.3},
            {0.0, 9.9}
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
    }
    // ================================================================
};
// ####################################################################