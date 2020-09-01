/* ELASTIC WAVE EQUATION */

#include <IBVP_WavesInSolids_Base.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF LEVEL SET FUNCTIONS/DOMAINS/UNKNOWN FIELDS ---------------
#define N_PHI 2
#define N_DOM 2
#define N_U (2*N_VS)
// --------------------------------------------------------------------

// SOME PARAMETERS ----------------------------------------------------
#define H 0.50
#define R 0.25
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

    }
    // ================================================================

    // LEVEL SET FUNCTION =============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
        const amrex::Real AMREX_D_DECL(x1 = x[0], x2 = x[1], x3 = x[2]);
#if (AMREX_SPACEDIM == 2)
        const amrex::Real PHI[N_PHI] =
        {
            x2-H,
            (x1-0.5)*(x1-0.5)+(x2-0.5)*(x2-0.5)-R*R
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real PHI[N_PHI] =
        {
            x3-H,
            (x1-0.5)*(x1-0.5)+(x2-0.5)*(x2-0.5)+(x3-0.5)*(x3-0.5)-R*R
        };
#endif
        return PHI[ph];
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
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

    amrex::Real F_EXACT_VOLUME(const int & dom) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            H,

            // dom = 1
            1.0-H
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            H,

            // dom = 1
            1.0-H
        };
#endif

        return EXACT_VOLUME[dom];
    }
    amrex::Real F_EXACT_SURFACE(const int & dom) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            1.0,

            // dom = 1
            1.0
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            1.0,

            // dom = 1
            1.0
        };
#endif

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
        return 0.0;
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUXES ===============
    // INTRAPHASE -----------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_ICS (We should not end up here)");
    }
    // ----------------------------------------------------------------

    // GRID BOUNDARIES ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_BCS (We should not end up here)");
    }
    // ----------------------------------------------------------------

    // INTERNAL BOUNDARIES --------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_PHI_BCS (We should not end up here)");
    }
    // ----------------------------------------------------------------

    // INTERNAL INTERFACE ---------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_PHI_ICS (We should not end up here)");
    }
    // ----------------------------------------------------------------
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
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
amrex::Abort("Hello! F_POINT_SOL (We should not end up here)");
    }

    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
        fields_domain.clear();
        fields_name.clear();
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
    }
    // ================================================================

};