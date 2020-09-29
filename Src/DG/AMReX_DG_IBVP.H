// AMReX_DG_IBVP.H

#ifndef BL_DG_IBVP_H_
#define BL_DG_IBVP_H_

namespace amrex
{
namespace DG
{

// IBVP BASE CLASS ####################################################
template<int N_PHI, int N_DOM, int N_U>
struct IBVP_BASE
{
    // LEVEL SET FUNCTION =============================================
    AMREX_GPU_HOST_DEVICE
    Real F_PHI(const int ph, const Real t, const Real * x) const
    {
        return -1.0;
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int dom, int * phi_info) const
    {
        phi_info[0] = 0;
        phi_info[1] = -1;
    }

    Real F_EXACT_VOLUME(const int dom) const
    {
        return 0.0;
    }

    Real F_EXACT_SURFACE(const int dom) const
    {
        return 0.0;
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int u) const
    {
        return 0;
    }
    // ================================================================

    // RELATIONSHIP BETWEEN NEIGHBORING DOMAINS =======================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int dom) const
    {
        return -1;
    }
    // ================================================================

    // INITIAL CONDITIONS =============================================
    AMREX_GPU_HOST_DEVICE
    Real F_U0(const int u, const Real * x) const
    {
        return 0.0;
    }
    // ================================================================

    // BOUNDARY CONDITIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const int dom, const Real t, const Real * x, const Real * un,
               const Real * U, Real * bcs_U) const
    {
Abort("Hello! IBVP_BASE.F_BCS (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS(const int dom, const Real t, const Real * x, const Real * un,
                   const Real * U, Real * bcs_U) const
    {
Abort("Hello! IBVP_BASE.F_PHI_BCS (We should not end up in here)");
    }
    // ================================================================

    // TROUBLED CELLS INDICATOR =======================================
    AMREX_GPU_HOST_DEVICE
    bool F_TROUBLED(const int dom, const Real t, const Real * x, const Real * U) const
    {
        return false;
    }
    // ================================================================

    // FUNCTIONS FOR POST-PROCESSING ==================================
    void F_POINT_SOL_DESCRIPTION(Vector<Array<Real, AMREX_SPACEDIM>> & point_fields_location,
                                 Vector<int> & point_fields_domain,
                                 Vector<Vector<std::string>> & point_fields_name) const
    {
        point_fields_location.clear();
    }

    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL(const int p,
                     const Real * PHI, AMREX_D_DECL(const Real * dPHIdx1, const Real * dPHIdx2, const Real * dPHIdx3),
                     const Real * U, AMREX_D_DECL(const Real * dUdx1, const Real * dUdx2, const Real * dUdx3),
                     Real * F) const
    {
Abort("Hello! IBVP_BASE.F_POINT_SOL (We should not end up in here)");
    }

    void F_SOL_DESCRIPTION(Vector<int> & fields_domain, Vector<std::string> & fields_name) const
    {
        fields_domain.clear();
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const Real t, const Real * x,
               const Real * PHI, AMREX_D_DECL(const Real * dPHIdx1, const Real * dPHIdx2, const Real * dPHIdx3),
               const Real * U, AMREX_D_DECL(const Real * dUdx1, const Real * dUdx2, const Real * dUdx3),
               Real * F) const
    {
Abort("Hello! IBVP_BASE.F_SOL (We should not end up in here)");
    }
    // ================================================================
};
// ####################################################################



// IBVP HYPERBOLIC BASE CLASS #########################################
// This data structure is a base class containing the base methods
// related to a hyperbolic system of PDEs of the form:
//
// U_{,t}+F_{i,i}(U)+R(U) = B
//
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct IBVP_HYPERBOLIC_BASE
:
public IBVP_BASE<N_PHI, N_DOM, N_U>
{
    // Fi TERMS =======================================================
    AMREX_GPU_HOST_DEVICE
    void F_F(const Real t, const Real * x,
             const Real * U,
             AMREX_D_DECL(Real * F1, Real * F2, Real * F3)) const
    {
        for (int u = 0; u < N_U; ++u)
        {
            AMREX_D_TERM
            (
                F1[u] = 0.0;,
                F2[u] = 0.0;,
                F3[u] = 0.0;
            )
        }
    }
    // ================================================================

    // R TERMS ========================================================
    AMREX_GPU_HOST_DEVICE
    void F_R(const Real t, const Real * x,
             const Real * U,
             Real * R) const
    {
        for (int u = 0; u < N_U; ++u)
        {
            R[u] = 0.0;
        }
    }
    // ================================================================

    // B TERMS ========================================================
    AMREX_GPU_HOST_DEVICE
    void F_B(const Real t, const Real * x,
             const Real * U,
             Real * B) const
    {
        for (int u = 0; u < N_U; ++u)
        {
            B[u] = 0.0;
        }
    }
    // ================================================================

    // CONSERVED-CHARACTERISTICS VARIABLES RELATIONSHIPS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC(const int dom,
                 const Real t, const Real * x, const Real * un,
                 const Real * U, const Real * DU, Real * DC) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DU2DC (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_1(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DU, Real * DC) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DU2DC_1 (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_2(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DU, Real * DC) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DU2DC_1 (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_DU2DC_3(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DU, Real * DC) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DU2DC_1 (We should not end up in here)");
    }

    AMREX_GPU_HOST_DEVICE
    void F_DC2DU(const int dom,
                 const Real t, const Real * x, const Real * un,
                 const Real * U, const Real * DC, Real * DU) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DC2DU (We should not end up in here)");
    }

    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_1(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DC, Real * DU) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DC2DU_1 (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_2(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DC, Real * DU) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DC2DU_1 (We should not end up in here)");
    }
    AMREX_GPU_HOST_DEVICE
    void F_DC2DU_3(const int dom,
                   const Real t, const Real * x,
                   const Real * U, const Real * DC, Real * DU) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_DC2DU_1 (We should not end up in here)");
    }
    // ================================================================

    // NUMERICAL FLUXES: INTRAPHASE ===================================
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const Real t, const Real * x, const Real * un,
                  const Real * U, const Real * nbr_U,
                  Real * NFn) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_NF_ICS (We should not end up in here)");
    }
    // ================================================================

    // NUMERICAL FLUXES: GRID BOUNDARIES ==============================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int dom,
                  const Real t, const Real * x, const Real * un,
                  const Real * U,
                  Real * NFn) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_NF_BCS (We should not end up in here)");
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL BOUNDARIES ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int dom,
                      const Real t, const Real * x, const Real * un,
                      const Real * U,
                      Real * NFn) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_NF_PHI_BCS (We should not end up in here)");
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL INTERFACE ===========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int dom, const int nbr_dom,
                      const Real t, const Real * x, const Real * un,
                      const Real * U, const Real * nbr_U,
                      Real * NFn) const
    {
Abort("Hello! IBVP_HYPERBOLIC_BASE.F_NF_PHI_ICS (We should not end up in here)");
    }
    // ================================================================
};
// ####################################################################

} // namespace DG
} // namespace amrex

#endif