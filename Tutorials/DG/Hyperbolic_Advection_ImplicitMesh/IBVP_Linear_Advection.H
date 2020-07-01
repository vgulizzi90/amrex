/* LINEAR ADVECTION EQUATION */

// IBVP PARAMETERS ####################################################
// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS
#define N_PHI 3
#define N_DOM 2
#define N_U 2
// ---------------------------------------------------

// PDES PARAMETERS --------
#define CONSTANT_VELOCITY 1
#define SINGLE_VORTEX 2
#define VELOCITY_FIELD 1
// ------------------------

// DG-METHOD PARAMETERS -----------------------------------------------
#define DG_NUM_FLUX_LF 0
#define DG_NUM_FLUX_UPW 1
#define DG_NUM_FLUX 1
// --------------------------------------------------------------------
// ####################################################################

// IBVP CLASS #########################################################
class LINADV
{
private:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES

    // AUXILIARY DATA MEMBERS
    // ================================================================
public:
    // CONSTRUCTOR ====================================================
    // ================================================================

    // DISTRUCTOR =====================================================
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
#if (AMREX_SPACEDIM == 2)
        const int n_holes = 3;
        const amrex::Real r[n_holes] = {0.15, 0.12, 0.10};
        const amrex::Real c0[n_holes] = {0.4, 0.75, 0.3};
        const amrex::Real c1[n_holes] = {0.7, 0.45, 0.2};
        amrex::Real phi;
        phi = r[0]*r[0]-(x[0]-c0[0])*(x[0]-c0[0])-(x[1]-c1[0])*(x[1]-c1[0]);
        for (int h = 1; h < n_holes; ++h)
        {
            phi = std::max(phi, r[h]*r[h]-(x[0]-c0[h])*(x[0]-c0[h])-(x[1]-c1[h])*(x[1]-c1[h]));
        }

        const amrex::Real PHI[N_PHI] =
        {
            phi,
            (x[0]+0.1*x[1]-0.41)*(0.67-x[0]+0.1*x[1]),
            -1.0
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int n_holes = 3;
        const amrex::Real r[n_holes] = {0.13, 0.11, 0.08};
        const amrex::Real c0[n_holes] = {0.28, 0.67, 0.32};
        const amrex::Real c1[n_holes] = {0.7, 0.45, 0.4};
        const amrex::Real c2[n_holes] = {0.3, 0.7, 0.5};
        amrex::Real phi;
        phi = r[0]*r[0]-(x[0]-c0[0])*(x[0]-c0[0])-(x[1]-c1[0])*(x[1]-c1[0])-(x[2]-c2[0])*(x[2]-c2[0]);
        for (int h = 1; h < n_holes; ++h)
        {
            phi = std::max(phi, r[h]*r[h]-(x[0]-c0[h])*(x[0]-c0[h])-(x[1]-c1[h])*(x[1]-c1[h])-(x[2]-c2[h])*(x[2]-c2[h]));
        }

        const amrex::Real PHI[N_PHI] =
        {
            phi,
            (x[0]+0.1*x[2]-0.41)*(0.67-x[0]+0.1*x[2]),
            -1.0
        };
#endif

        return PHI[ph];
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

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(0.25, 0.5, 0.5)};
#if (AMREX_SPACEDIM == 2)
        //amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
        //amrex::Real phi0 = std::exp(-r2);
        amrex::Real rho0 = std::sin(2.0*M_PI*x[0])*std::sin(2.0*M_PI*x[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        //amrex::Real r2 = 100.0*((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1])+(x[2]-xc[2])*(x[2]-xc[2]));
        //amrex::Real phi0 = std::exp(-r2);
        amrex::Real rho0 = std::sin(2.0*M_PI*x[0])*std::sin(2.0*M_PI*x[1])*std::sin(2.0*M_PI*x[2]);
#endif

        return rho0;
    }
    // ================================================================
    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
        const int DOM[N_U] =
        {
            // sol = 0
            0,

            // sol = 1
            1
        };

        return DOM[u];
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * U) const
    {
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-a[0]*t, x[1]-a[1]*t, x[2]-a[2]*t)};
        
        U[0] = this->F_U0(0, y);
        U[1] = this->F_U0(0, y);
    }

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
        const int ph = 0;
        const amrex::Real PHI = this->F_PHI(ph, t, x);

        amrex::Real Ue[N_U];

        this->F_EXACT(t, x, Ue);

        norm_x = Ue[0]*Ue[0];

        Ue[0] -= ((PHI < 0.0) ? U[0] : U[1]);
        err_x = Ue[0]*Ue[0];
    }
    // ================================================================

    // BOUNDARY CONDITIONS ============================================
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

    // PDES SYSTEM ====================================================
    AMREX_GPU_HOST_DEVICE
    void F_a(const amrex::Real & t, const amrex::Real * x,
             amrex::Real * a) const
    {
#if (VELOCITY_FIELD == CONSTANT_VELOCITY)
#if (AMREX_SPACEDIM == 2)
        a[0] = std::sqrt(2.0)/2.0;
        a[1] = std::sqrt(2.0)/2.0;
#endif
#if (AMREX_SPACEDIM == 3)
        a[0] = std::sqrt(3.0)/3.0;
        a[1] = std::sqrt(3.0)/3.0;
        a[2] = std::sqrt(3.0)/3.0;
#endif
#endif
#if (VELOCITY_FIELD == SINGLE_VORTEX)
        const amrex::Real ct = -std::cos(0.5*M_PI*t);
        const amrex::Real sx = std::sin(M_PI*x[0]);
        const amrex::Real sy = std::sin(M_PI*x[1]);
        const amrex::Real s2x = std::sin(2.0*M_PI*x[0]);
        const amrex::Real s2y = std::sin(2.0*M_PI*x[1]);
        AMREX_D_TERM
        (
            a[0] = -ct*sx*sx*s2y;,
            a[1] = +ct*s2x*sy*sy;,
            a[2] = 1.0;
        )
#endif
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_am(const amrex::Real & t, const amrex::Real * x) const
    {
        amrex::Real am, a[AMREX_SPACEDIM];
        this->F_a(t, x, a);

        am = std::sqrt(AMREX_D_PICK(a[0]*a[0], a[0]*a[0]+a[1]*a[1], a[0]*a[0]+a[1]*a[1]+a[2]*a[2]));
        return am;
    }

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_DT(const amrex::Real * dx,
                     const amrex::Real & t, const amrex::Real * x,
                     const amrex::Real * U) const
    {
        amrex::Real dt;
        const amrex::Real am = this->F_am(t, x);

#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
        dt = 0.5*h/am;
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
        dt = 0.25*h/am;
#endif
        
        return dt;
    }

    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);

        AMREX_D_TERM
        (
            F1[0] = a[0]*U[0];,
            F2[0] = a[1]*U[0];,
            F3[0] = a[2]*U[0];
        )
        AMREX_D_TERM
        (
            F1[1] = a[0]*U[1];,
            F2[1] = a[1]*U[1];,
            F3[1] = a[2]*U[1];
        )
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUX =================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BASE(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                   const amrex::Real * U, const amrex::Real * nbr_U,
                   AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                   AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                   AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
#if (DG_NUM_FLUX == DG_NUM_FLUX_LF)
        const amrex::Real mu = this->F_am(t, x);
#endif
#if (DG_NUM_FLUX == DG_NUM_FLUX_UPW)
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real a_dot_un = AMREX_D_PICK(a[0]*un[0], a[0]*un[0]+a[1]*un[1], a[0]*un[0]+a[1]*un[1]+a[2]*un[2]);
        const amrex::Real mu = std::abs(a_dot_un);
#endif

        // dom = 0
        AMREX_D_TERM
        (
            NF1[0] = 0.5*(F1[0]+nbr_F1[0])+0.5*mu*(U[0]-nbr_U[0])*un[0];,
            NF2[0] = 0.5*(F2[0]+nbr_F2[0])+0.5*mu*(U[0]-nbr_U[0])*un[1];,
            NF3[0] = 0.5*(F3[0]+nbr_F3[0])+0.5*mu*(U[0]-nbr_U[0])*un[2];
        )

        // dom = 1
        AMREX_D_TERM
        (
            NF1[1] = 0.5*(F1[1]+nbr_F1[1])+0.5*mu*(U[1]-nbr_U[1])*un[0];,
            NF2[1] = 0.5*(F2[1]+nbr_F2[1])+0.5*mu*(U[1]-nbr_U[1])*un[1];,
            NF3[1] = 0.5*(F3[1]+nbr_F3[1])+0.5*mu*(U[1]-nbr_U[1])*un[2];
        )
    }

    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BASE(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                       const amrex::Real * U, const amrex::Real * nbr_U,
                       AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                       AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                       AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
#if (DG_NUM_FLUX == DG_NUM_FLUX_LF)
        const amrex::Real mu = this->F_am(t, x);
#endif
#if (DG_NUM_FLUX == DG_NUM_FLUX_UPW)
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real a_dot_un = AMREX_D_PICK(a[0]*un[0], a[0]*un[0]+a[1]*un[1], a[0]*un[0]+a[1]*un[1]+a[2]*un[2]);
        const amrex::Real mu = std::abs(a_dot_un);
#endif

        // dom = 0
        AMREX_D_TERM
        (
            NF1[0] = 0.5*(F1[0]+nbr_F1[1])+0.5*mu*(U[0]-nbr_U[1])*un[0];,
            NF2[0] = 0.5*(F2[0]+nbr_F2[1])+0.5*mu*(U[0]-nbr_U[1])*un[1];,
            NF3[0] = 0.5*(F3[0]+nbr_F3[1])+0.5*mu*(U[0]-nbr_U[1])*un[2];
        )

        // dom = 1
        AMREX_D_TERM
        (
            NF1[1] = 0.5*(F1[1]+nbr_F1[0])+0.5*mu*(U[1]-nbr_U[0])*un[0];,
            NF2[1] = 0.5*(F2[1]+nbr_F2[0])+0.5*mu*(U[1]-nbr_U[0])*un[1];,
            NF3[1] = 0.5*(F3[1]+nbr_F3[0])+0.5*mu*(U[1]-nbr_U[0])*un[2];
        )
    }
    
    // INTRAPHASE
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->F_NF_BASE(t, x, un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
    }

    // GRID BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real a_dot_un = AMREX_D_PICK(a[0]*un[0], a[0]*un[0]+a[1]*un[1], a[0]*un[0]+a[1]*un[1]+a[2]*un[2]);
        const amrex::Real mu = std::abs(a_dot_un);
        amrex::Real U_exact[N_U], AMREX_D_DECL(F1_exact[N_U], F2_exact[N_U], F3_exact[N_U]);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));

        if (a_dot_un < 0.0)
        {
            this->F_EXACT(t, x, U_exact);
            this->F_F(t, x, U_exact, AMREX_D_DECL(F1_exact, F2_exact, F3_exact));

            // dom = 0
            AMREX_D_TERM
            (
                NF1[0] = 0.5*(F1[0]+F1_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[0];,
                NF2[0] = 0.5*(F2[0]+F2_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[1];,
                NF3[0] = 0.5*(F3[0]+F3_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[2];
            )

            // dom = 1
            AMREX_D_TERM
            (
                NF1[1] = 0.5*(F1[1]+F1_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[0];,
                NF2[1] = 0.5*(F2[1]+F2_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[1];,
                NF3[1] = 0.5*(F3[1]+F3_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[2];
            )
        }
        else
        {
            // dom = 0
            AMREX_D_TERM
            (
                NF1[0] = F1[0];,
                NF2[0] = F2[0];,
                NF3[0] = F3[0];
            )

            // dom = 1
            AMREX_D_TERM
            (
                NF1[1] = F1[1];,
                NF2[1] = F2[1];,
                NF3[1] = F3[1];
            )
        }
    }

    // INTERNAL BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real a_dot_un = AMREX_D_PICK(a[0]*un[0], a[0]*un[0]+a[1]*un[1], a[0]*un[0]+a[1]*un[1]+a[2]*un[2]);
        const amrex::Real mu = std::abs(a_dot_un);
        amrex::Real U_exact[N_U], AMREX_D_DECL(F1_exact[N_U], F2_exact[N_U], F3_exact[N_U]);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));

        if (a_dot_un < 0.0)
        {
            this->F_EXACT(t, x, U_exact);
            this->F_F(t, x, U_exact, AMREX_D_DECL(F1_exact, F2_exact, F3_exact));

            // dom = 0
            AMREX_D_TERM
            (
                NF1[0] = 0.5*(F1[0]+F1_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[0];,
                NF2[0] = 0.5*(F2[0]+F2_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[1];,
                NF3[0] = 0.5*(F3[0]+F3_exact[0])+0.5*mu*(U[0]-U_exact[0])*un[2];
            )

            // dom = 1
            AMREX_D_TERM
            (
                NF1[1] = 0.5*(F1[1]+F1_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[0];,
                NF2[1] = 0.5*(F2[1]+F2_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[1];,
                NF3[1] = 0.5*(F3[1]+F3_exact[1])+0.5*mu*(U[1]-U_exact[1])*un[2];
            )
        }
        else
        {
            // dom = 0
            AMREX_D_TERM
            (
                NF1[0] = F1[0];,
                NF2[0] = F2[0];,
                NF3[0] = F3[0];
            )

            // dom = 1
            AMREX_D_TERM
            (
                NF1[1] = F1[1];,
                NF2[1] = F2[1];,
                NF3[1] = F3[1];
            )
        }
    }

    // INTERNAL INTERFACE
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);
        amrex::Real a[AMREX_SPACEDIM];
        this->F_a(t, x, a);
        const amrex::Real a_dot_un = AMREX_D_PICK(a[0]*un[0], a[0]*un[0]+a[1]*un[1], a[0]*un[0]+a[1]*un[1]+a[2]*un[2]);

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        if (a_dot_un < 0.0)
        {
            this->F_NF_PHI_BASE(t, x, un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
        }
        else
        {
            // dom = 0
            AMREX_D_TERM
            (
                NF1[0] = F1[0];,
                NF2[0] = F2[0];,
                NF3[0] = F3[0];
            )

            // dom = 1
            AMREX_D_TERM
            (
                NF1[1] = F1[1];,
                NF2[1] = F2[1];,
                NF3[1] = F3[1];
            )
        }
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // PARAMETERS

        // EXACT SOLUTION
        amrex::Real Ue[N_U];

        this->F_EXACT(t, x, Ue);

        // FIELD EXPRESSIONS
        F[0] = U[0];
        F[1] = U[0]-Ue[0];
        F[2] = U[1];
        F[3] = U[1]-Ue[1];
    }
    // ================================================================
};
// ####################################################################
