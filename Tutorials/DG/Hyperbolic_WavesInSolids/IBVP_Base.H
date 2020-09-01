/* ELASTIC WAVES IN SOLIDS */


// IBVP PARAMETERS ####################################################
// NUMBER OF VELOCITY COMPONENTS AND STRESS COMPONENTS ----------------
#define N_V AMREX_SPACEDIM

#if (AMREX_SPACEDIM == 1)
#define N_E 1
#define N_S 1
#endif
#if (AMREX_SPACEDIM == 2)
#define N_E 3
#define N_S 3
#endif
#if (AMREX_SPACEDIM == 3)
#define N_E 6
#define N_S 6
#endif

#define N_VE (N_V+N_E)
#define N_VS (N_V+N_S)

// ORDERING OF THE PRIMARY VARIABLES
#if (AMREX_SPACEDIM == 1)
#define V1 0
#define E11 1
#define S11 1
#endif
#if (AMREX_SPACEDIM == 2)
#define V1 0
#define V2 1
#define E11 2
#define E22 3
#define E12 4
#define S11 2
#define S22 3
#define S12 4
#endif
#if (AMREX_SPACEDIM == 3)
#define V1 0
#define V2 1
#define V3 2
#define E11 3
#define E22 4
#define E33 5
#define E23 6
#define E13 7
#define E12 8
#define S11 3
#define S22 4
#define S33 5
#define S23 6
#define S13 7
#define S12 8
#endif

// ORDERING OF THE ELASTIC COEFFICIENTS
#if (AMREX_SPACEDIM == 2)
#define C11 0
#define C21 1
#define C61 2

#define C22 3
#define C62 4

#define C66 5
#endif
#if (AMREX_SPACEDIM == 3)
#define C11 0
#define C21 1
#define C31 2
#define C41 3
#define C51 4
#define C61 5

#define C22 6
#define C32 7
#define C42 8
#define C52 9
#define C62 10

#define C33 11
#define C43 12
#define C53 13
#define C63 14

#define C44 15
#define C54 16
#define C64 17

#define C55 18
#define C65 19

#define C66 20
#endif
// --------------------------------------------------------------------

// PDES PARAMETERS AND CONSTANT OPERATORS -----------------------------
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
template <int N_DOM>
class ELASTIC_WAVES_BASE
{
public:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    amrex::Real rho[N_DOM];
    amrex::Real cSE[N_DOM][AMREX_D_PICK(1, 6, 21)];
    amrex::Real sES[N_DOM][AMREX_D_PICK(1, 6, 21)];

    // LARGEST WAVE SPEED
    amrex::Real max_wave_speed;
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES_BASE(const amrex::Vector<std::string> & material_type,
                       const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    {
#if (AMREX_SPACEDIM == 1)
        std::string msg;
        msg  = "\n";
        msg += "ERROR: IBVP_Base_VS.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE\n";
        msg += "| ELASTIC_WAVES_BASE must be double-checked step-by-step when dimension is "+std::to_string(AMREX_SPACEDIM)+".\n";
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

            // AUXILIARY VARIABLES
            amrex::Real s3D[36], c3D[36];
            std::fill(s3D, s3D+36, 0.0);
            std::fill(c3D, c3D+36, 0.0);

            // DENSITY
            this->rho[d] = m_prop[0];

            // STIFFNESS MATRIX
            if (m_type.compare("Isotropic") == 0)
            {
                const amrex::Real E = m_prop[1];
                const amrex::Real nu = m_prop[2];
                const amrex::Real G = 0.5*E/(1.0+nu);

                s3D[0+0*6] = 1.0/E; s3D[0+1*6] = -nu/E; s3D[0+2*6] = -nu/E;
                s3D[1+0*6] = -nu/E; s3D[1+1*6] = 1.0/E; s3D[1+2*6] = -nu/E;
                s3D[2+0*6] = -nu/E; s3D[2+1*6] = -nu/E; s3D[2+2*6] = 1.0/E;
                s3D[3+3*6] = 1.0/G;
                s3D[4+4*6] = 1.0/G;
                s3D[5+5*6] = 1.0/G;

                amrex::DG_utils::matinv(6, s3D, c3D);

#if (AMREX_SPACEDIM == 2)
                this->cSE[d][0] = c3D[0+0*6];
                this->cSE[d][1] = c3D[1+0*6]; this->cSE[d][3] = c3D[1+1*6];
                this->cSE[d][2] = c3D[5+0*6]; this->cSE[d][4] = c3D[5+1*6]; this->cSE[d][5] = c3D[5+5*6];

                this->sES[d][0] = s3D[0+0*6];
                this->sES[d][1] = s3D[1+0*6]; this->sES[d][3] = s3D[1+1*6];
                this->sES[d][2] = s3D[5+0*6]; this->sES[d][4] = s3D[5+1*6]; this->sES[d][5] = s3D[5+5*6];
#endif
#if (AMREX_SPACEDIM == 3)
                std::copy(c3D, c3D+6, this->cSE[d]);
                std::copy(&c3D[1*6+1], &c3D[1*6+1+5], &this->cSE[d][6]);
                std::copy(&c3D[2*6+2], &c3D[2*6+2+4], &this->cSE[d][6+5]);
                std::copy(&c3D[3*6+3], &c3D[3*6+3+3], &this->cSE[d][6+5+4]);
                std::copy(&c3D[4*6+4], &c3D[4*6+4+2], &this->cSE[d][6+5+4+3]);
                std::copy(&c3D[5*6+5], &c3D[5*6+5+1], &this->cSE[d][6+5+4+3+2]);

                std::copy(s3D, s3D+6, this->sES[d]);
                std::copy(&s3D[1*6+1], &s3D[1*6+1+5], &this->sES[d][6]);
                std::copy(&s3D[2*6+2], &s3D[2*6+2+4], &this->sES[d][6+5]);
                std::copy(&s3D[3*6+3], &s3D[3*6+3+3], &this->sES[d][6+5+4]);
                std::copy(&s3D[4*6+4], &s3D[4*6+4+2], &this->sES[d][6+5+4+3]);
                std::copy(&s3D[5*6+5], &s3D[5*6+5+1], &this->sES[d][6+5+4+3+2]);
#endif
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

        // MAXIMUM WAVE SPEED -----------------------------------------
        this->max_wave_speed = 0.0;

        for (int d = 0; d < N_DOM; ++d)
        {
            // PARAMETERS
            const std::string & m_type = material_type[d];

            if (m_type.compare("Isotropic") == 0)
            {
                const amrex::Real density = this->rho[d];
#if (AMREX_SPACEDIM == 2)
                const amrex::Real G = this->cSE[d][C66];
                const amrex::Real lam = this->cSE[d][C21];
#endif
#if (AMREX_SPACEDIM == 3)
                const amrex::Real G = this->cSE[d][C44];
                const amrex::Real lam = this->cSE[d][C21];
#endif
                this->max_wave_speed = std::max(this->max_wave_speed, std::sqrt((2.0*G+lam)/density));
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

    // AUXILIARY FUNCTIONS ============================================
    // EVALUATE THE STRAIN VECTOR GIVEN THE STRESS VECTOR -------------
    AMREX_GPU_HOST_DEVICE
    void _eval_E_(const int dom, const amrex::Real * S, amrex::Real * E) const
    {
        // PARAMETERS
        const amrex::Real * s = this->sES[dom];

#if (AMREX_SPACEDIM == 1)
        E[0] = s[C11]*S[0];
#endif
#if (AMREX_SPACEDIM == 2)
        E[0] = s[C11]*S[0]+s[C21]*S[1]+s[C61]*S[2];
        E[1] = s[C21]*S[0]+s[C22]*S[1]+s[C62]*S[2];
        E[2] = s[C61]*S[0]+s[C62]*S[1]+s[C66]*S[2];
#endif
#if (AMREX_SPACEDIM == 3)
        E[0] = s[C11]*S[0]+s[C21]*S[1]+s[C31]*S[2]+s[C41]*S[3]+s[C51]*S[4]+s[C61]*S[5];
        E[1] = s[C21]*S[0]+s[C22]*S[1]+s[C32]*S[2]+s[C42]*S[3]+s[C52]*S[4]+s[C62]*S[5];
        E[2] = s[C31]*S[0]+s[C32]*S[1]+s[C33]*S[2]+s[C43]*S[3]+s[C53]*S[4]+s[C63]*S[5];
        E[3] = s[C41]*S[0]+s[C42]*S[1]+s[C43]*S[2]+s[C44]*S[3]+s[C54]*S[4]+s[C64]*S[5];
        E[4] = s[C51]*S[0]+s[C52]*S[1]+s[C53]*S[2]+s[C54]*S[3]+s[C55]*S[4]+s[C65]*S[5];
        E[6] = s[C61]*S[0]+s[C62]*S[1]+s[C63]*S[2]+s[C64]*S[3]+s[C65]*S[4]+s[C66]*S[5];
#endif
    }
    // ----------------------------------------------------------------
    // EVALUATE THE STRESS VECTOR GIVEN THE STRAIN VECTOR -------------
    AMREX_GPU_HOST_DEVICE
    void _eval_S_(const int dom, const amrex::Real * E, amrex::Real * S) const
    {
        // PARAMETERS
        const amrex::Real * c = this->cSE[dom];

#if (AMREX_SPACEDIM == 1)
        S[0] = c[C11]*E[0];
#endif
#if (AMREX_SPACEDIM == 2)
        S[0] = c[C11]*E[0]+c[C21]*E[1]+c[C61]*E[2];
        S[1] = c[C21]*E[0]+c[C22]*E[1]+c[C62]*E[2];
        S[2] = c[C61]*E[0]+c[C62]*E[1]+c[C66]*E[2];
#endif
#if (AMREX_SPACEDIM == 3)
        S[0] = c[C11]*E[0]+c[C21]*E[1]+c[C31]*E[2]+c[C41]*E[3]+c[C51]*E[4]+c[C61]*E[5];
        S[1] = c[C21]*E[0]+c[C22]*E[1]+c[C32]*E[2]+c[C42]*E[3]+c[C52]*E[4]+c[C62]*E[5];
        S[2] = c[C31]*E[0]+c[C32]*E[1]+c[C33]*E[2]+c[C43]*E[3]+c[C53]*E[4]+c[C63]*E[5];
        S[3] = c[C41]*E[0]+c[C42]*E[1]+c[C43]*E[2]+c[C44]*E[3]+c[C54]*E[4]+c[C64]*E[5];
        S[4] = c[C51]*E[0]+c[C52]*E[1]+c[C53]*E[2]+c[C54]*E[3]+c[C55]*E[4]+c[C65]*E[5];
        S[6] = c[C61]*E[0]+c[C62]*E[1]+c[C63]*E[2]+c[C64]*E[3]+c[C65]*E[4]+c[C66]*E[5];
#endif
    }
    // ----------------------------------------------------------------
    // ================================================================

    // BODY LOAD ======================================================
    AMREX_GPU_HOST_DEVICE
    void F_B(const amrex::Real t, const amrex::Real * x,
             amrex::Real * B) const
    {
        for (int u = 0; u < (N_DOM*N_VE); ++u)
        {
            B[u] = 0.0;
        }
    }
    // ================================================================

    // INITIAL CONDITIONS =============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        return 0.0;
    }
    // ================================================================

    // FUNCTIONS FOR POST-PROCESSING ==================================
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
amrex::Abort("Hello! ELASTIC_WAVES_BASE.F_POINT_SOL (We should not end up in here)");
    }

    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
        fields_domain.clear();
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
amrex::Abort("Hello! ELASTIC_WAVES_BASE.F_SOL (We should not end up in here)");
    }
    // ================================================================
};
// ####################################################################



// IBVP CLASS: VELOCITY-STRESS FORMULATION ############################
template <int N_DOM>
class ELASTIC_WAVES_BASE_VS
:
public ELASTIC_WAVES_BASE<N_DOM>
{
public:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES_BASE_VS(const amrex::Vector<std::string> & material_type,
                          const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    ELASTIC_WAVES_BASE<N_DOM>(material_type, material_properties)
    {}
    // ================================================================

    // REFERENCE SYSTEM TRANSFORMATIONS ===============================
    // ================================================================

    // EVAL FLUX FUNCTIONS ============================================
    // EVAL F1 --------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_F1_(const int dom, const amrex::Real * U, amrex::Real * F1) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];

#if (AMREX_SPACEDIM == 2)
        F1[V1] = -inv_rho*U[S11];
        F1[V2] = -inv_rho*U[S12];

        F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2];
        F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2];
        F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
        F1[V1] = -inv_rho*U[S11];
        F1[V2] = -inv_rho*U[S12];
        F1[V3] = -inv_rho*U[S13];

        F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2]-c[C51]*U[V3];
        F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2]-c[C52]*U[V3];
        F1[S33] = -c[C31]*U[V1]-c[C63]*U[V2]-c[C53]*U[V3];
        F1[S23] = -c[C41]*U[V1]-c[C64]*U[V2]-c[C54]*U[V3];
        F1[S13] = -c[C51]*U[V1]-c[C65]*U[V2]-c[C55]*U[V3];
        F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2]-c[C65]*U[V3];
#endif
    }
    // ----------------------------------------------------------------

    // EVAL F2 --------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_F2_(const int dom, const amrex::Real * U, amrex::Real * F2) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];

#if (AMREX_SPACEDIM == 2)
        F2[V1] = -inv_rho*U[S12];
        F2[V2] = -inv_rho*U[S22];

        F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2];
        F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2];
        F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
        F2[V1] = -inv_rho*U[S12];
        F2[V2] = -inv_rho*U[S22];
        F2[V3] = -inv_rho*U[S23];

        F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2]-c[C41]*U[V3];
        F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2]-c[C42]*U[V3];
        F2[S33] = -c[C63]*U[V1]-c[C32]*U[V2]-c[C43]*U[V3];
        F2[S23] = -c[C64]*U[V1]-c[C42]*U[V2]-c[C44]*U[V3];
        F2[S13] = -c[C65]*U[V1]-c[C52]*U[V2]-c[C54]*U[V3];
        F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2]-c[C64]*U[V3];
#endif
    }
    // ----------------------------------------------------------------

    // EVAL F3 --------------------------------------------------------
#if (AMREX_SPACEDIM > 2)
    AMREX_GPU_HOST_DEVICE
    void _eval_F3_(const int dom, const amrex::Real * U, amrex::Real * F3) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];

        F3[V1] = -inv_rho*U[S13];
        F3[V2] = -inv_rho*U[S23];
        F3[V3] = -inv_rho*U[S33];

        F3[S11] = -c[C51]*U[V1]-c[C41]*U[V2]-c[C31]*U[V3];
        F3[S22] = -c[C52]*U[V1]-c[C42]*U[V2]-c[C32]*U[V3];
        F3[S33] = -c[C53]*U[V1]-c[C43]*U[V2]-c[C33]*U[V3];
        F3[S23] = -c[C54]*U[V1]-c[C44]*U[V2]-c[C43]*U[V3];
        F3[S13] = -c[C55]*U[V1]-c[C54]*U[V2]-c[C53]*U[V3];
        F3[S12] = -c[C65]*U[V1]-c[C64]*U[V2]-c[C63]*U[V3];
    }
#endif
    // ----------------------------------------------------------------

    // EVAL AnU -------------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_AnU_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];

#if (AMREX_SPACEDIM == 2)
        AnU[V1] = -inv_rho*(U[S11]*un[0]+U[S12]*un[1]);
        AnU[V2] = -inv_rho*(U[S12]*un[0]+U[S22]*un[1]);

        AnU[S11] = -(c[C11]*U[V1]+c[C61]*U[V2])*un[0]-(c[C61]*U[V1]+c[C21]*U[V2])*un[1];
        AnU[S22] = -(c[C21]*U[V1]+c[C62]*U[V2])*un[0]-(c[C62]*U[V1]+c[C22]*U[V2])*un[1];
        AnU[S12] = -(c[C61]*U[V1]+c[C66]*U[V2])*un[0]-(c[C66]*U[V1]+c[C62]*U[V2])*un[1];
#endif
#if (AMREX_SPACEDIM == 3)
#endif
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
            const amrex::Real * Ud = &U[d*N_VS];
            AMREX_D_TERM
            (
                amrex::Real * F1d = &F1[d*N_VS];,
                amrex::Real * F2d = &F2[d*N_VS];,
                amrex::Real * F3d = &F3[d*N_VS];
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
};
// ####################################################################