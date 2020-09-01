/* ELASTIC WAVE EQUATION */

// IBVP PARAMETERS ####################################################
// NUMBER OF VELOCITY COMPONENTS AND STRESS COMPONENTS ----------------
#define N_V AMREX_SPACEDIM

#if (AMREX_SPACEDIM == 1)
#define N_S 1
#endif
#if (AMREX_SPACEDIM == 2)
#define N_S 3
#endif
#if (AMREX_SPACEDIM == 3)
#define N_S 6
#endif

#define N_VS (N_V+N_S)

#if (AMREX_SPACEDIM == 2)
#define V1 0
#define V2 1
#define S11 2
#define S22 3
#define S12 4
#endif
#if (AMREX_SPACEDIM == 3)
#define V1 0
#define V2 1
#define V3 2
#define S11 3
#define S22 4
#define S33 5
#define S23 6
#define S13 7
#define S12 8
#endif

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
        msg += "ERROR: IBVP_WavesInSolids_Base_VS.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE\n";
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
                msg += "ERROR: IBVP_WavesInSolids_Base_VS.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE\n";
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
                msg += "ERROR: IBVP_WavesInSolids_Base_VS.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE\n";
                msg += "| Unexpected material type: "+m_type+" with dimension: "+std::to_string(AMREX_SPACEDIM)+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    // EVALUATE THE STRESS GIVEN THE STRAIN VECTOR --------------------
    AMREX_GPU_HOST_DEVICE
    void _eval_E_(const int dom, const amrex::Real * S, amrex::Real * E) const
    {
        // PARAMETERS
        const amrex::Real * s = this->sES[dom];

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
        F1[V1] = -inv_rho*(c[C11]*U[E11]+c[C21]*U[E22]+c[C31]*U[E33]+c[C41]*U[E23]+c[C51]*U[E13]+c[C61]*U[E12]);
        F1[V2] = -inv_rho*(c[C61]*U[E11]+c[C62]*U[E22]+c[C63]*U[E33]+c[C64]*U[E23]+c[C65]*U[E13]+c[C66]*U[E12]);
        F1[V3] = -inv_rho*(c[C51]*U[E11]+c[C52]*U[E22]+c[C53]*U[E33]+c[C54]*U[E23]+c[C55]*U[E13]+c[C65]*U[E12]);

        F1[E11] = -U[V1];
        F1[E22] = 0.0;
        F1[E33] = 0.0;
        F1[E23] = 0.0;
        F1[E13] = -U[V3];
        F1[E12] = -U[V2];
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
        F2[V1] = -inv_rho*(c[C61]*U[E11]+c[C62]*U[E22]+c[C63]*U[E33]+c[C64]*U[E23]+c[C65]*U[E13]+c[C66]*U[E12]);
        F2[V2] = -inv_rho*(c[C21]*U[E11]+c[C22]*U[E22]+c[C32]*U[E33]+c[C42]*U[E23]+c[C52]*U[E13]+c[C62]*U[E12]);
        F2[V3] = -inv_rho*(c[C41]*U[E11]+c[C42]*U[E22]+c[C43]*U[E33]+c[C44]*U[E23]+c[C54]*U[E13]+c[C64]*U[E12]);

        F2[E11] = 0.0;
        F2[E22] = -U[V2];
        F2[E33] = 0.0;
        F2[E23] = -U[V3];
        F2[E13] = 0.0;
        F2[E12] = -U[V1];
#endif
    }
    // ----------------------------------------------------------------

    // EVAL F3 --------------------------------------------------------
#if (AMREX_SPACEDIM == 3)
    AMREX_GPU_HOST_DEVICE
    void _eval_F3_(const int dom, const amrex::Real * U, amrex::Real * F3) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];

        F3[V1] = -inv_rho*(c[C51]*U[E11]+c[C52]*U[E22]+c[C53]*U[E33]+c[C54]*U[E23]+c[C55]*U[E13]+c[C65]*U[E12]);
        F3[V2] = -inv_rho*(c[C41]*U[E11]+c[C42]*U[E22]+c[C43]*U[E33]+c[C44]*U[E23]+c[C54]*U[E13]+c[C64]*U[E12]);
        F3[V3] = -inv_rho*(c[C31]*U[E11]+c[C32]*U[E22]+c[C33]*U[E33]+c[C43]*U[E23]+c[C53]*U[E13]+c[C63]*U[E12]);

        F3[E11] = 0.0;
        F3[E22] = 0.0;
        F3[E33] = -U[V3];
        F3[E23] = -U[V2];
        F3[E13] = -U[V1];
        F3[E12] = 0.0;
    }
#endif
    // ----------------------------------------------------------------

    AMREX_GPU_HOST_DEVICE
    void _eval_AnU_(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/this->rho[dom];
        const amrex::Real * c = this->cSE[dom];
        const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);

#if (AMREX_SPACEDIM == 2)
        AnU[V1] = U[S11]*n1+U[S12]*n2;
        AnU[V2] = U[S12]*n1+U[S22]*n2;

        AnU[S11] = -(c[C11]*n1+c[C61]*n2)*U[V1]-(c[C61]*n1+c[C21]*n2)*U[V2];
        AnU[S22] = -(c[C21]*n1+c[C62]*n2)*U[V1]-(c[C62]*n1+c[C22]*n2)*U[V2];
        AnU[S12] = -(c[C61]*n1+c[C66]*n2)*U[V1]-(c[C66]*n1+c[C62]*n2)*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
        AnU[V1]  = (c[C11]*n1+c[C61]*n2+c[C51]*n3)*U[E11];
        AnU[V1] += (c[C21]*n1+c[C62]*n2+c[C52]*n3)*U[E22];
        AnU[V1] += (c[C31]*n1+c[C63]*n2+c[C53]*n3)*U[E33];
        AnU[V1] += (c[C41]*n1+c[C64]*n2+c[C54]*n3)*U[E23];
        AnU[V1] += (c[C51]*n1+c[C65]*n2+c[C55]*n3)*U[E13];
        AnU[V1] += (c[C61]*n1+c[C66]*n2+c[C65]*n3)*U[E12];
        
        AnU[V2]  = (c[C61]*n1+c[C21]*n2+c[C41]*n3)*U[E11];
        AnU[V2] += (c[C62]*n1+c[C22]*n2+c[C42]*n3)*U[E22];
        AnU[V2] += (c[C63]*n1+c[C32]*n2+c[C43]*n3)*U[E33];
        AnU[V2] += (c[C64]*n1+c[C42]*n2+c[C44]*n3)*U[E23];
        AnU[V2] += (c[C65]*n1+c[C52]*n2+c[C54]*n3)*U[E13];
        AnU[V2] += (c[C66]*n1+c[C62]*n2+c[C64]*n3)*U[E12];
        
        AnU[V3]  = (c[C51]*n1+c[C41]*n2+c[C31]*n3)*U[E11];
        AnU[V3] += (c[C52]*n1+c[C42]*n2+c[C32]*n3)*U[E22];
        AnU[V3] += (c[C53]*n1+c[C43]*n2+c[C33]*n3)*U[E33];
        AnU[V3] += (c[C54]*n1+c[C44]*n2+c[C43]*n3)*U[E23];
        AnU[V3] += (c[C55]*n1+c[C54]*n2+c[C53]*n3)*U[E13];
        AnU[V3] += (c[C65]*n1+c[C64]*n2+c[C63]*n3)*U[E12];
        
        AnU[E11] = -U[V1]*n1;
        AnU[E22] = -U[V2]*n2;
        AnU[E33] = -U[V3]*n3;
        AnU[E23] = -U[V3]*n2-U[V2]*n3;
        AnU[E13] = -U[V3]*n1-U[V1]*n3;
        AnU[E12] = -U[V2]*n1-U[V1]*n2;
#endif

        AMREX_D_TERM
        (
            AnU[V1] *= -inv_rho;,
            AnU[V2] *= -inv_rho;,
            AnU[V3] *= -inv_rho;
        )
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

    // PDES SYSTEM ====================================================
    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        for (int d = 0; d < N_DOM; ++d)
        {
            const amrex::Real * Ud = &U[d*N_VE];
            AMREX_D_TERM
            (
                amrex::Real * F1d = &F1[d*N_VE];,
                amrex::Real * F2d = &F2[d*N_VE];,
                amrex::Real * F3d = &F3[d*N_VE];
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

    // BODY LOAD ======================================================
    AMREX_GPU_HOST_DEVICE
    void F_B(const amrex::Real & t, const amrex::Real * x,
             amrex::Real * B) const
    {
        for (int u = 0; u < (N_DOM*N_VE); ++u)
        {
            B[u] = 0.0;
        }
    }
    // ================================================================
};
// ####################################################################