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
// --------------------------------------------------------------------

// PDES PARAMETERS ----------------------------------------------------
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
template <int N_DOM>
class ELASTIC_WAVES_BASE
{
protected:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    amrex::Real density[N_DOM];
    amrex::Real cSE[N_DOM][N_S*N_S], sES[N_DOM][N_S*N_S];

#if (AMREX_SPACEDIM == 2)
    const amrex::Real I1[N_S*N_V] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const amrex::Real I2[N_S*N_V] = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0};
#endif
#if (AMREX_SPACEDIM == 3)
    const amrex::Real I1[N_S*N_V] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    const amrex::Real I2[N_S*N_V] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const amrex::Real I3[N_S*N_V] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
#endif
    amrex::Real AMREX_D_DECL(A1[N_DOM][N_VS*N_VS], A2[N_DOM][N_VS*N_VS], A3[N_DOM][N_VS*N_VS]);
    amrex::Real AMREX_D_DECL(A1_w[N_DOM][N_VS], A2_w[N_DOM][N_VS], A3_w[N_DOM][N_VS]);
    amrex::Real AMREX_D_DECL(A1_v[N_DOM][N_VS*N_VS], A2_v[N_DOM][N_VS*N_VS], A3_v[N_DOM][N_VS*N_VS]);
    amrex::Real AMREX_D_DECL(iA1_v[N_DOM][N_VS*N_VS], iA2_v[N_DOM][N_VS*N_VS], iA3_v[N_DOM][N_VS*N_VS]);

    // LARGEST WAVE SPEED
    amrex::Real max_wave_speed;
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES_BASE(const amrex::Vector<std::string> & material_type,
                       const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real AMREX_D_DECL(I1T[N_V*N_S], I2T[N_V*N_S], I3T[N_V*N_S]);
        // ------------------------------------------------------------

        // AUXILIARY MATRICES
        AMREX_D_TERM
        (
            amrex::DG_utils::transpose(N_S, N_V, this->I1, I1T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I2, I2T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I3, I3T);
        )

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
            this->density[d] = m_prop[0];
            const amrex::Real inv_rho = 1.0/this->density[d];

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
                this->cSE[d][0+0*3] = c3D[0+0*6]; this->cSE[d][0+1*3] = c3D[0+1*6]; this->cSE[d][0+2*3] = c3D[0+5*6];
                this->cSE[d][1+0*3] = c3D[1+0*6]; this->cSE[d][1+1*3] = c3D[1+1*6]; this->cSE[d][1+2*3] = c3D[1+5*6];
                this->cSE[d][2+0*3] = c3D[5+0*6]; this->cSE[d][2+1*3] = c3D[5+1*6]; this->cSE[d][2+2*3] = c3D[5+5*6];

                amrex::DG_utils::matinv(3, this->cSE[d], this->sES[d]);
#endif
#if (AMREX_SPACEDIM == 3)
                std::copy(c3D, c3D+36, this->cSE[d]);
                std::copy(s3D, s3D+36, this->sES[d]);
#endif
            }
            else if ((m_type.compare("Hexagonal-2D") == 0) && (AMREX_SPACEDIM == 2))
            {
                const amrex::Real c11 = m_prop[1];
                const amrex::Real c33 = m_prop[2];
                const amrex::Real c44 = m_prop[3];
                const amrex::Real c13 = m_prop[4];

                this->cSE[d][0+0*3] = c11; this->cSE[d][0+1*3] = c13; this->cSE[d][0+2*3] = 0.0;
                this->cSE[d][1+0*3] = c13; this->cSE[d][1+1*3] = c33; this->cSE[d][1+2*3] = 0.0;
                this->cSE[d][2+0*3] = 0.0; this->cSE[d][2+1*3] = 0.0; this->cSE[d][2+2*3] = c44;

                amrex::DG_utils::matinv(3, this->cSE[d], this->sES[d]);
            }
            else
            {
                amrex::Print() << std::endl;
                amrex::Print() << "ERROR: IBVP_WavesInSolids.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE" << std::endl;
                amrex::Print() << "| Unexpected material type: " << m_type << " with dimension " << AMREX_SPACEDIM << "." << std::endl;
                exit(-1);
            }

            // AUXILIARY MATRICES
            AMREX_D_TERM
            (
                amrex::Real cI1[N_S*N_V];,
                amrex::Real cI2[N_S*N_V];,
                amrex::Real cI3[N_S*N_V];
            )
            AMREX_D_TERM
            (
                amrex::DG_utils::matmul(N_S, N_S, N_V, this->cSE[d], this->I1, cI1);,
                amrex::DG_utils::matmul(N_S, N_S, N_V, this->cSE[d], this->I2, cI2);,
                amrex::DG_utils::matmul(N_S, N_S, N_V, this->cSE[d], this->I3, cI3);
            )

            // OPERATORS
            AMREX_D_TERM
            (
                std::fill(this->A1[d], this->A1[d]+N_VS*N_VS, 0.0);,
                std::fill(this->A2[d], this->A2[d]+N_VS*N_VS, 0.0);,
                std::fill(this->A3[d], this->A3[d]+N_VS*N_VS, 0.0);
            )

            for (int c = N_V; c < N_VS; ++c)
            for (int r = 0; r < N_V; ++r)
            {
                AMREX_D_TERM
                (
                    this->A1[d][r+c*N_VS] = -inv_rho*I1T[r+(c-N_V)*N_V];,
                    this->A2[d][r+c*N_VS] = -inv_rho*I2T[r+(c-N_V)*N_V];,
                    this->A3[d][r+c*N_VS] = -inv_rho*I3T[r+(c-N_V)*N_V];
                )
            }
            for (int c = 0; c < N_V; ++c)
            for (int r = N_V; r < N_VS; ++r)
            {
                AMREX_D_TERM
                (
                    this->A1[d][r+c*N_VS] = -cI1[r-N_V+c*N_S];,
                    this->A2[d][r+c*N_VS] = -cI2[r-N_V+c*N_S];,
                    this->A3[d][r+c*N_VS] = -cI3[r-N_V+c*N_S];
                )
            }

            // EIGENVALUES
            {
                amrex::Real wIm[N_VS];

                AMREX_D_TERM
                (
                    amrex::DG_utils::eig(false, true, N_VS, this->A1[d], this->A1_w[d], wIm, nullptr, this->A1_v[d]);,
                    amrex::DG_utils::eig(false, true, N_VS, this->A2[d], this->A2_w[d], wIm, nullptr, this->A2_v[d]);,
                    amrex::DG_utils::eig(false, true, N_VS, this->A3[d], this->A3_w[d], wIm, nullptr, this->A3_v[d]);
                )
                AMREX_D_TERM
                (
                    amrex::DG_utils::matinv(N_VS, this->A1_v[d], this->iA1_v[d]);,
                    amrex::DG_utils::matinv(N_VS, this->A2_v[d], this->iA2_v[d]);,
                    amrex::DG_utils::matinv(N_VS, this->A3_v[d], this->iA3_v[d]);
                )
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
                const amrex::Real rho = this->density[d];
#if (AMREX_SPACEDIM == 2)
                const amrex::Real G = this->cSE[d][2+2*3];
                const amrex::Real lam = this->cSE[d][0+1*3];
#endif
#if (AMREX_SPACEDIM == 3)
                const amrex::Real G = this->cSE[d][5+5*6];
                const amrex::Real lam = this->cSE[d][0+1*6];
#endif
                this->max_wave_speed = std::max(this->max_wave_speed, std::sqrt((2.0*G+lam)/rho));
            }
            else if ((m_type.compare("Hexagonal-2D") == 0) && (AMREX_SPACEDIM == 2))
            {
                const int nth = 45;
                const amrex::Real dth = M_PI/(nth-1);
                amrex::Real un[AMREX_SPACEDIM], th;
                amrex::Real InTcIn[N_V*N_V], wRe[N_V], wIm[N_V], c;

                for (int i = 0; i < nth; ++i)
                {
                    th = i*dth;
                    un[0] = std::cos(th);
                    un[1] = std::sin(th);

                    this->_eval_InTcIn_(this->density[d], this->cSE[d], un, InTcIn);

                    amrex::DG_utils::eig(N_V, InTcIn, wRe, wIm);

                    wRe[0] = std::abs(wRe[0]);
                    wRe[1] = std::abs(wRe[1]);

                    if (wRe[0] > wRe[1]) std::swap(wRe[0], wRe[1]);

                    c = std::max(wRe[0], wRe[1]);
                }

                this->max_wave_speed = std::max(this->max_wave_speed, c);
            }
            else
            {
                amrex::Print() << std::endl;
                amrex::Print() << "ERROR: IBVP_WavesInSolids.H - ELASTIC_WAVES_BASE::ELASTIC_WAVES_BASE" << std::endl;
                amrex::Print() << "| Unexpected material type: " << m_type << " with dimension " << AMREX_SPACEDIM << "." << std::endl;
                exit(-1);
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void _eval_T_(const amrex::Real * un, amrex::Real * T, amrex::Real * iT) const
    {
        for (int k = 0; k < N_VS*N_VS; ++k)
        {
            T[k] = 0.0;
            iT[k] = 0.0;
        }
#if (AMREX_SPACEDIM == 2)
        T[0+0*N_VS] =  un[0]; T[0+1*N_VS] = un[1];
        T[1+0*N_VS] = -un[1]; T[1+1*N_VS] = un[0];
        
        T[2+2*N_VS] = un[0]*un[0]; T[2+3*N_VS] = un[1]*un[1]; T[2+4*N_VS] = 2.0*un[0]*un[1];
        T[3+2*N_VS] = un[1]*un[1]; T[3+3*N_VS] = un[0]*un[0]; T[3+4*N_VS] = -2.0*un[0]*un[1];
        T[4+2*N_VS] = -un[0]*un[1]; T[4+3*N_VS] = un[0]*un[1]; T[4+4*N_VS] = un[0]*un[0]-un[1]*un[1];

        iT[0+0*N_VS] = un[0]; iT[0+1*N_VS] = -un[1];
        iT[1+0*N_VS] = un[1]; iT[1+1*N_VS] = un[0];
        
        iT[2+2*N_VS] = un[0]*un[0]; iT[2+3*N_VS] = un[1]*un[1]; iT[2+4*N_VS] = -2.0*un[0]*un[1];
        iT[3+2*N_VS] = un[1]*un[1]; iT[3+3*N_VS] = un[0]*un[0]; iT[3+4*N_VS] = 2.0*un[0]*un[1];
        iT[4+2*N_VS] = un[0]*un[1]; iT[4+3*N_VS] = -un[0]*un[1]; iT[4+4*N_VS] = un[0]*un[0]-un[1]*un[1];
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "Hello! _eval_T_" << std::endl;
exit(-1);
#endif
    }

    void _eval_InTcIn_(const amrex::Real rho, const amrex::Real * cSE_, const amrex::Real * un, amrex::Real * InTcIn) const
    {
        const amrex::Real inv_rho = 1.0/rho;

        amrex::Real In[N_S*N_V], InT[N_S*N_V], cIn[N_S*N_V];

        for (int c = 0; c < N_V; ++c)
        for (int r = 0; r < N_S; ++r)
        {
            AMREX_D_TERM
            (
                In[r+c*N_S]  = this->I1[r+c*N_S]*un[0];,
                In[r+c*N_S] += this->I2[r+c*N_S]*un[1];,
                In[r+c*N_S] += this->I3[r+c*N_S]*un[2];
            )

            AMREX_D_TERM
            (
                InT[c+r*N_V]  = this->I1[r+c*N_S]*un[0];,
                InT[c+r*N_V] += this->I2[r+c*N_S]*un[1];,
                InT[c+r*N_V] += this->I3[r+c*N_S]*un[2];
            )
        }

        amrex::DG_utils::matmul(N_S, N_S, N_V, cSE_, In, cIn);
        amrex::DG_utils::matmul(N_V, N_S, N_V, InT, cIn, InTcIn);

        for (int k = 0; k < (N_V*N_V); ++k)
        {
            InTcIn[k] *= inv_rho;
        }
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
        dt = 0.3*h/(this->max_wave_speed);
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
            AMREX_D_TERM
            (
                amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1[d], &U[d*N_VS], &F1[d*N_VS]);,
                amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2[d], &U[d*N_VS], &F2[d*N_VS]);,
                amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A3[d], &U[d*N_VS], &F3[d*N_VS]);
            )
        }
    }
    // ================================================================
};
// ####################################################################