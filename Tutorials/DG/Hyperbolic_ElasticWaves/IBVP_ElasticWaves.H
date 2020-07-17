/* ELASTIC WAVE EQUATION */

// IBVP PARAMETERS ####################################################
// PROBLEM ------------------------------------------------------------
#define PROBLEM_SHM_PARTICLES_REF 1
#define PROBLEM_SHM_PARTICLES 11

#define PROBLEM 11
// --------------------------------------------------------------------

// DISTANCE FUNCTIONS DESCRIPTION -------------------------------------
#define PHI_TYPE_ONE_PHASE -1
#define PHI_TYPE_TWO_PHASES_PERIODIC 0
#define PHI_TYPE_PARTICLES 2

#if (PROBLEM == PROBLEM_SHM_PARTICLES_REF)

#define PHI_TYPE -1

#elif (PROBLEM == PROBLEM_SHM_PARTICLES)

#define PHI_TYPE 2

#else

#define PHI_TYPE 0

#endif
// --------------------------------------------------------------------

// NUMBER OF DISTANCE FUNCTIONS/DOMAINS/UNKNOWN FIELDS ----------------
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

#define N_PHI 1
#define N_DOM 1

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))

#define N_PHI 1
#define N_DOM 2

#endif


// NUMBER OF VELOCITY COMPONENTS AND STRESS COMPONENTS
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

#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

#define N_U N_VS

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))

#define N_U (2*N_VS)

#endif
// --------------------------------------------------------------------

// PDES PARAMETERS ----------------------------------------------------
// --------------------------------------------------------------------

// DG-METHOD PARAMETERS -----------------------------------------------
// --------------------------------------------------------------------
// ####################################################################


// PATCH INFORMATIONS #################################################
// ####################################################################


// IBVP CLASS #########################################################
class ELASTIC_WAVES
{
private:
    // DATA MEMBERS ===================================================
    // PARAMETERS

    // VARIABLES
    amrex::Real density;
    amrex::Real C[N_S*N_S], S[N_S*N_S];
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
    amrex::Real AMREX_D_DECL(A1[N_VS*N_VS], A2[N_VS*N_VS], A3[N_VS*N_VS]);
    amrex::Real AMREX_D_DECL(A1_w[N_VS], A2_w[N_VS], A3_w[N_VS]);
    amrex::Real AMREX_D_DECL(A1_v[N_VS*N_VS], A2_v[N_VS*N_VS], A3_v[N_VS*N_VS]);
    amrex::Real AMREX_D_DECL(iA1_v[N_VS*N_VS], iA2_v[N_VS*N_VS], iA3_v[N_VS*N_VS]);

    // VARIABLES FOR INITIAL CONDITIONS
    amrex::Real ICs_un[AMREX_SPACEDIM];
    amrex::Real ICs_P[N_VS*N_VS];
    amrex::Real ICs_P_w[N_VS];
    amrex::Real ICs_P_v[N_VS*N_VS];

    // LARGEST WAVE SPEED
    amrex::Real max_wave_speed;

    // AUXILIARY DATA MEMBERS
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const std::string & material_type,
                  const amrex::Real * material_properties,
                  const bool plane_stress_flag = false,
                  const amrex::Real shear_factor = 1.0)
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real rho = material_properties[0];
        const amrex::Real inv_rho = 1.0/rho;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        // MATERIAL PROPERTIES
        amrex::Real s3D[36], c3D[36];
#if (AMREX_SPACEDIM == 2)
        amrex::Real c2D[9];
#endif

        // AUXILIARY MATRICES
        AMREX_D_TERM
        (
            amrex::Real I1T[N_V*N_S];
            amrex::Real cI1[N_S*N_V];,
            amrex::Real I2T[N_V*N_S];
            amrex::Real cI2[N_S*N_V];,
            amrex::Real I3T[N_V*N_S];
            amrex::Real cI3[N_S*N_V];
        )
        // ------------------------------------------------------------

        // DENSITY ---------
        this->density = rho;
        // -----------------

        // INITIALIZATION ---------------------------------------------
        std::fill(s3D, s3D+36, 0.0);
        std::fill(c3D, c3D+36, 0.0);
#if (AMREX_SPACEDIM == 2)
        std::fill(c2D, c2D+9, 0.0);
#endif

        AMREX_D_TERM
        (
            amrex::DG_utils::transpose(N_S, N_V, this->I1, I1T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I2, I2T);,
            amrex::DG_utils::transpose(N_S, N_V, this->I3, I3T);
        )

        AMREX_D_TERM
        (
            std::fill(A1, A1+N_VS*N_VS, 0.0);,
            std::fill(A2, A2+N_VS*N_VS, 0.0);,
            std::fill(A3, A3+N_VS*N_VS, 0.0);
        )
        // ------------------------------------------------------------

        // COMPUTE THE COMPLIANCE/STIFFNESS MATRIX ON THE BASIS OF THE
        // INPUT MATERIAL TYPE ----------------------------------------
        if (material_type.compare("Isotropic") == 0)
        {
            // Input properties
            const amrex::Real E = material_properties[1];
            const amrex::Real nu = material_properties[2];
            const amrex::Real G = 0.5*E/(1.0+nu);

            s3D[0+0*6] = 1.0/E; s3D[0+1*6] = -nu/E; s3D[0+2*6] = -nu/E;
            s3D[1+0*6] = -nu/E; s3D[1+1*6] = 1.0/E; s3D[1+2*6] = -nu/E;
            s3D[2+0*6] = -nu/E; s3D[2+1*6] = -nu/E; s3D[2+2*6] = 1.0/E;
            s3D[3+3*6] = 1.0/G;
            s3D[4+4*6] = 1.0/G;
            s3D[5+5*6] = 1.0/G;
        }
        else
        {
            amrex::Print() << std::endl;
            amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::ELASTIC_WAVES" << std::endl;
            amrex::Print() << "| Unexpected material type: " << material_type << std::endl;
            exit(-1);
        }

        if ((material_type.compare("Isotropic") == 0) || (material_type.compare("Orthotropic") == 0))
        {
            if (plane_stress_flag)
            {
                s3D[0+2*6] = 0.0;
                s3D[1+2*6] = 0.0;
                s3D[2+0*6] = 0.0;
                s3D[2+1*6] = 0.0;

                s3D[3+3*6] = s3D[3+3*6]/shear_factor;
                s3D[4+4*6] = s3D[4+4*6]/shear_factor;
            }

            amrex::DG_utils::matinv(6, s3D, c3D);
        }

#if (AMREX_SPACEDIM == 2)
        c2D[0+0*3] = c3D[0+0*6]; c2D[0+1*3] = c3D[0+1*6]; c2D[0+2*3] = c3D[0+5*6];
        c2D[1+0*3] = c3D[1+0*6]; c2D[1+1*3] = c3D[1+1*6]; c2D[1+2*3] = c3D[1+5*6];
        c2D[2+0*3] = c3D[5+0*6]; c2D[2+1*3] = c3D[5+1*6]; c2D[2+2*3] = c3D[5+5*6];
#endif

        // STORE THE MATRICES
#if (AMREX_SPACEDIM == 2)
        amrex::DG_utils::matinv(3, c2D, this->S);
#endif
#if (AMREX_SPACEDIM == 3)
        std::copy(s3D, s3D+36, this->S);
#endif
        // ------------------------------------------------------------

        // AUXILIARY MATRICES -----------------------------------------
#if (AMREX_SPACEDIM == 2)
        amrex::DG_utils::matmul(N_S, N_S, N_V, c2D, this->I1, cI1);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c2D, this->I2, cI2);
#endif
#if (AMREX_SPACEDIM == 3)
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I1, cI1);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I2, cI2);
        amrex::DG_utils::matmul(N_S, N_S, N_V, c3D, this->I3, cI3);
#endif
        // ------------------------------------------------------------

        // DIFFERENTIAL OPERATORS -------------------------------------
        for (int c = N_V; c < (N_V+N_S); ++c)
        for (int r = 0; r < N_V; ++r)
        {
            AMREX_D_TERM
            (
                this->A1[r+c*N_VS] = -inv_rho*I1T[r+(c-N_V)*N_V];,
                this->A2[r+c*N_VS] = -inv_rho*I2T[r+(c-N_V)*N_V];,
                this->A3[r+c*N_VS] = -inv_rho*I3T[r+(c-N_V)*N_V];
            )
        }

        for (int c = 0; c < N_V; ++c)
        for (int r = N_V; r < (N_V+N_S); ++r)
        {
            AMREX_D_TERM
            (
                this->A1[r+c*N_VS] = -cI1[r-N_V+c*N_S];,
                this->A2[r+c*N_VS] = -cI2[r-N_V+c*N_S];,
                this->A3[r+c*N_VS] = -cI3[r-N_V+c*N_S];
            )
        }

        // AND THEIR EIGENVALUES
        {
            amrex::Real wIm[N_VS];

            AMREX_D_TERM
            (
                amrex::DG_utils::eig(false, true, N_VS, this->A1, this->A1_w, wIm, nullptr, this->A1_v);,
                amrex::DG_utils::eig(false, true, N_VS, this->A2, this->A2_w, wIm, nullptr, this->A2_v);,
                amrex::DG_utils::eig(false, true, N_VS, this->A3, this->A3_w, wIm, nullptr, this->A3_v);
            )
            AMREX_D_TERM
            (
                amrex::DG_utils::matinv(N_VS, this->A1_v, this->iA1_v);,
                amrex::DG_utils::matinv(N_VS, this->A2_v, this->iA2_v);,
                amrex::DG_utils::matinv(N_VS, this->A3_v, this->iA3_v);
            )
        }
        // ------------------------------------------------------------

        // INITIAL CONDITIONS -----------------------------------------
        AMREX_D_TERM
        (
            this->ICs_un[0] = 1.0;,
            this->ICs_un[1] = 0.0;,
            this->ICs_un[2] = 0.0;
        )
        {
            amrex::Real tmp = this->ICs_un[0]*this->ICs_un[0];
            for (int dim = 1; dim < AMREX_SPACEDIM; ++dim)
                tmp += this->ICs_un[dim]*this->ICs_un[dim];
            tmp = 1.0/std::sqrt(tmp);

            AMREX_D_TERM
            (
                this->ICs_un[0] *= tmp;,
                this->ICs_un[1] *= tmp;,
                this->ICs_un[2] *= tmp;
            )
        }

        for (int c = 0; c < N_VS; ++c)
        for (int r = 0; r < N_VS; ++r)
        {
            AMREX_D_TERM
            (
                this->ICs_P[r+c*N_VS]  = this->A1[r+c*N_VS]*this->ICs_un[0];,
                this->ICs_P[r+c*N_VS] += this->A2[r+c*N_VS]*this->ICs_un[1];,
                this->ICs_P[r+c*N_VS] += this->A3[r+c*N_VS]*this->ICs_un[2];
            )
        }

        // COMPUTE THE EIGENVALUES AND EIGENVECTOR
        // The i-th eigenvector will be stored in the i-th column of
        // the array ICs_P_v. Remember that we use row-major format.
        {
            amrex::Real wIm[N_VS];

            amrex::DG_utils::eig(false, true, N_VS, this->ICs_P, this->ICs_P_w, wIm, nullptr, this->ICs_P_v);
        }
        // ------------------------------------------------------------

        // FIND THE LARGEST WAVE SPEED --------------------------------
        if (material_type.compare("Isotropic") == 0)
        {
#if (AMREX_SPACEDIM == 2)
            const amrex::Real G = c2D[2+2*3];
            const amrex::Real lam = c2D[0+1*3];
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real G = c3D[5+5*6];
            const amrex::Real lam = c3D[0+1*6];
#endif
            this->max_wave_speed = std::sqrt((2.0*G+lam)/rho);
        }
        else
        {
            amrex::Print() << std::endl;
            amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::ELASTIC_WAVES" << std::endl;
            amrex::Print() << "| Unexpected material type: " << material_type << std::endl;
            exit(-1);
        }
        // ------------------------------------------------------------
    }

    ELASTIC_WAVES(const std::string & material_type,
                  const std::vector<amrex::Real> & material_properties,
                  const bool plane_stress_flag = false,
                  const amrex::Real shear_factor = 1.0)
    :
    ELASTIC_WAVES(material_type, material_properties.data(), plane_stress_flag, shear_factor)
    {}
    // ================================================================

    // DISTRUCTOR =====================================================
    // ================================================================

    // AUXILIARY FUNCTIONS ============================================
    AMREX_GPU_HOST_DEVICE
    void _eval_T_(const amrex::Real * un, amrex::Real * T) const
    {
        for (int k = 0; k < N_VS*N_VS; ++k)
        {
            T[k] = 0.0;
        }
#if (AMREX_SPACEDIM == 2)
        T[0+0*N_VS] =  un[0]; T[0+1*N_VS] = un[1];
        T[1+0*N_VS] = -un[1]; T[1+1*N_VS] = un[0];
        
        T[2+2*N_VS] = un[0]*un[0]; T[2+3*N_VS] = un[1]*un[1]; T[2+4*N_VS] = 2.0*un[0]*un[1];
        T[3+2*N_VS] = un[1]*un[1]; T[3+3*N_VS] = un[0]*un[0]; T[3+4*N_VS] = -2.0*un[0]*un[1];
        T[4+2*N_VS] = -un[0]*un[1]; T[4+3*N_VS] = un[0]*un[1]; T[4+4*N_VS] = un[0]*un[0]-un[1]*un[1];
#endif
    }
    
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
    }
    
    AMREX_GPU_HOST_DEVICE
    void _eval_P_(const amrex::Real * un, amrex::Real * P) const
    {
        for (int c = 0; c < N_VS; ++c)
        for (int r = 0; r < N_VS; ++r)
        {
            AMREX_D_TERM
            (
                P[r+c*N_VS]  = this->A1[r+c*N_VS]*un[0];,
                P[r+c*N_VS] += this->A2[r+c*N_VS]*un[1];,
                P[r+c*N_VS] += this->A3[r+c*N_VS]*un[2];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _eval_Pn_(const amrex::Real * un, amrex::Real * Pn) const
    {
        amrex::Real P[N_VS*N_VS], T[N_VS*N_VS], iT[N_VS*N_VS];
        this->_eval_P_(un, Pn);
        this->_eval_T_(un, T, iT);

        amrex::DG_utils::matmul(N_VS, N_VS, N_VS, Pn, iT, P);
        amrex::DG_utils::matmul(N_VS, N_VS, N_VS, T, P, Pn);
    }
    // ================================================================

    // GRID MAPPING ===================================================
    AMREX_GPU_HOST_DEVICE
    void F_MAP(const amrex::Real & t, const amrex::Real * xi,  amrex::Real * x) const
    {

    }
    // ================================================================

    // DISTANCE FUNCTION ==============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

        const amrex::Real PHI[N_PHI] = {-1.0};

#elif (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)

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

        const amrex::Real PHI[N_PHI] =
        {
            -phi
        };

#elif (PHI_TYPE == PHI_TYPE_PARTICLES)
        const int Np = 6;
        const amrex::Real rp[Np] = {0.02, 0.03, 0.026, 0.018, 0.022, 0.024};
        const amrex::Real x0[Np] = {0.07, 0.15, 0.03, 0.04, 0.17, 0.1};
        const amrex::Real x1[Np] = {-0.48, -0.45, -0.425, -0.55, -0.527, -0.57};
        const amrex::Real xi = x[0]-std::floor(5.0*x[0])*0.2;
        const amrex::Real eta = x[1];

        amrex::Real phi;

        phi = rp[0]*rp[0]-(xi-x0[0])*(xi-x0[0])-(eta-x1[0])*(eta-x1[0]);
        for (int p = 1; p < Np; ++p)
        {
            phi = std::max(phi, rp[p]*rp[p]-(xi-x0[p])*(xi-x0[p])-(eta-x1[p])*(eta-x1[p]));
        }

        const amrex::Real PHI[N_PHI] =
        {
            phi
        };

#else
        amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::F_PHI" << std::endl;
        amrex::Print() << "| Unexpected distance function(s)." << std::endl;
        exit(-1);
#endif



        return PHI[ph];
    }
    // ================================================================

    // RELATIONSHIP AMONG DISTANCE FUNCTIONS AND DOMAINS ==============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1
        };

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))

        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1,

            // dom = 1
            0, +1
        };

#else
        amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::F_DOM2PHI" << std::endl;
        amrex::Print() << "| Unexpected distance function(s) - domain(s) relationships." << std::endl;
        exit(-1);
#endif

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
#define ICS_ZERO -1
#define ICS_EIGEN_STATE 0

#if ((PROBLEM == PROBLEM_SHM_PARTICLES_REF) || \
     (PROBLEM == PROBLEM_SHM_PARTICLES))

#define ICS -1

#else

#define ICS 0

#endif

    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        amrex::Real res;
#if (ICS == ICS_ZERO)
        res = 0.0;
#endif
#if (ICS == ICS_EIGEN_STATE)
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ICs_un[0], this->ICs_un[1], this->ICs_un[2])};
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real f0 = std::sin(2.0*M_PI*x_dot_un);
#if (AMREX_SPACEDIM == 2)
        const int iv = 3;
#endif
#if (AMREX_SPACEDIM == 3)
        const int iv = 4;
#endif
        
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

        res = this->ICs_P_v[u+iv*N_VS]*f0;

#elif (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)

        res = (u < N_VS) ? (this->ICs_P_v[u+iv*N_VS]*f0) : (this->ICs_P_v[u-N_VS+iv*N_VS]*f0);

#endif

#endif
        return res;
    }
    // ================================================================
    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

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

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))

#if (AMREX_SPACEDIM == 2)
        const int DOM[N_U] =
        {
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1
        };
#endif

#else
        amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::F_U2DOM" << std::endl;
        amrex::Print() << "| Unexpected unknown field(s) - domain(s) relationships." << std::endl;
        exit(-1);
#endif

        return DOM[u];
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * U) const
    {
#if (ICS == ICS_EIGEN_STATE)
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ICs_un[0], this->ICs_un[1], this->ICs_un[2])};
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
#if (AMREX_SPACEDIM == 2)
        const int iv = 3;
#endif
#if (AMREX_SPACEDIM == 3)
        const int iv = 4;
#endif
        const amrex::Real w = this->ICs_P_w[iv];
        const amrex::Real f0 = std::sin(2.0*M_PI*(x_dot_un-w*t));

#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

        for (int u = 0; u < N_VS; ++u)
        {
            U[u] = this->ICs_P_v[u+iv*N_VS]*f0;
        }

#elif (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)

        for (int u = 0; u < N_VS; ++u)
        {
            U[u] = this->ICs_P_v[u+iv*N_VS]*f0;
        }

        for (int u = N_VS; u < (2*N_VS); ++u)
        {
            U[u] = this->ICs_P_v[u-N_VS+iv*N_VS]*f0;
        }

#endif

#endif
    }

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
#if (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)
        const int ph = 0;
        const amrex::Real phi = this->F_PHI(ph, t, x);
#endif
        
#if (ICS == ICS_EIGEN_STATE)
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ICs_un[0], this->ICs_un[1], this->ICs_un[2])};
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
#if (AMREX_SPACEDIM == 2)
        const int iv = 3;
#endif
#if (AMREX_SPACEDIM == 3)
        const int iv = 4;
#endif
        const amrex::Real w = this->ICs_P_w[iv];
        const amrex::Real f0 = std::sin(2.0*M_PI*(x_dot_un-w*t));
        amrex::Real Ue[N_VS], V[N_VS];

        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] = this->ICs_P_v[u+iv*N_VS]*f0;
        }
        
        AMREX_D_TERM
        (
            V[0] = this->density*Ue[0];,
            V[1] = this->density*Ue[1];,
            V[2] = this->density*Ue[2];
        )
        amrex::DG_utils::matmul(N_S, N_S, 1, this->S, &Ue[N_V], &V[N_V]);
        norm_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            norm_x += Ue[u]*V[u];
        }

#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)
        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] -= U[u];
        }
#elif (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)
        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] -= (phi < 0.0) ? U[u] : U[u+N_VS];
        }
#endif
        AMREX_D_TERM
        (
            V[0] = this->density*Ue[0];,
            V[1] = this->density*Ue[1];,
            V[2] = this->density*Ue[2];
        )
        amrex::DG_utils::matmul(N_S, N_S, 1, this->S, &Ue[N_V], &V[N_V]);
        err_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            err_x += Ue[u]*V[u];
        }
#else
#if (!AMREX_USE_CUDA)
amrex::Print() << "Hello! F_ERROR with ICS != ICS_EIGEN_STATE" << std::endl;
exit(-1);
#endif
#endif
    }
    // ================================================================

    // BOUNDARY CONDITIONS ============================================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)
        const int NBRDOM[N_DOM] =
        {
            -1
        };

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))

        const int NBRDOM[N_DOM] =
        {
            // dom = 0
            1,

            // dom = 1
            0
        };

#else
        amrex::Print() << "ERROR: IBVP_ElasticWaves.H - ELASTIC_WAVES::F_DOM2NBRDOM" << std::endl;
        amrex::Print() << "| Unexpected neighboring domain(s) relationships." << std::endl;
        exit(-1);
#endif
        
        return NBRDOM[dom];
    }
    // ================================================================

    // PDES SYSTEM ====================================================
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
        dt = 0.25*h/(this->max_wave_speed);
#endif
        
        return dt;
    }

    AMREX_GPU_HOST_DEVICE
    void F_F(const amrex::Real & t, const amrex::Real * x,
             const amrex::Real * U,
             AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)
        AMREX_D_TERM
        (
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1, U, F1);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2, U, F2);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A3, U, F3);
        )

#elif ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
       (PHI_TYPE == PHI_TYPE_PARTICLES))
        AMREX_D_TERM
        (
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1, U, F1);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2, U, F2);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A3, U, F3);
        )

        AMREX_D_TERM
        (
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1, &U[N_VS], &F1[N_VS]);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A2, &U[N_VS], &F2[N_VS]);,
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A3, &U[N_VS], &F3[N_VS]);
        )
#endif
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUX =================
#define GRID_BCS_ABSORBING 0
#define GRID_BCS_ENCASTRE 1
#define GRID_BCS_TRACTION_FREE 2
#define GRID_BCS 2

    AMREX_GPU_HOST_DEVICE
    void _NF_base_(const amrex::Real * un,
                   const amrex::Real * U, const amrex::Real * nbr_U,
                   AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                   AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                   AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        const amrex::Real mu = this->max_wave_speed;

        for (int u = 0; u < N_U; ++u)
        {
            AMREX_D_TERM
            (
                NF1[u] = 0.5*(F1[u]+nbr_F1[u])+0.5*mu*(U[u]-nbr_U[u])*un[0];,
                NF2[u] = 0.5*(F2[u]+nbr_F2[u])+0.5*mu*(U[u]-nbr_U[u])*un[1];,
                NF3[u] = 0.5*(F3[u]+nbr_F3[u])+0.5*mu*(U[u]-nbr_U[u])*un[2];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void _NF_PHI_base_(const amrex::Real * un,
                       const amrex::Real * U, const amrex::Real * nbr_U,
                       AMREX_D_DECL(const amrex::Real * F1, const amrex::Real * F2, const amrex::Real * F3),
                       AMREX_D_DECL(const amrex::Real * nbr_F1, const amrex::Real * nbr_F2, const amrex::Real * nbr_F3),
                       AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        const amrex::Real mu = this->max_wave_speed;

#if ((PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC) || \
     (PHI_TYPE == PHI_TYPE_PARTICLES))
        for (int u = 0; u < (N_VS); ++u)
        {
            AMREX_D_TERM
            (
                NF1[u] = 0.5*(F1[u]+nbr_F1[u+N_VS])+0.5*mu*(U[u]-nbr_U[u+N_VS])*un[0];,
                NF2[u] = 0.5*(F2[u]+nbr_F2[u+N_VS])+0.5*mu*(U[u]-nbr_U[u+N_VS])*un[1];,
                NF3[u] = 0.5*(F3[u]+nbr_F3[u+N_VS])+0.5*mu*(U[u]-nbr_U[u+N_VS])*un[2];
            )
        }

        for (int u = (N_VS); u < (2*N_VS); ++u)
        {
            AMREX_D_TERM
            (
                NF1[u] = 0.5*(F1[u]+nbr_F1[u-N_VS])+0.5*mu*(U[u]-nbr_U[u-N_VS])*un[0];,
                NF2[u] = 0.5*(F2[u]+nbr_F2[u-N_VS])+0.5*mu*(U[u]-nbr_U[u-N_VS])*un[1];,
                NF3[u] = 0.5*(F3[u]+nbr_F3[u-N_VS])+0.5*mu*(U[u]-nbr_U[u-N_VS])*un[2];
            )
        }

#else

#if (!AMREX_USE_CUDA)
amrex::Print() << "Hello! _NF_PHI_base_" << std::endl;
exit(-1);
#endif

#endif
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

        this->_NF_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
    }

    // GRID BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
        amrex::Real AMREX_D_DECL(F1[N_U], F2[N_U], F3[N_U]);
        amrex::Real AMREX_D_DECL(nbr_F1[N_U], nbr_F2[N_U], nbr_F3[N_U]);

#if ((PROBLEM == PROBLEM_SHM_PARTICLES_REF) || \
     (PROBLEM == PROBLEM_SHM_PARTICLES))
        amrex::Real cv[N_VS], nbr_U[N_VS];

        // ABSORBING BCS
        if (std::abs(un[0]) > 0.5)
        {
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->iA1_v, U, cv);
            for (int u = 0; u < N_VS; ++u)
            {
                if (this->A1_w[u]*un[0] < 0.0) cv[u] = 0.0;
            }
            amrex::DG_utils::matmul(N_VS, N_VS, 1, this->A1_v, cv, nbr_U);
        }
        // PRESCRIBED (ZERO) TRACTIONS
        else if (un[1] < -0.5)
        {
            const amrex::Real t1 = 0.0;
            const amrex::Real t2 = 0.0;
            
            nbr_U[0] = U[0];
            nbr_U[1] = U[1];
            nbr_U[2] = U[2];
            nbr_U[3] = 2.0*t2-U[3];
            nbr_U[4] = 2.0*t1-U[4];
        }
        // PRESCRIBED TRACTIONS
        else if (un[1] > 0.5)
        {
            const amrex::Real X = std::exp(-500.0*x[0]*x[0]);
            const amrex::Real f = 25.0;
            const amrex::Real T = std::exp(-1000.0*(t-0.1)*(t-0.1))*std::sin(f*2.0*M_PI*t);
            const amrex::Real t1 = 0.0;
            const amrex::Real t2 = X*T;

            nbr_U[0] = U[0];
            nbr_U[1] = U[1];
            nbr_U[2] = U[2];
            nbr_U[3] = 2.0*t2-U[3];
            nbr_U[4] = 2.0*t1-U[4];
        }

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->_NF_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
#else
amrex::Print() << "Hello! F_NF_BCS" << std::endl;
exit(-1);
#endif
    }
    /*
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
amrex::Print() << "Hello! F_NF_BCS" << std::endl;
exit(-1);
    }
    */

    // INTERNAL BOUNDARY
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      AMREX_D_DECL(amrex::Real * NF1, amrex::Real * NF2, amrex::Real * NF3)) const
    {
#if (!AMREX_USE_CUDA)
amrex::Print() << "Hello! F_NF_PHI_BCS" << std::endl;
exit(-1);
#endif
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

        this->F_F(t, x, U, AMREX_D_DECL(F1, F2, F3));
        this->F_F(t, x, nbr_U, AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3));

        this->_NF_PHI_base_(un, U, nbr_U, AMREX_D_DECL(F1, F2, F3), AMREX_D_DECL(nbr_F1, nbr_F2, nbr_F3), AMREX_D_DECL(NF1, NF2, NF3));
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
        
        // VARIABLES
        amrex::Real Ue[N_U];

        this->F_EXACT(t, x, Ue);

        // FIELD EXPRESSIONS
#if (PHI_TYPE == PHI_TYPE_ONE_PHASE)

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

#elif (PHI_TYPE == PHI_TYPE_TWO_PHASES_PERIODIC)

#if (AMREX_SPACEDIM == 2)
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];

        F[5] = U[0+N_VS];
        F[6] = U[1+N_VS];
        F[7] = U[2+N_VS];
        F[8] = U[3+N_VS];
        F[9] = U[4+N_VS];

        F[10] = U[0]-Ue[0];
        F[11] = U[2]-Ue[2];
        F[12] = U[3]-Ue[3];

        F[13] = U[0+N_VS]-Ue[0];
        F[14] = U[2+N_VS]-Ue[2];
        F[15] = U[3+N_VS]-Ue[3];
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

        F[9] = U[0+N_VS];
        F[10] = U[1+N_VS];
        F[11] = U[2+N_VS];
        F[12] = U[3+N_VS];
        F[13] = U[4+N_VS];
        F[14] = U[5+N_VS];
        F[15] = U[6+N_VS];
        F[16] = U[7+N_VS];
        F[17] = U[8+N_VS];

        F[18] = U[0]-Ue[0];
        F[19] = U[3]-Ue[3];

        F[20] = U[0+N_VS]-Ue[0];
        F[21] = U[3+N_VS]-Ue[3];
#endif

#elif (PHI_TYPE == PHI_TYPE_PARTICLES)

        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];

        F[5] = U[0+N_VS];
        F[6] = U[1+N_VS];
        F[7] = U[2+N_VS];
        F[8] = U[3+N_VS];
        F[9] = U[4+N_VS];

#endif
    }
    // ================================================================
};
// ####################################################################
