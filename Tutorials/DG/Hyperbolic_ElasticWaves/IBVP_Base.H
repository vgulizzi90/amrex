//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Base.H
 * \brief Contains base constants and data structures for IBVP of elastic wave propagation.
*/


// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
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
#define N_VS (N_V+N_S)

// SOLUTION FIELDS ORDER
#if (AMREX_SPACEDIM == 1)
#define V1 0
#define S11 1
#endif

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

// ORDERING OF THE ELASTIC COEFFICIENTS
#if (AMREX_SPACEDIM == 1)
#define C11 0
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

// MATERIAL TYPE FLAGS
#define MAT_TYPE_ISOTROPIC 0
// ####################################################################



/**
 * \brief This struct contains constants and methods for the considered IBVP. 
 *
*/
template <int N_DOM>
struct ELASTIC_SOLID_BASE
{
    // DATA MEMBERS ###################################################
    // VARIABLES
    amrex::Real rho[N_DOM];
    amrex::Real cSE[N_DOM][AMREX_D_PICK(1, 6, 21)];
    amrex::Real sES[N_DOM][AMREX_D_PICK(1, 6, 21)];

    // LARGEST WAVE SPEED
    amrex::Real max_wave_speed[N_DOM];
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    ELASTIC_SOLID_BASE()
    {}

    /**
     * \brief This method initializes the IBVP.
    */
    ELASTIC_SOLID_BASE(const amrex::Vector<amrex::Vector<int>> & int_params,
                       const amrex::Vector<amrex::Vector<amrex::Real>> & params)
    {
        // EVAL THE STIFFNESS MATRIX FOR EACH DOMAIN ==================
        for (int d = 0; d < N_DOM; ++d)
        {
            const int mat_type = int_params[d][0];

            amrex::Real s[N_S*N_S];
            std::fill(s, s+N_S*N_S, 0.0);
            
            if (mat_type == MAT_TYPE_ISOTROPIC)
            {
                const amrex::Real E = params[d][1];
                const amrex::Real nu = params[d][2];
                const amrex::Real G = 0.5*E/(1.0+nu);

                // DENSITY
                this->rho[d] = params[d][0];

                // COMPLIANCE MATRIX
#if (AMREX_SPACEDIM == 1)
                s[0] = 1.0/E;
#endif
#if (AMREX_SPACEDIM == 2)
                s[0+0*N_S] = 1.0/E; s[0+1*N_S] = -nu/E;
                s[1+0*N_S] = -nu/E; s[1+1*N_S] = 1.0/E;
                s[2+2*N_S] = 1.0/G;
#endif
#if (AMREX_SPACEDIM == 3)
                s[0+0*N_S] = 1.0/E; s[0+1*N_S] = -nu/E; s[0+2*N_S] = -nu/E;
                s[1+0*N_S] = -nu/E; s[1+1*N_S] = 1.0/E; s[1+2*N_S] = -nu/E;
                s[2+0*N_S] = -nu/E; s[2+1*N_S] = -nu/E; s[2+2*N_S] = 1.0/E;
                s[3+3*N_S] = 1.0/G;
                s[4+4*N_S] = 1.0/G;
                s[5+5*N_S] = 1.0/G;
#endif

#if (AMREX_SPACEDIM == 1)
                this->sES[d][C11] = s[0];
#endif
#if (AMREX_SPACEDIM == 2)
                this->sES[d][C11] = s[0];
                this->sES[d][C21] = s[1+0*N_S]; this->sES[d][C22] = s[1+1*N_S];
                this->sES[d][C61] = s[2+0*N_S]; this->sES[d][C62] = s[2+1*N_S]; this->sES[d][C66] = s[2+2*N_S];
#endif
#if (AMREX_SPACEDIM == 3)
                this->sES[d][C11] = s[0];
                this->sES[d][C21] = s[1+0*N_S]; this->sES[d][C22] = s[1+1*N_S];
                this->sES[d][C31] = s[2+0*N_S]; this->sES[d][C32] = s[2+1*N_S]; this->sES[d][C33] = s[2+2*N_S];
                this->sES[d][C41] = s[3+0*N_S]; this->sES[d][C42] = s[3+1*N_S]; this->sES[d][C43] = s[3+2*N_S]; this->sES[d][C44] = s[3+3*N_S];
                this->sES[d][C51] = s[4+0*N_S]; this->sES[d][C52] = s[4+1*N_S]; this->sES[d][C53] = s[4+2*N_S]; this->sES[d][C54] = s[4+3*N_S]; this->sES[d][C55] = s[4+4*N_S];
                this->sES[d][C61] = s[5+0*N_S]; this->sES[d][C62] = s[5+1*N_S]; this->sES[d][C63] = s[5+2*N_S]; this->sES[d][C64] = s[5+3*N_S]; this->sES[d][C65] = s[5+4*N_S]; this->sES[d][C66] = s[5+5*N_S];
#endif

                // Inverse of compliance matrix
                int ipiv[N_S], info;
                const int l_work = N_S*N_S;
                double work[N_S*N_S];
                
                amrex::DG::linalg::dgetrf(N_S, N_S, s, N_S, ipiv, info);
                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: IBVP_Base.H\n";
                    msg += "| Could not compute the factorization of the compliance matrix.\n";
                    msg += "| dgetrf info = "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

                amrex::DG::linalg::dgetri(N_S, s, N_S, ipiv, work, l_work, info);
                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg += "ERROR: IBVP_Base.H\n";
                    msg += "| Could not compute the inverse of the compliance matrix.\n";
                    msg += "| dgetrf info = "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

#if (AMREX_SPACEDIM == 1)
                this->cSE[d][C11] = s[0];
#endif
#if (AMREX_SPACEDIM == 2)
                this->cSE[d][C11] = s[0];
                this->cSE[d][C21] = s[1+0*N_S]; this->cSE[d][C22] = s[1+1*N_S];
                this->cSE[d][C61] = s[2+0*N_S]; this->cSE[d][C62] = s[2+1*N_S]; this->cSE[d][C66] = s[2+2*N_S];
#endif
#if (AMREX_SPACEDIM == 3)
                this->cSE[d][C11] = s[0];
                this->cSE[d][C21] = s[1+0*N_S]; this->cSE[d][C22] = s[1+1*N_S];
                this->cSE[d][C31] = s[2+0*N_S]; this->cSE[d][C32] = s[2+1*N_S]; this->cSE[d][C33] = s[2+2*N_S];
                this->cSE[d][C41] = s[3+0*N_S]; this->cSE[d][C42] = s[3+1*N_S]; this->cSE[d][C43] = s[3+2*N_S]; this->cSE[d][C44] = s[3+3*N_S];
                this->cSE[d][C51] = s[4+0*N_S]; this->cSE[d][C52] = s[4+1*N_S]; this->cSE[d][C53] = s[4+2*N_S]; this->cSE[d][C54] = s[4+3*N_S]; this->cSE[d][C55] = s[4+4*N_S];
                this->cSE[d][C61] = s[5+0*N_S]; this->cSE[d][C62] = s[5+1*N_S]; this->cSE[d][C63] = s[5+2*N_S]; this->cSE[d][C64] = s[5+3*N_S]; this->cSE[d][C65] = s[5+4*N_S]; this->cSE[d][C66] = s[5+5*N_S];
#endif
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: IBVP_Base.H\n";
                msg += "| Unexpected material type flag: "+std::to_string(mat_type)+".\n";
                amrex::Abort(msg);
            }
        }
        // ============================================================

        // EVAL THE LARGEST WAVE SPEED FOR EACH DOMAIN ================
        for (int d = 0; d < N_DOM; ++d)
        {
            const int mat_type = int_params[d][0];

            if (mat_type == MAT_TYPE_ISOTROPIC)
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
                this->max_wave_speed[d] = std::sqrt((2.0*G+lam)/density);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: IBVP_Base.H\n";
                msg += "| Unexpected material type flag: "+std::to_string(mat_type)+".\n";
                amrex::Abort(msg);
            }
        }
        // ============================================================
    }
    // ################################################################

    // TIME STEP ######################################################
    /**
     * \brief Eval the maximum time step.
     *
     * \param[in] dom: index of the considered domain.
     * \param[in] dx: pointer to memory containing the sizes of the grid's element.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables at (t,x).
     *
     * \return the maximum allowed time step.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real F_DT(const int dom,
                     const amrex::Real * dx,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/,
                     const amrex::Real * /*U*/) const
    {
        // VARIABLES
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = std::min(dx[0], dx[1]);
        const amrex::Real dt = h/(this->max_wave_speed[dom]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = std::min(dx[0], std::min(dx[1], dx[2]));
        const amrex::Real dt = h/(this->max_wave_speed[dom]);
#endif
        return dt;
    }
    // ################################################################

    // AUXILIARY FUNCTIONS ############################################
    /**
     * \brief Transform solution from global reference system to unit normal reference system.
     * 
     * \param[in] un: pointer to memory containing the unit normal.
     * \param[in] U: pointer to memory containing the unknown solution fields in the global reference
     *               system.
     * \param[out] Un: pointer to memory containing the unknown solution fields in the unit normal
     *                 reference system.
    */
    AMREX_GPU_HOST_DEVICE
    void eval_U2Un(const amrex::Real * un, const amrex::Real * U, amrex::Real * Un) const
    {
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "HELLO! eval_U2Un - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];
        const amrex::Real cs = c*s;

        Un[V1] =  c*U[V1]+s*U[V2];
        Un[V2] = -s*U[V1]+c*U[V2];
        Un[S11] = c*c*U[S11]+s*s*U[S22]  +2.0*cs*U[S12];
        Un[S22] = s*s*U[S11]+c*c*U[S22]  -2.0*cs*U[S12];
        Un[S12] = -cs*U[S11]+cs*U[S22]+(c*c-s*s)*U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        Un[V1] =  cph*sth*U[V1]+sph*sth*U[V2]+cth*U[V3];
        Un[V2] =     -sph*U[V1]    +cph*U[V2];
        Un[V3] = -cph*cth*U[V1]-cth*sph*U[V2]+sth*U[V3];
        Un[S11] =  cph*cph*sth*sth*U[S11]+sph*sph*sth*sth*U[S22]+cth*cth*U[S33]      +2.0*cth*sph*sth*U[S23]      +2.0*cph*cth*sth*U[S13]  +2.0*cph*sph*sth*sth*U[S12];
        Un[S22] =          sph*sph*U[S11]        +cph*cph*U[S22]                                                                                   -2.0*cph*sph*U[S12];
        Un[S33] =  cph*cph*cth*cth*U[S11]+cth*cth*sph*sph*U[S22]+sth*sth*U[S33]      -2.0*cth*sph*sth*U[S23]      -2.0*cph*cth*sth*U[S13]  +2.0*cph*cth*cth*sph*U[S12];
        Un[S23] =      cph*cth*sph*U[S11]    -cph*cth*sph*U[S22]                             +cph*sth*U[S23]              -sph*sth*U[S13]-(cph*cph-sph*sph)*cth*U[S12];
        Un[S13] = -cph*cph*cth*sth*U[S11]-cth*sph*sph*sth*U[S22]+cth*sth*U[S33]-(cth*cth-sth*sth)*sph*U[S23]-(cth*cth-sth*sth)*cph*U[S13]  -2.0*cph*cth*sph*sth*U[S12];
        Un[S12] =     -cph*sph*sth*U[S11]    +cph*sph*sth*U[S22]                             +cph*cth*U[S23]              -cth*sph*U[S13]+(cph*cph-sph*sph)*sth*U[S12];
#endif
    }

    /**
     * \brief Transform solution from unit normal reference system to global reference system.
     * 
     * \param[in] un: pointer to memory containing the unit normal.
     * \param[in] Un: pointer to memory containing the unknown solution fields in the unit normal
     *                reference system.
     * \param[out] U: pointer to memory containing the unknown solution fields in the global reference
     *                system.
    */
    AMREX_GPU_HOST_DEVICE
    void eval_Un2U(const amrex::Real * un, const amrex::Real * Un, amrex::Real * U) const
    {
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "HELLO! eval_Un2U - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];
        const amrex::Real cs = c*s;

        U[V1] = c*Un[V1]-s*Un[V2];
        U[V2] = s*Un[V1]+c*Un[V2];
        U[S11] = c*c*Un[S11]+s*s*Un[S22]  -2.0*cs*Un[S12];
        U[S22] = s*s*Un[S11]+c*c*Un[S22]  +2.0*cs*Un[S12];
        U[S12] =  cs*Un[S11]-cs*Un[S22]+(c*c-s*s)*Un[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        U[V1] = cph*sth*Un[V1]-sph*Un[V2]-cph*cth*Un[V3];
        U[V2] = sph*sth*Un[V1]+cph*Un[V2]-cth*sph*Un[V3];
        U[V3] = cth*Un[V1]                   +sth*Un[V3];
        U[S11] = cph*cph*sth*sth*Un[S11]+sph*sph*Un[S22]+cph*cph*cth*cth*Un[S33]      +2.0*cph*cth*sph*Un[S23]  -2.0*cph*cph*cth*sth*Un[S13]      -2.0*cph*sph*sth*Un[S12];
        U[S22] = sph*sph*sth*sth*Un[S11]+cph*cph*Un[S22]+cth*cth*sph*sph*Un[S33]      -2.0*cph*cth*sph*Un[S23]  -2.0*cth*sph*sph*sth*Un[S13]      +2.0*cph*sph*sth*Un[S12];
        U[S33] =         cth*cth*Un[S11]                        +sth*sth*Un[S33]                                        +2.0*cth*sth*Un[S13];
        U[S23] =     cth*sph*sth*Un[S11]                    -cth*sph*sth*Un[S33]              +cph*sth*Un[S23]+(sth*sth-cth*cth)*sph*Un[S13]              +cph*cth*Un[S12];
        U[S13] =     cph*cth*sth*Un[S11]                    -cph*cth*sth*Un[S33]              -sph*sth*Un[S23]+(sth*sth-cth*cth)*cph*Un[S13]              -cth*sph*Un[S12];
        U[S12] = cph*sph*sth*sth*Un[S11]-cph*sph*Un[S22]+cph*cth*cth*sph*Un[S33]+(sph*sph-cph*cph)*cth*Un[S23]  -2.0*cph*cth*sph*sth*Un[S13]+(cph*cph-sph*sph)*sth*Un[S12];
#endif
    }
    
    /**
     * \brief Eval the strain vector given the stress vector as input.
     *
     * \param[in] dom: index of the domain.
     * \param[in] S: pointer to memory containing the stress vector.
     * \param[out] E: pointer to memory that will contain the strain vector.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_strain(const int dom, const amrex::Real * S, amrex::Real * E) const
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
        E[5] = s[C61]*S[0]+s[C62]*S[1]+s[C63]*S[2]+s[C64]*S[3]+s[C65]*S[4]+s[C66]*S[5];
#endif
    }

    /**
     * \brief Eval the sum of kinetic energy and elastic energy.
     *
     * \param[in] dom: index of the domain.
     * \param[in] U: pointer to memory containing the conserved variables.
     *
     * \return the sum of kinetic energy and elastic energy.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval_total_energy(const int dom, const amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real density = this->rho[dom];
        const amrex::Real * S = &U[AMREX_SPACEDIM];

        // VARIABLES
        amrex::Real E[N_S];
        amrex::Real res;

        this->eval_strain(dom, S, E);

        AMREX_D_TERM
        (
            res  = U[V1]*U[V1];,
            res += U[V2]*U[V2];,
            res += U[V3]*U[V3];
        )
        res *= density;

        for (int k = 0; k < N_S; ++k)
        {
            res += S[k]*E[k];
        }

        return 0.5*res;
    }
    
    /**
     * \brief Eval the matrix (1/rho)*InT*cSE*In.
     *
     * \param[in] dom: index of the domain.
     * \param[in] un: pointer to memory containing the unit normal vector.
     * \param[out] InTcIn: pointer to memory that will be filled with the computed matrix.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_InTcIn(const int dom, const amrex::Real * un, amrex::Real * InTcIn) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];
        const amrex::Real un11 = un[0]*un[0]*inv_rho;
#if (AMREX_SPACEDIM > 1)
        const amrex::Real un12 = un[0]*un[1]*inv_rho;
        const amrex::Real un22 = un[1]*un[1]*inv_rho;
#endif
#if (AMREX_SPACEDIM > 2)
        const amrex::Real un23 = un[1]*un[2]*inv_rho;
        const amrex::Real un13 = un[0]*un[2]*inv_rho;
        const amrex::Real un33 = un[2]*un[2]*inv_rho;
#endif

#if (AMREX_SPACEDIM == 2)
        InTcIn[0+0*AMREX_SPACEDIM] = c[C11]*un11+(c[C61]+c[C61])*un12+c[C66]*un22;
        InTcIn[1+0*AMREX_SPACEDIM] = c[C61]*un11+(c[C21]+c[C66])*un12+c[C62]*un22;
        InTcIn[1+1*AMREX_SPACEDIM] = c[C66]*un11+(c[C62]+c[C62])*un12+c[C22]*un22;

        // Symmetric components
        InTcIn[0+1*AMREX_SPACEDIM] = InTcIn[1+0*AMREX_SPACEDIM];
#endif
#if (AMREX_SPACEDIM == 3)
        InTcIn[0+0*AMREX_SPACEDIM] = c[C11]*un11+c[C66]*un22+c[C55]*un33+(c[C61]+c[C61])*un12+(c[C51]+c[C51])*un13+(c[C65]+c[C65])*un23;
        InTcIn[1+0*AMREX_SPACEDIM] = c[C61]*un11+c[C62]*un22+c[C54]*un33+(c[C21]+c[C66])*un12+(c[C41]+c[C65])*un13+(c[C52]+c[C64])*un23;
        InTcIn[2+0*AMREX_SPACEDIM] = c[C51]*un11+c[C64]*un22+c[C53]*un33+(c[C41]+c[C65])*un12+(c[C31]+c[C55])*un13+(c[C63]+c[C54])*un23;
        InTcIn[1+1*AMREX_SPACEDIM] = c[C66]*un11+c[C22]*un22+c[C44]*un33+(c[C62]+c[C62])*un12+(c[C64]+c[C64])*un13+(c[C42]+c[C42])*un23;
        InTcIn[2+1*AMREX_SPACEDIM] = c[C65]*un11+c[C42]*un22+c[C43]*un33+(c[C52]+c[C64])*un12+(c[C63]+c[C54])*un13+(c[C32]+c[C44])*un23;
        InTcIn[2+2*AMREX_SPACEDIM] = c[C55]*un11+c[C44]*un22+c[C33]*un33+(c[C54]+c[C54])*un12+(c[C53]+c[C53])*un13+(c[C43]+c[C43])*un23;

        // Symmetric components
        InTcIn[0+1*AMREX_SPACEDIM] = InTcIn[1+0*AMREX_SPACEDIM];
        InTcIn[0+2*AMREX_SPACEDIM] = InTcIn[2+0*AMREX_SPACEDIM];
        InTcIn[1+2*AMREX_SPACEDIM] = InTcIn[2+1*AMREX_SPACEDIM];
#endif
    }

    /**
     * \brief Eval the matrix An
     *
     * \param[in] dom: index of the domain.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[out] An: pointer to memory that will contain the result.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_An(const int dom, const amrex::Real * un, amrex::Real * An) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];

        std::fill(An, An+N_VS*N_VS, 0.0);
#if (AMREX_SPACEDIM == 2)
        An[0+S11*N_VS] = -inv_rho*un[0]; An[0+S12*N_VS] = -inv_rho*un[1];
        An[1+S22*N_VS] = -inv_rho*un[1]; An[1+S12*N_VS] = -inv_rho*un[0];
        
        An[2+V1*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]); An[2+V2*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]);
        An[3+V1*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]); An[3+V2*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]);
        An[4+V1*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]); An[4+V2*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        An[V1+S11*N_VS] = -inv_rho*un[0]; An[V1+S12*N_VS] = -inv_rho*un[1]; An[V1+S13*N_VS] = -inv_rho*un[2];
        An[V2+S12*N_VS] = -inv_rho*un[0]; An[V2+S22*N_VS] = -inv_rho*un[1]; An[V2+S23*N_VS] = -inv_rho*un[2];
        An[V3+S13*N_VS] = -inv_rho*un[0]; An[V3+S23*N_VS] = -inv_rho*un[1]; An[V3+S33*N_VS] = -inv_rho*un[2];

        An[S11+V1*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2]); An[S11+V2*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2]); An[S11+V3*N_VS] = -(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2]);
        An[S22+V1*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2]); An[S22+V2*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2]); An[S22+V3*N_VS] = -(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2]);
        An[S33+V1*N_VS] = -(c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2]); An[S33+V2*N_VS] = -(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2]); An[S33+V3*N_VS] = -(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2]);
        An[S23+V1*N_VS] = -(c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2]); An[S23+V2*N_VS] = -(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2]); An[S23+V3*N_VS] = -(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2]);
        An[S13+V1*N_VS] = -(c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2]); An[S13+V2*N_VS] = -(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2]); An[S13+V3*N_VS] = -(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2]);
        An[S12+V1*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2]); An[S12+V2*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2]); An[S12+V3*N_VS] = -(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2]);
#endif

    }

    /**
     * \brief Eval the eigenvalues and left and right eigenvectors of An.
     *
     * \param[in] dom: index of the domain.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[out] lam: pointer to memory that will contain the eigenvalues.
     * \param[out] r: pointer to memory that will contain the right eigenvectors.
     * \param[out] l: pointer to memory that will contain the left eigenvectors.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eig_An(const int dom, const amrex::Real * un, amrex::Real * lam, amrex::Real * r, amrex::Real * l) const
    {
        // PARAMETERS
        const amrex::Real density = this->rho[dom];
        const amrex::Real * c = &this->cSE[dom][0];

        // VARIABLES
        amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM], zk;
        amrex::Real * vk;
        
        // EIGENVALUES AND EIGENVECTORS OF THE CHRISTOFFEL SYMBOL =====
        this->eig_InTcIn(dom, un, w, v);
        // ============================================================

        // EIGENVALUES AND EIGENVECTORS OF An =========================
        AMREX_D_TERM
        (
            lam[0] = -std::sqrt(w[0]);,
            lam[1] = -std::sqrt(w[1]);,
            lam[2] = -std::sqrt(w[2]);
        )
        AMREX_D_TERM
        (
            lam[0+AMREX_SPACEDIM] = -lam[0];,
            lam[1+AMREX_SPACEDIM] = -lam[1];,
            lam[2+AMREX_SPACEDIM] = -lam[2];
        )

        for (int k = 0; k < AMREX_SPACEDIM; ++k)
        {
            zk = density*lam[k];
            vk = &v[k*AMREX_SPACEDIM];

            // RIGHT EIGENVECTORS
            AMREX_D_TERM
            (
                r[V1+k*N_VS] = 0.5*std::sqrt(2.0)*vk[0];,
                r[V2+k*N_VS] = 0.5*std::sqrt(2.0)*vk[1];,
                r[V3+k*N_VS] = 0.5*std::sqrt(2.0)*vk[2];
            )
            AMREX_D_TERM
            (
                r[V1+(k+AMREX_SPACEDIM)*N_VS] = r[0+k*N_VS];,
                r[V2+(k+AMREX_SPACEDIM)*N_VS] = r[1+k*N_VS];,
                r[V3+(k+AMREX_SPACEDIM)*N_VS] = r[2+k*N_VS];
            )
#if (AMREX_SPACEDIM == 2)
            r[S11+k*N_VS] = (c[C11]*un[0]+c[C61]*un[1])*r[V1+k*N_VS]+(c[C61]*un[0]+c[C21]*un[1])*r[V2+k*N_VS];
            r[S22+k*N_VS] = (c[C21]*un[0]+c[C62]*un[1])*r[V1+k*N_VS]+(c[C62]*un[0]+c[C22]*un[1])*r[V2+k*N_VS];
            r[S12+k*N_VS] = (c[C61]*un[0]+c[C66]*un[1])*r[V1+k*N_VS]+(c[C66]*un[0]+c[C62]*un[1])*r[V2+k*N_VS];

            r[S11+k*N_VS] *= -density/zk;
            r[S22+k*N_VS] *= -density/zk;
            r[S12+k*N_VS] *= -density/zk;

            r[S11+(k+AMREX_SPACEDIM)*N_VS] = -r[S11+k*N_VS];
            r[S22+(k+AMREX_SPACEDIM)*N_VS] = -r[S22+k*N_VS];
            r[S12+(k+AMREX_SPACEDIM)*N_VS] = -r[S12+k*N_VS];

#endif
#if (AMREX_SPACEDIM == 3)
            r[S11+k*N_VS] = (c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2])*r[V1+k*N_VS]+(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2])*r[V2+k*N_VS]+(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2])*r[V3+k*N_VS];
            r[S22+k*N_VS] = (c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2])*r[V1+k*N_VS]+(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2])*r[V2+k*N_VS]+(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2])*r[V3+k*N_VS];
            r[S33+k*N_VS] = (c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2])*r[V1+k*N_VS]+(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2])*r[V2+k*N_VS]+(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2])*r[V3+k*N_VS];
            r[S23+k*N_VS] = (c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2])*r[V1+k*N_VS]+(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2])*r[V2+k*N_VS]+(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2])*r[V3+k*N_VS];
            r[S13+k*N_VS] = (c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2])*r[V1+k*N_VS]+(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2])*r[V2+k*N_VS]+(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2])*r[V3+k*N_VS];
            r[S12+k*N_VS] = (c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2])*r[V1+k*N_VS]+(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2])*r[V2+k*N_VS]+(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2])*r[V3+k*N_VS];
            
            r[S11+k*N_VS] *= -density/zk;
            r[S22+k*N_VS] *= -density/zk;
            r[S33+k*N_VS] *= -density/zk;
            r[S23+k*N_VS] *= -density/zk;
            r[S13+k*N_VS] *= -density/zk;
            r[S12+k*N_VS] *= -density/zk;

            r[S11+(k+AMREX_SPACEDIM)*N_VS] = -r[S11+k*N_VS];
            r[S22+(k+AMREX_SPACEDIM)*N_VS] = -r[S22+k*N_VS];
            r[S33+(k+AMREX_SPACEDIM)*N_VS] = -r[S33+k*N_VS];
            r[S23+(k+AMREX_SPACEDIM)*N_VS] = -r[S23+k*N_VS];
            r[S13+(k+AMREX_SPACEDIM)*N_VS] = -r[S13+k*N_VS];
            r[S12+(k+AMREX_SPACEDIM)*N_VS] = -r[S12+k*N_VS];
#endif
            // LEFT EIGENVECTORS
            AMREX_D_TERM
            (
                l[V1+k*N_VS] = r[V1+k*N_VS];,
                l[V2+k*N_VS] = r[V2+k*N_VS];,
                l[V3+k*N_VS] = r[V3+k*N_VS];
            )
            AMREX_D_TERM
            (
                l[V1+(k+AMREX_SPACEDIM)*N_VS] = r[V1+k*N_VS];,
                l[V2+(k+AMREX_SPACEDIM)*N_VS] = r[V2+k*N_VS];,
                l[V3+(k+AMREX_SPACEDIM)*N_VS] = r[V3+k*N_VS];
            )
#if (AMREX_SPACEDIM == 2)
            l[S11+k*N_VS] = un[0]*r[V1+k*N_VS];
            l[S22+k*N_VS] = un[1]*r[V2+k*N_VS];
            l[S12+k*N_VS] = un[0]*r[V2+k*N_VS]+un[1]*r[V1+k*N_VS];

            l[S11+k*N_VS] *= -1.0/zk;
            l[S22+k*N_VS] *= -1.0/zk;
            l[S12+k*N_VS] *= -1.0/zk;

            l[S11+(k+AMREX_SPACEDIM)*N_VS] = -l[S11+k*N_VS];
            l[S22+(k+AMREX_SPACEDIM)*N_VS] = -l[S22+k*N_VS];
            l[S12+(k+AMREX_SPACEDIM)*N_VS] = -l[S12+k*N_VS];
#endif
#if (AMREX_SPACEDIM == 3)
            l[S11+k*N_VS] = un[0]*r[V1+k*N_VS];
            l[S22+k*N_VS] = un[1]*r[V2+k*N_VS];
            l[S33+k*N_VS] = un[2]*r[V3+k*N_VS];
            l[S23+k*N_VS] = un[2]*r[V2+k*N_VS]+un[1]*r[V3+k*N_VS];
            l[S13+k*N_VS] = un[0]*r[V3+k*N_VS]+un[2]*r[V1+k*N_VS];
            l[S12+k*N_VS] = un[0]*r[V2+k*N_VS]+un[1]*r[V1+k*N_VS];

            l[S11+k*N_VS] *= -1.0/zk;
            l[S22+k*N_VS] *= -1.0/zk;
            l[S33+k*N_VS] *= -1.0/zk;
            l[S23+k*N_VS] *= -1.0/zk;
            l[S13+k*N_VS] *= -1.0/zk;
            l[S12+k*N_VS] *= -1.0/zk;

            l[S11+(k+AMREX_SPACEDIM)*N_VS] = -l[S11+k*N_VS];
            l[S22+(k+AMREX_SPACEDIM)*N_VS] = -l[S22+k*N_VS];
            l[S33+(k+AMREX_SPACEDIM)*N_VS] = -l[S33+k*N_VS];
            l[S23+(k+AMREX_SPACEDIM)*N_VS] = -l[S23+k*N_VS];
            l[S13+(k+AMREX_SPACEDIM)*N_VS] = -l[S13+k*N_VS];
            l[S12+(k+AMREX_SPACEDIM)*N_VS] = -l[S12+k*N_VS];
#endif
        }
        // ============================================================
    }

    /**
     * \brief Eval the product between the matrix An and the conserved variable vector U.
     *
     * \param[in] dom: index of the domain.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] AnU: pointer to memory that will contain the result.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_AnU(const int dom, const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];

#if (AMREX_SPACEDIM == 2)
        AnU[V1] = -inv_rho*(U[S11]*un[0]+U[S12]*un[1]);
        AnU[V2] = -inv_rho*(U[S12]*un[0]+U[S22]*un[1]);

        AnU[S11] = -(c[C11]*U[V1]+c[C61]*U[V2])*un[0]-(c[C61]*U[V1]+c[C21]*U[V2])*un[1];
        AnU[S22] = -(c[C21]*U[V1]+c[C62]*U[V2])*un[0]-(c[C62]*U[V1]+c[C22]*U[V2])*un[1];
        AnU[S12] = -(c[C61]*U[V1]+c[C66]*U[V2])*un[0]-(c[C66]*U[V1]+c[C62]*U[V2])*un[1];
#endif
#if (AMREX_SPACEDIM == 3)
        AnU[V1] = -inv_rho*(U[S11]*un[0]+U[S12]*un[1]+U[S13]*un[2]);
        AnU[V2] = -inv_rho*(U[S12]*un[0]+U[S22]*un[1]+U[S23]*un[2]);
        AnU[V3] = -inv_rho*(U[S13]*un[0]+U[S23]*un[1]+U[S33]*un[2]);

        AnU[S11] = -(c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2])*U[V1]-(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2])*U[V2]-(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2])*U[V3];
        AnU[S22] = -(c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2])*U[V1]-(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2])*U[V2]-(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2])*U[V3];
        AnU[S33] = -(c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2])*U[V1]-(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2])*U[V2]-(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2])*U[V3];
        AnU[S23] = -(c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2])*U[V1]-(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2])*U[V2]-(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2])*U[V3];
        AnU[S13] = -(c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2])*U[V1]-(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2])*U[V2]-(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2])*U[V3];
        AnU[S12] = -(c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2])*U[V1]-(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2])*U[V2]-(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2])*U[V3];
#endif

    }
    // ################################################################

    // EIGENVALUES AND EIGENVECTORS ###################################
    /**
     * \brief Computes the eigenvalues and eigenvectors of the matrix (1/rho)*InT*cSE*In.
     *
     * \param[in] dom: index of the domain.
     * \param[in] un: pointer to memory containing the unit normal vector.
     * \param[out] w: pointer to memory that will be filled with eigenvalues.
     * \param[out] v: pointer to memory that will be filled with eigenvectors.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eig_InTcIn(const int dom, const amrex::Real * un, amrex::Real * w, amrex::Real * v) const
    {
        // VARIABLES
        amrex::Real InTcIn[AMREX_SPACEDIM*AMREX_SPACEDIM];

        // EVAL THE MATRIX AND ITS EIGENVALUES AND EIGENVECTORS
        this->eval_InTcIn(dom, un, InTcIn);

#if (AMREX_SPACEDIM == 2)
        amrex::DG::linalg::dsy2ev(InTcIn, w, v);
#endif
#if (AMREX_SPACEDIM == 3)
        amrex::DG::linalg::dsy3ev(InTcIn, w, v);
#endif

/*
amrex::Print() << "un: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
amrex::Print() << "InTcIn: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, InTcIn);
amrex::Print() << "w: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, w);
amrex::Print() << "v: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, v);

for (int k = 0; k < AMREX_SPACEDIM; ++k)
{
amrex::Real tmp[AMREX_SPACEDIM*AMREX_SPACEDIM], I[AMREX_SPACEDIM*AMREX_SPACEDIM], res[AMREX_SPACEDIM];
amrex::DG::linalg::eye(AMREX_SPACEDIM, I);
for (int d = 0; d < AMREX_SPACEDIM; ++d)
{
    I[d+d*AMREX_SPACEDIM] *= w[k];
}
for (int c = 0; c < AMREX_SPACEDIM; ++c)
for (int r = 0; r < AMREX_SPACEDIM; ++r)
{
    tmp[r+c*AMREX_SPACEDIM] = I[r+c*AMREX_SPACEDIM]-InTcIn[r+c*AMREX_SPACEDIM];
}

amrex::DG::linalg::matmul(AMREX_SPACEDIM, AMREX_SPACEDIM, 1, tmp, &v[k*AMREX_SPACEDIM], res);
amrex::Print() << "I: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, I);
amrex::Print() << "res: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, 1, res);
amrex::Print() << std::endl;
}
exit(-1);
*/
    }
    // ################################################################

    // FLUXES #########################################################
    /**
     * \brief Eval the flux F1 for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F1(const int dom, const amrex::Real * U, amrex::Real * F1) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];

        AMREX_D_TERM
        (
            F1[V1] = -inv_rho*U[S11];,
            F1[V2] = -inv_rho*U[S12];,
            F1[V3] = -inv_rho*U[S13];
        )

#if (AMREX_SPACEDIM == 2)
        F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2];
        F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2];
        F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
        F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2]-c[C51]*U[V3];
        F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2]-c[C52]*U[V3];
        F1[S33] = -c[C31]*U[V1]-c[C63]*U[V2]-c[C53]*U[V3];
        F1[S23] = -c[C41]*U[V1]-c[C64]*U[V2]-c[C54]*U[V3];
        F1[S13] = -c[C51]*U[V1]-c[C65]*U[V2]-c[C55]*U[V3];
        F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2]-c[C65]*U[V3];
#endif
    }

#if (AMREX_SPACEDIM > 1)
    /**
     * \brief Eval the flux F2 for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F2: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F2(const int dom, const amrex::Real * U, amrex::Real * F2) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];

        AMREX_D_TERM
        (
            F2[V1] = -inv_rho*U[S12];,
            F2[V2] = -inv_rho*U[S22];,
            F2[V3] = -inv_rho*U[S23];
        )

#if (AMREX_SPACEDIM == 2)
        F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2];
        F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2];
        F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
        F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2]-c[C42]*U[V3];
        F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2]-c[C42]*U[V3];
        F2[S33] = -c[C63]*U[V1]-c[C32]*U[V2]-c[C43]*U[V3];
        F2[S23] = -c[C64]*U[V1]-c[C42]*U[V2]-c[C44]*U[V3];
        F2[S13] = -c[C65]*U[V1]-c[C52]*U[V2]-c[C54]*U[V3];
        F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2]-c[C64]*U[V3];
#endif
    }
#endif

#if (AMREX_SPACEDIM > 2)
    /**
     * \brief Eval the flux F3 for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F3: pointer to memory that will contain the flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F3(const int dom, const amrex::Real * U, amrex::Real * F3) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/(this->rho[dom]);
        const amrex::Real * c = &this->cSE[dom][0];

        F3[V1] = -inv_rho*U[S13];
        F3[V2] = -inv_rho*U[S23];
        F3[V3] = -inv_rho*U[S33];

        F3[S11] = -c[C51]*U[V1]-c[C42]*U[V2]-c[C31]*U[V3];
        F3[S22] = -c[C52]*U[V1]-c[C42]*U[V2]-c[C32]*U[V3];
        F3[S33] = -c[C53]*U[V1]-c[C43]*U[V2]-c[C33]*U[V3];
        F3[S23] = -c[C54]*U[V1]-c[C44]*U[V2]-c[C43]*U[V3];
        F3[S13] = -c[C55]*U[V1]-c[C54]*U[V2]-c[C53]*U[V3];
        F3[S12] = -c[C65]*U[V1]-c[C64]*U[V2]-c[C63]*U[V3];
    }
#endif

    /**
     * \brief Eval the fluxes for the IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE
    void F_F(const int dom,
             const amrex::Real /*t*/, const amrex::Real * /*x*/,
             const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        AMREX_D_TERM
        (
            this->eval_F1(dom, &U[dom*N_VS], &F1[dom*N_VS]);,
            this->eval_F2(dom, &U[dom*N_VS], &F2[dom*N_VS]);,
            this->eval_F3(dom, &U[dom*N_VS], &F3[dom*N_VS]);
        )
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    /**
     * \brief Eval the numerical flux at the interface between two adjacent states.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the unit normal of the interface.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] nbr_U: pointer to memory containing the neighbor conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real /*t*/, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real density = this->rho[dom];
        const amrex::Real * c = &this->cSE[dom][0];
        const amrex::Real * Ud = &U[dom*N_VS];
        const amrex::Real * nbr_Ud = &nbr_U[dom*N_VS];

        // VARIABLES
        amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM];
        amrex::Real lamk, zk, lk[N_VS], ak[AMREX_SPACEDIM];
        amrex::Real * vk;

        amrex::Real * NFnd = &NFn[dom*N_VS];

        // EIGENVALUES AND EIGENVECTORS OF THE CHRISTOFFEL SYMBOL =====
        this->eig_InTcIn(dom, un, w, v);
        // ============================================================

        // COEFFICIENTS OF THE RIGHT EIGENVECTORS =====================
        // Temporarily use NFn as an auxiliary variable where the jump
        // of the conserved variables is stored.
        AMREX_D_TERM
        (
            NFnd[V1] = Ud[V1]-nbr_Ud[V1];,
            NFnd[V2] = Ud[V2]-nbr_Ud[V2];,
            NFnd[V3] = Ud[V3]-nbr_Ud[V3];
        )
#if (AMREX_SPACEDIM == 2)
        NFnd[S11] = Ud[S11]-nbr_Ud[S11];
        NFnd[S22] = Ud[S22]-nbr_Ud[S22];
        NFnd[S12] = Ud[S12]-nbr_Ud[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        NFnd[S11] = Ud[S11]-nbr_Ud[S11];
        NFnd[S22] = Ud[S22]-nbr_Ud[S22];
        NFnd[S33] = Ud[S33]-nbr_Ud[S33];
        NFnd[S23] = Ud[S23]-nbr_Ud[S23];
        NFnd[S13] = Ud[S13]-nbr_Ud[S13];
        NFnd[S12] = Ud[S12]-nbr_Ud[S12];
#endif

        for (int k = 0; k < AMREX_SPACEDIM; ++k)
        {
            lamk = -std::sqrt(w[k]);
            zk = density*lamk;
            vk = &v[k*AMREX_SPACEDIM];

            AMREX_D_TERM
            (
                lk[V1] = 0.5*std::sqrt(2.0)*vk[0];,
                lk[V2] = 0.5*std::sqrt(2.0)*vk[1];,
                lk[V3] = 0.5*std::sqrt(2.0)*vk[2];
            )
#if (AMREX_SPACEDIM == 2)
            lk[S11] = un[0]*lk[V1];
            lk[S22] = un[1]*lk[V2];
            lk[S12] = un[0]*lk[V2]+un[1]*lk[V1];

            lk[S11] *= -1.0/zk;
            lk[S22] *= -1.0/zk;
            lk[S12] *= -1.0/zk;
#endif
#if (AMREX_SPACEDIM == 3)
            lk[S11] = un[0]*lk[V1];
            lk[S22] = un[1]*lk[V2];
            lk[S33] = un[2]*lk[V3];
            lk[S23] = un[2]*lk[V2]+un[1]*lk[V3];
            lk[S13] = un[0]*lk[V3]+un[2]*lk[V1];
            lk[S12] = un[0]*lk[V2]+un[1]*lk[V1];

            lk[S11] *= -1.0/zk;
            lk[S22] *= -1.0/zk;
            lk[S33] *= -1.0/zk;
            lk[S23] *= -1.0/zk;
            lk[S13] *= -1.0/zk;
            lk[S12] *= -1.0/zk;
#endif

//amrex::Print() << "lk: "; amrex::DG::IO::PrintRealArray2D(1, N_VS, lk);

            AMREX_D_TERM
            (
                ak[k]  = lk[V1]*NFnd[V1];,
                ak[k] += lk[V2]*NFnd[V2];,
                ak[k] += lk[V3]*NFnd[V3];
            )
#if (AMREX_SPACEDIM == 2)
            ak[k] += lk[S11]*NFnd[S11];
            ak[k] += lk[S22]*NFnd[S22];
            ak[k] += lk[S12]*NFnd[S12];
#endif
#if (AMREX_SPACEDIM == 3)
            ak[k] += lk[S11]*NFnd[S11];
            ak[k] += lk[S22]*NFnd[S22];
            ak[k] += lk[S33]*NFnd[S33];
            ak[k] += lk[S23]*NFnd[S23];
            ak[k] += lk[S13]*NFnd[S13];
            ak[k] += lk[S12]*NFnd[S12];
#endif
        }
        // ============================================================

        // NUMERICAL FLUX =============================================
        this->eval_AnU(dom, un, Ud, NFnd);

        // Use lk to store the right eigenvectors
        for (int k = 0; k < AMREX_SPACEDIM; ++k)
        {
            lamk = -std::sqrt(w[k]);
            zk = density*lamk;
            vk = &v[k*AMREX_SPACEDIM];

            AMREX_D_TERM
            (
                lk[V1] = 0.5*std::sqrt(2.0)*vk[0];,
                lk[V2] = 0.5*std::sqrt(2.0)*vk[1];,
                lk[V3] = 0.5*std::sqrt(2.0)*vk[2];
            )
#if (AMREX_SPACEDIM == 2)
            lk[S11] = (c[C11]*un[0]+c[C61]*un[1])*lk[V1]+(c[C61]*un[0]+c[C21]*un[1])*lk[V2];
            lk[S22] = (c[C21]*un[0]+c[C62]*un[1])*lk[V1]+(c[C62]*un[0]+c[C22]*un[1])*lk[V2];
            lk[S12] = (c[C61]*un[0]+c[C66]*un[1])*lk[V1]+(c[C66]*un[0]+c[C62]*un[1])*lk[V2];

            lk[S11] *= -density/zk;
            lk[S22] *= -density/zk;
            lk[S12] *= -density/zk;
#endif
#if (AMREX_SPACEDIM == 3)
            lk[S11] = (c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2])*lk[V1]+(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2])*lk[V2]+(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2])*lk[V3];
            lk[S22] = (c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2])*lk[V1]+(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2])*lk[V2]+(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2])*lk[V3];
            lk[S33] = (c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2])*lk[V1]+(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2])*lk[V2]+(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2])*lk[V3];
            lk[S23] = (c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2])*lk[V1]+(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2])*lk[V2]+(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2])*lk[V3];
            lk[S13] = (c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2])*lk[V1]+(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2])*lk[V2]+(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2])*lk[V3];
            lk[S12] = (c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2])*lk[V1]+(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2])*lk[V2]+(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2])*lk[V3];
            
            lk[S11] *= -density/zk;
            lk[S22] *= -density/zk;
            lk[S33] *= -density/zk;
            lk[S23] *= -density/zk;
            lk[S13] *= -density/zk;
            lk[S12] *= -density/zk;
#endif

//amrex::Print() << "rk: "; amrex::DG::IO::PrintRealArray2D(1, N_VS, lk);

            AMREX_D_TERM
            (
                NFnd[V1] -= ak[k]*lamk*lk[V1];,
                NFnd[V2] -= ak[k]*lamk*lk[V2];,
                NFnd[V3] -= ak[k]*lamk*lk[V3];
            )
#if (AMREX_SPACEDIM == 2)
            NFnd[S11] -= ak[k]*lamk*lk[S11];
            NFnd[S22] -= ak[k]*lamk*lk[S22];
            NFnd[S12] -= ak[k]*lamk*lk[S12];
#endif
#if (AMREX_SPACEDIM == 3)
            NFnd[S11] -= ak[k]*lamk*lk[S11];
            NFnd[S22] -= ak[k]*lamk*lk[S22];
            NFnd[S33] -= ak[k]*lamk*lk[S33];
            NFnd[S23] -= ak[k]*lamk*lk[S23];
            NFnd[S13] -= ak[k]*lamk*lk[S13];
            NFnd[S12] -= ak[k]*lamk*lk[S12];
#endif
        }
        // ============================================================

        // NUMERICAL FLUX (LAX) =======================================
        /*
        {
            amrex::Real avg_U[N_VS], delta_U[N_VS];

            AMREX_D_TERM
            (
                avg_U[V1] = 0.5*(Ud[V1]+nbr_Ud[V1]);,
                avg_U[V2] = 0.5*(Ud[V2]+nbr_Ud[V2]);,
                avg_U[V3] = 0.5*(Ud[V3]+nbr_Ud[V3]);
            )
#if (AMREX_SPACEDIM == 2)
            avg_U[S11] = 0.5*(Ud[S11]+nbr_Ud[S11]);
            avg_U[S22] = 0.5*(Ud[S22]+nbr_Ud[S22]);
            avg_U[S12] = 0.5*(Ud[S12]+nbr_Ud[S12]);
#endif
            AMREX_D_TERM
            (
                delta_U[V1] = Ud[V1]-nbr_Ud[V1];,
                delta_U[V2] = Ud[V2]-nbr_Ud[V2];,
                delta_U[V3] = Ud[V3]-nbr_Ud[V3];
            )
#if (AMREX_SPACEDIM == 2)
            delta_U[S11] = Ud[S11]-nbr_Ud[S11];
            delta_U[S22] = Ud[S22]-nbr_Ud[S22];
            delta_U[S12] = Ud[S12]-nbr_Ud[S12];
#endif

            this->eval_AnU(dom, un, avg_U, NFnd);

            AMREX_D_TERM
            (
                NFnd[V1] += 0.5*this->max_wave_speed[dom]*delta_U[V1];,
                NFnd[V2] += 0.5*this->max_wave_speed[dom]*delta_U[V2];,
                NFnd[V3] += 0.5*this->max_wave_speed[dom]*delta_U[V3];
            )
#if (AMREX_SPACEDIM == 2)
            NFnd[S11] += 0.5*this->max_wave_speed[dom]*delta_U[S11];
            NFnd[S22] += 0.5*this->max_wave_speed[dom]*delta_U[S22];
            NFnd[S12] += 0.5*this->max_wave_speed[dom]*delta_U[S12];
#endif
        }
        */
        // ============================================================

/*
//if ((un[0] < -0.5) && (un[1] > 0.05))
{
amrex::Real An[N_VS*N_VS], lam[2*AMREX_SPACEDIM], r[N_VS*2*AMREX_SPACEDIM], l[N_VS*2*AMREX_SPACEDIM];
amrex::Real rAn[N_VS*N_VS];

this->eval_An(dom, un, An);
this->eig_An(dom, un, lam, r, l);
amrex::Print() << "An: " << std::endl; amrex::DG::IO::PrintRealArray2D(N_VS, N_VS, An);
amrex::Print() << "lam: "; amrex::DG::IO::PrintRealArray2D(1, 2*AMREX_SPACEDIM, lam);
amrex::Print() << "r: " << std::endl; amrex::DG::IO::PrintRealArray2D(N_VS, 2*AMREX_SPACEDIM, r);
amrex::Print() << "l: " << std::endl; amrex::DG::IO::PrintRealArray2D(N_VS, 2*AMREX_SPACEDIM, l);

for (int k = 0; k < 2.0*AMREX_SPACEDIM; ++k)
{
    amrex::Real tmp[N_VS];
    amrex::DG::linalg::matmul(N_VS, N_VS, 1, An, &r[k*N_VS], tmp);
    for (int u = 0; u < N_VS; ++u)
    {
        tmp[u] -= lam[k]*r[u+k*N_VS];
    }
    amrex::Print() << " - tmp: "; amrex::DG::IO::PrintRealArray2D(1, N_VS, tmp);
}

for (int k = 0; k < 2.0*AMREX_SPACEDIM; ++k)
{
    amrex::Real tmp;
    amrex::DG::linalg::matmul(1, N_VS, 1, &r[k*N_VS], &l[k*N_VS], &tmp);
amrex::Print() << " - tmp: " << tmp << std::endl;
}

std::fill(rAn, rAn+N_VS*N_VS, 0.0);
for (int k = 0; k < 2.0*AMREX_SPACEDIM; ++k)
{
    amrex::DG::linalg::dgemm('N', 'N', N_VS, N_VS, 1, lam[k], &r[k*N_VS], N_VS, &l[k*N_VS], 1, 1.0, rAn, N_VS);
}
amrex::Print() << "rAn: " << std::endl; amrex::DG::IO::PrintRealArray2D(N_VS, N_VS, rAn);

for (int k = 0; k < N_VS*N_VS; ++k)
{
    rAn[k] -= An[k];
}
amrex::Print() << "rAn-An: " << std::endl; amrex::DG::IO::PrintRealArray2D(N_VS, N_VS, rAn);

amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
amrex::Print() << "un: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
amrex::Print() << "U    : "; amrex::DG::IO::PrintRealArray2D(1, N_DOM*N_VS, U);
amrex::Print() << "nbr_U: "; amrex::DG::IO::PrintRealArray2D(1, N_DOM*N_VS, nbr_U);
amrex::Print() << "w: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, w);
amrex::Print() << "v: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
amrex::Print() << "ak: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, ak);
amrex::Print() << "NFn: "; amrex::DG::IO::PrintRealArray2D(1, N_DOM*N_VS, NFn);
amrex::Print() << "HELLO - F_NF_ICS" << std::endl;
exit(-1);
}
*/
    }
    // ################################################################
};