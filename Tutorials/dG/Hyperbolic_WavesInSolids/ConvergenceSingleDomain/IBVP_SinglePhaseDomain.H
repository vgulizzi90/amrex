//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SinglePhaseDomain.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_ElasticSolid.H"


namespace single_phase_domain
{
// SINGLE PHASE DOMAIN: LEVEL SET #####################################
template <int N>
struct LevelSet
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    amrex::Real xc, yc, zc, r;
    // ================================================================


    // CONSTRUCTOR ====================================================
    LevelSet()
    :
    t{0.0},
    dom{0}
    {
        amrex::ParmParse pp;

        AMREX_D_TERM
        (
            pp.get("x_coord", this->xc);,
            pp.get("y_coord", this->yc);,
            pp.get("z_coord", this->zc);
        )
        pp.get("radius", this->r);
    }
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_time(const amrex::Real t_)
    {
        this->t = t_;
    }
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // EVALUATION =====================================================
    amrex::Real eval(const amrex::Real * x) const
    {
        const amrex::Real r2 = this->r*this->r;
        const amrex::Real dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->xc, x[1]-this->yc, x[2]-this->zc)};
        const amrex::Real res = r2-(AMREX_D_TERM(dx[0]*dx[0],+dx[1]*dx[1],+dx[2]*dx[2]));
        const amrex::Real sgn = AMREX_D_PICK(1.0, 1.0, -1.0);
        
        return sgn*res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real r2 = this->r*this->r;
        const T dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->xc, x[1]-this->yc, x[2]-this->zc)};
        const T res = r2-(AMREX_D_TERM(dx[0]*dx[0],+dx[1]*dx[1],+dx[2]*dx[2]));
        const amrex::Real sgn = AMREX_D_PICK(1.0, 1.0, -1.0);
        
        return sgn*res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
        T dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->xc, x[1]-this->yc, x[2]-this->zc)};
        const amrex::Real sgn = AMREX_D_PICK(1.0, 1.0, -1.0);
        const amrex::GpuArray<T, N> res = {AMREX_D_DECL(-2.0*dx[0]*sgn, -2.0*dx[1]*sgn, -2.0*dx[2]*sgn)};

        return res;
    }
    // ================================================================
};
// ####################################################################



// SINGLE PHASE DOMAIN: IBVP ##########################################
struct IBVP
{
    // DATA MEMBERS ===================================================
    amrex::Real density, c[NS*(NS+1)/2], max_wave_speed;
    
    // INITIAL CONDITION / EXACT SOLUTION
    amrex::Real wn[AMREX_SPACEDIM];
    amrex::Real w[N_VS], R[N_VS*N_VS], cR[N_VS];
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        amrex::ParmParse pp;
        std::string material_type;
        amrex::Vector<amrex::Real> material_constants;
        amrex::Real cSE[NS*NS];

        // READ MATERIAL PROPERTIES -----------------------------------
        pp.get("material_type", material_type);
        if (material_type.compare("isotropic") == 0)
        {
            amrex::Real E, nu;

            pp.getarr("material_constants", material_constants);

            this->density = material_constants[0];
            E = material_constants[1];
            nu = material_constants[2];
            
#if (AMREX_SPACEDIM == 2)
            linear_elasticity::get_stiffness_matrix_2d_isotropic(E, nu, cSE);
            //linear_elasticity::get_stiffness_matrix_2d_isotropic_plane_stress(E, nu, cSE);
#endif
#if (AMREX_SPACEDIM == 3)
            linear_elasticity::get_stiffness_matrix_3d_isotropic(E, nu, cSE);
#endif
        }
        else if (material_type.compare("orthotropic") == 0)
        {
#if (AMREX_SPACEDIM == 2)
            amrex::Real c11, c22, c12, c66;
#endif
#if (AMREX_SPACEDIM == 3)
            amrex::Real c11, c22, c33, c23, c13, c12, c44, c55, c66;
#endif
            pp.getarr("material_constants", material_constants);

            this->density = material_constants[0];
#if (AMREX_SPACEDIM == 2)
            c11 = material_constants[1];
            c22 = material_constants[2];
            c12 = material_constants[3];
            c66 = material_constants[4];
#endif
#if (AMREX_SPACEDIM == 3)
            c11 = material_constants[1];
            c22 = material_constants[2];
            c33 = material_constants[3];
            c23 = material_constants[4];
            c13 = material_constants[5];
            c12 = material_constants[6];
            c44 = material_constants[7];
            c55 = material_constants[8];
            c66 = material_constants[9];
#endif

#if (AMREX_SPACEDIM == 2)
            linear_elasticity::get_stiffness_matrix_2d_orthotropic(c11, c22, c12, c66, cSE);
#endif
#if (AMREX_SPACEDIM == 3)
            linear_elasticity::get_stiffness_matrix_3d_orthotropic(c11, c22, c33, c23, c13, c12, c44, c55, c66, cSE);
#endif
        }
        else if (material_type.compare("anisotropic") == 0)
        {
            const amrex::Real * cij;
            pp.getarr("material_constants", material_constants);

            this->density = material_constants[0];
            cij = &material_constants[1];

            amrex::Print() << "cij[3]: " << cij[3] << std::endl;

#if (AMREX_SPACEDIM == 2)
            cSE[0+0*NS] = cij[0]; cSE[0+1*NS] = cij[1]; cSE[0+2*NS] = cij[2];
            cSE[1+0*NS] = cij[1]; cSE[1+1*NS] = cij[3]; cSE[1+2*NS] = cij[4];
            cSE[2+0*NS] = cij[2]; cSE[2+1*NS] = cij[4]; cSE[2+2*NS] = cij[5];
#endif
#if (AMREX_SPACEDIM == 3)
            cSE[0+0*NS] = cij[0]; cSE[0+1*NS] = cij[1];  cSE[0+2*NS] = cij[2];  cSE[0+3*NS] = cij[3];  cSE[0+4*NS] = cij[4];  cSE[0+5*NS] = cij[5];
            cSE[1+0*NS] = cij[1]; cSE[1+1*NS] = cij[6];  cSE[1+2*NS] = cij[7];  cSE[1+3*NS] = cij[8];  cSE[1+4*NS] = cij[9];  cSE[1+5*NS] = cij[10];
            cSE[2+0*NS] = cij[2]; cSE[2+1*NS] = cij[7];  cSE[2+2*NS] = cij[11]; cSE[2+3*NS] = cij[12]; cSE[2+4*NS] = cij[13]; cSE[2+5*NS] = cij[14];
            cSE[3+0*NS] = cij[3]; cSE[3+1*NS] = cij[8];  cSE[3+2*NS] = cij[12]; cSE[3+3*NS] = cij[15]; cSE[3+4*NS] = cij[16]; cSE[3+5*NS] = cij[17];
            cSE[4+0*NS] = cij[4]; cSE[4+1*NS] = cij[9];  cSE[4+2*NS] = cij[13]; cSE[4+3*NS] = cij[16]; cSE[4+4*NS] = cij[18]; cSE[4+5*NS] = cij[19];
            cSE[5+0*NS] = cij[5]; cSE[5+1*NS] = cij[10]; cSE[5+2*NS] = cij[14]; cSE[5+3*NS] = cij[17]; cSE[5+4*NS] = cij[19]; cSE[5+5*NS] = cij[20];
#endif
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SinglePhaseDomain.H - IBVP::IBVP\n";
            msg += "| Unexpected input material type: "+material_type+".\n";
            amrex::Abort(msg);
        }

        /* DEBUG */
        amrex::Print() << "cSE: " << std::endl;
        amrex::dG::io::print_real_array_2d(NS, NS, cSE);
        /**/
        // ------------------------------------------------------------

        // STORE STIFFNESS MATRIX COEFFICIENTS ------------------------
        {
#if (AMREX_SPACEDIM == 2)
            this->c[C11] = cSE[0+0*NS];
            this->c[C21] = cSE[1+0*NS]; this->c[C22] = cSE[1+1*NS];
            this->c[C61] = cSE[2+0*NS]; this->c[C62] = cSE[2+1*NS]; this->c[C66] = cSE[2+2*NS];
#endif
#if (AMREX_SPACEDIM == 3)
            this->c[C11] = cSE[0+0*NS];
            this->c[C21] = cSE[1+0*NS]; this->c[C22] = cSE[1+1*NS];
            this->c[C31] = cSE[2+0*NS]; this->c[C32] = cSE[2+1*NS]; this->c[C33] = cSE[2+2*NS];
            this->c[C41] = cSE[3+0*NS]; this->c[C42] = cSE[3+1*NS]; this->c[C43] = cSE[3+2*NS]; this->c[C44] = cSE[3+3*NS];
            this->c[C51] = cSE[4+0*NS]; this->c[C52] = cSE[4+1*NS]; this->c[C53] = cSE[4+2*NS]; this->c[C54] = cSE[4+3*NS]; this->c[C55] = cSE[4+4*NS];
            this->c[C61] = cSE[5+0*NS]; this->c[C62] = cSE[5+1*NS]; this->c[C63] = cSE[5+2*NS]; this->c[C64] = cSE[5+3*NS]; this->c[C65] = cSE[5+4*NS]; this->c[C66] = cSE[5+5*NS];
#endif
        }
        // ------------------------------------------------------------

        // MAX WAVE SPEED ---------------------------------------------
        if (material_type.compare("isotropic") == 0)
        {
            const amrex::Real G = this->c[C66];
            const amrex::Real lam = this->c[C21];

            this->max_wave_speed = std::sqrt((2.0*G+lam)/this->density);
        }
        else
        {
            const int n_th = 360;
            const amrex::Real dth = 2.0*M_PI/n_th;
#if (AMREX_SPACEDIM == 3)
            const int n_ph = n_th/2;
            const amrex::Real dph = M_PI/n_ph;
#endif

            amrex::Real th;
#if (AMREX_SPACEDIM == 3)
            amrex::Real ph;
#endif
            amrex::Real un[AMREX_SPACEDIM];
            amrex::Real w2[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM];

            this->max_wave_speed = 0.0;
#if (AMREX_SPACEDIM == 2)
            for (int i = 0; i < n_th; ++i)
            {
                th = i*dth;
                un[0] = std::cos(th);
                un[1] = std::sin(th);
                elastic_solid::eval_eig_InTcIn_compact_c(this->density, this->c, un, w2, v);
                this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt(*std::max_element(w2, w2+AMREX_SPACEDIM)));
            }
#endif
#if (AMREX_SPACEDIM == 3)
            for (int j = 0; j < n_ph; ++j)
            for (int i = 0; i < n_th; ++i)
            {
                th = i*dth;
                ph = j*dph;
                un[0] = std::cos(th)*std::sin(ph);
                un[1] = std::sin(th)*std::sin(ph);
                un[2] = std::cos(ph);
                elastic_solid::eval_eig_InTcIn_compact_c(this->density, this->c, un, w2, v);
                this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt(*std::max_element(w2, w2+AMREX_SPACEDIM)));
            }
#endif
        }

        /* DEBUG */
        amrex::Print() << "Max wave speed: " << this->max_wave_speed << std::endl;
        /**/
        // ------------------------------------------------------------

        // INITIAL CONDITIONS / EXACT SOLUTION ------------------------
        // We consider the hyperbolic formulation of linear elasticity:
        //
        // U_t + (A_k U)_k = 0
        //
        // where
        //
        // U = {v_x, v_y, s_xx, s_yy, s_xy} (2D)
        // U = {v_x, v_y, v_z, s_xx, s_yy, s_zz, s_yz, s_xz, s_xy} (3D)
        // 
        // ------------------------------------------------------------
        {
            amrex::Real An[N_VS*N_VS], wAn[N_VS], vAn[N_VS*N_VS];
            
            // Wave number
#if (AMREX_SPACEDIM == 2)
            const amrex::Real th = M_PI/6.0;
            const amrex::Real cth = std::cos(th);
            const amrex::Real sth = std::sin(th);
            this->wn[0] = 2.0*M_PI*cth;
            this->wn[1] = 2.0*M_PI*sth;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real th = M_PI/3.0;
            const amrex::Real ph = M_PI/6.0;
            const amrex::Real cth = std::cos(th);
            const amrex::Real sth = std::sin(th);
            const amrex::Real cph = std::cos(ph);
            const amrex::Real sph = std::sin(ph);
            this->wn[0] = 2.0*M_PI*cth*sph;
            this->wn[1] = 2.0*M_PI*sth*sph;
            this->wn[2] = 2.0*M_PI*cph;
#endif

            // Evaluate An = A_k*wk[k]
            elastic_solid::eval_An_compact_c(this->density, this->c, this->wn, An);

            // Evaluate eigenvalues of An
            {
                char jobvl = 'N';
                char jobvr = 'V';
                int n = N_VS;
                amrex::Real An_copy[N_VS*N_VS];
                amrex::Real wAn_im[N_VS];
                amrex::Real work[N_VS*N_VS];
                int lwork = N_VS*N_VS;
                int info;

                std::copy(An, An+N_VS*N_VS, An_copy);
                
                dgeev_(&jobvl, &jobvr, &n, An_copy, &n, wAn, wAn_im, nullptr, &n, vAn, &n, work, &lwork, &info);
                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_SinglePhaseDomain.H - IBVP::IBVP\n";
                    msg += "| Something went wrong in the computation of the eigenvalues of An.\n";
                    amrex::Abort(msg);
                }

                // Store eigenvalues and eigenvectors
                std::copy(wAn, wAn+N_VS, this->w);
                std::copy(vAn, vAn+N_VS*N_VS, this->R);
            }

#if (AMREX_SPACEDIM == 2)
            this->cR[V1] = 1.0;
            this->cR[V2] = 1.0;

            this->cR[S11] = 1.0;
            this->cR[S22] = 1.0;
            this->cR[S12] = 1.0;
#endif
#if (AMREX_SPACEDIM == 3)
            this->cR[V1] = 1.0;
            this->cR[V2] = 1.0;
            this->cR[V3] = 1.0;

            this->cR[S11] = 1.0;
            this->cR[S22] = 1.0;
            this->cR[S33] = 1.0;
            this->cR[S23] = 1.0;
            this->cR[S13] = 1.0;
            this->cR[S12] = 1.0;
#endif

            /* DEBUG */
            amrex::Print() << "wn: " << std::endl;
            amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, this->wn);
            amrex::Print() << "An: " << std::endl;
            amrex::dG::io::print_real_array_2d(N_VS, N_VS, An);
            amrex::Print() << "Eigenvalues of An: " << std::endl;
            amrex::dG::io::print_real_array_2d(1, N_VS, this->w);
            amrex::Print() << "Right eigenvectors of An: " << std::endl;
            amrex::dG::io::print_real_array_2d(N_VS, N_VS, this->R);
            amrex::Print() << "Right eigenvectors coefficients: " << std::endl;
            amrex::dG::io::print_real_array_2d(1, N_VS, this->cR);
            {
                amrex::Real max_wn = *std::max_element(this->w, this->w+N_VS);
                amrex::Print() << "max wn: " << max_wn << " -> T: " << 2.0*M_PI/max_wn << std::endl;
            }
            /**/
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // SOLUTION =======================================================
    /**
     * \brief Return the number of considered domains.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_domains() const
    {
        return 1;
    }
    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        return N_VS;
    }
    /**
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int unknown_field_parent_domain(const int /*u*/) const
    {
        return 0;
    }
    /**
     * \brief Return the domain that is neighbor with domain dom.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_nbr_dom(const int /*dom*/) const
    {
        return -1;
    }

    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_exact_solution(const amrex::Real t, const amrex::Real * x, amrex::Real * U) const
    {
        const amrex::Real x_dot_wn = (AMREX_D_TERM(this->wn[0]*x[0],+this->wn[1]*x[1],+this->wn[2]*x[2]));
        
        for (int r = 0; r < N_VS; ++r)
        {
            U[r] = 0.0;
            for (int k = 0; k < N_VS; ++k)
            {
                U[r] += this->R[r+k*N_VS]*this->cR[k]*std::sin(this->w[k]*t-x_dot_wn);
            }
        }
    }

    /**
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs(const amrex::Real * x, amrex::Real * U) const
    {
        this->eval_exact_solution(0.0, x, U);
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error at (t, x) to be used for the L_inf norm error.
     *
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the value of the conserved variables at (t, x).
     * \param[out] err: error at (t, x).
     * \param[out] err_norm: error normalization at (t, x).
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_error_L_inf(const int /*dom*/,
                          const amrex::Real t, const amrex::Real * x,
                          const amrex::Real * U,
                          amrex::Real & err, amrex::Real & err_norm) const
    {
        amrex::Real U_exact[N_VS];

        this->eval_exact_solution(t, x, U_exact);

#if (AMREX_SPACEDIM == 2)
        err =                 std::abs(U[V1]-U_exact[V1]);
        err = amrex::max(err, std::abs(U[V2]-U_exact[V2]));
        err = amrex::max(err, std::abs(U[S11]-U_exact[S11]));
        err = amrex::max(err, std::abs(U[S22]-U_exact[S22]));
        err = amrex::max(err, std::abs(U[S12]-U_exact[S12]));

        err_norm =                      std::abs(U_exact[V1]);
        err_norm = amrex::max(err_norm, std::abs(U_exact[V2]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S11]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S22]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S12]));
#endif
#if (AMREX_SPACEDIM == 3)
        err =                 std::abs(U[V1]-U_exact[V1]);
        err = amrex::max(err, std::abs(U[V2]-U_exact[V2]));
        err = amrex::max(err, std::abs(U[V3]-U_exact[V3]));
        err = amrex::max(err, std::abs(U[S11]-U_exact[S11]));
        err = amrex::max(err, std::abs(U[S22]-U_exact[S22]));
        err = amrex::max(err, std::abs(U[S33]-U_exact[S33]));
        err = amrex::max(err, std::abs(U[S23]-U_exact[S23]));
        err = amrex::max(err, std::abs(U[S13]-U_exact[S13]));
        err = amrex::max(err, std::abs(U[S12]-U_exact[S12]));

        err_norm =                      std::abs(U_exact[V1]);
        err_norm = amrex::max(err_norm, std::abs(U_exact[V2]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[V3]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S11]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S22]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S33]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S23]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S13]));
        err_norm = amrex::max(err_norm, std::abs(U_exact[S12]));
#endif
    }
    // ================================================================


    // TIME INCREMENT =================================================
    /**
     * \brief Eval the time increment based on solution at (t, x).
     *
     * \param[in] dom: domain index.
     * \param[in] dx: pointer to memory containing the grid sizes.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the value of the conserved variables at (t, x).
     *
     * \return: time increment based on solution at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval_dt(const int /*dom*/,
                        const amrex::Real * dx,
                        const amrex::Real /*t*/, const amrex::Real * /*x*/,
                        const amrex::Real * U) const
    {
        const amrex::Real ws = this->max_wave_speed;
#if (AMREX_SPACEDIM == 1)
        const amrex::Real h = dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = amrex::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
#endif
        return (h/ws);
    }
    // ================================================================


    // GOVERNING EQUATIONS ============================================
    /**
     * \brief Eval the fluxes for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F(const int /*dom*/,
                const amrex::Real /*t*/, const amrex::Real * /*x*/,
                const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        elastic_solid::eval_F_compact_c(this->density, this->c, U, AMREX_D_DECL(F1, F2, F3));
    }
    // ================================================================


    // DG =============================================================
    /**
     * \brief Eval the numerical flux at the intraphase boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
     *                 unit normal.
     * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
     *                 unit normal.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_intraphase(const int /*dom*/,
                             const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn) const
    {
        elastic_solid::eval_NFn_Riemann_solver(this->density, this->c, un, m_U, p_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_BCs(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        amrex::Real U_exact[N_VS], U_ghost[N_VS];

        this->eval_exact_solution(t, x, U_exact);

        // Prescribed velocity
#if (AMREX_SPACEDIM == 2)
        U_ghost[V1] = 2.0*U_exact[V1]-U[V1];
        U_ghost[V2] = 2.0*U_exact[V2]-U[V2];

        U_ghost[S11] = U[S11];
        U_ghost[S22] = U[S22];
        U_ghost[S12] = U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        U_ghost[V1] = 2.0*U_exact[V1]-U[V1];
        U_ghost[V2] = 2.0*U_exact[V2]-U[V2];
        U_ghost[V3] = 2.0*U_exact[V3]-U[V3];

        U_ghost[S11] = U[S11];
        U_ghost[S22] = U[S22];
        U_ghost[S33] = U[S33];
        U_ghost[S23] = U[S23];
        U_ghost[S13] = U[S13];
        U_ghost[S12] = U[S12];
#endif

        this->eval_NFn_intraphase(dom, t, x, un, U, U_ghost, NFn);
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_EB(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        amrex::Real U_exact[N_VS], U_ghost[N_VS], Sn[NS], Sn_exact[NS];

        this->eval_exact_solution(t, x, U_exact);

        // Unit normal reference system
        linear_elasticity::S_to_Sn(un, &U[NV], Sn);
        linear_elasticity::S_to_Sn(un, &U_exact[NV], Sn_exact);

        // Prescribed tractions
        // We use NFn as a temporary variable
#if (AMREX_SPACEDIM == 2)
        U_ghost[V1] = U[V1];
        U_ghost[V2] = U[V2];

        NFn[S11] = 2.0*Sn_exact[0]-Sn[0];
        NFn[S22] = Sn[1];
        NFn[S12] = 2.0*Sn_exact[2]-Sn[2];
#endif
#if (AMREX_SPACEDIM == 3)
        U_ghost[V1] = U[V1];
        U_ghost[V2] = U[V2];
        U_ghost[V3] = U[V3];

        NFn[S11] = 2.0*Sn_exact[0]-Sn[0];
        NFn[S22] = Sn[1];
        NFn[S33] = Sn[2];
        NFn[S23] = Sn[3];
        NFn[S13] = 2.0*Sn_exact[4]-Sn[4];
        NFn[S12] = 2.0*Sn_exact[5]-Sn[5];
#endif

        // Global reference system
        linear_elasticity::Sn_to_S(un, &NFn[NV], &U_ghost[NV]);

        this->eval_NFn_intraphase(dom, t, x, un, U, U_ghost, NFn);
    }
    // ================================================================


    // FV =============================================================
    /**
     * \brief Reconstruct and limit the slopes for FV schemes.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] solution: Solution object.
     * \param[in] mesh: Mesh object.
     * \param[in] X: a MultiFabs containing the solution states.
     * \param[out] dX: a MultiFabs that will contain the solution slopes.
     *
    */
    void eval_FV_slopes(const amrex::Geometry & geom,
                        const amrex::dG::Mesh & mesh,
                        const amrex::dG::Solution & solution,
                        const amrex::MultiFab & X,
                        amrex::MultiFab & dX) const
    {
amrex::Print() << "eval_FV_slopes" << std::endl;
exit(-1);
    }
    // ================================================================


    // OUTPUT FIELDS ==================================================
    /**
     * \brief Return the number of output fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_output_fields() const
    {
        return N_VS+2;
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int /*u*/) const
    {
        return 0;
    }
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
#if (AMREX_SPACEDIM == 2)
        output_fields_names = {"v_x", "v_y",
                               "s_xx", "s_yy", "s_xy",
                               "v_x_error", "s_xx_error"};
#endif
#if (AMREX_SPACEDIM == 3)
        output_fields_names = {"v_x", "v_y", "v_z",
                               "s_xx", "s_yy", "s_zz", "s_yz", "s_xz", "s_xy",
                               "v_x_error", "s_xx_error"};
#endif
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const amrex::Real t, const amrex::Real * x,
                            const amrex::Real * U,
                            amrex::Real * F) const
    {
        amrex::Real U_exact[N_VS];

        this->eval_exact_solution(t, x, U_exact);

#if (AMREX_SPACEDIM == 2)
            F[V1] = U[V1];
            F[V2] = U[V2];

            F[S11] = U[S11];
            F[S22] = U[S22];
            F[S12] = U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
            F[V1] = U[V1];
            F[V2] = U[V2];
            F[V3] = U[V3];

            F[S11] = U[S11];
            F[S22] = U[S22];
            F[S33] = U[S33];
            F[S23] = U[S23];
            F[S13] = U[S13];
            F[S12] = U[S12];
#endif
            F[S12+1] = U[V1]-U_exact[V1];
            F[S12+2] = U[S11]-U_exact[S11];
    }
    // ================================================================
};
// ####################################################################



// SINGLE PHASE DOMAIN: AMR ###########################################
struct AMR
:
public amrex::dG::amr::SinglePatch
{
    // DATA MEMBERS ===================================================
    // LEVEL SET
    LevelSet<AMREX_SPACEDIM> level_set;
    
    // IBVP
    IBVP ibvp;

    // SOLUTION MULTIFABS
    amrex::Vector<amrex::MultiFab> X;

    // NORM
    std::string norm;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        // SOLUTION MULTIFABS
        this->X.resize(n_levels);

        // NORM
        {
            amrex::ParmParse pp;
            pp.get("norm", this->norm);
        }
    }
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize AMR data.
     * 
    */
    void init()
    {
        // INIT PARENT DATA STRUCTURE ---------------------------------
        amrex::dG::amr::SinglePatch::init();
        // ------------------------------------------------------------

        // INIT SOLUTION FROM SCRATCH ---------------------------------
        if (this->inputs.restart < 0)
        {
            const amrex::Real t = 0.0;
            this->InitFromScratch(t);
        }
        // ------------------------------------------------------------
        else
        // INIT SOLUTION FROM CHECKPOINT ------------------------------
        {
amrex::Print() << "INIT SOLUTION FROM CHECKPOINT" << std::endl;
exit(-1);
        }
        // ------------------------------------------------------------

        // UPDATE MASKS -----------------------------------------------
        this->update_masks();
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int mask_n_grow = 1;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::dG::Mesh & mesh = *this->meshes[lev];
        amrex::dG::Solution & solution = *this->solutions[lev];
        bool level_uses_FV;
        // ------------------------------------------------------------

        // FV LEVEL ---------------------------------------------------
        level_uses_FV = ((solution.params.space_p == 0) && solution.params.FV_is_active);
        // ------------------------------------------------------------

        // MESH -------------------------------------------------------
        if (mesh.uses_level_set())
        {
            mesh.make_from_scratch_by_level_set(time, this->geom[lev], ba, dm, this->ibvp, this->level_set);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::MakeNewLevelFromScratch\n";
            msg += "| Unexpected method used to define the embedded geometry.\n";
            msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
            amrex::Abort(msg);
        }
        this->SetBoxArray(lev, mesh.ba);
        this->SetDistributionMap(lev, mesh.dm);
        // ------------------------------------------------------------

        // SOLUTION ---------------------------------------------------
        solution.make_from_scratch(time, this->geom[lev], mesh, this->ibvp);
        solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);
        if (level_uses_FV)
        {
            solution.eval_space_centroids(this->geom[lev], mesh, this->ibvp, level_uses_FV);
        }

        // MULTIFABS
        solution.init_space_solution_multifabs(mesh, this->X[lev], this->ibvp);
        solution.project_initial_conditions(this->geom[lev], mesh, this->X[lev], this->ibvp, level_uses_FV);
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(mesh.ba, mesh.dm, 1, mask_n_grow);
        this->masks[lev] = 0;
        // ------------------------------------------------------------
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    void eval_error(const amrex::Real t, amrex::Real & err, amrex::Real & err_norm) const
    {
        if (this->norm.compare("L_inf") == 0)
        {
            this->eval_error_L_inf(t, this->X, err, err_norm, this->ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::eval_error\n";
            msg += "| Unexpected error norm.\n";
            msg += "| requested norm           : "+this->norm+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t) const
    {
        return amrex::dG::hyperbolic_pde::eval_dt(*this, t, this->X, this->ibvp);
    }

    void take_time_step(const amrex::Real t, const amrex::Real dt)
    {
        amrex::dG::hyperbolic_pde::take_time_step(*this, t, dt, this->X, this->ibvp);
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void export_mesh(const int n, const std::string & filename_root)
    {
        this->export_mesh_to_VTK(n, filename_root, this->ibvp);
    }

    void export_quadrature_points(const int n, const std::string & filename_root)
    {
        this->export_quadrature_points_to_VTK(n, filename_root, this->ibvp);
    }

    void export_solution(const int n, const std::string & filename_root, const amrex::Real t)
    {
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const amrex::dG::Mesh & mesh = *this->meshes[lev];
            const amrex::dG::Solution & solution = *this->solutions[lev];
            const amrex::iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            if (mesh.uses_level_set())
            {
                if ((solution.params.space_p == 0) && solution.params.FV_is_active)
                {
                    amrex::dG::solution_io::export_FV_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                                             t, this->geom[lev], mesh, solution, this->X[lev], mask, this->ibvp, this->level_set);
                }
                else
                {
                    amrex::dG::solution_io::export_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                                          t, this->geom[lev], solution, this->X[lev], mask, this->ibvp, this->level_set);
                }
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::export_solution\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
        }
    }
    // ================================================================
};
// ####################################################################

} // namespace single_phase_domain