//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_TwoPhaseDomain.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_ElasticSolid.H"

#define N_PHASES 2

namespace two_phase_domain
{
// TWO-PHASE DOMAIN: LEVEL SET ########################################
template <int N>
struct LevelSet
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    LevelSet()
    :
    t{0.0},
    dom{0}
    {}
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval(const amrex::Real t_, const amrex::Real * x) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real res = std::cos(2.0*M_PI*x[0])*std::cos(2.0*M_PI*x[1])-0.125;
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real res = -std::cos(2.0*M_PI*x[0])*std::cos(2.0*M_PI*x[1])*std::cos(2.0*M_PI*x[2])-0.125;
#endif
        //const amrex::Real res = +1.0;

        return res;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval(const amrex::Real * x) const
    {
        return this->eval(0.0, x);
    }
    // ================================================================
};
// ####################################################################



// TWO-PHASE DOMAIN: IBVP #############################################
struct IBVP
:
public amrex::dG::IBVP_Base
{
    // DATA MEMBERS ===================================================
    amrex::Real density[N_PHASES], c[N_PHASES][NS*(NS+1)/2], max_wave_speed;
    
    // INITIAL CONDITION / EXACT SOLUTION
    amrex::Real wn[AMREX_SPACEDIM];
    amrex::Real w[N_VS], R[N_VS*N_VS], cR[N_VS];
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        const amrex::Array<std::string, N_PHASES> phase_string = {"phase_a_", "phase_b_"};

        amrex::ParmParse pp;
        amrex::Array<std::string, N_PHASES> material_type;
        amrex::Array<amrex::Vector<amrex::Real>, N_PHASES> material_constants;
        amrex::Array<amrex::Array<amrex::Real, NS*NS>, N_PHASES> cSE;

        // READ MATERIAL PROPERTIES -----------------------------------
        for (int a = 0; a < N_PHASES; ++a)
        {
            const std::string phase_material_type = phase_string[a]+"material_type";
            const std::string phase_material_constants = phase_string[a]+"material_constants";
            
            pp.get(phase_material_type.c_str(), material_type[a]);

            if (material_type[a].compare("isotropic") == 0)
            {
                amrex::Real E, nu;

                pp.getarr(phase_material_constants.c_str(), material_constants[a]);

                this->density[a] = material_constants[a][0];
                E = material_constants[a][1];
                nu = material_constants[a][2];
                
#if (AMREX_SPACEDIM == 2)
                linear_elasticity::get_stiffness_matrix_2d_isotropic(E, nu, &(cSE[a][0]));
                //linear_elasticity::get_stiffness_matrix_2d_isotropic_plane_stress(E, nu, &(cSE[a][0]));
#endif
#if (AMREX_SPACEDIM == 3)
                linear_elasticity::get_stiffness_matrix_3d_isotropic(E, nu, &(cSE[a][0]));
#endif
            }
            else if (material_type[a].compare("orthotropic") == 0)
            {
#if (AMREX_SPACEDIM == 2)
                amrex::Real c11, c22, c12, c66;
#endif
#if (AMREX_SPACEDIM == 3)
                amrex::Real c11, c22, c33, c23, c13, c12, c44, c55, c66;
#endif
                pp.getarr(phase_material_constants.c_str(), material_constants[a]);

                this->density[a] = material_constants[a][0];
#if (AMREX_SPACEDIM == 2)
                c11 = material_constants[a][1];
                c22 = material_constants[a][2];
                c12 = material_constants[a][3];
                c66 = material_constants[a][4];
#endif
#if (AMREX_SPACEDIM == 3)
                c11 = material_constants[a][1];
                c22 = material_constants[a][2];
                c33 = material_constants[a][3];
                c23 = material_constants[a][4];
                c13 = material_constants[a][5];
                c12 = material_constants[a][6];
                c44 = material_constants[a][7];
                c55 = material_constants[a][8];
                c66 = material_constants[a][9];
#endif

#if (AMREX_SPACEDIM == 2)
                linear_elasticity::get_stiffness_matrix_2d_orthotropic(c11, c22, c12, c66, &(cSE[a][0]));
#endif
#if (AMREX_SPACEDIM == 3)
                linear_elasticity::get_stiffness_matrix_3d_orthotropic(c11, c22, c33, c23, c13, c12, c44, c55, c66, &(cSE[a][0]));
#endif
            }
            else if (material_type[a].compare("anisotropic") == 0)
            {
                const amrex::Real * cij;
                pp.getarr(phase_material_constants.c_str(), material_constants[a]);

                this->density[a] = material_constants[a][0];
                cij = &material_constants[a][1];

                amrex::Print() << "cij[3]: " << cij[3] << std::endl;

#if (AMREX_SPACEDIM == 2)
                cSE[a][0+0*NS] = cij[0]; cSE[a][0+1*NS] = cij[1]; cSE[a][0+2*NS] = cij[2];
                cSE[a][1+0*NS] = cij[1]; cSE[a][1+1*NS] = cij[3]; cSE[a][1+2*NS] = cij[4];
                cSE[a][2+0*NS] = cij[2]; cSE[a][2+1*NS] = cij[4]; cSE[a][2+2*NS] = cij[5];
#endif
#if (AMREX_SPACEDIM == 3)
                cSE[a][0+0*NS] = cij[0]; cSE[a][0+1*NS] = cij[1];  cSE[a][0+2*NS] = cij[2];  cSE[a][0+3*NS] = cij[3];  cSE[a][0+4*NS] = cij[4];  cSE[a][0+5*NS] = cij[5];
                cSE[a][1+0*NS] = cij[1]; cSE[a][1+1*NS] = cij[6];  cSE[a][1+2*NS] = cij[7];  cSE[a][1+3*NS] = cij[8];  cSE[a][1+4*NS] = cij[9];  cSE[a][1+5*NS] = cij[10];
                cSE[a][2+0*NS] = cij[2]; cSE[a][2+1*NS] = cij[7];  cSE[a][2+2*NS] = cij[11]; cSE[a][2+3*NS] = cij[12]; cSE[a][2+4*NS] = cij[13]; cSE[a][2+5*NS] = cij[14];
                cSE[a][3+0*NS] = cij[3]; cSE[a][3+1*NS] = cij[8];  cSE[a][3+2*NS] = cij[12]; cSE[a][3+3*NS] = cij[15]; cSE[a][3+4*NS] = cij[16]; cSE[a][3+5*NS] = cij[17];
                cSE[a][4+0*NS] = cij[4]; cSE[a][4+1*NS] = cij[9];  cSE[a][4+2*NS] = cij[13]; cSE[a][4+3*NS] = cij[16]; cSE[a][4+4*NS] = cij[18]; cSE[a][4+5*NS] = cij[19];
                cSE[a][5+0*NS] = cij[5]; cSE[a][5+1*NS] = cij[10]; cSE[a][5+2*NS] = cij[14]; cSE[a][5+3*NS] = cij[17]; cSE[a][5+4*NS] = cij[19]; cSE[a][5+5*NS] = cij[20];
#endif
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_TwoPhaseDomain.H - IBVP::IBVP\n";
                msg += "| Unexpected input material type: "+material_type[a]+".\n";
                amrex::Abort(msg);
            }
        }

        /* DEBUG */
        amrex::Print() << "Phase a - cSE: " << std::endl;
        amrex::dG::io::print_real_array_2d(NS, NS, &(cSE[0][0]));
        amrex::Print() << "Phase b - cSE: " << std::endl;
        amrex::dG::io::print_real_array_2d(NS, NS, &(cSE[1][0]));
        /**/
        // ------------------------------------------------------------

        // STORE STIFFNESS MATRIX COEFFICIENTS ------------------------
        for (int a = 0; a < N_PHASES; ++a)
        {
#if (AMREX_SPACEDIM == 2)
            this->c[a][C11] = cSE[a][0+0*NS];
            this->c[a][C21] = cSE[a][1+0*NS]; this->c[a][C22] = cSE[a][1+1*NS];
            this->c[a][C61] = cSE[a][2+0*NS]; this->c[a][C62] = cSE[a][2+1*NS]; this->c[a][C66] = cSE[a][2+2*NS];
#endif
#if (AMREX_SPACEDIM == 3)
            this->c[a][C11] = cSE[a][0+0*NS];
            this->c[a][C21] = cSE[a][1+0*NS]; this->c[a][C22] = cSE[a][1+1*NS];
            this->c[a][C31] = cSE[a][2+0*NS]; this->c[a][C32] = cSE[a][2+1*NS]; this->c[a][C33] = cSE[a][2+2*NS];
            this->c[a][C41] = cSE[a][3+0*NS]; this->c[a][C42] = cSE[a][3+1*NS]; this->c[a][C43] = cSE[a][3+2*NS]; this->c[a][C44] = cSE[a][3+3*NS];
            this->c[a][C51] = cSE[a][4+0*NS]; this->c[a][C52] = cSE[a][4+1*NS]; this->c[a][C53] = cSE[a][4+2*NS]; this->c[a][C54] = cSE[a][4+3*NS]; this->c[a][C55] = cSE[a][4+4*NS];
            this->c[a][C61] = cSE[a][5+0*NS]; this->c[a][C62] = cSE[a][5+1*NS]; this->c[a][C63] = cSE[a][5+2*NS]; this->c[a][C64] = cSE[a][5+3*NS]; this->c[a][C65] = cSE[a][5+4*NS]; this->c[a][C66] = cSE[a][5+5*NS];
#endif
        }
        // ------------------------------------------------------------

        // MAX WAVE SPEED ---------------------------------------------
        this->max_wave_speed = 0.0;

        for (int a = 0; a < N_PHASES; ++a)
        {
            if (material_type[a].compare("isotropic") == 0)
            {
                const amrex::Real G = this->c[a][C66];
                const amrex::Real lam = this->c[a][C21];

                this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt((2.0*G+lam)/this->density[a]));
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

#if (AMREX_SPACEDIM == 2)
                for (int i = 0; i < n_th; ++i)
                {
                    th = i*dth;
                    un[0] = std::cos(th);
                    un[1] = std::sin(th);
                    elastic_solid::eval_eig_InTcIn_compact_c(this->density[a], &(this->c[a][0]), un, w2, v);
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
                    elastic_solid::eval_eig_InTcIn_compact_c(this->density[a], &(this->c[a][0]), un, w2, v);
                    this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt(*std::max_element(w2, w2+AMREX_SPACEDIM)));
                }
#endif
            }
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
            const int a = 0;

            amrex::Real An[N_VS*N_VS], wAn[N_VS], vAn[N_VS*N_VS];
            
            // Wave number
#if (AMREX_SPACEDIM == 2)
            const amrex::Real th = 0.0*M_PI/6.0;
            const amrex::Real cth = std::cos(th);
            const amrex::Real sth = std::sin(th);
            this->wn[0] = 2.0*M_PI*cth;
            this->wn[1] = 2.0*M_PI*sth;
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real th = 0.0;
            const amrex::Real ph = M_PI/2.0;
            const amrex::Real cth = std::cos(th);
            const amrex::Real sth = std::sin(th);
            const amrex::Real cph = std::cos(ph);
            const amrex::Real sph = std::sin(ph);
            this->wn[0] = 2.0*M_PI*cth*sph;
            this->wn[1] = 2.0*M_PI*sth*sph;
            this->wn[2] = 2.0*M_PI*cph;
#endif

            // Evaluate An = A_k*wk[k]
            elastic_solid::eval_An_compact_c(this->density[a], &(this->c[a][0]), this->wn, An);

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

            this->cR[E11] = 1.0;
            this->cR[E22] = 1.0;
            this->cR[E12] = 1.0;
#endif
#if (AMREX_SPACEDIM == 3)
            this->cR[V1] = 1.0;
            this->cR[V2] = 1.0;
            this->cR[V3] = 1.0;

            this->cR[E11] = 1.0;
            this->cR[E22] = 1.0;
            this->cR[E33] = 1.0;
            this->cR[E23] = 1.0;
            this->cR[E13] = 1.0;
            this->cR[E12] = 1.0;
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
        return 2;
    }
    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        return N_PHASES*N_VS;
    }
    /**
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int dom, int & u_lo, int & u_hi) const
    {
        u_lo = dom*N_VS;
        u_hi = (dom+1)*N_VS;
    }
    /**
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int unknown_field_parent_domain(const int u) const
    {
        return ((u < N_VS) ? 0 : 1);
    }
    /**
     * \brief Return the domain that is neighbor with domain dom.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_nbr_dom(const int dom) const
    {
        return 1-dom;
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
        amrex::Real * Ua = U;
        amrex::Real * Ub = &U[N_VS];
        
        for (int r = 0; r < N_VS; ++r)
        {
            Ua[r] = 0.0;
            for (int k = 0; k < N_VS; ++k)
            {
                Ua[r] += this->R[r+k*N_VS]*this->cR[k]*std::sin(this->w[k]*t-x_dot_wn);
            }
            Ub[r] = Ua[r];
        }

        /* CONSTANT STATE
        for (int u = 0; u < N_PHASES*N_VS; ++u)
        {
            U[u] = 1.0;
        }
        */
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
    void eval_error_L_inf(const int dom,
                          const amrex::Real t, const amrex::Real * x,
                          const amrex::Real * U,
                          amrex::Real & err, amrex::Real & err_norm) const
    {
        const amrex::Real * Ua = U;
        const amrex::Real * Ub = &U[N_VS];
        amrex::Real U_exact[N_PHASES*N_VS];

        this->eval_exact_solution(t, x, U_exact);
        const amrex::Real * Ua_exact = U_exact;
        const amrex::Real * Ub_exact = &U_exact[N_VS];

        if (dom == 0)
        {
#if (AMREX_SPACEDIM == 2)
            err =                 std::abs(Ua[V1]-Ua_exact[V1]);
            err = amrex::max(err, std::abs(Ua[V2]-Ua_exact[V2]));
            err = amrex::max(err, std::abs(Ua[E11]-Ua_exact[E11]));
            err = amrex::max(err, std::abs(Ua[E22]-Ua_exact[E22]));
            err = amrex::max(err, std::abs(Ua[E12]-Ua_exact[E12]));

            err_norm =                      std::abs(Ua_exact[V1]);
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[V2]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E11]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E22]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E12]));
#endif
#if (AMREX_SPACEDIM == 3)
            err =                 std::abs(Ua[V1]-Ua_exact[V1]);
            err = amrex::max(err, std::abs(Ua[V2]-Ua_exact[V2]));
            err = amrex::max(err, std::abs(Ua[V3]-Ua_exact[V3]));
            err = amrex::max(err, std::abs(Ua[E11]-Ua_exact[E11]));
            err = amrex::max(err, std::abs(Ua[E22]-Ua_exact[E22]));
            err = amrex::max(err, std::abs(Ua[E33]-Ua_exact[E33]));
            err = amrex::max(err, std::abs(Ua[E23]-Ua_exact[E23]));
            err = amrex::max(err, std::abs(Ua[E13]-Ua_exact[E13]));
            err = amrex::max(err, std::abs(Ua[E12]-Ua_exact[E12]));

            err_norm =                      std::abs(Ua_exact[V1]);
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[V2]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[V3]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E11]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E22]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E33]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E23]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E13]));
            err_norm = amrex::max(err_norm, std::abs(Ua_exact[E12]));
#endif
        }
        else
        {
#if (AMREX_SPACEDIM == 2)
            err =                 std::abs(Ub[V1]-Ub_exact[V1]);
            err = amrex::max(err, std::abs(Ub[V2]-Ub_exact[V2]));
            err = amrex::max(err, std::abs(Ub[E11]-Ub_exact[E11]));
            err = amrex::max(err, std::abs(Ub[E22]-Ub_exact[E22]));
            err = amrex::max(err, std::abs(Ub[E12]-Ub_exact[E12]));

            err_norm =                      std::abs(Ub_exact[V1]);
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[V2]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E11]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E22]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E12]));
#endif
#if (AMREX_SPACEDIM == 3)
            err =                 std::abs(Ub[V1]-Ub_exact[V1]);
            err = amrex::max(err, std::abs(Ub[V2]-Ub_exact[V2]));
            err = amrex::max(err, std::abs(Ub[V3]-Ub_exact[V3]));
            err = amrex::max(err, std::abs(Ub[E11]-Ub_exact[E11]));
            err = amrex::max(err, std::abs(Ub[E22]-Ub_exact[E22]));
            err = amrex::max(err, std::abs(Ub[E33]-Ub_exact[E33]));
            err = amrex::max(err, std::abs(Ub[E23]-Ub_exact[E23]));
            err = amrex::max(err, std::abs(Ub[E13]-Ub_exact[E13]));
            err = amrex::max(err, std::abs(Ub[E12]-Ub_exact[E12]));

            err_norm =                      std::abs(Ub_exact[V1]);
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[V2]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[V3]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E11]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E22]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E33]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E23]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E13]));
            err_norm = amrex::max(err_norm, std::abs(Ub_exact[E12]));
#endif
        }
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
    void eval_F(const int dom,
                const amrex::Real /*t*/, const amrex::Real * /*x*/,
                const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
        if (dom == 0)
        {
            elastic_solid::eval_F_compact_c(this->density[dom], &(this->c[dom][0]), U, AMREX_D_DECL(F1, F2, F3));
        }
        else
        {
            elastic_solid::eval_F_compact_c(this->density[dom], &(this->c[dom][0]), &U[N_VS], AMREX_D_DECL(&F1[N_VS], &F2[N_VS], &F3[N_VS]));
        }
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
    void eval_NFn_intraphase(const int dom,
                             const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn) const
    {
        if (dom == 0)
        {
            elastic_solid::eval_NFn_Riemann_solver(this->density[dom], &(this->c[dom][0]), un, m_U, p_U, NFn);
        }
        else
        {
            elastic_solid::eval_NFn_Riemann_solver(this->density[dom], &(this->c[dom][0]), un, &m_U[N_VS], &p_U[N_VS], &NFn[N_VS]);
        }
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
amrex::Print() << "IBVP.eval_NFn_BCs" << std::endl;
exit(-1);
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
amrex::Print() << "IBVP.eval_NFn_EB" << std::endl;
exit(-1);
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] nbr_dom: index of the domain that is neighbor with dom.
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
    void eval_NFn_EB(const int dom, const int nbr_dom,
                     const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                     const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn) const
    {
        if (dom == 0)
        {
            elastic_solid::eval_NFn_Riemann_solver(this->density[dom], &(this->c[dom][0]), un, m_U, &p_U[N_VS], NFn);
        }
        else
        {
            elastic_solid::eval_NFn_Riemann_solver(this->density[dom], &(this->c[dom][0]), un, &m_U[N_VS], p_U, &NFn[N_VS]);
        }
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
        return N_PHASES*(N_VS+2);
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int u) const
    {
        return ((u < (N_VS+2)) ? 0 : 1);
    }
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
#if (AMREX_SPACEDIM == 2)
        output_fields_names = {"phase_a_m_x", "phase_a_m_y",
                               "phase_a_g_xx", "phase_a_g_yy", "phase_a_g_xy",
                               "phase_a_m_x_error", "phase_a_g_xx_error",
                               "phase_b_m_x", "phase_b_m_y",
                               "phase_b_g_xx", "phase_b_g_yy", "phase_b_g_xy",
                               "phase_b_m_x_error", "phase_b_g_xx_error"};
#endif
#if (AMREX_SPACEDIM == 3)
        output_fields_names = {"phase_a_m_x", "phase_a_m_y", "phase_a_m_z",
                               "phase_a_g_xx", "phase_a_g_yy", "phase_a_g_zz", "phase_a_g_yz", "phase_a_g_xz", "phase_a_g_xy",
                               "phase_a_m_x_error", "phase_a_g_xx_error",
                               "phase_b_m_x", "phase_b_m_y", "phase_b_m_z",
                               "phase_b_g_xx", "phase_b_g_yy", "phase_b_g_zz", "phase_b_g_yz", "phase_b_g_xz", "phase_b_g_xy",
                               "phase_b_m_x_error", "phase_b_g_xx_error"};
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
    void eval_output_fields(const int dom, const amrex::Real t, const amrex::Real * x,
                            const amrex::Real * U,
                            amrex::Real * F) const
    {
        const amrex::Real * Ua = U;
        const amrex::Real * Ub = &U[N_VS];
        amrex::Real * Fa = F;
        amrex::Real * Fb = &F[N_VS+2];
        amrex::Real U_exact[N_PHASES*N_VS];

        this->eval_exact_solution(t, x, U_exact);
        const amrex::Real * Ua_exact = U_exact;
        const amrex::Real * Ub_exact = &U_exact[N_VS];

        if (dom == 0)
        {
#if (AMREX_SPACEDIM == 2)
            Fa[V1] = Ua[V1];
            Fa[V2] = Ua[V2];

            Fa[E11] = Ua[E11];
            Fa[E22] = Ua[E22];
            Fa[E12] = Ua[E12];
#endif
#if (AMREX_SPACEDIM == 3)
            Fa[V1] = Ua[V1];
            Fa[V2] = Ua[V2];
            Fa[V3] = Ua[V3];

            Fa[E11] = Ua[E11];
            Fa[E22] = Ua[E22];
            Fa[E33] = Ua[E33];
            Fa[E23] = Ua[E23];
            Fa[E13] = Ua[E13];
            Fa[E12] = Ua[E12];
#endif
            Fa[E12+1] = Ua[V1]-Ua_exact[V1];
            Fa[E12+2] = Ua[E11]-Ua_exact[E11];
        }
        else
        {
#if (AMREX_SPACEDIM == 2)
            Fb[V1] = Ub[V1];
            Fb[V2] = Ub[V2];

            Fb[E11] = Ub[E11];
            Fb[E22] = Ub[E22];
            Fb[E12] = Ub[E12];
#endif
#if (AMREX_SPACEDIM == 3)
            Fb[V1] = Ub[V1];
            Fb[V2] = Ub[V2];
            Fb[V3] = Ub[V3];

            Fb[E11] = Ub[E11];
            Fb[E22] = Ub[E22];
            Fb[E33] = Ub[E33];
            Fb[E23] = Ub[E23];
            Fb[E13] = Ub[E13];
            Fb[E12] = Ub[E12];
#endif
            Fb[E12+1] = Ub[V1]-Ub_exact[V1];
            Fb[E12+2] = Ub[E11]-Ub_exact[E11];
        }
    }
    // ================================================================
};
// ####################################################################



// TWO-PHASE DOMAIN: AMR ##############################################
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
    amrex::Vector<amrex::MultiFab> X, L;

    // NORM
    std::string norm;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        // SOLUTION / LEVEL SET MULTIFABS
        this->X.resize(n_levels);
        this->L.resize(n_levels);

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
        // ------------------------------------------------------------

        // MESH -------------------------------------------------------
        if (mesh.uses_projected_level_set())
        {
            mesh.init_projected_level_set_multifabs(ba, dm, this->L[lev]);
            mesh.project_space_level_set(time, this->geom[lev], this->L[lev], this->level_set);
            mesh.make_from_scratch_by_projected_level_set(time, this->geom[lev], ba, dm, this->L[lev], this->ibvp);
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

        // MULTIFABS
        solution.init_space_solution_multifabs(mesh, this->X[lev], this->ibvp);
        solution.project_initial_conditions(this->geom[lev], mesh, this->X[lev], this->ibvp);
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

            if (mesh.uses_projected_level_set())
            {
                amrex::dG::solution_io::export_using_projected_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                                                t, this->geom[lev], mesh, solution, this->X[lev], this->L[lev], mask, this->ibvp);
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

} // namespace two_phase_domain