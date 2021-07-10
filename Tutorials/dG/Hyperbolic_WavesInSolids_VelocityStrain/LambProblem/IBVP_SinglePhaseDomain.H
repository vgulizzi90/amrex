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
    amrex::Real eval(const amrex::Real /*t_*/, const amrex::Real * x) const
    {
        const amrex::Real th = 10.0*M_PI/180.0;
        const amrex::Real tth = std::tan(th);
        const amrex::Real res = x[1]-2000.0-tth*x[0];

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



// SINGLE PHASE DOMAIN: IBVP ##########################################
struct IBVP
:
public amrex::dG::IBVP_Base
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
        else if (material_type.compare("isotropic_wave_velocities") == 0)
        {
            amrex::Real cP, cS;

            pp.getarr("material_constants", material_constants);

            this->density = material_constants[0];
            cP = material_constants[1];
            cS = material_constants[2];
            
#if (AMREX_SPACEDIM == 2)
            linear_elasticity::get_stiffness_matrix_2d_isotropic_wave_velocities(this->density, cP, cS, cSE);
#endif
#if (AMREX_SPACEDIM == 3)
            linear_elasticity::get_stiffness_matrix_3d_isotropic_wave_velocities(this->density, cP, cS, cSE);
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
        if ((material_type.compare("isotropic") == 0) ||
            (material_type.compare("isotropic_wave_velocities") == 0))
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
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int /*dom*/, int & u_lo, int & u_hi) const
    {
        u_lo = 0;
        u_hi = N_VS;
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
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs(const amrex::Real * /*x*/, amrex::Real * U) const
    {
#if (AMREX_SPACEDIM == 2)
        U[V1] = 0.0; // hpAMR
        U[V2] = 0.0; // hpAMR
        U[E11] = 0.0;
        U[E22] = 0.0;
        U[E12] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        U[V1] = 0.0;
        U[V2] = 0.0;
        U[V3] = 0.0;
        U[E11] = 0.0;
        U[E22] = 0.0;
        U[E33] = 0.0;
        U[E23] = 0.0;
        U[E13] = 0.0;
        U[E12] = 0.0;
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
                        const amrex::Real * /*U*/) const
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

    /**
     * \brief Eval the source for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] S: pointer to memory that will contain the source term.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_S(const int /*dom*/,
                const amrex::Real t, const amrex::Real * /*x*/,
                const amrex::Real * /*U*/, amrex::Real * S) const
    {
        const amrex::Real a1 = -2000.0;
        const amrex::Real f = 14.5;
        const amrex::Real t0 = 0.08;
        const amrex::Real Rw = elastic_solid::Ricker_wavelet(a1, f, t0, t);

#if (AMREX_SPACEDIM == 2)
        const amrex::Real th = 10.0*M_PI/180.0;
        const amrex::Real cth = std::cos(th);
        const amrex::Real sth = std::sin(th);

        S[V1] = -Rw*sth;
        S[V2] = Rw*cth;
        S[E11] = 0.0;
        S[E22] = 0.0;
        S[E12] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        S[V1] = 0.0;
        S[V2] = 0.0;
        S[V3] = Rw;
        S[E11] = 0.0;
        S[E22] = 0.0;
        S[E33] = 0.0;
        S[E23] = 0.0;
        S[E13] = 0.0;
        S[E12] = 0.0;
#endif
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
    void eval_NFn_BCs(const int /*dom*/,
                      const amrex::Real /*t*/, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
#if (AMREX_SPACEDIM == 2)
        /* // hpAMR */
        if (un[1] > 0.5)
        {
            const amrex::Real T[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
            elastic_solid::eval_NFn_Riemann_solver_prescribed_tractions(this->density, this->c, un, U, T, NFn);
        }
        else
        {
            elastic_solid::eval_NFn_Riemann_solver_absorbing_boundary(this->density, this->c, un, U, NFn);
        }
        /**/

        // hpAMR
        //amrex::Real U_0[N_VS];
        //this->eval_ICs(x, U_0);
        //elastic_solid::eval_NFn_Riemann_solver(this->density, this->c, un, U, U_0, NFn);
        
#endif
#if (AMREX_SPACEDIM == 3)
amrex::Print() << "eval_NFn_BCs" << std::endl;
exit(-1);
#endif
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
    void eval_NFn_EB(const int /*dom*/,
                      const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        const amrex::Real T[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
        elastic_solid::eval_NFn_Riemann_solver_prescribed_tractions(this->density, this->c, un, U, T, NFn);
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
    void eval_NFn_EB(const int /*dom*/, const int /*nbr_dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                     const amrex::Real * /*m_U*/, const amrex::Real * /*p_U*/, amrex::Real * /*NFn*/) const
    {
amrex::Print() << "eval_NFn_EB" << std::endl;
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
        return N_VS+1;
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
        output_fields_names = {"m_x", "m_y",
                               "g_xx", "g_yy", "g_xy", "total_energy"};
#endif
#if (AMREX_SPACEDIM == 3)
        output_fields_names = {"m_x", "m_y", "m_z",
                               "g_xx", "g_yy", "s_zz", "g_yz", "g_xz", "g_xy", "total_energy"};
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
    void eval_output_fields(const int /*dom*/, const amrex::Real /*t*/, const amrex::Real * /*x*/,
                            const amrex::Real * U,
                            amrex::Real * F) const
    {
#if (AMREX_SPACEDIM == 2)
        F[V1] = U[V1];
        F[V2] = U[V2];

        F[E11] = U[E11];
        F[E22] = U[E22];
        F[E12] = U[E12];
#endif
#if (AMREX_SPACEDIM == 3)
        F[V1] = U[V1];
        F[V2] = U[V2];
        F[V3] = U[V3];

        F[E11] = U[E11];
        F[E22] = U[E22];
        F[E33] = U[E33];
        F[E23] = U[E23];
        F[E13] = U[E13];
        F[E12] = U[E12];
#endif
        F[E12+1] = elastic_solid::eval_total_energy_compact_c(this->density, this->c, U);
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
    amrex::Real t;

    // LEVEL SET
    LevelSet<AMREX_SPACEDIM> level_set;
    
    // IBVP
    IBVP ibvp;

    // SOLUTION MULTIFABS
    amrex::Vector<amrex::MultiFab> X, L;

    // AMR
    amrex::Vector<amrex::Real> threshold_amr;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        // SOLUTION MULTIFABS
        this->X.resize(n_levels);
        this->L.resize(n_levels);

        // AMR
        this->threshold_amr.resize(n_levels);
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

        // THRESHOLD PARAMETERS ---------------------------------------
        if (this->max_level > 0)
        {
            amrex::ParmParse pp;
            pp.getarr("threshold_amr", this->threshold_amr);
        }
        // ------------------------------------------------------------

        // INIT SOLUTION FROM SCRATCH ---------------------------------
        if (this->inputs.restart < 0)
        {
            this->t = 0.0;
            this->InitFromScratch(this->t);
        }
        // ------------------------------------------------------------
        else
        // INIT SOLUTION FROM CHECKPOINT ------------------------------
        {
            if (this->meshes[0]->uses_projected_level_set())
            {
                this->init_from_checkpoint_using_projected_level_set(this->inputs.restart, "checkpoint", this->X, this->L, this->ibvp);
                this->t = this->inputs.restart_time;
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::init\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+this->meshes[0]->params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------

        // UPDATE MASKS -----------------------------------------------
        this->update_masks();
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
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
            mesh.project_space_level_set(this->t, this->geom[lev], this->L[lev], this->level_set);
            mesh.make_from_scratch_by_projected_level_set(this->t, this->geom[lev], ba, dm, this->L[lev], this->ibvp);
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
        solution.make_from_scratch(this->t, this->geom[lev], mesh, this->ibvp);
        solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);

        // INTERPOLATION OPERATOR
        if (lev > 0)
        {
            this->eval_space_interpolation(lev, this->ibvp);
        }

        // MULTIFABS
        solution.init_space_solution_multifabs(mesh, this->X[lev], this->ibvp);
        solution.project_initial_conditions(this->geom[lev], mesh, this->X[lev], this->ibvp);
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(mesh.ba, mesh.dm, 1, mask_n_grow);
        this->masks[lev] = __DG_CELL_NOT_MASKED__;
        // ------------------------------------------------------------
    }
    // ================================================================


    // TAG CELL FOR REFINEMENT ========================================
    virtual void ErrorEst(int lev, amrex::TagBoxArray & tags, amrex::Real /*time*/, int /*ngr*/) override
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("single_phase_domain::AMR::ErrorEst(...)");
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const char non_tagged = amrex::TagBox::CLEAR;
        const char tagged = amrex::TagBox::SET;

        const amrex::dG::Mesh & mesh = *this->meshes[lev];
        const amrex::dG::Solution & solution = *this->solutions[lev];

        // DOMAIN
        const int dom = 0;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        tags.setVal(non_tagged);
        // ------------------------------------------------------------

        // MANUAL TAGGING AT THE CONCENTRATED SOURCE ------------------
        if (this->t < 0.2) // hpAMR
        {
            const amrex::Real th = 10.0*M_PI/180.0;
            const amrex::Real tth = std::tan(th);
            const amrex::Real x1S = 1720.0;
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xs = {AMREX_D_DECL(x1S, 2000.0+tth*x1S, 0.0)}; // hpAMR
            //const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xs = {x1S, 2000.0+tth*x1S-2*2000.0*t}; // hpAMR

            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->geom[lev].CellSizeArray();
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->geom[lev].ProbLoArray();

            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();

                amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
                amrex::Array4<char> const & tags_fab = tags.array(mfi);
                
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                    if (amrex::dG::ELM_IS_NOT_EMPTY(etype) && (tags_fab(i,j,k) != tagged))
                    {
                        // LOCAL PARAMETERS
                        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                             prob_lo[1]+(j+0.5)*dx[1],
                                                                             prob_lo[2]+(k+0.5)*dx[2])};
                        const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(xc[0]-xs[0], xc[1]-xs[1], xc[2]-xs[2])};
                        const amrex::Real d = std::sqrt(AMREX_D_TERM(y[0]*y[0],+y[1]*y[1],+y[2]*y[2]));

                        if (d < 2.0*amrex::max(AMREX_D_DECL(dx[0], dx[1], dx[2])))
                        {
                            tags_fab(i,j,k) = tagged;
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
            tags.FillBoundary(this->geom[lev].periodicity());
        }
        // ------------------------------------------------------------

        // TAGGING ----------------------------------------------------
        elastic_solid::tag_cells_single_phase_domain(this->ibvp.density, this->ibvp.c, this->Geom(lev), mesh, solution, this->X[lev], this->threshold_amr[lev], tags, tagged); // hpAMR
        // ------------------------------------------------------------
    }
    // ================================================================


    // REMAKE AN EXISTING LEVEL =======================================
    virtual void RemakeLevel(int lev, amrex::Real /*time*/, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("single_phase_domain::AMR::RemakeLevel(...)");
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const int mask_n_grow = 1;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::dG::Mesh & mesh = *this->meshes[lev];
        amrex::dG::Solution & solution = *this->solutions[lev];
        // ------------------------------------------------------------

        // LEVEL = 0
        if (lev == 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::RemakeLevel\n";
            msg += "| Do not know how to remake level 0.\n";
            amrex::Abort(msg);
        }
        // LEVEL > 0
        else
        {
            // MESH ---------------------------------------------------
            if (mesh.uses_projected_level_set())
            {
                amrex::dG::amr_utils::remake_level_set(this->t, ba, dm,
                                                       this->geom[lev-1], *this->meshes[lev-1], this->L[lev-1],
                                                       this->refRatio(lev-1),
                                                       this->geom[lev], mesh, this->L[lev],
                                                       ibvp);
                mesh.make_from_scratch_by_projected_level_set(this->t, this->geom[lev], ba, dm, this->L[lev], this->ibvp);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::RemakeLevel\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
            this->SetBoxArray(lev, mesh.ba);
            this->SetDistributionMap(lev, mesh.dm);
            // --------------------------------------------------------

            // SOLUTION -----------------------------------------------
            solution.remake(this->t, this->geom[lev], mesh, this->ibvp);
            solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);

            // INTERPOLATION OPERATOR
            this->eval_space_interpolation(lev, this->ibvp);

            // MULTIFABS
            amrex::dG::amr_utils::remake_solution(this->t, this->geom[lev-1], *this->solutions[lev-1], *this->meshes[lev-1], this->X[lev-1],
                                                  this->refRatio(lev-1),
                                                  this->geom[lev], solution, mesh, this->X[lev],
                                                  ibvp);
            // --------------------------------------------------------

            // MASKS --------------------------------------------------
            this->masks[lev].define(mesh.ba, mesh.dm, 1, mask_n_grow);
            this->masks[lev] = __DG_CELL_NOT_MASKED__;
            // --------------------------------------------------------
        }
    }
    // ================================================================


    // CLEAR LEVEL ====================================================
    virtual void ClearLevel(int /*lev*/) override
    {}
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t_) const
    {
        return amrex::dG::hyperbolic_pde::eval_dt(*this, t_, this->X, this->ibvp);
    }

    void take_time_step(const amrex::Real t_, const amrex::Real dt)
    {
        amrex::dG::hyperbolic_pde::take_time_step(*this, t_, dt, this->X, this->ibvp);
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void write_sampled_solution(const int lev, const int n, const amrex::Real t_) const
    {
        // PARAMETERS -------------------------------------------------
        // CURRENT PROCESSOR
        const int rank = amrex::ParallelDescriptor::MyProc();

        // OUTPUT FILEPATH
        const std::string filepath = amrex::dG::io::make_path({this->inputs.output_folderpath, "Level_"+std::to_string(lev)+"_sampled_solution.txt"});

        // DOMAIN
        const int dom = 0;

        // RECEIVERS INFO
        const amrex::Real th = 10.0*M_PI/180.0;
        const amrex::Real tth = std::tan(th);
        const int n_sampling_points = 1;
#if (AMREX_SPACEDIM == 2)
        const amrex::Real x1R = 2694.96;
        const amrex::Vector<amrex::Real> sampling_points_coordinates = {x1R, 2000+tth*x1R};
#endif
        const int n_sampled_fields_per_point = 2;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Vector<int> sampling_points_rank(n_sampling_points);
        amrex::Vector<amrex::Real> sampled_fields(n_sampling_points*n_sampled_fields_per_point);

        // WRITING TOOLS
        std::ofstream fp;
        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
        // ------------------------------------------------------------

        // SAMPLE THE FIELDS ------------------------------------------
        std::fill(sampling_points_rank.begin(), sampling_points_rank.end(), -1);
        std::fill(sampled_fields.begin(), sampled_fields.end(), std::sqrt(-1.0));

        {
            // GRID
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->Geom(lev).CellSizeArray();
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->Geom(lev).ProbLoArray();

            // ORDER
            const int sp = this->solutions[lev]->params.space_p;

            for (amrex::MFIter mfi(this->X[lev]); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();
                const amrex::Dim3 lo = lbound(bx);
                const amrex::Dim3 hi = ubound(bx);

                amrex::Array4<short const> const & elm_type_fab = this->solutions[lev]->elm_type.array(mfi);
                amrex::Array4<amrex::Real const> const & X_fab = this->X[lev].array(mfi);

                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    // ELEMENT TYPE
                    const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                    // LOCAL PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};
                    const amrex::Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(cell_lo[0]+dx[0],
                                                                              cell_lo[1]+dx[1],
                                                                              cell_lo[2]+dx[2])};
                    
                    // LOCAL VARIABLES
                    bool cond;
                    amrex::Real U[N_VS];

                    // UNKNOWN FIELDS
                    amrex::dG::space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);

                    for (int s = 0; s < n_sampling_points; ++s)
                    {
                        const amrex::Real * xs = &sampling_points_coordinates[AMREX_SPACEDIM*s];

                        AMREX_D_TERM
                        (
                            cond =         (cell_lo[0] <= xs[0]) && (xs[0] <= cell_hi[0]);,
                            cond = cond && (cell_lo[1] <= xs[1]) && (xs[1] <= cell_hi[1]);,
                            cond = cond && (cell_lo[2] <= xs[2]) && (xs[2] <= cell_hi[2]);
                        )
                        cond = cond && (sampling_points_rank[s] == -1);

                        if (cond)
                        {
                            sampling_points_rank[s] = rank;
                            
                            bfu.eval(xs, 0, N_VS, U);
#if (AMREX_SPACEDIM == 2)
                            sampled_fields[n_sampled_fields_per_point*s] = U[V1];
                            sampled_fields[n_sampled_fields_per_point*s+1] = U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
#endif
                        }
                    }
                }
            }
        }
        // ------------------------------------------------------------

        // SEND INFO TO IO PROCESSOR ----------------------------------
        amrex::ParallelDescriptor::ReduceIntMax(sampling_points_rank.data(), n_sampling_points);

        /*
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            for (int s = 0; s < n_sampling_points; ++s)
            {
                amrex::Print() << " sampling_points_rank[" << s << "]: " << sampling_points_rank[s] << std::endl;
            }
        }
        */
        
        for (int s = 0; s < n_sampling_points; ++s)
        {
            if (sampling_points_rank[s] != -1)
            {
                if ((sampling_points_rank[s] != rank) && (amrex::ParallelDescriptor::IOProcessor()))
                {
                    amrex::ParallelDescriptor::Recv(&sampled_fields[n_sampled_fields_per_point*s], n_sampled_fields_per_point, sampling_points_rank[s], 0);
                }
                else if ((sampling_points_rank[s] == rank) && (!amrex::ParallelDescriptor::IOProcessor()))
                {
                    amrex::ParallelDescriptor::Send(&sampled_fields[n_sampled_fields_per_point*s], n_sampled_fields_per_point, amrex::ParallelDescriptor::IOProcessorNumber(), 0);
                }
            }
        }
        // ------------------------------------------------------------

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // OPEN FILE FOR WRITING ----------------------------------
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            if (n == 0)
            {
                fp.open(filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

                if (!fp.good())
                {
                    amrex::FileOpenFailed(filepath);
                }

                fp << "SAMPLED SOLUTION\n";
                for (int s = 0; s < n_sampling_points; ++s)
                {
                    const amrex::Real * xs = &sampling_points_coordinates[AMREX_SPACEDIM*s];
#if (AMREX_SPACEDIM == 2)
                    fp << "R" << std::to_string(s) << ": " << xs[0] << " " << xs[1] << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
                    fp << "R" << std::to_string(s) << ": " << xs[0] << " " << xs[1] << " " << xs[2] << std::endl;
#endif
                }

                fp << "t ";
                for (int s = 0; s < n_sampling_points; ++s)
                {
#if (AMREX_SPACEDIM == 2)
                    fp << "v1(t,R" << std::to_string(s) << ") " << "v2(t,R" << std::to_string(s) << ") ";
#endif
#if (AMREX_SPACEDIM == 3)
#endif
                }
                fp << std::endl;
            }
            else
            {
                fp.open(filepath.c_str(), std::ofstream::app | std::ofstream::binary);

                if (!fp.good())
                {
                    amrex::FileOpenFailed(filepath);
                }
            }

            fp << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << t_;
            for (int f = 0; f < n_sampling_points*n_sampled_fields_per_point; ++f)
            {
                fp << " " << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << sampled_fields[f];
            }
            fp << std::endl;
            // --------------------------------------------------------

            // CLOSE FILE
            fp.close();
            // ----------
        }
    }

    void write_sampled_solution(const int n, const amrex::Real t_) const
    {
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            this->write_sampled_solution(lev, n, t_);
        }
    }

    void export_mesh(const int n, const std::string & filename_root)
    {
        this->export_mesh_to_VTK(n, filename_root, this->ibvp);
    }

    void export_quadrature_points(const int n, const std::string & filename_root)
    {
        this->export_quadrature_points_to_VTK(n, filename_root, this->ibvp);
    }

    void export_solution(const int n, const std::string & filename_root, const amrex::Real t_)
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
                                                                                t_, this->geom[lev], mesh, solution, this->X[lev], this->L[lev], mask, this->ibvp);
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

    void write_checkpoint(const int n, const amrex::Real t_)
    {
        if (this->meshes[0]->uses_projected_level_set())
        {
            this->write_checkpoint_using_projected_level_set(n, t_, "checkpoint", this->X, this->L);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SinglePhaseDomain.H - AMR::write_checkpoint\n";
            msg += "| Unexpected method used to define the embedded geometry.\n";
            msg += "| Unexpected method: "+this->meshes[0]->params.embedded_geometry_defined_by+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================
};
// ####################################################################

} // namespace single_phase_domain