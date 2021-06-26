//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_TimeModulatedComposite.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_ElasticSolid.H"

#define MAX_N_PHASES 2

namespace space_time_modulated_composite
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
    amrex::Real eval(const amrex::Real * /*x*/) const
    {
        const amrex::Real res = -1.0;
        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        const T res = (x[0]-x[0]-1.0);
        
        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
        const amrex::GpuArray<T, N> res = {AMREX_D_DECL(x[0]-x[0], x[0]-x[0], x[0]-x[0])};

        return res;
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
    const int domain_type_homogeneous = 0;
    const int domain_type_time_laminate = 1;
    const int domain_type_space_laminate = 2;
    const int domain_type_space_time_laminate = 3;

    int domain_type_flag;
    amrex::Real density[MAX_N_PHASES], c[MAX_N_PHASES][NS*(NS+1)/2], max_wave_speed;

    amrex::Real space_box, time_box, space_eps, time_eps;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        amrex::ParmParse pp;
        std::string domain_type;
        int n_phases;

        // READ DOMAIN TYPE -------------------------------------------
        {
            pp.get("domain_type", domain_type);

            if (domain_type.compare("homogeneous") == 0)
            {
                this->domain_type_flag = this->domain_type_homogeneous;
            }
            else if (domain_type.compare("time_laminate") == 0)
            {
                this->domain_type_flag = this->domain_type_time_laminate;
            }
            else if (domain_type.compare("space_laminate") == 0)
            {
                this->domain_type_flag = this->domain_type_space_laminate;
            }
            else if (domain_type.compare("space_time_laminate") == 0)
            {
                this->domain_type_flag = this->domain_type_space_time_laminate;
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedComposites.H - IBVP::IBVP\n";
                msg += "| Unexpected domain type: "+domain_type+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------


        // READ NUMBER OF PHASES --------------------------------------
        {
            pp.get("n_phases", n_phases);
        }
        // ------------------------------------------------------------

        // READ MATERIAL PROPERTIES -----------------------------------
        for (int a = 0; a < n_phases; ++a)
        {
            amrex::ParmParse phase_pp("phase["+std::to_string(a)+"]");
            std::string material_type;
            amrex::Vector<amrex::Real> material_constants;
            amrex::Real cSE[NS*NS];

            phase_pp.get("material_type", material_type);

            if (material_type.compare("isotropic") == 0)
            {
#if (AMREX_SPACEDIM > 1)
                amrex::Real E, nu;
#endif

                phase_pp.getarr("material_constants", material_constants);

                this->density[a] = material_constants[0];
#if (AMREX_SPACEDIM > 1)
                E = material_constants[1];
                nu = material_constants[2];
#endif
                
#if (AMREX_SPACEDIM == 1)
                cSE[0] = material_constants[1];
#endif
#if (AMREX_SPACEDIM == 2)
                linear_elasticity::get_stiffness_matrix_2d_isotropic(E, nu, cSE);
#endif
#if (AMREX_SPACEDIM == 3)
                linear_elasticity::get_stiffness_matrix_3d_isotropic(E, nu, cSE);
#endif
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedComposites.H - IBVP::IBVP\n";
                msg += "| Unexpected input material type: "+material_type+" for phase: "+std::to_string(a)+".\n";
                amrex::Abort(msg);
            }

            // STORE STIFFNESS CONSTANTS
#if (AMREX_SPACEDIM == 1)
            this->c[a][C11] = cSE[0];
#endif
#if (AMREX_SPACEDIM == 2)
            this->c[a][C11] = cSE[0+0*NS];
            this->c[a][C21] = cSE[1+0*NS]; this->c[a][C22] = cSE[1+1*NS];
            this->c[a][C61] = cSE[2+0*NS]; this->c[a][C62] = cSE[2+1*NS]; this->c[a][C66] = cSE[2+2*NS];
#endif
#if (AMREX_SPACEDIM == 3)
            this->c[a][C11] = cSE[0+0*NS];
            this->c[a][C21] = cSE[1+0*NS]; this->c[a][C22] = cSE[1+1*NS];
            this->c[a][C31] = cSE[2+0*NS]; this->c[a][C32] = cSE[2+1*NS]; this->c[a][C33] = cSE[2+2*NS];
            this->c[a][C41] = cSE[3+0*NS]; this->c[a][C42] = cSE[3+1*NS]; this->c[a][C43] = cSE[3+2*NS]; this->c[a][C44] = cSE[3+3*NS];
            this->c[a][C51] = cSE[4+0*NS]; this->c[a][C52] = cSE[4+1*NS]; this->c[a][C53] = cSE[4+2*NS]; this->c[a][C54] = cSE[4+3*NS]; this->c[a][C55] = cSE[4+4*NS];
            this->c[a][C61] = cSE[5+0*NS]; this->c[a][C62] = cSE[5+1*NS]; this->c[a][C63] = cSE[5+2*NS]; this->c[a][C64] = cSE[5+3*NS]; this->c[a][C65] = cSE[5+4*NS]; this->c[a][C66] = cSE[5+5*NS];
#endif

            /* DEBUG */
            amrex::Print() << "Phase " << a << " - cSE: " << std::endl;
            amrex::dG::io::print_real_array_2d(NS, NS, cSE);
            /**/
        }
        // ------------------------------------------------------------

        // MAX WAVE SPEED ---------------------------------------------
        this->max_wave_speed = 0.0;

        for (int a = 0; a < n_phases; ++a)
        {
#if (AMREX_SPACEDIM == 1)
            this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt(this->c[a][C11]/this->density[a]));

#else
            amrex::ParmParse phase_pp("phase["+std::to_string(a)+"]");
            std::string material_type;

            phase_pp.get("material_type", material_type);

            if (material_type.compare("isotropic") == 0)
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
#endif
        }
        
        amrex::Print() << "Max wave speed: " << this->max_wave_speed << std::endl;
        // ------------------------------------------------------------

        // SPACE AND TIME BOXES ---------------------------------------
        if (domain_type.compare("homogeneous") == 0)
        {
            this->space_box = 0.0;
            this->time_box  = 0.0;

            this->space_eps = 0.0;
            this->time_eps = 0.0;
        }
        else if (domain_type.compare("time_laminate") == 0)
        {
            this->space_box = 0.0;
            pp.get("time_box", this->time_box);

            this->space_eps = 0.0;
            this->time_eps = 1.0e-5*this->time_box;
        }
        else if (domain_type.compare("space_laminate") == 0)
        {
            pp.get("space_box", this->space_box);
            this->time_box  = 0.0;

            this->space_eps = 1.0e-5*this->space_box;
            this->time_eps = 0.0;
        }
        else if (domain_type.compare("space_time_laminate") == 0)
        {
            pp.get("space_box", this->space_box);
            pp.get("time_box", this->time_box);

            this->space_eps = 1.0e-5*this->space_box;
            this->time_eps = 1.0e-5*this->time_box;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SpaceTimeModulatedComposites.H - IBVP::IBVP\n";
            msg += "| Unexpected domain type: "+domain_type+".\n";
            amrex::Abort(msg);
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
        return N_VS+AMREX_SPACEDIM;
    }
    /**
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int /*dom*/, int & u_lo, int & u_hi) const
    {
        u_lo = 0;
        u_hi = N_VS+AMREX_SPACEDIM;
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
        U[V1] = 0.0;
        U[E11] = 0.0;
        U[E11+1] = 0.0;
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
        const amrex::Real x0 = 1.425;
        const amrex::Real s2 = 10000.0;
        const amrex::Real f = std::exp(-s2*(x[0]-x0)*(x[0]-x0));
        const amrex::Real df = -2.0*s2*(x[0]-x0)*std::exp(-s2*(x[0]-x0)*(x[0]-x0));

        U[V1] = 0.0;
        U[E11] = df;
        U[E11+1] = f;
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
        err = 1.0;
        err_norm = 1.0;
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
     * \brief Eval the phase at (t, x).
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int eval_phase(const amrex::Real t_, const amrex::Real * x) const
    {
        const amrex::Real t = t_+this->time_eps;
        int a;

        if (this->domain_type_flag == this->domain_type_homogeneous)
        {
            a = 0;
        }
        else if (this->domain_type_flag == this->domain_type_space_laminate)
        {
            const amrex::Real sb = this->space_box;

#if (AMREX_SPACEDIM == 1)

            const int ix = (int) std::floor(x[0]/sb);

            //amrex::Print() << "sb: " << sb << std::endl;
            //amrex::Print() << "x[0]: " << x[0] << std::endl;
            //amrex::Print() << "ix: " << ix << std::endl;

            a = ix%2;
#else
amrex::Print() << "eval_phase" << std::endl;
exit(-1);
#endif
        }
        else if (this->domain_type_flag == this->domain_type_time_laminate)
        {
            const amrex::Real tb = this->time_box;

#if (AMREX_SPACEDIM == 1)

            const int it = (int) std::floor(t/tb);

            //amrex::Print() << "tb: " << tb << std::endl;
            //amrex::Print() << "t: " << t << std::endl;
            //amrex::Print() << "it: " << it << std::endl;

            a = it%2;
#else
amrex::Print() << "eval_phase" << std::endl;
exit(-1);
#endif
        }
        else if (this->domain_type_flag == this->domain_type_space_time_laminate)
        {
            const amrex::Real sb = this->space_box;
            const amrex::Real tb = this->time_box;

#if (AMREX_SPACEDIM == 1)

            const int ix = (int) std::floor(x[0]/sb);
            const int it = (int) std::floor(t/tb);

            //amrex::Print() << "sb: " << sb << std::endl;
            //amrex::Print() << "x[0]: " << x[0] << std::endl;
            //amrex::Print() << "ix: " << ix << std::endl;
            //amrex::Print() << "tb: " << tb << std::endl;
            //amrex::Print() << "t: " << t << std::endl;
            //amrex::Print() << "it: " << it << std::endl;

            a = ((ix%2)^(it%2));
#else
amrex::Print() << "eval_phase" << std::endl;
exit(-1);
#endif
        }
        else
        {

amrex::Print() << "eval_phase" << std::endl;
exit(-1);
        }

        return a;
    }

    /**
     * \brief Eval the phase at the boundary at (t, x) with unit normal un.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_phase(const amrex::Real t, const amrex::Real * x, const amrex::Real * un, int & m_a) const
    {
        const amrex::Real dx[AMREX_SPACEDIM] = {AMREX_D_DECL(this->space_eps*un[0], this->space_eps*un[1], this->space_eps*un[2])};
        const amrex::Real m_x[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-dx[0], x[1]-dx[1], x[2]-dx[2])};

        m_a = this->eval_phase(t, m_x);
    }
    
    /**
     * \brief Eval the phases at the interface at (t, x) with unit normal un.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_phase(const amrex::Real t, const amrex::Real * x, const amrex::Real * un, int & m_a, int & p_a) const
    {
        const amrex::Real dx[AMREX_SPACEDIM] = {AMREX_D_DECL(this->space_eps*un[0], this->space_eps*un[1], this->space_eps*un[2])};
        const amrex::Real m_x[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-dx[0], x[1]-dx[1], x[2]-dx[2])};
        const amrex::Real p_x[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]+dx[0], x[1]+dx[1], x[2]+dx[2])};

        m_a = this->eval_phase(t, m_x);
        p_a = this->eval_phase(t, p_x);
    }
    
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
    void eval_FR(const int /*dom*/,
                 const amrex::Real t, const amrex::Real * x,
                 const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3), amrex::Real * R) const
    {
        const int a = this->eval_phase(t, x);
#if (AMREX_SPACEDIM == 1)
        elastic_solid::eval_F_compact_c(this->density[a], this->c[a], U, AMREX_D_DECL(F1, F2, F3));
        
        R[V1] = 0.0;
        R[E11] = 0.0;
        R[E11+1] = -U[V1]/this->density[a];
#else
amrex::Print() << "eval_FR" << std::endl;
exit(-1);
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
                             const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn) const
    {
        int m_a, p_a;
        this->eval_phase(t, x, un, m_a, p_a);
        
        elastic_solid::eval_NFn_Riemann_solver(this->density[m_a], this->c[m_a], this->density[p_a], this->c[p_a], un, m_U, p_U, NFn);
        NFn[E11+1] = 0.0;
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
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        const amrex::Real T[AMREX_SPACEDIM] = {0.0};
        int a;
        
        this->eval_phase(t, x, un, a);

        elastic_solid::eval_NFn_Riemann_solver_prescribed_tractions(this->density[a], this->c[a], un, U, T, NFn);
        NFn[E11+1] = 0.0;
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
        return N_VS+AMREX_SPACEDIM+2;
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
#if (AMREX_SPACEDIM == 1)
        output_fields_names = {"mx", "gxx", "ux", "density", "stiffness"};
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
        int a;
        
        a = this->eval_phase(t, x);

#if (AMREX_SPACEDIM == 1)
        F[V1] = U[V1];
        F[E11] = U[E11];
        F[E11+1] = U[E11+1];
        F[E11+2] = this->density[a];
        F[E11+3] = this->c[a][C11];
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
                            const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * /*U_1*/, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                            amrex::Real * /*F*/) const
    {
amrex::Print() << "eval_output_fields" << std::endl;
exit(-1);
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


    // CHECK SPACE-TIME BOXES =========================================
    void check_space_time_boxes() const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real T = this->inputs.time.T;
        const amrex::Box domain = this->Geom(0).Domain();
        const amrex::IntVect n_cells = domain.bigEnd()-domain.smallEnd()+1;
        const amrex::Real * prob_lo = this->Geom(0).ProbLo();
        const amrex::Real * prob_hi = this->Geom(0).ProbHi();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->Geom(0).CellSizeArray();
        const amrex::Real sb = this->ibvp.space_box;
        const amrex::Real tb = this->ibvp.time_box;
        // ------------------------------------------------------------

        if (this->ibvp.domain_type_flag == this->ibvp.domain_type_homogeneous)
        {
            // All good.
        }
        
        if ((this->ibvp.domain_type_flag == this->ibvp.domain_type_space_laminate) || 
            (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_time_laminate))
        {

            const amrex::Real n_cells_per_sb[AMREX_SPACEDIM] = {AMREX_D_DECL(sb/dx[0], sb/dx[1], sb/dx[2])};

#if (AMREX_SPACEDIM == 1)
            //amrex::Print() << "n_cells_per_sb: " << n_cells_per_sb[0] << std::endl;
            if (std::abs(n_cells_per_sb[0]-std::round(n_cells_per_sb[0])) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedComposites.H - AMR::check_space_time_boxes\n";
                msg += "| The discretization is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[0])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[0])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[0])+"\n";
                msg += "|      dx:"+std::to_string(dx[0])+"\n";
                msg += "| space_box:"+std::to_string(sb)+"\n";
                msg += "| space_box/dx:"+std::to_string(sb/dx[0])+"\n";
                msg += "| The ratio between space_box and dx must be an integer.\n";
                amrex::Abort(msg);
            }
#else
amrex::Print() << "check_space_boxes" << std::endl;
exit(-1);
#endif
        }
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
            msg +=  "ERROR: IBVP_TimeModulatedComposite.H - AMR::MakeNewLevelFromScratch\n";
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
            msg +=  "ERROR: IBVP_TimeModulatedComposite.H - AMR::eval_error\n";
            msg += "| Unexpected error norm.\n";
            msg += "| requested norm           : "+this->norm+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t) const
    {
        amrex::Real dt;
        
        dt = amrex::dG::hyperbolic_pde::eval_dt(*this, t, this->X, this->ibvp);


        if ((this->ibvp.domain_type_flag == this->ibvp.domain_type_time_laminate) || 
            (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_time_laminate))
        {
            const amrex::Real tb = this->ibvp.time_box;
            const int it = (int) std::floor(t/tb);
            const amrex::Real next_time_interface = (it+1)*tb;

            if ((t < next_time_interface) && ((t+dt) > next_time_interface))
            {
                
                //amrex::Print() << "t: " << t << std::endl;
                //amrex::Print() << "dt: " << dt << std::endl;
                //amrex::Print() << "tb: " << tb << std::endl;
                //amrex::Print() << "it: " << it << std::endl;
                //amrex::Print() << "next_time_interface: " << next_time_interface << std::endl;

                dt = amrex::min(t+dt, next_time_interface)-t;
            }
        }

        return dt;
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
                msg +=  "ERROR: IBVP_TimeModulatedComposite.H - AMR::export_solution\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
        }
    }
    // ================================================================
};
// ####################################################################

} // namespace space_time_modulated_composite