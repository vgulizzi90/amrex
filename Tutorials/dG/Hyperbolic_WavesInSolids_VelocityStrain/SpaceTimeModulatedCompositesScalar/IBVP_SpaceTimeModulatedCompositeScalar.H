//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SpaceTimeModulatedCompositeScalar.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define N_U (1+AMREX_SPACEDIM+1)

// SOLUTION FIELDS ORDER
#if (AMREX_SPACEDIM == 1)
#define MM 0
#define E11 1
#define UU 2
#endif
#if (AMREX_SPACEDIM == 2)
#define MM 0
#define E11 1
#define E22 2
#define UU 3
#endif
#if (AMREX_SPACEDIM == 3)
#define MM 0
#define E11 1
#define E22 2
#define E33 3
#define UU 4
#endif
// ####################################################################

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
    const int domain_type_space_laminate_x = 20;
    const int domain_type_space_laminate_y = 21;
    const int domain_type_space_laminate_z = 22;
    const int domain_type_space_time_checkerboard = 3;
    const int domain_type_space_checkerboard = 4;

    int domain_type_flag;
    amrex::Real density[MAX_N_PHASES], stiffness[MAX_N_PHASES], max_wave_speed;

    amrex::Real space_box[AMREX_SPACEDIM], time_box, space_eps, time_eps;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP(const amrex::Real * dx)
    {
        amrex::ParmParse pp;
        std::string domain_type;
        int n_phases;

        // READ DOMAIN TYPE -------------------------------------------
        {
            pp.get("domain_type", domain_type);

            // HOMOGENEOUS DOMAIN
            if (domain_type.compare("homogeneous") == 0)
            {
                this->domain_type_flag = this->domain_type_homogeneous;
            }
            // TIME LAMINATE
            else if (domain_type.compare("time_laminate") == 0)
            {
                this->domain_type_flag = this->domain_type_time_laminate;
            }
            // SPACE LAMINATE
            else if (domain_type.compare("space_laminate") == 0)
            {
                this->domain_type_flag = this->domain_type_space_laminate;
            }
            // SPACE LAMINATE X
            else if (domain_type.compare("space_laminate_x") == 0)
            {
                this->domain_type_flag = this->domain_type_space_laminate_x;
            }
            // SPACE LAMINATE Y
            else if (domain_type.compare("space_laminate_y") == 0)
            {
                this->domain_type_flag = this->domain_type_space_laminate_y;
            }
            // SPACE LAMINATE Z
            else if (domain_type.compare("space_laminate_z") == 0)
            {
                this->domain_type_flag = this->domain_type_space_laminate_z;
            }
            // SPACE-TIME CHECKERBOARD
            else if (domain_type.compare("space_time_checkerboard") == 0)
            {
                this->domain_type_flag = this->domain_type_space_time_checkerboard;
            }
            // SPACE CHECKERBOARD
            else if (domain_type.compare("space_checkerboard") == 0)
            {
                this->domain_type_flag = this->domain_type_space_checkerboard;
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - IBVP::IBVP\n";
                msg += "| Unexpected domain type: "+domain_type+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------


        // READ NUMBER OF PHASES --------------------------------------
        {
            pp.get("n_phases", n_phases);

            if (n_phases > MAX_N_PHASES)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - IBVP::IBVP\n";
                msg += "| MAX_N_PHASES should be increased to at least: "+std::to_string(n_phases)+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------

        // READ MATERIAL PROPERTIES -----------------------------------
        for (int a = 0; a < n_phases; ++a)
        {
            amrex::ParmParse phase_pp("phase["+std::to_string(a)+"]");
            std::string material_type;
            amrex::Vector<amrex::Real> material_constants;

            phase_pp.get("material_type", material_type);

            if (material_type.compare("isotropic") == 0)
            {
                phase_pp.getarr("material_constants", material_constants);

                this->density[a] = material_constants[0];
                this->stiffness[a] = material_constants[1];
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - IBVP::IBVP\n";
                msg += "| Unexpected input material type: "+material_type+" for phase: "+std::to_string(a)+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------

        // MAX WAVE SPEED ---------------------------------------------
        this->max_wave_speed = 0.0;

        for (int a = 0; a < n_phases; ++a)
        {
            this->max_wave_speed = amrex::max(this->max_wave_speed, std::sqrt(this->stiffness[a]/this->density[a]));
        }
        
        amrex::Print() << "Max wave speed: " << this->max_wave_speed << std::endl;
        // ------------------------------------------------------------

        // SPACE AND TIME BOXES ---------------------------------------
        const amrex::Real max_real = std::numeric_limits<amrex::Real>::max();

        // HOMOGENEOUS DOMAIN
        if (domain_type.compare("homogeneous") == 0)
        {
            AMREX_D_TERM
            (
                this->space_box[0] = max_real;,
                this->space_box[1] = max_real;,
                this->space_box[2] = max_real;
            )
            this->time_box = max_real;

            this->space_eps = 0.0;
            this->time_eps = 0.0;
        }
        // TIME LAMINATE
        else if (domain_type.compare("time_laminate") == 0)
        {
            AMREX_D_TERM
            (
                this->space_box[0] = max_real;,
                this->space_box[1] = max_real;,
                this->space_box[2] = max_real;
            )
            pp.get("time_box", this->time_box);

            this->space_eps = 0.0;
            this->time_eps = 1.0e-5*this->time_box;
        }
        // SPACE LAMINATE
        else if (domain_type.compare("space_laminate") == 0)
        {
            pp.get("space_box", this->space_box[0]);
            this->time_box = max_real;

            this->space_eps = 1.0e-5*dx[0];
            this->time_eps = 0.0;

#if (AMREX_SPACEDIM != 1)
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - IBVP::IBVP\n";
            msg += "| The space_laminate is supposed to work with AMREX_SPACEDIM = 1.\n";
            amrex::Abort(msg);
#endif
        }
        // SPACE LAMINATE X
        else if (domain_type.compare("space_laminate_x") == 0)
        {
            amrex::Real sb;
            pp.get("space_box", sb);
            AMREX_D_TERM
            (
                this->space_box[0] = sb;,
                this->space_box[1] = max_real;,
                this->space_box[2] = max_real;
            )
            this->time_box = max_real;

            this->space_eps = 1.0e-5*dx[0];
            this->time_eps = 0.0;
        }
        // SPACE-TIME CHECKERBOARD
        else if (domain_type.compare("space_time_checkerboard") == 0)
        {
            amrex::Vector<amrex::Real> sb;
            pp.getarr("space_box", sb);
            AMREX_D_TERM
            (
                this->space_box[0] = sb[0];,
                this->space_box[1] = sb[1];,
                this->space_box[2] = sb[2];
            )
            pp.get("time_box", this->time_box);

#if (AMREX_SPACEDIM == 1)
            this->space_eps = 1.0e-5*dx[0];
#else
            this->space_eps = 1.0e-5*amrex::min(AMREX_D_DECL(dx[0], dx[1], dx[2]));
#endif
            this->time_eps = 1.0e-5*this->time_box;
        }
        // SPACE CHECKERBOARD
        else if (domain_type.compare("space_checkerboard") == 0)
        {
            amrex::Vector<amrex::Real> sb;
            pp.getarr("space_box", sb);
            AMREX_D_TERM
            (
                this->space_box[0] = sb[0];,
                this->space_box[1] = sb[1];,
                this->space_box[2] = sb[2];
            )

            this->time_box = max_real;

            this->space_eps = 1.0e-5*dx[0];
            this->time_eps = 0.0;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - IBVP::IBVP\n";
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
        return N_U;
    }
    /**
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int /*dom*/, int & u_lo, int & u_hi) const
    {
        u_lo = 0;
        u_hi = N_U;
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
    void eval_exact_solution(const amrex::Real /*t*/, const amrex::Real * /*x*/, amrex::Real * /*U*/) const
    {
amrex::Print() << "eval_exact_solution" << std::endl;
exit(-1);
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
#if (AMREX_SPACEDIM == 1)
        const amrex::Real x0 = 0.75;
        const amrex::Real s2 = 10.0*10.0;
        const amrex::Real f = std::exp(-s2*(x[0]-x0)*(x[0]-x0));
        const amrex::Real dfdx = -2.0*s2*(x[0]-x0)*std::exp(-s2*(x[0]-x0)*(x[0]-x0));
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real x0 = 1.5;
        const amrex::Real x1 = 1.5;
        const amrex::Real s2 = 10.0*10.0;
        const amrex::Real f = std::exp(-s2*(x[0]-x0)*(x[0]-x0)-s2*(x[1]-x1)*(x[1]-x1));
        const amrex::Real dfdx = -2.0*s2*(x[0]-x0)*std::exp(-s2*(x[0]-x0)*(x[0]-x0)-s2*(x[1]-x1)*(x[1]-x1));
        const amrex::Real dfdy = -2.0*s2*(x[1]-x1)*std::exp(-s2*(x[0]-x0)*(x[0]-x0)-s2*(x[1]-x1)*(x[1]-x1));
#endif

        U[MM] = 0.0;
        AMREX_D_TERM
        (
            U[E11] = dfdx;,
            U[E22] = dfdy;,
            U[E33] = dfdz;
        )
        U[UU] = f;
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
                          const amrex::Real /*t*/, const amrex::Real * /*x*/,
                          const amrex::Real * /*U*/,
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

        // HOMOGENEOUS DOMAIN
        if (this->domain_type_flag == this->domain_type_homogeneous)
        {
            a = 0;
        }
        // TIME LAMINATE
        else if (this->domain_type_flag == this->domain_type_time_laminate)
        {
            const amrex::Real tb = this->time_box;
            
            const int it = (int) std::floor(t/tb);

            //amrex::Print() << "tb: " << tb << std::endl;
            //amrex::Print() << "t: " << t << std::endl;
            //amrex::Print() << "it: " << it << std::endl;

            a = it%2;
        }
        // SPACE LAMINATE
        else if (this->domain_type_flag == this->domain_type_space_laminate)
        {
            const amrex::Real * sb = this->space_box;
            
            const int ix = (int) std::floor(x[0]/sb[0]);

            //amrex::Print() << "sb: " << sb << std::endl;
            //amrex::Print() << "x[0]: " << x[0] << std::endl;
            //amrex::Print() << "ix: " << ix << std::endl;

            a = ix%2;
        }
        // SPACE LAMINATE X
        else if (this->domain_type_flag == this->domain_type_space_laminate_x)
        {
            const amrex::Real * sb = this->space_box;
            
            const int ix = (int) std::floor(x[0]/sb[0]);

            //amrex::Print() << "sb: " << sb << std::endl;
            //amrex::Print() << "x[0]: " << x[0] << std::endl;
            //amrex::Print() << "ix: " << ix << std::endl;

            a = abs(ix%2);
        }
        // SPACE-TIME CHECKERBOARD
        else if (this->domain_type_flag == this->domain_type_space_time_checkerboard)
        {
            const amrex::Real * sb = this->space_box;
            const amrex::Real tb = this->time_box;

#if (AMREX_SPACEDIM == 1)

            const int ix = (int) std::floor(x[0]/sb[0]);
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
        // SPACE CHECKERBOARD
        else if (this->domain_type_flag == this->domain_type_space_checkerboard)
        {
            const amrex::Real * sb = this->space_box;

#if (AMREX_SPACEDIM == 2)

            const int ix = (int) std::floor(x[0]/sb[0]);
            const int iy = (int) std::floor(x[1]/sb[1]);

            //amrex::Print() << "sb: " << sb << std::endl;
            //amrex::Print() << "x[0]: " << x[0] << std::endl;
            //amrex::Print() << "ix: " << ix << std::endl;
            //amrex::Print() << "tb: " << tb << std::endl;
            //amrex::Print() << "t: " << t << std::endl;
            //amrex::Print() << "it: " << it << std::endl;

            a = ((ix%2)^(iy%2));
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
        const amrex::Real rho = this->density[a];
        const amrex::Real E = this->stiffness[a];

#if (AMREX_SPACEDIM == 1)
        F1[MM] = -E*U[E11];
        F1[E11] = -U[MM]/rho;
        F1[UU] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
        F1[MM] = -E*U[E11];
        F1[E11] = -U[MM]/rho;
        F1[E22] = 0.0;
        F1[UU] = 0.0;

        F2[MM] = -E*U[E22];
        F2[E11] = 0.0;
        F2[E22] = -U[MM]/rho;
        F2[UU] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        F1[MM] = -E*U[E11];
        F1[E11] = -U[MM]/rho;
        F1[E22] = 0.0;
        F1[E33] = 0.0;
        F1[UU] = 0.0;

        F2[MM] = -E*U[E22];
        F2[E11] = 0.0;
        F2[E22] = -U[MM]/rho;
        F2[E33] = 0.0;
        F2[UU] = 0.0;

        F3[MM] = -E*U[E33];
        F3[E11] = 0.0;
        F3[E22] = 0.0;
        F3[E33] = -U[MM]/rho;
        F3[UU] = 0.0;
#endif

        R[MM] = 0.0;
        AMREX_D_TERM
        (
            R[E11] = 0.0;,
            R[E22] = 0.0;,
            R[E33] = 0.0;
        )
        R[UU] = -U[MM]/rho;
    }
    // ================================================================


    // DG =============================================================
    /**
    * \brief Fill memory pointed to by AnU with the product of the hyperbolic operator An and U.
    *
    * \param[in] rho: density.
    * \param[in] E: stiffness.
    * \param[in] U: pointer to memory containing the conserved variables.
    * \param[out] AnU: pointer to memory that will contain AnU.
    *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_AnU(const amrex::Real rho, const amrex::Real E, const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU) const
    {
        // PARAMETERS
        const amrex::Real inv_rho = 1.0/rho;
        AMREX_D_TERM
        (
            const amrex::Real n1 = un[0];,
            const amrex::Real n2 = un[1];,
            const amrex::Real n3 = un[2];
        )

#if (AMREX_SPACEDIM == 1)
        AnU[MM] = -n1*E*U[E11];
        AnU[E11] = -n1*inv_rho*U[MM];
        AnU[UU] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
        AnU[MM] = -n1*E*U[E11]-n2*E*U[E22];
        AnU[E11] = -n1*inv_rho*U[MM];
        AnU[E22] = -n2*inv_rho*U[MM];
        AnU[UU] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        AnU[MM] = -n1*E*U[E11]-n2*E*U[E22]-n3*E*U[E33];
        AnU[E11] = -n1*inv_rho*U[MM];
        AnU[E22] = -n2*inv_rho*U[MM];
        AnU[E33] = -n3*inv_rho*U[MM];
        AnU[UU] = 0.0;
#endif
    }

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
        const amrex::Real m_rho = this->density[m_a];
        const amrex::Real m_E = this->stiffness[m_a];
        const amrex::Real p_rho = this->density[p_a];
        const amrex::Real p_E = this->stiffness[p_a];
        const amrex::Real rho = 2.0/(1.0/m_rho+1.0/p_rho);
        const amrex::Real E = 0.5*(m_E+p_E);
        const amrex::Real w_max = this->max_wave_speed;
        //const amrex::Real w_max = 0.5*(std::sqrt(m_E/m_rho)+std::sqrt(p_E/p_rho));

        amrex::Real m_AnU[N_U], p_AnU[N_U];

        this->eval_AnU(rho, E, un, m_U, m_AnU);
        this->eval_AnU(rho, E, un, p_U, p_AnU);

        for (int u = 0; u < (1+AMREX_SPACEDIM); ++u)
        {
            NFn[u] = 0.5*(m_AnU[u]+p_AnU[u])+0.5*w_max*(m_U[u]-p_U[u]);
        }
        NFn[UU] = 0.0;
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
        int a;
        amrex::Real U_ghost[N_U];
        amrex::Real AnU[N_U], AnU_ghost[N_U];
        
        this->eval_phase(t, x, un, a);
        const amrex::Real rho = this->density[a];
        const amrex::Real E = this->stiffness[a];
        const amrex::Real w_max = this->max_wave_speed;

        U_ghost[MM] = -U[MM];
        AMREX_D_TERM
        (
            U_ghost[E11] = U[E11];,
            U_ghost[E22] = U[E22];,
            U_ghost[E33] = U[E33];
        )
        U_ghost[UU] = U[UU];

        this->eval_AnU(rho, E, un, U, AnU);
        this->eval_AnU(rho, E, un, U_ghost, AnU_ghost);

        for (int u = 0; u < (1+AMREX_SPACEDIM); ++u)
        {
            NFn[u] = 0.5*(AnU[u]+AnU_ghost[u])+0.5*w_max*(U[u]-U_ghost[u]);
        }
        NFn[UU] = 0.0;
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
        return N_U+2;
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
        output_fields_names = {"m", AMREX_D_DECL("gx", "gy", "gz"), "u", "density", "stiffness"};
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

        F[MM] = U[MM];
        AMREX_D_TERM
        (
            F[E11] = U[E11];,
            F[E22] = U[E22];,
            F[E33] = U[E33];
        )
        F[UU] = U[UU];
        F[UU+1] = this->density[a];
        F[UU+2] = this->stiffness[a];
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
    amrex::dG::amr::SinglePatch::SinglePatch(),
    ibvp(this->geom[0].CellSize())
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
    void check_space_boxes() const
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Box domain = this->Geom(0).Domain();
        const amrex::IntVect n_cells = domain.bigEnd()-domain.smallEnd()+1;
        const amrex::Real * prob_lo = this->Geom(0).ProbLo();
        const amrex::Real * prob_hi = this->Geom(0).ProbHi();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->Geom(0).CellSizeArray();
        const amrex::Real * sb = this->ibvp.space_box;
        // ------------------------------------------------------------

        // HOMOGENEOUS DOMAIN
        if (this->ibvp.domain_type_flag == this->ibvp.domain_type_homogeneous)
        {
            // All good.
        }
        // SPACE LAMINATE
        else if (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_laminate)
        {
            const amrex::Real n_cells_per_sb = sb[0]/dx[0];

            if (std::abs(n_cells_per_sb-std::round(n_cells_per_sb)) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - AMR::check_space_boxes\n";
                msg += "| The discretization is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[0])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[0])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[0])+"\n";
                msg += "|      dx:"+std::to_string(dx[0])+"\n";
                msg += "| space_box:"+std::to_string(sb[0])+"\n";
                msg += "| space_box/dx:"+std::to_string(sb[0]/dx[0])+"\n";
                msg += "| The ratio between space_box and dx must be an integer.\n";
                amrex::Abort(msg);
            }
        }
        // SPACE LAMINATE X
        else if (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_laminate_x)
        {
            const amrex::Real n_cells_per_sb = sb[0]/dx[0];

            if (std::abs(n_cells_per_sb-std::round(n_cells_per_sb)) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - AMR::check_space_boxes\n";
                msg += "| The discretization is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[0])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[0])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[0])+"\n";
                msg += "|      dx:"+std::to_string(dx[0])+"\n";
                msg += "| space_box:"+std::to_string(sb[0])+"\n";
                msg += "| space_box/dx:"+std::to_string(sb[0]/dx[0])+"\n";
                msg += "| The ratio between space_box and dx must be an integer.\n";
                amrex::Abort(msg);
            }
        }
        // SPACE-TIME CHECKERBOARD
        else if (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_time_checkerboard)
        {
            const amrex::Real n_cells_per_sb[AMREX_SPACEDIM] = {AMREX_D_DECL(sb[0]/dx[0], sb[1]/dx[1], sb[2]/dx[2])};

            if (std::abs(n_cells_per_sb[0]-std::round(n_cells_per_sb[0])) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - AMR::check_space_boxes\n";
                msg += "| The discretization in the x direction is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[0])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[0])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[0])+"\n";
                msg += "|      dx:"+std::to_string(dx[0])+"\n";
                msg += "| space_box:"+std::to_string(sb[0])+"\n";
                msg += "| space_box/dx:"+std::to_string(sb[0]/dx[0])+"\n";
                msg += "| The ratio between space_box and dx must be an integer.\n";
                amrex::Abort(msg);
            }
#if (AMREX_SPACEDIM > 1)
            if (std::abs(n_cells_per_sb[1]-std::round(n_cells_per_sb[1])) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - AMR::check_space_boxes\n";
                msg += "| The discretization in the y direction is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[1])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[1])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[1])+"\n";
                msg += "|      dy:"+std::to_string(dx[1])+"\n";
                msg += "| space_box:"+std::to_string(sb[1])+"\n";
                msg += "| space_box/dy:"+std::to_string(sb[1]/dx[1])+"\n";
                msg += "| The ratio between space_box and dy must be an integer.\n";
                amrex::Abort(msg);
            }
#endif
#if (AMREX_SPACEDIM > 2)
            if (std::abs(n_cells_per_sb[2]-std::round(n_cells_per_sb[2])) > 1.0e-8)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SpaceTimeModulatedCompositeScalar.H - AMR::check_space_boxes\n";
                msg += "| The discretization in the z direction is incompatible with the space box size: \n";
                msg += "| prob_lo:"+std::to_string(prob_lo[2])+"\n";
                msg += "| prob_hi:"+std::to_string(prob_hi[2])+"\n";
                msg += "| n_cells:"+std::to_string(n_cells[2])+"\n";
                msg += "|      dz:"+std::to_string(dx[2])+"\n";
                msg += "| space_box:"+std::to_string(sb[2])+"\n";
                msg += "| space_box/dz:"+std::to_string(sb[2]/dx[2])+"\n";
                msg += "| The ratio between space_box and dz must be an integer.\n";
                amrex::Abort(msg);
            }
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
            (this->ibvp.domain_type_flag == this->ibvp.domain_type_space_time_checkerboard))
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