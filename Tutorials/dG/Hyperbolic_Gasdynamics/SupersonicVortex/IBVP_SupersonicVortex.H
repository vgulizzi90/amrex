//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SupersonicVortex.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_IdealGas.H"

namespace SupersonicVortex
{

// SUPERSONIC VORTEX: PROBLEM PARAMETERS ##############################
struct ProblemParameters
{
    // DATA MEMBERS ===================================================
    amrex::Real gamma;
    amrex::Real rho_inner, Mach_inner, r_inner, r_outer;
    std::string norm;
    // ================================================================

    // CONSTRUCTOR ====================================================
    ProblemParameters()
    {
        amrex::ParmParse pp;

        pp.get("gamma", this->gamma);
        
        pp.get("rho_inner", this->rho_inner);
        pp.get("Mach_inner", this->Mach_inner);
        pp.get("r_inner", this->r_inner);
        pp.get("r_outer", this->r_outer);
        pp.get("norm", this->norm);
    }
    // ================================================================
};
// ####################################################################



// SUPERSONIC VORTEX: IBVP ############################################
template <int N>
struct Levelset
{
    // DATA MEMBERS ===================================================
    amrex::Real ri, ro;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    Levelset(const amrex::Real r_inner, const amrex::Real r_outer)
    :
    ri{r_inner},
    ro{r_outer},
    dom{0}
    {}
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_time(const amrex::Real /*t*/)
    {
        // ...
    }
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // EVALUATION =====================================================
    amrex::Real eval(const amrex::Real * x) const
    {
        const amrex::Real ri2 = this->ri*this->ri;
        const amrex::Real ro2 = this->ro*this->ro;
        const amrex::Real r2 = x[0]*x[0]+x[1]*x[1];
        const amrex::Real res = (r2-ri2)*(r2-ro2);

        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real ri2 = this->ri*this->ri;
        const amrex::Real ro2 = this->ro*this->ro;
        const T r2 = x[0]*x[0]+x[1]*x[1];
        const T res = (r2-ri2)*(r2-ro2);

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real ri2 = this->ri*this->ri;
        const amrex::Real ro2 = this->ro*this->ro;
        const T r2 = x[0]*x[0]+x[1]*x[1];
        const T dr2dx = x[0]+x[0];
        const T dr2dy = x[1]+x[1];
        const T rr = (2.0*r2-ri2-ro2);
        const amrex::GpuArray<T, N> res = {rr*dr2dx, rr*dr2dy};

        return res;
    }
    // ================================================================
};

struct IBVP
{
    // DATA MEMBERS ===================================================
    ProblemParameters problem_params;
    Levelset<AMREX_SPACEDIM> level_set;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    :
    level_set(this->problem_params.r_inner, this->problem_params.r_outer)
    {}
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
        return IDEAL_GAS_NU;
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
    void eval_exact_solution(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->problem_params.gamma;
        const amrex::Real g2 = 0.5*(g-1.0);
        const amrex::Real ig = 1.0/(g-1.0);
        const amrex::Real rho_i = this->problem_params.rho_inner;
        const amrex::Real M_i = this->problem_params.Mach_inner;

        const amrex::Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]);
        const amrex::Real cth = x[0]/r;
        const amrex::Real sth = x[1]/r;
        const amrex::Real ri = this->problem_params.r_inner;
        const amrex::Real rr2 = (ri/r)*(ri/r);

        const amrex::Real rho = rho_i*std::pow(1.0+g2*M_i*M_i*(1.0-rr2), ig);
        const amrex::Real a_i = 1.0;
        const amrex::Real uth = a_i*M_i*(ri/r);
        const amrex::Real u = -uth*sth;
        const amrex::Real v = uth*cth;
        const amrex::Real p = std::pow(rho, g)/g;

        U[RHO] = rho;
        U[RHOV1] = rho*u;
        U[RHOV2] = rho*v;
        U[TE] = ig*p+0.5*rho*(u*u+v*v);

        /* Constant initial state
        U[RHO] = 1.0;
        U[RHOV1] = 0.0;
        U[RHOV2] = 0.0;
        U[TE] = ig;
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
    void eval_error_L_inf(const int /*dom*/,
                          const amrex::Real t, const amrex::Real * x,
                          const amrex::Real * U,
                          amrex::Real & err, amrex::Real & err_norm) const
    {
        amrex::Real U_exact[IDEAL_GAS_NU];

        this->eval_exact_solution(t, x, U_exact);

        err = std::abs(U[RHO]-U_exact[RHO]);
        err_norm = std::abs(U_exact[RHO]);
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
        const amrex::Real w = ideal_gas::eval_largest_wave_speed(this->problem_params.gamma, U);
#if (AMREX_SPACEDIM == 1)
        const amrex::Real h = dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real h = amrex::min(dx[0], dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real h = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
#endif
        return (h/w);
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
        ideal_gas::eval_F(this->problem_params.gamma, U, AMREX_D_DECL(F1, F2, F3));
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
        ideal_gas::eval_NFn_Riemann_solver(this->problem_params.gamma, un, m_U, p_U, NFn);
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
        // VARIABLES --------------------------------------------------
        amrex::Real U_exact[IDEAL_GAS_NU];
        // ------------------------------------------------------------

        // EXACT SOLUTION ---------------------------------------------
        this->eval_exact_solution(t, x, U_exact);
        // ------------------------------------------------------------

        // BOUNDARY FLUX ----------------------------------------------
        ideal_gas::eval_NFn_Riemann_solver(this->problem_params.gamma, un, U, U_exact, NFn);
        // ------------------------------------------------------------
    }

    /**
     * \brief Eval the numerical flux at the interfaces.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_ICs(const int /*dom*/,
                      const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        ideal_gas::eval_NFn_wall(this->problem_params.gamma, un, U, NFn);
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
        ideal_gas::eval_FV_slopes(this->problem_params.gamma, geom, mesh, solution, X, dX);
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
        return IDEAL_GAS_NU+4+1;
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
        output_fields_names = {"density",
                               "velocity_x", "velocity_y",
                               "pressure",
                               "mach",
                               "total_energy",
                               "internal_energy",
                               "density_error",
                               "level_set"};
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
        // PARAMETERS -------------------------------------------------
        const amrex::Real g = this->problem_params.gamma;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p, a, M;
        amrex::Real U_exact[IDEAL_GAS_NU], phi;
        // ------------------------------------------------------------

        // PRIMITIVE VARIABLES
        ideal_gas::eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);

        // SOUND SPEED AND MACH NUMBER
        a = std::sqrt(g*p/rho);
        M = std::sqrt(AMREX_D_TERM(v1*v1,+v2*v2,+v3*v3))/a;

        // EXACT SOLUTION
        this->eval_exact_solution(t, x, U_exact);

        // LEVEL SET
        phi = this->level_set.eval(x);

        // OUTPUT FIELDS
        F[RHO] = rho;
        AMREX_D_TERM
        (
            F[RHOV1] = v1;,
            F[RHOV2] = v2;,
            F[RHOV3] = v3;
        )
        F[TE] = p;
        F[TE+1] = M;
        F[TE+2] = U[TE];
        F[TE+3] = p/(rho*(g-1.0));
        F[TE+4] = rho-U_exact[RHO];
        F[TE+5] = phi;
    }
    // ================================================================
};
// ####################################################################



// SUPERSONIC VORTEX: AMR #############################################
struct AMR
:
public amrex::dG::amr::SinglePatch
{
    // DATA MEMBERS ===================================================
    // IBVP
    IBVP ibvp;

    // SOLUTION MULTIFABS
    amrex::Vector<amrex::MultiFab> X;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        this->X.resize(n_levels);
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

        // MESH -------------------------------------------------------
        this->meshes[lev]->make_from_scratch(time, this->Geom(lev), ba, dm, this->ibvp);
        // ------------------------------------------------------------

        // SOLUTION ---------------------------------------------------
        this->solutions[lev]->make_from_scratch(time, this->Geom(lev), *this->meshes[lev], this->X[lev], this->ibvp);
        this->solutions[lev]->eval_space_mass_matrix(time, this->Geom(lev), *this->meshes[lev], this->ibvp);
        this->solutions[lev]->project_initial_conditions(this->Geom(lev), *this->meshes[lev], this->X[lev], this->ibvp);

        if ((this->solutions[lev]->params.space_p == 0) && this->solutions[lev]->params.FV_is_active)
        {
            this->solutions[lev]->eval_space_centroids(this->Geom(lev), *this->meshes[lev], this->ibvp);
        }
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(this->meshes[lev]->ba, this->meshes[lev]->dm, 1, mask_n_grow);
        this->masks[lev] = 0;
        // ------------------------------------------------------------

        // SET BOX ARRAY AND DISTRIBUTION MAP -------------------------
        this->SetBoxArray(lev, this->meshes[lev]->ba);
        this->SetDistributionMap(lev, this->meshes[lev]->dm);
        // ------------------------------------------------------------
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    void eval_error(const amrex::Real t, amrex::Real & err, amrex::Real & err_norm) const
    {
        if (this->ibvp.problem_params.norm.compare("L_inf") == 0)
        {
            this->eval_error_L_inf(t, this->X, err, err_norm, this->ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SupersonicVortex.H - AMR::eval_error\n";
            msg += "| Unexpected error norm.\n";
            msg += "| requested norm           : "+this->ibvp.problem_params.norm+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t) const
    {
        return amrex::dG::hyperbolic_pde::amr_utils::eval_dt(*this, t, this->X, this->ibvp);
    }

    void take_time_step(const amrex::Real t, const amrex::Real dt)
    {
        amrex::dG::hyperbolic_pde::amr_utils::take_time_step(*this, t, dt, this->X, this->ibvp);
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
        this->export_solution_to_VTK(n, filename_root, t, this->X, this->ibvp);
    }
    // ================================================================
};
// ####################################################################

} // SupersonicVortex