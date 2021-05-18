//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SingleDomain.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_ElasticSolid.H"

namespace SingleDomain
{
// SINGLE DOMAIN: PROBLEM PARAMETERS ##################################
struct ProblemParameters
{
    // DATA MEMBERS ===================================================
    std::string shape;
    amrex::Vector<amrex::Real> geo_parameters;

    int n_domains;
    amrex::Vector<std::string> material_type;
    amrex::Vector<amrex::Vector<amrex::Real>> material_constants;
    
    std::string norm;
    // ================================================================

    // CONSTRUCTOR ====================================================
    ProblemParameters()
    {
        amrex::ParmParse pp;

        pp.get("shape", this->shape);
        pp.getarr("geo_parameters", this->geo_parameters);

        pp.get("n_domains", this->n_domains);
        this->material_type.resize(this->n_domains);
        this->material_constants.resize(this->n_domains);
        for (int dom = 0; dom < this->n_domains; ++dom)
        {
            amrex::ParmParse pp_dom("domain["+std::to_string(dom)+"]");
            pp_dom.get("material_type", this->material_type[dom]);
            pp_dom.getarr("material_constants", this->material_constants[dom]);
        }

        pp.get("norm", this->norm);
    }
    // ================================================================
};
// ####################################################################



// SINGLE DOMAIN: LEVEL SET ###########################################
template <int N>
struct LevelSet
{
    // DATA MEMBERS ===================================================
    int flag;
    amrex::Vector<amrex::Real> params;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    LevelSet(const std::string & shape, const amrex::Vector<amrex::Real> & geo_parameters)
    :
    flag{-1},
    dom{0}
    {
        if (shape.compare("none") == 0)
        {
            this->flag = -1;
            this->params.clear();
        }
        else if (shape.compare("circle") == 0)
        {
            this->flag = 0;
            this->params.resize(AMREX_SPACEDIM+1);
            AMREX_D_TERM
            (
                this->params[0] = geo_parameters[0];,
                this->params[1] = geo_parameters[1];,
                this->params[2] = geo_parameters[2];
            )
            this->params[AMREX_SPACEDIM] = geo_parameters[AMREX_SPACEDIM];
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_SingleDomain.H - LevelSet::LevelSet\n";
            msg += "| Unexpected shape for the level set.\n";
            msg += "| shape: "+shape+".\n";
            amrex::Abort(msg);
        }
    }
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
        amrex::Real res;

        switch(this->flag)
        {
            case -1:
                res = -1.0;
                
                break;
            
            case 0:
                const amrex::Real dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->params[0],
                                                                     x[1]-this->params[1],
                                                                     x[2]-this->params[2])};
                const amrex::Real r = this->params[AMREX_SPACEDIM];
                res = r*r-(AMREX_D_TERM(dx[0]*dx[0],+dx[1]*dx[1],+dx[2]*dx[2]));
            
                break;
            
            default:
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SingleDomain.H - LevelSet::eval\n";
                msg += "| Unexpected geometry flag.\n";
                msg += "| this->flag: "+std::to_string(this->flag)+".\n";
                amrex::Abort(msg);
        }

        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        T res;

        switch(this->flag)
        {
            case -1:
                res = -1.0;
                
                break;
            
            case 0:
                const T dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->params[0],
                                                           x[1]-this->params[1],
                                                           x[2]-this->params[2])};
                const amrex::Real r = this->params[AMREX_SPACEDIM];
                res = r*r-(AMREX_D_TERM(dx[0]*dx[0],+dx[1]*dx[1],+dx[2]*dx[2]));
            
                break;
            
            default:
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SingleDomain.H - LevelSet::operator()\n";
                msg += "| Unexpected geometry flag.\n";
                msg += "| this->flag: "+std::to_string(this->flag)+".\n";
                amrex::Abort(msg);
        }

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
        amrex::GpuArray<T, N> res;

        switch(this->flag)
        {
            case -1:
                AMREX_D_TERM
                (
                    res[0] = 0.0;,
                    res[1] = 0.0;,
                    res[2] = 0.0;
                )
                
                break;
            
            case 0:
                const T dx[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-this->params[0],
                                                           x[1]-this->params[1],
                                                           x[2]-this->params[2])};
            
                AMREX_D_TERM
                (
                    res[0] = -2.0*dx[0];,
                    res[1] = -2.0*dx[1];,
                    res[2] = -2.0*dx[2];
                )
                
                break;
            
            default:
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_SingleDomain.H - LevelSet::grad\n";
                msg += "| Unexpected geometry flag.\n";
                msg += "| this->flag: "+std::to_string(this->flag)+".\n";
                amrex::Abort(msg);
        }

        return res;
    }
    // ================================================================
};
// ####################################################################



// SINGLE DOMAIN: IBVP ################################################
struct IBVP
{
    // DATA MEMBERS ===================================================
    ProblemParameters problem_params;
    LevelSet<AMREX_SPACEDIM> level_set;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    :
    level_set(this->problem_params.shape, this->problem_params.geo_parameters)
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
        return ELASTIC_SOLID_NU;
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
     * \param[out] U: pointer to memory that will be filled with the value of the unknown fields at
     *                (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_exact_solution(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * U) const
    {

        /* Constant initial state */
        AMREX_D_TERM
        (
            U[V1] = 1.0;,
            U[V2] = 1.0;,
            U[V3] = 1.0;
        )
        AMREX_D_TERM
        (
            U[S11] = 0.0;,
            U[S22] = 0.0;,
            U[S33] = 0.0;
        )
#if (AMREX_SPACEDIM == 2)
        U[S12] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        U[S23] = 0.0;
        U[S13] = 0.0;
        U[S12] = 0.0;
#endif
    }

    /**
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the unknown fields at x.
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
     * \param[in] U: pointer to memory containing the value of the unknown fields at (t, x).
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
        const int dom = 0;

        amrex::Real U_exact[ELASTIC_SOLID_NU];
        amrex::Real e, e_exact;

        this->eval_exact_solution(t, x, U_exact);

        //e_exact = elastic_solid::eval_total_energy(rho, c, U_exact);

amrex::Print() << "eval_error_L_inf" << std::endl;
exit(-1);
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
     * \param[in] U: pointer to memory containing the value of the unknown fields at (t, x).
     *
     * \return: time increment based on solution at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval_dt(const int /*dom*/,
                        const amrex::Real * dx,
                        const amrex::Real /*t*/, const amrex::Real * /*x*/,
                        const amrex::Real * U) const
    {
amrex::Print() << "eval_dt" << std::endl;
exit(-1);
        const amrex::Real w = 1.0;//elastic_solid::eval_largest_wave_speed(this->problem_params.gamma, U);
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
     * \param[in] U: pointer to memory containing the unknown fields.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F(const int /*dom*/,
                const amrex::Real /*t*/, const amrex::Real * /*x*/,
                const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3)) const
    {
amrex::Print() << "eval_F" << std::endl;
exit(-1);
        //ideal_gas::eval_F(this->problem_params.gamma, U, AMREX_D_DECL(F1, F2, F3));
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
     * \param[in] m_U: pointer to memory containing the unknown fields on the negative side of the
     *                 unit normal.
     * \param[in] p_U: pointer to memory containing the unknown fields on the positive side of the
     *                 unit normal.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_intraphase(const int /*dom*/,
                             const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn) const
    {
amrex::Print() << "eval_NFn_intraphase" << std::endl;
exit(-1);
        //ideal_gas::eval_NFn_Riemann_solver(this->problem_params.gamma, un, m_U, p_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the unknown fields.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_BCs(const int /*dom*/,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
        // VARIABLES --------------------------------------------------
        amrex::Real U_exact[ELASTIC_SOLID_NU];
        // ------------------------------------------------------------

        // EXACT SOLUTION ---------------------------------------------
        this->eval_exact_solution(t, x, U_exact);
        // ------------------------------------------------------------

amrex::Print() << "eval_NFn_BCs" << std::endl;
exit(-1);
        // BOUNDARY FLUX ----------------------------------------------
        //ideal_gas::eval_NFn_Riemann_solver(this->problem_params.gamma, un, U, U_exact, NFn);
        // ------------------------------------------------------------
    }

    /**
     * \brief Eval the numerical flux at the interfaces.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the unknown fields.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_ICs(const int /*dom*/,
                      const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                      const amrex::Real * U, amrex::Real * NFn) const
    {
amrex::Print() << "eval_NFn_ICs" << std::endl;
exit(-1);
        //ideal_gas::eval_NFn_wall(this->problem_params.gamma, un, U, NFn);
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
        //ideal_gas::eval_FV_slopes(this->problem_params.gamma, geom, mesh, solution, X, dX);
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
        return ELASTIC_SOLID_NU+1+1;
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
        output_fields_names = {"velocity_x", "velocity_y",
                               "sigma_xx", "sigma_yy", "sigma_xy",
                               "sigma_xx_error",
                               "level_set"};
#endif
#if (AMREX_SPACEDIM == 3)
        output_fields_names = {"velocity_x", "velocity_y", "velocity_z",
                               "sigma_xx", "sigma_yy", "sigma_zz", "sigma_yz", "sigma_xy", "sigma_xy",
                               "sigma_xx_error",
                               "level_set"};
#endif
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the unknown fields at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const amrex::Real t, const amrex::Real * x,
                            const amrex::Real * U,
                            amrex::Real * F) const
    {
        // VARIABLES --------------------------------------------------
        amrex::Real U_exact[ELASTIC_SOLID_NU], phi;
        // ------------------------------------------------------------

        // EXACT SOLUTION
        this->eval_exact_solution(t, x, U_exact);

        // LEVEL SET
        phi = this->level_set.eval(x);

        // OUTPUT FIELDS
        AMREX_D_TERM
        (
            F[V1] = U[V1];,
            F[V2] = U[V2];,
            F[V3] = U[V3];
        )
        AMREX_D_TERM
        (
            F[S11] = U[S11];,
            F[S22] = U[S22];,
            F[S33] = U[S33];
        )
#if (AMREX_SPACEDIM == 2)
        F[S12] = U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        F[S23] = U[S23];
        F[S13] = U[S13];
        F[S12] = U[S12];
#endif
        F[ELASTIC_SOLID_NU] = U[S11]-U_exact[S11];
        F[ELASTIC_SOLID_NU+1] = phi;
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

        // VARIABLES --------------------------------------------------
        amrex::dG::Mesh & mesh = *this->meshes[lev];
        amrex::dG::Solution & solution = *this->solutions[lev];
        bool level_uses_FV;
        // ------------------------------------------------------------

        // FV LEVEL ---------------------------------------------------
        level_uses_FV = ((solution.params.space_p == 0) && solution.params.FV_is_active);
        // ------------------------------------------------------------

        // MESH -------------------------------------------------------
        mesh.make_from_scratch(time, this->geom[lev], ba, dm, this->ibvp);
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
        this->export_solution_to_VTK(n, filename_root, t, this->X, this->ibvp);
    }
    // ================================================================
};
// ####################################################################
}
