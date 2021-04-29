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
// SUPERSONIC VORTEX: LEVEL SET #######################################
template <int N>
struct LevelSet
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    amrex::Real ri, ro;
    // ================================================================


    // CONSTRUCTOR ====================================================
    LevelSet()
    :
    t{0.0},
    dom{0}
    {
        amrex::ParmParse pp;

        pp.get("r_inner", this->ri);
        pp.get("r_outer", this->ro);
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
// ####################################################################



// SUPERSONIC VORTEX: IBVP ############################################
struct IBVP
{
    // DATA MEMBERS ===================================================
    amrex::Real gamma;
    amrex::Real rho_inner, Mach_inner, r_inner, r_outer;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        amrex::ParmParse pp;
        
        pp.get("gamma", this->gamma);
        
        pp.get("rho_inner", this->rho_inner);
        pp.get("Mach_inner", this->Mach_inner);
        pp.get("r_inner", this->r_inner);
        pp.get("r_outer", this->r_outer);
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
        const amrex::Real g = this->gamma;
        const amrex::Real g2 = 0.5*(g-1.0);
        const amrex::Real ig = 1.0/(g-1.0);
        const amrex::Real rho_i = this->rho_inner;
        const amrex::Real M_i = this->Mach_inner;

        const amrex::Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]);
        const amrex::Real cth = x[0]/r;
        const amrex::Real sth = x[1]/r;
        const amrex::Real ri = this->r_inner;
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
        const amrex::Real w = ideal_gas::eval_largest_wave_speed(this->gamma, U);
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
        ideal_gas::eval_F(this->gamma, U, AMREX_D_DECL(F1, F2, F3));
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
        ideal_gas::eval_NFn_Riemann_solver(this->gamma, un, m_U, p_U, NFn);
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
        ideal_gas::eval_NFn_Riemann_solver(this->gamma, un, U, U_exact, NFn);
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
        ideal_gas::eval_NFn_wall(this->gamma, un, U, NFn);
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
        ideal_gas::eval_FV_slopes(this->gamma, geom, mesh, solution, X, dX);
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
        return IDEAL_GAS_NU+4;
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
                               "density_error"};
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
        const amrex::Real g = this->gamma;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p, a, M;
        amrex::Real U_exact[IDEAL_GAS_NU];
        // ------------------------------------------------------------

        // PRIMITIVE VARIABLES
        ideal_gas::eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);

        // SOUND SPEED AND MACH NUMBER
        a = std::sqrt(g*p/rho);
        M = std::sqrt(AMREX_D_TERM(v1*v1,+v2*v2,+v3*v3))/a;

        // EXACT SOLUTION
        this->eval_exact_solution(t, x, U_exact);

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
            msg +=  "ERROR: IBVP_SupersonicVortex.H - AMR::MakeNewLevelFromScratch\n";
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
            msg +=  "ERROR: IBVP_SupersonicVortex.H - AMR::eval_error\n";
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
                msg +=  "ERROR: IBVP_SupersonicVortex.H - AMR::export_solution\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
        }
    }
    // ================================================================
};
// ####################################################################

} // SupersonicVortex