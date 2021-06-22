//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_HeatEquation1D.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#define N_U 1

namespace heat_equation_1d
{

// HEAT EQUATION 1D: LEVEL SET ########################################
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
        const amrex::Real res = -1.0;

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



// HEAT EQUATION 1D: IBVP #############################################
struct IBVP
:
public amrex::dG::IBVP_Base
{
    // DATA MEMBERS ===================================================
    // PENALTY PARAMETERS
    amrex::Real muD, muI;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        amrex::ParmParse pp;
        amrex::Real mu;

        pp.get("penalty", mu);

        this->muD = mu;
        this->muI = mu;
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
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
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
        const amrex::Real k = 2.0*M_PI;

        U[0] = std::sin(k*x[0])*std::exp(-k*k*t);
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
        amrex::Real U_exact[N_U];

        this->eval_exact_solution(t, x, U_exact);

        err = std::abs(U[0]-U_exact[0]);
        err_norm = std::abs(U_exact[0]);
    }

    /**
     * \brief Eval the error at (t, x) to be used for the L_2 norm error.
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
    void eval_error_L_2(const int /*dom*/,
                        const amrex::Real t, const amrex::Real * x,
                        const amrex::Real * U,
                        amrex::Real & err, amrex::Real & err_norm) const
    {
        amrex::Real U_exact[N_U];

        this->eval_exact_solution(t, x, U_exact);

        err = (U[0]-U_exact[0])*(U[0]-U_exact[0]);
        err_norm = (U_exact[0])*(U_exact[0]);
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
        const amrex::Real h = dx[0];

        return h*h;
    }
    // ================================================================


    // GOVERNING EQUATIONS ============================================
    /**
     * \brief Eval the diffusive/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the solution.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the space-derivatives of the solution.
     * \param[out] D1, D2, D3: pointer to memory that will contain the diffusive terms.
     * \param[out] R: pointer to memory that will contain the reactive term.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_DR(const int /*dom*/,
                 const amrex::Real /*t*/, const amrex::Real * /*x*/,
                 const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                 AMREX_D_DECL(amrex::Real * D1, amrex::Real * /*D2*/, amrex::Real * /*D3*/), amrex::Real * R) const
    {
        D1[0] = U_1[0];
        R[0] = 0.0;
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
     * \param[in] m_U: pointer to memory containing the solution on the negative side of the unit
     *                 normal.
     * \param[in] m_U_1, m_U_2, m_U_3: pointer to memory containing the derivatives of the solution
     *                                 on the negative side of the unit normal.
     * \param[in] p_U: pointer to memory containing the solution on the positive side of the unit
     *                 normal.
     * \param[in] p_U_1, p_U_2, p_U_3: pointer to memory containing the derivatives of the solution
     *                                 on the positive side of the unit normal.
     * \param[out] m_ND1, m_ND2, m_ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] m_NFn: pointer to memory that will contain the numerical flux.
     * \param[out] p_ND1, p_ND2, p_ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] p_NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_intraphase(const int /*dom*/,
                            const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                            const amrex::Real * m_U, AMREX_D_DECL(const amrex::Real * m_U_1, const amrex::Real * /*m_U_2*/, const amrex::Real * /*m_U_3*/),
                            const amrex::Real * p_U, AMREX_D_DECL(const amrex::Real * p_U_1, const amrex::Real * /*p_U_2*/, const amrex::Real * /*p_U_3*/),
                            AMREX_D_DECL(amrex::Real * m_ND1, amrex::Real * /*m_ND2*/, amrex::Real * /*m_ND3*/), amrex::Real * m_NFn,
                            AMREX_D_DECL(amrex::Real * p_ND1, amrex::Real * /*p_ND2*/, amrex::Real * /*p_ND3*/), amrex::Real * p_NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muI;

        // VARIABLES
        amrex::Real dU[N_U], T[N_U];

        // JUMP AND AVERAGE OF NORMAL FLUX
        dU[0] = m_U[0]-p_U[0];
        T[0] = 0.5*(m_U_1[0]+p_U_1[0])*un[0];

        // NUMERICAL FLUXES
        m_ND1[0] = 0.5*dU[0]*un[0];
        m_NFn[0] = T[0]-mu*dU[0];

        p_ND1[0] = 0.5*dU[0]*un[0];
        p_NFn[0] = -T[0]+mu*dU[0];
    }

    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the solution.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the solution.
     * \param[out] ND1, ND2, ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_BCs(const int /*dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                     AMREX_D_DECL(amrex::Real * ND1, amrex::Real * /*ND2*/, amrex::Real * /*ND3*/),
                     amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muD;

        // VARIABLES
        amrex::Real dU[N_U];

        // JUMP
        dU[0] = U[0];

        // NUMERICAL FLUXES
        ND1[0] = dU[0]*un[0];

        NFn[0] = un[0]*U_1[0]-mu*dU[0];
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
        return N_U+1;
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
        output_fields_names = {"u", "u_error"};
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
        amrex::Real U_exact[N_U];

        this->eval_exact_solution(t, x, U_exact);

        F[0] = U[0];
        F[1] = U[0]-U_exact[0];
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
    amrex::Vector<amrex::MultiFab> X, L;

    // NORM
    //std::string norm;
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
            //pp.get("norm", this->norm);
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
    void eval_error(const amrex::Real t, amrex::Real & err_L_inf, amrex::Real & err_L_inf_norm, amrex::Real & err_L_2, amrex::Real & err_L_2_norm) const
    {
        this->eval_error_L_inf(t, this->X, err_L_inf, err_L_inf_norm, this->ibvp);
        this->eval_error_L_2(t, this->X, err_L_2, err_L_2_norm, this->ibvp);
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t) const
    {
        return amrex::dG::eval_dt(*this, t, this->X, this->ibvp);
    }

    void take_time_step(const amrex::Real t, const amrex::Real dt)
    {
        amrex::dG::parabolic_pde::take_time_step(*this, t, dt, this->X, this->ibvp);
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

} // namespace heat_equation_1d