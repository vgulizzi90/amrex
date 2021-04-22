//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_EmbeddedSodsTube.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_IdealGas.H"

namespace EmbeddedSodsTube
{

// SUPERSONIC VORTEX: PROBLEM PARAMETERS ##############################
struct ProblemParameters
{
    // DATA MEMBERS ===================================================
    amrex::Real gamma;
    amrex::Real rhoL, vL, pL, rhoR, vR, pR;
    amrex::Real diam, theta, phi;
    // ================================================================

    // CONSTRUCTOR ====================================================
    ProblemParameters()
    {
        amrex::ParmParse pp;

        pp.get("gamma", this->gamma);

        pp.get("rhoL", this->rhoL);
        pp.get("vL", this->vL);
        pp.get("pL", this->pL);
        pp.get("rhoR", this->rhoR);
        pp.get("vR", this->vR);
        pp.get("pR", this->pR);

        pp.get("diam", this->diam);
        pp.get("theta", this->theta);
        this->theta *= M_PI/180.0;
#if (AMREX_SPACEDIM == 3)
        pp.get("phi", this->phi);
        this->phi *= M_PI/180.0;
#endif
    }
    // ================================================================
};
// ####################################################################



// SUPERSONIC VORTEX: IBVP ############################################
template <int N>
struct Levelset
{
    // DATA MEMBERS ===================================================
    amrex::Real diam, theta, phi;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
#if (AMREX_SPACEDIM == 2)
    Levelset(const amrex::Real diam_, const amrex::Real theta_)
    :
    diam{diam_},
    theta{theta_},
    phi{0.0}
    {}
#endif
#if (AMREX_SPACEDIM == 3)
    Levelset(const amrex::Real diam_, const amrex::Real theta_, const amrex::Real phi_)
    :
    diam{diam_},
    theta{theta_},
    phi{phi_}
    {}
#endif
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
        const amrex::Real r = 0.5*this->diam;
#if (AMREX_SPACEDIM == 2)
        const amrex::Real th = this->theta;
        const amrex::Real xi[2] = {std::cos(th), std::sin(th)};
        const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
        const amrex::Real y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
        const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real th = this->theta;
        const amrex::Real ph = this->phi;
        const amrex::Real xi[3] = {std::cos(th)*std::cos(ph), std::sin(th)*std::cos(ph), std::sin(ph)};
        const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
        const amrex::Real y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
        const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]);
#endif
        const amrex::Real res = d2-r*r;

        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real r = 0.5*this->diam;
#if (AMREX_SPACEDIM == 2)
        const amrex::Real th = this->theta;
        const amrex::Real xi[2] = {std::cos(th), std::sin(th)};
        const T s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
        const T y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
        const T d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real th = this->theta;
        const amrex::Real ph = this->phi;
        const amrex::Real xi[3] = {std::cos(th)*std::cos(ph), std::sin(th)*std::cos(ph), std::sin(ph)};
        const T s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
        const T y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
        const T d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]);
#endif
        const T res = d2-r*r;

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real th = this->theta;
        const amrex::Real xi[2] = {std::cos(th), std::sin(th)};
        const T s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
        const T y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
        const T dd[2] = {(x[0]-y[0])*(1.0-xi[0]*xi[0])+(x[1]-y[1])*(   -xi[1]*xi[0]),
                         (x[0]-y[0])*(   -xi[0]*xi[1])+(x[1]-y[1])*(1.0-xi[1]*xi[1])};
        const amrex::GpuArray<T, N> res = {2.0*dd[0], 2.0*dd[1]};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real th = this->theta;
        const amrex::Real ph = this->phi;
        const amrex::Real xi[3] = {std::cos(th)*std::cos(ph), std::sin(th)*std::cos(ph), std::sin(ph)};
        const T s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
        const T y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
        const amrex::GpuArray<T, N> res = {};
#endif

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
    level_set(AMREX_D_DECL(this->problem_params.diam, this->problem_params.theta, this->problem_params.phi))
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
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs(const amrex::Real * x, amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->problem_params.gamma;
        const amrex::Real rhoL = this->problem_params.rhoL;
        const amrex::Real vL = this->problem_params.vL;
        const amrex::Real pL = this->problem_params.pL;
        const amrex::Real rhoR = this->problem_params.rhoR;
        const amrex::Real vR = this->problem_params.vR;
        const amrex::Real pR = this->problem_params.pR;

        const amrex::Real th = this->problem_params.theta;
        const amrex::Real cth = std::cos(th);
        const amrex::Real sth = std::sin(th);
#if (AMREX_SPACEDIM == 3)
        const amrex::Real ph = this->problem_params.phi;
        const amrex::Real cph = std::cos(ph);
        const amrex::Real sph = std::sin(ph);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
        const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-0.5, x[1]-0.5, x[2]-0.5)};
        const amrex::Real xi = AMREX_D_TERM(xi_dir[0]*y[0],+xi_dir[1]*y[1],+xi_dir[2]*y[2]);

        amrex::Real Uxi[IDEAL_GAS_NU];

        if (xi > 0.0)
        {
            Uxi[RHO] = rhoR;
            AMREX_D_TERM
            (
                Uxi[RHOV1] = rhoR*vR;,
                Uxi[RHOV2] = 0.0;,
                Uxi[RHOV3] = 0.0;
            )
            Uxi[TE] = pR/(g-1.0)+0.5*rhoR*vR*vR;
        }
        else
        {
            Uxi[RHO] = rhoL;
            AMREX_D_TERM
            (
                Uxi[RHOV1] = rhoL*vL;,
                Uxi[RHOV2] = 0.0;,
                Uxi[RHOV3] = 0.0;
            )
            Uxi[TE] = pL/(g-1.0)+0.5*rhoL*vL*vL;
        }

        ideal_gas::Un_to_U(xi_dir, Uxi, U);

        /* Constant initial state */
        U[RHO] = 1.0;
        U[RHOV1] = 0.0;
        U[RHOV2] = 0.0;
        U[TE] = 1.0/(g-1.0);
        /**/
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
        amrex::Real U_ghost[IDEAL_GAS_NU];
        // ------------------------------------------------------------

        // INITIAL CONDITIONS SOLUTION --------------------------------
        this->eval_ICs(x, U_ghost);
        // ------------------------------------------------------------

        // BOUNDARY FLUX ----------------------------------------------
        ideal_gas::eval_NFn_Riemann_solver(this->problem_params.gamma, un, U, U_ghost, NFn);
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
        return IDEAL_GAS_NU+3+AMREX_SPACEDIM+1;
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
                               AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                               "pressure",
                               "mach",
                               "total_energy",
                               "internal_energy",
                               AMREX_D_DECL("velocity_xi", "velocity_eta", "velocity_zeta"),
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
        const amrex::Real th = this->problem_params.theta;
        const amrex::Real cth = std::cos(th);
        const amrex::Real sth = std::sin(th);
#if (AMREX_SPACEDIM == 3)
        const amrex::Real ph = this->problem_params.phi;
        const amrex::Real cph = std::cos(ph);
        const amrex::Real sph = std::sin(ph);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Real rho, AMREX_D_DECL(v1, v2, v3), p, a, M;
        amrex::Real phi, Uxi[IDEAL_GAS_NU];
        // ------------------------------------------------------------

        // PRIMITIVE VARIABLES
        ideal_gas::eval_primitive_variables(g, U, rho, AMREX_D_DECL(v1, v2, v3), p);

        // SOUND SPEED AND MACH NUMBER
        a = std::sqrt(g*p/rho);
        M = std::sqrt(AMREX_D_TERM(v1*v1,+v2*v2,+v3*v3))/a;

        // SOLUTION STATE IN THE TUBE'S REFERENCE SYSTEM
        ideal_gas::U_to_Un(xi_dir, U, Uxi);

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
        AMREX_D_TERM
        (
            F[TE+4] = Uxi[RHOV1]/rho;,
            F[TE+5] = Uxi[RHOV2]/rho;,
            F[TE+6] = Uxi[RHOV3]/rho;
        )
        F[TE+4+AMREX_SPACEDIM] = phi;
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

    // USED FOR THE AMR
    amrex::Vector<amrex::Real> thr;

    // USED FOR THE SHOCK SENSOR
    struct
    {
        amrex::Vector<amrex::dG::Solution *> solutions;
        amrex::Vector<amrex::MultiFab> X;
        amrex::Vector<amrex::Real> thr;
    }
    sensor;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        this->X.resize(n_levels);
        this->thr.resize(n_levels);
        
        this->sensor.solutions.resize(n_levels);
        for (int lev = 0; lev < n_levels; ++lev)
        {
            this->sensor.solutions[lev] = new amrex::dG::Solution();
        }
        this->sensor.X.resize(n_levels);
        this->sensor.thr.resize(n_levels);
    }
    // ================================================================


    // CONSTRUCTOR ====================================================
    ~AMR()
    {
        const int n_levels = this->max_level+1;

        for (int lev = 0; lev < n_levels; ++lev)
        {
            if (this->sensor.solutions[lev] != nullptr)
            {
                delete this->sensor.solutions[lev];
            }
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

        // PARAMETERS -------------------------------------------------
        const int n_levels = this->max_level+1;
        // ------------------------------------------------------------

        // INIT SOLUTIONS INPUTS FOR THE SHOCK SENSOR -----------------
        for (int lev = 0; lev < n_levels; ++lev)
        {
            amrex::ParmParse pp("solution["+std::to_string(lev)+"]");
            this->sensor.solutions[lev]->read_input_file(pp);
            this->sensor.solutions[lev]->params.space_p -= 1;
        }
        // ------------------------------------------------------------

        // THRESHOLD PARAMETERS ---------------------------------------
        {
            amrex::ParmParse pp;
            pp.getarr("threshold_amr", this->thr);
            pp.getarr("threshold_sensor", this->sensor.thr);
        }
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
        mesh.make_from_scratch(time, this->geom[lev], ba, dm, this->ibvp);
        // ------------------------------------------------------------

        // SOLUTION ---------------------------------------------------
        solution.make_from_scratch(time, this->geom[lev], mesh, this->X[lev], this->ibvp);
        solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);
        solution.project_initial_conditions(this->geom[lev], mesh, this->X[lev], this->ibvp);

        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            solution.eval_space_centroids(this->geom[lev], mesh, this->ibvp);
        }
        // ------------------------------------------------------------

        // INTERPOLATION OPERATOR -------------------------------------
        // Interpolation operator used for the shock sensor
        if (lev == 0)
        {
            amrex::dG::Solution & sensor_solution = *this->sensor.solutions[lev];
            
            sensor_solution.make_from_scratch(time, this->geom[lev], mesh, this->sensor.X[lev], this->ibvp);
            sensor_solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);

            amrex::dG::amr_utils::eval_space_interpolation(this->geom[lev], mesh, solution,
                                                           amrex::IntVect::TheUnitVector(),
                                                           this->geom[lev], mesh, sensor_solution,
                                                           ibvp);
        }

        // Proper interpolation operator
        if (lev > 1)
        {
            this->eval_space_interpolation(lev, this->ibvp);
        }
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(mesh.ba, mesh.dm, 1, mask_n_grow);
        this->masks[lev] = 0;
        // ------------------------------------------------------------

        // SET BOX ARRAY AND DISTRIBUTION MAP -------------------------
        this->SetBoxArray(lev, mesh.ba);
        this->SetDistributionMap(lev, mesh.dm);
        // ------------------------------------------------------------
    }
    // ================================================================


    // TAG CELL FOR REFINEMENT ========================================
    virtual void ErrorEst(int lev, amrex::TagBoxArray & tags, amrex::Real time, int /*ngr*/) override
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("EmbeddedSodsTube::AMR::ErrorEst(...)");
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const char non_tagged = amrex::TagBox::CLEAR;
        const char tagged = amrex::TagBox::SET;

        const amrex::dG::Mesh & mesh = *this->meshes[lev];
        const amrex::dG::Solution & solution = *this->solutions[lev];
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        tags.setVal(non_tagged);
        // ------------------------------------------------------------

        // TAG CELLS FOR REFINEMENT -----------------------------------
        // SHOCK SENSOR
        if (lev == 0)
        {
            const amrex::dG::Solution & sensor_solution = *this->sensor.solutions[lev];

            amrex::dG::amr_utils::interpolate(this->geom[lev], solution, this->X[lev],
                                              amrex::IntVect::TheUnitVector(),
                                              this->geom[lev], sensor_solution, this->sensor.X[lev],
                                              ibvp);
            ideal_gas::shock_sensor(this->geom[lev], mesh, solution, this->X[lev], this->sensor.X[lev], this->sensor.thr[lev], tags, tagged);
        }
        // ------------------------------------------------------------

        // MANUALLY TRACK THE SHOCK AT t = 0 --------------------------
        {
            const int dom = 0;
            const amrex::Real th = this->ibvp.problem_params.theta;
            const amrex::Real cth = std::cos(th);
            const amrex::Real sth = std::sin(th);
#if (AMREX_SPACEDIM == 3)
            const amrex::Real ph = this->ibvp.problem_params.phi;
            const amrex::Real cph = std::cos(ph);
            const amrex::Real sph = std::sin(ph);
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
            if (time < 1.0e-12)
            {
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
                            const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(xc[0]-0.5, xc[1]-0.5, xc[2]-0.5)};
                            const amrex::Real xi = AMREX_D_TERM(xi_dir[0]*y[0],+xi_dir[1]*y[1],+xi_dir[2]*y[2]);

                            if (std::abs(xi) < amrex::max(AMREX_D_DECL(dx[0], dx[1], dx[2])))
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
        }
        // ------------------------------------------------------------
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

} // EmbeddedSodsTube