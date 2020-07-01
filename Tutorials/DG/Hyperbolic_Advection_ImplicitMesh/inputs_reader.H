// inputs_reader.H

#include <AMReX_ParmParse.H>
#include <AMReX_BC_TYPES.H>

// SPACE ##############################################################
struct space_struct
{
    amrex::Vector<int> is_periodic;
    int coord_sys;
    amrex::Vector<amrex::Real> prob_lo, prob_hi;
    amrex::Vector<int> bc_lo, bc_hi;

    space_struct()
    :
    is_periodic({AMREX_D_DECL(1, 1, 1)}),
    coord_sys(0),
    prob_lo({AMREX_D_DECL( 0.0,  0.0,  0.0)}),
    prob_hi({AMREX_D_DECL(-1.0, -1.0, -1.0)}),
    bc_lo({AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir)}),
    bc_hi({AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir)})
    {}
};
// ####################################################################



// TIME ###############################################################
struct time_struct
{
    amrex::Real T;

    time_struct()
    :
    T(0.0)
    {}
};
// ####################################################################



// SPACE AND TIME DISCRETIZATION ######################################
struct mesh_struct
{
    amrex::IntVect n_cells;
    int max_grid_size, n_time_steps;

    mesh_struct():
    n_cells(AMREX_D_DECL(0, 0, 0)),
    max_grid_size(0),
    n_time_steps(0)
    {}
};
// ####################################################################



// DISCONTINOUS GALERKIN METHOD #######################################
struct dG_struct
{
    int phi_space_p, space_p, time_p;
    int space_q, time_q;

    dG_struct()
    :
    phi_space_p(-1),
    space_p(-1),
    time_p(-1)
    {}
};
// ####################################################################



// INPUTS #############################################################
struct inputs_struct
{
    // DATA MEMBERS ===================================================
    space_struct space;
    time_struct time;
    mesh_struct mesh;
    dG_struct dG;
    int plot_int;
    // ================================================================

    // CONSTRUCTOR ====================================================
    inputs_struct()
    :
    plot_int(-1)
    {
        // VARIABLES --------------------------------------------------
        amrex::ParmParse pp;
        amrex::ParmParse pp_space("space");
        amrex::ParmParse pp_time("time");
        amrex::ParmParse pp_mesh("mesh");
        amrex::ParmParse pp_dG("dG");

        amrex::Vector<int> vector_int;
        // ------------------------------------------------------------

        // SPACE ------------------------------------------------------
        pp_space.queryarr("is_periodic", this->space.is_periodic);
        pp_space.query("coord_sys", this->space.coord_sys);
        pp_space.getarr("prob_lo", this->space.prob_lo);
        pp_space.getarr("prob_hi", this->space.prob_hi);
        pp_space.queryarr("bc_lo", this->space.bc_lo);
        pp_space.queryarr("bc_hi", this->space.bc_hi);

        // MAKE SURE BCS ARE GIVEN CONSISTENTLY
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            if ((this->space.bc_lo[d] == amrex::BCType::int_dir) && (this->space.bc_hi[d] == amrex::BCType::int_dir))
            {
                this->space.is_periodic[d] = 1;
            }
            else if ((this->space.bc_lo[d] != amrex::BCType::int_dir) && (this->space.bc_hi[d] != amrex::BCType::int_dir))
            {
                this->space.is_periodic[d] = 0;
            }
        }
        // ------------------------------------------------------------

        // TIME -------------------------------------------------------
        pp_time.get("T", this->time.T);
        // ------------------------------------------------------------

        // SPACE AND TIME DISCRETIZATION ------------------------------
        vector_int.resize(AMREX_SPACEDIM);
        pp_mesh.getarr("n_cells", vector_int);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) this->mesh.n_cells[d] = vector_int[d];
        pp_mesh.get("max_grid_size", this->mesh.max_grid_size);
        pp_mesh.get("n_time_steps", this->mesh.n_time_steps);

        // DG
        pp_dG.get("phi_space_p", this->dG.phi_space_p);
        pp_dG.get("space_p", this->dG.space_p);
        pp_dG.get("time_p", this->dG.time_p);

        this->dG.space_q = std::max(this->dG.phi_space_p+2, this->dG.space_p+2);
        // ------------------------------------------------------------

        // POST-PROCESSING INFO ---------------------------------------
        pp.query("plot_int", this->plot_int);
        // ------------------------------------------------------------
    }
    // ================================================================

    // PRINT TO SCREEN ================================================
    void print() const
    {
        amrex::Print() << "# ANALYSIS PARAMETERS                                                  " << std::endl;
        amrex::Print() << "# SPACE ===============================================================" << std::endl;
        amrex::Print() << "# space.is_periodic = " << AMREX_D_TERM(this->space.is_periodic[0], << ", " << this->space.is_periodic[1], << ", " << this->space.is_periodic[2]) << std::endl;
        amrex::Print() << "# space.coord_sys = " << this->space.coord_sys << std::endl;
        amrex::Print() << "# space.prob_lo = " << AMREX_D_TERM(this->space.prob_lo[0], << ", " << this->space.prob_lo[1], << ", " << this->space.prob_lo[2]) << std::endl;
        amrex::Print() << "# space.prob_hi = " << AMREX_D_TERM(this->space.prob_hi[0], << ", " << this->space.prob_hi[1], << ", " << this->space.prob_hi[2]) << std::endl;
        amrex::Print() << "# space.bc_lo = " << AMREX_D_TERM(this->space.bc_lo[0], << ", " << this->space.bc_lo[1], << ", " << this->space.bc_lo[2]) << std::endl;
        amrex::Print() << "# space.bc_hi = " << AMREX_D_TERM(this->space.bc_hi[0], << ", " << this->space.bc_hi[1], << ", " << this->space.bc_hi[2]) << std::endl;
        amrex::Print() << "#" << std::endl;
        amrex::Print() << "# TIME ================================================================" << std::endl;
        amrex::Print() << "# time.T = " << this->time.T << std::endl;
        amrex::Print() << "#" << std::endl;
        amrex::Print() << "# =====================================================================" << std::endl;
        amrex::Print() << "# SPACE AND TIME DISCRETIZATION =======================================" << std::endl;
        amrex::Print() << "# mesh.n_cells = " << AMREX_D_TERM(this->mesh.n_cells[0], << ", " << this->mesh.n_cells[1], << ", " << this->mesh.n_cells[2]) << std::endl;
        amrex::Print() << "# mesh.max_grid_size = " << this->mesh.max_grid_size << std::endl;
        amrex::Print() << "# mesh.n_time_steps  = " << this->mesh.n_time_steps << std::endl;
        amrex::Print() << "#" << std::endl;
        amrex::Print() << "# dG.phi_space_p = " << this->dG.phi_space_p << std::endl;
        amrex::Print() << "# dG.space_p     = " << this->dG.space_p << std::endl;
        amrex::Print() << "# dG.time_p      = " << this->dG.time_p << std::endl;
        amrex::Print() << "#" << std::endl;
        amrex::Print() << "# =====================================================================" << std::endl;
        amrex::Print() << "# POSTPROCESSING INFO =================================================" << std::endl;
        amrex::Print() << "# plot_int = " << this->plot_int << std::endl;
        amrex::Print() << "#" << std::endl;
        amrex::Print() << "# =====================================================================" << std::endl;
        amrex::Print() << "#######################################################################" << std::endl;
    }
    // ================================================================
};
// ####################################################################