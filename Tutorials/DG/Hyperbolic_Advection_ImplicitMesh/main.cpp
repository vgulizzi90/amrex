#include <AMReX_PlotFileUtil.H>
#include <AMReX_Geometry.H>

#include <AMReX_Print.H>

#include <AMReX_DG.H>

#include "inputs_reader.H"

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving an advection equation of the form
//
// phi_{,t}+(u_i phi)_{,i} = 0.
//
// where phi is the advected field and u = {u_i} is a divergence-free
// velocity field.
// The problem is subjected to:
// i) either periodic or inflow boundary conditions
// ii) initial conditions: phi(0,x) = phi0(x)
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include <IBVP_Linear_Advection.H>
// ====================================================================
// ####################################################################

// ACTUAL MAIN PROGRAM ################################################
void main_main()
{
    // HEADING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# AMREX & DG PROJECT                                                   " << std::endl;
amrex::Print() << "# Author: Vincenzo Gulizzi (vgulizzi@lbl.gov)                          " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# SUMMARY:                                                             " << std::endl;
amrex::Print() << "# In this tutorial, we are solving an advection equation of the form   " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# rho_{,t}+(a_i rho)_{,i} = 0                                          " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# where rho is the advected field and a = {a_i} is a divergence-free   " << std::endl;
amrex::Print() << "# velocity field.                                                      " << std::endl;
amrex::Print() << "# The problem is subjected to:                                         " << std::endl;
amrex::Print() << "# i) either periodic or inflow boundary conditions                     " << std::endl;
amrex::Print() << "# ii) initial conditions: rho(0,x) = rho0(x)                           " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // GENERAL PARAMETERS =============================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    // ================================================================

    // GENERAL VARIABLES ==============================================
    amrex::Real start_time, stop_time;
    amrex::Real start_time_per_step, stop_time_per_step, time_per_step;
    // ================================================================

    // INPUTS DATA STRUCTURE
    inputs_struct inputs;

    // REPORT TO SCREEN
    inputs.print();
    // =====================

    // TIC ======================
    start_time = amrex::second();
    // ==========================

    // PHYSICAL BOX ===================================================
    amrex::RealBox real_box(inputs.space.prob_lo.data(),
                            inputs.space.prob_hi.data());
    // ================================================================

    // BOX ARRAY ======================================================
    amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, inputs.mesh.n_cells-1);

    amrex::BoxArray ba;
    ba.define(indices_box);
    ba.maxSize(inputs.mesh.max_grid_size);
    // ================================================================

    // GEOMETRY =======================================================
    amrex::Geometry geom;
    geom.define(indices_box, &real_box, inputs.space.coord_sys, inputs.space.is_periodic.data());
    // ================================================================
    
    // BOXES DISTRIBUTION AMONG MPI PROCESSES =========================
    amrex::DistributionMapping dm(ba);
    // ================================================================

    // INIT DG DATA STRUCTURES ========================================
    amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, inputs.dG.phi_space_p, inputs.dG.space_q);
    amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, inputs.dG.space_p, inputs.dG.time_p, inputs.dG.space_q, 0);
    amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta");
    dG.InitData(iGeom, MatFactory);
    // ================================================================

    // DESTINATION FOLDER =============================================
    const std::string dG_order = "p"+std::to_string(inputs.dG.space_p);
    std::string ics_type;
    std::string bcs_type;
    std::string dst_folder;

    ics_type = "ICS_periodic";
    bcs_type = "BCS_periodic";
    dst_folder = "./IBVP_"+std::to_string(AMREX_SPACEDIM)+"d_Linear_Advection/"+ics_type+"_"+bcs_type+"_"+dG_order+"/";

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if (!amrex::UtilCreateDirectory(dst_folder, 0755))
        {
            amrex::CreateDirectoryFailed(dst_folder);
        }
    }
    amrex::ParallelDescriptor::Barrier();
    // ================================================================

    // INIT IBVP DATA STRUCTURE =======================================
    LINADV LinAdv;
    // ================================================================

    // INIT FIELDS' DATA WITH INITIAL CONDITIONS ======================
    iGeom.ProjectDistanceFunctions(LinAdv);
    iGeom.EvalImplicitMesh(LinAdv);
    MatFactory.Eval(iGeom);
    dG.SetICs(iGeom, MatFactory, LinAdv);

    // WRITE TO OUTPUT
    if (inputs.plot_int > 0)
    {
        int n = 0;
        amrex::Real time = 0.0;
        std::vector<int> field_domains = {0, 0, 1, 1};
        std::vector<std::string> field_names = {"rho_a", "err_a", "rho_b", "err_b"};

        iGeom.Export_VTK_Mesh(dst_folder, "Mesh", n, inputs.mesh.n_time_steps);
        //dG.Export_VTK(dst_folder, "Solution", n, inputs.mesh.n_time_steps, field_domains, field_names, time, iGeom, MatFactory, LinAdv);
    }
exit(-1);
    // ================================================================

    // START THE ANALYSIS (ADVANCE IN TIME) ===========================
amrex::Print() << "# START OF THE ANALYSIS                                                " << std::endl;
    
    // VARIABLES --------
    int n;
    amrex::Real time, dt;
    // ------------------

    // TIME MARCHING TIC -----------------
    start_time_per_step = amrex::second();
    // -----------------------------------

    // ADVANCE IN TIME ------------------------------------------------
    n = 0;
    time = 0.0;
    while ((time < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.mesh.n_time_steps))
    {
        // COMPUTE NEXT TIME STEP
        dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, LinAdv);
        dt = std::min(time+dt, inputs.time.T)-time;

        // REPORT TO SCREEN
amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1 << " time step: " << dt << ", time = " << time+dt << std::endl;

        // TIME STEP
        dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, LinAdv);

        // UPDATE TIME AND STEP
        n += 1;
        time += dt;

        // COMPUTE ERROR
        if (std::abs(time/inputs.time.T-1.0) < 1.0e-12)
        {
            amrex::Real err;
            err = dG.EvalError(time, iGeom, MatFactory, LinAdv);
amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << std::sqrt(err) << std::endl;
        }

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0 && n%inputs.plot_int == 0)
        {
            std::vector<int> field_domains = {0, 0, 1, 1};
            std::vector<std::string> field_names = {"rho_a", "err_a", "rho_b", "err_b"};

            dG.Export_VTK(dst_folder, "Solution", n, inputs.mesh.n_time_steps, field_domains, field_names, time, iGeom, MatFactory, LinAdv);
        }
    }
    // ----------------------------------------------------------------

    // TIME MARCHING TOC ----------------------------------------------
    stop_time_per_step = amrex::second();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time_per_step, IOProc);
    time_per_step = (stop_time_per_step-start_time_per_step)/n;
    // ----------------------------------------------------------------

amrex::Print() << "# END OF THE ANALYSIS                                                  " << std::endl;
    // ================================================================

    // TOC ============================================================
    stop_time = amrex::second();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
    // ================================================================

    // CLOSING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "# Number of steps = " << n << std::endl;
amrex::Print() << "# Time          = " << std::scientific << std::setprecision(5) << std::setw(12) << (stop_time-start_time) << " s" << std::endl;
amrex::Print() << "# Time per step = " << std::scientific << std::setprecision(5) << std::setw(12) << time_per_step << " s" << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

}
// ####################################################################


// DUMMY MAIN #########################################################
int main(int argc, char* argv[])
{
    // INIT AMREX ================
    amrex::Initialize(argc, argv);
    // ===========================

    main_main();

    // END AMREX =====
    amrex::Finalize();
    // ===============

    return 0;
}
// ####################################################################