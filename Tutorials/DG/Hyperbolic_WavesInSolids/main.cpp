#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>

#include <AMReX_BC_TYPES.H>

#include <AMReX_Print.H>

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving elastic wave propagations
//
// ####################################################################
// SELECT SET OF PDES =================================================
#define PROBLEM_ONE_PHASE 0
#define PROBLEM_ONE_ANISOTROPIC_PHASE 10
#define PROBLEM_ONE_INTERFACE 1
#define PROBLEM_HP_CONVERGENCE 23
#define PROBLEM_BCC_LATTICE 233

#define PROBLEM 10

#if (PROBLEM == PROBLEM_ONE_PHASE)
#include <IBVP_WavesInSolids_OnePhase.H>
#elif (PROBLEM == PROBLEM_ONE_ANISOTROPIC_PHASE)
#include <IBVP_WavesInSolids_OneAnisotropicPhase.H>
#elif (PROBLEM == PROBLEM_ONE_INTERFACE)
#include <IBVP_WavesInSolids_OneInterface.H>
#elif (PROBLEM == PROBLEM_HP_CONVERGENCE)
#include <IBVP_WavesInSolids_hp_Convergence.H>
#elif (PROBLEM == PROBLEM_BCC_LATTICE)
#include <IBVP_WavesInSolids_BCC_lattice.H>
#endif

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
amrex::Print() << "# In this tutorial, we are solving elastic wave propagations.          " << std::endl;
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

    // DG MODEL =======================================================
    amrex::DG::InputReader dG_inputs;
    // ================================================================

    // TIC ======================
    start_time = amrex::second();
    // ==========================

    // PHYSICAL BOX ===================================================
    amrex::RealBox real_box(dG_inputs.space[0].prob_lo.data(),
                            dG_inputs.space[0].prob_hi.data());
    // ================================================================

    // BOX ARRAY ======================================================
    amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, dG_inputs.mesh[0].n_cells-1);

    amrex::BoxArray ba;
    ba.define(indices_box);
    ba.maxSize(dG_inputs.mesh[0].max_grid_size);
    // ================================================================

    // GEOMETRY =======================================================
    amrex::Geometry geom;
    geom.define(indices_box, &real_box, dG_inputs.space[0].coord_sys, dG_inputs.space[0].is_periodic.data());
    // ================================================================
    
    // BOXES DISTRIBUTION AMONG MPI PROCESSES =========================
    amrex::DistributionMapping dm(ba);
    // ================================================================

    // INIT DG DATA STRUCTURES ========================================
    amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, dG_inputs.dG[0].phi_space_p, dG_inputs.dG[0].space_q);
    amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, dG_inputs.dG[0].space_p, dG_inputs.dG[0].time_p, dG_inputs.dG[0].space_q, dG_inputs.dG[0].time_q);
    amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta");
    dG.InitData(iGeom, MatFactory);
    // ================================================================

    // DESTINATION FOLDER AND OUTPUT DATA =============================
    std::string dst_folder;
    
    const std::string dG_order = "p"+std::to_string(dG_inputs.dG[0].space_p);
    const std::string dG_mesh = AMREX_D_TERM(std::to_string(dG_inputs.mesh[0].n_cells[0]),+"x"+
                                             std::to_string(dG_inputs.mesh[0].n_cells[1]),+"x"+
                                             std::to_string(dG_inputs.mesh[0].n_cells[2]));

#if (PROBLEM == PROBLEM_ONE_PHASE)
    const std::string problem = "PROBLEM_OnePhase";
#elif (PROBLEM == PROBLEM_ONE_ANISOTROPIC_PHASE)
    const std::string problem = "PROBLEM_OneAnisotropicPhase";
#elif (PROBLEM == PROBLEM_ONE_INTERFACE)
    const std::string problem = "PROBLEM_OneInterface";
#elif (PROBLEM == PROBLEM_HP_CONVERGENCE)
    const std::string problem = "PROBLEM_hp_Convergence";
#elif (PROBLEM == PROBLEM_BCC_LATTICE)
    const std::string problem = "PROBLEM_BCC_lattice";
#endif

    dst_folder = "./IBVP_"+std::to_string(AMREX_SPACEDIM)+"d/"+problem+"_"+dG_mesh+"_"+dG_order+"/";

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
#if ((PROBLEM == PROBLEM_ONE_PHASE) || \
     (PROBLEM == PROBLEM_BCC_LATTICE))
    const amrex::Vector<std::string> material_type = {"Isotropic"};
    const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{1.0, 1.0, 0.33}};
#elif (PROBLEM == PROBLEM_ONE_ANISOTROPIC_PHASE)
    const amrex::Vector<std::string> material_type = {"Hexagonal-2D"};
    const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{32.0, 16.7, 14.0, 6.63, 6.6}};
#elif (PROBLEM == PROBLEM_HP_CONVERGENCE)
    const amrex::Vector<std::string> material_type = {"Isotropic", "Isotropic"};
    const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{1.0, 1.0, 0.33}, {1.0, 1.0, 0.33}};
#elif (PROBLEM == PROBLEM_ONE_INTERFACE)
    const amrex::Vector<std::string> material_type = {"Hexagonal-2D", "Hexagonal-2D"};
    const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{71.0, 16.5, 6.2, 3.96, 5.00},
                                                                           {71.0, 16.5, 16.5, 3.96, 8.58}};
#endif

    ELASTIC_WAVES Waves(material_type, material_properties);
    // ================================================================

    // INIT OUTPUT DATA INFORMATION ===================================
    dG.SetOutput(dst_folder, "PointSolution", iGeom, Waves);
    // ================================================================

    // INIT FIELDS' DATA WITH INITIAL CONDITIONS ======================
    iGeom.ProjectDistanceFunctions(Waves);
    iGeom.EvalImplicitMesh(Waves);
    MatFactory.Eval(iGeom);
    dG.SetICs(iGeom, MatFactory, Waves);
    
    // WRITE TO OUTPUT
    dG.PrintPointSolution(0, 0.0, iGeom, MatFactory, Waves);

    if (dG_inputs.plot_int > 0)
    {
        int n = 0;
        amrex::Real time = 0.0;

        iGeom.Export_VTK_Mesh(dst_folder, "Mesh", n, dG_inputs.time.n_steps);
        dG.Export_VTK(dst_folder, "Solution", n, dG_inputs.time.n_steps, time, iGeom, MatFactory, Waves);
    }
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
    while ((time < dG_inputs.time.T*(1.0-1.0e-12)) && (n < dG_inputs.time.n_steps))
    {
        // COMPUTE NEXT TIME STEP
        dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, Waves);
        dt = std::min(time+dt, dG_inputs.time.T)-time;

        // REPORT TO SCREEN
amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1 << " time step: " << dt << ", time = " << time+dt << std::endl;

        // TIME STEP
        dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, Waves);

        // UPDATE TIME AND STEP
        n += 1;
        time += dt;

#if (PROBLEM == PROBLEM_HP_CONVERGENCE)
        // COMPUTE ERROR
        if (std::abs(time/dG_inputs.time.T-1.0) < 1.0e-12)
        {
            amrex::Real err;
            err = dG.EvalError(time, iGeom, MatFactory, Waves);
amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << std::sqrt(err) << std::endl;
        }
#endif

        // WRITE TO OUTPUT
        dG.PrintPointSolution(n, time, iGeom, MatFactory, Waves);

        if ((dG_inputs.plot_int > 0) && (n%dG_inputs.plot_int == 0))
        {
            dG.Export_VTK(dst_folder, "Solution", n, dG_inputs.time.n_steps, time, iGeom, MatFactory, Waves);
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
