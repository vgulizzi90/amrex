#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the Sod's
// tube problem in tilted geometries.
//
// ####################################################################
// SELECT SET OF PDES =================================================
//#include "IBVP_utils.H"
#include "IBVP_SodsTube.H"
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
amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for the Sod's   " << std::endl;
amrex::Print() << "# tube problem in tilted geometries.                                   " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    
    const std::string problem = "PROBLEM_SodsTube";

    // NUMBER OF GHOST ROWS
    const int ngr = 1;

    // IBVP
    const int X_n_comp = N_SOL;
    const amrex::Real gamma = 1.4;
    // ================================================================

    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;
    std::string output_folderpath;

    // INPUTS
    amrex::DG::InputReader inputs;

    // SOLUTION MULTIFAB
    amrex::MultiFab X;

    // IBVP
    IDEAL_GAS IG(gamma, inputs.problem.params);
    // ================================================================

    // DO THE ANALYSIS
    {
        // MAKE OUTPUT FOLDER =========================================
        {
            const amrex::Real diam = inputs.problem.params[7];
            const amrex::Real theta = inputs.problem.params[8];

            const std::string mesh_info = AMREX_D_TERM(std::to_string(inputs.grid.n_cells[0]),+"x"+
                                                       std::to_string(inputs.grid.n_cells[1]),+"x"+
                                                       std::to_string(inputs.grid.n_cells[2]));
            const std::string diam_info = "D0"+std::to_string((int) std::round(diam*100));
            const std::string theta_info = "TH"+std::to_string((int) std::round(theta));

            output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info+"_"+diam_info+"_"+theta_info});

            amrex::DG::IO::MakeFolder(output_folderpath);
        }
        // ============================================================

        // OPEN STATISTICS FILE =======================================
        std::ofstream fp;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            const std::string stats_filepath = amrex::DG::IO::MakePath({output_folderpath, "Stats.txt"});

            time_t date_and_time = time(0);
            char * date_and_time_ = ctime(&date_and_time);
            
            fp.open(stats_filepath, std::ofstream::app);
            fp << std::endl << "ANALYSIS STATISTICS - " << date_and_time_ << "\n";
        }
        // ============================================================

        // HEADER =====================================================
        {
            const amrex::Real diam = inputs.problem.params[7];
            const amrex::Real theta = inputs.problem.params[8];
            
            const std::string diam_info = "0."+std::to_string((int) std::round(diam*100));
            const std::string theta_info = std::to_string((int) std::round(theta));

            amrex::Print() << std::endl;
            amrex::Print() << "# SOD'S TUBE TEST" << " - DIAMETER: " << diam_info << " - INCLINATION: " << theta_info << std::endl;
        }
        // ============================================================

        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================

        // MAKE THE IMPLICIT-MESH =====================================
        amrex::DG::ImplicitMesh mesh(inputs);

        mesh.MakeFromScratch(IG);

        // WRITE THE LEVELSETS TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            ExportBase_VTK(output_folderpath, "Levelset", n, inputs.time.n_steps,
                           t, mesh.geom, mesh.std_elem, 1, mesh.PHI, "LS_");
            ExportImplicitMesh_VTK(output_folderpath, "ImplicitMesh", n, inputs.time.n_steps,
                                   t, mesh);
        }
        // ============================================================

        // INIT THE MATRIX FACTORY ====================================
        amrex::DG::MatrixFactory matfactory(inputs);

        matfactory.EvalMassMatrices(mesh);
        // ============================================================

        // INIT MULTIFAB ==============================================
        X.define(mesh.cc_ba, mesh.dm, X_n_comp, ngr);
        // ============================================================

        // SET INITIAL CONDITIONS =====================================
        //ProjectInitialConditions(mesh, N_SOL, X, IG);

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            /*
            Export_VTK(output_folderpath, "Solution", n, inputs.time.n_steps,
                       t, mesh, N_SOL, X,
                       IG);
            */
        }
        // ============================================================

        // TOC ========================================================
        stop_time = amrex::second();
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
        // ============================================================

        // CLOSING ====================================================
        amrex::Print() << "# Time = " << std::scientific << std::setprecision(5) << std::setw(12) << (stop_time-start_time) << " s" << std::endl;
        // ============================================================

        // CLOSE STATISTICS FILE ======================================
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            fp.close();
        }
        // ============================================================
    }

    // CLOSING ========================================================
amrex::Print() << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================
}
// ####################################################################



// DUMMY MAIN #########################################################
int main(int argc, char * argv[])
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