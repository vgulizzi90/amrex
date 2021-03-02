#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we showcase some geometry-generation capabilities
// of the implicit-mesh adaptive mesh refinement framework.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_Geometry.H"
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
    amrex::Print() << "# In this tutorial, we showcase some geometry-generation capabilities  " << std::endl;
    amrex::Print() << "# of the implicit-mesh adaptive mesh refinement framework.             " << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
    amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    // ================================================================

    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;
    AMR_GEOMETRY amr;
    // ================================================================

    // DO THE ANALYSIS
    {
        // MAKE OUTPUT FOLDER =========================================
        {
            amrex::DG::IO::MakeFolder(amr.inputs.plot_filepath);
        }
        // ============================================================

        // OPEN STATISTICS FILE =======================================
        std::ofstream fp;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            const std::string stats_filepath = amrex::DG::IO::MakePath({amr.inputs.plot_filepath, "Stats.txt"});

            time_t date_and_time = time(0);
            char * date_and_time_ = ctime(&date_and_time);
            
            fp.open(stats_filepath, std::ofstream::app);
            fp << std::endl << "ANALYSIS STATISTICS - " << date_and_time_ << "\n";
#ifdef AMREX_DEBUG
            fp << "| Debug active: true" << std::endl;
#else
            fp << "| Debug active: false" << std::endl;
#endif
#ifdef AMREX_USE_GPU
            fp << "| Using GPUs: true" << std::endl;
#else
            fp << "| Using GPUs: false" << std::endl;
#endif
            fp << "| Number of MPI ranks: " << amrex::ParallelDescriptor::NProcs() << std::endl;
            fp << "| Number of AMR levels: " << amr.maxLevel()+1 << std::endl;
        }
        // ============================================================

        // HEADER =====================================================
        {
            amrex::Print() << "# GEOMETRY GENERATION TEST" << std::endl;
        }
        // ============================================================

        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================

        // GENERATE GEOMETRY ==========================================
        amr.Init();
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::DG::Export_VTK(amr.inputs.plot_filepath, n, amr.inputs.time.n_steps,
                                  t, N_SOL, amr, amr.GG);
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