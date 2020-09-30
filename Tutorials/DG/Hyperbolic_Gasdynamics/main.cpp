#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>

#include <AMReX_BC_TYPES.H>

#include <AMReX_Print.H>

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving Gasdynamics equations.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#define PROBLEM_SODS_TUBE_REF 0
#define PROBLEM_SODS_TUBE 1
#define PROBLEM_EB_COMPARISON 2

#define PROBLEM 1

#if (PROBLEM == PROBLEM_SODS_TUBE_REF)
#include <IBVP_SodsTube_Reference.H>
#elif (PROBLEM == PROBLEM_SODS_TUBE)
#include <IBVP_SodsTube.H>
#elif (PROBLEM == PROBLEM_EB_COMPARISON)
#include <IBVP_EB_Comparison.H>
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
amrex::Print() << "# In this tutorial, we are solving Gasdynamics equations.              " << std::endl;
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

    // INPUTS
    amrex::DG::InputReader inputs;
    // ================================================================

    // TIC ============================================================
    start_time = amrex::second();
    // ================================================================

    // OPEN FILE WHERE TO WRITE THE HP RESULTS ========================
    std::ofstream fp;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        time_t date_and_time = time(0);
        char * date_and_time_ = ctime(&date_and_time);
#if (PROBLEM == PROBLEM_SODS_TUBE_REF)
        fp.open("hp_SodsTube_Reference.txt", std::ofstream::app);
#elif (PROBLEM == PROBLEM_SODS_TUBE)
        fp.open("hp_SodsTube.txt", std::ofstream::app);
#elif (PROBLEM == PROBLEM_EB_COMPARISON)
        fp.open("hp_EB_Comparison.txt", std::ofstream::app);
#endif
        fp << std::endl << "HP ANALYSIS: " << date_and_time_ << "\n";
    }
    amrex::ParallelDescriptor::Barrier();
    // ================================================================

    // PERFORM THE ANALYSIS ===========================================
    {
        // WRITE TO FILE ----------------------------------------------
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            fp << "p = " << inputs.dG[0].space_p << "\n";
        }
        // ------------------------------------------------------------

        {
            // DESTINATION FOLDER AND OUTPUT DATA ---------------------
            std::string dst_folder;
    
            const std::string dG_order = "p"+std::to_string(inputs.dG[0].space_p);
            const std::string dG_mesh = AMREX_D_TERM(std::to_string(inputs.mesh[0].n_cells[0]),+"x"+
                                                     std::to_string(inputs.mesh[0].n_cells[1]),+"x"+
                                                     std::to_string(inputs.mesh[0].n_cells[2]));

#if (PROBLEM == PROBLEM_SODS_TUBE_REF)
            const std::string problem = "PROBLEM_SodsTube_Reference";
#elif (PROBLEM == PROBLEM_SODS_TUBE)
            const std::string problem = "PROBLEM_SodsTube";
#elif (PROBLEM == PROBLEM_EB_COMPARISON)
            const std::string problem = "PROBLEM_EB_Comparison";
#endif
            dst_folder = amrex::DG::MakePath({".", "IBVP_"+std::to_string(AMREX_SPACEDIM)+"d/"+problem+"_"+dG_mesh+"_"+dG_order});

            if (amrex::ParallelDescriptor::IOProcessor())
            {
                if (!amrex::UtilCreateDirectory(dst_folder, 0755))
                {
                    amrex::CreateDirectoryFailed(dst_folder);
                }
            }
            amrex::ParallelDescriptor::Barrier();
            // --------------------------------------------------------

            // INIT IBVP DATA STRUCTURE -------------------------------
#if ((PROBLEM == PROBLEM_SODS_TUBE_REF) || (PROBLEM == PROBLEM_SODS_TUBE))
            const amrex::Vector<std::string> material_type = {"Ideal gas"};
            const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{1.4}};
#elif (PROBLEM == PROBLEM_EB_COMPARISON)
            const amrex::Vector<std::string> material_type = {"Ideal gas"};
            const amrex::Vector<amrex::Vector<amrex::Real>> material_properties = {{5.0/3.0}};
#endif
            GASDYNAMICS GD(material_type, material_properties);
            // --------------------------------------------------------

            // INIT DG OBJECT -----------------------------------------
            amrex::DG::DG<N_PHI, N_DOM, N_U> dG(inputs);
            // --------------------------------------------------------

            // INIT OUTPUT DATA INFORMATION ---------------------------
            dG.SetOutput(dst_folder, GD);
            // --------------------------------------------------------

            // INIT FIELDS' DATA WITH INITIAL CONDITIONS --------------
            dG.SetICs(GD);

            // SOME REPORTS
            dG.PrintMemoryReport();
            dG.mesh.CheckQuadratureRules(GD);

            // WRITE TO OUTPUT
            {
                int n = 0;
                amrex::Real time = 0.0;

                dG.PrintPointSolution("PointSolution", n, time, GD);

                if (inputs.plot_int > 0)
                {
                    dG.ExportMesh("Mesh", n, time, GD);
                    dG.ExportSolution("Solution", n, time, GD);
                }
            }
            // --------------------------------------------------------

            // START THE ANALYSIS (ADVANCE IN TIME) -------------------
            amrex::Print() << "# START OF THE ANALYSIS                                                " << std::endl;

            // VARIABLES
            int n;
            amrex::Real time, dt;
            amrex::Real start_clock_time_per_step, stop_clock_time_per_step, clock_time_per_time_step;

            // INIT CLOCK TIME PER STEP
            clock_time_per_time_step = 0.0;

            // ADVANCE IN TIME
            n = 0;
            time = 0.0;
            while ((time < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                start_clock_time_per_step = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = dG.Compute_dt(time+0.5*dt, GD);
                dt = std::min(time+dt, inputs.time.T)-time;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTING TIME STEP: n = " << n+1;
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << ", dt = " << dt << ", time = " << time+dt
                               << ", clock time per time step = " << clock_time_per_time_step << std::endl;

                // TIME STEP
                dG.TakeTimeStep(dt, time, GD);

                // UPDATE TIME AND STEP
                n += 1;
                time += dt;
                
                // COMPUTE ERROR
                if (std::abs(time/inputs.time.T-1.0) < 1.0e-12)
                {
                    // WRITE TO FILE
                    if (amrex::ParallelDescriptor::IOProcessor())
                    {
                        fp << "Ne = " << AMREX_D_TERM(inputs.mesh[0].n_cells[0], << "x" << inputs.mesh[0].n_cells[1], << "x" << inputs.mesh[0].n_cells[2]);
                    }
                }

                // WRITE TO OUTPUT
                dG.PrintPointSolution("PointSolution", n, time, GD);

                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(time/inputs.time.T-1.0) < 1.0e-12)))
                {
                    dG.ExportSolution("Solution", n, time, GD);
                }

                // CLOCK TIME PER TIME STEP TOC
                stop_clock_time_per_step = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(stop_clock_time_per_step, IOProc);

                clock_time_per_time_step = (clock_time_per_time_step*n+(stop_clock_time_per_step-start_clock_time_per_step))/(n+1);

                // WRITE TO FILE
                if ((std::abs(time/inputs.time.T-1.0) < 1.0e-12) && (amrex::ParallelDescriptor::IOProcessor()))
                {
                    fp << ", clock time per time step = " << std::scientific << std::setprecision(5) << std::setw(12) << clock_time_per_time_step << "\n";
                }
            }

            // END OF ANALYSIS
            amrex::Print() << "# END OF THE ANALYSIS                                                  " << std::endl;
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // CLOSE FILE =====================================================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        fp.close();
    }
    // ================================================================

    // TOC ============================================================
    stop_time = amrex::second();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
    // ================================================================

    // CLOSING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "# Time = " << std::scientific << std::setprecision(5) << std::setw(12) << (stop_time-start_time) << " s" << std::endl;
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