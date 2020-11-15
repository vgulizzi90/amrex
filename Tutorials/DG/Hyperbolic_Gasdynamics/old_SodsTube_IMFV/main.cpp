#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the Sod's
// tube problem in tilted geometries.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_utils.H"
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
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = inputs.problem.params[9];
#endif

            const std::string mesh_info = AMREX_D_TERM(std::to_string(inputs.grid.n_cells[0]),+"x"+
                                                       std::to_string(inputs.grid.n_cells[1]),+"x"+
                                                       std::to_string(inputs.grid.n_cells[2]));
            const std::string diam_info = "D0"+std::to_string((int) std::round(diam*100));
            const std::string theta_info = "TH"+std::to_string((int) std::round(theta));
#if (AMREX_SPACEDIM == 3)
            const std::string phi_info = "PH"+std::to_string((int) std::round(phi));
#endif

#if (AMREX_SPACEDIM == 2)
            output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info+"_"+diam_info+"_"+theta_info});
#endif
#if (AMREX_SPACEDIM == 3)
            output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info+"_"+diam_info+"_"+theta_info+"_"+phi_info});
#endif

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
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = inputs.problem.params[9];
#endif
            
            const std::string diam_info = "0."+std::to_string((int) std::round(diam*100));
            const std::string theta_info = std::to_string((int) std::round(theta));
#if (AMREX_SPACEDIM == 3)
            const std::string phi_info = std::to_string((int) std::round(phi));
#endif

            amrex::Print() << std::endl;
#if (AMREX_SPACEDIM == 2)
            amrex::Print() << "# SOD'S TUBE TEST" << " - DIAMETER: " << diam_info << " - INCLINATION: " << theta_info << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
            amrex::Print() << "# SOD'S TUBE TEST" << " - DIAMETER: " << diam_info << " - AZIMUTH: " << theta_info << " - ELEVATION: " << phi_info << std::endl;
#endif
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
            amrex::DG::ExportBase_VTK(output_folderpath, "Levelset", n, inputs.time.n_steps,
                                      t, mesh.geom, mesh.std_elem, 1, mesh.PHI, "LS_");
            amrex::DG::ExportImplicitMesh_VTK(output_folderpath, "ImplicitMesh", n, inputs.time.n_steps,
                                              t, mesh);
        }
        // ============================================================

        // CHECK THE COMPUTED QUADRATURE RULES ========================
        {
            const amrex::Real diam = inputs.problem.params[7];
            const amrex::Real theta = inputs.problem.params[8]*M_PI/180.0;
            const amrex::Real cth = std::cos(theta);
            const amrex::Real sth = std::sin(theta);
            const amrex::Real tth = std::tan(theta);

            const amrex::Real Ay = 0.5*(1.0-diam/cth+tth);
            const amrex::Real By = 0.5*(1.0+diam/cth+tth);

            amrex::Real volume = 0.0;
            amrex::Real surface = 0.0;

            // We assume theta >= 0
            if (Ay > 1.0)
            {
#if (AMREX_SPACEDIM == 2)
                volume = diam/sth;
                surface = 2.0/sth;
#endif
#if (AMREX_SPACEDIM == 3)
                volume = 0.25*M_PI*diam*diam/sth;
                surface = M_PI*diam/sth;
#endif
            }
            else if (By < 1.0)
            {
#if (AMREX_SPACEDIM == 2)
                volume = diam/cth;
                surface = 2.0/cth;
#endif
#if (AMREX_SPACEDIM == 3)
                volume = 0.25*M_PI*diam*diam/cth;
                surface = M_PI*diam/cth;
#endif
            }
            else
            {
#if (AMREX_SPACEDIM == 2)
                const amrex::Real xc = 0.5*(1.0+1.0/tth-diam/sth);

                volume = 0.25*(2.0+2.0*diam/sth+2.0*diam/cth-(1.0+diam*diam)/(cth*sth));
                surface = 2.0*xc/cth;
#endif
#if (AMREX_SPACEDIM == 3)
#endif
            }

            mesh.CheckQuadratureRules(volume, surface);
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
        amrex::DG::ProjectInitialConditions(mesh, matfactory, N_SOL, X, IG);

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::DG::Export_VTK(output_folderpath, "Solution", n, inputs.time.n_steps,
                                  t, mesh, matfactory, N_SOL, X,
                                  IG);
        }

        if (X.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: main.cpp\n";
            msg += "| X contains nans.\n";
            amrex::Abort(msg);
        }
        // ============================================================

        // ADVANCE IN TIME ============================================
        amrex::Print() << "# START OF THE ANALYSIS" << std::endl;
        {
            int n = 0;
            amrex::Real t, dt;
            amrex::Real tps_start, tps_stop, tps;

            // INIT CLOCK TIME PER STEP
            tps = 0.0;

            // ADVANCE IN TIME
            n = 0;
            t = 0.0;
            dt = 0.0;
            while ((t < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                tps_start = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = amrex::DG::Compute_dt(t+0.5*dt, mesh, matfactory, N_SOL, X, IG);
                dt *= inputs.grid.CFL;
                dt = std::min(t+dt, inputs.time.T)-t;

                // TIME STEP
                amrex::DG::TakeTimeStep(dt, t, mesh, matfactory, N_SOL, X, IG);

                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // WRITE TO OUTPUT
                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(t/inputs.time.T-1.0) < 1.0e-12)))
                {
                    amrex::DG::Export_VTK(output_folderpath, "Solution", n, inputs.time.n_steps,
                                          t, mesh, matfactory, N_SOL, X,
                                          IG);
                }

                // CLOCK TIME PER TIME STEP TOC
                tps_stop = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(tps_stop, IOProc);

                tps = (tps*n+(tps_stop-tps_start))/(n+1);

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << dt << ", t = " << t
                               << ", clock time per time step = " << tps << std::endl;
            }

        }
        amrex::Print() << "# END OF THE ANALYSIS" << std::endl;
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