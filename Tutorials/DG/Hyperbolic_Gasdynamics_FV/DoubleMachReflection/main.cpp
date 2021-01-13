#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the double
// Mach reflection problem using a Finite Volume scheme and the
// implicitly defined mesh.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_DoubleMachReflection.H"
#include "../IBVP_utils.H"
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
amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for the double  " << std::endl;
amrex::Print() << "# Mach reflection problem using a Finite Volume scheme and the         " << std::endl;
amrex::Print() << "# implicitly defined mesh.                                             " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

#if (AMREX_SPACEDIM != 2)
    // THIS TEST IS SUPPOSED TO RUN ONLY FOR AMREX_SPACEDIM = 2 =======
    std::string msg;
    msg  = "\n";
    msg += "ERROR: main.cpp\n";
    msg += "| This problem can be run only for AMREX_SPACEDIM = 2.\n";
    amrex::Abort(msg);
    // ================================================================
#endif

    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    
    const std::string problem = "PROBLEM_DoubleMachReflection";

    // NUMBER OF GHOST ROWS
    const int ngr = 1;
    // ================================================================

    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;
    std::string output_folderpath;

    // INPUTS
    amrex::DG::InputReader inputs;

    // SOLUTION MULTIFAB
    amrex::MultiFab X;

    // IBVP
    IDEAL_GAS IG(inputs.problem.int_params, inputs.problem.params);
    // ================================================================

    // CHECK INPUT ====================================================
    if (inputs.dG.space_p != 0)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: main.cpp\n";
        msg += "| This problem must be run with inputs.dG.space_p = 0.\n";
        amrex::Abort(msg);
    }
    // ================================================================

    // DO THE ANALYSIS
    {
        // MAKE OUTPUT FOLDER =========================================
        {
            const amrex::Real theta = inputs.problem.params[9];

            const std::string mesh_info = "m"+AMREX_D_TERM(std::to_string(inputs.grid.n_cells[0]),+"x"+
                                                           std::to_string(inputs.grid.n_cells[1]),+"x"+
                                                           std::to_string(inputs.grid.n_cells[2]));
            const std::string theta_info = "TH"+std::to_string((int) std::round(theta));

            if (inputs.problem.int_params[0] == 0)
            {
                output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info});
            }
            else
            {
                output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info+"_"+theta_info});
            }

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
        }
        // ============================================================

        // HEADER =====================================================
        {
            const amrex::Real theta = inputs.problem.params[9];
            
            const std::string theta_info = std::to_string((int) std::round(theta));

            amrex::Print() << std::endl;
            if (inputs.problem.int_params[0] == 0)
            {
                amrex::Print() << "# DOUBLE MACH REFLECTION TEST" << std::endl;
            }
            else
            {
                amrex::Print() << "# DOUBLE MACH REFLECTION TEST" << " - p: " << inputs.dG.space_p << " - INCLINATION: " << theta_info << std::endl;
            }
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
            amrex::DG::ExportImplicitMesh_VTK(output_folderpath, n, inputs.time.n_steps, "ImplicitMesh",
                                              t, mesh);
        }
        // ============================================================

        // CHECK THE COMPUTED QUADRATURE RULES ========================
        {
            const amrex::Real theta = (inputs.problem.params[9])*M_PI/180.0;
            const amrex::Real Px = inputs.problem.params[8];
            const amrex::Real L = inputs.space.hi[0]-Px;
            const amrex::Real H = inputs.space.hi[1];
            const amrex::Real alpha = std::atan(H/L);

            amrex::Real volume;
            amrex::Real surface;

            if (inputs.problem.int_params[0] == 0)
            {
                volume = (Px+L)*H;
                surface = 0.0;
            }
            else
            {
                if (theta < alpha)
                {
                    const amrex::Real h = L*std::tan(theta);

                    volume = (Px+L)*H-0.5*L*h;
                    surface = L/std::cos(theta);
                }
                else
                {
                    const amrex::Real l = H/std::tan(theta);

                    volume = 0.5*(Px+Px+l)*H;
                    surface = H/std::sin(theta);
                }
            }

            mesh.CheckQuadratureRules(volume, surface);
        }
        // ============================================================

        // INIT THE MATRIX FACTORY ====================================
        amrex::DG::MatrixFactory matfactory(inputs);

        matfactory.EvalMassMatrices(mesh);
        // ============================================================

        // INIT MULTIFAB ==============================================
        {
            const int p = inputs.dG.space_p;
            const int X_n_comp = DG_N_SOL*(AMREX_D_PICK(1+p, (1+p)*(1+p), (1+p)*(1+p)*(1+p)));
            X.define(mesh.cc_ba, mesh.dm, X_n_comp, ngr);
        }
        // ============================================================

        // SET INITIAL CONDITIONS =====================================
        amrex::Print() << "# COMPUTING PROJECTED INITIAL CONDITIONS " << std::endl;
        
        amrex::FV::SetInitialConditions(mesh, matfactory, X, IG);

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::FV::Export_VTK(output_folderpath, n, inputs.time.n_steps, "Solution",
                                  t, mesh, matfactory, X,
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
            amrex::Real tps_start, tps_stop, tps, eta;

            // INIT CLOCK TIME PER STEP AND ESTIMATED TIME
            tps = 0.0;
            eta = 0.0;

            // ADVANCE IN TIME
            n = 0;
            t = 0.0;
            dt = 0.0;
            while ((t < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                tps_start = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = amrex::FV::Compute_dt(t+0.5*dt, mesh, matfactory, X, IG);
                dt *= inputs.grid.CFL;
                dt = std::min(t+dt, inputs.time.T)-t;

                // TIME STEP
                amrex::FV::TakeRungeKuttaTimeStep(dt, t,
                                                  mesh, matfactory,
                                                  X,
                                                  IG);
                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // WRITE TO OUTPUT
                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(t/inputs.time.T-1.0) < 1.0e-12)))
                {
                    amrex::FV::Export_VTK(output_folderpath, n, inputs.time.n_steps, "Solution",
                                          t, mesh, matfactory, X,
                                          IG);
                }

                // CLOCK TIME PER TIME STEP TOC
                tps_stop = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(tps_stop, IOProc);

                tps = (tps*n+(tps_stop-tps_start))/(n+1);
                eta = (inputs.time.T-t)/dt*tps;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << dt << ", t = " << t
                               << ", tts [s] = " << tps 
                               << ", eta = " << amrex::DG::IO::Seconds2HoursMinutesSeconds(eta) << std::endl;
            }

            fp << "| clock time per time step: " << std::scientific << std::setprecision(5) << std::setw(12) << tps << " s\n";

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