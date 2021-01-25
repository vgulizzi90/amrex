#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for an airfoil
// using a Finite Volume scheme and the implicitly defined mesh.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_Airfoil.H"
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
amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for an airfoil  " << std::endl;
amrex::Print() << "# using a Finite Volume scheme and the implicitly defined mesh.        " << std::endl;
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
    
    const std::string problem = "PROBLEM";

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
            output_folderpath = inputs.plot_filepath;
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
            amrex::Print() << std::endl;
            if (inputs.problem.int_params[0] == -1)
            {
                amrex::Print() << "# EMPTY DOMAIN TEST" << std::endl;
            }
            else if (inputs.problem.int_params[0] == 0)
            {
                amrex::Print() << "# ELLIPSE TEST" << std::endl;
            }
            else if (inputs.problem.int_params[0] == 4)
            {
                amrex::Print() << "# NACA 4-DIGIT AIRFOIL TEST" << std::endl;
            }
            else
            {
amrex::Print() << "HERE WE ARE - HEADER" << std::endl;
exit(-1);
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
            const amrex::Real len[AMREX_SPACEDIM] = {AMREX_D_DECL(inputs.space.hi[0]-inputs.space.lo[0],
                                                                  inputs.space.hi[1]-inputs.space.lo[1],
                                                                  inputs.space.hi[2]-inputs.space.lo[2])};

            amrex::Real volume;
            amrex::Real surface;

            if (inputs.problem.int_params[0] == -1)
            {
                volume = AMREX_D_TERM(len[0],*len[1],*len[2]);
                surface = 0.0;
            }
            else if (inputs.problem.int_params[0] == 0)
            {
                const amrex::Real a = inputs.problem.params[3];
                const amrex::Real b = inputs.problem.params[4];
                const amrex::Real h = (a-b)*(a-b)/((a+b)*(a+b));
                volume = 8.0-M_PI*a*b;
                surface = M_PI*(a+b)*(1.0+0.25*h+0.015625*h*h+0.00390625*h*h*h);
            }
            else if (inputs.problem.int_params[0] == 4)
            {
                volume = 64.0-0.081706;
                surface = 2.039548931147967;
            }
            else
            {
amrex::Print() << "HERE WE ARE - CHECK THE COMPUTED QUADRATURE RULES" << std::endl;
exit(-1);
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
        
        amrex::DG::ProjectInitialConditions(mesh, matfactory, DG_N_SOL, X, IG);

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
        amrex::Print() << "# START OF THE ANALYSIS" << std::endl;
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