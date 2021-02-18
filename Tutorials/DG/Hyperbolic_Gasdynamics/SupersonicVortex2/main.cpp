#include <tr1/cmath>
#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the
// supersonic vortex problem using the discontinuous Galerkin method
// and the implicitly defined mesh.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_SupersonicVortex.H"
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
    amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for the         " << std::endl;
    amrex::Print() << "# supersonic vortex problem using the discontinuous Galerkin method    " << std::endl;
    amrex::Print() << "# and the implicitly defined mesh.                                     " << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
    amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================


    // THIS TEST IS SUPPOSED TO RUN ONLY FOR AMREX_SPACEDIM = 2 =======
#if (AMREX_SPACEDIM != 2)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: main.cpp\n";
        msg += "| This problem can be run only for AMREX_SPACEDIM = 2.\n";
        amrex::Abort(msg);
    }
#endif
    // ================================================================


    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    // ================================================================


    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;
    int RK_order;

    AMR_IDEAL_GAS amr;
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
            amrex::Print() << "# SUPERSONIC VORTEX TEST" << std::endl;
        }
        // ============================================================


        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================


        // SET INITIAL CONDITIONS =====================================
        amr.Init();

        // CHECK QUADRATURE
        {
            const amrex::Real ri = amr.inputs.problem.params[3];
            const amrex::Real ro = amr.inputs.problem.params[4];

            amrex::Real volume;
            amrex::Real surface;

            volume = 0.25*M_PI*(ro*ro-ri*ri);
            surface = 0.5*M_PI*(ro+ri);

            amr.CheckQuadratureRules(volume, surface);
        }

        // EVAL ERROR
        {
            const amrex::Real t = 0.0;
            amrex::Real dom_err;
            
            dom_err = amrex::DG::AMR::EvalError(t, DG_N_SOL, amr, amr.IG);
            dom_err = std::sqrt(dom_err);
                      
            amrex::Print() << "| dom_err: " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << std::endl;

            if (amrex::ParallelDescriptor::IOProcessor())
            {
                fp << "| dom_err(t = 0): " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << "\n";
            }
        }

        // EXPORT
        if (amr.inputs.Plot())
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::DG::Export_VTK(amr.inputs.plot_filepath, n, amr.inputs.time.n_steps,
                                  t, DG_N_SOL, amr, amr.IG);
        }

        if (amr.contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: main.cpp\n";
            msg += "| X contains nans.\n";
            amrex::Abort(msg);
        }
        // ============================================================


        // GET THE ORDER OF THE RUNGE-KUTTA SCHEME ====================
        {
            int p;

            // LEVEL 0
            p = amr.inputs.dG[0].space_p;
            RK_order = (p == 0) ? p+2 : p+1;

            // FINER LEVELS 
            for (int lev = 1; lev <= amr.maxLevel(); ++lev)
            {
                if (amr.LevelIsValid(lev))
                {
                    p = amr.inputs.dG[lev].space_p;
                    RK_order = std::max(RK_order, (p == 0) ? p+2 : p+1);
                }
            }
        }
        // ============================================================


        // ADVANCE IN TIME ============================================
        amrex::Print() << "# START OF THE ANALYSIS" << std::endl;
        {
            int n;
            amrex::Real t, dt, dom_err;
            amrex::Real tps_start, tps_stop, tps, eta;

            // INIT CLOCK TIME PER STEP AND ESTIMATED TIME
            tps = 0.0;
            eta = 0.0;

            // ADVANCE IN TIME
            n = 0;
            t = 0.0;
            dt = 0.0;
            while ((t < amr.inputs.time.T*(1.0-1.0e-12)) && (n < amr.inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                tps_start = amrex::second();

                // COMPUTE NEXT TIME STEP
                // CFL condition is taken care of inside Compute_dt
                dt = amrex::DG::Hyperbolic::Explicit::Compute_dt(t+0.5*dt, DG_N_SOL, amr, amr.IG);
                dt = std::min(t+dt, amr.inputs.time.T)-t;

                // TIME STEP
                amrex::DG::Hyperbolic::Explicit::TakeRungeKuttaTimeStep(RK_order, dt, t,
                                                                        amr.refRatio(), amr.meshes, amr.matfactories,
                                                                        DG_N_SOL, amr.Xs, amr.masks, amr.IG);

                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // EVAL ERROR
                {
                    dom_err = amrex::DG::AMR::EvalError(t, DG_N_SOL, amr, amr.IG);
                    dom_err = std::sqrt(dom_err);
                }
                
                // WRITE TO OUTPUT
                if (amr.inputs.Plot(n, t))
                {
                    amrex::DG::Export_VTK(amr.inputs.plot_filepath, n, amr.inputs.time.n_steps,
                                          t, DG_N_SOL, amr, amr.IG);
                }

                // CLOCK TIME PER TIME STEP TOC
                tps_stop = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(tps_stop, IOProc);

                tps = (tps*n+(tps_stop-tps_start))/(n+1);
                eta = (amr.inputs.time.T-t)/dt*tps;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << dt << ", t = " << t << ", dom_err = " << dom_err
                               << ", tts [s] = " << tps 
                               << ", eta = " << amrex::DG::IO::Seconds2HoursMinutesSeconds(eta) << std::endl;
            }

            if (amrex::ParallelDescriptor::IOProcessor())
            {
                fp << "| dom_err(t = T): " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << "\n";
                fp << "| clock time per time step: " << std::scientific << std::setprecision(5) << std::setw(12) << tps << " s\n";
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