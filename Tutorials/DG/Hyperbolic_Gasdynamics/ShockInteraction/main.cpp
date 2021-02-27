#include <tr1/cmath>
#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for shocks
// interacting with solid bodies using the discontinuous Galerkin
// method and the implicitly defined mesh.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_ShockInteraction.H"
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
    amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for shocks      " << std::endl;
    amrex::Print() << "# interacting with solid bodies using the discontinuous Galerkin       " << std::endl;
    amrex::Print() << "# method and the implicitly defined mesh.                              " << std::endl;
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
    int RK_order;

    AMR_IDEAL_GAS amr;
    // ================================================================


    // DO THE ANALYSIS
    {
        // MAKE OUTPUT FOLDER =========================================
        if (amr.inputs.restart > 0)
        {
            // Do nothing
        }
        else
        {
            amrex::DG::IO::MakeFolder(amr.inputs.plot_filepath);
        }
        // ============================================================


        // OPEN STATISTICS FILE =======================================
        std::ofstream fp;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            time_t date_and_time = time(0);
            char * date_and_time_ = ctime(&date_and_time);
            
            std::string stats_filepath;
            if (amr.inputs.restart > 0)
            {
                stats_filepath = amrex::DG::IO::MakePath({amr.inputs.plot_filepath, "Stats_"+std::to_string(amr.inputs.restart)+".txt"});
            }
            else
            {
                stats_filepath = amrex::DG::IO::MakePath({amr.inputs.plot_filepath, "Stats.txt"});
            }
            
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
            amrex::Print() << "# SOD'S TUBE TEST" << std::endl;
        }
        // ============================================================


        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================


        // SET INITIAL CONDITIONS =====================================
        amr.Init();

        // CHECK QUADRATURE
        {
            const amrex::Real * prob_lo = amr.Geom(0).ProbLo();
            const amrex::Real * prob_hi = amr.Geom(0).ProbHi();
            const amrex::Real len[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_hi[0]-prob_lo[0],
                                                                  prob_hi[1]-prob_lo[1],
                                                                  prob_hi[2]-prob_lo[2])};

            amrex::Real volume;
            amrex::Real surface;

            if (amr.IG.eb_flag == -1)
            {
                volume = AMREX_D_TERM(len[0],*len[1],*len[2]);
                surface = 0.0;
            }
            else if (amr.IG.eb_flag == 0)
            {
                const amrex::Real r = amr.IG.params[5];

                volume = AMREX_D_TERM(len[0],*len[1],*len[2])-AMREX_D_PICK(2.0*r, 0.5*M_PI*r*r, 0.25*(4.0/3.0)*M_PI*r*r*r);
                surface = AMREX_D_PICK(0.0, M_PI*r, M_PI*r*r);
            }
            else
            {
amrex::Print() << "main.cpp - CHECK THE COMPUTED QUADRATURE RULES" << std::endl;
exit(-1);
            }

            amr.CheckQuadratureRules(volume, surface);
        }

        // EXPORT
        if (amr.inputs.Plot())
        {
            const int n = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
            const amrex::Real t = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);
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
            const int max_level = amr.maxLevel();

            int n;
            amrex::Real t, dt;
            amrex::Real tps_start, tps_stop, tps, eta;

            // INIT CLOCK TIME PER STEP AND ESTIMATED TIME
            tps = 0.0;
            eta = 0.0;

            // ADVANCE IN TIME
            n = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
            t = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);
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
                
                // WRITE TO OUTPUT
                if (amr.inputs.Plot(n, t))
                {
                    amrex::DG::Export_VTK(amr.inputs.plot_filepath, n, amr.inputs.time.n_steps,
                                          t, DG_N_SOL, amr, amr.IG);
                }
                // WRITE CHECKPOINT
                if (amr.inputs.Checkpoint(n, t))
                {
                    amrex::DG::WriteCheckpoint(n, t, amr);
                }

                // REGRID
                if ((max_level > 0) && (amr.inputs.regrid_int > 0))
                {
                    if (n%amr.inputs.regrid_int == 0)
                    {
                        amr.regrid(0, t);
                        amr.UpdateMasks();
                    }
                }

                // CLOCK TIME PER TIME STEP TOC
                tps_stop = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(tps_stop, IOProc);

                {
                    const int m = n-((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
                    tps = (tps*(m-1)+(tps_stop-tps_start))/(1.0*m);
                }
                eta = (amr.inputs.time.T-t)/dt*tps;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << dt << ", t = " << t
                               << ", tts [s] = " << tps 
                               << ", eta = " << amrex::DG::IO::Seconds2HoursMinutesSeconds(eta) << std::endl;
            }

            if (amrex::ParallelDescriptor::IOProcessor())
            {
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