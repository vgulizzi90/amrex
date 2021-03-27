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

    // NUMBER OF GHOST ROWS
    const int ngr = 1;
    // ================================================================


    // VARIABLES ======================================================
    amrex::Real start_time, stop_time;

    // INPUTS
    amrex::DG::InputReader inputs;

    // SOLUTION MULTIFAB
    amrex::MultiFab X;

    // IBVP
    IDEAL_GAS IG(inputs.problem.int_params, inputs.problem.params);
    // ================================================================


    // DO THE ANALYSIS
    {
        // MAKE OUTPUT FOLDER =========================================
        {
            amrex::DG::IO::MakeFolder(inputs.plot_filepath);
        }
        // ============================================================


        // OPEN STATISTICS FILE =======================================
        std::ofstream fp;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            const std::string stats_filepath = amrex::DG::IO::MakePath({inputs.plot_filepath, "Stats.txt"});

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
            amrex::Print() << "# SUPERSONIC VORTEX TEST" << std::endl;
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
            amrex::DG::ExportImplicitMesh_VTK(inputs.plot_filepath, n, inputs.time.n_steps, "ImplicitMesh", t, mesh);
        }
        // ============================================================


        // CHECK THE COMPUTED QUADRATURE RULES ========================
        {
            const amrex::Real ri = inputs.problem.params[3];
            const amrex::Real ro = inputs.problem.params[4];

            amrex::Real volume;
            amrex::Real surface;

            volume = 0.25*M_PI*(ro*ro-ri*ri);
            surface = 0.5*M_PI*(ro+ri);

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

        // EVAL ERROR
        {
            amrex::Real dom_err;
            dom_err = amrex::DG::EvalErrorInfNorm(0.0, mesh, matfactory, DG_N_SOL, X, IG);
                      
            amrex::Print() << "| dom_err: " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << std::endl;

            if (amrex::ParallelDescriptor::IOProcessor())
            {
                fp << "| dom_err(t = 0): " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << "\n";
            }
        }

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::DG::Export_VTK(inputs.plot_filepath, n, inputs.time.n_steps, "Solution",
                                  t, mesh, matfactory, DG_N_SOL, X,
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
            const int p = inputs.dG.space_p;
            const int RK_order = p+1;

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
            while ((t < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                tps_start = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = amrex::DG::Hyperbolic::Explicit::Compute_dt(t+0.5*dt, mesh, matfactory, DG_N_SOL, X, IG);
                dt *= inputs.grid.CFL/(1.0+2.0*p);
                dt = std::min(t+dt, inputs.time.T)-t;

                // TIME STEP
                amrex::DG::Hyperbolic::Explicit::TakeRungeKuttaTimeStep(RK_order, dt, t,
                                                                        mesh, matfactory,
                                                                        DG_N_SOL, X,
                                                                        IG);

                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // EVAL ERROR
                {
                    dom_err = amrex::DG::EvalErrorInfNorm(t, mesh, matfactory, DG_N_SOL, X, IG);
                }

                // WRITE TO OUTPUT
                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(t/inputs.time.T-1.0) < 1.0e-12)))
                {
                    amrex::DG::Export_VTK(inputs.plot_filepath, n, inputs.time.n_steps, "Solution",
                                          t, mesh, matfactory, DG_N_SOL, X,
                                          IG);
                }

                // CLOCK TIME PER TIME STEP TOC
                tps_stop = amrex::second();
                amrex::ParallelDescriptor::ReduceRealMax(tps_stop, IOProc);

                tps = (tps*(n-1)+(tps_stop-tps_start))/(1.0*n);
                eta = (inputs.time.T-t)/dt*tps;

                // REPORT TO SCREEN
                amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
                amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                               << dt << ", t = " << t << ", dom_err = " << dom_err
                               << ", tts [s] = " << tps 
                               << ", eta = " << amrex::DG::IO::Seconds2HoursMinutesSeconds(eta) << std::endl;
            }

            amrex::Print() << "| dom_err: " << std::scientific << std::setprecision(5) << std::setw(12) << dom_err << std::endl;

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