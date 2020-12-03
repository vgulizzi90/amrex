#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the advection equations in structured
// grids using the discontinuous Galerkin method.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_Vortex.H"
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
amrex::Print() << "# In this tutorial, we solve the advection equations in structured     " << std::endl;
amrex::Print() << "# grids using the discontinuous Galerkin method.                       " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // PARAMETERS =====================================================
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    
    const std::string problem = "PROBLEM_Vortex";

    // AUXILIARY TABLES TO TEST THE DIFFERENT POLYNOMIAL ORDERS
    const amrex::Vector<int> table_p = {0, 1, 2, 3};
    const int n_p = table_p.size();

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
    ADVECTION ADV;
    // ================================================================

    // DO THE ANALYSES
    for (int ip = 0; ip < n_p; ++ip)
    {
        // SET THE ORDER IN SPACE AND TIME ============================
        inputs.dG.space_p = table_p[ip];
        inputs.dG.time_p = table_p[ip];
        // ============================================================
        
        // MAKE OUTPUT FOLDER =========================================
        {
            const std::string mesh_info = "m"+AMREX_D_TERM(std::to_string(inputs.grid.n_cells[0]),+"x"+
                                                           std::to_string(inputs.grid.n_cells[1]),+"x"+
                                                           std::to_string(inputs.grid.n_cells[2]));
            const std::string p_info = "p"+std::to_string(inputs.dG.space_p);

            output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info+"_"+p_info});

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
            amrex::Print() << std::endl;
            amrex::Print() << "# ADVECTION #" << ip << " - p: " << inputs.dG.space_p << std::endl;
        }
        // ============================================================

        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================

        // MAKE THE IMPLICIT-MESH =====================================
        amrex::DG::ImplicitMesh mesh(inputs);

        mesh.MakeFromScratch(ADV);

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
            const amrex::Real volume = 1.0;
            const amrex::Real surface = 0.0;

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
        
        amrex::DG::ProjectInitialConditions(mesh, matfactory, DG_N_SOL, X, ADV);
        
        {
            const amrex::Real err = amrex::DG::EvalError(0.0, mesh, matfactory, DG_N_SOL, X, ADV);
            amrex::Print() << "| err: " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;
        }

        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            amrex::DG::Export_VTK(output_folderpath, n, inputs.time.n_steps, "Solution",
                                  t, mesh, matfactory, DG_N_SOL, X,
                                  ADV);
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

            const amrex::Real dt_max_change = 1.1;

            int n = 0;
            amrex::Real t, dt, dt_old;
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
                dt_old = dt;
                dt = amrex::DG::Hyperbolic::Explicit::Compute_dt(t+0.5*dt, mesh, matfactory, DG_N_SOL, X, ADV);
                dt *= inputs.grid.CFL/(1.0+2.0*p);
                dt = std::min(t+dt, inputs.time.T)-t;
                if (n > 0)
                {
                    dt = std::min(dt, dt_old*dt_max_change);
                }

                // TIME STEP
                amrex::DG::Hyperbolic::Explicit::TakeRungeKuttaTimeStep(RK_order, dt, t,
                                                                        mesh, matfactory,
                                                                        DG_N_SOL,
                                                                        X,
                                                                        ADV);
                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // WRITE TO OUTPUT
                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(t/inputs.time.T-1.0) < 1.0e-12)))
                {
                    amrex::DG::Export_VTK(output_folderpath, n, inputs.time.n_steps, "Solution",
                                          t, mesh, matfactory, DG_N_SOL, X,
                                          ADV);
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
                               << ", clock time per time step = " << tps 
                               << ", estimated remaining time = " << eta << std::endl;
            }

        }
        amrex::Print() << "# END OF THE ANALYSIS" << std::endl;
        // ============================================================

        // TOC ========================================================
        stop_time = amrex::second();
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
        // ============================================================

        {
            const amrex::Real err = amrex::DG::EvalError(inputs.time.T, mesh, matfactory, DG_N_SOL, X, ADV);
            amrex::Print() << "| err: " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;
        }

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