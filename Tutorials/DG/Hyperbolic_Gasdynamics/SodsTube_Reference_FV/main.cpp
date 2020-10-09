#include <AMReX_PlotFileUtil.H>

#include <AMReX_Print.H>

#include <AMReX_DG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the Sod's
// tube problem.
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
amrex::Print() << "# tube problem.                                                        " << std::endl;
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

    // AUXILIARY TABLES TO TEST THE DIFFERENT ORIENTATIONS
#if (AMREX_SPACEDIM == 2)
    const amrex::Real table_hi[AMREX_SPACEDIM*AMREX_SPACEDIM] = {1.0, 0.1, 0.1, 1.0};
    const int table_n_cells[AMREX_SPACEDIM*AMREX_SPACEDIM] = {64, 4, 4, 64};
#endif
#if (AMREX_SPACEDIM == 3)
    const amrex::Real table_hi[AMREX_SPACEDIM*AMREX_SPACEDIM] = {1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0};
    const int table_n_cells[AMREX_SPACEDIM*AMREX_SPACEDIM] = {64, 4, 4, 4, 64, 4, 4, 4, 64};
#endif

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

    // STANDARD ELEMENT
    amrex::DG::StandardRectangle<AMREX_SPACEDIM> std_elem;

    // GEOMETRY, BOXARRAY, DISTRIBUTION MAPPING
    amrex::RealBox rbx;
    amrex::Box ibx;
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;

    // SOLUTION MULTIFAB
    amrex::MultiFab X;

    // IBVP
    IDEAL_GAS IG(gamma, inputs.problem.params);
    // ================================================================

    // WE TEST ALL POSSIBLE ORIENTATIONS OF THE TUBE
    for (int ori = 0; ori < AMREX_SPACEDIM; ++ori)
    {
        // SET SOD'S TUBE ORIENTATION =================================
        IG.set_ori(ori);
        // ============================================================

        // SET THE PROBLEM SIZE AND THE NUMBER OF CELLS ===============
#if (AMREX_SPACEDIM == 2)
        inputs.space.hi[0] = table_hi[ori];
        inputs.space.hi[1] = table_hi[ori+AMREX_SPACEDIM];
        inputs.mesh.n_cells[0] = table_n_cells[ori];
        inputs.mesh.n_cells[1] = table_n_cells[ori+AMREX_SPACEDIM];
#endif
#if (AMREX_SPACEDIM == 3)
        inputs.space.hi[0] = table_hi[ori];
        inputs.space.hi[1] = table_hi[ori+AMREX_SPACEDIM];
        inputs.space.hi[2] = table_hi[ori+2*AMREX_SPACEDIM];
        inputs.mesh.n_cells[0] = table_n_cells[ori];
        inputs.mesh.n_cells[1] = table_n_cells[ori+AMREX_SPACEDIM];
        inputs.mesh.n_cells[2] = table_n_cells[ori+2*AMREX_SPACEDIM];
#endif
        // ============================================================

        // MAKE OUTPUT FOLDER =========================================
        {
            const std::string mesh_info = AMREX_D_TERM(std::to_string(inputs.mesh.n_cells[0]),+"x"+
                                                       std::to_string(inputs.mesh.n_cells[1]),+"x"+
                                                       std::to_string(inputs.mesh.n_cells[2]));
            output_folderpath = amrex::DG::IO::MakePath({".", problem+"_"+mesh_info});
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
        const char ori_description = ((ori == 0) ? 'X' : ((ori == 1) ? 'Y' : 'Z'));
        amrex::Print() << std::endl;
        amrex::Print() << "# SOD'S TUBE TEST #" << ori << " - ORIENTATION: " << ori_description << std::endl;
        // ============================================================

        // TIC ========================================================
        start_time = amrex::second();
        // ============================================================

        // INIT THE STANDARD ELEMENT ==================================
        {
            AMREX_D_TERM
            (
                const amrex::Real dx1 = (inputs.space.hi[0]-inputs.space.lo[0])/inputs.mesh.n_cells[0];,
                const amrex::Real dx2 = (inputs.space.hi[1]-inputs.space.lo[1])/inputs.mesh.n_cells[1];,
                const amrex::Real dx3 = (inputs.space.hi[2]-inputs.space.lo[2])/inputs.mesh.n_cells[2];
            )
            const amrex::Real dx[AMREX_SPACEDIM] = {AMREX_D_DECL(dx1, dx2, dx3)};

            std_elem.Init(dx);
        }
        // ============================================================

        // INIT GEOMETRY AND DISTRIBUTION MAPPING =====================
        rbx.setLo(inputs.space.lo.data());
        rbx.setHi(inputs.space.hi.data());
        
        ibx.setSmall({AMREX_D_DECL(0, 0, 0)});
        ibx.setBig(inputs.mesh.n_cells-1);
        geom.define(ibx, &rbx, inputs.space.coord_sys, inputs.space.is_periodic.data());

        // BOX ARRAY
        ba.define(ibx);
        ba.maxSize(inputs.mesh.max_grid_size);

        // DISTRIBUTION MAPPING
        dm.define(ba);
        // ============================================================

        // INIT MULTIFAB ==============================================
        X.define(ba, dm, X_n_comp, ngr);
        // ============================================================

        // SET INITIAL CONDITIONS =====================================
        ProjectInitialConditionsOverGrid(geom, std_elem, N_SOL, X, IG);
        
        // WRITE TO OUTPUT
        if (inputs.plot_int > 0)
        {
            const int n = 0;
            const amrex::Real t = 0.0;
            Export_VTK(output_folderpath, "Solution", n, inputs.time.n_steps,
                       t, geom, std_elem, N_SOL, X,
                       IG);
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
            while ((t < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.time.n_steps))
            {
                // CLOCK TIME PER TIME STEP TIC
                tps_start = amrex::second();

                // COMPUTE NEXT TIME STEP
                dt = amrex::DG::Compute_dt(t+0.5*dt, geom, std_elem, N_SOL, X, IG);
                dt *= inputs.mesh.CFL;
                dt = std::min(t+dt, inputs.time.T)-t;

                // TIME STEP
                amrex::DG::TakeTimeStep(dt, t, geom, N_SOL, X, IG);

                // UPDATE TIME STEP
                n += 1;
                t += dt;

                // WRITE TO OUTPUT
                if ((inputs.plot_int > 0) && ((n%inputs.plot_int == 0) || (std::abs(t/inputs.time.T-1.0) < 1.0e-12)))
                {
                    Export_VTK(output_folderpath, "Solution", n, inputs.time.n_steps,
                               t, geom, std_elem, N_SOL, X,
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