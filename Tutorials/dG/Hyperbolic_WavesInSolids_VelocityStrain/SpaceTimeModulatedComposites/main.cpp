#include <AMReX_Print.H>
#include <AMReX_dG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we study the performance of discontinuous Galerkin
// methods for the solution of the elastic wave equation in
// space-time-modulated composites.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_SpaceTimeModulatedComposite.H"
// ====================================================================
// ####################################################################


// ACTUAL MAIN PROGRAM ################################################
void main_main()
{
    // OPENING ========================================================
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# AMREX & DG PROJECT                                                   " << std::endl;
    amrex::Print() << "# Author: Vincenzo Gulizzi (vgulizzi@lbl.gov)                          " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# SUMMARY:                                                             " << std::endl;
    amrex::Print() << "# In this tutorial, we study the performance of discontinuous Galerkin " << std::endl;
    amrex::Print() << "# methods for the solution of the elastic wave equation in             " << std::endl;
    amrex::Print() << "# space-time-modulated composites.                                     " << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
    amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================


    // PARAMETERS =====================================================
    // ================================================================


    // VARIABLES ======================================================
    amrex::dG::TimeKeeper time_keeper;

    // USER-DEFINED AMR
    space_time_modulated_composite::AMR amr;
    
    // RESTART INFO
    int n0;
    amrex::Real t0;

    // ERROR
    amrex::Real err, err_norm;
    // ================================================================


    // TIC ============================================================
    time_keeper.tic();
    // ================================================================


    // INITIAL CONDITIONS =============================================
    amr.init();

    // RESTART INFO
    n0 = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
    t0 = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);

    // EXPORT
    {
        const int n = n0;
        const amrex::Real t = t0;

        if (amr.inputs.plot(n, t))
        {
            amr.make_step_output_folder(n, t);
            amr.export_solution(n, "solution", t);
        }
    }

    // EVAL ERROR
    {
        const amrex::Real t = t0;

        amr.eval_error(t, err, err_norm);
        err = err/err_norm;

        amrex::Print() << "INITIAL ERROR REPORT:" << std::endl;
        amrex::Print() << "| err(t = " << t << "): " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;
    }

    // CHECKS
    {
        amr.check_space_time_boxes();
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Print() << "# START OF THE ANALYSIS" << std::endl;
    {
        // TIME STEP / TIME / TIME INCREMENT
        int n;
        amrex::Real t, dt;
        
        // CLOCK TIME PER TIME STEP / ETA
        amrex::Real ct, ct_avg, eta;

        // ADVANCE IN TIME
        n = n0;
        t = t0;
        dt = 0.0;
        ct_avg = 0.0;
        eta = 0.0;
        while (amr.advance_in_time_continues(n, t))
        {
            // TIME STEP TIC
            time_keeper.tic();

            // COMPUTE TIME INCREMENT
            dt = amr.eval_dt(t);
            dt = amrex::min(t+dt, amr.inputs.time.T)-t;

            // TAKE TIME STEP
            amr.take_time_step(t, dt);

            // UPDATE TIME STEP
            n += 1;
            t += dt;

            // EXPORT
            if (amr.inputs.plot(n, t))
            {
                amr.make_step_output_folder(n, t);
                amr.export_solution(n, "solution", t);
            }

            // EVAL ERROR
            amr.eval_error(t, err, err_norm);
            err = err/err_norm;

            // TIME STEP TOC
            time_keeper.toc();

            // CLOCK TIME PER TIME STEP / ETA
            ct = time_keeper.get_elapsed_time_in_seconds();
            ct_avg = (ct_avg*(n-n0-1)+ct)/(1.0*(n-n0));
            eta = amrex::min((amr.inputs.time.T-t)/dt, 1.0*(amr.inputs.time.n_steps-n))*ct_avg;

            // REPORT TO SCREEN
            amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
            amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                            << dt << ", t = " << t << ", err = " << err
                            << ", ct [s] = " << ct_avg 
                            << ", eta = " << amrex::dG::seconds_to_hms(eta) << std::endl;
        }
    }
    amrex::Print() << "# END OF THE ANALYSIS" << std::endl;
    // ================================================================


    // TOC ============================================================
    time_keeper.toc();
    // ================================================================


    // CLOSING ========================================================
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# END OF THE TUTORIAL                                                  " << std::endl;
    amrex::Print() << "# Elapsed time: " << time_keeper.get_elapsed_time_in_hms() << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================
}
// ####################################################################



// DUMMY MAIN #########################################################
int main(int argc, char * argv[])
{
    amrex::Initialize(argc, argv);
    
    main_main();
    
    amrex::Finalize();

    return 0;
}
// ####################################################################