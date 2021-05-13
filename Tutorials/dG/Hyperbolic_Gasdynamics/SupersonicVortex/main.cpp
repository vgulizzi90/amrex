#include <AMReX_Print.H>
#include <AMReX_dG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we solve the Gasdynamics equations for the
// supersonic vortex problem using the discontinuous Galerkin method
// and the embedded-geometry approach.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_SupersonicVortex.H"
// ====================================================================
// ####################################################################



// ACTUAL MAIN PROGRAM ################################################
void main_main()
{
    // PARAMETERS =====================================================
    // ================================================================


    // VARIABLES ======================================================
    amrex::dG::TimeKeeper time_keeper;

    // USER-DEFINED AMR
    supersonic_vortex::AMR amr;
    
    // RESTART INFO
    int n0;
    amrex::Real t0;

    // ERROR
    amrex::Real err_old, err_new, err_norm;
    // ================================================================


    // OPENING ========================================================
    // TIC -----------
    time_keeper.tic();
    // ---------------

    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# AMREX & DG PROJECT                                                   " << std::endl;
    amrex::Print() << "# Author: Vincenzo Gulizzi (vgulizzi@lbl.gov)                          " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# SUMMARY:                                                             " << std::endl;
    amrex::Print() << "# In this tutorial, we solve the Gasdynamics equations for the         " << std::endl;
    amrex::Print() << "# supersonic vortex problem using the discontinuous Galerkin method    " << std::endl;
    amrex::Print() << "# and the embedded-geometry approach.                                  " << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
    amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================


    // INITIAL CONDITIONS =============================================
    amr.init();

    // RESTART INFO
    n0 = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
    t0 = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);

    // CHECK QUADRATURE
    {
        const amrex::Real ri = amr.ibvp.r_inner;
        const amrex::Real ro = amr.ibvp.r_outer;

        amrex::Real volume;
        amrex::Real surface;

        volume = 0.25*M_PI*(ro*ro-ri*ri);
        surface = 0.5*M_PI*(ro+ri);

        amr.check_quadrature_rules(volume, surface);
    }

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

        err_old = 0.0;
        amr.eval_error(t, err_new, err_norm);
        err_new = err_new/err_norm;

        amrex::Print() << "INITIAL ERROR REPORT:" << std::endl;
        amrex::Print() << "| err(t = " << t << "): " << std::scientific << std::setprecision(5) << std::setw(12) << err_new << std::endl;
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

            // SWAP OLD AND NEW ERROR
            std::swap(err_old, err_new);

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
            amr.eval_error(t, err_new, err_norm);
            err_new = err_new/err_norm;

            // TIME STEP TOC
            time_keeper.toc();

            // CLOCK TIME PER TIME STEP / ETA
            ct = time_keeper.get_elapsed_time_in_seconds();
            ct_avg = (ct_avg*(n-n0-1)+ct)/(1.0*(n-n0));
            eta = amrex::min((amr.inputs.time.T-t)/dt, 1.0*(amr.inputs.time.n_steps-n))*ct_avg;

            // REPORT TO SCREEN
            amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
            amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                            << dt << ", t = " << t << ", err = " << err_new
                            << ", ct [s] = " << ct_avg 
                            << ", eta = " << amrex::dG::seconds_to_hms(eta) << std::endl;
        }
    }
    amrex::Print() << "# END OF THE ANALYSIS" << std::endl;
    // ================================================================


    // CLOSING ========================================================
    // TOC -----------
    time_keeper.toc();
    // ---------------

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