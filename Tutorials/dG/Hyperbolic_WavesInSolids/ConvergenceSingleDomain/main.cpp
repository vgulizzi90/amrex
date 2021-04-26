#include <AMReX_Print.H>
#include <AMReX_dG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we study the hp-convergence performance of
// discontinuous Galerkin methods for the solution of the elastic wave
// equation over domains with embedded geometries.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_SingleDomain.H"
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
    SingleDomain::AMR amr;
    
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
    amrex::Print() << "# In this tutorial, we study the hp-convergence performance of         " << std::endl;
    amrex::Print() << "# discontinuous Galerkin methods for the solution of the elastic wave  " << std::endl;
    amrex::Print() << "# equation over domains with embedded geometries.                      " << std::endl;
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
        const amrex::Real * prob_lo = amr.Geom(0).ProbLo();
        const amrex::Real * prob_hi = amr.Geom(0).ProbHi();
        const amrex::Real prob_len[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_hi[0]-prob_lo[0],
                                                                   prob_hi[1]-prob_lo[1],
                                                                   prob_hi[2]-prob_lo[2])};
        const amrex::Real prob_volume = AMREX_D_TERM(prob_len[0],*prob_len[1],*prob_len[2]);

        amrex::Real volume;
        amrex::Real surface;

        if (amr.ibvp.problem_params.shape.compare("none") == 0)
        {
            surface = 0.0;
            volume = prob_volume;
        }
        else if (amr.ibvp.problem_params.shape.compare("circle") == 0)
        {
            const amrex::Real r = amr.ibvp.level_set.params[AMREX_SPACEDIM];
            surface = AMREX_D_PICK(0.0, 2.0*M_PI*r, 4.0*M_PI*r*r);
            volume = prob_volume-(AMREX_D_PICK(2.0*r, M_PI*r*r, 4.0/3.0*M_PI*r*r*r));
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: main.cpp\n";
            msg += "| Unexpected geometry shape.\n";
            msg += "| shape: "+amr.ibvp.problem_params.shape+".\n";
            amrex::Abort(msg);
        }

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
/*

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
*/

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