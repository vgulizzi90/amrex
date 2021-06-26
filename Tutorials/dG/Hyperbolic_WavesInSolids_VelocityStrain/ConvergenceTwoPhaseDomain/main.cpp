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
#include "IBVP_TwoPhaseDomain.H"
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


    // PARAMETERS =====================================================
    // ================================================================


    // VARIABLES ======================================================
    amrex::dG::TimeKeeper time_keeper;

    // USER-DEFINED AMR
    two_phase_domain::AMR amr;
    
    // RESTART INFO
    int n0;
    amrex::Real t0;

    // ERROR
    amrex::Real err_L_inf, err_L_inf_norm, err_L_2, err_L_2_norm;
    // ================================================================


    // TIC ============================================================
    time_keeper.tic();
    // ================================================================


    // TEST THE RIEMANN SOLVER ========================================
    /*
    {
        const amrex::Real th = M_PI/3.0;
        const amrex::Real ph = M_PI/6.0;
        const amrex::Real cth = std::cos(th);
        const amrex::Real sth = std::sin(th);
        const amrex::Real cph = std::cos(ph);
        const amrex::Real sph = std::sin(ph);
        const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*sph, sth*sph, cph)};
        //const amrex::Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(1.0, 0.0, 0.0)};
        amrex::Real An[N_VS*N_VS], wAn[N_VS], vAn[N_VS*N_VS], m_U[N_VS], p_U[N_VS], NFn[N_VS];
        amrex::Print() << "un: "; amrex::dG::io::print_reals(AMREX_SPACEDIM, un); amrex::Print() << std::endl;
        elastic_solid::eval_An_compact_c(amr.ibvp.density[0], &(amr.ibvp.c[0][0]), un, An);
        amrex::Print() << "An: " << std::endl;
        amrex::dG::io::print_real_array_2d(N_VS, N_VS, An);

        // Evaluate eigenvalues of An
        {
            char jobvl = 'N';
            char jobvr = 'V';
            int n = N_VS;
            amrex::Real An_copy[N_VS*N_VS];
            amrex::Real wAn_im[N_VS];
            amrex::Real work[N_VS*N_VS];
            int lwork = N_VS*N_VS;
            int info;

            std::copy(An, An+N_VS*N_VS, An_copy);
            
            dgeev_(&jobvl, &jobvr, &n, An_copy, &n, wAn, wAn_im, nullptr, &n, vAn, &n, work, &lwork, &info);
            if (info != 0)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: main.cpp\n";
                msg += "| Something went wrong in the computation of the eigenvalues of An.\n";
                amrex::Abort(msg);
            }
        }

        amrex::Print() << "wAn: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, N_VS, wAn);
        amrex::Print() << "vAn: " << std::endl;
        amrex::dG::io::print_real_array_2d(N_VS, N_VS, vAn);

#if (AMREX_SPACEDIM == 3)
        m_U[V1] = 1.0;
        m_U[V2] = 1.0;
        m_U[V3] = 1.0;
        m_U[S11] = 1.0;
        m_U[S22] = 1.0;
        m_U[S33] = 1.0;
        m_U[S23] = 1.0;
        m_U[S13] = 1.0;
        m_U[S12] = 1.0;

        p_U[V1] = 0.0;
        p_U[V2] = 0.0;
        p_U[V3] = 0.0;
        p_U[S11] = 0.0;
        p_U[S22] = 0.0;
        p_U[S33] = 0.0;
        p_U[S23] = 0.0;
        p_U[S13] = 0.0;
        p_U[S12] = 0.0;
#endif
        
        amrex::Print() << "m_U: "; amrex::dG::io::print_reals(N_VS, m_U); amrex::Print() << std::endl;
        amrex::Print() << "p_U: "; amrex::dG::io::print_reals(N_VS, p_U); amrex::Print() << std::endl;
        
        elastic_solid::eval_NFn_Riemann_solver(amr.ibvp.density[0], &(amr.ibvp.c[0][0]), un, m_U, p_U, NFn);

        amrex::Print() << "NFn    : "; amrex::dG::io::print_reals(N_VS, NFn); amrex::Print() << std::endl;

        return;
    }
    */
    // ================================================================


    // INITIAL CONDITIONS =============================================
    amr.init();

    // RESTART INFO
    n0 = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
    t0 = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);
    
    // CHECK QUADRATURE
    {
        amrex::Real volume[N_PHASES];
        amrex::Real surface[N_PHASES];

#if (AMREX_SPACEDIM == 2)
        volume[0] = 0.6132111710287008;
        volume[1] = 1.0-volume[0];
        surface[0] = 3.2001998801421565;
        surface[1] = surface[0];
#endif
#if (AMREX_SPACEDIM == 3)
        volume[0] = 0.7049227561311002;
        volume[1] = 1.0-volume[0];
        surface[0] = 3.5087190200183827;
        surface[1] = surface[0];
#endif

        amr.check_quadrature_rules(N_PHASES, volume, surface);
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

        amr.eval_error(t, err_L_inf, err_L_inf_norm, err_L_2, err_L_2_norm);
        err_L_inf = err_L_inf/err_L_inf_norm;
        err_L_2 = std::sqrt(err_L_2/err_L_2_norm);

        amrex::Print() << "INITIAL ERROR REPORT:" << std::endl;
        amrex::Print() << "| err_L_inf(t = " << t << "): " << std::scientific << std::setprecision(5) << std::setw(12) << err_L_inf << std::endl;
        amrex::Print() << "|   err_L_2(t = " << t << "): " << std::scientific << std::setprecision(5) << std::setw(12) << err_L_2 << std::endl;
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
            amr.eval_error(t, err_L_inf, err_L_inf_norm, err_L_2, err_L_2_norm);
            err_L_inf = err_L_inf/err_L_inf_norm;
            err_L_2 = std::sqrt(err_L_2/err_L_2_norm);

            // TIME STEP TOC
            time_keeper.toc();

            // CLOCK TIME PER TIME STEP / ETA
            ct = time_keeper.get_elapsed_time_in_seconds();
            ct_avg = (ct_avg*(n-n0-1)+ct)/(1.0*(n-n0));
            eta = amrex::min((amr.inputs.time.T-t)/dt, 1.0*(amr.inputs.time.n_steps-n))*ct_avg;

            // REPORT TO SCREEN
            amrex::Print() << "| COMPUTED TIME STEP: n = "+std::to_string(n)+", dt = ";
            amrex::Print() << std::scientific << std::setprecision(5) << std::setw(12)
                            << dt << ", t = " << t << ", err_L_inf = " << err_L_inf << ", err_L_2 = " << err_L_2
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