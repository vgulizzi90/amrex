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
    SupersonicVortex::AMR amr;
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

    // CHECK QUADRATURE
    {
        const amrex::Real ri = amr.ibvp.problem_params.r_inner;
        const amrex::Real ro = amr.ibvp.problem_params.r_outer;

        amrex::Real volume;
        amrex::Real surface;

        volume = 0.25*M_PI*(ro*ro-ri*ri);
        surface = 0.5*M_PI*(ro+ri);

        amr.check_quadrature_rules(volume, surface);
    }

    // EVAL ERROR
    {

    }

    // EXPORT
    {
        const int n = ((amr.inputs.restart > 0) ? amr.inputs.restart : 0);
        const amrex::Real t = ((amr.inputs.restart > 0) ? amr.inputs.restart_time : 0.0);

        amr.export_mesh(n, t, "mesh");
    }
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