#include <AMReX_PlotFileUtil.H>
#include <AMReX_Geometry.H>

#include <AMReX_Print.H>

#include <AMReX_DG.H>

#include "inputs_reader.H"

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we are solving an advection equation of the form
//
// phi_{,t}+(u_i phi)_{,i} = 0.
//
// where phi is the advected field and u = {u_i} is a divergence-free
// velocity field.
// The problem is subjected to:
// i) either periodic or inflow boundary conditions
// ii) initial conditions: phi(0,x) = phi0(x)
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include <IBVP_Linear_Advection.H>
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
amrex::Print() << "# In this tutorial, we are solving an advection equation of the form   " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# rho_{,t}+(a_i rho)_{,i} = 0                                          " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "# where rho is the advected field and a = {a_i} is a divergence-free   " << std::endl;
amrex::Print() << "# velocity field.                                                      " << std::endl;
amrex::Print() << "# The problem is subjected to:                                         " << std::endl;
amrex::Print() << "# i) either periodic or inflow boundary conditions                     " << std::endl;
amrex::Print() << "# ii) initial conditions: rho(0,x) = rho0(x)                           " << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
amrex::Print() << "#                                                                      " << std::endl;
amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================

    // GENERAL VARIABLES =============
    amrex::Real start_time, stop_time;
    // ===============================

    // INPUTS DATA STRUCTURE
    inputs_struct inputs;

    // REPORT TO SCREEN
    inputs.print();
    // =====================

    // TIC ======================
    start_time = amrex::second();
    // ==========================

    // PHYSICAL BOX ===================================================
    amrex::RealBox real_box(inputs.space.prob_lo.data(),
                            inputs.space.prob_hi.data());
    // ================================================================

    // INIT IBVP DATA STRUCTURE =======================================
    LINADV LinAdv;
    // ================================================================

    // RANGE OF DG DEGREES p AND NUMBER OF MESH ELEMENTS ==============
    amrex::Vector<int> p_degrees = {0, 1, 2, 3};
    amrex::Vector<amrex::Vector<int>> n_mesh_elements = 
    {
        {8, 16, 32},
        {8},
        {8},
        {8}
    };
    // ================================================================

    // OPEN FILE WHERE TO WRITE THE HP-CONVERGENCE RESULTS ============
    std::ofstream fp;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        fp.open("hp-Convergence.txt");
        fp << "HP-CONVERGENCE ANALYSIS\n";
    }
    amrex::ParallelDescriptor::Barrier();
    // ================================================================

    // PERFORM THE HP-CONVERGENCE ANALYSIS ============================
    const int n_p_degrees = p_degrees.size();
    for (int ip = 0; ip < n_p_degrees; ++ip)
    {
        const int phi_space_p = 1;
        const int space_p = p_degrees[ip];
        const int space_q = std::max(phi_space_p+2, space_p+2);
        const int time_p = space_p;
amrex::Print() << "# SETTING DG ORDER ====================================================" << std::endl;
amrex::Print() << "# phi_space_p = " << phi_space_p << std::endl;
amrex::Print() << "# space_p = " << space_p << std::endl;
amrex::Print() << "# space_q = " << space_q << std::endl;
amrex::Print() << "# time_p = " << time_p << std::endl;

        // WRITE TO FILE ----------------------------------------------
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            fp << "p = " << space_p << "\n";
        }
        amrex::ParallelDescriptor::Barrier();
        // ------------------------------------------------------------
        
        const int n_n_mesh_elements = n_mesh_elements[ip].size();
        for (int iNe = 0; iNe < n_n_mesh_elements; ++iNe)
        {
            const int Ne = n_mesh_elements[ip][iNe];
amrex::Print() << "# - Setting number of elements Ne = " << Ne << std::endl;

amrex::Print() << "# - Initializing geometry data structures " << std::endl;
            // INIT GEOMETRY DATA STRUCTURES --------------------------
            amrex::IntVect n_cells(AMREX_D_DECL(Ne, Ne, Ne));
            amrex::Box indices_box({AMREX_D_DECL(0, 0, 0)}, n_cells-1);

            amrex::BoxArray ba;
            ba.define(indices_box);
            ba.maxSize(inputs.mesh.max_grid_size);

            amrex::Geometry geom;
            geom.define(indices_box, &real_box, inputs.space.coord_sys, inputs.space.is_periodic.data());
            // --------------------------------------------------------

            // DISTRIBUTION MAPPING -----------------------------------
            amrex::DistributionMapping dm(ba);
            // --------------------------------------------------------

amrex::Print() << "# - Initializing dG data structures " << std::endl;
            // INIT DG DATA STRUCTURES --------------------------------
            amrex::DG::ImplicitGeometry<N_PHI, N_DOM> iGeom(indices_box, real_box, ba, dm, geom, phi_space_p, space_q);
            amrex::DG::MatrixFactory<N_PHI, N_DOM> MatFactory(indices_box, real_box, ba, dm, geom, space_p, time_p, space_q, 0);
            amrex::DG::DG<N_PHI, N_DOM, N_U> dG("Hyperbolic", "Runge-Kutta");
            dG.InitData(iGeom, MatFactory);
            // --------------------------------------------------------

            // INIT FIELDS' DATA WITH INITIAL CONDITIONS --------------
            iGeom.ProjectDistanceFunctions(LinAdv);
            iGeom.EvalImplicitMesh(LinAdv, false);
            MatFactory.Eval(iGeom);
            dG.SetICs(iGeom, MatFactory, LinAdv);
            // --------------------------------------------------------

            // START THE ANALYSIS (ADVANCE IN TIME) -------------------
            amrex::Print() << std::endl;

            amrex::Print() << " -- START OF THE ANALYSIS " << std::endl;

            // VARIABLES
            int n;
            amrex::Real time, dt;

            // ADVANCE IN TIME
            n = 0;
            time = 0.0;
            while ((time < inputs.time.T*(1.0-1.0e-12)) && (n < inputs.mesh.n_time_steps))
            {
                // COMPUTE NEXT TIME STEP
                dt = dG.Compute_dt(time+0.5*dt, iGeom, MatFactory, LinAdv);
                dt = std::min(time+dt, inputs.time.T)-time;

                // TIME STEP
                dG.TakeTimeStep_Hyperbolic(dt, time, iGeom, MatFactory, LinAdv);

                // UPDATE TIME AND STEP
                n += 1;
                time += dt;

                // COMPUTE ERROR
                if (std::abs(time/inputs.time.T-1.0) < 1.0e-12)
                {
                    amrex::Real err;
                    err = dG.EvalError(time, iGeom, MatFactory, LinAdv);
                    amrex::Print() << "| Error: " << std::scientific << std::setprecision(5) << std::setw(12) << err << std::endl;

                    // WRITE TO FILE
                    if (amrex::ParallelDescriptor::IOProcessor())
                    {
                        fp << "Ne = " << Ne << ", error = " << std::scientific << std::setprecision(5) << std::setw(12) << err << "\n";
                    }
                    amrex::ParallelDescriptor::Barrier();
                }
            }

            amrex::Print() << " -- END OF THE ANALYSIS " << std::endl;

            amrex::Print() << std::endl;
            // --------------------------------------------------------

        }
amrex::Print() << "# =====================================================================" << std::endl;
    }
    // ================================================================

    // CLOSE FILE ===============================
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        fp.close();
    }
    // ==========================================

    // TOC ============================================================
    stop_time = amrex::second();
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
    // ================================================================

    // CLOSING ========================================================
amrex::Print() << "#######################################################################" << std::endl;
amrex::Print() << "# END OF TUTORIAL                                                      " << std::endl;
amrex::Print() << "# Time = " << stop_time-start_time << " s" << std::endl;
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