#include <AMReX_Print.H>
#include <AMReX_dG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we test different libraries used to generate the
// high-order quadrature rules for embedded-boundary discontinuous
// Galerkin methods with mapped geometries.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_MappedGeometry.H"
// ====================================================================
// ####################################################################



// ACTUAL MAIN PROGRAM ################################################
void main_main()
{
    // PARAMETERS =====================================================
    // ================================================================


    // VARIABLES ======================================================
    amrex::dG::TimeKeeper time_keeper;

    // INPUTS
    amrex::Geometry geom;
    amrex::IntVect n_cell, max_grid_size;
    std::string which_embedded_geometry;
    amrex::dG::Mesh mesh;
    amrex::dG::Solution solution;
    amrex::dG::InputReaderBase inputs;

    // BOX ARRAY AND DISTRIBUTION MAPPING
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;

    // USER-DEFINED IBVP
    MappedGeometry::IBVP ibvp;

    // MULTIFABS STORING THE SOLUTION AND THE LEVEL SET
    amrex::MultiFab X, L;
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
    amrex::Print() << "# In this tutorial, we test different libraries used to generate the   " << std::endl;
    amrex::Print() << "# high-order quadrature rules for embedded-boundary discontinuous      " << std::endl;
    amrex::Print() << "# Galerkin methods with mapped geometries.                             " << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    amrex::Print() << "# The selected space dimension at compile time is                      " << std::endl;
    amrex::Print() << "# AMREX_SPACEDIM = " << AMREX_SPACEDIM << std::endl;
    amrex::Print() << "#                                                                      " << std::endl;
    amrex::Print() << "#######################################################################" << std::endl;
    // ================================================================


    // READ INPUTS ====================================================
    {
        amrex::ParmParse pp;
        amrex::Vector<int> iv;

        // GEOMETRY
        geom.Setup();
        pp.getarr("n_cell", iv);
        AMREX_D_TERM
        (
            n_cell[0] = iv[0];,
            n_cell[1] = iv[1];,
            n_cell[2] = iv[2];
        )
        pp.getarr("max_grid_size", iv);
        AMREX_D_TERM
        (
            max_grid_size[0] = iv[0];,
            max_grid_size[1] = iv[1];,
            max_grid_size[2] = iv[2];
        )
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(n_cell-1));
        geom.define(domain);

        // BOX ARRAY AND DISTRIBUTION MAPPING
        ba.define(domain);
        ba.maxSize(max_grid_size);
        dm.define(ba);

        // MESH
        mesh.read_input_file();

        // SOLUTION
        solution.read_input_file();
    }

    // INIT THE MULTIFAB THAT WILL CONTAIN THE PROJECTED LEVEL SET
    {
        if (mesh.uses_embedded_geometries())
        {
            if (mesh.uses_projected_level_set())
            {
                mesh.init_projected_level_set_multifabs(ba, dm, L);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: main.cpp\n";
            msg += "| At this moment embedded geometries must be active.\n";
            amrex::Abort(msg);
        }
    }

    // MESH
    {
        bool mesh_has_been_generated;

        const amrex::Real t = 0.0;
        
        amrex::ParmParse pp;
        pp.get("which_embedded_geometry", which_embedded_geometry);

        mesh_has_been_generated = false;
        if (mesh.uses_embedded_geometries())
        {
            if (mesh.uses_projected_level_set() && (which_embedded_geometry.compare("custom") == 0))
            {
                MappedGeometry::CustomLevelSet level_set;
                mesh.project_space_level_set(t, geom, L, level_set);
                mesh.make_from_scratch_by_projected_level_set(t, geom, ba, dm, L, ibvp);

                mesh_has_been_generated = true;
            }

            if (!mesh_has_been_generated)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: main.cpp\n";
                msg += "| Mesh was not generated.\n";
                amrex::Abort(msg);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: main.cpp\n";
            msg += "| At this moment embedded geometries must be active.\n";
            amrex::Abort(msg);
        }
    }

    // POST-PROCESSING
    inputs.read_input_file();
    // ================================================================


    // CHECK QUADRATURE ===============================================
    {
        const amrex::Real * prob_lo = geom.ProbLo();
        const amrex::Real * prob_hi = geom.ProbHi();
        const amrex::Real prob_len[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_hi[0]-prob_lo[0],
                                                                   prob_hi[1]-prob_lo[1],
                                                                   prob_hi[2]-prob_lo[2])};
        const amrex::Real prob_volume = AMREX_D_TERM(prob_len[0],*prob_len[1],*prob_len[2]);

        const int n_domains = (ibvp.use_two_phases ? 2 : 1);
        amrex::Vector<amrex::Real> volume(2);
        amrex::Vector<amrex::Real> surface(2);

        if (mesh.uses_embedded_geometries())
        {
            
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: main.cpp\n";
            msg += "| At this moment embedded geometries must be active.\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // SOLUTION =======================================================
    {
        const amrex::Real t = 0.0;
        const bool FV_is_active = ((solution.params.space_p == 0) && solution.params.FV_is_active);

        solution.make_from_scratch(t, geom, mesh, ibvp);
        solution.eval_space_mass_matrix(geom, mesh, ibvp);
        if (FV_is_active)
        {
            solution.eval_space_centroids(geom, mesh, ibvp, FV_is_active);
        }

        solution.init_space_solution_multifabs(mesh, X, ibvp);
        solution.project_initial_conditions(geom, mesh, X, ibvp, FV_is_active);
    }
    // ================================================================


    // EXPORT =========================================================
    {
        const int n = 0;
        const std::string level_folderpath = inputs.get_output_folderpath();
        const std::string level_step_folderpath = inputs.get_output_step_folderpath(n);
        const std::string step_string = inputs.get_step_string(n);
        const amrex::Real t = 0.0;

        inputs.make_step_output_folder(n);
        amrex::dG::mesh_io::export_to_VTK(level_folderpath, level_step_folderpath, step_string, "mesh",
                                          geom, mesh, ibvp);
        amrex::dG::mesh_io::export_quadrature_points_to_VTK(level_folderpath, level_step_folderpath, step_string, "mesh_quadrature",
                                                            geom, mesh, ibvp);

        const bool FV_is_active = ((solution.params.space_p == 0) && solution.params.FV_is_active);

        if (mesh.uses_projected_level_set())
        {
            if (FV_is_active)
            {
            }
            else
            {
                amrex::dG::solution_io::export_using_projected_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, "solution",
                                                                                t, geom, mesh, solution, X, L, ibvp);
            }
        }
    }
    // ================================================================


    // AUXILIARY EVALUATION FOR MAPPED GEOMETRIES =====================
    /*
    {
        const amrex::Real xi[2] = {0.08660254038, 0.15};
        amrex::Real x0[3], un[3], x[3], gl[9];

        ibvp.eval_shell_map_and_unit_normal(xi, x0, un);
        x[0] = x0[0]+0.05*un[0];
        x[1] = x0[1]+0.05*un[1];
        x[2] = x0[2]+0.05*un[2];
        ibvp.eval_map_covariant_basis(xi, gl);

amrex::Print() << "xi: "; amrex::dG::io::print_real_array_2d(1, 2, xi);
amrex::Print() << "x0: "; amrex::dG::io::print_real_array_2d(1, 3, x0);
amrex::Print() << "un: "; amrex::dG::io::print_real_array_2d(1, 3, un);
amrex::Print() << "x: "; amrex::dG::io::print_real_array_2d(1, 3, x);
amrex::Print() << "gl: " << std::endl; amrex::dG::io::print_real_array_2d(3, 3, gl);

    }
    */
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