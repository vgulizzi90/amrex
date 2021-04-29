#include <AMReX_Print.H>
#include <AMReX_dG.H>

// PDES INFORMATION ###################################################
// SUMMARY:
// In this tutorial, we test different libraries used to generate the
// high-order quadrature rules for embedded-boundary discontinuous
// Galerkin methods.
//
// ####################################################################
// SELECT SET OF PDES =================================================
#include "IBVP_QuadratureRules.H"
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
    amrex::dG::InputReaderBase inputs;

    // BOX ARRAY AND DISTRIBUTION MAPPING
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;

    // USER-DEFINED AMR
    QuadratureRules::IBVP ibvp;
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
    amrex::Print() << "# Galerkin methods.                                                    " << std::endl;
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
    }

    // MESH
    {
        const amrex::Real t = 0.0;
        
        amrex::ParmParse pp;
        pp.get("which_embedded_geometry", which_embedded_geometry);
        
        mesh.read_input_file();
        
        if (mesh.uses_embedded_geometries())
        {
            if (mesh.uses_level_set() && (which_embedded_geometry.compare("custom") == 0))
            {
                QuadratureRules::CustomLevelSet level_set;
                mesh.make_from_scratch_by_level_set(t, geom, ba, dm, ibvp, level_set);
            }

#if (AMREX_SPACEDIM == 2)
            if (mesh.uses_level_set() && (which_embedded_geometry.compare("circle") == 0))
            {
                QuadratureRules::Circle level_set;
                mesh.make_from_scratch_by_level_set(t, geom, ba, dm, ibvp, level_set);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (mesh.uses_level_set() && (which_embedded_geometry.compare("sphere") == 0))
            {
                QuadratureRules::Sphere level_set;
                mesh.make_from_scratch_by_level_set(t, geom, ba, dm, ibvp, level_set);
            }
#endif
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
#if (AMREX_SPACEDIM == 2)
            if (mesh.uses_level_set() && (which_embedded_geometry.compare("circle") == 0))
            {
                QuadratureRules::Circle level_set;
                //const amrex::Real * c = level_set.c;
                const amrex::Real r = level_set.r;

                volume[0] = M_PI*r*r;
                surface[0] = 2.0*M_PI*r;

                volume[1] = prob_volume-volume[0];
                surface[1] = surface[0];

                mesh.check_quadrature_rules(geom, n_domains, volume.data(), surface.data());
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (mesh.uses_level_set() && (which_embedded_geometry.compare("sphere") == 0))
            {
                QuadratureRules::Sphere level_set;
                //const amrex::Real * c = level_set.c;
                const amrex::Real r = level_set.r;

                volume[0] = (4.0/3.0)*M_PI*r*r*r;
                surface[0] = 4.0*M_PI*r*r;

                volume[1] = prob_volume-volume[0];
                surface[1] = surface[0];

                mesh.check_quadrature_rules(geom, n_domains, volume.data(), surface.data());
            }
#endif
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


    // EXPORT =========================================================
    {
        const int n = 0;
        const std::string level_folderpath = inputs.get_output_folderpath();
        const std::string level_step_folderpath = inputs.get_output_step_folderpath(n);
        const std::string step_string = inputs.get_step_string(n);

        inputs.make_step_output_folder(n);
        amrex::dG::mesh_io::export_to_VTK(level_folderpath, level_step_folderpath, step_string, "mesh",
                                          geom, mesh, ibvp);
        amrex::dG::mesh_io::export_quadrature_points_to_VTK(level_folderpath, level_step_folderpath, step_string, "mesh_quadrature",
                                                            geom, mesh, ibvp);
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