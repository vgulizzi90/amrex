//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_InputReader.H
 * \brief A file containing data structures used for reading input files.
*/

#include <AMReX_ParmParse.H>

#ifndef AMREX_DG_INPUT_READER_H_
#define AMREX_DG_INPUT_READER_H_

namespace amrex
{
namespace DG
{

/**
 * \brief Inputs for Partial Differential Equations information
 *
*/
struct PDEsInput
{
    std::string description;

    PDEsInput()
    :
    description{""}
    {}
};

/**
 * \brief Inputs for Space information
 *
*/
struct SpaceInput
{
    Vector<int> is_periodic;
    int coord_sys;
    Vector<Real> lo, hi;

    SpaceInput()
    :
    is_periodic{AMREX_D_DECL(0, 0, 0)},
    coord_sys{0},
    lo{AMREX_D_DECL( 0.0,  0.0,  0.0)},
    hi{AMREX_D_DECL(+1.0, +1.0, +1.0)}
    {}
};

/**
 * \brief Inputs for Time information
 *
*/
struct TimeInput
{
    Real T;
    int n_steps;

    TimeInput()
    :
    T{0.0},
    n_steps{0}
    {}
};

/**
 * \brief Inputs for Grid information
 *
*/
struct GridInput
{
    IntVect n_cells, max_box_size;
    Real CFL;
    int space_p, time_p;
    int space_q, time_q;

    GridInput()
    :
    n_cells(AMREX_D_DECL(0, 0, 0)),
    max_box_size(AMREX_D_DECL(0, 0, 0)),
    CFL{1.0},
    space_p{-1},
    time_p{-1},
    space_q{-1},
    time_q{-1}
    {}
};

/**
 * \brief Inputs for ImplicitMesh information
 *
*/
struct ImplicitMeshInput
{
    int space_q, space_q_ie, time_q;
    Real VFT;

    ImplicitMeshInput()
    :
    space_q{-1},
    space_q_ie{-1},
    time_q{-1},
    VFT{0.33}
    {}
};

/**
 * \brief Inputs for discontinuous Galerkin information
 *
*/
struct DGInput
{
    int space_p, time_p;

    DGInput()
    :
    space_p{-1},
    time_p{-1}
    {}
};

/**
 * \brief Inputs for Problem information
 *
*/
struct ProblemInput
{
    Vector<int> int_params;
    Vector<Real> params;

    ProblemInput()
    {}
};

/**
 * \brief Base version of the input reader
 *
*/
struct InputReaderBase
{
    // DATA MEMBERS ###################################################
    int verbosity;

    int plot_int;
    int checkpoint_int;
    // ################################################################

    // CONSTRUCTOR ####################################################
    InputReaderBase()
    :
    verbosity{1},
    plot_int{-1},
    checkpoint_int{-1}
    {
        // VARIABLES ==================================================
        amrex::ParmParse pp;
        // ============================================================
        
        // VERBOSITY ==================================================
        pp.query("verbosity", this->verbosity);
        // ============================================================

        // OUTPUT FILE INTERVAL =======================================
        pp.query("plot_int", this->plot_int);
        // ============================================================

        // CHECKPOINT FILE INTERVAL ===================================
        pp.query("checkpoint_int", this->checkpoint_int);
        // ============================================================
    }
    // ################################################################
};

/**
 * \brief Input reader for single-level single-patch problem
 *
*/
struct InputReader
:
public InputReaderBase
{
    // DATA MEMBERS ###################################################
    SpaceInput space;
    TimeInput time;
    GridInput grid;
    ImplicitMeshInput implicit_mesh;
    DGInput dG;
    ProblemInput problem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    InputReader()
    :
    InputReaderBase()
    {
        // SPACE ======================================================
        {
            ParmParse pp("space");

            pp.queryarr("is_periodic", this->space.is_periodic);
            pp.query("coord_sys", this->space.coord_sys);
            pp.getarr("lo", this->space.lo);
            pp.getarr("hi", this->space.hi);
        }
        // ============================================================

        // TIME =======================================================
        {
            ParmParse pp("time");

            pp.get("T", this->time.T);
            pp.get("n_steps", this->time.n_steps);
        }
        // ============================================================

        // GRID =======================================================
        {
            Vector<int> aux;

            ParmParse pp("grid");

            if (pp.queryarr("n_cells", aux))
            {
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                {
                    this->grid.n_cells[dim] = aux[dim];
                    this->grid.max_box_size[dim] = aux[dim];
                }
            }

            if (pp.queryarr("max_box_size", aux))
            {
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                {
                    this->grid.max_box_size[dim] = aux[dim];
                }
            }

            pp.query("CFL", this->grid.CFL);

            pp.query("space_p", this->grid.space_p);

            this->grid.time_p = this->grid.space_p;
            pp.query("time_p", this->grid.time_p);

            this->grid.space_q = this->grid.space_p;
            pp.query("space_q", this->grid.space_q);
            
            this->grid.time_q = this->grid.space_p;
            pp.query("time_q", this->grid.time_q);
        }
        // ============================================================

        // MESH =======================================================
        {
            Vector<int> aux;

            ParmParse pp("implicit_mesh");

            pp.query("space_q", this->implicit_mesh.space_q);

            this->implicit_mesh.space_q_ie = this->implicit_mesh.space_q;
            pp.query("space_q_ie", this->implicit_mesh.space_q_ie);

            pp.query("volume_fraction_threshold", this->implicit_mesh.VFT);
        }
        // ============================================================

        // DISCONTINUOUS GALERKIN METHODS =============================
        {
            ParmParse pp("dG");

            if (pp.contains("space_p"))
            {
                pp.get("space_p", this->dG.space_p);
                
                this->dG.time_p = this->dG.space_p;
                pp.query("time_p", this->dG.time_p);
            }
        }
        // ============================================================

        // PROBLEM ====================================================
        {
            ParmParse pp("problem");

            pp.queryarr("int_params", this->problem.int_params);
            pp.queryarr("params", this->problem.params);
        }
        // ============================================================
    }
    // ################################################################
};

} // namespace DG
} // namespace amrex

#endif