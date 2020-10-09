//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_InputReader.H
 * \brief A file containing data structures used for reading input files.
*/

#ifndef AMREX_DG_INPUT_READER_H_
#define AMREX_DG_INPUT_READER_H_

#include <AMReX_ParmParse.H>

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
 * \brief Inputs for Mesh information
 *
*/
struct MeshInput
{
    IntVect n_cells, max_grid_size;
    Real CFL;

    MeshInput()
    :
    n_cells(AMREX_D_DECL(0, 0, 0)),
    max_grid_size(AMREX_D_DECL(0, 0, 0)),
    CFL{0.0}
    {}
};

/**
 * \brief Inputs for Problem information
 *
*/
struct ProblemInput
{
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
    MeshInput mesh;
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

        // MESH =======================================================
        {
            Vector<int> aux;

            ParmParse pp("mesh");

            pp.getarr("n_cells", aux);
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                this->mesh.n_cells[dim] = aux[dim];
                this->mesh.max_grid_size[dim] = aux[dim];
            }

            if (pp.queryarr("max_grid_size", aux))
            {
                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
                {
                    this->mesh.max_grid_size[dim] = aux[dim];
                }
            }

            pp.get("CFL", this->mesh.CFL);
        }
        // ============================================================

        // PROBLEM ====================================================
        {
            ParmParse pp("problem");

            pp.getarr("params", this->problem.params);
        }
        // ============================================================
    }
    // ################################################################
};

} // namespace DG
} // namespace amrex

#endif