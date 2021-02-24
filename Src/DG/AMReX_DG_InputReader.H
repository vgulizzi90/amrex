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
    bool use_sub_cell_limiter;

    std::string plot_filepath;
    int plot_int;

    std::string checkpoint_filepath;
    int checkpoint_int;

    int regrid_int;

    int restart;
    Real restart_time;
    // ################################################################

    // CONSTRUCTOR ####################################################
    InputReaderBase(const bool read_from_file = true)
    :
    use_sub_cell_limiter{false},
    plot_int{-1},
    checkpoint_int{-1},
    regrid_int{-1},
    restart{-1},
    restart_time{0.0}
    {
        if (read_from_file)
        {
            this->ReadFromInputFile();
        }
    }
    // ################################################################

    // READ INPUT FILE ################################################
    void ReadFromInputFile()
    {
        // VARIABLES ==================================================
        const time_t date_and_time = time(0);
        
        amrex::ParmParse pp;
        // ============================================================

        // SUBCELL LIMITER ============================================
        {
            std::string tmp;
            pp.query("use_sub_cell_limiter", tmp);
            this->use_sub_cell_limiter = (tmp.compare("true") == 0);
        }
        // ============================================================

        // OUTPUT FILE INTERVAL =======================================
        this->plot_filepath = "plt"+std::to_string(date_and_time);
        pp.query("plot_int", this->plot_int);
        pp.query("plot_filepath", this->plot_filepath);
        // ============================================================

        // CHECKPOINT FILE INTERVAL ===================================
        this->checkpoint_filepath = "chk"+std::to_string(date_and_time);
        pp.query("checkpoint_int", this->checkpoint_int);
        pp.query("checkpoint_filepath", this->checkpoint_filepath);
        // ============================================================

        // REGRIDDING =================================================
        pp.query("regrid_int", this->regrid_int);
        // ============================================================

        // RESTART ====================================================
        pp.query("restart", this->restart);
        // ============================================================
    }
    // ################################################################
};

/**
 * \brief Input reader for single-patch problem
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
    InputReader(const bool read_from_file = true)
    :
    InputReaderBase(read_from_file)
    {
        if (read_from_file)
        {
            this->ReadFromInputFile();
        }
    }
    // ################################################################

    // AUXILIARY FUNCTION #############################################
    bool Plot() const
    {
        return (this->plot_int > 0);
    }

    bool Plot(const int n, const Real t) const
    {
        bool res;
        res = (n%(this->plot_int) == 0);
        res = res || (n == this->time.n_steps);
        res = res || (std::abs(t/this->time.T-1.0) < 1.0e-12);
        res = res && (this->plot_int > 0);

        return res;
    }

    bool Checkpoint(const int n, const Real t) const
    {
        bool res;
        res = (n%(this->checkpoint_int) == 0);
        res = res || (n == this->time.n_steps);
        res = res || (std::abs(t/this->time.T-1.0) < 1.0e-12);
        res = res && (this->checkpoint_int > 0);

        return res;
    }
    // ################################################################

    // READ INPUT FILE ################################################
    void ReadFromInputFile()
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

            pp.get("CFL", this->grid.CFL);

            pp.get("space_p", this->grid.space_p);

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

            pp.get("space_q", this->implicit_mesh.space_q);

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

namespace AMR
{

/**
 * \brief Input reader for single-patch AMR problem
 *
*/
struct InputReader
:
public InputReaderBase
{
    // DATA MEMBERS ###################################################
    SpaceInput space;
    TimeInput time;
    Vector<GridInput> grids;
    Vector<ImplicitMeshInput> implicit_meshes;
    Vector<DGInput> dG;
    ProblemInput problem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    InputReader(const int n_levels, const bool read_from_file = true)
    :
    InputReaderBase(read_from_file)
    {
        if (read_from_file)
        {
            this->ReadFromInputFile(n_levels);
        }
    }
    // ################################################################

    // AUXILIARY FUNCTION #############################################
    bool Plot() const
    {
        return (this->plot_int > 0);
    }

    bool Plot(const int n, const Real t) const
    {
        bool res;
        res = (n%(this->plot_int) == 0);
        res = res || (n == this->time.n_steps);
        res = res || (std::abs(t/this->time.T-1.0) < 1.0e-12);
        res = res && (this->plot_int > 0);

        return res;
    }

    bool Checkpoint(const int n, const Real t) const
    {
        bool res;
        res = (n%(this->checkpoint_int) == 0);
        res = res || (n == this->time.n_steps);
        res = res || (std::abs(t/this->time.T-1.0) < 1.0e-12);
        res = res && (this->checkpoint_int > 0);

        return res;
    }
    // ################################################################

    // READ INPUT FILE ################################################
    void ReadFromInputFile(const int n_levels)
    {
        // SPACE ======================================================
        {
            ParmParse pp("geometry");

            pp.queryarr("is_periodic", this->space.is_periodic);
            pp.query("coord_sys", this->space.coord_sys);
            pp.getarr("prob_lo", this->space.lo);
            pp.getarr("prob_hi", this->space.hi);
        }
        // ============================================================

        // TIME =======================================================
        {
            ParmParse pp("time");

            pp.get("T", this->time.T);
            pp.get("n_steps", this->time.n_steps);
        }
        // ============================================================

        // GRIDS ======================================================
        // INITIALIZATION
        {
            this->grids.resize(n_levels);
        }

        // READ FROM FILE
        {
            Vector<int> int_aux;
            Vector<Real> aux;

            ParmParse pp("grids");

            pp.getarr("CFL", aux);
            for (int l = 0; l < n_levels; ++l)
            {
                this->grids[l].CFL = aux[l];
            }

            pp.getarr("space_p", int_aux);
            for (int l = 0; l < n_levels; ++l)
            {
                this->grids[l].space_p = int_aux[l];
                this->grids[l].space_q = int_aux[l];
            }

            if (pp.contains("space_q"))
            {
                pp.getarr("space_q", int_aux);
                for (int l = 0; l < n_levels; ++l)
                {
                    this->grids[l].space_q = int_aux[l];
                }
            }
        }
        // ============================================================
        
        // MESHES =======================================================
        // INITIALIZATION
        {
            this->implicit_meshes.resize(n_levels);
        }

        // READ FROM FILE
        {
            Vector<int> int_aux;
            Vector<Real> aux;

            ParmParse pp("implicit_meshes");

            pp.getarr("space_q", int_aux);
            for (int l = 0; l < n_levels; ++l)
            {
                this->implicit_meshes[l].space_q = int_aux[l];
                this->implicit_meshes[l].space_q_ie = int_aux[l];
            }

            if (pp.contains("space_q_ie"))
            {
                pp.queryarr("space_q_ie", int_aux);
                for (int l = 0; l < n_levels; ++l)
                {
                    this->implicit_meshes[l].space_q_ie = int_aux[l];
                }
            }

            if (pp.contains("volume_fraction_threshold"))
            {
                pp.queryarr("volume_fraction_threshold", aux);
                for (int l = 0; l < n_levels; ++l)
                {
                    this->implicit_meshes[l].VFT = aux[l];
                }
            }
        }
        // ============================================================

        // DISCONTINUOUS GALERKIN METHODS =============================
        // INITIALIZATION
        {
            this->dG.resize(n_levels);
        }

        // READ FROM FILE
        {
            ParmParse pp("dG");

            Vector<int> int_aux;

            pp.getarr("space_p", int_aux);
            for (int l = 0; l < n_levels; ++l)
            {
                this->dG[l].space_p = int_aux[l];
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

} // namespace AMR

} // namespace DG
} // namespace amrex

#endif