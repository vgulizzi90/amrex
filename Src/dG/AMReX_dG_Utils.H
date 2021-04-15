//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Utils.H
 * \brief A file that defines a few utilities.
*/

#ifndef AMREX_DG_UTILS_H_
#define AMREX_DG_UTILS_H_

#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_Utility.H>

#include <AMReX_dG_Constants.H>

namespace amrex
{
namespace dG
{

// AUXILIARY UTILITIES ################################################
std::string second_to_hms(const Real s);
std::string replace_input_keywords(const std::string & is);
// ####################################################################



namespace io
{

// INPUT/OUTPUT ROUTINES FOR FOLDERS ##################################
std::string make_path(const std::initializer_list<std::string>);
// ####################################################################


// INPUT/OUTPUT ROUTINES FOR ARRAYS ###################################
void print_reals(const int rank, const MPI_Comm comm, const int n, const Real * src, std::ostream & os = OutStream());
void print_reals(const int rank, const int n, const Real * src, std::ostream & os = OutStream());
void print_reals(const int n, const Real * src, std::ostream & os = OutStream());
void print_real_array_2d(const int rank, const MPI_Comm comm, const int Nr, const int Nc, const Real * src, std::ostream & os = OutStream());
void print_real_array_2d(const int rank, const int Nr, const int Nc, const Real * src, std::ostream & os = OutStream());
void print_real_array_2d(const int Nr, const int Nc, const Real * src, std::ostream & os = OutStream());
// ####################################################################

} // namespace io



// TIME KEEPER ########################################################
/**
 * \brief Time keeper class.
 *
*/
struct TimeKeeper
{
    // DATA MEMBERS ===================================================
    Vector<Real> start_time;
    Real elapsed_time;
    // ================================================================

    
    // CONSTRUCTOR ====================================================
    TimeKeeper()
    {}
    // ================================================================

    
    // DESTRUCTOR =====================================================
    ~TimeKeeper()
    {}
    // ================================================================

    
    // TIC/TOC ========================================================
    void tic();
    void toc();
    // ================================================================


    // ELAPSED TIME ===================================================
    Real get_elapsed_time_in_seconds() const;
    std::string get_elapsed_time_in_hms() const;
    // ================================================================
};
// ####################################################################



// INPUT READERS ######################################################
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
 * \brief Base version of the input reader
 *
*/
struct InputReaderBase
{
    // DATA MEMBERS ===================================================
    Real start_time_s;

    Real wall_time;
    std::string wall_time_units;
    Real wall_time_s;
    
    std::string output_folderpath;
    std::string checkpoint_filename;
    int checkpoint_int;
    std::string plot_filename;
    int plot_int;
    bool output_overwrite;

    int restart;
    Real restart_time;
    // ================================================================


    // CONSTRUCTOR ====================================================
    InputReaderBase()
    :
    start_time_s{second()},
    wall_time{std::numeric_limits<Real>::max()},
    wall_time_units{"s"},
    wall_time_s{this->wall_time},
    output_folderpath{""},
    checkpoint_filename{""},
    checkpoint_int{-1},
    plot_filename{""},
    plot_int{-1},
    output_overwrite{true},
    restart{-1},
    restart_time{0.0}
    {}
    // ================================================================


    // READ INPUT FILE ================================================
    void read_input_file();
    // ================================================================
};


/**
 * \brief Input reader for single-patch problem
 *
*/
struct InputReaderSinglePatch
:
public InputReaderBase
{
    // DATA MEMBERS ===================================================
    TimeInput time;
    // ================================================================


    // CONSTRUCTOR ====================================================
    InputReaderSinglePatch()
    {}
    // ================================================================


    // READ INPUT FILE ================================================
    void read_input_file();
    // ================================================================


    // READERS ========================================================
    bool plot(const int n, const Real t) const;
    // ================================================================


    // OUTPUT FOLDERS =================================================
    std::string get_step_string(const int n) const;
    std::string get_level_folderpath(const int lev) const;
    std::string get_level_step_folderpath(const int lev, const int n) const;
    // ================================================================
};
// ####################################################################

} // namespace dG
} // namespace amrex

#endif