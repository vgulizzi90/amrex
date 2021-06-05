//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Pardiso.H
 * \brief Contains utilities to call pardiso linear solver.
*/

#ifndef AMREX_DG_PARDISO_H_
#define AMREX_DG_PARDISO_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

/* PARDISO prototype. */
extern "C" void pardisoinit(void *, int *, int *, int *, double *, int *);
extern "C" void pardiso_chkmatrix(int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);
extern "C" void pardiso(void *, int *, int *, int *, int *, int *, double *, int *, int *, int *, int *, int *, int *, double *, double *, int *, double *);

#define __DG_PARDISO_REAL_STRUCTURAL_SYMMETRIC_MATRIX__ 1
#define __DG_PARDISO_REAL_SYMMETRIC_POS_DEFINITE_MATRIX__ 2
#define __DG_PARDISO_REAL_SYMMETRIC_INDEFINITE_MATRIX__ -2
#define __DG_PARDISO_REAL_NONSYMMETRIC_MATRIX__ 11

namespace amrex
{
namespace dG
{

// PARDISO ############################################################
struct LinearSolverPardiso
{
    // DATA MEMBERS ===================================================
    int mtype;

    int n;
    Vector<int> ia, ja;
    Vector<Real> A, B, X;
    int nnz;
    int n_rhs;

    void * pt[64];

    int i_params[64];
    Real d_params[64];
    int maxfct, mnum, phase, error, msglvl, solver;

    int num_procs;

    Vector<iMultiFab> dom_info;
    Vector<Array<iMultiFab, AMREX_SPACEDIM>> bou_info;

    bool first_call;
    // ================================================================


    // CONSTRUCTOR ====================================================
    // ================================================================


    // DESTRUCTOR =====================================================
    // ================================================================


    // READERS ========================================================
    bool matrix_is_real_symmetric() const;
    bool matrix_is_real_nonsymmetric() const;
    void print_stats();
    // ================================================================


    // INITIALIZATION =================================================
    void read_input_file();
    void init();
    // ================================================================


    // CHECKS =========================================================
    void matrix_check();
    // ================================================================


    // SOLUTION =======================================================
    void solve(const bool actually_solve = true);
    // ================================================================


    // OUTPUT =========================================================
    void print_to_file_sparse(const std::string & filename_root);
    // ================================================================


    // TERMINATION ====================================================
    void terminate();
    // ================================================================
};
// ####################################################################

} // namespace dG
} // namespace amrex

#endif