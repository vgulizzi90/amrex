//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Hypre.H
 * \brief Contains utilities to call hypre linear solver.
*/

#ifndef AMREX_DG_HYPRE_H_
#define AMREX_DG_HYPRE_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <AMReX_HypreIJIface.H>

#include <AMReX_dG_Utils.H>

#define __DG_HYPRE_REAL_STRUCTURAL_SYMMETRIC_MATRIX__ 1
#define __DG_HYPRE_REAL_SYMMETRIC_POS_DEFINITE_MATRIX__ 2
#define __DG_HYPRE_REAL_SYMMETRIC_INDEFINITE_MATRIX__ -2
#define __DG_HYPRE_REAL_NONSYMMETRIC_MATRIX__ 11

namespace amrex
{
namespace dG
{


// HYPRE ##############################################################
struct LinearSolverHypre
{
    // DATA MEMBERS ===================================================
    Vector<int> posX_offset;

    int mtype;

    int n;
    Vector<int> ia, ja;
    Vector<Real> A, B, X;
    int nnz;
    int n_rhs;

    int msglvl;

    Vector<iMultiFab> dom_info;
    Vector<Array<iMultiFab, AMREX_SPACEDIM>> bou_info;

    int rlo, rhi;
    HYPRE_IJMatrix AIJ;

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
    void factorize();
    void solve();
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