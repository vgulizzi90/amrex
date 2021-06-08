//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Hypre.cpp
 * \brief Contains implementations of functions for hypre linear solver.
*/

#include <AMReX_dG_Hypre.H>

namespace amrex
{
namespace dG
{

// HYPRE ##############################################################
    // INITIALIZATION =================================================
    void LinearSolverHypre::read_input_file()
    {
        ParmParse pp("hypre");

        pp.get("mtype", this->mtype);
        pp.get("n_rhs", this->n_rhs);

        this->msglvl = 1;
        pp.query("msglvl", this->msglvl);
    }

    void LinearSolverHypre::init()
    {
        // PARAMETERS -------------------------------------------------
        const int n_ranks = ParallelDescriptor::NProcs();
        // ------------------------------------------------------------

        // DOF POSITION IN THE GLOBAL SYSTEM --------------------------
        this->posX_offset.resize(n_ranks);
        // ------------------------------------------------------------

        // HYPRE INITIALIZATION ---------------------------------------
        HYPRE_Init();
        // ------------------------------------------------------------

        // FIRST CALL FLAG --------------------------------------------
        this->first_call = true;
        // ------------------------------------------------------------
    }
    // ================================================================


    // READERS ========================================================
    bool LinearSolverHypre::matrix_is_real_symmetric() const
    {
        bool cond;

        cond =         (this->mtype == __DG_HYPRE_REAL_STRUCTURAL_SYMMETRIC_MATRIX__);
        cond = cond || (this->mtype == __DG_HYPRE_REAL_SYMMETRIC_POS_DEFINITE_MATRIX__);
        cond = cond || (this->mtype == __DG_HYPRE_REAL_SYMMETRIC_INDEFINITE_MATRIX__);

        return cond;
    }

    bool LinearSolverHypre::matrix_is_real_nonsymmetric() const
    {
        bool cond;

        cond = (this->mtype == __DG_HYPRE_REAL_NONSYMMETRIC_MATRIX__);
        
        return cond;
    }

    void LinearSolverHypre::print_stats()
    {
Print() << "LinearSolverHypre::init(): " << std::endl;
exit(-1);
    }
    // ================================================================


    // CHECKS =========================================================
    void LinearSolverHypre::matrix_check()
    {
        // PARAMETERS -------------------------------------------------
        const int rank = ParallelDescriptor::MyProc();
        // ------------------------------------------------------------
        
        // GET THE ROW PARTITIONING -----------------------------------
        if (rank == 0)
        {
            this->rlo = 0;
        }
        else
        {
            this->rlo = this->posX_offset[rank-1];
        }

        this->rhi = this->posX_offset[rank]-1;
        // ------------------------------------------------------------

        // CREATE MATRIX ----------------------------------------------
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, this->rlo, this->rhi, this->rlo, this->rhi, &this->AIJ);
        HYPRE_IJMatrixSetObjectType(this->AIJ, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(this->AIJ);
        // ------------------------------------------------------------


        // SET VALUES -------------------------------------------------
        {
Print(rank) << "n(" << rank << "): " << this->ia.size() << std::endl;
        }
        // ------------------------------------------------------------

        
Print() << "LinearSolverHypre::matrix_check(): " << std::endl;
exit(-1);
    }
    // ================================================================


    // SOLUTION =======================================================
    void LinearSolverHypre::factorize()
    {
Print() << "LinearSolverHypre::factorize(): " << std::endl;
exit(-1);
    }
    
    void LinearSolverHypre::solve()
    {
Print() << "LinearSolverHypre::solve(): " << std::endl;
exit(-1);
    }
    // ================================================================


    // OUTPUT =========================================================
    void LinearSolverHypre::print_to_file_sparse(const std::string & filename_root)
    {
Print() << "LinearSolverHypre::print_to_file_sparse(): " << std::endl;
exit(-1);
    }
    // ================================================================


    // TERMINATION ====================================================
    void LinearSolverHypre::terminate()
    {
Print() << "LinearSolverHypre::terminate(): " << std::endl;
exit(-1);
    }
    // ================================================================
// ####################################################################

} // namespace dG
} // namespace amrex