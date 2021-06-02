//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Pardiso.cpp
 * \brief Contains implementations of functions for pardiso linear solver.
*/

#include <AMReX_dG_Pardiso.H>

namespace amrex
{
namespace dG
{

// PARDISO ############################################################
    // INITIALIZATION =================================================
    void LinearSolverPardiso::init()
    {
        // VARIABLES --------------------------------------------------
        char * c_var;
        // ------------------------------------------------------------

        // PARDISO BASE CONTROL PARAMETERS ----------------------------
        this->error = 0;
        this->solver = 0;
        pardisoinit(this->pt,
                    &this->mtype,
                    &this->solver,
                    this->i_params,
                    this->d_params,
                    &this->error);

        if (this->error != 0) 
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.H - LinearSolverPardiso::init\n";

            if (this->error == -10)
            {
                msg += "| No license file found \n";
            }
            if (this->error == -11)
            {
                msg += "| License is expired \n";
            }
            if (this->error == -12)
            {
                msg += "| Wrong username or hostname \n";
            }
            Abort(msg);
        }

        // NUMBER OF OPENMP PROCESSORS
        c_var = getenv("OMP_NUM_THREADS");

        if (c_var != NULL)
        {
            sscanf(c_var, "%d", &this->num_procs);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR.H - LinearSolverPardiso::init\n";
            msg += "| Set environment OMP_NUM_THREADS to 1 \n";
            Abort(msg);
        }
        this->i_params[2] = this->num_procs;

        this->maxfct = 1;
        this->mnum = 1;

        this->msglvl = 1;
        this->error = 0;
        // ------------------------------------------------------------

        // FIRST CALL FLAG --------------------------------------------
        this->first_call = true;
        // ------------------------------------------------------------
    }
    // ================================================================


    // READERS ========================================================
    bool LinearSolverPardiso::matrix_is_real_symmetric() const
    {
        bool cond;

        cond =         (this->mtype == __DG_PARDISO_REAL_STRUCTURAL_SYMMETRIC_MATRIX__);
        cond = cond || (this->mtype == __DG_PARDISO_REAL_SYMMETRIC_POS_DEFINITE_MATRIX__);
        cond = cond || (this->mtype == __DG_PARDISO_REAL_SYMMETRIC_INDEFINITE_MATRIX__);

        return cond;
    }

    bool LinearSolverPardiso::matrix_is_real_nonsymmetric() const
    {
        bool cond;

        cond = (this->mtype == __DG_PARDISO_REAL_NONSYMMETRIC_MATRIX__);

        return cond;
    }
    // ================================================================
// ####################################################################

} // namespace dG
} // namespace amrex