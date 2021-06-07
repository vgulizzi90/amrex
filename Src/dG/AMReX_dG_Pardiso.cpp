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
    void LinearSolverPardiso::read_input_file()
    {
        ParmParse pp("pardiso");

        pp.get("mtype", this->mtype);
        pp.get("n_rhs", this->n_rhs);

        this->msglvl = 1;
        pp.query("msglvl", this->msglvl);
    }
    
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

    void LinearSolverPardiso::print_stats()
    {
        // PRINT STATISTICS -------------------------------------------
        pardiso_printstats(&this->mtype,
                           &this->n,
                           this->A.data(),
                           this->ia.data(),
                           this->ja.data(),
                           &this->n_rhs,
                           this->B.data(),
                           &this->error);
        if (this->error != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::print_stats\n";
            msg += "| Error during pardiso_printstats.\n";
            msg += "| Error: "+std::to_string(this->error)+".\n";
            Abort(msg);
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // CHECKS =========================================================
    void LinearSolverPardiso::matrix_check()
    {
        // PARAMETERS -------------------------------------------------
        Real d_dum;
        int i_dum;
        // ------------------------------------------------------------

        // IF 0-BASED, CONVERT TO 1-BASED INDEXING --------------------
        if (this->ia[0] == 0)
        {
            for (int i = 0; i < this->n+1; i++)
            {
                this->ia[i] += 1;
            }
            for (int i = 0; i < this->nnz; i++)
            {
                this->ja[i] += 1;
            }
        }
        // ------------------------------------------------------------

        // CHECK MATRIX -----------------------------------------------
        pardiso_chkmatrix(&this->mtype,
                          &this->n,
                          this->A.data(),
                          this->ia.data(),
                          this->ja.data(),
                          &this->error);
        if (this->error != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::matrix_check\n";
            msg += "| Error in consistency of the matrix.\n";
            msg += "| Error: "+std::to_string(this->error)+".\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // REORDERING AND SYMBOLIC FACTORIZATION ----------------------
        this->phase = 11;
        pardiso(this->pt,
                &this->maxfct,
                &this->mnum,
                &this->mtype,
                &this->phase,
                &this->n,
                this->A.data(),
                this->ia.data(),
                this->ja.data(),
                &i_dum,
                &this->n_rhs,
                this->i_params,
                &this->msglvl,
                &d_dum,
                &d_dum,
                &this->error,
                this->d_params);
        if (this->error != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::matrix_check\n";
            msg += "| Error in phase 11: Reordering and symbolic factorization.\n";
            msg += "| Error: "+std::to_string(this->error)+".\n";
            Abort(msg);
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // SOLUTION =======================================================
    void LinearSolverPardiso::factorize()
    {
        // PARAMETERS -------------------------------------------------
        Real d_dum;
        int i_dum;
        // ------------------------------------------------------------

        // NUMERICAL FACTORIZATION ------------------------------------
        this->phase = 22;
        pardiso(this->pt,
                &this->maxfct,
                &this->mnum,
                &this->mtype,
                &this->phase,
                &this->n,
                this->A.data(),
                this->ia.data(),
                this->ja.data(),
                &i_dum,
                &this->n_rhs,
                this->i_params,
                &this->msglvl,
                &d_dum,
                &d_dum,
                &this->error,
                this->d_params);

        if (this->error != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::factorize\n";
            msg += "| Error in phase 22: Numerical factorization.\n";
            msg += "| Error: "+std::to_string(this->error)+".\n";
            Abort(msg);
        }
        // ------------------------------------------------------------
    }
    
    void LinearSolverPardiso::solve()
    {
        // PARAMETERS -------------------------------------------------
        int i_dum;
        // ------------------------------------------------------------

        // TRY ITERATIVE SOLUTION FIRST -------------------------------
        this->i_params[3] = 61;
        this->phase = 33;

        pardiso(this->pt,
                &this->maxfct,
                &this->mnum,
                &this->mtype,
                &this->phase,
                &this->n,
                this->A.data(),
                this->ia.data(),
                this->ja.data(),
                &i_dum,
                &this->n_rhs,
                this->i_params,
                &this->msglvl,
                this->B.data(),
                this->X.data(),
                &this->error,
                this->d_params);

        if (this->error != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::solve\n";
            msg += "| Error in phase 33: Back substitution and iterative refinement - 1.\n";
            msg += "| Error: "+std::to_string(this->error)+".\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        if (this->i_params[19] < 0)
        {
            this->i_params[3] = 0;

            // NUMERICAL FACTORIZATION --------------------------------
            this->factorize();
            // --------------------------------------------------------

            // BACK SUBSTITUTION AND ITERATIVE REFINEMENT -------------
            this->phase = 33;

            pardiso(this->pt,
                    &this->maxfct,
                    &this->mnum,
                    &this->mtype,
                    &this->phase,
                    &this->n,
                    this->A.data(),
                    this->ia.data(),
                    this->ja.data(),
                    &i_dum,
                    &this->n_rhs,
                    this->i_params,
                    &this->msglvl,
                    this->B.data(),
                    this->X.data(),
                    &this->error,
                    this->d_params);

            if (this->error != 0)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: AMReX_dG_Pardiso.cpp - LinearSolverPardiso::solve\n";
                msg += "| Error in phase 33: Back substitution and iterative refinement - 2.\n";
                msg += "| Error: "+std::to_string(this->error)+".\n";
                Abort(msg);
            }
            // --------------------------------------------------------
        }

        this->first_call = false;
    }
    // ================================================================


    // OUTPUT =========================================================
    void LinearSolverPardiso::print_to_file_sparse(const std::string & filename_root)
    {
        const std::string filename_ia = filename_root+"_ia.txt";
        const std::string filename_ja = filename_root+"_ja.txt";
        const std::string filename_A = filename_root+"_A.txt";

        std::ofstream fp;
        
        fp.open(filename_ia);
        for (int i = 0; i < this->ia.size(); ++i)
        {
            //fp << std::scientific << std::setprecision(11) << std::setw(18) << std::showpos << this->ia[i] << "\n";
            fp << this->ia[i] << "\n";
        }
        fp << "\n";
        fp.close();

        fp.open(filename_ja);
        for (int i = 1; i < this->ia.size(); ++i)
        {
            const int pos = this->ia[i-1]-1;
            const int nc = this->ia[i]-this->ia[i-1];

            for (int j = 0; j < nc; ++j)
            {
                fp << this->ja[j+pos] << " ";
            }
            fp << "\n";
        }
        fp << "\n";
        fp.close();

        fp.open(filename_A);
        for (int i = 1; i < this->ia.size(); ++i)
        {
            const int pos = this->ia[i-1]-1;
            const int nc = this->ia[i]-this->ia[i-1];

            for (int j = 0; j < nc; ++j)
            {
                fp << std::scientific << std::setprecision(3) << std::setw(10) << std::showpos << this->A[j+pos] << " ";
            }
            fp << "\n";
        }
        fp << "\n";
        fp.close();
    }
    // ================================================================


    // TERMINATION ====================================================
    void LinearSolverPardiso::terminate()
    {
        Real d_dum;
        int i_dum;

        this->phase = -1;
        pardiso(this->pt,
                &this->maxfct,
                &this->mnum,
                &this->mtype,
                &this->phase,
                &this->n,
                &d_dum,
                this->ia.data(),
                this->ja.data(),
                &i_dum,
                &this->n_rhs,
                this->i_params,
                &this->msglvl,
                &d_dum,
                &d_dum,
                &this->error,
                this->d_params);
    }
    // ================================================================
// ####################################################################

} // namespace dG
} // namespace amrex