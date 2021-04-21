//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_AMR.H
 * \brief Basis data structures and routines for AMR computations.
*/

#ifndef AMREX_DG_AMR_H_
#define AMREX_DG_AMR_H_

#include <AMReX_MultiFabUtil.H>
#include <AMReX_AmrCore.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_dG_Utils.H>
#include <AMReX_dG_Mesh.H>
#include <AMReX_dG_Mesh_Output.H>
#include <AMReX_dG_Solution.H>
#include <AMReX_dG_Solution_Output.H>

namespace amrex
{
namespace dG
{
namespace amr
{

// ADAPTIVE MESH REFINEMENT CLASS #####################################
struct SinglePatch
:
public AmrCore
{
    // DATA MEMBERS ===================================================
    InputReaderSinglePatch inputs;
    Vector<Mesh *> meshes;
    Vector<Solution *> solutions;
    
    Vector<iMultiFab> masks;
    // ================================================================


    // CONSTRUCTOR ====================================================
    SinglePatch();
    // ================================================================


    // DESTRUCTOR =====================================================
    ~SinglePatch();
    // ================================================================


    // INITIALIZATION =================================================
    void init_inputs();
    void init();
    // ================================================================


    // READERS ========================================================
    bool advance_in_time_continue(const int n, const Real t) const;

    int get_largest_dG_space_p() const;
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.MakeNewLevelFromScratch" << std::endl;
exit(-1);
    }

    /**
     * \brief Update the masks between levels.
     *
    */
    void update_masks()
    {
        for (int lev = 1; lev <= this->finest_level; ++lev)
        {
            const sMultiFab & c_cell_type = this->meshes[lev-1]->cell_type;
            const sMultiFab & f_cell_type = this->meshes[lev]->cell_type;
            
            this->masks[lev-1] = makeFineMask(c_cell_type, f_cell_type,
                                              c_cell_type.n_grow, this->ref_ratio[lev-1],
                                              this->Geom(lev-1).periodicity(),
                                              __DG_CELL_NON_MASKED__, __DG_CELL_MASKED__);
        }
    }
    // ================================================================


    // TAG CELLS FOR REFINEMENT =======================================
    virtual void ErrorEst(int /*lev*/, TagBoxArray & /*tags*/, Real /*time*/, int /*ngr*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.ErrorEst" << std::endl;
exit(-1);
    }
    // ================================================================

    
    // MAKE NEW LEVEL FROM A COARSE LEVEL =============================
    virtual void MakeNewLevelFromCoarse(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.MakeNewLevelFromCoarse" << std::endl;
exit(-1);
    }
    // ================================================================


    // REMAKE AN EXISTING LEVEL =======================================
    virtual void RemakeLevel(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.RemakeLevel" << std::endl;
exit(-1);
    }
    // ================================================================


    // CLEAR LEVEL ====================================================
    virtual void ClearLevel(int /*lev*/) override
    {
Print() << "amrex::dG::amr::SinglePatch.ClearLevel" << std::endl;
exit(-1);
    }
    // ================================================================


    // CHECK QUADRATURE RULES =========================================
    void check_quadrature_rules(const int N_DOM,
                                const Real * exact_volume,
                                const Real * exact_surface) const;
    void check_quadrature_rules(const Real exact_volume,
                                const Real exact_surface) const;
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error among the levels using the L_inf norm.
     *
     * \param[in] t: time.
     * \param[in] X: a vector of MultiFabs containing the solution coefficients.
     * \param[out] err: L_inf-based error computed over the levels.
     * \param[out] err_norm: L_inf-based error normalization computed over the levels.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_error_L_inf(const Real t, const Vector<MultiFab> & X, Real & err, Real & err_norm, const IBVP & ibvp) const
    {
        err = 0.0;
        err_norm = 0.0;

        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const Solution & solution = *this->solutions[lev];
            const iMultiFab & mask = this->masks[lev];
            
            Real err_lev, err_norm_lev;

            solution.eval_error_L_inf(t, this->Geom(lev), mesh, X[lev], mask, err_lev, err_norm_lev, ibvp);

            err = amrex::max(err, err_lev);
            err_norm = amrex::max(err_norm, err_norm_lev);
        }
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void make_step_output_folder(const int n, const Real t) const;
    
    /**
     * \brief Export meshes to VTK format.
     *
     * \param[in] n: time step index.
     * \param[in] filename_root: root of the filename containing the mesh output.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_mesh_to_VTK(const int n, const std::string & filename_root, const IBVP & ibvp) const
    {
        // EXPORT THE MESHES ------------------------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            mesh_io::export_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                   this->Geom(lev), mesh, mask, ibvp);
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Export the meshes quadrature points to VTK format.
     *
     * \param[in] n: time step index.
     * \param[in] filename_root: root of the filename containing the mesh output.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_quadrature_points_to_VTK(const int n, const std::string & filename_root, const IBVP & ibvp) const
    {
        // EXPORT THE MESHES QUADRATURE POINTS ------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            mesh_io::export_quadrature_points_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                     this->Geom(lev), mesh, mask, ibvp);
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Export solutions to VTK format.
     *
     * \param[in] n: time step index.
     * \param[in] filename_root: root of the filename containing the mesh output.
     * \param[in] t: time.
     * \param[in] X: a vector of MultiFabs containing the solution coefficients.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_solution_to_VTK(const int n, const std::string & filename_root, const Real t, const Vector<MultiFab> & X, const IBVP & ibvp) const
    {
        // EXPORT THE SOLUTIONS ---------------------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const Mesh & mesh = *this->meshes[lev];
            const Solution & solution = *this->solutions[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            if (mesh.uses_level_set())
            {
                if ((solution.params.space_p == 0) && solution.params.FV_is_active)
                {
                    solution_io::export_FV_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                                  t, this->Geom(lev), mesh, solution, X[lev], mask, ibvp);
                }
                else
                {
                    solution_io::export_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                               t, this->Geom(lev), solution, X[lev], mask, ibvp);
                }
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: AMReX_dG_AMR.H - SinglePatch::export_solution_to_VTK\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                Abort(msg);
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================
};
// ####################################################################

} // namespace amr
} // namespace dG
} // namespace amrex

#endif