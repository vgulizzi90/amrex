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

namespace amrex
{
namespace dG
{
namespace AMR
{

// ADAPTIVE MESH REFINEMENT CLASS #####################################
struct SinglePatch
:
public AmrCore
{
    // DATA MEMBERS ===================================================
    InputReaderSinglePatch inputs;
    Vector<StructuredMesh *> meshes;
    
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


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::AMR::SinglePatch.MakeNewLevelFromScratch" << std::endl;
exit(-1);
    }

    /**
     * \brief Update the masks between levels.
     *
    */
    void update_masks()
    {
        for (int lev = 1; lev <= this->max_level; ++lev)
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
Print() << "amrex::dG::AMR::SinglePatch.ErrorEst" << std::endl;
exit(-1);
    }
    // ================================================================

    
    // MAKE NEW LEVEL FROM A COARSE LEVEL =============================
    virtual void MakeNewLevelFromCoarse(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::AMR::SinglePatch.MakeNewLevelFromCoarse" << std::endl;
exit(-1);
    }
    // ================================================================


    // REMAKE AN EXISTING LEVEL =======================================
    virtual void RemakeLevel(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "amrex::dG::AMR::SinglePatch.RemakeLevel" << std::endl;
exit(-1);
    }
    // ================================================================


    // CLEAR LEVEL ====================================================
    virtual void ClearLevel(int /*lev*/) override
    {
Print() << "amrex::dG::AMR::SinglePatch.ClearLevel" << std::endl;
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


    // INPUT/OUTPUT ===================================================
    void make_step_output_folder(const int n, const amrex::Real t) const;
    
    /**
    * \brief Export meshes to VTK format.
    *
    * \param[in] n: time step index.
    * \param[in] t: time.
    * \param[in] filename_root: root of the filename containing the mesh output.
    * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_mesh_to_VTK(const int n, const amrex::Real t, const std::string & filename_root, IBVP & ibvp) const
    {
        // MAKE SURE THE FOLDER EXISTS --------------------------------
        this->make_step_output_folder(n, t);
        // ------------------------------------------------------------

        // EXPORT THE MESHES ------------------------------------------
        for (int lev = 0; lev <= this->max_level; ++lev)
        {
            const StructuredMesh & mesh = *this->meshes[lev];
            const iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            mesh.export_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                               this->Geom(lev), mask, ibvp);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
};
// ####################################################################

} // namespace AMR
} // namespace dG
} // namespace amrex

#endif