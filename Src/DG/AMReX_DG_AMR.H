//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_AMR.H
 * \brief Basis data structure for AMR computations.
*/

#ifndef AMREX_DG_AMR_H_
#define AMREX_DG_AMR_H_

#include <AMReX_MultiFabUtil.H>
#include <AMReX_AmrCore.H>

#include <AMReX_DG_Constants.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_Solution.H>

namespace amrex
{
namespace DG
{
namespace AMR
{

// ####################################################################
// ADAPTIVE MESH REFINEMENT CLASS #####################################
// ####################################################################
/**
 * \brief Adaptive Mesh Refinement base class.
 *
*/
struct Base
:
public AmrCore
{
    // DATA MEMBERS ###################################################
    InputReader inputs;
    
    Vector<ImplicitMesh *> meshes;
    Vector<MatrixFactory *> matfactories;
    Vector<iMultiFab> masks;
    Vector<MultiFab> Xs;
    // ################################################################

    // CONSTRUCTOR ####################################################
    Base()
    :
    inputs(this->max_level+1)
    {
        const int n_levels = this->max_level+1;

        this->meshes.resize(n_levels);
        this->matfactories.resize(n_levels);
        this->masks.resize(n_levels);
        this->Xs.resize(n_levels);
    }

    virtual ~Base()
    {
        for (int lev = 0; lev <= this->max_level; ++lev)
        {
            if (this->meshes[lev] != nullptr)
            {
                delete this->meshes[lev];
            }
            if (this->matfactories[lev] != nullptr)
            {
                delete this->matfactories[lev];
            }
        }
    }
    // ################################################################

    // AUXILIARY FUNCTION #############################################
    bool contains_nan() const
    {
        bool res = false;

        for (int lev = 0; lev <= this->max_level; ++lev)
        {
            if (this->LevelIsValid(lev))
            {
                res = res || this->Xs[lev].contains_nan();
            }
        }

        return res;
    }

    void ExtractLevelInputs(const int lev, amrex::DG::InputReader & level_inputs) const
    {
        level_inputs.space = this->inputs.space;
        level_inputs.time = this->inputs.time;
        level_inputs.grid = this->inputs.grids[lev];
        level_inputs.implicit_mesh = this->inputs.implicit_meshes[lev];
        level_inputs.dG = this->inputs.dG[lev];
        level_inputs.problem = this->inputs.problem;
    }
    // ################################################################

    // MAKE NEW LEVEL FROM SCRATCH ####################################
    virtual void MakeNewLevelFromScratch(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "Base.MakeNewLevelFromScratch" << std::endl;
exit(-1);
    }
    // ################################################################

    // TAG CELL FOR REFINEMENT ########################################
    
    
    virtual void ErrorEst(int /*lev*/, TagBoxArray & /*tags*/, Real /*time*/, int /*ngr*/) override
    {
Print() << "Base.ErrorEst" << std::endl;
exit(-1);
    }

    /**
     * \brief Update the masks between levels.
     *
    */
    void UpdateMasks()
    {
        for (int lev = 1; lev <= this->max_level; ++lev)
        {
            if (this->LevelIsValid(lev))
            {
                const MultiFab & cX = this->Xs[lev-1];
                const MultiFab & fX = this->Xs[lev];
                const ImplicitMesh & c_mesh = *this->meshes[lev-1];
                
                this->masks[lev-1] = makeFineMask(cX, fX, cX.n_grow, this->ref_ratio[lev-1],
                                                  c_mesh.geom.periodicity(),
                                                  __DG_CELL_UNMASKED__, __DG_CELL_MASKED__);
            }
        }
    }

    /**
     * \brief Tag i) small cells merged with tagged cells and ii) cells that extend onto tagged small cells.
     *
     * \param[in] mesh: ImplicitMesh data structure containing element type information.
     * \param[inout] tags: TagBoxArray containing tagging information.
     *
    */
    template <typename TAG_MF, typename TAG_T>
    void FixTagging(const ImplicitMesh & mesh, TAG_MF & tags, const TAG_T tagval) const
    {
        // PARAMETERS =================================================
        const int N_DOM = mesh.eType.n_comp/__DG_ELM_TYPE_N_COMP_PER_DOM__;
        // ============================================================

        // TAG SMALL CELLS THAT ARE MERGED WITH TAGGED CELLS ==========
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            auto tags_fab = tags.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_not_tagged = (tags_fab(i,j,k) != tagval);

                    if (ELM_IS_SMALL(etype) && elm_is_not_tagged)
                    {
                        // LOCAL PARAMETERS
                        const int merged_b = etype/10;

                        // LOCAL VARIABLES
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        bool nbr_is_tagged;

                        NBR_CELL(i, j, k, merged_b, nbr_i, nbr_j, nbr_k, nbr_b);

                        nbr_is_tagged = (tags_fab(nbr_i,nbr_j,nbr_k) == tagval);

                        if (nbr_is_tagged)
                        {
                            tags_fab(i,j,k) = tagval;
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        // ============================================================

        // TAG CELLS EXTENDING ONTO TAGGED SMALL CELLS ================
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            auto tags_fab = tags.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_not_tagged = (tags_fab(i,j,k) != tagval);

                    if (ELM_IS_EXTENDED(etype) && elm_is_not_tagged)
                    {
                        // LOCAL VARIABLES
                        short nbr_etype;
                        int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                        bool nbr_is_small, nbr_is_tagged;

                        for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                        {
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                            nbr_is_small = ELM_IS_SMALL(nbr_etype);
                            nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                            nbr_is_tagged = (tags_fab(nbr_i,nbr_j,nbr_k) == tagval);

                            if ((nbr_merged_b == nbr_b) && nbr_is_tagged)
                            {
                                tags_fab(i,j,k) = tagval;
                            }
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        tags.FillBoundary(mesh.geom.periodicity());
        // ============================================================
    }
    // ################################################################

    // MAKE NEW LEVEL FROM A COARSE LEVEL #############################
    virtual void MakeNewLevelFromCoarse(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "Base.MakeNewLevelFromCoarse" << std::endl;
exit(-1);
    }
    // ################################################################

    // REMAKE AN EXISTING LEVEL #######################################
    virtual void RemakeLevel(int /*lev*/, Real /*time*/, const BoxArray & /*ba*/, const DistributionMapping & /*dm*/) override
    {
Print() << "Base.RemakeLevel" << std::endl;
exit(-1);
    }
    // ################################################################

    // DELETE LEVEL DATA ##############################################
    virtual void ClearLevel(int /*lev*/) override
    {
Print() << "Base.ClearLevel" << std::endl;
exit(-1);
    }
    // ################################################################

    // AUXILIARY FUNCTIONS ############################################
    bool LevelIsValid(const int lev) const
    {
        const bool res = ((this->meshes[lev] != nullptr) &&
                          (this->matfactories[lev] != nullptr));

        return res;
    }
    // ################################################################

    // CHECK QUADRATURE RULES #########################################
    /**
     * \brief Check the quadrature rules by comparing the domains' volume and surface with exact values.
     *
     * \param[in] N_DOM: number of domains to be considered.
     * \param[in] exact_volume: exact_volume[dom] should contain the exact value of the volume of the
     *                          dom-th domain.
     * \param[in] exact_surface: exact_surface[dom] should contain the exact value of the surface of the
     *                           dom-th domain.
     *
    */
    void CheckQuadratureRules(const int N_DOM,
                              const Real * exact_volume,
                              const Real * exact_surface) const
    {
        // VARIABLES ==================================================
        Vector<Real> computed_volume(N_DOM);
        Vector<Real> computed_volume_div(N_DOM);
        Vector<Real> computed_surface(N_DOM);
        // ============================================================

        // INITIALIZATION =============================================
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            computed_volume[dom] = 0.0;
            computed_volume_div[dom] = 0.0;
            computed_surface[dom] = 0.0;
        }
        // ============================================================

        // COMPUTE VALUES =============================================
        for (int lev = 0; lev <= this->max_level; ++lev)
        {
            const iMultiFab & mask = this->masks[lev];

            Vector<Real> tmp_computed_volume(N_DOM);
            Vector<Real> tmp_computed_volume_div(N_DOM);
            Vector<Real> tmp_computed_surface(N_DOM);
            
            if (this->LevelIsValid(lev))
            {
                this->meshes[lev]->EvalVolumesAndSurfaces(N_DOM, mask,
                                                          tmp_computed_volume.data(),
                                                          tmp_computed_volume_div.data(),
                                                          tmp_computed_surface.data());

                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    computed_volume[dom] += tmp_computed_volume[dom];
                    computed_volume_div[dom] += tmp_computed_volume_div[dom];
                    computed_surface[dom] += tmp_computed_surface[dom];
                }
            }
        }
        // ============================================================

        // PRINT REPORT ===============================================
        Print() << "IMPLICIT-MESH QUADRATURE REPORT:" << std::endl;
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            Real err;

            Print() << "| Domain " << dom << std::endl;
            
            Print() << "|  Volume: " << std::endl;
            
            Print() << "|  - Reference: " << exact_volume[dom] << std::endl;

            err = 100.0*std::abs(computed_volume[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via dom. quadrature: " << computed_volume[dom] << ", error (%): " << err << std::endl;
            
            err = 100.0*std::abs(computed_volume_div[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via bou. quadrature: " << computed_volume_div[dom] << ", error (%): " << err << std::endl;

            Print() << "|  Surface: " << std::endl;

            if (exact_surface[dom] == 0.0)
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = std::abs(computed_surface[dom]-exact_surface[dom]);
                Print() << "|  - Computed: " << computed_surface[dom] << ", error: " << err << std::endl;
            }
            else
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = 100.0*std::abs(computed_surface[dom]-exact_surface[dom])/exact_surface[dom];
                Print() << "|  - Computed: " << computed_surface[dom] << ", error (%): " << err << std::endl;
            }
        }
        Print() << std::endl;
        // ============================================================
    }

    /**
     * \brief Check the quadrature rules by comparing the domains' volume and surface with exact values.
     *
     * \param[in] exact_volume: exact value of the volume of the domain.
     * \param[in] exact_surface: exact value of the surface of the domain.
     *
    */
    void CheckQuadratureRules(const Real exact_volume,
                              const Real exact_surface) const
    {
        const int N_DOM = 1;
        
        this->CheckQuadratureRules(N_DOM, &exact_volume, &exact_surface);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################



// ####################################################################
// AUXILIARY ROUTINES #################################################
// ####################################################################
AMREX_GPU_HOST_DEVICE
template <typename L>
void CoarseFineFaceLoop(const IntVect & rr, const int dir,
                        const int c_i, const int c_j, const int c_k,
                        L && f)
{
#if (AMREX_SPACEDIM == 1)
    const int rr_1 = rr[0];
    const int f_i_0 = c_i*rr_1;
    
Print() << "CoarseFineFaceLoop DIM = 1" << std::endl;
exit(-1);

#endif
#if (AMREX_SPACEDIM == 2)
    const int rr_1 = rr[0], rr_2 = rr[1];
    const int f_i_0 = c_i*rr_1, f_j_0 = c_j*rr_2, f_k_0 = c_k;

    switch (dir)
    {
        case 0:
        {
            for (int f_j = 0; f_j < rr_2; ++f_j)
            {
                f(f_i_0, f_j_0+f_j, f_k_0);
            }
            break;
        }
        case 1:
        {
            for (int f_i = 0; f_i < rr_1; ++f_i)
            {
                f(f_i_0+f_i, f_j_0, f_k_0);
            }
            break;
        }
    }

#endif
#if (AMREX_SPACEDIM == 3)
    const int rr_1 = rr[0], rr_2 = rr[1], rr_3 = rr[2];
    const int f_i_0 = c_i*rr_1, f_j_0 = c_j*rr_2, f_k_0 = c_k*rr_3;

Print() << "CoarseFineFaceLoop DIM = 3" << std::endl;
exit(-1);

    switch (dir)
    {
        case 0:
        {
            break;
        }
        case 1:
        {
            break;
        }
        case 2:
        {
            break;
        }
    }
#endif
}
// ####################################################################
// ####################################################################


// ####################################################################
// INTERPOLATION/RESTRICTION ##########################################
// ####################################################################
void Interpolate(const int N_SOL,
                 const Gpu::ManagedVector<int> & Sol2Dom,
                 const ImplicitMesh & c_mesh,
                 const MatrixFactory & c_matfactory,
                 const MultiFab & c_X,
                 const IntVect rr,
                 const ImplicitMesh & f_mesh,
                 const MatrixFactory & f_matfactory,
                 MultiFab & f_X);

void Restrict(const int N_SOL,
              const Gpu::ManagedVector<int> & Sol2Dom,
              const ImplicitMesh & f_mesh,
              const MatrixFactory & f_matfactory,
              const MultiFab & f_X,
              const IntVect rr,
              const ImplicitMesh & c_mesh,
              const MatrixFactory & c_matfactory,
              const iMultiFab & c_mask,
              MultiFab & c_X);

// ####################################################################
// ####################################################################

} // namespace AMR
} // namespace DG
} // namespace amrex

#endif