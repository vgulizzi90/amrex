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

    Vector<charMultiFab> cached_tags;
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

        this->cached_tags.resize(n_levels);
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
     * \brief Copy tags from a multifab into a TagBoxArray.
     *
     * \param[out] dst: the TagBoxArray where information will be copied.
     * \param[in] src: the input multifab.
     *
    */
    template <typename TAG_MF>
    void CopyTagging(TagBoxArray & dst, const TAG_MF & src) const
    {
        for (MFIter mfi(dst); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            auto const & dst_fab = dst.array(mfi);
            auto const & src_fab = src.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                dst_fab(i,j,k) = src_fab(i,j,k);
            });
            Gpu::synchronize();
        }
    }
    
    /**
     * \brief Tag cells that n_grow cells away from currently tagged cells.
     *
     * \param[in] mesh: ImplicitMesh data structure containing element type information.
     * \param[inout] tags: an object containing tagging information.
     * \param[in] tagval: the value used to tag the cells.
     * \param[in] n_grow: width of layer of cells that will be tagged.
     *
    */
    template <typename TAG_MF, typename TAG_T>
    void GrowTagging(const ImplicitMesh & mesh, TAG_MF & tags, const TAG_T tagval, const int n_grow) const
    {
        TAG_MF tags_helper(tags.boxarray, tags.distributionMap, 1, tags.n_grow);

        for (int it = 0; it < n_grow; ++it)
        {
            Copy(tags_helper, tags, 0, 0, 1, tags.n_grow);

            for (MFIter mfi(tags); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                const auto & old_tags_fab = tags_helper.array(mfi);
                auto const & tags_fab = tags.array(mfi);

                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // ELEMENT TYPE
                    const bool elm_is_not_tagged = (tags_fab(i,j,k) != tagval);

                    if (elm_is_not_tagged)
                    {
                        for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                        {
                            const int nbr_i = i+extended_stencil_table_i[n];
                            const int nbr_j = j+extended_stencil_table_j[n];
                            const int nbr_k = k+extended_stencil_table_k[n];
                            const bool nbr_is_tagged = (old_tags_fab(nbr_i,nbr_j,nbr_k) == tagval);

                            if (nbr_is_tagged)
                            {
                                tags_fab(i,j,k) = tagval;
                            }
                        }
                    }
                });
                Gpu::synchronize();
            }
            tags.FillBoundary(mesh.geom.periodicity());
        }
    }

    /**
     * \brief Tag i) small cells merged with tagged cells and ii) cells that extend onto tagged small cells.
     *
     * \param[in] mesh: ImplicitMesh data structure containing element type information.
     * \param[inout] tags: an object containing tagging information.
     * \param[in] tagval: the value used to tag the cells.
     *
    */
    template <typename TAG_MF, typename TAG_T>
    void TagSmallOrExtendedElements(const ImplicitMesh & mesh, TAG_MF & tags, const TAG_T tagval) const
    {
        // PARAMETERS =================================================
        const int N_DOM = mesh.eType.n_comp/__DG_ELM_TYPE_N_COMP_PER_DOM__;
        // ============================================================

        // TAG CELLS EXTENDING ONTO TAGGED SMALL CELLS ================
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            auto const & tags_fab = tags.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    // ELEMENT TYPE
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_not_tagged = (tags_fab(i,j,k) != tagval);

/*
if (i == 11 && j == 11 && k == 11)
{
Print() << ELM_TYPE_DESCRIPTION(etype) << std::endl;
Print() << "elm_is_tagged: " << (elm_is_not_tagged ? "no" : "yes") << std::endl;
}
*/

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

        // TAG SMALL CELLS THAT ARE MERGED WITH TAGGED CELLS ==========
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            auto const & tags_fab = tags.array(mfi);

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
/*
if (i == 12 && j == 11 && k == 11)
{
Print() << ELM_TYPE_DESCRIPTION(etype) << std::endl;
Print() << "nbr_is_tagged: " << (nbr_is_tagged ? "yes" : "no") << std::endl;
}
*/
                        if (nbr_is_tagged)
                        {
                            tags_fab(i,j,k) = tagval;
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        tags.FillBoundary(mesh.geom.periodicity());
        // ============================================================
    }

    /**
     * \brief Fix the tagging in such a way that merging does not happen across levels.
     *
     * \param[in] mesh: ImplicitMesh data structure containing element type information.
     * \param[inout] tags: an object containing tagging information.
     * \param[in] tagval: the value used to tag the cells.
     *
    */
    template <typename TAG_MF, typename TAG_T>
    void FixTagging(const ImplicitMesh & mesh, TAG_MF & tags, const TAG_T tagval) const
    {
        this->TagSmallOrExtendedElements(mesh, tags, tagval);

        //IO::PrintMultiFabEntry(tags, 11, 11, 11, 0);
        //IO::PrintMultiFabEntry(tags, 12, 11, 11, 0);

        // REPORT TAGGED CELLS ========================================
        /*
        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            const auto & tags_fab = tags.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const bool elm_is_tagged = (tags_fab(i,j,k) == tagval);

                if (elm_is_tagged)
                {
                    Print() << "tagged(" << i << "," << j << "," << k << ")" << std::endl;
                }
            });
            Gpu::synchronize();
        }
        */
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



// ####################################################################
// ERROR EVALUATION ###################################################
// ####################################################################
/**
 * \brief Compute the error between the exact solution and the AMR solution.
 *
 * \param[in] t: time.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: an AMR object that contains the solution at the different mesh levels.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real EvalError(const Real t,
               const int N_DOM,
               const int N_SOL,
               const AMR_CLASS & AMR,
               const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    const int finest_level = AMR.maxLevel();
    // ================================================================

    // VARIABLES ======================================================
    Real max_error = 0.0;
    // ================================================================

    // CYCLE OVER THE LEVELS ==========================================
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (AMR.LevelIsValid(lev))
        {
            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MatrixFactory & matfactory = *AMR.matfactories[lev];
            //const iMultiFab & mask = AMR.masks[lev];
            const MultiFab & X = AMR.Xs[lev];

            Real level_error;

            if (matfactory.std_elem.p == 0)
            {
                level_error = EvalError_FV(t, mesh, N_DOM, N_SOL, X, IBVP);
            }
            else
            {
                level_error = EvalError(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
            }

            max_error += level_error;
        }
    }
    // ================================================================

    return max_error;
}

/**
 * \brief Compute the error between the exact solution and the AMR solution.
 *
 * \param[in] t: time.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: an AMR object that contains the solution at the different mesh levels.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real EvalError(const Real t,
               const int N_SOL,
               const AMR_CLASS & AMR,
               const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    return EvalError(t, N_DOM, N_SOL, AMR, IBVP);
}

/**
 * \brief Compute the error between the exact solution and the AMR solution using infinite norm.
 *
 * \param[in] t: time.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: an AMR object that contains the solution at the different mesh levels.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real EvalErrorInfNorm(const Real t,
                      const int N_DOM,
                      const int N_SOL,
                      const AMR_CLASS & AMR,
                      const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    const int finest_level = AMR.maxLevel();
    // ================================================================

    // VARIABLES ======================================================
    Real max_error = 0.0;
    // ================================================================

    // CYCLE OVER THE LEVELS ==========================================
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (AMR.LevelIsValid(lev))
        {
            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MatrixFactory & matfactory = *AMR.matfactories[lev];
            //const iMultiFab & mask = AMR.masks[lev];
            const MultiFab & X = AMR.Xs[lev];

            Real level_error;

            if (matfactory.std_elem.p == 0)
            {
                level_error = EvalErrorInfNorm_FV(t, mesh, N_DOM, N_SOL, X, IBVP);
            }
            else
            {
                level_error = EvalErrorInfNorm(t, mesh, matfactory, N_DOM, N_SOL, X, IBVP);
            }

            max_error = std::max(max_error, level_error);
        }
    }
    // ================================================================

    return max_error;
}

/**
 * \brief Compute the error between the exact solution and the AMR solution using infinite norm.
 *
 * \param[in] t: time.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: an AMR object that contains the solution at the different mesh levels.
 * \param[in] IBVP: a class that must have a method with the following prototype:
 *
 *                  Real F_ERROR(const int dom, const Real t, const Real * x, const Real * SOL, Real & err_x, Real & norm_x) const;
 *
 * \return the error.
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
Real EvalErrorInfNorm(const Real t,
                      const int N_SOL,
                      const AMR_CLASS & AMR,
                      const IBVP_CLASS & IBVP)
{
    const int N_DOM = 1;
    return EvalErrorInfNorm(t, N_DOM, N_SOL, AMR, IBVP);
}
// ####################################################################
// ####################################################################


// ####################################################################
// AUXILIARY FUNCTIONS ################################################
// ####################################################################
void CheckMergeLeaking(const ImplicitMesh & mesh, const int N_DOM, const iMultiFab & mask);
// ####################################################################
// ####################################################################

} // namespace AMR
} // namespace DG
} // namespace amrex

#endif