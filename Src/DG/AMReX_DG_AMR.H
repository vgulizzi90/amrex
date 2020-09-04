// AMREX_DG_AMR.H

#ifndef BL_DG_AMR_H_
#define BL_DG_AMR_H_

#include <AMReX_AmrCore.H>

namespace amrex
{
namespace DG
{

// ####################################################################
// ADAPTIVE MESH REFINEMENT DATA STRUCTURE ############################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct AMR
:
public AmrCore
{
    // DATA MEMBERS ###################################################
    // ################################################################

    // CONSTRUCTOR ####################################################
    AMR(const Geometry & geom_, const AmrInfo & amr_info)
    :
    AmrCore(geom_, amr_info)
    {}
    // ################################################################

    // MAKE NEW GRIDS FROM SCRATCH ####################################
    void InitData()
    {
        const Real time = 0.0;
        this->InitFromScratch(time);
    }
    // ################################################################

    // OVERRIDE VIRTUAL: MAKE NEW LEVEL FROM COARSE ###################
    virtual
    void MakeNewLevelFromCoarse(const int lev, const Real time,
                                const BoxArray & ba,
                                const DistributionMapping & dm) override
    {
Print() << "HERE WE ARE: AmrCore.MakeNewLevelFromCoarse" << std::endl;
exit(-1);
    }
    // ################################################################

    // OVERRIDE VIRTUAL: REMAKE LEVEL #################################
    virtual
    void RemakeLevel(const int lev, const Real time,
                     const BoxArray & ba,
                     const DistributionMapping & dm) override
    {
Print() << "HERE WE ARE: AmrCore.RemakeLevel" << std::endl;
exit(-1);
    }
    // ################################################################

    // OVERRIDE VIRTUAL: CLEAR LEVEL ##################################
    virtual
    void ClearLevel(const int lev) override
    {
Print() << "HERE WE ARE: AmrCore.ClearLevel" << std::endl;
exit(-1);
    }
    // ################################################################

    // OVERRIDE VIRTUAL: MAKE NEW LEVEL FROM SCRATCH ##################
    virtual
    void MakeNewLevelFromScratch(const int lev, const Real time,
                                 const BoxArray & ba,
                                 const DistributionMapping & dm) override
    {
Print() << "HERE WE ARE: AmrCore.MakeNewLevelFromScratch" << std::endl;
exit(-1);
    }
    // ################################################################

    // OVERRIDE VIRTUAL: MARK CELLS FOR REFINEMENT ####################
    virtual
    void ErrorEst(const int lev, TagBoxArray & tags, const Real time, const int ngr) override
    {
        this->TagCellsForRefinement(lev, tags, time, ngr);
    }
    void TagCellsForRefinement(const int lev, TagBoxArray & tags, const  Real time, const int ngr)
    {
Print() << "HERE WE ARE: AmrCore.TagCellsForRefinement" << std::endl;
exit(-1);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif