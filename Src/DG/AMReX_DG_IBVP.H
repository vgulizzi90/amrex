//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_IBVP.H
 * \brief Contains constants and data structures for Initial-Boundary Value problems.
*/

#ifndef AMREX_DG_IBVP_H_
#define AMREX_DG_IBVP_H_

namespace amrex
{
namespace DG
{

/**
 * \brief Base Initial-Boundary Value problem class.
 *
*/
struct IBVP_BASE
{
    // LEVEL SET FUNCTIONS AND IMPLICITLY-DEFINED DOMAINS #############
    /**
     * \brief Fill PHI with values of the level sets at (t,x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] PHI: pointer to memory that will be filled with the value of the level sets at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_PHI(const amrex::Real /*t*/, const amrex::Real * /*x*/, amrex::Real * /*PHI*/) const
    {
        Abort("IBVP_BASE.F_PHI: We should have not ended up in here!");
    }

    /**
     * \brief Fill phi_info with information about which domain is associated to which level set.
     * 
     * \param[in] dom: domain index.
     * \param[out] phi_info: pointer to memory that will be filled domain info. More specifically:
     *
     *             phi_info[0] = ph means that the dom-th domain is defined by the ph-th levelset;
     *             phi_info[1] = -1 means that the dom-th domain is defined where the ph-th levelset is negative;
     *             phi_info[1] = +1 means that the dom-th domain is defined where the ph-th levelset is positive;
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_DOM2PHI(const int /*dom*/, int * /*phi_info*/) const
    {
        Abort("IBVP_BASE.F_PHI: We should have not ended up in here!");
    }

    /**
     * \brief Returns the index nbr_dom of the domain that is neighbor with the current domain.
     * 
     * \param[in] dom: index of the current domain.
     *
     * \return the index nbr_dom of the domain that is neighbor with domain dom.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int F_DOM2NBRDOM(const int /*dom*/) const
    {
        Abort("IBVP_BASE.F_DOM2NBRDOM: We should have not ended up in here!");
        return -1;
    }
    // ################################################################

    // TIME STEP ######################################################
    /**
     * \brief Eval the maximum time step.
     *
     * \param[in] dom: index of the considered domain.
     * \param[in] dx: pointer to memory containing the sizes of the grid's element.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables at (t,x).
     *
     * \return the maximum allowed time step.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real F_DT(const int /*dom*/,
                     const amrex::Real * /*dx*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/,
                     const amrex::Real * /*U*/) const
    {
        Abort("IBVP_BASE.F_DT: We should have not ended up in here!");
        return 0.0;
    }
    // ################################################################
};


namespace Hyperbolic
{

namespace Explicit
{

/**
 * \brief Base Initial-Boundary Value problem class for explicit hyperbolic problems.
 *
*/
struct IBVP
:
public IBVP_BASE
{
};


} // namespace Explicit

} // namespace Hyperbolic


} // namespace DG
} // namespace amrex

#endif