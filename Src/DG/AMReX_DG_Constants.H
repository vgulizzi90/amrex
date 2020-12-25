//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Constants.H
 * \brief Definitions, contants and typedefs.
*/

#include <algoim_quad.hpp>

#ifndef AMREX_DG_CONSTANTS_H_
#define AMREX_DG_CONSTANTS_H_


namespace amrex
{
namespace DG
{

// CONSTANTS ##########################################################
#define __DG_BASE_STENCIL_N_NBR__ AMREX_D_PICK(2, 4, 6)

#define __DG_STD_ELEM_N_SPACE_BOUNDARIES__ (2*AMREX_SPACEDIM)

#define __DG_PHI_MAX_SPACE_p__ 8
#if (AMREX_SPACEDIM == 1)
#define __DG_PHI_MAX_SPACE_Np__ (1+__DG_PHI_MAX_SPACE_p__)
#endif
#if (AMREX_SPACEDIM == 2)
#define __DG_PHI_MAX_SPACE_Np__ ((1+__DG_PHI_MAX_SPACE_p__)*(1+__DG_PHI_MAX_SPACE_p__))
#endif
#if (AMREX_SPACEDIM == 3)
#define __DG_PHI_MAX_SPACE_Np__ ((1+__DG_PHI_MAX_SPACE_p__)*(1+__DG_PHI_MAX_SPACE_p__)*(1+__DG_PHI_MAX_SPACE_p__))
#endif

#define __DG_SOL_MAX_SPACE_p__ 3
#if (AMREX_SPACEDIM == 1)
#define __DG_SOL_MAX_SPACE_Np__ (1+__DG_SOL_MAX_SPACE_p__)
#endif
#if (AMREX_SPACEDIM == 2)
#define __DG_SOL_MAX_SPACE_Np__ ((1+__DG_SOL_MAX_SPACE_p__)*(1+__DG_SOL_MAX_SPACE_p__))
#endif
#if (AMREX_SPACEDIM == 3)
#define __DG_SOL_MAX_SPACE_Np__ ((1+__DG_SOL_MAX_SPACE_p__)*(1+__DG_SOL_MAX_SPACE_p__)*(1+__DG_SOL_MAX_SPACE_p__))
#endif

#define __DG_MAX_N_PHI__ 10
#define __DG_MAX_N_SOL__ 10
#define __DG_MAX_N_FIELDS__ 18

// IMPLICIT MESH
#define __DG_ELM_TYPE_N_COMP_PER_DOM__ 1

#define __DG_ELM_TYPE_UNDEFINED__ -2
#define __DG_ELM_TYPE_EMPTY__ -1
#define __DG_ELM_TYPE_ENTIRE__ 1
#define __DG_ELM_TYPE_LARGE__ 2
#define __DG_ELM_TYPE_SMALL__ 3
#define __DG_ELM_TYPE_PARTIAL__ 5

#define __DG_ELM_BOU_TYPE_N_COMP_PER_DOM__ 1

#define __DG_ELM_BOU_TYPE_UNDEFINED__ -2
#define __DG_ELM_BOU_TYPE_EMPTY__ -1
#define __DG_ELM_BOU_TYPE_ENTIRE__ 1
#define __DG_ELM_BOU_TYPE_PARTIAL__ 5
#define __DG_ELM_BOU_TYPE_PARTIAL_INVALID__ 6

#define __DG_ELM_INFO_N_COMP_PER_DOM__ (2+AMREX_SPACEDIM)
#define __DG_ELM_BOU_INFO_N_COMP_PER_DOM__ (AMREX_SPACEDIM)
#define __DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__ (AMREX_SPACEDIM+AMREX_SPACEDIM)

#define __DG_ELM_DOM_QUAD_N_COMP_PER_DOM__ 1
#define __DG_ELM_BOU_QUAD_N_COMP_PER_DOM__ 2
#define __DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__ 1

#define __DG_ELM_FV_INFO_N_COMP_PER_DOM__ 1

#define __DG_ELM_COLOR_N_COMP_PER_DOM__ 1

#define __DG_ELM_COLOR_UNDEFINED__ -1
// ####################################################################

// TYPEDEFS ###########################################################
typedef FabArray<BaseFab<char>> charMultiFab;
typedef FabArray<BaseFab<short>> shortMultiFab;
typedef FabArray<BaseFab<long>> longMultiFab;

// FOR ALGOIM
typedef Algoim::QuadratureRule<AMREX_SPACEDIM> iQuadRule;
// ####################################################################

// AUXILIARY CONSTANTS ################################################
// NEIGHBORS INFO IN STENCIL
const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> base_stencil_table_i =
{
#if (AMREX_SPACEDIM == 1)
    -1, +1
#endif
#if (AMREX_SPACEDIM == 2)
    -1, +1, 0, 0
#endif
#if (AMREX_SPACEDIM == 3)
    -1, +1, 0, 0, 0, 0
#endif
};
const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> base_stencil_table_j =
{
#if (AMREX_SPACEDIM == 1)
    0, 0
#endif
#if (AMREX_SPACEDIM == 2)
    0, 0, -1, +1
#endif
#if (AMREX_SPACEDIM == 3)
    0, 0, -1, +1, 0, 0
#endif
};
const GpuArray<int, __DG_BASE_STENCIL_N_NBR__> base_stencil_table_k = 
{
#if (AMREX_SPACEDIM == 1)
    0, 0
#endif
#if (AMREX_SPACEDIM == 2)
    0, 0, 0, 0
#endif
#if (AMREX_SPACEDIM == 3)
    0, 0, 0, 0, -1, +1
#endif
};
// ####################################################################


} // namespace DG
} // namespace amrex

#endif