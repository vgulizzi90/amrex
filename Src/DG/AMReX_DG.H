//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG.H
 * \brief A file that defines a bunch constants and loads all header files.
*/

// CONSTANTS ##########################################################
#define __DG_STD_ELEM_N_SPACE_BOUNDARIES__ (2*AMREX_SPACEDIM)

#define __DG_PHI_MAX_SPACE_p__ 3
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

#define __DG_MAX_N_SOL__ 10
#define __DG_MAX_N_FIELDS__ 10
// ####################################################################

#include <AMReX_VTK_utils.H>
#include <AMReX_DG_IO.H>
#include <AMReX_DG_Quadrature.H>
#include <AMReX_DG_BasisFunctions.H>
#include <AMReX_DG_LinAlg.H>
#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_BFX.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_Solution.H>
#include <AMReX_DG_Solution_Output.H>