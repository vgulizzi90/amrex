//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Constants.H
 * \brief Definitions, contants and typedefs.
*/

#ifndef AMREX_DG_CONSTANTS_H_
#define AMREX_DG_CONSTANTS_H_

#include <algoim_quad.hpp>

namespace amrex
{
namespace dG
{

#ifdef WIN32
#define OS_SEP '\\'
#else
#define OS_SEP '/'
#endif

#define AMREX_DG_CHECK_FOR_NAN
#define AMREX_DG_CHECK_FOR_LARGE_NUMBERS
#ifdef AMREX_DG_CHECK_FOR_LARGE_NUMBERS
#define __DG_LARGE_NUMBER__ 1.0e8
#endif

// TYPEDEFS ###########################################################
// MULTIFABS ==========================================================
typedef FabArray<BaseFab<char>> cMultiFab;
typedef FabArray<BaseFab<short int>> sMultiFab;
typedef FabArray<BaseFab<unsigned int>> uMultiFab;
typedef FabArray<BaseFab<long>> lMultiFab;
// ====================================================================


// ALGOIM =============================================================
typedef Algoim::QuadratureRule<AMREX_SPACEDIM> AlgoimQuadRule;
// ====================================================================
// ####################################################################

} // namespace dG
} // namespace amrex

#endif