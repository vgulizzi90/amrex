//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG.H
 * \brief A file that defines a bunch constants and loads all header files.
*/

#include <AMReX_DG_Constants.H>
#include <AMReX_DG_IO.H>
#include <AMReX_VTK_utils.H>
#include <AMReX_DG_Quadrature.H>
#include <AMReX_DG_BasisFunctions.H>
#include <AMReX_DG_LinAlg.H>
#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_BFX.H>
#include <AMReX_DG_IBVP.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_Solution.H>
#include <AMReX_DG_AMR.H>
#include <AMReX_DG_Solution_Hyperbolic.H>
#include <AMReX_DG_Solution_Parabolic.H>
#include <AMReX_DG_Solution_Output.H>