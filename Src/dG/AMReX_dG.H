//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG.H
 * \brief A file that loads all header files.
*/

#include <AMReX.H>

#include <AMReX_dG_Constants.H>
#include <AMReX_dG_Tables.H>
#include <AMReX_dG_Utils.H>
#include <AMReX_VTK_Utils.H>
#include <AMReX_dG_Pardiso.H>
#include <AMReX_dG_LinearDirectSolver.H>
#include <AMReX_dG_BasisFunctions.H>
#include <AMReX_dG_LinAlg.H>
#include <AMReX_dG_StandardElements.H>
#include <AMReX_dG_Cell_BFX.H>
#include <AMReX_dG_Mesh.H>
#include <AMReX_dG_Mesh_Output.H>
#include <AMReX_dG_BFX.H>
#include <AMReX_dG_Solution.H>
#include <AMReX_dG_Solution_Utils.H>
#include <AMReX_dG_Solution_RungeKutta.H>
#include <AMReX_dG_Solution_CentralDifference.H>
#include <AMReX_dG_Solution_Newmark.H>
#include <AMReX_dG_Solution_Output.H>
#include <AMReX_dG_AMR.H>
#include <AMReX_dG_AMR_Utils.H>