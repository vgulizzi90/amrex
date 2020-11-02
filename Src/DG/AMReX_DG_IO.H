//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_IO.H
 * \brief Header file for some auxiliary input/output routines.
*/

#include <string>

#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_MultiFab.H>

#ifndef AMREX_DG_IO_H_
#define AMREX_DG_IO_H_

namespace amrex
{
namespace DG
{
namespace IO
{

void PrintInts(const int /*rank*/, const MPI_Comm, const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintInts(const int /*rank*/, const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintInts(const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*rank*/, const MPI_Comm, const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*rank*/, const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*rank*/, const MPI_Comm, const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*rank*/, const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());

std::string MakePath(const std::initializer_list<std::string>);
void MakeFolder(const std::string &);

void PrintMultiFabEntry(const MultiFab & /*mf*/, const int /*ii*/, const int /*jj*/, const int /*kk*/, const int /*comp*/);


} // namespace IO
} // namespace DG
} // namespace amrex

#endif