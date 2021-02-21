//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_IO.H
 * \brief Header file for some auxiliary input/output routines.
*/

#ifndef AMREX_DG_IO_H_
#define AMREX_DG_IO_H_

#include <string>

#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_MultiFab.H>

namespace amrex
{
namespace DG
{
namespace IO
{

std::string Seconds2HoursMinutesSeconds(const Real /*seconds*/);

void PrintInts(const int /*rank*/, const MPI_Comm, const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintInts(const int /*rank*/, const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintInts(const int /*n*/, const int *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*rank*/, const MPI_Comm, const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*rank*/, const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintReals(const int /*n*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*rank*/, const MPI_Comm, const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*rank*/, const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());
void PrintRealArray2D(const int /*Nr*/, const int /*Nc*/, const Real *, std::ostream & os = amrex::OutStream());

std::string MakeLocalOutputFilename(const std::string & /*filename*/,
                                    const std::string & /*file_fmt*/);
std::string MakeGlobalOutputFilename(const std::string & /*filename*/,
                                     const std::string & /*file_fmt*/);
std::string MakeLocalOutputFilepath(const std::string & /*folderpath_root*/,
                                    const int /*n*/,
                                    const int /*n_steps*/,
                                    const std::string & /*filename*/,
                                    const std::string & /*file*/);
std::string MakeGlobalOutputFilepath(const std::string & /*folderpath_root*/,
                                     const int /*n*/,
                                     const int /*n_steps*/,
                                     const std::string & /*filename*/,
                                     const std::string & /*file*/);

std::string MakePath(const std::initializer_list<std::string>);
void MakeFolder(const std::string &);
void MakeStepFolder(const std::string & /*folderpath_root*/, const int /*n*/, const int /*n_steps*/);

/**
 * \brief Print (ii,jj,kk,comp) of multifab.
*/
template <typename MF>
void PrintMultiFabEntry(const MF & mf, const int ii, const int jj, const int kk, const int comp, const std::string where = "")
{
    // PARAMETERS
    const int n_comp = mf.n_comp;

    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        auto fab = mf.array(mfi);

        ParallelFor(bx, n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int u) noexcept
        {
            if (i == ii && j == jj && k == kk && u == comp)
            {
                Print() << where << "mf(" << i << "," << j << "," << k << "," << u << "): " << (fab(i,j,k,u)) << std::endl;
            }
        });
        Gpu::synchronize();
    }
}

/**
 * \brief Report nan components of a multifab.
*/
template <typename MF>
void PrintNaNs(const MF & mf)
{
    // PARAMETERS
    const int n_comp = mf.n_comp;

    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        auto fab = mf.array(mfi);

        ParallelFor(bx, n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int u) noexcept
        {
            if (std::isnan(fab(i,j,k,u)))
            {
                Print() << "mf(" << i << "," << j << "," << k << "," << u << ") is nan." << std::endl;
            }
        });
        Gpu::synchronize();
    }
}


} // namespace IO
} // namespace DG
} // namespace amrex

#endif