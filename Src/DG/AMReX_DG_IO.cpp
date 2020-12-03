//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_IO.cpp
 * \brief Implementation of some auxiliary input/output routines.
*/

#include <AMReX_DG_IO.H>

namespace amrex
{
namespace DG
{
namespace IO
{
/**
 * \brief Print a list of n space-separated integers on process rank of communicator comm.
 *
 * \param[in] rank: process identifier;
 * \param[in] comm: communicator identifier;
 * \param[in] n: number of integers to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintInts(const int rank, const MPI_Comm comm, const int n, const int * x, std::ostream & os)
{
    for (int i = 0; i < (n-1); ++i)
    {
        Print(rank, comm, os) << x[i] << " ";
    }
    Print(rank, comm, os) << x[n-1];
}

/**
 * \brief Print a list of n space-separated integers on process rank of the default communicator.
 *
 * \param[in] rank: process identifier;
 * \param[in] n: number of integers to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintInts(const int rank, const int n, const int * x, std::ostream & os)
{
    PrintInts(rank, ParallelContext::CommunicatorSub(), n, x, os);
}

/**
 * \brief Print a list of n space-separated integers on I/O process of the default communicator.
 *
 * \param[in] n: number of integers to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintInts(const int n, const int * x, std::ostream & os)
{
    PrintInts(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), n, x, os);
}

/**
 * \brief Print a list of n space-separated reals on process rank of communicator comm.
 *
 * \param[in] rank: process identifier;
 * \param[in] comm: communicator identifier;
 * \param[in] n: number of reals to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintReals(const int rank, const MPI_Comm comm, const int n, const Real * x, std::ostream & os)
{
    for (int i = 0; i < (n-1); ++i)
    {
        Print(rank, comm, os) << std::scientific << std::setprecision(11) << std::setw(18) << std::showpos << x[i] << " ";
    }
    Print(rank, comm, os) << std::scientific << std::setprecision(11) << std::setw(18) << std::showpos << x[n-1];
}

/**
 * \brief Print a list of n space-separated reals on process rank of the default communicator.
 *
 * \param[in] rank: process identifier;
 * \param[in] n: number of reals to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintReals(const int rank, const int n, const Real * x, std::ostream & os)
{
    PrintReals(rank, ParallelContext::CommunicatorSub(), n, x, os);
}

/**
 * \brief Print a list of n space-separated reals on I/O process of the default communicator.
 *
 * \param[in] n: number of reals to be printed;
 * \param[in] x: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintReals(const int n, const Real * x, std::ostream & os)
{
    PrintReals(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), n, x, os);
}

/**
 * \brief Print an array or reals with column-major order on process rank of communicator comm.
 *
 * \param[in] rank: process identifier;
 * \param[in] comm: communicator identifier;
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] a: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintRealArray2D(const int rank, const MPI_Comm comm, const int Nr, const int Nc, const Real * a, std::ostream & os)
{
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c)
        {
            Print(rank, comm, os) << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << a[r+c*Nr] << " ";
        }
        Print(rank, comm, os) << std::endl;
    }
}

/**
 * \brief Print an array or reals with column-major order on process rank of the default
 *        communicator.
 *
 * \param[in] rank: process identifier;
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] a: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintRealArray2D(const int rank, const int Nr, const int Nc, const Real * a, std::ostream & os)
{
    PrintRealArray2D(rank, ParallelContext::CommunicatorSub(), Nr, Nc, a, os);
}

/**
 * \brief Print an array or reals with column-major order on I/O process of the default
 *        communicator.
 *
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] a: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void PrintRealArray2D(const int Nr, const int Nc, const Real * a, std::ostream & os)
{
    PrintRealArray2D(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), Nr, Nc, a, os);
}

/**
 * \brief Build a path from a set of input strings.
 *
 * \param[in] list: the set of input strings.
 *
 * \return a string containing the path.
*/
std::string MakePath(const std::initializer_list<std::string> list)
{
    int cnt;
    std::string path;

    cnt = 0;
    for (const std::string & str : list)
    {
        if (cnt == 0)
        {
            path += str;
        }
        else
        {
            if (str.front() == '/')
            {
                path += str;
            }
            else
            {
                path += '/'+str;
            }
        }

        if (str.back() == '/')
        {
            path.pop_back();
        }

        cnt += 1;
    }

    return path;
}

/**
 * \brief Make a string containing the filename_root suffixed with rank and format.
 *
 * \param[in] filename: name of the file without rank and format.
 * \param[in] file_fmt: format appended to the filename.
 *
 * \return a string containing the output filename.
*/
std::string MakeLocalOutputFilename(const std::string & filename,
                                    const std::string & file_fmt)
{
    const int proc_id = ParallelDescriptor::MyProc();

    return filename+"_proc_"+std::to_string(proc_id)+"."+file_fmt;
}

/**
 * \brief Make a string containing the filename_root suffixed with format.
 *
 * \param[in] filename: name of the file without format.
 * \param[in] file_fmt: format appended to the filename.
 *
 * \return a string containing the output filename.
*/
std::string MakeGlobalOutputFilename(const std::string & filename,
                                     const std::string & file_fmt)
{
    return filename+"."+file_fmt;
}

/**
 * \brief Make a filepath where the folder is suffixed with index denoting the time step.
 *
 * \param[in] folderpath_root: root of the folder path; it will be suffixed with an index denoting
 *                             the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename: name of the file without format.
 * \param[in] file_fmt: format appended to the filename.
 *
 * \return a string containing the output filepath.
*/
std::string MakeLocalOutputFilepath(const std::string & folderpath_root,
                                    const int n,
                                    const int n_steps,
                                    const std::string & filename,
                                    const std::string & file_fmt)
{
    int ndigits;
    std::string folderpath;

    ndigits = 1;
    while (std::pow(10, ndigits) < n_steps) ndigits += 1;

    folderpath = Concatenate(folderpath_root+"/", n, ndigits);

    return MakePath({folderpath, MakeLocalOutputFilename(filename, file_fmt)});
}

/**
 * \brief Make a filepath where the folder is suffixed with index denoting the time step.
 *
 * \param[in] folderpath_root: root of the folder path; it will be suffixed with an index denoting
 *                             the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename: name of the file without format.
 * \param[in] file_fmt: format appended to the filename.
 *
 * \return a string containing the output filepath.
*/
std::string MakeGlobalOutputFilepath(const std::string & folderpath_root,
                                     const int n, 
                                     const int n_steps,
                                     const std::string & filename,
                                     const std::string & file_fmt)
{
    int ndigits;
    std::string folderpath;

    ndigits = 1;
    while (std::pow(10, ndigits) < n_steps) ndigits += 1;

    folderpath = Concatenate(folderpath_root+"/", n, ndigits);

    return MakePath({folderpath, MakeGlobalOutputFilename(filename, file_fmt)});
}

/**
 * \brief Make folder at the input folderpath.
 *
 * \param[in] folderpath: input path of the folder.
*/
void MakeFolder(const std::string & folderpath)
{
    if (ParallelDescriptor::IOProcessor())
    {
        if (!UtilCreateDirectory(folderpath, 0755))
        {
            CreateDirectoryFailed(folderpath);
        }
    }
    ParallelDescriptor::Barrier();
}

/**
 * \brief Make folder at the input folderpath accounting for current step and total number of steps.
 *
 * \param[in] folderpath_root: input path of the folder.
 * \param[in] n: current step.
 * \param[in] n_steps: total number of steps.
*/
void MakeStepFolder(const std::string & folderpath_root, const int n, const int n_steps)
{
    int ndigits;
    std::string folderpath;

    ndigits = 1;
    while (std::pow(10, ndigits) < n_steps) ndigits += 1;

    folderpath = Concatenate(folderpath_root+"/", n, ndigits);

    if (ParallelDescriptor::IOProcessor())
    {
        if (!UtilCreateDirectory(folderpath, 0755))
        {
            CreateDirectoryFailed(folderpath);
        }
    }
    ParallelDescriptor::Barrier();
}

/**
 * \brief Print (ii,jj,kk,comp) of multifab.
*/
void PrintMultiFabEntry(const MultiFab & mf, const int ii, const int jj, const int kk, const int comp)
{
    // PARAMETERS
    const int n_comp = mf.n_comp;

    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        Array4<Real const> const & fab = mf.array(mfi);

        ParallelFor(bx, n_comp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int u) noexcept
        {
            if (i == ii && j == jj && k == kk && u == comp)
            {
                Print() << "mf(" << i << "," << j << "," << k << "," << u << "): " << fab(i,j,k,u) << std::endl;
            }
        });
        Gpu::synchronize();
    }
}

} // namespace IO
} // namespace DG
} // namespace amrex