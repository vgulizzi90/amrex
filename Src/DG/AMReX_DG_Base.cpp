// AMReX_DG_Base.cpp

#include <AMReX_DG_Base.H>

namespace amrex
{
namespace DG
{


// ####################################################################
// EXPORT ROUTINES ####################################################
// ####################################################################
// BUILD THE STRING CONTAINING THE FILEPATH FOR A GENERIC PLOT FILE ===
std::string GetPlotFilepath(const std::string & dst_folder,
                            const std::string & filename_root,
                            const int & time_id,
                            const int & time_id_max)
{
    // VARIABLES ------------------
    int ndigits;
    std::string filename, filepath;
    // ----------------------------

    // BUILD THE STRING -----------------------------------------------
    ndigits = 1;
    while (std::pow(10, ndigits) < time_id_max) ndigits += 1;

    filename = Concatenate(filename_root+"_plt_", time_id, ndigits);
    filepath = dst_folder+filename;
    // ----------------------------------------------------------------

    return filepath;
}
// ====================================================================

// BUILD THE STRING CONTAINING THE FILEPATH FOR A GENERIC OUTPUT FILE =
std::string GetOutputFilepath(const std::string & dst_folder,
                              const std::string & filename_root,
                              const int & time_id,
                              const int & time_id_max)
{
    // PARAMETERS --------------------------------
    const int rank = ParallelDescriptor::MyProc();
    // -------------------------------------------

    // VARIABLES ------------------
    int ndigits;
    std::string filename, filepath;
    // ----------------------------

    // BUILD THE STRING -----------------------------------------------
    ndigits = 1;
    while (std::pow(10, ndigits) < time_id_max) ndigits += 1;

    filename = Concatenate(filename_root+"_proc_"+std::to_string(rank)+"_", time_id, ndigits);
    filepath = dst_folder+filename;
    // ----------------------------------------------------------------

    return filepath;
}
// ====================================================================

// VTK RELATED ROUTINES ===============================================
Gpu::ManagedVector<int> Get_VTK_BaseCell_Connectivity(const int & ne)
{
    // PARAMETERS -----------------------------------------------------
#if (AMREX_SPACEDIM != 1)
    const int nn = ne+1;
#endif
    const int cell_n_hyperrects = AMREX_D_PICK(ne, ne*ne, ne*ne*ne);
    const int cell_conn_len = AMREX_D_PICK(2, 4, 8)*cell_n_hyperrects;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Gpu::ManagedVector<int> cell_conn(cell_conn_len);
    // ----------------------------------------------------------------

#if (AMREX_SPACEDIM == 1)
    for (int i = 0; i < ne; ++i)
    {
        cell_conn[2*i] = i;
        cell_conn[2*i+1] = i+1;
    }
#endif
#if (AMREX_SPACEDIM == 2)
    int c;

    for (int j = 0; j < ne; ++j)
    for (int i = 0; i < ne; ++i)
    {
        c = i+j*ne;
        cell_conn[4*c+0] = i+j*nn;
        cell_conn[4*c+1] = (i+1)+j*nn;
        cell_conn[4*c+2] = (i+1)+(j+1)*nn;
        cell_conn[4*c+3] = i+(j+1)*nn;
    }
#endif
#if (AMREX_SPACEDIM == 3)
    int c;

    for (int k = 0; k < ne; ++k)
    for (int j = 0; j < ne; ++j)
    for (int i = 0; i < ne; ++i)
    {
        c = i+j*ne+k*ne*ne;
        cell_conn[8*c+0] = i+j*nn+k*nn*nn;
        cell_conn[8*c+1] = (i+1)+j*nn+k*nn*nn;
        cell_conn[8*c+2] = (i+1)+(j+1)*nn+k*nn*nn;
        cell_conn[8*c+3] = i+(j+1)*nn+k*nn*nn;
        cell_conn[8*c+4] = i+j*nn+(k+1)*nn*nn;
        cell_conn[8*c+5] = (i+1)+j*nn+(k+1)*nn*nn;
        cell_conn[8*c+6] = (i+1)+(j+1)*nn+(k+1)*nn*nn;
        cell_conn[8*c+7] = i+(j+1)*nn+(k+1)*nn*nn;
    }
#endif
    return cell_conn;
}

Gpu::ManagedVector<int> Get_VTK_BaseBouCell_Connectivity(const int & ne)
{
    // PARAMETERS -----------------------------------------------------
#if (AMREX_SPACEDIM == 3)
    const int nn = ne+1;
#endif
    const int cell_n_hyperrects = AMREX_D_PICK(1, ne, ne*ne);
    const int cell_bou_conn_len = AMREX_D_PICK(1, 2*cell_n_hyperrects, 4*cell_n_hyperrects);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Gpu::ManagedVector<int> bou_cell_conn(cell_bou_conn_len);
    // ----------------------------------------------------------------

#if (AMREX_SPACEDIM == 1)
    bou_cell_conn[0] = 0;
#endif
#if (AMREX_SPACEDIM == 2)
    for (int i = 0; i < ne; ++i)
    {
        bou_cell_conn[2*i] = i;
        bou_cell_conn[2*i+1] = i+1;
    }
#endif
#if (AMREX_SPACEDIM == 3)
    int c;
    for (int j = 0; j < ne; ++j)
    for (int i = 0; i < ne; ++i)
    {
        c = i+j*ne;
        bou_cell_conn[4*c+0] = i+j*nn;
        bou_cell_conn[4*c+1] = (i+1)+j*nn;
        bou_cell_conn[4*c+2] = (i+1)+(j+1)*nn;
        bou_cell_conn[4*c+3] = i+(j+1)*nn;
    }
#endif

    return bou_cell_conn;
}
// ====================================================================
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex