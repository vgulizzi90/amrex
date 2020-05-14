// AMReX_VTK_utils.cpp

#include <AMReX_VTK_utils.H>

namespace amrex
{
namespace VTK
{
// ####################################################################
// AUXILIARY ROUTINES FOR OUTPUT FILE/FOLDER CREATIONS ################
// ####################################################################
std::string MakeLocalOutputFilename(const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int & time_id,
                                    const int & time_id_max)
{
    // PARAMETERS ------------------------------------------
    const int proc_id = amrex::ParallelDescriptor::MyProc();
    // -----------------------------------------------------

    // VARIABLES --------
    int ndigits;
    std::string filename;
    // ------------------

    // BUILD THE STRING -----------------------------------------------
    ndigits = 1;
    while (std::pow(10, ndigits) < time_id_max) ndigits += 1;

    filename = amrex::Concatenate(filename_root+"_proc_"+std::to_string(proc_id)+"_", time_id, ndigits);
    filename += "."+file_fmt;
    // ----------------------------------------------------------------

    return filename;
}

std::string MakeGlobalOutputFilename(const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int & time_id,
                                     const int & time_id_max)
{
    // VARIABLES --------
    int ndigits;
    std::string filename;
    // ------------------

    // BUILD THE STRING -----------------------------------------------
    ndigits = 1;
    while (std::pow(10, ndigits) < time_id_max) ndigits += 1;

    filename = amrex::Concatenate(filename_root+"_", time_id, ndigits);
    filename += "."+file_fmt;
    // ----------------------------------------------------------------

    return filename;
}

std::string MakeLocalOutputFilepath(const std::string & dst_folder,
                                    const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int & time_id,
                                    const int & time_id_max)
{
    std::string filepath = dst_folder+"/"+MakeLocalOutputFilename(filename_root, file_fmt, time_id, time_id_max);

    return filepath;
}

std::string MakeGlobalOutputFilepath(const std::string & dst_folder,
                                     const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int & time_id,
                                     const int & time_id_max)
{
    std::string filepath = dst_folder+"/"+MakeGlobalOutputFilename(filename_root, file_fmt, time_id, time_id_max);

    return filepath;
}
// ####################################################################
// ####################################################################



// ####################################################################
// VTU OUTPUT #########################################################
// ####################################################################
void PrintHeaderFile_VTU(const std::string & dst_folder,
                         const std::string & filename_root,
                         const int & time_id,
                         const int & time_id_max,
                         const amrex::Vector<std::string> & cell_field_name,
                         const amrex::Vector<std::string> & nodal_field_name)
{
    // PROFILE --------------------------------------------------------
    const std::string which_function = "PrintHeaderFile_VTU(const std::string &, const std::string &, ...)";
    BL_PROFILE(which_function);
    const std::string which_file = "PolyExa_IO_VTK.cpp";
    const std::string header = which_file+" - "+which_function;
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    // NUMBER OF PROCESSORS
    const int n_procs = amrex::ParallelDescriptor::NProcs();

    // FILEPATH
    const std::string headerpath = MakeGlobalOutputFilepath(dst_folder, filename_root, "pvtu", time_id, time_id_max);

    // WRITING TOOL
    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);

    // CELL DATA
    const int n_cell_fields = cell_field_name.size();

    // NODAL DATA
    const int n_nodal_fields = nodal_field_name.size();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    std::string filename;

    std::ofstream fp;
    // ----------------------------------------------------------------

    // OPEN FILE ------------------------------------------------------
    fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    fp.open(headerpath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!fp.good())
    {
        amrex::FileOpenFailed(headerpath);
    }
    // ----------------------------------------------------------------

    // WRITE FILE -----------------------------------------------------
    fp.precision(17);

    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"" << Header_t_description << "\">\n";
    fp << "<PUnstructuredGrid GhostLevel=\"0\">\n";
    
    fp << "  <PPoints>\n";
    fp << "    <PDataArray type=\"" << Float_t_description << "\" Name=\"Position\" NumberOfComponents=\"3\"/>\n"; 
    fp << "  </PPoints>\n";

    if (n_nodal_fields > 0)
    {
        fp << "  <PPointData Scalars=\"Scalars\">\n";
        for (int i = 0; i < n_nodal_fields; i++)
        {
            fp << "    <PDataArray type=\"" << Float_t_description << "\" Name=\""+nodal_field_name[i]+"\" NumberOfComponents=\"1\"/>\n";
        }
        fp << "  </PPointData>\n";
    }
    
    fp << "  <PCells>\n"; 
    fp << "    <PDataArray type=\"" << Cell_conn_t_description << "\" Name=\"connectivity\" NumberOfComponents=\"1\"/>\n";
    fp << "    <PDataArray type=\"" << Cell_offs_t_description << "\" Name=\"offsets\" NumberOfComponents=\"1\"/>\n";
    fp << "    <PDataArray type=\"" << Cell_type_t_description << "\" Name=\"types\" NumberOfComponents=\"1\"/>\n";
    fp << "  </PCells>\n";

    if (n_cell_fields > 0)
    {
        fp << "  <PCellData Scalars=\"Scalars\">\n";
        for (int i = 0; i < n_cell_fields; i++)
        {
            fp << "    <PDataArray type=\"" << Int_t_description << "\" Name=\""+cell_field_name[i]+"\" NumberOfComponents=\"1\"/>\n";
        }
        fp << "  </PCellData>\n";
    }

    for (int i = 0; i < n_procs; i++)
    {
        filename = MakeGlobalOutputFilename(filename_root+"_proc_"+std::to_string(i), "vtu", time_id, time_id_max);

        fp << "  <Piece Source=\"" << filename << "\"/>\n";
    }
    
    fp << "</PUnstructuredGrid>\n";
    
    fp << "</VTKFile>\n";
    
    fp << '\n';
    // ----------------------------------------------------------------

    // CLOSE FILE
    fp.close();
    // ----------
}

void PrintUnstructuredGridData_VTU(const std::string & filepath,
                                   const Cell_conn_t & n_nodes,
                                   const Cell_offs_t & n_cells,
                                   const amrex::Vector<Float_t> & nodes,
                                   const amrex::Vector<Cell_conn_t> & cell_conn,
                                   const amrex::Vector<Cell_offs_t> & cell_offset,
                                   const amrex::Vector<Cell_type_t> & cell_type,
                                   const amrex::Vector<amrex::Vector<Int_t>> & cell_field,
                                   const amrex::Vector<std::string> & cell_field_name,
                                   const amrex::Vector<amrex::Vector<Float_t>> & nodal_field,
                                   const amrex::Vector<std::string> & nodal_field_name,
                                   const std::string & fmt)
{
    // PROFILE --------------------------------------------------------
    const std::string which_function = "PrintUnstructuredGridData_VTU(const std::string &, ...)";
    BL_PROFILE(which_function);
    const std::string which_file = "PolyExa_IO_VTK.cpp";
    const std::string header = which_file+" - "+which_function;
    // ----------------------------------------------------------------
    
    // PARAMETERS -----------------------------------------------------
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
    std::ofstream fp;
    // ----------------------------------------------------------------

    // CHECK FORMAT ---------------------------------------------------
    if ((fmt.compare("ascii") != 0) && (fmt.compare("binary") != 0))
    {
        Print() << std::endl;
        Print() << "ERROR: AMReX_VTK_utils.cpp - PrintUnstructuredGridData_VTU" << std::endl;
        Print() << "| Unexpected file format: " << fmt << std::endl;
        Print() << std::endl;
        exit(-1);
    }
    // ----------------------------------------------------------------

    // OPEN FILE ------------------------------------------------------
    fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    fp.open(filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!fp.good())
    {
        amrex::FileOpenFailed(filepath);
    }
    // ----------------------------------------------------------------

    // ASCII FORMAT ---------------------------------------------------
    if (fmt.compare("ascii") == 0)
    {
        PrintUnstructuredGridData_VTU_ascii(fp,
                                            n_nodes,
                                            n_cells,
                                            nodes,
                                            cell_conn,
                                            cell_offset,
                                            cell_type,
                                            cell_field,
                                            cell_field_name,
                                            nodal_field,
                                            nodal_field_name);
    }
    // ----------------------------------------------------------------
    // BINARY FORMAT --------------------------------------------------
    else if (fmt.compare("binary") == 0)
    {
        PrintUnstructuredGridData_VTU_binary(fp,
                                             n_nodes,
                                             n_cells,
                                             nodes,
                                             cell_conn,
                                             cell_offset,
                                             cell_type,
                                             cell_field,
                                             cell_field_name,
                                             nodal_field,
                                             nodal_field_name);
    }
    // ----------------------------------------------------------------

    // CLOSE FILE
    fp.close();
    // ----------
}

void PrintUnstructuredGridData_VTU_ascii(std::ofstream & fp,
                                         const Cell_conn_t & n_nodes,
                                         const Cell_offs_t & n_cells,
                                         const amrex::Vector<Float_t> & nodes,
                                         const amrex::Vector<Cell_conn_t> & cell_conn,
                                         const amrex::Vector<Cell_offs_t> & cell_offset,
                                         const amrex::Vector<Cell_type_t> & cell_type,
                                         const amrex::Vector<amrex::Vector<Int_t>> & cell_field,
                                         const amrex::Vector<std::string> & cell_field_name,
                                         const amrex::Vector<amrex::Vector<Float_t>> & nodal_field,
                                         const amrex::Vector<std::string> & nodal_field_name)
{
    // PROFILE --------------------------------------------------------
    const std::string which_function = "PrintUnstructuredGridData_VTU_ascii(std::ofstream &, ...)";
    BL_PROFILE(which_function);
    const std::string which_file = "PolyExa_IO_VTK.cpp";
    const std::string header = which_file+" - "+which_function;
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    const int cell_offset_stride = 20;
    const int cell_type_stride = 20;
    const int cell_field_stride = 20;
    const int nodal_field_stride = 10;

    const int n_cell_fields = cell_field.size();
    const int n_nodal_fields = nodal_field.size();
    // ----------------------------------------------------------------

    // SETTINGS -------------------------------------------------------
    fp.precision(17);
    // ----------------------------------------------------------------

    // HEADER ---------------------------------------------------------
    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    fp << "<UnstructuredGrid>\n";

    fp << "<Piece NumberOfPoints=\"" << n_nodes << "\" NumberOfCells=\"" << n_cells << "\">\n";
    // ----------------------------------------------------------------

    // NODES ----------------------------------------------------------
    fp << "<Points>\n";
    fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\"Position\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (Cell_conn_t n = 0; n < n_nodes; ++n)
    {
        fp << "  " << nodes[3*n] << " " << nodes[3*n+1] << " " << nodes[3*n+2] << "\n";
    }
    fp << "  </DataArray>\n";
    fp << "</Points>\n";
    // ----------------------------------------------------------------

    // CELLS ----------------------------------------------------------
    fp << "<Cells>\n";
    // CONNECTIVITY
    fp << "  <DataArray type=\"" << Cell_conn_t_description << "\" Name=\"connectivity\" format=\"ascii\">\n";
    
    for (Cell_offs_t c = 0; c < n_cells; ++c)
    {
        const Cell_offs_t n_comp = cell_offset[c+1]-cell_offset[c];
        fp << "  ";
        for (Cell_offs_t k = 0; k < (n_comp-1); ++k)
        {
            fp << cell_conn[cell_offset[c]+k] << " ";
        }
        fp << cell_conn[cell_offset[c]+n_comp-1] << "\n";
    }
    fp << "  </DataArray>\n";

    // OFFSET
    fp << "  <DataArray type=\"" << Cell_offs_t_description << "\" Name=\"offsets\" format=\"ascii\">\n";
    fp << "  ";
    {
        Cell_offs_t c = 1;
        while (c < (n_cells+1))
        {
            fp << cell_offset[c];
            c += 1;
            fp << ((c%cell_offset_stride == 0)? "\n  " : " ");
        }
    }
    fp << "\n  </DataArray>\n";

    // TYPE
    fp << "  <DataArray type=\"" << Cell_type_t_description << "\" Name=\"types\" format=\"ascii\">\n";
    fp << "  ";
    {
        Cell_offs_t c = 0;
        while (c < n_cells)
        {
            fp << (int) cell_type[c];
            c += 1;
            fp << ((c%cell_type_stride == 0)? "\n  " : " ");
        }
    }
    fp << "\n  </DataArray>\n";

    fp << "</Cells>\n";
    // ----------------------------------------------------------------

    // CELLS DATA -----------------------------------------------------
    fp << "<CellData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_cell_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Int_t_description << "\" Name=\""+cell_field_name[f]+"\" format=\"ascii\">\n";
        fp << "  ";
        {
            Cell_offs_t c = 0;
            while (c < n_cells)
            {
                fp << cell_field[f][c];
                c += 1;
                fp << ((c%cell_field_stride == 0)? "\n  " : " ");
            }
        }
        fp << "\n  </DataArray>\n";
    }
    fp << "</CellData>" <<"\n";
    // ----------------------------------------------------------------

    // POINTS DATA ----------------------------------------------------
    fp << "<PointData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_nodal_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\""+nodal_field_name[f]+"\" format=\"ascii\">\n";
        fp << "  ";
        {
            Cell_conn_t n = 0;
            while (n < n_nodes)
            {
                fp << nodal_field[f][n];
                n += 1;
                fp << ((n%nodal_field_stride == 0)? "\n  " : " ");
            }
        }
        fp << "\n  </DataArray>\n";
    }
    fp << "</PointData>" <<"\n";
    // ----------------------------------------------------------------

    // CLOSING -------------------
    fp << "</Piece>\n";
    fp << "</UnstructuredGrid>\n";
    fp << "</VTKFile>\n";
    // ---------------------------   
}

void PrintUnstructuredGridData_VTU_binary(std::ofstream & fp,
                                          const Cell_conn_t & n_nodes,
                                          const Cell_offs_t & n_cells,
                                          const amrex::Vector<Float_t> & nodes,
                                          const amrex::Vector<Cell_conn_t> & cell_conn,
                                          const amrex::Vector<Cell_offs_t> & cell_offset,
                                          const amrex::Vector<Cell_type_t> & cell_type,
                                          const amrex::Vector<amrex::Vector<Int_t>> & cell_field,
                                          const amrex::Vector<std::string> & cell_field_name,
                                          const amrex::Vector<amrex::Vector<Float_t>> & nodal_field,
                                          const amrex::Vector<std::string> & nodal_field_name)
{
    // PROFILE --------------------------------------------------------
    const std::string which_function = "PrintUnstructuredGridData_VTU_binary(std::ofstream &, ...)";
    BL_PROFILE(which_function);
    const std::string which_file = "PolyExa_IO_VTK.cpp";
    const std::string header = which_file+" - "+which_function;
    // ----------------------------------------------------------------

    // PARAMETERS -----------------------------------------------------
    const int n_cell_fields = cell_field.size();
    const int n_nodal_fields = nodal_field.size();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    long binary_offset = 0L;
    Header_t n_bytes;
    // ----------------------------------------------------------------

    // HEADER ---------------------------------------------------------
    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"" << Header_t_description << "\">\n";
    fp << "<UnstructuredGrid>\n";

    fp << "<Piece NumberOfPoints=\"" << n_nodes << "\" NumberOfCells=\"" << n_cells << "\">\n";
    // ----------------------------------------------------------------

    // NODES ----------------------------------------------------------
    fp << "<Points>\n";
    fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";
    fp << "</Points>\n";

    binary_offset += nodes.size()*sizeof(Float_t)+sizeof(n_bytes);
    // ----------------------------------------------------------------

    // CELLS ----------------------------------------------------------
    fp << "<Cells>\n";

    // CONNECTIVITY
    fp << "  <DataArray type=\"" << Cell_conn_t_description << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";

    binary_offset += cell_conn.size()*sizeof(Cell_conn_t)+sizeof(n_bytes);

    // OFFSET
    fp << "  <DataArray type=\"" << Cell_offs_t_description << "\" Name=\"offsets\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";

    binary_offset += (cell_offset.size()-1)*sizeof(Cell_offs_t)+sizeof(n_bytes);

    // TYPE
    fp << "  <DataArray type=\"" << Cell_type_t_description << "\" Name=\"types\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << " </DataArray>\n";

    binary_offset += cell_type.size()*sizeof(Cell_type_t)+sizeof(n_bytes);

    fp << "</Cells>\n";
    // ----------------------------------------------------------------

    // CELLS DATA -----------------------------------------------------
    fp << "<CellData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_cell_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Int_t_description << "\" Name=\""+cell_field_name[f]+"\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
        fp << "  </DataArray>\n";
        binary_offset += cell_field[f].size()*sizeof(Int_t)+sizeof(n_bytes);
    }
    fp << "</CellData>" <<"\n";
    // ----------------------------------------------------------------

    // POINTS DATA ----------------------------------------------------
    fp << "<PointData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_nodal_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\""+nodal_field_name[f]+"\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
        fp << "  </DataArray>\n";
        binary_offset += nodal_field[f].size()*sizeof(Float_t)+sizeof(n_bytes);
    }
    fp << "</PointData>" <<"\n";
    // ----------------------------------------------------------------

    // CLOSING --------------------------------------------------------
    fp << "</Piece>\n";
    fp << "</UnstructuredGrid>\n";

    // APPENDED BINARY DATA
    fp << "<AppendedData encoding=\"raw\">\n";
    fp << "_";

    n_bytes = nodes.size()*sizeof(Float_t);
    if (n_bytes > 0)
    {
        fp.write((char *)&n_bytes, sizeof(n_bytes));
        fp.write((char *)&nodes[0], n_bytes);
    }
    
    n_bytes = cell_conn.size()*sizeof(Cell_conn_t);
    if (n_bytes > 0)
    {
        fp.write((char *)&n_bytes, sizeof(n_bytes));
        fp.write((char *)&cell_conn[0], n_bytes);
    }
    
    n_bytes = (cell_offset.size()-1)*sizeof(Cell_offs_t);
    if (n_bytes > 0)
    {
        fp.write((char *)&n_bytes, sizeof(n_bytes));
        fp.write((char *)&cell_offset[1], n_bytes);
    }
    
    n_bytes = cell_type.size()*sizeof(Cell_type_t);
    if (n_bytes > 0)
    {
        fp.write((char *)&n_bytes, sizeof(n_bytes));
        fp.write((char *)&cell_type[0], n_bytes);
    }
    
    for (int f = 0; f < n_cell_fields; ++f)
    {
        n_bytes = cell_field[f].size()*sizeof(Int_t);
        if (n_bytes > 0)
        {
            fp.write((char *)&n_bytes, sizeof(n_bytes));
            fp.write((char *)&cell_field[f][0], n_bytes);
        }
    }

    for (int f = 0; f < n_nodal_fields; ++f)
    {
        n_bytes = nodal_field[f].size()*sizeof(Float_t);
        if (n_bytes > 0)
        {
            fp.write((char *)&n_bytes, sizeof(n_bytes));
            fp.write((char *)&nodal_field[f][0], n_bytes);
        }
    }
    fp << "\n";
    fp << "</AppendedData>\n";

    fp << "</VTKFile>\n";
    // ----------------------------------------------------------------
}
// ####################################################################
// ####################################################################

}
}