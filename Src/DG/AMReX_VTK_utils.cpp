//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_VTK_utils.cpp
 * \brief Implementations of the functions contained in AMReX_VTK_utils.H.
*/

#include <AMReX_VTK_utils.H>

namespace amrex
{
namespace VTK
{

// AUXILIARY FUNCTIONS ################################################
/**
 * \brief Make a string containing the filename_root suffixed with rank, time step index and format.
 *
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the local rank and the time step.
 * \param[in] file_fmt: filename format appended to the filename.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 *
 * \return a string containing the output filename.
*/
std::string MakeLocalOutputFilename(const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int n, 
                                    const int n_steps)
{
    const int proc_id = ParallelDescriptor::MyProc();

    int ndigits;
    std::string filename;

    ndigits = 1;
    while (std::pow(10, ndigits) < n_steps) ndigits += 1;

    filename = Concatenate(filename_root+"_proc_"+std::to_string(proc_id)+"_", n, ndigits);
    filename += "."+file_fmt;

    return filename;
}

/**
 * \brief Make a string containing the filename_root suffixed with time step index and format.
 *
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the time step.
 * \param[in] file_fmt: filename format appended to the filename.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 *
 * \return a string containing the output filename.
*/
std::string MakeGlobalOutputFilename(const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int n,
                                     const int n_steps)
{
    int ndigits;
    std::string filename;

    ndigits = 1;
    while (std::pow(10, ndigits) < n_steps) ndigits += 1;

    filename = Concatenate(filename_root+"_", n, ndigits);
    filename += "."+file_fmt;

    return filename;
}

/**
 * \brief Same as MakeLocalOutputFilename but the folderpath is prefixed.
 *
 * \param[in] folderpath: path to the output folder.
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the local rank and the time step.
 * \param[in] file_fmt: filename format appended to the filename.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 *
 * \return a string containing the output filepath.
*/
std::string MakeLocalOutputFilepath(const std::string & folderpath,
                                    const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int n,
                                    const int n_steps)
{
    std::string filepath = folderpath+"/"+MakeLocalOutputFilename(filename_root, file_fmt, n, n_steps);

    return filepath;
}

/**
 * \brief Same as MakeGlobalOutputFilename but the folderpath is prefixed.
 *
 * \param[in] folderpath: path to the output folder.
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the time step.
 * \param[in] file_fmt: filename format appended to the filename.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 *
 * \return a string containing the output filepath.
*/
std::string MakeGlobalOutputFilepath(const std::string & folderpath,
                                     const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int n, 
                                     const int n_steps)
{
    std::string filepath = folderpath+"/"+MakeGlobalOutputFilename(filename_root, file_fmt, n, n_steps);

    return filepath;
}
// ####################################################################

// VTU CELLS ##########################################################
/**
 * \brief Return the connectivity of a VTK line divided using ne lines.
 *
 * \param[in] ne: grid size.
 * 
 * \return the connectivity of the ne lines.
*/
Gpu::ManagedVector<int> GriddedLine_Connectivity(const int ne)
{
    // PARAMETERS =====================================================
    const int n_sublines = ne;
    const int conn_len = 2*n_sublines;
    // ================================================================

    // VARIABLES ======================================================
    Gpu::ManagedVector<int> conn(conn_len);
    // ================================================================

    // EVAL CONNECTIVITY ==============================================
    for (int i = 0; i < ne; ++i)
    {
        conn[2*i] = i;
        conn[2*i+1] = i+1;
    }
    // ================================================================

    return conn;
}

/**
 * \brief Return the connectivity of a VTK quad divided using a ne x ne grid.
 *
 * \param[in] ne: grid size.
 * 
 * \return the connectivity of the ne x ne grid.
*/
Gpu::ManagedVector<int> GriddedQuad_Connectivity(const int ne)
{
    // PARAMETERS =====================================================
    const int nn = ne+1;
    const int n_subquads = ne*ne;
    const int conn_len = 4*n_subquads;
    // ================================================================

    // VARIABLES ======================================================
    int c;
    Gpu::ManagedVector<int> conn(conn_len);
    // ================================================================

    // EVAL CONNECTIVITY ==============================================
    for (int j = 0; j < ne; ++j)
    for (int i = 0; i < ne; ++i)
    {
        c = i+j*ne;
        conn[4*c+0] = i+j*nn;
        conn[4*c+1] = (i+1)+j*nn;
        conn[4*c+2] = (i+1)+(j+1)*nn;
        conn[4*c+3] = i+(j+1)*nn;
    }
    // ================================================================

    return conn;
}

/**
 * \brief Return the connectivity of a VTK hexahedron divided using a ne x ne x ne grid.
 *
 * \param[in] ne: grid size.
 * 
 * \return the connectivity of the ne x ne x ne grid.
*/
Gpu::ManagedVector<int> GriddedHexahedron_Connectivity(const int ne)
{
    // PARAMETERS =====================================================
    const int nn = ne+1;
    const int n_subquads = ne*ne*ne;
    const int conn_len = 8*n_subquads;
    // ================================================================
    
    // VARIABLES ======================================================
    int c;
    Gpu::ManagedVector<int> conn(conn_len);
    // ================================================================

    // EVAL CONNECTIVITY ==============================================
    for (int k = 0; k < ne; ++k)
    for (int j = 0; j < ne; ++j)
    for (int i = 0; i < ne; ++i)
    {
        c = i+j*ne+k*ne*ne;
        conn[8*c+0] = i+j*nn+k*nn*nn;
        conn[8*c+1] = (i+1)+j*nn+k*nn*nn;
        conn[8*c+2] = (i+1)+(j+1)*nn+k*nn*nn;
        conn[8*c+3] = i+(j+1)*nn+k*nn*nn;
        conn[8*c+4] = i+j*nn+(k+1)*nn*nn;
        conn[8*c+5] = (i+1)+j*nn+(k+1)*nn*nn;
        conn[8*c+6] = (i+1)+(j+1)*nn+(k+1)*nn*nn;
        conn[8*c+7] = i+(j+1)*nn+(k+1)*nn*nn;
    }
    // ================================================================

    return conn;
}
// ####################################################################

// VTU OUTPUT #########################################################
/**
 * \brief Print header file for VTU output.
 *
 * \param[in] folderpath: output folder path where the file will be dumped.
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] cell_fields_names: vector of strings containing the description of the cell fields.
 * \param[in] nodal_fields_names: vector of strings containing the description of the nodal fields.
*/
void PrintHeaderFile_VTU(const std::string & folderpath,
                         const std::string & filename_root,
                         const int n,
                         const int n_steps,
                         const Vector<std::string> & nodal_fields_names,
                         const Vector<std::string> & cell_fields_names)
{
    // PARAMETERS =====================================================
    const int n_procs = ParallelDescriptor::NProcs();

    // FILEPATH
    const std::string filepath = MakeGlobalOutputFilepath(folderpath, filename_root, "pvtu", n, n_steps);

    // NODAL DATA
    const int n_nodal_fields = nodal_fields_names.size();
    
    // CELL DATA
    const int n_cell_fields = cell_fields_names.size();
    // ================================================================

    // VARIABLES ======================================================
    std::string tmp_filename;

    // WRITING TOOLS
    std::ofstream fp;
    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
    // ================================================================

    // OPEN FILE ======================================================
    fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    fp.open(filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!fp.good())
    {
        FileOpenFailed(filepath);
    }
    // ================================================================

    // WRITE FILE =====================================================
    fp.precision(17);

    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"" << Header_t_description << "\">\n";
    fp << "<PUnstructuredGrid GhostLevel=\"0\">\n";
    
    {
        fp << "  <PPoints>\n";
        fp << "    <PDataArray type=\"" << Float_t_description << "\" Name=\"Position\" NumberOfComponents=\"3\"/>\n"; 
        fp << "  </PPoints>\n";
    }

    if (n_nodal_fields > 0)
    {
        fp << "  <PPointData Scalars=\"Scalars\">\n";
        for (int i = 0; i < n_nodal_fields; i++)
        {
            fp << "    <PDataArray type=\"" << Float_t_description << "\" Name=\""+nodal_fields_names[i]+"\" NumberOfComponents=\"1\"/>\n";
        }
        fp << "  </PPointData>\n";
    }
    
    {
        fp << "  <PCells>\n"; 
        fp << "    <PDataArray type=\"" << Cell_conn_t_description << "\" Name=\"connectivity\" NumberOfComponents=\"1\"/>\n";
        fp << "    <PDataArray type=\"" << Cell_offs_t_description << "\" Name=\"offsets\" NumberOfComponents=\"1\"/>\n";
        fp << "    <PDataArray type=\"" << Cell_type_t_description << "\" Name=\"types\" NumberOfComponents=\"1\"/>\n";
        fp << "  </PCells>\n";
    }

    if (n_cell_fields > 0)
    {
        fp << "  <PCellData Scalars=\"Scalars\">\n";
        for (int i = 0; i < n_cell_fields; i++)
        {
            fp << "    <PDataArray type=\"" << Int_t_description << "\" Name=\""+cell_fields_names[i]+"\" NumberOfComponents=\"1\"/>\n";
        }
        fp << "  </PCellData>\n";
    }

    for (int i = 0; i < n_procs; i++)
    {
        tmp_filename = MakeGlobalOutputFilename(filename_root+"_proc_"+std::to_string(i), "vtu", n, n_steps);

        fp << "  <Piece Source=\"" << tmp_filename << "\"/>\n";
    }
    
    fp << "</PUnstructuredGrid>\n";
    
    fp << "</VTKFile>\n";
    
    fp << '\n';
    // ================================================================

    // CLOSE FILE
    fp.close();
    // ==========
}

/**
 * \brief Print unstructured grid data VTU output.
 *
 * \param[in] filepath: output file path.
 * \param[in] n_nodes: number of VTU nodes.
 * \param[in] n_cells: number of VTU cells.
 * \param[in] nodes: vector containing nodes coordinates.
 * \param[in] cell_conn: vector containing cell connectivity.
 * \param[in] cell_offset: vector containing cell offset.
 * \param[in] cell_type: vector containing cell type.
 * \param[in] nodal_fields: vector containing nodal fields.
 * \param[in] nodal_fields_names: vector of strings containing the description of the nodal fields.
 * \param[in] cell_fields: vector containing cell fields.
 * \param[in] cell_fields_names: vector of strings containing the description of the cell fields.
 * \param[in] fmt: either "ascii" or "binary".
*/
void PrintUnstructuredGridData_VTU(const std::string & filepath,
                                   const Cell_conn_t & n_nodes,
                                   const Cell_offs_t & n_cells,
                                   const Vector<Float_t> & nodes,
                                   const Vector<Cell_conn_t> & cell_conn,
                                   const Vector<Cell_offs_t> & cell_offset,
                                   const Vector<Cell_type_t> & cell_type,
                                   const Vector<Vector<Float_t>> & nodal_fields,
                                   const Vector<std::string> & nodal_fields_names,
                                   const Vector<Vector<Int_t>> & cell_fields,
                                   const Vector<std::string> & cell_fields_names,
                                   const std::string & fmt)
{
    // VARIABLES ======================================================
    // WRITING TOOLS
    std::ofstream fp;
    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
    // ================================================================

    // CHECK FORMAT ===================================================
    if (fmt.compare("binary") != 0)
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_VTK_utils.cpp - PrintUnstructuredGridData_VTU\n";
        msg += "| Unexpected file format: "+fmt+"\n";
        Abort(msg);
    }
    // ================================================================

    // OPEN FILE ======================================================
    fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    fp.open(filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!fp.good())
    {
        FileOpenFailed(filepath);
    }
    // ================================================================

    // BINARY FORMAT
    if (fmt.compare("binary") == 0)
    {
        PrintUnstructuredGridData_VTU_binary(fp,
                                             n_nodes,
                                             n_cells,
                                             nodes,
                                             cell_conn,
                                             cell_offset,
                                             cell_type,
                                             nodal_fields,
                                             nodal_fields_names,
                                             cell_fields,
                                             cell_fields_names);
    }

    // CLOSE FILE
    fp.close();
    // ==========
}

/**
 * \brief Print unstructured grid data VTU output in binary format.
 *
 * \param[in] fp: output stream where the data will be written.
 * \param[in] n_nodes: number of VTU nodes.
 * \param[in] n_cells: number of VTU cells.
 * \param[in] nodes: vector containing nodes coordinates.
 * \param[in] cell_conn: vector containing cell connectivity.
 * \param[in] cell_offset: vector containing cell offset.
 * \param[in] cell_type: vector containing cell type.
 * \param[in] nodal_fields: vector containing nodal fields.
 * \param[in] nodal_fields_names: vector of strings containing the description of the nodal fields.
 * \param[in] cell_fields: vector containing cell fields.
 * \param[in] cell_fields_names: vector of strings containing the description of the cell fields.
*/
void PrintUnstructuredGridData_VTU_binary(std::ofstream & fp,
                                          const Cell_conn_t & n_nodes,
                                          const Cell_offs_t & n_cells,
                                          const Vector<Float_t> & nodes,
                                          const Vector<Cell_conn_t> & cell_conn,
                                          const Vector<Cell_offs_t> & cell_offset,
                                          const Vector<Cell_type_t> & cell_type,
                                          const Vector<Vector<Float_t>> & nodal_fields,
                                          const Vector<std::string> & nodal_fields_names,
                                          const Vector<Vector<Int_t>> & cell_fields,
                                          const Vector<std::string> & cell_fields_names)
{
    // PARAMETERS =====================================================
    const int n_nodal_fields = nodal_fields.size();
    const int n_cell_fields = cell_fields.size();
    // ================================================================

    // VARIABLES ======================================================
    long binary_offset = 0L;
    Header_t n_bytes;
    // ================================================================

    // HEADER =========================================================
    fp << "<?xml version=\"1.0\"?>\n";
    fp << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"" << Header_t_description << "\">\n";
    fp << "<UnstructuredGrid>\n";

    fp << "<Piece NumberOfPoints=\"" << n_nodes << "\" NumberOfCells=\"" << n_cells << "\">\n";
    // ================================================================

    // NODES ==========================================================
    fp << "<Points>\n";

    fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";

    if (n_nodes > 0)
    {
        binary_offset += nodes.size()*sizeof(Float_t)+sizeof(n_bytes);
    }

    fp << "</Points>\n";
    // ================================================================

    // CELLS ==========================================================
    fp << "<Cells>\n";

    // CONNECTIVITY
    fp << "  <DataArray type=\"" << Cell_conn_t_description << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";

    if (n_cells > 0)
    {
        binary_offset += cell_conn.size()*sizeof(Cell_conn_t)+sizeof(n_bytes);
    }

    // OFFSET
    fp << "  <DataArray type=\"" << Cell_offs_t_description << "\" Name=\"offsets\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << "  </DataArray>\n";

    if (n_cells > 0)
    {
        binary_offset += (cell_offset.size()-1)*sizeof(Cell_offs_t)+sizeof(n_bytes);
    }

    // TYPE
    fp << "  <DataArray type=\"" << Cell_type_t_description << "\" Name=\"types\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
    fp << " </DataArray>\n";

    if (n_cells > 0)
    {
        binary_offset += cell_type.size()*sizeof(Cell_type_t)+sizeof(n_bytes);
    }

    fp << "</Cells>\n";
    // ================================================================

    // POINTS DATA ====================================================
    fp << "<PointData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_nodal_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Float_t_description << "\" Name=\""+nodal_fields_names[f]+"\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
        fp << "  </DataArray>\n";
        binary_offset += nodal_fields[f].size()*sizeof(Float_t)+sizeof(n_bytes);
    }
    fp << "</PointData>" <<"\n";
    // ================================================================

    // CELLS DATA =====================================================
    fp << "<CellData Scalars=\"Scalars\">\n";
    for (int f = 0; f < n_cell_fields; ++f)
    {
        fp << "  <DataArray type=\"" << Int_t_description << "\" Name=\""+cell_fields_names[f]+"\" format=\"appended\" offset=\"" << binary_offset << "\">\n";
        fp << "  </DataArray>\n";
        binary_offset += cell_fields[f].size()*sizeof(Int_t)+sizeof(n_bytes);
    }
    fp << "</CellData>" <<"\n";
    // ================================================================

    // CLOSING ========================================================
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

    for (int f = 0; f < n_nodal_fields; ++f)
    {
        n_bytes = nodal_fields[f].size()*sizeof(Float_t);
        if (n_bytes > 0)
        {
            fp.write((char *)&n_bytes, sizeof(n_bytes));
            fp.write((char *)&nodal_fields[f][0], n_bytes);
        }
    }
    
    for (int f = 0; f < n_cell_fields; ++f)
    {
        n_bytes = cell_fields[f].size()*sizeof(Int_t);
        if (n_bytes > 0)
        {
            fp.write((char *)&n_bytes, sizeof(n_bytes));
            fp.write((char *)&cell_fields[f][0], n_bytes);
        }
    }

    fp << "\n";
    fp << "</AppendedData>\n";

    fp << "</VTKFile>\n";
    // ================================================================
}
// ####################################################################

} // namespace VTK
} // namespace amrex

#include <AMReX_VTK_utils.H>