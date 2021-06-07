//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_VTK_utils.H
 * \brief Contains utilities to export in VTK format.
*/

#ifndef AMREX_VTK_UTILS_H_
#define AMREX_VTK_UTILS_H_

#include <AMReX_Vector.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_dG_Utils.H>

namespace amrex
{
namespace vtk
{

// TYPEDEFS AND CONSTANTS #############################################
#define VTK_VERTEX_ID 1
#define VTK_VERTEX_CONN_LEN 1

#define VTK_LINE_ID 3
#define VTK_LINE_CONN_LEN 2

#define VTK_QUAD_ID 9
#define VTK_QUAD_CONN_LEN 4

#define VTK_HEXAHEDRON_ID 12
#define VTK_HEXAHEDRON_CONN_LEN 8

typedef uint32_t Header_t;
typedef Real Float_t;
typedef int Int_t;
typedef uint32_t Cell_conn_t;
typedef uint32_t Cell_offs_t;
typedef uint8_t Cell_type_t;

const std::string Header_t_description = "UInt32";
const std::string Float_t_description = "Float"+std::to_string(sizeof(Float_t)*8);
const std::string Int_t_description = "Int"+std::to_string(sizeof(Int_t)*8);
const std::string Cell_conn_t_description = "Int"+std::to_string(sizeof(Cell_conn_t)*8);
const std::string Cell_offs_t_description = "Int"+std::to_string(sizeof(Cell_offs_t)*8);
const std::string Cell_type_t_description = "UInt"+std::to_string(sizeof(Cell_type_t)*8);
// ####################################################################



// VTU CELLS ##########################################################
Gpu::ManagedVector<int> GriddedLine_Connectivity(const int ne);
Gpu::ManagedVector<int> GriddedQuad_Connectivity(const int ne);
Gpu::ManagedVector<int> GriddedHexahedron_Connectivity(const int ne);
Gpu::ManagedVector<int> GriddedHexahedron_Connectivity(const int ne, const int ne3);
// ####################################################################



// VTU OUTPUT #########################################################
void print_header_vtu(const std::string & folderpath,
                      const std::string & step_folderpath,
                      const std::string & step_string,
                      const std::string & filename,
                      const Vector<std::string> & nodal_fields_names,
                      const Vector<std::string> & cell_fields_names);

void print_unstructured_grid_data_vtu(const std::string & filepath,
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
                                      const std::string & fmt = "binary");
void print_unstructured_grid_data_vtu_ascii(std::ofstream & fp,
                                            const Cell_conn_t & n_nodes,
                                            const Cell_offs_t & n_cells,
                                            const Vector<Float_t> & nodes,
                                            const Vector<Cell_conn_t> & cell_conn,
                                            const Vector<Cell_offs_t> & cell_offset,
                                            const Vector<Cell_type_t> & cell_type,
                                            const Vector<Vector<Float_t>> & nodal_fields,
                                            const Vector<std::string> & nodal_fields_names,
                                            const Vector<Vector<Int_t>> & cell_fields,
                                            const Vector<std::string> & cell_fields_names);
void print_unstructured_grid_data_vtu_binary(std::ofstream & fp,
                                             const Cell_conn_t & n_nodes,
                                             const Cell_offs_t & n_cells,
                                             const Vector<Float_t> & nodes,
                                             const Vector<Cell_conn_t> & cell_conn,
                                             const Vector<Cell_offs_t> & cell_offset,
                                             const Vector<Cell_type_t> & cell_type,
                                             const Vector<Vector<Float_t>> & nodal_fields,
                                             const Vector<std::string> & nodal_fields_names,
                                             const Vector<Vector<Int_t>> & cell_fields,
                                             const Vector<std::string> & cell_fields_names);
// ####################################################################

} // namespace vtk
} // namespace amrex

#endif