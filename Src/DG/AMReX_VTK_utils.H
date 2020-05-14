// AMReX_VTK_utils.H

#include <AMReX_PlotFileUtil.H>

#ifndef BL_VTK_UTILS_H_
#define BL_VTK_UTILS_H_

namespace amrex
{
namespace VTK
{
// ####################################################################
// CONSTANTS ##########################################################
// ####################################################################
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
// ####################################################################


// ####################################################################
// AUXILIARY ROUTINES FOR OUTPUT FILE/FOLDER CREATIONS ################
// ####################################################################
std::string MakeLocalOutputFilename(const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int & time_id,
                                    const int & time_id_max);

std::string MakeGlobalOutputFilename(const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int & time_id,
                                     const int & time_id_max);

std::string MakeLocalOutputFilepath(const std::string & dst_folder,
                                    const std::string & filename_root,
                                    const std::string & file_fmt,
                                    const int & time_id,
                                    const int & time_id_max);

std::string MakeGlobalOutputFilepath(const std::string & dst_folder,
                                     const std::string & filename_root,
                                     const std::string & file_fmt,
                                     const int & time_id, 
                                     const int & time_id_max);
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
                         const amrex::Vector<std::string> & nodal_field_name);
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
                                   const std::string & fmt = "ascii");
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
                                         const amrex::Vector<std::string> & nodal_field_name);
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
                                          const amrex::Vector<std::string> & nodal_field_name);
// ####################################################################
// ####################################################################
}
}

#endif