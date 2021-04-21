//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Mesh_Output.H
 * \brief Functions to output meshes with embedded geometries.
*/

#ifndef AMREX_DG_MESH_OUTPUT_H_
#define AMREX_DG_MESH_OUTPUT_H_

#include <AMReX_VTK_Utils.H>

#include <AMReX_dG_Mesh.H>

namespace amrex
{
namespace dG
{
namespace mesh_io
{

// MESH INPUT/OUTPUT TO VTK ###########################################
// DOMAIN / BOUNDARY DATA =============================================
/**
 * \brief Export the mes domain data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_dom_data_to_VTK(const std::string & level_folderpath,
                            const std::string & level_step_folderpath,
                            const std::string & step_string,
                            const std::string & filename,
                            const Geometry & geom,
                            const Mesh & mesh,
                            const iMultiFab & mask,
                            IBVP & ibvp)
{
    // PARAMETERS -------------------------------------------------
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(VTK_LINE_ID, VTK_QUAD_ID, VTK_HEXAHEDRON_ID);
    const int VTK_cell_conn_len = AMREX_D_PICK(VTK_LINE_CONN_LEN, VTK_QUAD_CONN_LEN, VTK_HEXAHEDRON_CONN_LEN);

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ------------------------------------------------------------

    // VARIABLES --------------------------------------------------
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    vtk::Cell_conn_t n_VTK_nodes;
    vtk::Cell_offs_t n_VTK_cells;

    Vector<vtk::Float_t> VTK_nodes;
    Vector<vtk::Cell_conn_t> VTK_cell_conn;
    Vector<vtk::Cell_offs_t> VTK_cell_offset;
    Vector<vtk::Cell_type_t> VTK_cell_type;
    Vector<Vector<vtk::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<vtk::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_cell_conn_pos, VTK_cell_offset_pos, VTK_cell_type_pos;
    long VTK_nodal_field_pos;
    long VTK_cell_field_pos;

    long VTK_cell_conn_offset;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_dom_Ng, cell_e, cell_dom_Ne;
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], 1);

    cell_dom_Ng = std_elm.dom_Ng;
    cell_e = std_elm.g;
    cell_dom_Ne = AMREX_D_TERM(cell_e,*cell_e,*cell_e);

#if (AMREX_SPACEDIM == 1)
    subgrid_conn = vtk::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 2)
    subgrid_conn = vtk::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    subgrid_conn = vtk::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    VTK_cell_fields_names.push_back("rank");
    VTK_cell_fields_names.push_back("box");
    VTK_cell_fields_names.push_back("index_i");
    VTK_cell_fields_names.push_back("index_j");
    VTK_cell_fields_names.push_back("index_k");
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("cell_type_dom_"+std::to_string(dom));
    }
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("Nq_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            VTK_cell_conn_size += cell_dom_Ne*VTK_cell_conn_len;

            n_VTK_nodes += cell_dom_Ng;
            n_VTK_cells += cell_dom_Ne;
        }
    }
    // ------------------------------------------------------------

    // ALLOCATE MEMORY --------------------------------------------
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ------------------------------------------------------------

    // EVAL OUTPUT FIELDS -----------------------------------------
    long bx_id;
    bx_id = 0L;

    vtk::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    vtk::Cell_conn_t * VTK_cell_conn_ptr = VTK_cell_conn.data();
    
    vtk::Cell_offs_t * VTK_cell_offset_ptr = VTK_cell_offset.data();
    VTK_cell_offset_ptr[0] = 0;
    
    vtk::Cell_type_t * VTK_cell_type_ptr = VTK_cell_type.data();

    Vector<vtk::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<vtk::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;
    VTK_cell_conn_pos = 0L;
    VTK_cell_offset_pos = 1L;
    VTK_cell_type_pos = 0L;

    VTK_nodal_field_pos = 0L;
    VTK_cell_field_pos = 0L;

    VTK_cell_conn_offset = 0L;

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = std_elm.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = std_elm.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = std_elm.dom_XI3g.data();
    )

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & cell_type_fab = mesh.cell_type.array(mfi);
        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                prob_lo[1]+j*dx[1],
                                                                prob_lo[2]+k*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM];

            // LOOP OVER THE GRID POINTS
            for (int g = 0; g < cell_dom_Ng; ++g)
            {
                // GRID POINT
                AMREX_D_TERM
                (
                    x[0] = dom_XI1g_ptr[g]+cell_lo[0];,
                    x[1] = dom_XI2g_ptr[g]+cell_lo[1];,
                    x[2] = dom_XI3g_ptr[g]+cell_lo[2];
                )

                // VTK NODE
#if (AMREX_SPACEDIM == 1)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif

                // VTK NODAL FIELDS
                // ...

                // MOVE OFFSET
                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;

            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }

                // VTK CELL OFFSET
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;

                // VTK CELL TYPE
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;

                // VTK CELL FIELDS
                // RANK
                VTK_cell_fields_ptr[0][VTK_cell_field_pos] = rank;

                // BOX
                VTK_cell_fields_ptr[1][VTK_cell_field_pos] = bx_id;

                // I
                VTK_cell_fields_ptr[2][VTK_cell_field_pos] = i;
                // J
                VTK_cell_fields_ptr[3][VTK_cell_field_pos] = j;
                // K
                VTK_cell_fields_ptr[4][VTK_cell_field_pos] = k;

                // CELL TYPE
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[5+dom][VTK_cell_field_pos] = cell_type_fab(i,j,k,CELL_TYPE(dom));
                }

                // NUMBER OF QUADRATURE POINTS
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[5+n_domains+dom][VTK_cell_field_pos] = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                }

                // MASK
                VTK_cell_fields_ptr[5+n_domains+n_domains][VTK_cell_field_pos] = mask_fab(i,j,k);

                // MOVE OFFSET
                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            // MOVE OFFSET
            VTK_cell_conn_offset += cell_dom_Ng;
        }

        bx_id += 1;
    }
    // ------------------------------------------------------------

    // HEADER FILE ------------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {   
        vtk::print_header_vtu(level_folderpath, level_step_folderpath, step_string, filename,
                                VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ------------------------------------------------------------

    // DATA FILE --------------------------------------------------
    {
        const std::string filepath = dG::io::make_path({level_step_folderpath, filename+"_"+std::to_string(rank)+".vtu"});
        
        vtk::print_unstructured_grid_data_vtu(filepath,
                                                n_VTK_nodes,
                                                n_VTK_cells,
                                                VTK_nodes,
                                                VTK_cell_conn,
                                                VTK_cell_offset,
                                                VTK_cell_type,
                                                VTK_nodal_fields,
                                                VTK_nodal_fields_names,
                                                VTK_cell_fields,
                                                VTK_cell_fields_names);
    }
    // ------------------------------------------------------------
}

/**
 * \brief Export the mesh boundary data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_bou_data_to_VTK(const std::string & level_folderpath,
                            const std::string & level_step_folderpath,
                            const std::string & step_string,
                            const std::string & filename,
                            const Geometry & geom,
                            const Mesh & mesh,
                            const iMultiFab & mask,
                            IBVP & ibvp)
{
    // PARAMETERS -------------------------------------------------
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(VTK_VERTEX_ID, VTK_LINE_ID, VTK_QUAD_ID);
    const int VTK_cell_conn_len = AMREX_D_PICK(VTK_VERTEX_CONN_LEN, VTK_LINE_CONN_LEN, VTK_QUAD_CONN_LEN);

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ------------------------------------------------------------

    // VARIABLES --------------------------------------------------
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    vtk::Cell_conn_t n_VTK_nodes;
    vtk::Cell_offs_t n_VTK_cells;

    Vector<vtk::Float_t> VTK_nodes;
    Vector<vtk::Cell_conn_t> VTK_cell_conn;
    Vector<vtk::Cell_offs_t> VTK_cell_offset;
    Vector<vtk::Cell_type_t> VTK_cell_type;
    Vector<Vector<vtk::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<vtk::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_cell_conn_pos, VTK_cell_offset_pos, VTK_cell_type_pos;
    long VTK_nodal_field_pos;
    long VTK_cell_field_pos;

    long VTK_cell_conn_offset;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_bou_Ng, cell_bou_Ne;
#if (AMREX_SPACEDIM > 1)
    int cell_e;
#endif
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], 1);

    cell_bou_Ng = std_elm.bou_Ng;
#if (AMREX_SPACEDIM > 1)
    cell_e = std_elm.g;
#endif
    cell_bou_Ne = AMREX_D_TERM(1,*cell_e,*cell_e);
#if (AMREX_SPACEDIM == 1)
    subgrid_conn = {0};
#endif
#if (AMREX_SPACEDIM == 2)
    subgrid_conn = vtk::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    subgrid_conn = vtk::GriddedQuad_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("cell_bou_type_dom_"+std::to_string(dom));
    }
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("Nq_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;

    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                VTK_cell_conn_size += cell_bou_Ne*VTK_cell_conn_len;

                n_VTK_nodes += cell_bou_Ng;
                n_VTK_cells += cell_bou_Ne;
            }
        }
    }
    // ------------------------------------------------------------

    // ALLOCATE MEMORY --------------------------------------------
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ------------------------------------------------------------

    // EVAL OUTPUT FIELDS -----------------------------------------
    vtk::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    vtk::Cell_conn_t * VTK_cell_conn_ptr = VTK_cell_conn.data();
    
    vtk::Cell_offs_t * VTK_cell_offset_ptr = VTK_cell_offset.data();
    VTK_cell_offset_ptr[0] = 0;
    
    vtk::Cell_type_t * VTK_cell_type_ptr = VTK_cell_type.data();

    Vector<vtk::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<vtk::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;
    VTK_cell_conn_pos = 0L;
    VTK_cell_offset_pos = 1L;
    VTK_cell_type_pos = 0L;

    VTK_nodal_field_pos = 0L;
    VTK_cell_field_pos = 0L;

    VTK_cell_conn_offset = 0L;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        AMREX_D_TERM
        (
            Real const * bou_XI1g_ptr = std_elm.bou_XI1g[dir].data();,
            Real const * bou_XI2g_ptr = std_elm.bou_XI2g[dir].data();,
            Real const * bou_XI3g_ptr = std_elm.bou_XI3g[dir].data();
        )

        for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & cell_bou_type_fab = mesh.cell_bou_type[dir].array(mfi);
            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // LOCAL PARAMETERS
                const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                    prob_lo[1]+fj*dx[1],
                                                                    prob_lo[2]+fk*dx[2])};

                // LOCAL VARIABLES
                int mi, mj, mk, pi, pj, pk;
                int m_mask, p_mask;
                Real x[AMREX_SPACEDIM];

                // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                m_mask = mask_fab(mi,mj,mk);
                p_mask = mask_fab(pi,pj,pk);

                // LOOP OVER THE GRID POINTS
                for (int g = 0; g < cell_bou_Ng; ++g)
                {
                    // GRID POINT
                    AMREX_D_TERM
                    (
                        x[0] = bou_XI1g_ptr[g]+face_lo[0];,
                        x[1] = bou_XI2g_ptr[g]+face_lo[1];,
                        x[2] = bou_XI3g_ptr[g]+face_lo[2];
                    )

                    // VTK NODE
#if (AMREX_SPACEDIM == 1)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif

                    // VTK NODAL FIELDS
                    // ...

                    // MOVE OFFSET
                    VTK_nodes_pos += 3;
                    VTK_nodal_field_pos += 1;
                }

                // LOOP OVER THE SUB CELLS
                for (int e = 0; e < cell_bou_Ne; ++e)
                {
                    // VTK CELL CONNECTIVITY
                    const int offset = e*VTK_cell_conn_len;
                    for (int c = 0; c < VTK_cell_conn_len; ++c)
                    {
                        VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                    }

                    // VTK CELL OFFSET
                    VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;

                    // VTK CELL TYPE
                    VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;

                    // VTK CELL FIELDS
                    // CELL BOU TYPE
                    for (int dom = 0; dom < n_domains; ++dom)
                    {
                        VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = cell_bou_type_fab(fi,fj,fk,CELL_BOU_TYPE(dom));
                    }

                    // NUMBER OF QUADRATURE POINTS
                    for (int dom = 0; dom < n_domains; ++dom)
                    {
                        VTK_cell_fields_ptr[n_domains+dom][VTK_cell_field_pos] = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    }

                    // MASK
                    VTK_cell_fields_ptr[n_domains+n_domains][VTK_cell_field_pos] = (m_mask == p_mask) ? m_mask : __DG_LEVEL_BOUNDARY__;

                    // MOVE OFFSET
                    VTK_cell_conn_pos += VTK_cell_conn_len;
                    VTK_cell_offset_pos += 1;
                    VTK_cell_type_pos += 1;
                    VTK_cell_field_pos += 1;
                }

                // MOVE OFFSET
                VTK_cell_conn_offset += cell_bou_Ng;
            }
        }
    }
    // ------------------------------------------------------------

    // HEADER FILE ------------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {   
        vtk::print_header_vtu(level_folderpath, level_step_folderpath, step_string, filename,
                                VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ------------------------------------------------------------

    // DATA FILE --------------------------------------------------
    {
        const std::string filepath = dG::io::make_path({level_step_folderpath, filename+"_"+std::to_string(rank)+".vtu"});
        
        vtk::print_unstructured_grid_data_vtu(filepath,
                                                n_VTK_nodes,
                                                n_VTK_cells,
                                                VTK_nodes,
                                                VTK_cell_conn,
                                                VTK_cell_offset,
                                                VTK_cell_type,
                                                VTK_nodal_fields,
                                                VTK_nodal_fields_names,
                                                VTK_cell_fields,
                                                VTK_cell_fields_names);
    }
    // ------------------------------------------------------------
}

/**
 * \brief Export the mesh to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename_root: root of the name of the output files.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_to_VTK(const std::string & level_folderpath,
                    const std::string & level_step_folderpath,
                    const std::string & step_string,
                    const std::string & filename_root,
                    const Geometry & geom,
                    const Mesh & mesh,
                    const iMultiFab & mask,
                    IBVP & ibvp)
{
    const std::string dom_data_filename = filename_root+"_dom_data";
    const std::string bou_data_filename = filename_root+"_bou_data";

    export_dom_data_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                           geom, mesh, mask, ibvp);
    export_bou_data_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                           geom, mesh, mask, ibvp);
}
// ====================================================================


// QUADRATURE POINTS ==================================================
/**
 * \brief Export the mesh domain quadrature points to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_dom_quad_points_to_VTK(const std::string & level_folderpath,
                                   const std::string & level_step_folderpath,
                                   const std::string & step_string,
                                   const std::string & filename,
                                   const Geometry & geom,
                                   const Mesh & mesh,
                                   const iMultiFab & mask,
                                   IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    vtk::Cell_conn_t n_VTK_nodes;
    vtk::Cell_offs_t n_VTK_cells;

    Vector<vtk::Float_t> VTK_nodes;
    Vector<vtk::Cell_conn_t> VTK_cell_conn;
    Vector<vtk::Cell_offs_t> VTK_cell_offset;
    Vector<vtk::Cell_type_t> VTK_cell_type;
    Vector<Vector<vtk::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<vtk::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_nodal_field_pos;
    // ----------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA --------------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    VTK_nodal_fields_names.push_back("w");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ----------------------------------------------------------------

    // EVAL REQUIRED MEMORY -------------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            if (CELL_IS_NOT_MASKED(mask_fab(i,j,k,dom)))
            {
                n_VTK_nodes += cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
            }
        }
    }
    // ----------------------------------------------------------------

    // ALLOCATE MEMORY ------------------------------------------------
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ----------------------------------------------------------------

    // EVAL OUTPUT FIELDS ---------------------------------------------
    vtk::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    Vector<vtk::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<vtk::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;

    VTK_nodal_field_pos = 0L;

#ifdef AMREX_USE_GPU
    Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
    Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            if (CELL_IS_NOT_MASKED(mask_fab(i,j,k,dom)))
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;

                // LOOP OVER THE QUADRATURE POINTS
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // QUADRATURE POINT
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // VTK NODE
#if (AMREX_SPACEDIM == 1)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif

                    // VTK NODAL FIELDS
                    // QUADRATURE POINT WEIGHT
                    VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = w;

                    // MOVE OFFSET
                    VTK_nodes_pos += 3;
                    VTK_nodal_field_pos += 1;
                }

                // LOOP OVER THE SUB CELLS
                // ...
            }
        }
    }
    // ----------------------------------------------------------------

    // HEADER FILE ----------------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {   
        vtk::print_header_vtu(level_folderpath, level_step_folderpath, step_string, filename,
                                VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------

    // DATA FILE ------------------------------------------------------
    {
        const std::string filepath = dG::io::make_path({level_step_folderpath, filename+"_"+std::to_string(rank)+".vtu"});
        
        vtk::print_unstructured_grid_data_vtu(filepath,
                                              n_VTK_nodes,
                                              n_VTK_cells,
                                              VTK_nodes,
                                              VTK_cell_conn,
                                              VTK_cell_offset,
                                              VTK_cell_type,
                                              VTK_nodal_fields,
                                              VTK_nodal_fields_names,
                                              VTK_cell_fields,
                                              VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Export the mesh boundary quadrature points to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_bou_quad_points_to_VTK(const std::string & level_folderpath,
                                   const std::string & level_step_folderpath,
                                   const std::string & step_string,
                                   const std::string & filename,
                                   const Geometry & geom,
                                   const Mesh & mesh,
                                   const iMultiFab & mask,
                                   IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    vtk::Cell_conn_t n_VTK_nodes;
    vtk::Cell_offs_t n_VTK_cells;

    Vector<vtk::Float_t> VTK_nodes;
    Vector<vtk::Cell_conn_t> VTK_cell_conn;
    Vector<vtk::Cell_offs_t> VTK_cell_offset;
    Vector<vtk::Cell_type_t> VTK_cell_type;
    Vector<Vector<vtk::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<vtk::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_nodal_field_pos;
    // ----------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA --------------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    VTK_nodal_fields_names.push_back("w");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ----------------------------------------------------------------

    // EVAL REQUIRED MEMORY -------------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // LOCAL VARIABLES
                int mi, mj, mk, pi, pj, pk;
                int m_mask, p_mask;

                // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                m_mask = mask_fab(mi,mj,mk);
                p_mask = mask_fab(pi,pj,pk);

                if (CELL_IS_NOT_MASKED(m_mask) && CELL_IS_NOT_MASKED(p_mask))
                {
                    n_VTK_nodes += cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                }
            }
        }
    }
    // ----------------------------------------------------------------

    // ALLOCATE MEMORY ------------------------------------------------
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ----------------------------------------------------------------

    // EVAL OUTPUT FIELDS ---------------------------------------------
    vtk::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    Vector<vtk::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<vtk::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;

    VTK_nodal_field_pos = 0L;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
#endif

        for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // LOCAL PARAMETERS
                const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                   prob_lo[1]+fj*dx[1],
                                                                   prob_lo[2]+fk*dx[2])};
                const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                int mi, mj, mk, pi, pj, pk;
                int m_mask, p_mask;
                Real x[AMREX_SPACEDIM], w;

                // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                m_mask = mask_fab(mi,mj,mk);
                p_mask = mask_fab(pi,pj,pk);

                if (CELL_IS_NOT_MASKED(m_mask) && CELL_IS_NOT_MASKED(p_mask))
                {
                    // LOOP OVER THE QUADRATURE POINTS
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        // QUADRATURE POINT
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0]+face_lo[0];,
                            x[1] = x_ptr[x_pos+1]+face_lo[1];,
                            x[2] = x_ptr[x_pos+2]+face_lo[2];
                        )
                        w = x_ptr[x_pos+AMREX_SPACEDIM];

                        // VTK NODE
#if (AMREX_SPACEDIM == 1)
                        VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                        VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                        VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                        VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                        VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                        VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                        VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                        VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                        VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif

                        // VTK NODAL FIELDS
                        // QUADRATURE POINT WEIGHT
                        VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = w;

                        // MOVE OFFSET
                        VTK_nodes_pos += 3;
                        VTK_nodal_field_pos += 1;
                    }

                    // LOOP OVER THE SUB CELLS
                    // ...
                }
            }
        }
    }
    // ----------------------------------------------------------------

    // HEADER FILE ----------------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {   
        vtk::print_header_vtu(level_folderpath, level_step_folderpath, step_string, filename,
                                VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------

    // DATA FILE ------------------------------------------------------
    {
        const std::string filepath = dG::io::make_path({level_step_folderpath, filename+"_"+std::to_string(rank)+".vtu"});
        
        vtk::print_unstructured_grid_data_vtu(filepath,
                                              n_VTK_nodes,
                                              n_VTK_cells,
                                              VTK_nodes,
                                              VTK_cell_conn,
                                              VTK_cell_offset,
                                              VTK_cell_type,
                                              VTK_nodal_fields,
                                              VTK_nodal_fields_names,
                                              VTK_cell_fields,
                                              VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Export the mesh internal boundary quadrature points to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_int_bou_quad_points_to_VTK(const std::string & level_folderpath,
                                       const std::string & level_step_folderpath,
                                       const std::string & step_string,
                                       const std::string & filename,
                                       const Geometry & /*geom*/,
                                       const Mesh & mesh,
                                       const iMultiFab & mask,
                                       IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // DOMAINS
    const int n_domains = ibvp.get_number_of_domains();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    vtk::Cell_conn_t n_VTK_nodes;
    vtk::Cell_offs_t n_VTK_cells;

    Vector<vtk::Float_t> VTK_nodes;
    Vector<vtk::Cell_conn_t> VTK_cell_conn;
    Vector<vtk::Cell_offs_t> VTK_cell_offset;
    Vector<vtk::Cell_type_t> VTK_cell_type;
    Vector<Vector<vtk::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<vtk::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_nodal_field_pos;
    // ----------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA --------------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    VTK_nodal_fields_names.push_back("w");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ----------------------------------------------------------------

    // EVAL REQUIRED MEMORY -------------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            if (CELL_IS_NOT_MASKED(mask_fab(i,j,k,dom)))
            {
                n_VTK_nodes += cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
            }
        }
    }
    // ----------------------------------------------------------------

    // ALLOCATE MEMORY ------------------------------------------------
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_conn.resize(VTK_cell_conn_size);
    VTK_cell_offset.resize(n_VTK_cells+1);
    VTK_cell_type.resize(n_VTK_cells);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }

    VTK_cell_fields.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields[f].resize(n_VTK_cells);
    }
    // ----------------------------------------------------------------

    // EVAL OUTPUT FIELDS ---------------------------------------------
    vtk::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    Vector<vtk::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<vtk::Int_t *> VTK_cell_fields_ptr;
    VTK_cell_fields_ptr.resize(n_VTK_cell_fields);
    for (int f = 0; f < n_VTK_cell_fields; ++f)
    {
        VTK_cell_fields_ptr[f] = VTK_cell_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;

    VTK_nodal_field_pos = 0L;

#ifdef AMREX_USE_GPU
    Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_dev_mem.data();
#else
    Real const * cell_int_bou_quad_mem_ptr = mesh.cell_int_bou_quad_host_mem.data();
#endif

    for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long const> const & cell_int_bou_quad_info_fab = mesh.cell_int_bou_quad_info.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            if (CELL_IS_NOT_MASKED(mask_fab(i,j,k,dom)))
            {
                // LOCAL PARAMETERS
                const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;

                // LOOP OVER THE QUADRATURE POINTS
                for (int q = 0; q < bou_Nq; ++q)
                {
                    // QUADRATURE POINT
                    const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0];,
                        x[1] = x_ptr[x_pos+1];,
                        x[2] = x_ptr[x_pos+2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // VTK NODE
#if (AMREX_SPACEDIM == 1)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = x[0];
                    VTK_nodes_ptr[VTK_nodes_pos+1] = x[1];
                    VTK_nodes_ptr[VTK_nodes_pos+2] = x[2];
#endif

                    // VTK NODAL FIELDS
                    // QUADRATURE POINT WEIGHT
                    VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = w;

                    // MOVE OFFSET
                    VTK_nodes_pos += 3;
                    VTK_nodal_field_pos += 1;
                }

                // LOOP OVER THE SUB CELLS
                // ...
            }
        }
    }
    // ----------------------------------------------------------------

    // HEADER FILE ----------------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {   
        vtk::print_header_vtu(level_folderpath, level_step_folderpath, step_string, filename,
                                VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------

    // DATA FILE ------------------------------------------------------
    {
        const std::string filepath = dG::io::make_path({level_step_folderpath, filename+"_"+std::to_string(rank)+".vtu"});
        
        vtk::print_unstructured_grid_data_vtu(filepath,
                                              n_VTK_nodes,
                                              n_VTK_cells,
                                              VTK_nodes,
                                              VTK_cell_conn,
                                              VTK_cell_offset,
                                              VTK_cell_type,
                                              VTK_nodal_fields,
                                              VTK_nodal_fields_names,
                                              VTK_cell_fields,
                                              VTK_cell_fields_names);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Export the mesh quadrature points to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename_root: root of the name of the output files.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_quadrature_points_to_VTK(const std::string & level_folderpath,
                                     const std::string & level_step_folderpath,
                                     const std::string & step_string,
                                     const std::string & filename_root,
                                     const Geometry & geom,
                                     const Mesh & mesh,
                                     const iMultiFab & mask,
                                     IBVP & ibvp)
{
    const std::string dom_data_filename = filename_root+"_dom_data";
    const std::string bou_data_filename = filename_root+"_bou_data";
    const std::string int_bou_data_filename = filename_root+"_int_bou_data";

    export_dom_quad_points_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                  geom, mesh, mask, ibvp);
    export_bou_quad_points_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                                  geom, mesh, mask, ibvp);
    export_int_bou_quad_points_to_VTK(level_folderpath, level_step_folderpath, step_string, int_bou_data_filename,
                                      geom, mesh, mask, ibvp);
}
// ====================================================================
// ####################################################################

} // namespace mesh_io
} // namespace dG
} // namespace amrex

#endif