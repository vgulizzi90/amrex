//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution_Output.H
 * \brief Functions to output solutions.
*/

#ifndef AMREX_DG_SOLUTION_OUTPUT_H_
#define AMREX_DG_SOLUTION_OUTPUT_H_

#include <AMReX_VTK_Utils.H>

#include <AMReX_dG_Solution.H>

namespace amrex
{
namespace dG
{
namespace solution_io
{
// ELEMENT CENTROIDS ##################################################
/**
 * \brief Export the elements' centroids.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] solution: Solution object.
 * \param[in] ibvp: initial boundary value problem object.
 *
*/
template <typename IBVP>
void export_centroids_to_VTK(const std::string & level_folderpath,
                             const std::string & level_step_folderpath,
                             const std::string & step_string,
                             const std::string & filename,
                             const Solution & solution,
                             const IBVP & ibvp)
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

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ----------------------------------------------------------------

    // EVAL REQUIRED MEMORY -------------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            if (!ELM_IS_SMALL(elm_type_fab(i,j,k,ELM_TYPE(dom))))
            {
                n_VTK_nodes += 1;
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
    
    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);

        for (int dom = 0; dom < n_domains; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
            
            if (!ELM_IS_SMALL(etype))
            {
                // VTK NODE
#if (AMREX_SPACEDIM == 1)
                VTK_nodes_ptr[VTK_nodes_pos+0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));
                VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                VTK_nodes_ptr[VTK_nodes_pos+0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));
                VTK_nodes_ptr[VTK_nodes_pos+1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));
                VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                VTK_nodes_ptr[VTK_nodes_pos+0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));
                VTK_nodes_ptr[VTK_nodes_pos+1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));
                VTK_nodes_ptr[VTK_nodes_pos+2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
#endif

                // VTK NODAL FIELDS
                // ...

                // MOVE OFFSET
                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;

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
// ####################################################################



// SOLUTION INPUT/OUTPUT TO VTK #######################################
// DOMAIN / BOUNDARY DATA USING LEVEL SET =============================
/**
 * \brief Export the solution domain data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
*/
template <typename IBVP, typename LEVEL_SET>
void export_dom_data_using_level_set_to_VTK(const std::string & level_folderpath,
                                            const std::string & level_step_folderpath,
                                            const std::string & step_string,
                                            const std::string & filename,
                                            const Real t,
                                            const Geometry & geom,
                                            const Solution & solution,
                                            const MultiFab & X,
                                            const iMultiFab & mask,
                                            const IBVP & ibvp,
                                            const LEVEL_SET & level_set)
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

    // DOMAINS / UNKNOWN FIELDS / OUTPUT FIELDS
    const int n_domains = ibvp.get_number_of_domains();
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
    const int n_output_fields = ibvp.get_number_of_output_fields();

    // ORDER
    const int sp = solution.params.space_p;
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

    // OUTPUT FIELD NAMES
    Vector<std::string> output_fields_names;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_dom_Ng, cell_e, cell_dom_Ne;
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], solution.params.post_processing_grid_order);

    cell_dom_Ng = std_elm.dom_Ng;
    cell_e = std_elm.g;
    cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);

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

    // IBVP OUTPUT FIELD NAMES ------------------------------------
    ibvp.get_output_fields_names(output_fields_names);
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int o = 0; o < n_output_fields; ++o)
    {
        VTK_nodal_fields_names.push_back(output_fields_names[o]);
    }
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_type_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
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

#ifdef AMREX_USE_GPU
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_dev_mem.data();
#else
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_host_mem.data();
#endif

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);

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
            Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
            Real F[__DG_MAX_N_OUTPUT_FIELDS__];

            // LEVEL SETS
            // ...
            
            // UNKNOWN FIELDS
            space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, sp, X_fab);

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
                for (int o = 0; o < n_output_fields; ++o)
                {
                    const int dom = parent_dom_mem_ptr[o];
                    const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                    
                    bfu.set_support(&prob_lo[0], &dx[0], i, j, k, etype);
                    bfu.eval(x, 0, n_unknown_fields, U);

                    ibvp.eval_output_fields(dom, t, x, U, F);
                    VTK_nodal_fields_ptr[o][VTK_nodal_field_pos] = F[o];
                }

                // LEVEL SET
                VTK_nodal_fields_ptr[n_output_fields][VTK_nodal_field_pos] = level_set.eval(x);

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
                // ELEMENT TYPE
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_type_fab(i,j,k,ELM_TYPE(dom));
                }
                // MASK
                VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = mask_fab(i,j,k);

                // MOVE OFFSET
                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            // MOVE OFFSET
            VTK_cell_conn_offset += cell_dom_Ng;
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
 * \brief Export the solution domain data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
*/
template <typename IBVP, typename LEVEL_SET>
void export_dom_data_using_level_set_and_derivatives_to_VTK(const std::string & level_folderpath,
                                                            const std::string & level_step_folderpath,
                                                            const std::string & step_string,
                                                            const std::string & filename,
                                                            const Real t,
                                                            const Geometry & geom,
                                                            const Solution & solution,
                                                            const MultiFab & X,
                                                            const iMultiFab & mask,
                                                            const IBVP & ibvp,
                                                            const LEVEL_SET & level_set)
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

    // DOMAINS / UNKNOWN FIELDS / OUTPUT FIELDS
    const int n_domains = ibvp.get_number_of_domains();
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
    const int n_output_fields = ibvp.get_number_of_output_fields();

    // ORDER
    const int sp = solution.params.space_p;
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

    // OUTPUT FIELD NAMES
    Vector<std::string> output_fields_names;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_dom_Ng, cell_e, cell_dom_Ne;
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], solution.params.post_processing_grid_order);

    cell_dom_Ng = std_elm.dom_Ng;
    cell_e = std_elm.g;
    cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);

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

    // IBVP OUTPUT FIELD NAMES ------------------------------------
    ibvp.get_output_fields_names(output_fields_names);
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int o = 0; o < n_output_fields; ++o)
    {
        VTK_nodal_fields_names.push_back(output_fields_names[o]);
    }
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_type_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("id");
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
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

#ifdef AMREX_USE_GPU
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_dev_mem.data();
#else
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_host_mem.data();
#endif

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & elm_id_fab = solution.elm_id.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);

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
            Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
            Real AMREX_D_DECL(U_1[__DG_MAX_N_UNKNOWN_FIELDS__], U_2[__DG_MAX_N_UNKNOWN_FIELDS__], U_3[__DG_MAX_N_UNKNOWN_FIELDS__]);
            Real F[__DG_MAX_N_OUTPUT_FIELDS__];

            // LEVEL SETS
            // ...
            
            // UNKNOWN FIELDS
            space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, sp, X_fab);

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
                for (int o = 0; o < n_output_fields; ++o)
                {
                    const int dom = parent_dom_mem_ptr[o];
                    const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                    
                    bfu.set_support(&prob_lo[0], &dx[0], i, j, k, etype);
                    
                    bfu.eval_bf_table(x);
                    bfu.eval_unsafe(0, n_unknown_fields, U);
                    bfu.eval_derivatives_unsafe(0, n_unknown_fields, AMREX_D_DECL(U_1, U_2, U_3));

                    ibvp.eval_output_fields(dom, t, x, U, AMREX_D_DECL(U_1, U_2, U_3), F);
                    VTK_nodal_fields_ptr[o][VTK_nodal_field_pos] = F[o];
                }

                // LEVEL SET
                VTK_nodal_fields_ptr[n_output_fields][VTK_nodal_field_pos] = level_set.eval(x);

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
                // ELEMENT TYPE
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_type_fab(i,j,k,ELM_TYPE(dom));
                }
                // ID
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[n_domains+dom][VTK_cell_field_pos] = elm_id_fab(i,j,dom);
                }
                // MASK
                VTK_cell_fields_ptr[2*n_domains][VTK_cell_field_pos] = mask_fab(i,j,k);

                // MOVE OFFSET
                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            // MOVE OFFSET
            VTK_cell_conn_offset += cell_dom_Ng;
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
 * \brief Export the solution boundary data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
*/
template <typename IBVP, typename LEVEL_SET>
void export_bou_data_using_level_set_to_VTK(const std::string & level_folderpath,
                                            const std::string & level_step_folderpath,
                                            const std::string & step_string,
                                            const std::string & filename,
                                            const Real /*t*/,
                                            const Geometry & geom,
                                            const Solution & solution,
                                            const MultiFab & /*X*/,
                                            const iMultiFab & mask,
                                            const IBVP & ibvp,
                                            const LEVEL_SET & level_set)
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

    // ORDER
    //const int sp = solution.params.space_p;
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
    std_elm.set_grid(&dx[0], solution.params.post_processing_grid_order);

    cell_bou_Ng = std_elm.bou_Ng;
#if (AMREX_SPACEDIM > 1)
    cell_e = std_elm.g;
#endif
    cell_bou_Ne = AMREX_D_PICK(1, cell_e, cell_e*cell_e);
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
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_bou_type_dom_"+std::to_string(dom));
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
        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
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

        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
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
                    // LEVEL SET
                    VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = level_set.eval(x);

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
                    // ELM BOU TYPE
                    for (int dom = 0; dom < n_domains; ++dom)
                    {
                        VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }

                    // MASK
                    VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = (m_mask == p_mask) ? m_mask : __DG_LEVEL_BOUNDARY__;

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
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
 * \param[in] use_derivatives: if true, the derivatives of the solution are also provided to the
 *                             user.
 *
*/
template <typename IBVP, typename LEVEL_SET>
void export_using_level_set_to_VTK(const std::string & level_folderpath,
                                    const std::string & level_step_folderpath,
                                    const std::string & step_string,
                                    const std::string & filename_root,
                                    const Real t,
                                    const Geometry & geom,
                                    const Solution & solution,
                                    const MultiFab & X,
                                    const iMultiFab & mask,
                                    const IBVP & ibvp,
                                    const LEVEL_SET & level_set,
                                    const bool use_derivatives = false)
{
    const std::string dom_data_filename = filename_root+"_dom_data";
    const std::string bou_data_filename = filename_root+"_bou_data";

    if (use_derivatives)
    {
        export_dom_data_using_level_set_and_derivatives_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                                               t, geom, solution, X, mask, ibvp, level_set);
    }
    else
    {
        export_dom_data_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                               t, geom, solution, X, mask, ibvp, level_set);
    }
    export_bou_data_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                                           t, geom, solution, X, mask, ibvp, level_set);
}
// ====================================================================


// DOMAIN / BOUNDARY DATA USING PROJECTED LEVEL SET ===================
/**
 * \brief Export the solution domain data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] L: a MultiFabs containing the level set coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_dom_data_using_projected_level_set_to_VTK(const std::string & level_folderpath,
                                                      const std::string & level_step_folderpath,
                                                      const std::string & step_string,
                                                      const std::string & filename,
                                                      const Real t,
                                                      const Geometry & geom,
                                                      const Mesh & mesh,
                                                      const Solution & solution,
                                                      const MultiFab & X,
                                                      const MultiFab & L,
                                                      const iMultiFab & mask,
                                                      const IBVP & ibvp)
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

    // DOMAINS / UNKNOWN FIELDS / OUTPUT FIELDS
    const int n_domains = ibvp.get_number_of_domains();
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
    const int n_output_fields = ibvp.get_number_of_output_fields();

    // ORDER
    const int sp = solution.params.space_p;
    const int p = mesh.params.projected_level_set_p;
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

    // OUTPUT FIELD NAMES
    Vector<std::string> output_fields_names;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_dom_Ng, cell_e, cell_dom_Ne;
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], solution.params.post_processing_grid_order);

    cell_dom_Ng = std_elm.dom_Ng;
    cell_e = std_elm.g;
    cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);

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

    // IBVP OUTPUT FIELD NAMES ------------------------------------
    ibvp.get_output_fields_names(output_fields_names);
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int o = 0; o < n_output_fields; ++o)
    {
        VTK_nodal_fields_names.push_back(output_fields_names[o]);
    }
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_type_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
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

#ifdef AMREX_USE_GPU
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_dev_mem.data();
#else
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_host_mem.data();
#endif

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & L_fab = L.array(mfi);

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
            Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
            Real F[__DG_MAX_N_OUTPUT_FIELDS__];

            // UNKNOWN FIELDS
            space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, sp, X_fab);

            // LEVEL SET FUNCTION
            space_cell_bfx_level_set phi(&prob_lo[0], &dx[0], i, j, k, p, L_fab);
            phi.u = 0;

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
                for (int o = 0; o < n_output_fields; ++o)
                {
                    const int dom = parent_dom_mem_ptr[o];
                    const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                    
                    bfu.set_support(&prob_lo[0], &dx[0], i, j, k, etype);
                    bfu.eval(x, 0, n_unknown_fields, U);

                    ibvp.eval_output_fields(dom, t, x, U, F);
                    VTK_nodal_fields_ptr[o][VTK_nodal_field_pos] = F[o];
                }

                // LEVEL SET
                VTK_nodal_fields_ptr[n_output_fields][VTK_nodal_field_pos] = phi.eval(x);

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
                // ELEMENT TYPE
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_type_fab(i,j,k,ELM_TYPE(dom));
                }
                // MASK
                VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = mask_fab(i,j,k);

                // MOVE OFFSET
                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            // MOVE OFFSET
            VTK_cell_conn_offset += cell_dom_Ng;
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
 * \brief Export the solution boundary data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] L: a MultiFabs containing the level set coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
*/
template <typename IBVP>
void export_bou_data_using_projected_level_set_to_VTK(const std::string & level_folderpath,
                                                      const std::string & level_step_folderpath,
                                                      const std::string & step_string,
                                                      const std::string & filename,
                                                      const Real /*t*/,
                                                      const Geometry & geom,
                                                      const Mesh & mesh,
                                                      const Solution & solution,
                                                      const MultiFab & /*X*/,
                                                      const MultiFab & L,
                                                      const iMultiFab & mask,
                                                      const IBVP & ibvp)
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

    // ORDER
    const int p = mesh.params.projected_level_set_p;
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
    std_elm.set_grid(&dx[0], solution.params.post_processing_grid_order);

    cell_bou_Ng = std_elm.bou_Ng;
#if (AMREX_SPACEDIM > 1)
    cell_e = std_elm.g;
#endif
    cell_bou_Ne = AMREX_D_PICK(1, cell_e, cell_e*cell_e);
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
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_bou_type_dom_"+std::to_string(dom));
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
        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
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

        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<Real const> const & L_fab = L.array(mfi);
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

                // LEVEL SET FUNCTION
                space_cell_bfx_level_set m_phi(&prob_lo[0], &dx[0], mi, mj, mk, p, L_fab);
                space_cell_bfx_level_set p_phi(&prob_lo[0], &dx[0], pi, pj, pk, p, L_fab);
                m_phi.u = 0;
                p_phi.u = 0;

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
                    // LEVEL SET
                    VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = 0.5*(m_phi.eval(x)+p_phi.eval(x));

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
                    // ELM BOU TYPE
                    for (int dom = 0; dom < n_domains; ++dom)
                    {
                        VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }

                    // MASK
                    VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = (m_mask == p_mask) ? m_mask : __DG_LEVEL_BOUNDARY__;

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
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] L: a MultiFabs containing the level set coefficients.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 *
*/
template <typename IBVP>
void export_using_projected_level_set_to_VTK(const std::string & level_folderpath,
                                             const std::string & level_step_folderpath,
                                             const std::string & step_string,
                                             const std::string & filename_root,
                                             const Real t,
                                             const Geometry & geom,
                                             const Mesh & mesh,
                                             const Solution & solution,
                                             const MultiFab & X,
                                             const MultiFab & L,
                                             const iMultiFab & mask,
                                             const IBVP & ibvp)
{
    const std::string dom_data_filename = filename_root+"_dom_data";
    const std::string bou_data_filename = filename_root+"_bou_data";

    export_dom_data_using_projected_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                                     t, geom, mesh, solution, X, L, mask, ibvp);
    export_bou_data_using_projected_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                                                     t, geom, mesh, solution, X, L, mask, ibvp);
}

/**
 * \brief Export the mesh to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename_root: root of the name of the output files.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution coefficients.
 * \param[in] L: a MultiFabs containing the level set coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
*/
template <typename IBVP>
void export_using_projected_level_set_to_VTK(const std::string & level_folderpath,
                                             const std::string & level_step_folderpath,
                                             const std::string & step_string,
                                             const std::string & filename_root,
                                             const Real t,
                                             const Geometry & geom,
                                             const Mesh & mesh,
                                             const Solution & solution,
                                             const MultiFab & X,
                                             const MultiFab & L,
                                             const IBVP & ibvp)
{
    iMultiFab mask(X.boxarray, X.distributionMap, 1, X.n_grow);
    mask = __DG_CELL_NOT_MASKED__;

    export_using_projected_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                            t, geom, mesh, solution, X, L, mask, ibvp);
}
// ====================================================================
// ####################################################################



// SOLUTION INPUT/OUTPUT TO VTK - FINITE VOLUME SCHEME ################
// DOMAIN / BOUNDARY DATA USING LEVEL SET =============================
/**
 * \brief Export the solution domain data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution states.
 * \param[in] dX: a MultiFabs containing the solution slopes.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
*/
template <typename IBVP, typename LEVEL_SET>
void export_FV_dom_data_using_level_set_to_VTK(const std::string & level_folderpath,
                                               const std::string & level_step_folderpath,
                                               const std::string & step_string,
                                               const std::string & filename,
                                               const Real t,
                                               const Geometry & geom,
                                               const Solution & solution,
                                               const MultiFab & X,
                                               const MultiFab & dX,
                                               const iMultiFab & mask,
                                               const IBVP & ibvp,
                                               const LEVEL_SET & level_set)
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

    // DOMAINS / UNKNOWN FIELDS / OUTPUT FIELDS
    const int n_domains = ibvp.get_number_of_domains();
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
    const int n_output_fields = ibvp.get_number_of_output_fields();
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

    // OUTPUT FIELD NAMES
    Vector<std::string> output_fields_names;

    // STANDARD ELEMENT
    StandardRectangle<AMREX_SPACEDIM> std_elm;
    int cell_dom_Ng, cell_e, cell_dom_Ne;
    Gpu::ManagedVector<int> subgrid_conn;
    // ------------------------------------------------------------

    // INITIALIZE THE GRID POINTS OF THE STANDARD ELEMENT ---------
    std_elm.set_grid(&dx[0], 1);

    cell_dom_Ng = std_elm.dom_Ng;
    cell_e = std_elm.g;
    cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);

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

    // IBVP OUTPUT FIELD NAMES ------------------------------------
    ibvp.get_output_fields_names(output_fields_names);
    // ------------------------------------------------------------

    // GET NODAL FIELDS AND CELL FIELDS DATA ----------------------
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int o = 0; o < n_output_fields; ++o)
    {
        VTK_nodal_fields_names.push_back(output_fields_names[o]);
    }
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_type_dom_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("mask");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ------------------------------------------------------------

    // EVAL REQUIRED MEMORY ---------------------------------------
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
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

#ifdef AMREX_USE_GPU
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_dev_mem.data();
#else
    int const * parent_dom_mem_ptr = solution.output_field_parent_domain_host_mem.data();
#endif

    for (MFIter mfi(solution.elm_type); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        Array4<Real const> const & elm_centroid_fab = solution.elm_centroid.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], xc[AMREX_SPACEDIM];
            Real U[__DG_MAX_N_UNKNOWN_FIELDS__];
            Real F[__DG_MAX_N_OUTPUT_FIELDS__];

            // LEVEL SETS
            // ...
            
            // UNKNOWN FIELDS
            space_elm_rx ru(i, j, k, X_fab, dX_fab);

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
                for (int o = 0; o < n_output_fields; ++o)
                {
                    const int dom = parent_dom_mem_ptr[o];
                    AMREX_D_TERM
                    (
                        xc[0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        xc[1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        xc[2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    
                    ru.set_centroid(xc);
                    ru.eval(x, 0, n_unknown_fields, U);

                    ibvp.eval_output_fields(dom, t, x, U, F);
                    VTK_nodal_fields_ptr[o][VTK_nodal_field_pos] = F[o];
                }

                // LEVEL SET
                VTK_nodal_fields_ptr[n_output_fields][VTK_nodal_field_pos] = level_set.eval(x);

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
                // ELEMENT TYPE
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_type_fab(i,j,k,ELM_TYPE(dom));
                }
                // MASK
                VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = mask_fab(i,j,k);

                // MOVE OFFSET
                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            // MOVE OFFSET
            VTK_cell_conn_offset += cell_dom_Ng;
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
 * \brief Export the solution boundary data to VTK format.
 *
 * \param[in] level_folderpath: path where the .pvtu files will be written.
 * \param[in] level_step_folderpath: path where the .vtu files will be written.
 * \param[in] step_string: a zero-padded string containing the step number.
 * \param[in] filename: name of the output file.
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution states.
 * \param[in] dX: a MultiFabs containing the solution slopes.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
*/
template <typename IBVP, typename LEVEL_SET>
void export_FV_bou_data_using_level_set_to_VTK(const std::string & level_folderpath,
                                               const std::string & level_step_folderpath,
                                               const std::string & step_string,
                                               const std::string & filename,
                                               const Real /*t*/,
                                               const Geometry & geom,
                                               const Solution & solution,
                                               const MultiFab & /*X*/,
                                               const MultiFab & /*dX*/,
                                               const iMultiFab & mask,
                                               const IBVP & ibvp,
                                               const LEVEL_SET & level_set)
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

    // ORDER
    //const int sp = solution.params.space_p;
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
    cell_bou_Ne = AMREX_D_PICK(1, cell_e, cell_e*cell_e);
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
    VTK_nodal_fields_names.push_back("level_set");

    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < n_domains; ++dom)
    {
        VTK_cell_fields_names.push_back("elm_bou_type_dom_"+std::to_string(dom));
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
        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
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

        for (MFIter mfi(solution.elm_bou_type[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
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
                    // LEVEL SET
                    VTK_nodal_fields_ptr[0][VTK_nodal_field_pos] = level_set.eval(x);

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
                    // ELM BOU TYPE
                    for (int dom = 0; dom < n_domains; ++dom)
                    {
                        VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }

                    // MASK
                    VTK_cell_fields_ptr[n_domains][VTK_cell_field_pos] = (m_mask == p_mask) ? m_mask : __DG_LEVEL_BOUNDARY__;

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
 * \param[in] t: time.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution states.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] ibvp: initial boundary value problem object.
 * \param[in] level_set: level set object.
 *
*/
template <typename IBVP, typename LEVEL_SET>
void export_FV_using_level_set_to_VTK(const std::string & level_folderpath,
                                      const std::string & level_step_folderpath,
                                      const std::string & step_string,
                                      const std::string & filename_root,
                                      const Real t,
                                      const Geometry & geom,
                                      const Mesh & mesh,
                                      const Solution & solution,
                                      const MultiFab & X,
                                      const iMultiFab & mask,
                                      const IBVP & ibvp,
                                      const LEVEL_SET & level_set)
{
    // PARAMETERS -----------------------------------------------------
    const std::string dom_data_filename = filename_root+"_dom_data";
    const std::string bou_data_filename = filename_root+"_bou_data";
    const std::string centroid_data_filename = filename_root+"_centroids";

    // UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    MultiFab dX(X.boxarray, X.distributionMap, AMREX_SPACEDIM*n_unknown_fields, 1);
    // ----------------------------------------------------------------

    // EVAL THE RECONSTRUCTED SLOPES ----------------------------------
    ibvp.eval_FV_slopes(geom, mesh, solution, X, dX);
    // ----------------------------------------------------------------
    
    // EXPORT SOLUTION ------------------------------------------------
    export_FV_dom_data_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                              t, geom, solution, X, dX, mask, ibvp, level_set);
    export_FV_bou_data_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                                              t, geom, solution, X, dX, mask, ibvp, level_set);

    export_centroids_to_VTK(level_folderpath, level_step_folderpath, step_string, centroid_data_filename,
                            solution, ibvp);
    // ----------------------------------------------------------------
}
// ====================================================================
// ####################################################################

} // namespace solution_io
} // namespace dG
} // namespace amrex

#endif