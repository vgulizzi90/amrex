//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution_Output.H
 * \brief Contains output functions for implicit-mesh discontinuous Galerkin methods.
*/

#ifndef AMREX_DG_SOLUTION_OUTPUT_H_
#define AMREX_DG_SOLUTION_OUTPUT_H_

namespace amrex
{
namespace DG
{

/**
 * \brief Export solution contained in a MultiFab to VTK format.
 *
 * \param[in] folderpath: output folder path where the file will be dumped.
 * \param[in] filename_root: root of the output filename; it will be suffixed with an index denoting
 *                           the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names)
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F)
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath,
                const std::string & filename_root,
                const int n, 
                const int n_steps,
                const Real t,
                const Geometry & geom,
                const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_dom_Ng = std_elem.dom_Ng;
    const int cell_e = std_elem.g;
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = std_elem.dom_XI3g.data();
    )

    // DG
    const int sp = std_elem.p;

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(VTK_LINE_ID, VTK_QUAD_ID, VTK_HEXAHEDRON_ID);
    const int VTK_cell_conn_len = AMREX_D_PICK(VTK_LINE_CONN_LEN, VTK_QUAD_CONN_LEN, VTK_HEXAHEDRON_CONN_LEN);
    // ================================================================

    // VARIABLES ======================================================
    long VTK_cell_conn_size;

    int n_VTK_nodal_fields, n_VTK_cell_fields;
    
    VTK::Cell_conn_t n_VTK_nodes;
    VTK::Cell_offs_t n_VTK_cells;

    Vector<VTK::Float_t> VTK_nodes;
    Vector<VTK::Cell_conn_t> VTK_cell_conn;
    Vector<VTK::Cell_offs_t> VTK_cell_offset;
    Vector<VTK::Cell_type_t> VTK_cell_type;
    Vector<Vector<VTK::Float_t>> VTK_nodal_fields;
    Vector<std::string> VTK_nodal_fields_names;
    Vector<Vector<VTK::Int_t>> VTK_cell_fields;
    Vector<std::string> VTK_cell_fields_names;

    long VTK_nodes_pos;
    long VTK_cell_conn_pos, VTK_cell_offset_pos, VTK_cell_type_pos;
    long VTK_nodal_field_pos;
    long VTK_cell_field_pos;

    long VTK_cell_conn_offset;

    std::string filepath;
    // ================================================================

    // GET INFO FROM THE IBVP CLASS ===================================
    IBVP.F_FIELDS_NAMES(VTK_nodal_fields_names);
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    n_VTK_cell_fields = 0;
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
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
    // ================================================================

    // ALLOCATE MEMORY ================================================
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
    // ================================================================
    
    // EVAL OUTPUT FIELDS =============================================
    VTK::Float_t * VTK_nodes_ptr = VTK_nodes.data();
    
    VTK::Cell_conn_t * VTK_cell_conn_ptr = VTK_cell_conn.data();
    
    VTK::Cell_offs_t * VTK_cell_offset_ptr = VTK_cell_offset.data();
    VTK_cell_offset_ptr[0] = 0;
    
    VTK::Cell_type_t * VTK_cell_type_ptr = VTK_cell_type.data();

    Vector<VTK::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }

    Vector<VTK::Int_t *> VTK_cell_fields_ptr;
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

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & X_fab = X.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM];
            Real SOL[__DG_MAX_N_SOL__];
            Real FIELDS[__DG_MAX_N_FIELDS__];

            // SOLUTION
            DG_SOL_space_BFX sol(sp, cell_lo, cell_hi, X_fab, i, j, k);

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

                // VTK NODES ------------------------------------------
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
                // ----------------------------------------------------

                // EVAL SOLUTION --------------------------------------
                sol.eval(x, 0, N_SOL, SOL);
                // ----------------------------------------------------

                // VTK NODAL FIELDS -----------------------------------
                IBVP.F_FIELDS(t, x, SOL, FIELDS);

                for (int f = 0; f < n_VTK_nodal_fields; ++f)
                {
                    VTK_nodal_fields_ptr[f][VTK_nodal_field_pos] = FIELDS[f];
                }
                // ----------------------------------------------------
                
                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;
            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY ------------------------------
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                // VTK CELL FIELDS ------------------------------------
                
                // ----------------------------------------------------

                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            VTK_cell_conn_offset += cell_dom_Ng;
        }
    }
    // ================================================================

    // DUMP HEADER FILE ===============================================
    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath, filename_root, n, n_steps,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = VTK::MakeLocalOutputFilepath(folderpath, filename_root, "vtu", n, n_steps);

        VTK::PrintUnstructuredGridData_VTU(filepath,
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
    // ================================================================
}

} // namespace DG
} // namespace amrex

#endif