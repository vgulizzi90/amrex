//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_Solution_Output.H
 * \brief Contains output functions for implicit-mesh discontinuous Galerkin methods.
*/

#ifndef AMREX_DG_SOLUTION_OUTPUT_H_
#define AMREX_DG_SOLUTION_OUTPUT_H_

//#define __DG_EXPORT_IMPLICIT_MESH_DOMDATA__
#define __DG_EXPORT_IMPLICIT_MESH_BOUDATA__
//#define __DG_EXPORT_IMPLICIT_MESH_CENTROIDS__

#include <AMReX_DG_IO.H>

namespace amrex
{
namespace DG
{

/**
 * \brief Export solution contained in a MultiFab to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const std::string & filename_root,
                const Real t,
                const Geometry & geom,
                const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                const iMultiFab & mask,
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
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
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
    // NODAL DATA
    IBVP.F_FIELDS_NAMES(VTK_nodal_fields_names);
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names = {"TAG"};

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    if (N_SOL > __DG_MAX_N_SOL__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution_Hyperbolic.H - Compute_dt\n";
        msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
        msg += "| N_SOL           : "+std::to_string(N_SOL)+".\n";
        msg += "| __DG_MAX_N_SOL__: "+std::to_string(__DG_MAX_N_SOL__)+".\n";
        Abort(msg);
    }
    if (n_VTK_nodal_fields > __DG_MAX_N_FIELDS__)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_DG_Solution_Output.H - Export_VTK\n";
        msg += "| The number of output fields exceeds the maximum allowed number.\n";
        msg += "| n_VTK_nodal_fields : "+std::to_string(n_VTK_nodal_fields)+".\n";
        msg += "| __DG_MAX_N_FIELDS__: "+std::to_string(__DG_MAX_N_FIELDS__)+".\n";
        Abort(msg);
    }
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
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
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

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
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                // VTK CELL FIELDS ------------------------------------
                // TAG
                VTK_cell_fields_ptr[0][VTK_cell_field_pos] = mask_fab(i,j,k);
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
    IO::MakeStepFolder(folderpath_root, n, n_steps);
    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export solution contained in a MultiFab to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const std::string & filename_root,
                const Real t,
                const Geometry & geom,
                const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    iMultiFab mask(X.boxarray, X.distributionMap, 1, X.n_grow);
    mask = __DG_CELL_UNMASKED__;

    Export_VTK(folderpath_root, n, n_steps, filename_root, t, geom, std_elem, mask, N_SOL, X, IBVP);
}

/**
 * \brief Auxiliary class used for exporting fields contained in a MultiFab.
*/
struct IBVP_FIELDS
{
    // DATA MEMBERS
    const int n_fields;
    const std::string field_root;

    // CONSTRUCTOR
    IBVP_FIELDS(const int n, const std::string fr)
    :
    n_fields{n},
    field_root{fr}
    {}

    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void F_FIELDS_NAMES(amrex::Vector<std::string> & fields_names) const
    {
        fields_names.resize(this->n_fields);
        for (int n = 0; n < (this->n_fields); ++n)
        {
            fields_names[n] = this->field_root+std::to_string(n);
        }
    }

    /**
     * \brief Fill F with fields to be output.
     *
     * This method just copies SOL into F.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_FIELDS(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                  const amrex::Real * SOL,
                  amrex::Real * F) const
    {
        for (int n = 0; n < (this->n_fields); ++n)
        {
            F[n] = SOL[n];
        }
    }
};

/**
 * \brief Export fields contained in a MultiFab to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] geom: a Geometry object that contains geometry information.
 * \param[in] std_elem: a StandardRectangle<AMREX_SPACEDIM> object that contains evaluation grid
 *                      points and basis functions data.
 * \param[in] N_F: Number of fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               output fields.
 * \param[in] field_root: a string that will be used as a root for the names of the output fields.
*/
void ExportBase_VTK(const std::string & folderpath_root,
                    const int n, 
                    const int n_steps,
                    const std::string & filename_root,
                    const Real t,
                    const Geometry & geom,
                    const StandardRectangle<AMREX_SPACEDIM> & std_elem,
                    const int N_F,
                    const MultiFab & X,
                    const std::string & field_root)
{
    const IBVP_FIELDS FIELDS(N_F, field_root);
    Export_VTK(folderpath_root, n, n_steps, filename_root, t, geom, std_elem, N_F, X, FIELDS);
}

/**
 * \brief Export the domain data of the implicitly-defined mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_DOM: number of domains to be considered.
*/
void ExportImplicitMesh_DomData_VTK(const std::string & folderpath_root,
                                    const int n, 
                                    const int n_steps,
                                    const std::string & filename_root,
                                    const Real /*t*/,
                                    const ImplicitMesh & mesh,
                                    const iMultiFab & mask,
                                    const int N_PHI,
                                    const int N_DOM)
{
    // PARAMETERS =====================================================
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_dom_Ng = mesh.ls_std_elem.dom_Ng;
    const int cell_e = mesh.ls_std_elem.g;
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = mesh.ls_std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = mesh.ls_std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = mesh.ls_std_elem.dom_XI3g.data();
    )

    // DG
    const int sp = mesh.ls_std_elem.p;

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

    std::string filepath, folderpath;
    // ================================================================

    // GET NODAL FIELDS AND CELL FIELDS DATA ==========================
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int ph = 0; ph < N_PHI; ++ph)
    {
        VTK_nodal_fields_names.push_back("PHI_"+std::to_string(ph));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    VTK_cell_fields_names.push_back("RANK");
    VTK_cell_fields_names.push_back("BOX");
    VTK_cell_fields_names.push_back("I");
    VTK_cell_fields_names.push_back("J");
    VTK_cell_fields_names.push_back("K");
    for (int dom = 0; dom < N_DOM; ++dom)
    {
        VTK_cell_fields_names.push_back("ELM_TYPE_DOM_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("NQ");
    VTK_cell_fields_names.push_back("TAG");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================
    
    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(mesh.PHI); mfi.isValid(); ++mfi)
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
    long bx_id;
    bx_id = 0L;

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
    for (MFIter mfi(mesh.PHI); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

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
            Real PHI[__DG_MAX_N_PHI__];

            // LEVELSETS
            DG_PHI_space_BFX phi(sp, cell_lo, cell_hi, PHI_fab, i, j, k);

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

                // EVAL LEVELSETS -------------------------------------
                phi.eval(x, 0, N_PHI, PHI);
                // ----------------------------------------------------

                // VTK NODAL FIELDS -----------------------------------
                for (int ph = 0; ph < N_PHI; ++ph)
                {
                    VTK_nodal_fields_ptr[ph][VTK_nodal_field_pos] = PHI[ph];
                }
                // ----------------------------------------------------
                
                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;
            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY ------------------------------
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                // VTK CELL FIELDS ------------------------------------
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

                // ELM_TYPE
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    VTK_cell_fields_ptr[5+dom][VTK_cell_field_pos] = eType_fab(i,j,k,ELM_TYPE(dom))%10;
                }

                // NUMBER OF QUADRATURE POINTS
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    VTK_cell_fields_ptr[5+N_DOM+dom][VTK_cell_field_pos] = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                }

                // TAG
                VTK_cell_fields_ptr[5+N_DOM+N_DOM][VTK_cell_field_pos] = mask_fab(i,j,k);
                // ----------------------------------------------------

                VTK_cell_conn_pos += VTK_cell_conn_len;
                VTK_cell_offset_pos += 1;
                VTK_cell_type_pos += 1;
                VTK_cell_field_pos += 1;
            }

            VTK_cell_conn_offset += cell_dom_Ng;
        }

        bx_id += 1;
    }
    // ================================================================

    // DUMP HEADER FILE ===============================================
    IO::MakeStepFolder(folderpath_root, n, n_steps);

    if (ParallelDescriptor::IOProcessor())
    {   
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export the boundary data of the implicitly-defined mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_DOM: number of domains to be considered.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
*/
void ExportImplicitMesh_BouData_VTK(const std::string & folderpath_root,
                                    const int n, 
                                    const int n_steps,
                                    const std::string & filename_root,
                                    const Real /*t*/,
                                    const ImplicitMesh & mesh,
                                    const iMultiFab & mask,
                                    const int N_PHI,
                                    const int N_DOM)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_bou_Ng = mesh.ls_std_elem.bou_Ng;
#if (AMREX_SPACEDIM > 1)
    const int cell_e = mesh.ls_std_elem.g;
#endif
    const int cell_bou_Ne = AMREX_D_PICK(1, cell_e, cell_e*cell_e);
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = {0};
#endif
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * bou_XI1g_ptr = mesh.ls_std_elem.bou_XI1g.data();,
        Real const * bou_XI2g_ptr = mesh.ls_std_elem.bou_XI2g.data();,
        Real const * bou_XI3g_ptr = mesh.ls_std_elem.bou_XI3g.data();
    )

    // DG
    const int sp = mesh.ls_std_elem.p;

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(VTK_VERTEX_ID, VTK_LINE_ID, VTK_QUAD_ID);
    const int VTK_cell_conn_len = AMREX_D_PICK(VTK_VERTEX_CONN_LEN, VTK_LINE_CONN_LEN, VTK_QUAD_CONN_LEN);
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

    std::string filepath, folderpath;
    // ================================================================

    // GET NODAL FIELDS AND CELL FIELDS DATA ==========================
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int ph = 0; ph < N_PHI; ++ph)
    {
        VTK_nodal_fields_names.push_back("PHI_"+std::to_string(ph));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    for (int dom = 0; dom < N_DOM; ++dom)
    {
        VTK_cell_fields_names.push_back("BOU_TYPE_DOM_"+std::to_string(dom));
    }
    for (int dom = 0; dom < N_DOM; ++dom)
    {
        VTK_cell_fields_names.push_back("VALID_DOM_"+std::to_string(dom));
    }
    VTK_cell_fields_names.push_back("TAG");

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        for (MFIter mfi(mesh.eBouType[dir]); mfi.isValid(); ++mfi)
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

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;

        for (MFIter mfi(mesh.eBouType[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
            //Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<short const> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // LOCAL VARIABLES
                int mi, mj, mk, pi, pj, pk;
                Real m_cell_lo[AMREX_SPACEDIM], m_cell_hi[AMREX_SPACEDIM];
                Real p_cell_lo[AMREX_SPACEDIM], p_cell_hi[AMREX_SPACEDIM];
                /*
                int m_etype, p_etype;
                bool m_elm_is_small, p_elm_is_small, cells_are_merged;
                int m_merged_b, p_merged_b;
                */
                int m_mask, p_mask;
                Real x[AMREX_SPACEDIM];
                Real m_PHI[__DG_MAX_N_PHI__], p_PHI[__DG_MAX_N_PHI__];

                // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                FACE2NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);

                AMREX_D_TERM
                (
                    m_cell_lo[0] = prob_lo[0]+mi*dx[0];,
                    m_cell_lo[1] = prob_lo[1]+mj*dx[1];,
                    m_cell_lo[2] = prob_lo[2]+mk*dx[2];
                )
                AMREX_D_TERM
                (
                    m_cell_hi[0] = prob_lo[0]+(mi+1)*dx[0];,
                    m_cell_hi[1] = prob_lo[1]+(mj+1)*dx[1];,
                    m_cell_hi[2] = prob_lo[2]+(mk+1)*dx[2];
                )
                AMREX_D_TERM
                (
                    p_cell_lo[0] = prob_lo[0]+pi*dx[0];,
                    p_cell_lo[1] = prob_lo[1]+pj*dx[1];,
                    p_cell_lo[2] = prob_lo[2]+pk*dx[2];
                )
                AMREX_D_TERM
                (
                    p_cell_hi[0] = prob_lo[0]+(pi+1)*dx[0];,
                    p_cell_hi[1] = prob_lo[1]+(pj+1)*dx[1];,
                    p_cell_hi[2] = prob_lo[2]+(pk+1)*dx[2];
                )

                // LEVELSETS
                DG_PHI_space_BFX m_phi(sp, m_cell_lo, m_cell_hi, PHI_fab, mi, mj, mk);
                DG_PHI_space_BFX p_phi(sp, p_cell_lo, p_cell_hi, PHI_fab, pi, pj, pk);

                // LOOP OVER THE GRID POINTS
                for (int g = 0; g < cell_bou_Ng; ++g)
                {
                    // GRID POINT
                    AMREX_D_TERM
                    (
                        x[0] = bou_XI1g_ptr[b*cell_bou_Ng+g]+p_cell_lo[0];,
                        x[1] = bou_XI2g_ptr[b*cell_bou_Ng+g]+p_cell_lo[1];,
                        x[2] = bou_XI3g_ptr[b*cell_bou_Ng+g]+p_cell_lo[2];
                    )

                    // VTK NODES --------------------------------------
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
                    // ------------------------------------------------

                    // EVAL LEVELSETS ---------------------------------
                    m_phi.eval(x, 0, N_PHI, m_PHI);
                    p_phi.eval(x, 0, N_PHI, p_PHI);
                    // ------------------------------------------------

                    // VTK NODAL FIELDS -------------------------------
                    for (int ph = 0; ph < N_PHI; ++ph)
                    {
                        VTK_nodal_fields_ptr[ph][VTK_nodal_field_pos] = 0.5*(m_PHI[ph]+p_PHI[ph]);
                    }
                    // ------------------------------------------------
                    
                    VTK_nodes_pos += 3;
                    VTK_nodal_field_pos += 1;
                }

                // LOOP OVER THE SUB CELLS
                for (int e = 0; e < cell_bou_Ne; ++e)
                {
                    // VTK CELL CONNECTIVITY --------------------------
                    const int offset = e*VTK_cell_conn_len;
                    for (int c = 0; c < VTK_cell_conn_len; ++c)
                    {
                        VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                    }
                    // ------------------------------------------------

                    // VTK CELL OFFSET --------------------------------
                    VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                    // ------------------------------------------------

                    // VTK CELL TYPE ----------------------------------
                    VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                    // ------------------------------------------------

                    // VTK CELL FIELDS --------------------------------
                    // BOU_ELM_TYPE
                    for (int dom = 0; dom < N_DOM; ++dom)
                    {
                        VTK_cell_fields_ptr[dom][VTK_cell_field_pos] = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));
                    }

                    // VALID-VS-INVALID BOUNDARY
                    {
                        for (int dom = 0; dom < N_DOM; ++dom)
                        {
                            const short bou_type = eBouType_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                            if (ELM_BOU_IS_VALID(bou_type))
                            {
                                VTK_cell_fields_ptr[N_DOM+dom][VTK_cell_field_pos] = 1;
                            }
                            else
                            {
                                VTK_cell_fields_ptr[N_DOM+dom][VTK_cell_field_pos] = 0;
                            }
                        }
                    }

                    // TAG
                    {
                        // elm^-
                        m_mask = mask_fab(mi,mj,mk);

                        // elm^+
                        p_mask = mask_fab(pi,pj,pk);

                        // If both cells are tagged, so is their shared
                        // boundary
                        if ((m_mask == __DG_CELL_MASKED__) && (p_mask == __DG_CELL_MASKED__))
                        {
                            VTK_cell_fields_ptr[N_DOM+N_DOM][VTK_cell_field_pos] = __DG_CELL_MASKED__;
                        }
                        else if ((m_mask == __DG_CELL_UNMASKED__) && (p_mask == __DG_CELL_UNMASKED__))
                        {
                            VTK_cell_fields_ptr[N_DOM+N_DOM][VTK_cell_field_pos] = __DG_CELL_UNMASKED__;
                        }
                        else
                        {
                            VTK_cell_fields_ptr[N_DOM+N_DOM][VTK_cell_field_pos] = __DG_LEVEL_BOUNDARY__;
                        }
                    }
                    // ------------------------------------------------

                    VTK_cell_conn_pos += VTK_cell_conn_len;
                    VTK_cell_offset_pos += 1;
                    VTK_cell_type_pos += 1;
                    VTK_cell_field_pos += 1;
                }

                VTK_cell_conn_offset += cell_bou_Ng;
            }
        }   
    }
    // ================================================================

    // DUMP HEADER FILE ===============================================
    IO::MakeStepFolder(folderpath_root, n, n_steps);
    
    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export the centroids of the implicitly-defined mesh elements to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_DOM: number of domains to be considered.
*/
void ExportImplicitMesh_Centroids_VTK(const std::string & folderpath_root,
                                      const int n, 
                                      const int n_steps,
                                      const std::string & filename_root,
                                      const Real /*t*/,
                                      const ImplicitMesh & mesh,
                                      const int N_DOM)
{
    // PARAMETERS =====================================================
    // ================================================================

    // VARIABLES ======================================================
    int n_VTK_nodal_fields;
    
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
    long VTK_nodal_field_pos;

    std::string filepath, folderpath;
    // ================================================================

    // GET NODAL FIELDS AND CELL FIELDS DATA ==========================
    // NODAL DATA
    VTK_nodal_fields_names.clear();
    for (int dom = 0; dom < N_DOM; ++dom)
    {
        VTK_nodal_fields_names.push_back("VALID_DOM_"+std::to_string(dom));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // CELL DATA
    VTK_cell_fields_names.clear();
    // ================================================================
    
    // EVAL REQUIRED MEMORY ===========================================
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int dom = 0; dom < N_DOM; ++dom)
            {
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

                if (elm_is_not_empty)
                {
                    n_VTK_nodes += 1;
                }
            }
        }
    }
    // ================================================================

    // ALLOCATE MEMORY ================================================
    VTK_nodes.resize(3*n_VTK_nodes);
    VTK_cell_offset.resize(n_VTK_cells+1);

    VTK_nodal_fields.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields[f].resize(n_VTK_nodes);
    }
    // ================================================================

    // EVAL OUTPUT FIELDS =============================================
    VTK::Float_t * VTK_nodes_ptr = VTK_nodes.data();

    Vector<VTK::Float_t *> VTK_nodal_fields_ptr;
    VTK_nodal_fields_ptr.resize(n_VTK_nodal_fields);
    for (int f = 0; f < n_VTK_nodal_fields; ++f)
    {
        VTK_nodal_fields_ptr[f] = VTK_nodal_fields[f].data();
    }
    
    VTK_nodes_pos = 0L;

    VTK_nodal_field_pos = 0L;

    // DOMAIN
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int dom = 0; dom < N_DOM; ++dom)
            {
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);
                const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);

                if (elm_is_not_empty)
                {
                    // VTK NODES --------------------------------------
#if (AMREX_SPACEDIM == 1)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));
                    VTK_nodes_ptr[VTK_nodes_pos+1] = 0.0;
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 2)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));
                    VTK_nodes_ptr[VTK_nodes_pos+1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));
                    VTK_nodes_ptr[VTK_nodes_pos+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
                    VTK_nodes_ptr[VTK_nodes_pos+0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom));
                    VTK_nodes_ptr[VTK_nodes_pos+1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom));
                    VTK_nodes_ptr[VTK_nodes_pos+2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom));
#endif
                    // ------------------------------------------------

                    // VTK NODAL FIELDS -------------------------------
                    if (elm_is_small)
                    {
                        VTK_nodal_fields_ptr[dom][VTK_nodal_field_pos] = 0.0;
                    }
                    else
                    {
                        VTK_nodal_fields_ptr[dom][VTK_nodal_field_pos] = 1.0;
                    }
                    // ------------------------------------------------

                    VTK_nodes_pos += 3;
                    VTK_nodal_field_pos += 1;
                }
            }
        }
    }
    // ================================================================

    // DUMP HEADER FILE ===============================================
    IO::MakeStepFolder(folderpath_root, n, n_steps);

    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export the implicitly-defined mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_DOM: number of domains to be considered.
*/
void ExportImplicitMesh_VTK(const std::string & folderpath_root,
                            const int n, 
                            const int n_steps,
                            const std::string & filename_root,
                            const Real t,
                            const ImplicitMesh & mesh,
                            const iMultiFab & mask,
                            const int N_PHI,
                            const int N_DOM)
{
#ifdef __DG_EXPORT_IMPLICIT_MESH_DOMDATA__
    ExportImplicitMesh_DomData_VTK(folderpath_root, n, n_steps, filename_root+"Dom",
                                   t, mesh, mask, N_PHI, N_DOM);
#endif
#ifdef __DG_EXPORT_IMPLICIT_MESH_BOUDATA__
    ExportImplicitMesh_BouData_VTK(folderpath_root, n, n_steps, filename_root+"Bou",
                                   t, mesh, mask, N_PHI, N_DOM);
#endif
#ifdef __DG_EXPORT_IMPLICIT_MESH_CENTROIDS__
    ExportImplicitMesh_Centroids_VTK(folderpath_root, n, n_steps, filename_root+"Centroids",
                                     t, mesh, N_DOM);
#endif
}

/**
 * \brief Export the implicitly-defined mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_DOM: number of domains to be considered.
*/
void ExportImplicitMesh_VTK(const std::string & folderpath_root,
                            const int n, 
                            const int n_steps,
                            const std::string & filename_root,
                            const Real t,
                            const ImplicitMesh & mesh,
                            const int N_PHI,
                            const int N_DOM)
{
    iMultiFab mask(mesh.PHI.boxarray, mesh.PHI.distributionMap, 1, mesh.PHI.n_grow);
    mask = __DG_CELL_UNMASKED__;

    ExportImplicitMesh_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_DOM);
}

/**
 * \brief Export the implicitly-defined mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: the ImplicitMesh object that will be exported.
*/
void ExportImplicitMesh_VTK(const std::string & folderpath_root,
                            const int n, 
                            const int n_steps,
                            const std::string & filename_root,
                            const Real t,
                            const ImplicitMesh & mesh)
{
    const int N_PHI = 1;
    const int N_DOM = 1;
    iMultiFab mask(mesh.PHI.boxarray, mesh.PHI.distributionMap, 1, mesh.PHI.n_grow);
    mask = __DG_CELL_UNMASKED__;

    ExportImplicitMesh_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_DOM);
}

/**
 * \brief Export solution contained in a MultiFab and defined over an implicit mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const std::string & filename_root,
                const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const iMultiFab & mask,
                const int N_PHI,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_dom_Ng = matfactory.pp_std_elem.dom_Ng;
    const int cell_e = matfactory.pp_std_elem.g;
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = matfactory.pp_std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = matfactory.pp_std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = matfactory.pp_std_elem.dom_XI3g.data();
    )

    // DG
    const int phi_sp = mesh.ls_std_elem.p;
    const int sol_sp = matfactory.std_elem.p;

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

    std::string filepath, folderpath;

    int n_IBVP_fields;
    Vector<int> nodal_fields_domains;
    // ================================================================
    
    // GET INFO FROM THE IBVP CLASS ===================================
    // NODAL DATA
    IBVP.F_FIELDS_NAMES(VTK_nodal_fields_names);
    n_IBVP_fields = VTK_nodal_fields_names.size();

    for (int ph = 0; ph < N_PHI; ++ph)
    {
        VTK_nodal_fields_names.push_back("PHI_"+std::to_string(ph));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // DOMAINS OF DEFINITION FOR THE NODAL FIELDS
    IBVP.F_FIELDS_DOMAINS(nodal_fields_domains);

    // CELL DATA
    VTK_cell_fields_names = {"TAG"};

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Output.H", "Export_VTK", N_SOL, n_IBVP_fields);
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
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

    const int * nodal_fields_domains_ptr = nodal_fields_domains.data();

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
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

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
            Real PHI[__DG_MAX_N_PHI__];
            Real SOL[__DG_MAX_N_SOL__];
            Real FIELDS[__DG_MAX_N_FIELDS__];

            // LEVELSETS
            DG_PHI_space_BFX phi(phi_sp, cell_lo, cell_hi, PHI_fab, i, j, k);
            
            // SOLUTION
            DG_SOL_space_BFX sol(sol_sp, X_fab);

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

                // EVAL LEVELSETS -------------------------------------
                phi.eval(x, 0, N_PHI, PHI);
                // ----------------------------------------------------

                // VTK NODAL FIELDS -----------------------------------
                for (int f = 0; f < n_IBVP_fields; ++f)
                {
                    const int dom = nodal_fields_domains_ptr[f];
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                    int BF_i, BF_j, BF_k;
                    Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    
                    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k);
                    sol.set_support(BF_lo, BF_hi);

                    // EVAL SOLUTION
                    sol.eval(x, 0, N_SOL, SOL);

                    // EVAL IBVP FIELDS
                    IBVP.F_FIELDS(t, x, SOL, FIELDS);

                    VTK_nodal_fields_ptr[f][VTK_nodal_field_pos] = FIELDS[f];
                }

                for (int ph = 0; ph < N_PHI; ++ph)
                {
                    VTK_nodal_fields_ptr[n_IBVP_fields+ph][VTK_nodal_field_pos] = PHI[ph];
                }
                // ----------------------------------------------------

                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;
            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY ------------------------------
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                // VTK CELL FIELDS ------------------------------------
                // TAG
                VTK_cell_fields_ptr[0][VTK_cell_field_pos] = mask_fab(i,j,k);
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
    IO::MakeStepFolder(folderpath_root, n, n_steps);

    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export FV solution contained in a MultiFab and defined over an implicit mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] mask: a iMultiFab object that contains a single value for each cell.
 * \param[in] N_PHI: number of levelsets contained in mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the state vectors for the unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK_FV(const std::string & folderpath_root,
                   const int n, 
                   const int n_steps,
                   const std::string & filename_root,
                   const Real t,
                   const ImplicitMesh & mesh,
                   const iMultiFab & mask,
                   const int N_PHI,
                   const int N_SOL,
                   const MultiFab & X,
                   const IBVP_CLASS & IBVP)
{
    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

    // STANDARD ELEMENT
    const int cell_dom_Ng = std::min(mesh.ls_std_elem.dom_Ng, mesh.im_std_elem.dom_Ng);
    const int cell_e = std::min(mesh.ls_std_elem.g, mesh.im_std_elem.g);
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
#if (AMREX_SPACEDIM == 1)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedLine_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 2)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedQuad_Connectivity(cell_e);
#endif
#if (AMREX_SPACEDIM == 3)
    const Gpu::ManagedVector<int> subgrid_conn = VTK::GriddedHexahedron_Connectivity(cell_e);
#endif
    const int * subgrid_conn_ptr = subgrid_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = (mesh.ls_std_elem.g < mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI1g.data() : mesh.im_std_elem.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = (mesh.ls_std_elem.g < mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI2g.data() : mesh.im_std_elem.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = (mesh.ls_std_elem.g < mesh.im_std_elem.g) ? mesh.ls_std_elem.dom_XI3g.data() : mesh.im_std_elem.dom_XI3g.data();
    )

    // DG
    const int phi_sp = mesh.ls_std_elem.p;

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

    std::string filepath, folderpath;

    int n_IBVP_fields;
    Vector<int> nodal_fields_domains;

    MultiFab dX(X.boxarray, X.distributionMap, AMREX_SPACEDIM*N_SOL, 1);
    // ================================================================
    
    // GET INFO FROM THE IBVP CLASS ===================================
    // NODAL DATA
    IBVP.F_FIELDS_NAMES(VTK_nodal_fields_names);
    n_IBVP_fields = VTK_nodal_fields_names.size();

    for (int ph = 0; ph < N_PHI; ++ph)
    {
        VTK_nodal_fields_names.push_back("PHI_"+std::to_string(ph));
    }
    
    n_VTK_nodal_fields = VTK_nodal_fields_names.size();

    // DOMAINS OF DEFINITION FOR THE NODAL FIELDS
    IBVP.F_FIELDS_DOMAINS(nodal_fields_domains);

    // CELL DATA
    VTK_cell_fields_names = {"TAG"};

    n_VTK_cell_fields = VTK_cell_fields_names.size();
    // ================================================================

    // EVAL THE SLOPES ================================================
    {
        int N_DOM = std::set<int>(nodal_fields_domains.begin(), nodal_fields_domains.end()).size();
        IBVP.F_R_SLOPES(t, mesh, N_DOM, N_SOL, X, dX);
    }
    // ================================================================

    // MAKE SURE THE PROBLEM CAN BE HANDLED ===========================
    IsProblemManageable("AMReX_DG_Solution_Output.H", "Export_VTK_FV", N_SOL, n_IBVP_fields);
    // ================================================================

    // EVAL REQUIRED MEMORY ===========================================
    VTK_cell_conn_size = 0L;
    
    n_VTK_nodes = 0L;
    n_VTK_cells = 0L;

    // DOMAIN
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
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

    const int * nodal_fields_domains_ptr = nodal_fields_domains.data();

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
        const Box & bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real const> const & dX_fab = dX.array(mfi);
        Array4<int const> const & mask_fab = mask.array(mfi);

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
            Real PHI[__DG_MAX_N_PHI__];
            Real SOL[__DG_MAX_N_SOL__];
            Real FIELDS[__DG_MAX_N_FIELDS__];

            // LEVELSETS
            DG_PHI_space_BFX phi(phi_sp, cell_lo, cell_hi, PHI_fab, i, j, k);
            
            // SOLUTION
            DG_SOL_space_RX sol(X_fab, dX_fab);

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

                // EVAL LEVELSETS -------------------------------------
                phi.eval(x, 0, N_PHI, PHI);
                // ----------------------------------------------------

                // VTK NODAL FIELDS -----------------------------------
                for (int f = 0; f < n_IBVP_fields; ++f)
                {
                    const int dom = nodal_fields_domains_ptr[f];
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                    int BF_i, BF_j, BF_k;
                    Real xc[AMREX_SPACEDIM];
                    
                    BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);
                    AMREX_D_TERM
                    (
                        xc[0] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X1(dom));,
                        xc[1] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X2(dom));,
                        xc[2] = eInfo_fab(BF_i,BF_j,BF_k,ELM_CENTROID_X3(dom));
                    )
                    sol.set_fab_ptrs(BF_i, BF_j, BF_k);
                    sol.set_centroid(xc);

                    // EVAL SOLUTION
                    sol.eval(x, 0, N_SOL, SOL);

                    // EVAL IBVP FIELDS
                    IBVP.F_FIELDS(t, x, SOL, FIELDS);

                    VTK_nodal_fields_ptr[f][VTK_nodal_field_pos] = FIELDS[f];
                }

                for (int ph = 0; ph < N_PHI; ++ph)
                {
                    VTK_nodal_fields_ptr[n_IBVP_fields+ph][VTK_nodal_field_pos] = PHI[ph];
                }
                // ----------------------------------------------------

                VTK_nodes_pos += 3;
                VTK_nodal_field_pos += 1;
            }

            // LOOP OVER THE SUB CELLS
            for (int e = 0; e < cell_dom_Ne; ++e)
            {
                // VTK CELL CONNECTIVITY ------------------------------
                const int offset = e*VTK_cell_conn_len;
                for (int c = 0; c < VTK_cell_conn_len; ++c)
                {
                    VTK_cell_conn_ptr[VTK_cell_conn_pos+c] = subgrid_conn_ptr[offset+c]+VTK_cell_conn_offset;
                }
                // ----------------------------------------------------

                // VTK CELL OFFSET ------------------------------------
                VTK_cell_offset_ptr[VTK_cell_offset_pos] = VTK_cell_offset_ptr[VTK_cell_offset_pos-1]+VTK_cell_conn_len;
                // ----------------------------------------------------

                // VTK CELL TYPE --------------------------------------
                VTK_cell_type_ptr[VTK_cell_type_pos] = VTK_cell_id;
                // ----------------------------------------------------

                // VTK CELL FIELDS ------------------------------------
                // TAG
                VTK_cell_fields_ptr[0][VTK_cell_field_pos] = mask_fab(i,j,k);
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
    IO::MakeStepFolder(folderpath_root, n, n_steps);

    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(folderpath_root, n, n_steps, filename_root,
                                 VTK_nodal_fields_names, VTK_cell_fields_names);
    }
    // ================================================================

    // DUMP DATA FILE =================================================
    {
        filepath = IO::MakeLocalOutputFilepath(folderpath_root, n, n_steps, filename_root, "vtu");

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

/**
 * \brief Export solution contained in a MultiFab and defined over an implicit mesh to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] filename_root: root of the output filename.
 * \param[in] t: time.
 * \param[in] mesh: an ImplicitMesh object containing the information about the implicitly-defined
 *                  mesh.
 * \param[in] matfactory: a MatrixFactory object containing the operator for the implicitly-defined
 *                        mesh.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] X: a MultiFab object that contains the coefficients of the basis functions for the
 *               unknown solution fields.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const std::string & filename_root,
                const Real t,
                const ImplicitMesh & mesh,
                const MatrixFactory & matfactory,
                const int N_SOL,
                const MultiFab & X,
                const IBVP_CLASS & IBVP)
{
    const int N_PHI = 1;

    iMultiFab mask(X.boxarray, X.distributionMap, 1, X.n_grow);
    mask = __DG_CELL_UNMASKED__;

    Export_VTK(folderpath_root, n, n_steps, filename_root,
               t,
               mesh,
               matfactory,
               mask,
               N_PHI,
               N_SOL,
               X,
               IBVP);
}

/**
 * \brief Export solution contained in an AMR data structure to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] t: time.
 * \param[in] N_PHI: Number of level set functions.
 * \param[in] N_DOM: Number of domains.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const Real t,
                const int N_PHI,
                const int N_DOM,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP)
{
    const int finest_level = AMR.maxLevel();

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        std::string filename_root;
        
        if (AMR.LevelIsValid(lev))
        {
            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MatrixFactory & matfactory = *AMR.matfactories[lev];
            const iMultiFab & mask = AMR.masks[lev];
            const MultiFab & X = AMR.Xs[lev];

            filename_root = "ImplicitMesh_Lv"+std::to_string(lev)+"_";
            ExportImplicitMesh_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_DOM);
            
            filename_root = "Solution_Lv"+std::to_string(lev);
            if (matfactory.std_elem.p == 0)
            {
                Export_VTK_FV(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_SOL, X, IBVP);
            }
            else
            {
                Export_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, matfactory, mask, N_PHI, N_SOL, X, IBVP);
            }
        }
    }
}

/**
 * \brief Export solution contained in an AMR data structure to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] t: time.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename AMR_CLASS, typename IBVP_CLASS>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const Real t,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP)
{
    const int N_PHI = 1;
    const int N_DOM = 1;

    Export_VTK(folderpath_root, n, n_steps,
               t,
               N_PHI,
               N_DOM,
               N_SOL,
               AMR,
               IBVP);
}

/**
 * \brief Save solution contained in an AMR data structure.
 *
 * \param[in] n: time step index.
 * \param[in] t: time.
 * \param[in] AMR: the input AMR class.
*/
template <typename AMR_CLASS>
void WriteCheckpoint(const int n, const Real t, const AMR_CLASS & AMR)
{
    const std::string folderpath_root = AMR.inputs.checkpoint_filepath;
    const int n_steps = AMR.inputs.time.n_steps;
    const int finest_level = AMR.maxLevel();

    // STEP FOLDER
    IO::MakeStepFolder(folderpath_root, n, n_steps);

    // HEADER FILE
    {
        if (ParallelDescriptor::IOProcessor())
        {   
            const std::string header_filepath = IO::MakeGlobalOutputFilepath(folderpath_root, n, n_steps, "Checkpoint_Header", "txt");
            time_t date_and_time = time(0);
            char * date_and_time_ = ctime(&date_and_time);
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
            
            std::ofstream fp;
            fp.precision(17);
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

            fp.open(header_filepath.c_str(), std::ofstream::app);
            if(!fp.good())
            {
                FileOpenFailed(header_filepath);
            }
            
            fp << std::endl << "CHECKPOINT HEADER FILE - " << date_and_time_ << "\n";
            fp << "| Number of levels: " << finest_level+1 << std::endl;
            fp << "| time: " << t << std::endl;
            fp.close();
        }

        // HEADER FILE FOR THE MULTIFABS
        if (ParallelDescriptor::IOProcessor())
        {
            const std::string header_filepath = IO::MakeGlobalOutputFilepath(folderpath_root, n, n_steps, "Checkpoint_Header_MF", "txt");
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

            std::ofstream fp;
            fp.precision(17);
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            
            fp.open(header_filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            if(!fp.good())
            {
                FileOpenFailed(header_filepath);
            }

            fp << "Checkpoint file\n";
            fp << finest_level << "\n";
            for (int lev = 0; lev <= finest_level; ++lev)
            {
                if (AMR.LevelIsValid(lev))
                {
                    AMR.boxArray(lev).writeOn(fp);
                    fp << '\n';
                    fp << AMR.meshes[lev]->PHI.n_comp << "\n";
                    fp << AMR.meshes[lev]->PHI.n_grow << "\n";
                    fp << AMR.Xs[lev].n_comp << "\n";
                    fp << AMR.Xs[lev].n_grow << "\n";
                }
            }
            
            fp.close();
        }
    }
    
    // LEVELSET FUNCTIONS AND SOLUTIONS MULTIFABS
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        std::string filename_root;
        
        if (AMR.LevelIsValid(lev))
        {
            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MultiFab & PHI = mesh.PHI;
            const MultiFab & X = AMR.Xs[lev];

            const std::string PHI_filepath = IO::MakeGlobalOutputFilepath(folderpath_root, n, n_steps, "Checkpoint_Lv"+std::to_string(lev)+"_PHI", "");
            const std::string X_filepath = IO::MakeGlobalOutputFilepath(folderpath_root, n, n_steps, "Checkpoint_Lv"+std::to_string(lev)+"_X", "");

            VisMF::Write(PHI, PHI_filepath);
            VisMF::Write(X, X_filepath);
        }
    }
}

/**
 * \brief Export solution contained in an AMR data structure to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] t: time.
 * \param[in] N_PHI: Number of level set functions.
 * \param[in] N_DOM: Number of domains.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename AMR_CLASS, typename IBVP_CLASS, typename SLOPE_LIMITER>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const Real t,
                const int N_PHI,
                const int N_DOM,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP,
                const SLOPE_LIMITER & LIMITER)
{
    const int finest_level = AMR.maxLevel();

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        std::string filename_root;
        
        if (AMR.LevelIsValid(lev))
        {
            const ImplicitMesh & mesh = *AMR.meshes[lev];
            const MatrixFactory & matfactory = *AMR.matfactories[lev];
            const iMultiFab & mask = AMR.masks[lev];
            const MultiFab & X = AMR.Xs[lev];

            filename_root = "ImplicitMesh_Lv"+std::to_string(lev)+"_";
            ExportImplicitMesh_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_DOM);
            
            filename_root = "Solution_Lv"+std::to_string(lev);
            Export_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, matfactory, mask, N_PHI, N_SOL, X, IBVP);
        }
    }

    {
        const int lev = AMR.maxLevel()+1;
        std::string filename_root;

        const ImplicitMesh & mesh = LIMITER.mesh;
        const MatrixFactory & matfactory = LIMITER.matfactory;
        const iMultiFab & mask = LIMITER.mask;
        const MultiFab & X = LIMITER.X;

        filename_root = "ImplicitMesh_Lv"+std::to_string(lev)+"_";
        ExportImplicitMesh_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, mask, N_PHI, N_DOM);
        
        filename_root = "Solution_Lv"+std::to_string(lev);
        Export_VTK(folderpath_root, n, n_steps, filename_root, t, mesh, matfactory, mask, N_PHI, N_SOL, X, IBVP);
    }
}

/**
 * \brief Export solution contained in an AMR data structure to VTK format.
 *
 * \param[in] folderpath_root: root of the output folder path where the file will be dumped; it will
 *                             be suffixed with an index denoting the time step.
 * \param[in] n: time step index.
 * \param[in] n_steps: maximum number of time steps.
 * \param[in] t: time.
 * \param[in] N_SOL: Number of unknown solution fields.
 * \param[in] AMR: the input AMR class.
 * \param[in] IBVP: a class that must have two methods with the following prototypes:
 *
 *                  void F_FIELDS_NAMES(Vector<std::string> & fields_names) const;
 *                  void F_FIELDS(const Real t, const Real * x, const Real * PHI, const Real * SOL, Real * F) const;
*/
template <typename AMR_CLASS, typename IBVP_CLASS, typename SLOPE_LIMITER>
void Export_VTK(const std::string & folderpath_root,
                const int n, 
                const int n_steps,
                const Real t,
                const int N_SOL,
                const AMR_CLASS & AMR,
                const IBVP_CLASS & IBVP,
                const SLOPE_LIMITER & LIMITER)
{
    const int N_PHI = 1;
    const int N_DOM = 1;

    Export_VTK(folderpath_root, n, n_steps,
               t,
               N_PHI,
               N_DOM,
               N_SOL,
               AMR,
               IBVP,
               LIMITER);
}

} // namespace DG
} // namespace amrex

#endif