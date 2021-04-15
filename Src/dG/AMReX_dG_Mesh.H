//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Mesh.H
 * \brief Contains constants, data structures and functions for meshes with embedded geometries.
*/

#ifndef AMREX_DG_MESH_H_
#define AMREX_DG_MESH_H_

#include <string>

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <AMReX_VTK_Utils.H>

#include <AMReX_dG_Constants.H>
#include <AMReX_dG_StandardElements.H>

#define __DG_STD_ELEM_N_SPACE_BOUNDARIES__ (2*AMREX_SPACEDIM)

// CELL TYPE
#define __DG_CELL_TYPE_N_COMP_PER_DOM__ 1
#define __DG_CELL_TYPE_UNDEFINED__ -2
#define __DG_CELL_TYPE_EMPTY__ -1
#define __DG_CELL_TYPE_ENTIRE__ 1
#define __DG_CELL_TYPE_CUT__ 5

// CELL BOUNDARY TYPE
#define __DG_CELL_BOU_TYPE_N_COMP_PER_DOM__ 1
#define __DG_CELL_BOU_TYPE_UNDEFINED__ -2
#define __DG_CELL_BOU_TYPE_EMPTY__ -1
#define __DG_CELL_BOU_TYPE_ENTIRE__ 1
#define __DG_CELL_BOU_TYPE_CUT__ 5

// QUADRATURE INFO
#define __DG_CELL_DOM_QUAD_N_COMP_PER_DOM__ 2
#define __DG_CELL_BOU_QUAD_N_COMP_PER_DOM__ 2
#define __DG_CELL_INT_BOU_QUAD_N_COMP_PER_DOM__ 2

// MASK
#define __DG_CELL_NON_MASKED__ 0
#define __DG_CELL_MASKED__ 1
#define __DG_LEVEL_BOUNDARY__ 2

namespace amrex
{
namespace dG
{
// AUXILIARY FUNCTIONS ################################################
// ALGOIM =============================================================
template<typename F>
AlgoimQuadRule AlgoimQuadGen(const F & phi, const Real * lo, const Real * hi, const int dim, const int side, const int qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real, AMREX_SPACEDIM>(lo, hi), dim, side, qo);
}
// ====================================================================


// CELL TYPE ==========================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_TYPE(const int dom)
{
    return (__DG_CELL_TYPE_N_COMP_PER_DOM__*dom);
}
// ====================================================================


// CELL BOU TYPE ======================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_BOU_TYPE(const int dom)
{
    return (__DG_CELL_BOU_TYPE_N_COMP_PER_DOM__*dom);
}
// ====================================================================


// QUADRATURE INFO ====================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_DOM_QUAD_NQ(const int dom)
{
    return (__DG_CELL_DOM_QUAD_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_DOM_QUAD_POS(const int dom)
{
    return (__DG_CELL_DOM_QUAD_N_COMP_PER_DOM__*dom+1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_BOU_QUAD_NQ(const int dom)
{
    return (__DG_CELL_BOU_QUAD_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_BOU_QUAD_POS(const int dom)
{
    return (__DG_CELL_BOU_QUAD_N_COMP_PER_DOM__*dom+1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_INT_BOU_QUAD_NQ(const int dom)
{
    return (__DG_CELL_INT_BOU_QUAD_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_INT_BOU_QUAD_POS(const int dom)
{
    return (__DG_CELL_INT_BOU_QUAD_N_COMP_PER_DOM__*dom+1);
}
// ====================================================================


// NEIGHBORS INFORMATION ==============================================
/**
 * \brief Return the indices of the cell that shares the b-th boundary of the current cell.
 *
 * \param[in] (i,j,k): indices of the current cell.
 * \param[in] b: boundary of the current cell.
 * \param[out] (nbr_i,nbr_j,nbr_k): indices of the adjacent neighboring cell.
 * \param[out] nbr_b: boundary as seen by the neighboring cell.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void NBR_CELL(const int i, const int j, const int k, const int b,
              int & nbr_i, int & nbr_j, int & nbr_k, int & nbr_b)
{
    if (b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__)
    {
        // Neighboring cell id
        // b = 0 <-> nbr_i = i-1, nbr_j = j,   nbr_k = k
        // b = 1 <-> nbr_i = i+1, nbr_j = j,   nbr_k = k
        // b = 2 <-> nbr_i = i,   nbr_j = j-1, nbr_k = k
        // b = 3 <-> nbr_i = i,   nbr_j = j+1, nbr_k = k
        // b = 4 <-> nbr_i = i,   nbr_j = j,   nbr_k = k-1
        // b = 5 <-> nbr_i = i,   nbr_j = j,   nbr_k = k+1
        nbr_i = i+(-1+2*(b%2))*((b/2)-1)*((b/2)-2)/2;
        nbr_j = j+(-1+2*(b%2))*(2-(b/2))*(b/2);
        nbr_k = k+(-1+2*(b%2))*((b/2)-1)*(b/2)/2;

        // Boundary id as seen by the neighboring cell
        // b = 0 <-> nbr_b = 1
        // b = 1 <-> nbr_b = 0
        // b = 2 <-> nbr_b = 3
        // b = 3 <-> nbr_b = 2
        // b = 4 <-> nbr_b = 5
        // b = 5 <-> nbr_b = 4
        nbr_b = b+1-2*(b%2);
    }
    else
    {
#if (AMREX_SPACEDIM == 2)
        // Neighboring cell id
        // b = 4+0 <-> nbr_i = i-1, nbr_j = j-1, nbr_k = k
        // b = 4+1 <-> nbr_i = i+1, nbr_j = j-1, nbr_k = k
        // b = 4+2 <-> nbr_i = i-1, nbr_j = j+1, nbr_k = k
        // b = 4+3 <-> nbr_i = i+1, nbr_j = j+1, nbr_k = k
        nbr_i = i+2*(b%2)-1;
        nbr_j = j+2*((b-4)/2)-1;
        nbr_k = k;
#endif
#if (AMREX_SPACEDIM == 3)
        switch (b)
        {
            case 6:
                nbr_i = i;
                nbr_j = j-1;
                nbr_k = k-1;
                break;
            case 7:
                nbr_i = i;
                nbr_j = j+1;
                nbr_k = k-1;
                break;
            case 8:
                nbr_i = i-1;
                nbr_j = j;
                nbr_k = k-1;
                break;
            case 9:
                nbr_i = i+1;
                nbr_j = j;
                nbr_k = k-1;
                break;
                
            case 10:
                nbr_i = i-1;
                nbr_j = j-1;
                nbr_k = k;
                break;
            case 11:
                nbr_i = i+1;
                nbr_j = j-1;
                nbr_k = k;
                break;
            case 12:
                nbr_i = i-1;
                nbr_j = j+1;
                nbr_k = k;
                break;
            case 13:
                nbr_i = i+1;
                nbr_j = j+1;
                nbr_k = k;
                break;
                
            case 14:
                nbr_i = i;
                nbr_j = j-1;
                nbr_k = k+1;
                break;
            case 15:
                nbr_i = i;
                nbr_j = j+1;
                nbr_k = k+1;
                break;
            case 16:
                nbr_i = i-1;
                nbr_j = j;
                nbr_k = k+1;
                break;
            case 17:
                nbr_i = i+1;
                nbr_j = j;
                nbr_k = k+1;
                break;

            case 18:
                nbr_i = i-1;
                nbr_j = j-1;
                nbr_k = k-1;
                break;
            case 19:
                nbr_i = i+1;
                nbr_j = j-1;
                nbr_k = k-1;
                break;
            case 20:
                nbr_i = i-1;
                nbr_j = j+1;
                nbr_k = k-1;
                break;
            case 21:
                nbr_i = i+1;
                nbr_j = j+1;
                nbr_k = k-1;
                break;

            case 22:
                nbr_i = i-1;
                nbr_j = j-1;
                nbr_k = k+1;
                break;
            case 23:
                nbr_i = i+1;
                nbr_j = j-1;
                nbr_k = k+1;
                break;
            case 24:
                nbr_i = i-1;
                nbr_j = j+1;
                nbr_k = k+1;
                break;
            case 25:
                nbr_i = i+1;
                nbr_j = j+1;
                nbr_k = k+1;
                break;
        }
#endif

        // Boundary id as seen by the neighboring cell
        nbr_b = -1;
    }
}

/**
 * \brief Return the indices of the neighbor cells sharing the face (fi,fj,fk).
 *
 * \param[in] (fi,fj,fk): indices of the current cell.
 * \param[in] dir: direction of the unit normal of the face.
 * \param[out] (min_i,min_j,min_k): indices of the "minus" cell.
 * \param[out] (pls_i,pls_j,pls_k): indices of the "plus" cell.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FACE_TO_NBRS(const int fi, const int fj, const int fk, const int dir,
                  int & min_i, int & min_j, int & min_k,
                  int & pls_i, int & pls_j, int & pls_k)
{
    // Given an edge identified by the face-centered indexing tuple
    // (fi,fj,fk), the neighboring elements that share that face are
    // referred to as the "plus" and the "minus" elements. The "plus"
    // element is the one that lies on the same side of the positive
    // unit normal, whereas the "minus" element is the one that lies on
    // the same side of the negative unit normal as shown in the sketch
    // below:
    //
    //                        |
    //       -----------------+-----------------
    //                        | un = {+1,0,0}
    //                        |--->
    //                        |
    //          (elm^-)   (fi,fj,fk)  (elm^+)
    //                        |
    //                    <---|
    //          un = {-1,0,0} |
    //       -----------------+-----------------
    //                        |

    // dir = 0 <-> min_i = fi-1, min_j = fj,   min_k = fk
    // dir = 1 <-> min_i = fi,   min_j = fj-1, min_k = fk
    // dir = 2 <-> min_i = fi,   min_j = fj,   min_k = fk-1
    // dir = 0 <-> pls_i = fi,   pls_j = fj,   pls_k = fk
    // dir = 1 <-> pls_i = fi,   pls_j = fj,   pls_k = fk
    // dir = 2 <-> pls_i = fi,   pls_j = fj,   pls_k = fk
    min_i = fi-(dir-1)*(dir-2)/2;
    min_j = fj-(2-dir)*dir;
    min_k = fk-(dir-1)*dir/2;
    pls_i = fi;
    pls_j = fj;
    pls_k = fk;
}
// ====================================================================


// MASK ===============================================================
/**
 * \brief Given the cell mask value, return true if the cell is masked, false otherwise.
 *
 * \param[in] val: cell mask value.
 *
 * \return true if the cell is masked, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_MASKED(const int val)
{
    return (val == __DG_CELL_MASKED__);
}

/**
 * \brief Given the cell mask value, return true if the cell is not masked, false otherwise.
 *
 * \param[in] val: cell mask value.
 *
 * \return true if the cell is not masked, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_NOT_MASKED(const int val)
{
    return (val == __DG_CELL_NON_MASKED__);
}
// ====================================================================
// ####################################################################



// STRUCTURED MESH CLASS ##############################################
/**
 * \brief Structured mesh parameters class.
 *
*/
struct StructuredMeshParameters
{
    // DATA MEMBERS ===================================================
    bool embedded_geometry_is_active;
    std::string embedded_geometry_defined_by;
    bool box_pruning_is_active;
    
    int quadrature_order_regular_elements, quadrature_order_cut_elements;
    // ================================================================


    // CONSTRUCTOR ====================================================
    StructuredMeshParameters()
    :
    embedded_geometry_is_active{false},
    embedded_geometry_defined_by{""},
    box_pruning_is_active{false},
    quadrature_order_regular_elements{-1},
    quadrature_order_cut_elements{-1}
    {}
    // ================================================================
};

/**
 * \brief Structured mesh class.
 *
*/
struct StructuredMesh
{
    // DATA MEMBERS ===================================================
    static const int dom_data_n_grow = 1;
    static const int bou_data_n_grow = 0;

    StructuredMeshParameters params;

    // BOXARRAY AND DISTRIBUTION MAPPING
    BoxArray pruned_ba;
    DistributionMapping pruned_dm;

    // CELL TYPE
    sMultiFab cell_type;
    Array<sMultiFab, AMREX_SPACEDIM> cell_bou_type;

    // QUADRATURE INFO
    lMultiFab cell_dom_quad_info;
    Array<lMultiFab, AMREX_SPACEDIM> cell_bou_quad_info;
    lMultiFab cell_int_bou_quad_info;

    Gpu::HostVector<Real> cell_dom_quad_host_mem;
    Array<Gpu::HostVector<Real>, AMREX_SPACEDIM> cell_bou_quad_host_mem;
    Gpu::HostVector<Real> cell_int_bou_quad_host_mem;

    Gpu::DeviceVector<Real> cell_dom_quad_dev_mem;
    Array<Gpu::DeviceVector<Real>, AMREX_SPACEDIM> cell_bou_quad_dev_mem;
    Gpu::DeviceVector<Real> cell_int_bou_quad_dev_mem;
    // ================================================================


    // CONSTRUCTOR ====================================================
    // ================================================================


    // INITIALIZATION =================================================
    void read_input_file(const ParmParse & pp);
    // ================================================================


    // READERS ========================================================
    bool uses_embedded_geometries() const;
    bool uses_levelset() const;
    bool uses_box_pruning() const;
    // ================================================================


    // INITIALIZE THE MULTIFABS =======================================
    /**
     * \brief Initialize the multifabs that will contain the mesh information.
     *
     * \param[in] ba: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void init_multifabs(const BoxArray & ba, const DistributionMapping & dm, const IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();
        
        // COMPONENTS OF THE CELL/CELL BOU TYPE MULTIFABS
        // Type
        const int cell_type_n_comp = n_domains*__DG_CELL_TYPE_N_COMP_PER_DOM__;
        const int cell_bou_type_n_comp = n_domains*__DG_CELL_BOU_TYPE_N_COMP_PER_DOM__;
        
        // COMPONENTS OF THE QUADRATURE MULTIFABS
        // Number of quadrature points
        // Position in memory
        const int cell_dom_quad_info_n_comp = n_domains*__DG_CELL_DOM_QUAD_N_COMP_PER_DOM__;
        const int cell_bou_quad_info_n_comp = n_domains*__DG_CELL_BOU_QUAD_N_COMP_PER_DOM__;
        const int cell_int_bou_quad_info_n_comp = n_domains*__DG_CELL_INT_BOU_QUAD_N_COMP_PER_DOM__;

        // FACE-CENTERED BOX ARRAYS
#if (AMREX_SPACEDIM == 1)
        const BoxArray fc_ba[1] = {convert(ba, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(ba, IntVect(1,0)),
                                   convert(ba, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(ba, IntVect(1,0,0)),
                                   convert(ba, IntVect(0,1,0)),
                                   convert(ba, IntVect(0,0,1))};
#endif
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        // CELL TYPE
        this->cell_type.define(ba, dm, cell_dom_quad_info_n_comp, this->dom_data_n_grow);
        this->cell_type = __DG_CELL_TYPE_UNDEFINED__;
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->cell_bou_type[dim].define(fc_ba[dim], dm, cell_bou_quad_info_n_comp, this->bou_data_n_grow);
            this->cell_bou_type[dim] = __DG_CELL_BOU_TYPE_UNDEFINED__;
        }
        
        // QUADRATURE INFO
        this->cell_dom_quad_info.define(ba, dm, cell_dom_quad_info_n_comp, this->dom_data_n_grow);
        this->cell_dom_quad_info = -1L;

        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->cell_bou_quad_info[dim].define(fc_ba[dim], dm, cell_bou_quad_info_n_comp, this->bou_data_n_grow);
            this->cell_bou_quad_info[dim] = -1L;
        }

        this->cell_int_bou_quad_info.define(ba, dm, cell_int_bou_quad_info_n_comp, this->dom_data_n_grow);
        this->cell_int_bou_quad_info = -1L;
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE FROM SCRATCH ==============================================
    /**
     * \brief Make the data structures that will store the mesh info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] ba: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void make_from_scratch_by_levelset(const Real t, const Geometry & geom, const BoxArray & ba, const DistributionMapping & dm, IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
        const Real atol = 1.0e-12;

        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // QUADRATURE
        const int qo_reg = this->params.quadrature_order_regular_elements+1;
        const int qo_cut = this->params.quadrature_order_cut_elements+1;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        StandardRectangle<AMREX_SPACEDIM> std_elm;
        long cell_dom_quad_mem_pos;
        Array<long, AMREX_SPACEDIM> cell_bou_quad_mem_pos;
        long cell_int_bou_quad_mem_pos;
        // ------------------------------------------------------------

        // PRUNE BOX ARRAY AND DISTRIBUTION MAPPING (IF REQUESTED) ----
        if (this->uses_box_pruning())
        {
            BoxList old_bl(ba);
            BoxList new_bl;

            for (const Box & bx : old_bl)
            {
                const Box grown_bx = grow(bx, this->dom_data_n_grow);
                const Dim3 lo = lbound(grown_bx);
                const Dim3 hi = ubound(grown_bx);
                bool bx_is_empty;

                bx_is_empty = true;
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    ibvp.levelset.set_domain(dom);

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
                        const Real entire_cell_volume = AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);

                        // LOCAL VARIABLES
                        Real cell_volume, volume_fraction;

                        // ALGOIM QUADRATURE RULE
                        const int qo = 2;
                        AlgoimQuadRule quad = AlgoimQuadGen(ibvp.levelset, cell_lo, cell_hi, -1, -1, qo);

                        cell_volume = quad([](const auto & /*x*/) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;
                        if (volume_fraction > atol)
                        {
                            bx_is_empty = false;
                        }
                    }
                }

                if (!bx_is_empty)
                {
                    new_bl.push_back(bx);
                }
            }
            
            // SHARE THE INFO
            AllGatherBoxes(new_bl.data());

            // NEW BOXARRAY AND DISTRIBUTION MAPPING
            this->pruned_ba = BoxArray(new_bl);
            this->pruned_dm.define(this->pruned_ba);
        }
        else
        {
            this->pruned_ba = ba;
            this->pruned_dm = dm;
        }
        // ------------------------------------------------------------

        // INIT THE MULTIFAB ------------------------------------------
        this->init_multifabs(this->pruned_ba, this->pruned_dm, ibvp);
        // ------------------------------------------------------------

        // INITIALIZE THE QUADRATURE FOR THE STANDARD ELEMENT ---------
        std_elm.set_quadrature(&dx[0], qo_reg);
        // ------------------------------------------------------------

        // FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO -----
        cell_dom_quad_mem_pos = 0L;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            cell_bou_quad_mem_pos[dir] = 0L;
        }
        cell_int_bou_quad_mem_pos = 0L;

        // DOMAIN QUADRATURE
        {
            // QUADRATURE INFO
            const int dom_Nq = std_elm.dom_Nq;

            // Resize memory
            this->cell_dom_quad_host_mem.resize((AMREX_SPACEDIM+1)*dom_Nq);

            // Store data
            for (int q = 0; q < dom_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->cell_dom_quad_host_mem[(AMREX_SPACEDIM+1)*q+0] = std_elm.dom_XI1q[q];,
                    this->cell_dom_quad_host_mem[(AMREX_SPACEDIM+1)*q+1] = std_elm.dom_XI2q[q];,
                    this->cell_dom_quad_host_mem[(AMREX_SPACEDIM+1)*q+2] = std_elm.dom_XI3q[q];
                )
                this->cell_dom_quad_host_mem[(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = std_elm.dom_Wq[q];
            }

            // Update memory (nodes coordinates + weights)
            cell_dom_quad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
        }

        // BOUNDARY QUADRATURE
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            // QUADRATURE INFO
            const int bou_Nq = std_elm.bou_Nq;

            // Resize memory
            this->cell_bou_quad_host_mem[dir].resize((AMREX_SPACEDIM+1)*bou_Nq);

            // Store data
            for (int q = 0; q < bou_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->cell_bou_quad_host_mem[dir][(AMREX_SPACEDIM+1)*q+0] = std_elm.bou_XI1q[dir][q];,
                    this->cell_bou_quad_host_mem[dir][(AMREX_SPACEDIM+1)*q+1] = std_elm.bou_XI2q[dir][q];,
                    this->cell_bou_quad_host_mem[dir][(AMREX_SPACEDIM+1)*q+2] = std_elm.bou_XI3q[dir][q];
                )
                this->cell_bou_quad_host_mem[dir][(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = std_elm.bou_Wq[dir][q];
            }

            // Update memory (nodes coordinates + weights)
            cell_bou_quad_mem_pos[dir] += (AMREX_SPACEDIM+1)*bou_Nq;
        }
        // ------------------------------------------------------------

        // CELL-CENTERED QUADRATURE INFO: DOMAIN + INTERNAL BOUNDARIES 
        // NOTE: We must loop over the fabbox because the quadrature
        //       info will not be shared among the processors.
        // ------------------------------------------------------------
        const int std_elm_dom_Nq = std_elm.dom_Nq;

        const Real entire_cell_volume = AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);
        
        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short> const & cell_type_fab = this->cell_type.array(mfi);
            Array4<long> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
            Array4<long> const & cell_int_bou_quad_info_fab = this->cell_int_bou_quad_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            {
                ibvp.levelset.set_domain(dom);

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
                    bool cell_is_cut;
                    Real cell_volume, volume_fraction;

                    // DOMAIN QUADRATURE
                    {
                        // Algoim data structure
                        AlgoimQuadRule quad = AlgoimQuadGen(ibvp.levelset, cell_lo, cell_hi, -1, -1, qo_cut);

                        // Compute the volume fraction
                        cell_volume = quad([](const auto & /*x*/) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;

                        // EMPTY CELL
                        if (volume_fraction < atol)
                        {
                            cell_is_cut = false;

                            // CELL TYPE
                            cell_type_fab(i,j,k,CELL_TYPE(dom)) = __DG_CELL_TYPE_EMPTY__;

                            // QUADRATURE INFO
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom)) = 0L;
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom)) = -1L;
                        }
                        // ENTIRE CELL
                        else if (std::abs(volume_fraction-1.0) < atol)
                        {
                            cell_is_cut = false;

                            // CELL TYPE
                            cell_type_fab(i,j,k,CELL_TYPE(dom)) = __DG_CELL_TYPE_ENTIRE__;

                            // QUADRATURE INFO
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom)) = std_elm_dom_Nq;
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom)) = 0L;
                        }
                        // CUT CELL
                        else
                        {
                            cell_is_cut = true;

                            // CELL TYPE
                            cell_type_fab(i,j,k,CELL_TYPE(dom)) = __DG_CELL_TYPE_CUT__;

                            // QUADRATURE INFO
                            const int dom_Nq = quad.nodes.size();

                            // Store memory offset
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom)) = dom_Nq;
                            cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom)) = cell_dom_quad_mem_pos;

                            // Expand memory
                            this->cell_dom_quad_host_mem.resize(cell_dom_quad_mem_pos+(AMREX_SPACEDIM+1)*dom_Nq);

                            // Store data: remember that the abscissae
                            // are referred to a reference system
                            // centered at cell_lo.
                            for (int q = 0; q < dom_Nq; ++q)
                            {
                                const unsigned int pos = cell_dom_quad_mem_pos+(AMREX_SPACEDIM+1)*q;

                                AMREX_D_TERM
                                (
                                    this->cell_dom_quad_host_mem[pos+0] = (quad.nodes[q].x[0]-cell_lo[0]);,
                                    this->cell_dom_quad_host_mem[pos+1] = (quad.nodes[q].x[1]-cell_lo[1]);,
                                    this->cell_dom_quad_host_mem[pos+2] = (quad.nodes[q].x[2]-cell_lo[2]);
                                )
                                this->cell_dom_quad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                            }

                            // Update memory (nodes coordinates + weights)
                            cell_dom_quad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
                        }
                    }

                    // INTERNAL BOUNDARY QUADRATURE
                    if (cell_is_cut)
                    {
                        // Algoim data structure
                        AlgoimQuadRule quad = AlgoimQuadGen(ibvp.levelset, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo_cut);

                        // QUADRATURE INFO
                        const int bou_Nq = quad.nodes.size();
                        GpuArray<Real, AMREX_SPACEDIM> un;
                        Real tmp;

                        // Store memory offset
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom)) = bou_Nq;
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom)) = cell_int_bou_quad_mem_pos;

                        // Expand memory
                        this->cell_int_bou_quad_host_mem.resize(cell_int_bou_quad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq);

                        // Store data
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long pos = cell_int_bou_quad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;

                            AMREX_D_TERM
                            (
                                this->cell_int_bou_quad_host_mem[pos+0] = quad.nodes[q].x[0];,
                                this->cell_int_bou_quad_host_mem[pos+1] = quad.nodes[q].x[1];,
                                this->cell_int_bou_quad_host_mem[pos+2] = quad.nodes[q].x[2];
                            )
                            this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;

                            un = ibvp.levelset.grad(quad.nodes[q].x);
                            tmp = 1.0/std::sqrt(AMREX_D_PICK(un[0]*un[0], un[0]*un[0]+un[1]*un[1], un[0]*un[0]+un[1]*un[1]+un[2]*un[2]));
                            AMREX_D_TERM
                            (
                                un[0] *= tmp;,
                                un[1] *= tmp;,
                                un[2] *= tmp;
                            )

                            AMREX_D_TERM
                            (
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+0] = un[0];,
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+1] = un[1];,
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+2] = un[2];
                            )
                        }

                        // Update memory (nodes coordinates + weights +
                        // unit normal components)
                        cell_int_bou_quad_mem_pos += (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq;
                    }
                }
            }
        }
        // ------------------------------------------------------------

        // FACE-CENTERED QUADRATURE INFO: CELL BOUNDARIES -------------
        const int std_elm_bou_Nq = std_elm.bou_Nq;

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {

#if (AMREX_SPACEDIM == 2)
            const int ii = 1-dir;
#endif
#if (AMREX_SPACEDIM == 3)
            const int ii = ((2-dir)*(1-dir))/2;
            const int jj = (4+dir-dir*dir)/2;
#endif
            const Real entire_boundary_surface = AMREX_D_PICK(1.0, dx[ii], dx[ii]*dx[jj]);

            for (MFIter mfi(this->cell_bou_type[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<short> const & cell_bou_type_fab = this->cell_bou_type[dir].array(mfi);
                Array4<long> const & cell_bou_quad_info_fab = this->cell_bou_quad_info[dir].array(mfi);

                for (int dom = 0; dom < n_domains; ++dom)
                {
                    ibvp.levelset.set_domain(dom);

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
                        Real boundary_surface, surface_fraction;

                        // BOUNDARY QUADRATURE
                        {
                            // Algoim data structure
                            AlgoimQuadRule quad = AlgoimQuadGen(ibvp.levelset, cell_lo, cell_hi, dir, 0, qo_cut);

                            // Compute the surface fraction
                            boundary_surface = quad([](const auto & /*x*/) {return 1.0;});
                            surface_fraction = boundary_surface/entire_boundary_surface;

                            // EMPTY CELL BOUNDARY
                            if (surface_fraction < atol)
                            {
                                // CELL BOUNDARY TYPE
                                cell_bou_type_fab(i,j,k,CELL_BOU_TYPE(dom)) = __DG_CELL_BOU_TYPE_EMPTY__;

                                // QUADRATURE INFO
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_NQ(dom)) = 0L;
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_POS(dom)) = -1L;
                            }
                            // ENTIRE CELL BOUNDARY
                            else if (surface_fraction > (1.0-atol))
                            {
                                // CELL BOUNDARY TYPE
                                cell_bou_type_fab(i,j,k,CELL_BOU_TYPE(dom)) = __DG_CELL_BOU_TYPE_ENTIRE__;

                                // QUADRATURE INFO
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_NQ(dom)) = std_elm_bou_Nq;
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_POS(dom)) = 0L;
                            }
                            // CUT CELL BOUNDARY
                            else
                            {
                                // CELL BOUNDARY TYPE
                                cell_bou_type_fab(i,j,k,CELL_BOU_TYPE(dom)) = __DG_CELL_BOU_TYPE_CUT__;

                                // QUADRATURE INFO
                                const int bou_Nq = quad.nodes.size();

                                // Store memory offset
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_NQ(dom)) = bou_Nq;
                                cell_bou_quad_info_fab(i,j,k,CELL_BOU_QUAD_POS(dom)) = cell_bou_quad_mem_pos[dir];

                                // Expand memory
                                this->cell_bou_quad_host_mem[dir].resize(cell_bou_quad_mem_pos[dir]+(AMREX_SPACEDIM+1)*bou_Nq);

                                // Store data: remember that the
                                // abscissae are referred to a
                                // reference system centered at
                                // cell_lo.
                                for (int q = 0; q < bou_Nq; ++q)
                                {
                                    const long pos = cell_bou_quad_mem_pos[dir]+(AMREX_SPACEDIM+1)*q;

                                    AMREX_D_TERM
                                    (
                                        this->cell_bou_quad_host_mem[dir][pos+0] = (quad.nodes[q].x[0]-cell_lo[0]);,
                                        this->cell_bou_quad_host_mem[dir][pos+1] = (quad.nodes[q].x[1]-cell_lo[1]);,
                                        this->cell_bou_quad_host_mem[dir][pos+2] = (quad.nodes[q].x[2]-cell_lo[2]);
                                    )
                                    this->cell_bou_quad_host_mem[dir][pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                                }

                                // Update memory (nodes coordinates + weights)
                                cell_bou_quad_mem_pos[dir] += (AMREX_SPACEDIM+1)*bou_Nq;
                            }
                        }
                    }
                }
            }
        }
        // ------------------------------------------------------------

        // COPY MEMORY TO THE DEVICE ----------------------------------
#ifdef AMREX_USE_GPU
        this->cell_dom_quad_dev_mem.resize(this->cell_dom_quad_host_mem.size());
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            this->cell_bou_quad_dev_mem[dir].resize(this->cell_bou_quad_host_mem[dir].size());
        }
        this->cell_int_bou_quad_dev_mem.resize(this->cell_int_bou_quad_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->cell_dom_quad_host_mem.begin(), this->cell_dom_quad_host_mem.end(), this->cell_dom_quad_dev_mem.begin());
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            Gpu::copy(Gpu::hostToDevice, this->cell_bou_quad_host_mem[dir].begin(), this->cell_bou_quad_host_mem[dir].end(), this->cell_bou_quad_dev_mem[dir].begin());
        }
        Gpu::copy(Gpu::hostToDevice, this->cell_int_bou_quad_host_mem.begin(), this->cell_int_bou_quad_host_mem.end(), this->cell_int_bou_quad_dev_mem.begin());
#endif
        // ------------------------------------------------------------

    }
    
    /**
     * \brief Make the data structures that will store the mesh info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] ba: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void make_from_scratch(const Real t, const Geometry & geom, const BoxArray & ba, const DistributionMapping & dm, IBVP & ibvp)
    {
        // EVAL THE QUADRATURE RULES ----------------------------------
        if (this->uses_embedded_geometries())
        {
            if (this->uses_levelset())
            {
                this->make_from_scratch_by_levelset(t, geom, ba, dm, ibvp);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: AMReX_dG_Mesh.H - StructuredMesh::make_from_scratch\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+this->params.embedded_geometry_defined_by+".\n";
                Abort(msg);
            }
        }
        else
        {
            Abort("To be implemented!");
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // COMPUTE GEOMETRICAL PROPERTIES =================================
    void eval_volumes_and_surfaces(const Geometry & geom,
                                   const int n_domains,
                                   const iMultiFab & mask,
                                   Real * computed_volume,
                                   Real * computed_volume_div,
                                   Real * computed_surface) const;
    // ================================================================


    // INPUT/OUTPUT ===================================================
    /**
    * \brief Export the mesh to VTK format.
    *
    * \param[in] level_folderpath: path where the .pvtu files will be written.
    * \param[in] level_step_folderpath: path where the .vtu files will be written.
    * \param[in] step_string: a zero-padded string containing the step number.
    * \param[in] filename: name of the output file.
    * \param[in] geom: amrex geometry object.
    * \param[in] mask: a iMultiFab object that contains a single value for each cell.
    * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_dom_data_to_VTK(const std::string & level_folderpath,
                                const std::string & level_step_folderpath,
                                const std::string & step_string,
                                const std::string & filename,
                                const Geometry & geom,
                                const iMultiFab & mask,
                                IBVP & ibvp) const
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

        AMREX_D_TERM
        (
            Real const * dom_XI1g_ptr = std_elm.dom_XI1g.data();,
            Real const * dom_XI2g_ptr = std_elm.dom_XI2g.data();,
            Real const * dom_XI3g_ptr = std_elm.dom_XI3g.data();
        )
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

        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
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

        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & cell_type_fab = this->cell_type.array(mfi);
            Array4<long const> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
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
    * \brief Export the mesh to VTK format.
    *
    * \param[in] level_folderpath: path where the .pvtu files will be written.
    * \param[in] level_step_folderpath: path where the .vtu files will be written.
    * \param[in] step_string: a zero-padded string containing the step number.
    * \param[in] filename: name of the output file.
    * \param[in] geom: amrex geometry object.
    * \param[in] mask: a iMultiFab object that contains a single value for each cell.
    * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_bou_data_to_VTK(const std::string & level_folderpath,
                                const std::string & level_step_folderpath,
                                const std::string & step_string,
                                const std::string & filename,
                                const Geometry & geom,
                                const iMultiFab & mask,
                                IBVP & ibvp) const
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
            for (MFIter mfi(this->cell_bou_type[dir]); mfi.isValid(); ++mfi)
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

            for (MFIter mfi(this->cell_bou_type[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<short const> const & cell_bou_type_fab = this->cell_bou_type[dir].array(mfi);
                Array4<long const> const & cell_bou_quad_info_fab = this->cell_bou_quad_info[dir].array(mfi);
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
    * \param[in] mask: a iMultiFab object that contains a single value for each cell.
    * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void export_to_VTK(const std::string & level_folderpath,
                       const std::string & level_step_folderpath,
                       const std::string & step_string,
                       const std::string & filename_root,
                       const Geometry & geom,
                       const iMultiFab & mask,
                       IBVP & ibvp) const
    {
        const std::string dom_data_filename = filename_root+"_dom_data";
        const std::string bou_data_filename = filename_root+"_bou_data";

        this->export_dom_data_to_VTK(level_folderpath, level_step_folderpath, step_string, dom_data_filename,
                                     geom, mask, ibvp);
        this->export_bou_data_to_VTK(level_folderpath, level_step_folderpath, step_string, bou_data_filename,
                                     geom, mask, ibvp);
    }
    // ================================================================

};
// ####################################################################


} // namespace dG
} // namespace amrex

#endif