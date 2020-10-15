//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_ImplicitMesh.H
 * \brief Contains constants, data structures and functions to manage the implicitly-defined mesh.
*/

#include <AMReX_iMultiFab.H>

#include <AMReX_DG_InputReader.H>

#ifndef AMREX_DG_IMPLICIT_MESH_H_
#define AMREX_DG_IMPLICIT_MESH_H_

namespace amrex
{
namespace DG
{

// AUXILIARY FUNCTIONS ################################################
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

/**
 * \brief Return the face indices corresponding to the b-th boundary of the current cell.
 *
 * \param[in] (i,j,k): indices of the current cell.
 * \param[in] b: boundary of the current cell.
 * \param[out] (fi,fj,fk): indices of the face.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void GRID_FACE(const int i, const int j, const int k, const int b,
               int & fi, int & fj, int & fk)
{
    // Grid face id
    // b = 0 <-> fi = i,   fj = j,   fk = k
    // b = 1 <-> fi = i+1, fj = j,   fk = k
    // b = 2 <-> fi = i,   fj = j,   fk = k
    // b = 3 <-> fi = i,   fj = j+1, fk = k
    // b = 4 <-> fi = i,   fj = j,   fk = k
    // b = 5 <-> fi = i,   fj = j,   fk = k+1
    fi = i+(b%2)*((b/2)-1)*((b/2)-2)/2;
    fj = j+(b%2)*(2-(b/2))*(b/2);
    fk = k+(b%2)*((b/2)-1)*(b/2)/2;
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
void FACE2NBRS(const int fi, const int fj, const int fk, const int dir,
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

/**
 * \brief Get indices of the cell providing support for the basis functions over the cell (i,j,k).
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell; remember that if the cell (i,j,k) is small, then the support
 *                   of the basis functions is that of the cells that extends into it.
 * \param[out] (BF_i,BF_j,BF_k): indices of the cell that provides the support for the current cell.
 * \param[out] BF_lo: pointer to memory that will contain the lower end of the basis functions'
 *                    support.
 * \param[out] BF_hi: pointer to memory that will contain the higher end of the basis functions'
 *                    support.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BF_CELL(const int i, const int j, const int k, const int etype,
             int & BF_i, int & BF_j, int & BF_k)
{
    const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);

    if (elm_is_small)
    {
        const int b = etype/10;
        int nbr_b;
        NBR_CELL(i, j, k, b, BF_i, BF_j, BF_k, nbr_b);
    }
    else
    {
        BF_i = i;
        BF_j = j;
        BF_k = k;
    }
}
/**
 * \brief Get indices and support of the basis functions over the cell (i,j,k).
 *
 * \param[in] prob_lo: pointer to memory containing the lower end of the problem.
 * \param[in] dx: pointer to memory containing the grid's cell size.
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell; remember that if the cell (i,j,k) is small, then the support
 *                   of the basis functions is that of the cells that extends into it.
 * \param[out] (BF_i,BF_j,BF_k): indices of the cell that provides the support for the current cell.
 * \param[out] BF_lo: pointer to memory that will contain the lower end of the basis functions'
 *                    support.
 * \param[out] BF_hi: pointer to memory that will contain the higher end of the basis functions'
 *                    support.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BF_CELL(const Real * prob_lo, const Real * dx,
             const int i, const int j, const int k, const int etype,
             int & BF_i, int & BF_j, int & BF_k,
             Real * BF_lo, Real * BF_hi)
{
    BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

    AMREX_D_TERM
    (
        BF_lo[0] = prob_lo[0]+BF_i*dx[0];
        BF_hi[0] = prob_lo[0]+(BF_i+1)*dx[0];,
        BF_lo[1] = prob_lo[1]+BF_j*dx[1];
        BF_hi[1] = prob_lo[1]+(BF_j+1)*dx[1];,
        BF_lo[2] = prob_lo[2]+BF_k*dx[2];
        BF_hi[2] = prob_lo[2]+(BF_k+1)*dx[2];
    )
}
/**
 * \brief Get support of the basis functions for the cell (i,j,k).
 *
 * \param[in] prob_lo: pointer to memory containing the lower end of the problem.
 * \param[in] dx: pointer to memory containing the grid's cell size.
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell; remember that if the cell (i,j,k) is small, then the support
 *                   of the basis functions is that of the cells that extends into it.
 * \param[out] BF_lo: pointer to memory that will contain the lower end of the basis functions'
 *                    support.
 * \param[out] BF_hi: pointer to memory that will contain the higher end of the basis functions'
 *                    support.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BF_CELL(const Real * prob_lo, const Real * dx,
             const int i, const int j, const int k, const int etype,
             Real * BF_lo, Real * BF_hi)
{
    int BF_i, BF_j, BF_k;
    BF_CELL(prob_lo, dx, i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
}

// FOR ALGOIM
template<typename F>
iQuadRule iQuadGen(const F & phi, const Real * lo, const Real * hi, const int & dim, const int & side, const int & qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real,AMREX_SPACEDIM>(lo, hi), dim, side, qo);
}

// IMPLICIT MESH ELEMENTS' INFO
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_TYPE(const int dom)
{
    return (__DG_ELM_TYPE_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_TYPE(const int dom)
{
    return (__DG_ELM_BOU_TYPE_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_VOLUME_FRACTION(const int dom)
{
    return (__DG_ELM_INFO_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_VOLUME(const int dom)
{
    return (__DG_ELM_INFO_N_COMP_PER_DOM__*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X1(const int dom)
{
    return (__DG_ELM_INFO_N_COMP_PER_DOM__*dom+1+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X2(const int dom)
{
    return (__DG_ELM_INFO_N_COMP_PER_DOM__*dom+1+2);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X3(const int dom)
{
    return (__DG_ELM_INFO_N_COMP_PER_DOM__*dom+1+3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_DOM_QUAD_NQ(const int dom)
{
    return (__DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_DOM_QUAD_POS(const int dom)
{
    return (__DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_MINUS_QUAD_NQ(const int dom)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_MINUS_QUAD_POS(const int dom)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_PLUS_QUAD_NQ(const int dom)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_PLUS_QUAD_POS(const int dom)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_QUAD_NQ(const int dom)
{
    return (__DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_QUAD_POS(const int dom)
{
    return (__DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*dom);
}
// ####################################################################



// ####################################################################
// IMPLICIT MESH CLASS ################################################
// ####################################################################
/**
 * \brief Implicitly-defined mesh class.
 *
*/
struct ImplicitMesh
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;
    static const int bou_data_ngr = 0;
    
    // GRID
    Geometry geom;
    const Real abs_tol = 1.0e-12;
    Real rel_tol;

    BoxArray cc_ba;
    Array<BoxArray, AMREX_SPACEDIM> fc_ba;
    DistributionMapping dm;

    StandardRectangle<AMREX_SPACEDIM> std_elem;

    // LEVEL SETS
    MultiFab PHI;

    // IMPLICIT MESH INFO AND DATA STRUCTURES
    int space_q_im;
    Real volume_fraction_threshold;

    shortMultiFab eType;
    Array<shortMultiFab, AMREX_SPACEDIM> eBouType;

    MultiFab eInfo;

    iMultiFab eDomQuad_Nq;
    longMultiFab eDomQuad_pos;
    Array<iMultiFab, AMREX_SPACEDIM> eBouQuad_Nq;
    Array<longMultiFab, AMREX_SPACEDIM> eBouQuad_pos;
    iMultiFab eIntBouQuad_Nq;
    longMultiFab eIntBouQuad_pos;

    iMultiFab eColor;
    
    int n_cut_elms, n_cut_elm_bou, n_entire_extended_elms;
    long mem_for_eDomQuad, mem_for_eBouQuad, mem_for_eIntBouQuad;

    Gpu::HostVector<Real> eDomQuad_host_mem;
    Gpu::HostVector<Real> eBouQuad_host_mem;
    Gpu::HostVector<Real> eIntBouQuad_host_mem;
    
    Gpu::DeviceVector<Real> eDomQuad_device_mem;
    Gpu::DeviceVector<Real> eBouQuad_device_mem;
    Gpu::DeviceVector<Real> eIntBouQuad_device_mem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    ImplicitMesh()
    {}

    ImplicitMesh(const InputReader & inputs)
    {
        this->define(inputs);
    }
    // ################################################################

    // DESTRUCTOR #####################################################
    ~ImplicitMesh()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void define(const InputReader & inputs)
    {
        // VARIABLES ==================================================
        RealBox rbx;
        Box ibx;
        // ============================================================
        
        // GEOMETRY ===================================================
        rbx.setLo(inputs.space.lo.data());
        rbx.setHi(inputs.space.hi.data());
        
        AMREX_D_TERM
        (
            ibx.setSmall(0, 0);,
            ibx.setSmall(1, 0);,
            ibx.setSmall(2, 0);
        )
        ibx.setBig(inputs.grid.n_cells-1);

        this->geom.define(ibx, &rbx, inputs.space.coord_sys, inputs.space.is_periodic.data());
        this->rel_tol = (this->abs_tol)*(*std::min_element(this->geom.CellSize(),this->geom.CellSize()+AMREX_SPACEDIM));
        // ============================================================

        // BOX ARRAYS AND DISTRIBUTION MAPPING ========================
        this->cc_ba.define(ibx);
        this->cc_ba.maxSize(inputs.grid.max_box_size);

#if (AMREX_SPACEDIM == 1)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1));
#endif
#if (AMREX_SPACEDIM == 2)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1));
#endif
#if (AMREX_SPACEDIM == 3)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1,0));
        this->fc_ba[2] = convert(this->cc_ba, IntVect(0,0,1));
#endif

        this->dm.define(this->cc_ba);
        // ============================================================

        // STANDARD ELEMENT ===========================================
        this->std_elem.define(this->geom.CellSize(), inputs.grid.space_p, inputs.grid.space_q);
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->space_q_im = inputs.grid.space_q_im;

        this->volume_fraction_threshold = inputs.grid.VFT;

        this->n_cut_elms = 0;
        this->n_cut_elm_bou = 0;
        this->n_entire_extended_elms = 0;

        this->mem_for_eDomQuad = 0L;
        this->mem_for_eBouQuad = 0L;
        this->mem_for_eIntBouQuad = 0L;
        // ============================================================
    }
    // ################################################################

    // MAKE THE IMPLICIT MESH #########################################
    /**
     * \brief Eval and store the implicitly-defined mesh using N_PHI level sets and N_DOM domains.
     *
     * \param[in] N_PHI: number of level sets that will be projected.
     * \param[in] N_DOM: number of domains that will be considered.
     * \param[in] IBVP: a class that must have a method with the following prototype:
     *
     *                  void F_DOM2PHI(const int dom, int * phi_info) const;
     *
     * \param[in] im_flag: implicit mesh flag. If set to false, all cells are considered as entire
     *                     cells.
     */
    template <typename IBVP_CLASS>
    void Make(const int N_PHI, const int N_DOM, const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        // PROFILING ------------------------------------------------------
        BL_PROFILE("Make(const int, const int, const IBVP_CLASS &, const bool)");
        // ----------------------------------------------------------------

#if (AMREX_SPACEDIM == 1)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_DG_ImplicitMesh.H - ImplicitMesh::Make\n";
            msg += "| The generation of the implicit mesh must be thoroughly checked to handle AMREX_SPACEDIM = 1.\n";
            Abort(msg);
        }
#endif

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->geom.ProbLoArray();
        const Real atol = this->abs_tol;

        // STANDARD ELEMENT
        const int p = this->std_elem.p;
        const int std_elem_dom_Nq = this->std_elem.dom_Nq;
        const int std_elem_bou_Nq = this->std_elem.bou_Nq;
    
        // MULTIFAB INFO
        const int eType_n_comp = __DG_ELM_TYPE_N_COMP_PER_DOM__*N_DOM;
        const int eBouType_n_comp = __DG_ELM_BOU_TYPE_N_COMP_PER_DOM__*N_DOM;

        const int eInfo_n_comp = __DG_ELM_INFO_N_COMP_PER_DOM__*N_DOM;

        const int eDomQuad_Nq_n_comp = __DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eDomQuad_pos_n_comp = __DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eBouQuad_Nq_n_comp = __DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eBouQuad_pos_n_comp = __DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eIntBouQuad_Nq_n_comp = __DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eIntBouQuad_pos_n_comp = __DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;

        // IMPLICIT MESH VOLUME FRACTION THRESHOLD
        const Real vft = this->volume_fraction_threshold;

        // IMPLICIT MESH QUADRATURE ORDER
        const int qo = this->space_q_im;
        // ============================================================
        
        // VARIABLES ==================================================
        long eDomQuad_mem_pos, eBouQuad_mem_pos, eIntBouQuad_mem_pos;
        // ============================================================

        // INITIALIZATION =============================================
        this->eType.define(this->cc_ba, this->dm, eType_n_comp, this->dom_data_ngr);
        this->eType = __DG_ELM_TYPE_UNDEFINED__;

        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->eBouType[dim].define(this->fc_ba[dim], this->dm, eBouType_n_comp, this->bou_data_ngr);
            this->eBouType[dim] = __DG_ELM_BOU_TYPE_UNDEFINED__;
        }
        
        this->eInfo.define(this->cc_ba, this->dm, eInfo_n_comp, this->dom_data_ngr);
        this->eInfo = 0.0;

        this->eDomQuad_Nq.define(this->cc_ba, this->dm, eDomQuad_Nq_n_comp, this->dom_data_ngr);
        this->eDomQuad_Nq = -1;
        this->eDomQuad_pos.define(this->cc_ba, this->dm, eDomQuad_pos_n_comp, this->dom_data_ngr);
        this->eDomQuad_pos = -1L;

        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->eBouQuad_Nq[dim].define(this->fc_ba[dim], this->dm, eBouQuad_Nq_n_comp, this->bou_data_ngr);
            this->eBouQuad_Nq[dim] = -1;
            this->eBouQuad_pos[dim].define(this->fc_ba[dim], this->dm, eBouQuad_pos_n_comp, this->bou_data_ngr);
            this->eBouQuad_pos[dim] = -1L;
        }

        this->eIntBouQuad_Nq.define(this->cc_ba, this->dm, eIntBouQuad_Nq_n_comp, this->dom_data_ngr);
        this->eIntBouQuad_Nq = -1;
        this->eIntBouQuad_pos.define(this->cc_ba, this->dm, eIntBouQuad_pos_n_comp, this->dom_data_ngr);
        this->eIntBouQuad_pos = -1L;

        this->n_cut_elms = 0;
        this->n_cut_elm_bou = 0;
        this->n_entire_extended_elms = 0;

        eDomQuad_mem_pos = 0L;
        eBouQuad_mem_pos = 0L;
        eIntBouQuad_mem_pos = 0L;
        // ============================================================

        // THE FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO =
        // DOMAIN QUADRATURE POINTS
        {
            // QUADRATURE INFO
            const int dom_Nq = this->std_elem.dom_Nq;

            // Resize memory
            this->eDomQuad_host_mem.resize((AMREX_SPACEDIM+1)*dom_Nq);

            // Store data
            for (int q = 0; q < dom_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+0] = this->std_elem.dom_XI1q[q];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+1] = this->std_elem.dom_XI2q[q];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+2] = this->std_elem.dom_XI3q[q];
                )
                this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = this->std_elem.dom_Wq[q];
            }

            // Update memory (nodes coordinates + weights)
            eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
        }

        // BOUNDARY QUADRATURE POINTS
        {
            // QUADRATURE INFO
            const int bou_Nq = this->std_elem.bou_Nq;

            // Resize memory
            this->eBouQuad_host_mem.resize((AMREX_SPACEDIM+1)*bou_Nq*__DG_STD_ELEM_N_SPACE_BOUNDARIES__);

            // Store data
            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            for (int q = 0; q < bou_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+0] = this->std_elem.bou_XI1q[bou_Nq*b+q];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+1] = this->std_elem.bou_XI2q[bou_Nq*b+q];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+2] = this->std_elem.bou_XI3q[bou_Nq*b+q];
                )
                this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = this->std_elem.bou_Wq[bou_Nq*b+q];
            }

            // Update memory (nodes coordinates + weights)
            eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq*__DG_STD_ELEM_N_SPACE_BOUNDARIES__;
        }
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: DOMAIN (CELL-CENTERED) DATA ==
        // NOTE: We are looping over the fabbox because the implicit
        //       mesh information regarding the quadrature will not be
        //       shared among the processors.
        // ============================================================
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<Real const> const & PHI_fab = this->PHI.array(mfi);

            Array4<short> const & eType_fab = this->eType.array(mfi);
            Array4<Real> const & eInfo_fab = this->eInfo.array(mfi);
            Array4<int> const & eDomQuad_Nq_fab = this->eDomQuad_Nq.array(mfi);
            Array4<long> const & eDomQuad_pos_fab = this->eDomQuad_pos.array(mfi);
            Array4<int> const & eIntBouQuad_Nq_fab = this->eIntBouQuad_Nq.array(mfi);
            Array4<long> const & eIntBouQuad_pos_fab = this->eIntBouQuad_pos.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                  prob_lo[1]+(j+0.5)*dx[1],
                                                                  prob_lo[2]+(k+0.5)*dx[2])};
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                                   prob_lo[1]+(j+1)*dx[1],
                                                                   prob_lo[2]+(k+1)*dx[2])};
                const Real entire_cell_volume = AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);

                // LOCAL VARIABLES
                bool cell_is_empty, cell_is_entire, cell_is_cut;
                Real cell_volume, volume_fraction;
                int phi_info[2];

                // WHICH LEVELSET DEFINES THE dom-th DOMAIN
                IBVP.F_DOM2PHI(dom, phi_info);

                // LEVELSETS
                algoim_DG_phi_space_BFX phi(p, cell_lo, cell_hi, PHI_fab, i, j, k);
                phi.set_u(phi_info[0]);
                phi.set_sign(phi_info[1]);

                // ELEMENTS' QUADRATURE POINTS
                {
                    // Algoim data structure
                    iQuadRule quad;
                    if (im_flag)
                    {
                        quad = iQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                        // Compute the volume fraction
                        cell_volume = quad([](const auto & /*x*/) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;
                    }
                    else
                    {
                        volume_fraction = 1.0;
                    }

                    // EMPTY ELEMENTS ---------------------------------
                    if (volume_fraction < atol)
                    {
                        cell_is_empty = true;
                        cell_is_entire = false;
                        cell_is_cut = false;

                        // ELEMENT TYPE
                        eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_EMPTY__;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom)) = 0.0;
                        eInfo_fab(i,j,k,ELM_VOLUME(dom)) = cell_volume;

                        // QUADRATURE INFO
                        eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom)) = 0;
                        eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom)) = -1L;
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENTS --------------------------------
                    else if (volume_fraction > (1.0-atol))
                    {
                        cell_is_empty = false;
                        cell_is_entire = true;
                        cell_is_cut = false;

                        // ELEMENT INFO
                        eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_ENTIRE__;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom)) = 1.0;
                        eInfo_fab(i,j,k,ELM_VOLUME(dom)) = cell_volume;

                        // QUADRATURE INFO
                        eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom)) = std_elem_dom_Nq;
                        eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom)) = 0L;
                    }
                    // ------------------------------------------------
                    // PARTIAL ELEMENTS -------------------------------
                    else
                    {
                        cell_is_empty = false;
                        cell_is_entire = false;
                        cell_is_cut = true;

                        this->n_cut_elms += 1;
                            
                        // ELEMENT TYPE
                        eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_PARTIAL__;
                        
                        // ELEMENT INFO
                        eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom)) = volume_fraction;
                        eInfo_fab(i,j,k,ELM_VOLUME(dom)) = cell_volume;

                        // QUADRATURE INFO
                        const int dom_Nq = quad.nodes.size();
                        
                        // Store memory offset
                        eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom)) = dom_Nq;
                        eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom)) = eDomQuad_mem_pos;

                        // Expand memory
                        this->eDomQuad_host_mem.resize(eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*dom_Nq);

                        // Store data: remember that the quad abscissae
                        // are referred to a reference system centered
                        // at cell_lo.
                        for (int q = 0; q < dom_Nq; ++q)
                        {
                            const long pos = eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*q;

                            AMREX_D_TERM
                            (
                                this->eDomQuad_host_mem[pos+0] = (quad.nodes[q].x[0]-cell_lo[0]);,
                                this->eDomQuad_host_mem[pos+1] = (quad.nodes[q].x[1]-cell_lo[1]);,
                                this->eDomQuad_host_mem[pos+2] = (quad.nodes[q].x[2]-cell_lo[2]);
                            )
                            this->eDomQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                        }

                        // Update memory (nodes coordinates + weights)
                        eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
                    }
                    // ------------------------------------------------
                }

                // ELEMENTS' CENTROIDS
                {
                    // EMPTY ELEMENTS ---------------------------------
                    if (cell_is_empty)
                    {
                        // STORE CENTROID (ALL ZERO)
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) = 0.0;,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) = 0.0;,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) = 0.0;
                        )
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENTS --------------------------------
                    else if (cell_is_entire)
                    {
                        // STORE CENTROID
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) += cell_m[0];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) += cell_m[1];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) += cell_m[2];
                        )
                    }
                    // ------------------------------------------------
                    // PARTIAL ELEMENTS -------------------------------
                    else
                    {
                        // PARAMETERS
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                        const long pos0 = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));

                        // VARIABLES
                        Real x[AMREX_SPACEDIM], w, S[AMREX_SPACEDIM];

                        // EVAL THE STATIC MOMENTS
                        AMREX_D_TERM
                        (
                            S[0] = 0.0;,
                            S[1] = 0.0;,
                            S[2] = 0.0;
                        )

                        for (int q = 0; q < dom_Nq; ++q)
                        {
                            const long pos = pos0+(AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = this->eDomQuad_host_mem[pos+0]+cell_lo[0];,
                                x[1] = this->eDomQuad_host_mem[pos+1]+cell_lo[1];,
                                x[2] = this->eDomQuad_host_mem[pos+2]+cell_lo[2];
                            )
                            w = this->eDomQuad_host_mem[pos+AMREX_SPACEDIM];
                            AMREX_D_TERM
                            (
                                S[0] += x[0]*w;,
                                S[1] += x[1]*w;,
                                S[2] += x[2]*w;
                            )
                        }

                        // EVAL AND STORE CENTROID
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) += S[0]/cell_volume;,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) += S[1]/cell_volume;,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) += S[2]/cell_volume;
                        )
                    }
                    // ------------------------------------------------
                }

                // INTERNAL BOUNDARY (IF ANY)
                if (cell_is_cut)
                {
                    iQuadRule quad = iQuadGen(phi, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo);

                    // QUADRATURE INFO
                    const int bou_Nq = quad.nodes.size();
                    GpuArray<Real, AMREX_SPACEDIM> un;
                    Real tmp;

                    // Store memory offset
                    eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom)) = bou_Nq;
                    eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom)) = eIntBouQuad_mem_pos;

                    // Expand memory
                    this->eIntBouQuad_host_mem.resize(eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq);

                    // Store data
                    for (int q = 0; q < bou_Nq; ++q)
                    {
                        const long pos = eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;

                        AMREX_D_TERM
                        (
                            this->eIntBouQuad_host_mem[pos+0] = quad.nodes[q].x[0];,
                            this->eIntBouQuad_host_mem[pos+1] = quad.nodes[q].x[1];,
                            this->eIntBouQuad_host_mem[pos+2] = quad.nodes[q].x[2];
                        )
                        this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;

                        un = phi.grad(quad.nodes[q].x);
                        tmp = 1.0/std::sqrt(AMREX_D_PICK(un[0]*un[0], un[0]*un[0]+un[1]*un[1], un[0]*un[0]+un[1]*un[1]+un[2]*un[2]));
                        AMREX_D_TERM
                        (
                            un[0] *= tmp;,
                            un[1] *= tmp;,
                            un[2] *= tmp;
                        )

                        AMREX_D_TERM
                        (
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+0] = un[0];,
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+1] = un[1];,
                            this->eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+2] = un[2];
                        )
                    }

                    // Update memory (nodes coordinates + weights +
                    // unit normal components)
                    eIntBouQuad_mem_pos += (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq;
                }
            }
        }
        // ============================================================

        // STORE MEMORY INFO ==========================================
        this->mem_for_eDomQuad = eDomQuad_mem_pos;
        this->mem_for_eIntBouQuad = eIntBouQuad_mem_pos;
        // ============================================================

        // STORE THE IMPLICIT MESH INFO: BOUNDARY (FACE-CENTERED) DATA 
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            const int b = 2*dir;
            const int flag[AMREX_SPACEDIM] = {AMREX_D_DECL((int) (dir == 0), (int) (dir == 1), (int) (dir == 2))};
            const Real delta[AMREX_SPACEDIM] = {AMREX_D_DECL(dx[0]*flag[0], dx[1]*flag[1], dx[2]*flag[2])};

            for (MFIter mfi(this->eBouQuad_Nq[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<Real const> const & PHI_fab = this->PHI.array(mfi);

                Array4<short> const & eBouType_fab = this->eBouType[dir].array(mfi);
                Array4<int> const & eBouQuad_Nq_fab = this->eBouQuad_Nq[dir].array(mfi);
                Array4<long> const & eBouQuad_pos_fab = this->eBouQuad_pos[dir].array(mfi);

                for (int dom = 0; dom < N_DOM; ++dom)
                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    // ------------------------------------------------
                    // NOTE: Is it ok to use the indices defined from
                    //       the face-centered box to access the data
                    //       stored in the cell-centered PHI_fab?
                    // ------------------------------------------------

                    // LOCAL PARAMETERS
                    const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                      prob_lo[1]+(j+0.5)*dx[1],
                                                                      prob_lo[2]+(k+0.5)*dx[2])};
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                                       prob_lo[1]+(j+1)*dx[1],
                                                                       prob_lo[2]+(k+1)*dx[2])};
#if (AMREX_SPACEDIM == 2)
                    const int ii = 1-dir;
#endif
#if (AMREX_SPACEDIM == 3)
                    const int ii = ((2-dir)*(1-dir))/2;
                    const int jj = (4+dir-dir*dir)/2;
#endif
                    const Real entire_boundary_surface = AMREX_D_PICK(1.0, dx[ii], dx[ii]*dx[jj]);

                    // LOCAL VARIABLES
                    Real boundary_surface, surface_fraction;
                    int phi_info[2];

                    // WHICH LEVELSET DEFINES THE dom-th DOMAIN
                    IBVP.F_DOM2PHI(dom, phi_info);

                    // LEVELSETS
                    algoim_DG_phi_space_BFX phi(p, cell_lo, cell_hi, PHI_fab, i, j, k);
                    phi.set_u(phi_info[0]);
                    phi.set_sign(phi_info[1]);

                    // BOUNDARY QUADRATURE POINTS
                    {
                        // Algoim data structure
                        iQuadRule quad;
                        if (im_flag)
                        {
                            quad = iQuadGen(phi, cell_lo, cell_hi, b/2, b%2, qo);

                            // Compute the surface fraction
                            boundary_surface = quad([](const auto & /*x*/) {return 1.0;});
                            surface_fraction = boundary_surface/entire_boundary_surface;
                        }
                        else
                        {
                            surface_fraction = 1.0;
                        }

                        // EMPTY BOUNDARY -----------------------------
                        if (surface_fraction < atol)
                        {
                            // ELEMENT BOUNDARY TYPE
                            eBouType_fab(i,j,k,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_EMPTY__;

                            // QUADRATURE INFO
                            // BOUNDARY^-
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_MINUS_QUAD_NQ(dom)) = 0;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_MINUS_QUAD_POS(dom)) = -1L;

                            // BOUNDARY^+
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_PLUS_QUAD_NQ(dom)) = 0;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_PLUS_QUAD_POS(dom)) = -1L;
                        }
                        // --------------------------------------------
                        // ENTIRE BOUNDARY ----------------------------
                        else if (surface_fraction > (1.0-atol))
                        {
                            // ELEMENT BOUNDARY TYPE
                            eBouType_fab(i,j,k,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_ENTIRE__;

                            // QUADRATURE INFO
                            const int bou_Nq = std_elem_bou_Nq;
                            
                            // BOUNDARY^-
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_MINUS_QUAD_NQ(dom)) = bou_Nq;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_MINUS_QUAD_POS(dom)) = (AMREX_SPACEDIM+1)*bou_Nq*(b+1);

                            // BOUNDARY^+
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_PLUS_QUAD_NQ(dom)) = bou_Nq;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_PLUS_QUAD_POS(dom)) = (AMREX_SPACEDIM+1)*bou_Nq*b;
                        }
                        // --------------------------------------------
                        // PARTIAL BOUNDARY ---------------------------
                        else
                        {
                            // ELEMENT BOUNDARY TYPE
                            eBouType_fab(i,j,k,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL__;

                            // QUADRATURE INFO
                            const int bou_Nq = quad.nodes.size();

                            // BOUNDARY^-
                            // Store memory offset
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_MINUS_QUAD_NQ(dom)) = bou_Nq;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_MINUS_QUAD_POS(dom)) = eBouQuad_mem_pos;

                            // Expand memory
                            this->eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                            // Store data: remember that the quad
                            // abscissae are referred to a reference
                            // system centered at cell_lo.
                            for (int q = 0; q < bou_Nq; ++q)
                            {
                                const long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*q;

                                AMREX_D_TERM
                                (
                                    this->eBouQuad_host_mem[pos+0] = (quad.nodes[q].x[0]-cell_lo[0]+delta[0]);,
                                    this->eBouQuad_host_mem[pos+1] = (quad.nodes[q].x[1]-cell_lo[1]+delta[1]);,
                                    this->eBouQuad_host_mem[pos+2] = (quad.nodes[q].x[2]-cell_lo[2]+delta[2]);
                                )
                                this->eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                            }

                            // Update memory (nodes coordinates + weights)
                            eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;

                            // BOUNDARY^+
                            // Store memory offset
                            eBouQuad_Nq_fab(i,j,k,ELM_BOU_PLUS_QUAD_NQ(dom)) = bou_Nq;
                            eBouQuad_pos_fab(i,j,k,ELM_BOU_PLUS_QUAD_POS(dom)) = eBouQuad_mem_pos;

                            // Expand memory
                            this->eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                            // Store data: remember that the quad
                            // abscissae are referred to a reference
                            // system centered at cell_lo.
                            for (int q = 0; q < bou_Nq; ++q)
                            {
                                const long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*q;

                                AMREX_D_TERM
                                (
                                    this->eBouQuad_host_mem[pos+0] = (quad.nodes[q].x[0]-cell_m[0]);,
                                    this->eBouQuad_host_mem[pos+1] = (quad.nodes[q].x[1]-cell_m[1]);,
                                    this->eBouQuad_host_mem[pos+2] = (quad.nodes[q].x[2]-cell_m[2]);
                                )
                                this->eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                            }

                            // Update memory (nodes coordinates + weights)
                            eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;
                        }
                        // --------------------------------------------
                    }
                }
            }
        }
        // ============================================================

        // STORE MEMORY INFO ==========================================
        this->mem_for_eBouQuad = eBouQuad_mem_pos;
        // ============================================================

        // SEARCH FOR SMALL ELEMENTS AND SETUP THE MERGING INFO =======
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short> const & eType_fab = this->eType.array(mfi);
            
            Array4<Real const> const & eInfo_fab = this->eInfo.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const bool elm_is_partial = (eType_fab(i,j,k,ELM_TYPE(dom)) == __DG_ELM_TYPE_PARTIAL__);
                
                const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b;
                Real nbr_vf, best_nbr_vf;
                bool elm_has_been_merged, nbr_is_valid;

                // CONDITIONS FOR SMALL ELEMENTS
                if (elm_is_partial && (vf < vft))
                {
                    elm_has_been_merged = false;
                    best_nbr_vf = vft;

                    // Search among the neighbors. We currently limit the
                    // search to the neighboring cells that share a 
                    // boundary with the current cell.
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        // Neighboring cell id and boundary
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        nbr_is_valid = true;
                        nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME_FRACTION(dom));

                        if ((nbr_vf > best_nbr_vf) && nbr_is_valid)
                        {
                            eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                            elm_has_been_merged = true;
                            best_nbr_vf = nbr_vf;
                        }
                    }

                    if (elm_has_been_merged)
                    {
                        // I would like to mark an element as extended
                        // without falling into any raise condition. The
                        // conflict can be raised because an element can be
                        // extended to multiple small elements.
                    }
                    else
                    {
                        printf("\n");
                        printf("ERROR: AMReX_DG_ImplicitMesh.H - ImplicitMesh::Make\n");
                        printf("| Could not find a neighboring cell suitable for merging:\n");
                        printf("| dom: %d\n", dom);
                        printf("| i,j,k: %d,%d,%d\n", i, j, k);
                        printf("| Volume fraction: %f\n", vf);
                        for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                        {
                            // Neighboring cell id and boundary
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME_FRACTION(dom));
                            printf("| Volume fraction (nbr = %d): %f\n", b, nbr_vf);
                        }
                        printf("\n");
                        Abort("");
                    }
                }
                // ELSE THE ELEMENT IS LARGE
                else if (elm_is_partial && (vf >= vft))
                {
                    eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_LARGE__;
                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());
        // ============================================================

        // UPDATE THE CENTROIDS OF THE EXTENDED ELEMENTS ==============
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short> const & eType_fab = this->eType.array(mfi);

            Array4<Real> const & eInfo_fab = this->eInfo.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool elm_is_valid = elm_is_entire || elm_is_large;

                // LOCAL VARIABLES
                short nbr_etype;
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                bool nbr_is_small;

                // Cycle over the boundaries, check whether the valid
                // element is also extended and mark them. At the same
                // time, add the small element contribution to the
                // centroid.
                if (elm_is_valid)
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        // MARK THE ELEMENT AS EXTENDED
                        eType_fab(i,j,k,ELM_TYPE(dom)) = etype+10;

                        // UPDATE ITS CENTROID
                        const Real small_V = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME(dom));
                        const Real V = eInfo_fab(i,j,k,ELM_VOLUME(dom));
                        
                        Real small_S[AMREX_SPACEDIM], S[AMREX_SPACEDIM];

                        AMREX_D_TERM
                        (
                            small_S[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom))*small_V;,
                            small_S[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom))*small_V;,
                            small_S[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom))*small_V;
                        )
                        AMREX_D_TERM
                        (
                            S[0] = eInfo_fab(i,j,k,ELM_CENTROID_X1(dom))*V;,
                            S[1] = eInfo_fab(i,j,k,ELM_CENTROID_X2(dom))*V;,
                            S[2] = eInfo_fab(i,j,k,ELM_CENTROID_X3(dom))*V;
                        )
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) = (S[0]+small_S[0])/(V+small_V);,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) = (S[1]+small_S[1])/(V+small_V);,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) = (S[2]+small_S[2])/(V+small_V);
                        )
                    }
                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());
        this->eInfo.FillBoundary(this->geom.periodicity());
        // ============================================================

        // COUNT THE NUMBER OF ENTIRE EXTENDED ELEMENTS ===============
        this->n_entire_extended_elms = 0;

        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & eType_fab = this->eType.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool elm_is_valid = elm_is_entire || elm_is_large;
                const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

                if (elm_is_entire && elm_is_extended) this->n_entire_extended_elms += 1;
            }
        }
        // ============================================================

        // PERFORM THE MESH PRUNING ===================================
        // REMOVE ALL BOXES WHOSE ELEMENTS ARE EMPTY
        // (THE ELEMENTS MUST BE EMPTY FOR EACH AND EVERY DOMAIN)
        // ============================================================
        if (true)
        {
            BoxList bl;

            for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.fabbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<short const> const & eType_fab = this->eType.array(mfi);

                bool bx_is_empty;

                bx_is_empty = true;
                for (int dom = 0; dom < N_DOM; ++dom)
                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

                    if (elm_is_not_empty)
                    {
                        bx_is_empty = false;
                    }
                }

                if (!bx_is_empty)
                {
                    bl.push_back(mfi.validbox());
                }
            }

            // SHARE THE INFO
            AllGatherBoxes(bl.data());

            // CREATE THE NEW BOX ARRAYS AND DISTRIBUTION MAPPING
            BoxArray new_cc_ba;
            Array<BoxArray, AMREX_SPACEDIM> new_fc_ba;
            DistributionMapping new_dm;

            new_cc_ba = BoxArray(bl);
    #if (AMREX_SPACEDIM == 1)
            new_fc_ba[0] = convert(new_cc_ba, IntVect(1));
    #endif
    #if (AMREX_SPACEDIM == 2)
            new_fc_ba[0] = convert(new_cc_ba, IntVect(1,0));
            new_fc_ba[1] = convert(new_cc_ba, IntVect(0,1));
    #endif
    #if (AMREX_SPACEDIM == 3)
            new_fc_ba[0] = convert(new_cc_ba, IntVect(1,0,0));
            new_fc_ba[1] = convert(new_cc_ba, IntVect(0,1,0));
            new_fc_ba[2] = convert(new_cc_ba, IntVect(0,0,1));
    #endif
            new_dm.define(new_cc_ba);

            // UPDATE ALL THE MULTIFABS RELATED TO THE IMPLICIT MESH
            // LEVEL SETS
            {
                const int ngr = this->dom_data_ngr;
                const int n_comp = (this->std_elem.Np)*N_PHI;

                MultiFab real_mf;
                
                real_mf.define(new_cc_ba, new_dm, n_comp, ngr);
                real_mf.ParallelCopy(this->PHI, 0, 0, n_comp, ngr, ngr, this->geom.periodicity());
                this->PHI.define(new_cc_ba, new_dm, n_comp, ngr);
                MultiFab::Copy(this->PHI, real_mf, 0, 0, n_comp, ngr);
            }

            // IMPLICIT MESH DOMAIN (CELL-CENTERED) INFO
            {
                const int ngr = this->dom_data_ngr;
                
                shortMultiFab short_mf;
                iMultiFab int_mf;
                longMultiFab long_mf;
                MultiFab real_mf;
                
                short_mf.define(new_cc_ba, new_dm, eType_n_comp, ngr);
                short_mf.ParallelCopy(this->eType, 0, 0, eType_n_comp, ngr, ngr, this->geom.periodicity());
                this->eType.define(new_cc_ba, new_dm, eType_n_comp, ngr);
                Copy(this->eType, short_mf, 0, 0, eType_n_comp, ngr);

                real_mf.define(new_cc_ba, new_dm, eInfo_n_comp, ngr);
                real_mf.ParallelCopy(this->eInfo, 0, 0, eInfo_n_comp, ngr, ngr, this->geom.periodicity());
                this->eInfo.define(new_cc_ba, new_dm, eInfo_n_comp, ngr);
                MultiFab::Copy(this->eInfo, real_mf, 0, 0, eInfo_n_comp, ngr);

                int_mf.define(new_cc_ba, new_dm, eDomQuad_Nq_n_comp, ngr);
                int_mf.ParallelCopy(this->eDomQuad_Nq, 0, 0, eDomQuad_Nq_n_comp, ngr, ngr, this->geom.periodicity());
                this->eDomQuad_Nq.define(new_cc_ba, new_dm, eDomQuad_Nq_n_comp, ngr);
                iMultiFab::Copy(this->eDomQuad_Nq, int_mf, 0, 0, eDomQuad_Nq_n_comp, ngr);

                long_mf.define(new_cc_ba, new_dm, eDomQuad_pos_n_comp, ngr);
                long_mf.ParallelCopy(this->eDomQuad_pos, 0, 0, eDomQuad_pos_n_comp, ngr, ngr);
                this->eDomQuad_pos.define(new_cc_ba, new_dm, eDomQuad_pos_n_comp, ngr);
                Copy(this->eDomQuad_pos, long_mf, 0, 0, eDomQuad_pos_n_comp, ngr);

                int_mf.define(new_cc_ba, new_dm, eIntBouQuad_Nq_n_comp, ngr);
                int_mf.ParallelCopy(this->eIntBouQuad_Nq, 0, 0, eIntBouQuad_Nq_n_comp, ngr, ngr, this->geom.periodicity());
                this->eIntBouQuad_Nq.define(new_cc_ba, new_dm, eIntBouQuad_Nq_n_comp, ngr);
                iMultiFab::Copy(this->eIntBouQuad_Nq, int_mf, 0, 0, eIntBouQuad_Nq_n_comp, ngr);

                long_mf.define(new_cc_ba, new_dm, eIntBouQuad_pos_n_comp, ngr);
                long_mf.ParallelCopy(this->eIntBouQuad_pos, 0, 0, eIntBouQuad_pos_n_comp, ngr, ngr);
                this->eIntBouQuad_pos.define(new_cc_ba, new_dm, eIntBouQuad_pos_n_comp, ngr);
                Copy(this->eIntBouQuad_pos, long_mf, 0, 0, eIntBouQuad_pos_n_comp, ngr);
            }
            
            // IMPLICIT MESH BOUNDARY (FACE-CENTERED) INFO
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
            {
                const int ngr = this->bou_data_ngr;

                shortMultiFab short_mf;
                iMultiFab int_mf;
                longMultiFab long_mf;

                short_mf.define(new_fc_ba[dim], new_dm, eBouType_n_comp, ngr);
                short_mf.ParallelCopy(this->eBouType[dim], 0, 0, eBouType_n_comp, ngr, ngr, this->geom.periodicity());
                this->eBouType[dim].define(new_fc_ba[dim], new_dm, eBouType_n_comp, ngr);
                Copy(this->eBouType[dim], short_mf, 0, 0, eBouType_n_comp, ngr);

                int_mf.define(new_fc_ba[dim], new_dm, eBouQuad_Nq_n_comp, ngr);
                int_mf.ParallelCopy(this->eBouQuad_Nq[dim], 0, 0, eBouQuad_Nq_n_comp, ngr, ngr, this->geom.periodicity());
                this->eBouQuad_Nq[dim].define(new_fc_ba[dim], new_dm, eBouQuad_Nq_n_comp, ngr);
                iMultiFab::Copy(this->eBouQuad_Nq[dim], int_mf, 0, 0, eBouQuad_Nq_n_comp, ngr);

                long_mf.define(new_fc_ba[dim], new_dm, eBouQuad_pos_n_comp, ngr);
                long_mf.ParallelCopy(this->eBouQuad_pos[dim], 0, 0, eBouQuad_pos_n_comp, ngr, ngr);
                this->eBouQuad_pos[dim].define(new_fc_ba[dim], new_dm, eBouQuad_pos_n_comp, ngr);
                Copy(this->eBouQuad_pos[dim], long_mf, 0, 0, eBouQuad_pos_n_comp, ngr);
            }

            // UPDATE IMPLICIT-MESH BOX ARRAYS AND DISTRIBUTION MAPPING
            this->cc_ba = new_cc_ba;
            AMREX_D_TERM
            (
                this->fc_ba[0] = new_fc_ba[0];,
                this->fc_ba[1] = new_fc_ba[1];,
                this->fc_ba[2] = new_fc_ba[2];
            )
            this->dm = new_dm;
        }
        // ============================================================

        // COPY MEMORY TO THE DEVICE ==================================
    #ifdef AMREX_USE_CUDA
        this->eDomQuad_device_mem.resize(this->eDomQuad_host_mem.size());
        this->eBouQuad_device_mem.resize(this->eBouQuad_host_mem.size());
        this->eIntBouQuad_device_mem.resize(this->eIntBouQuad_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->eDomQuad_host_mem.begin(), this->eDomQuad_host_mem.end(), this->eDomQuad_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->eBouQuad_host_mem.begin(), this->eBouQuad_host_mem.end(), this->eBouQuad_device_mem.begin());
        Gpu::copy(Gpu::hostToDevice, this->eIntBouQuad_host_mem.begin(), this->eIntBouQuad_host_mem.end(), this->eIntBouQuad_device_mem.begin());
    #endif
        // ============================================================
    }
    
    /**
     * \brief Eval and store the implicitly-defined mesh using N_PHI level sets and N_DOM domains.
     *
     * \param[in] t: time.
     * \param[in] N_PHI: number of level sets that will be projected.
     * \param[in] N_DOM: number of domains that will be considered.
     * \param[in] IBVP: a class that must have a method with the following prototype:
     *
     *                  void F_PHI(const Real t, const Real * x, Real * PHI) const;
     *                  void F_DOM2PHI(const int dom, int * phi_info) const;
     *
     * \param[in] im_flag: implicit mesh flag. If set to false, all cells are considered as entire
     *                     cells.
     */
    template <typename IBVP_CLASS>
    void MakeFromScratch(const Real t, const int N_PHI, const int N_DOM, const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        // PARAMETERS =================================================
        const int PHI_n_comp = (this->std_elem.Np)*N_PHI;
        const int PHI_ngr = this->dom_data_ngr;
        // ============================================================
        
        // PROJECT THE LEVEL SETS OVER THE GRID =======================
        // INIT THE MULTIFAB THAT WILL CONTAIN THE PROJECTED LEVELSETS
        this->PHI.define(this->cc_ba, this->dm, PHI_n_comp, PHI_ngr);

        // EVAL THE PROJECTION
        ProjectLevelsetsOverGrid(t, this->geom, this->std_elem, N_PHI, this->PHI, IBVP);
        // ============================================================

        // MAKE THE IMPLICIT MESH USING THE PROJECTED LEVEL SETS ======
        this->Make(N_PHI, N_DOM, IBVP, im_flag);
        // ============================================================
    }

    /**
     * \brief Eval and store the implicitly-defined mesh using one level set and one domain.
     *
     * \param[in] t: time.
     * \param[in] IBVP: a class that must have a method with the following prototype:
     *
     *                  void F_PHI(const Real t, const Real * x, Real * PHI) const;
     *                  void F_DOM2PHI(const int dom, int * phi_info) const;
     *
     * \param[in] im_flag: implicit mesh flag. If set to false, all cells are considered as entire
     *                     cells.
     *
     */
    template <typename IBVP_CLASS>
    void MakeFromScratch(const Real t, const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        this->MakeFromScratch(t, 1, 1, IBVP, im_flag);
    }

    /**
     * \brief Eval and store the implicitly-defined mesh using one level set, one domain, at time = 0.
     *
     * \param[in] IBVP: a class that must have a method with the following prototype:
     *
     *                  void F_PHI(const Real t, const Real * x, Real * PHI) const;
     *                  void F_DOM2PHI(const int dom, int * phi_info) const;
     *
     * \param[in] im_flag: implicit mesh flag. If set to false, all cells are considered as entire
     *                     cells.
     *
     */
    template <typename IBVP_CLASS>
    void MakeFromScratch(const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        this->MakeFromScratch(0.0, 1, 1, IBVP, im_flag);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################


} // namespace DG
} // namespace amrex

#endif