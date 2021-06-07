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
#include <AMReX_dG_Cell_BFX.H>

#define __DG_STD_ELEM_N_SPACE_BOUNDARIES__ (2*AMREX_SPACEDIM)
#define __DG_STD_ELEM_N_SPACE_EDGES__ AMREX_D_PICK(2, 4, 12)
#define __DG_STD_ELEM_N_SPACE_CORNERS__ AMREX_D_PICK(2, 4, 8)

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

// GHOST
#define __DG_CELL_GHOST__ 1000

// MASK
#define __DG_CELL_NOT_MASKED__ 0
#define __DG_CELL_MASKED__ 1
#define __DG_LEVEL_BOUNDARY__ 2

// ELEMENT TYPE
#define __DG_ELM_TYPE_N_COMP_PER_DOM__ 1
#define __DG_ELM_TYPE_UNDEFINED__ -2
#define __DG_ELM_TYPE_EMPTY__ -1
#define __DG_ELM_TYPE_ENTIRE__ 1
#define __DG_ELM_TYPE_LARGE__ 2
#define __DG_ELM_TYPE_SMALL__ 3
#define __DG_ELM_TYPE_PARTIAL__ 5

// ELEMENT BOUNDARY TYPE
#define __DG_ELM_BOU_TYPE_N_COMP_PER_DOM__ 1
#define __DG_ELM_BOU_TYPE_UNDEFINED__ -2
#define __DG_ELM_BOU_TYPE_EMPTY__ -1
#define __DG_ELM_BOU_TYPE_ENTIRE__ 1
#define __DG_ELM_BOU_TYPE_PARTIAL__ 5
#define __DG_ELM_BOU_TYPE_PARTIAL_INVALID__ 6
#define __DG_ELM_BOU_TYPE_POS_WALL__ 7
#define __DG_ELM_BOU_TYPE_NEG_WALL__ 8

// ELEMENT ID / COLOR
#define __DG_ELM_ID_UNDEFINED__ -1

#define __DG_CELL_NBR_INFO_N_COMP_PER_DOM__ AMREX_D_PICK(2, 4, 6)
#define __DG_ELM_MAX_N_NBR_PER_DOM__ AMREX_D_PICK(2, 16, 64)
#define __DG_ELM_NBR_INFO_N_COMP_PER_DOM__ (1+2*__DG_ELM_MAX_N_NBR_PER_DOM__)

#define __DG_ELM_COLOR_N_COMP_PER_DOM__ 1
#define __DG_NBR_COLOR_N_COMP_PER_DOM__ AMREX_D_PICK(2, 4, 6)
#define __DG_ELM_COLOR_UNDEFINED__ -99999

// CSR MATRIX FORMAT
#define __DG_CSR_DOM_INFO_N_COMP_PER_DOM__ 3
#define __DG_CSR_BOU_INFO_N_COMP_PER_DOM__ 2

// DEBUG
#define __i__ 2
#define __j__ 2
#define __k__ 0

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


// GHOST CELLS ========================================================
/**
 * \brief Given the cell_type, return true if the cell is a ghost cell, false otherwise.
 *
 * \param[in] cell_type: cell type.
 *
 * \return true if the cell is a ghost cell, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_GHOST(const short cell_type)
{
    const bool cell_is_ghost = (cell_type/__DG_CELL_GHOST__ == 1);

    return cell_is_ghost;
}
// ====================================================================


// CELL TYPE ==========================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_TYPE(const int dom)
{
    return (__DG_CELL_TYPE_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_EMPTY(const short ctype)
{
    return (ctype == __DG_CELL_TYPE_EMPTY__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_ENTIRE(const short ctype)
{
    return (ctype%10 == __DG_CELL_TYPE_ENTIRE__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_IS_CUT(const short ctype)
{
    return (ctype%10 == __DG_CELL_TYPE_CUT__);
}
// ====================================================================


// CELL BOU TYPE ======================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_BOU_TYPE(const int dom)
{
    return (__DG_CELL_BOU_TYPE_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_BOU_IS_EMPTY(const short cbtype)
{
    return (cbtype == __DG_CELL_BOU_TYPE_EMPTY__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_BOU_IS_ENTIRE(const short cbtype)
{
    return (cbtype == __DG_CELL_BOU_TYPE_ENTIRE__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELL_BOU_IS_CUT(const short cbtype)
{
    return (cbtype == __DG_CELL_BOU_TYPE_CUT__);
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


// NEIGHBORS INFORMATION (CELLS) ======================================
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
    return (val == __DG_CELL_NOT_MASKED__);
}
// ====================================================================


// ELEMENT TYPE =======================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_TYPE(const int dom)
{
    return (__DG_ELM_TYPE_N_COMP_PER_DOM__*dom);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_EMPTY(const short etype)
{
    return (etype == __DG_ELM_TYPE_EMPTY__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_NOT_EMPTY(const short etype)
{
    return (etype != __DG_ELM_TYPE_EMPTY__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_ENTIRE(const short etype)
{
    return (etype%10 == __DG_ELM_TYPE_ENTIRE__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_PARTIAL(const short etype)
{
    return (etype == __DG_ELM_TYPE_PARTIAL__);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_SMALL(const short etype)
{
    return (etype%10 == __DG_ELM_TYPE_SMALL__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_LARGE(const short etype)
{
    return (etype%10 == __DG_ELM_TYPE_LARGE__);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_LARGE_OR_SMALL(const short etype)
{
    return ((etype%10 == __DG_ELM_TYPE_LARGE__) || (etype%10 == __DG_ELM_TYPE_SMALL__));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_VALID(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_not_ghost = (etype/__DG_CELL_GHOST__ != 1);
    
    return ((elm_is_entire || elm_is_large) && elm_is_not_ghost);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_EXTENDED(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    
    return ((elm_is_entire || elm_is_large) && (etype/10 == 1));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_ENTIRE_EXTENDED(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    
    return (elm_is_entire && (etype/10 == 1));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_GHOST(const short etype)
{
    const bool elm_is_ghost = (etype/__DG_CELL_GHOST__ == 1);

    return elm_is_ghost;
}
// ====================================================================


// ELEMENT CENTROID ===================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X1(const int dom)
{
    return (AMREX_SPACEDIM*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X2(const int dom)
{
    return (AMREX_SPACEDIM*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_CENTROID_X3(const int dom)
{
    return (AMREX_SPACEDIM*dom+2);
}
// ====================================================================


// NEIGHBORS INFORMATION (ELEMENTS) ===================================
/**
 * \brief Get indices of the cell providing support for the basis functions over the cell (i,j,k).
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell; remember that if the cell (i,j,k) is small, then the support
 *                   of the basis functions is that of the cells that extends into it.
 * \param[out] (bf_i,bf_j,bf_k): indices of the cell that provides the support for the current cell.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void BF_CELL(const int i, const int j, const int k, const short etype,
             int & bf_i, int & bf_j, int & bf_k)
{
    if (ELM_IS_SMALL(etype))
    {
        const int b = etype/10;
        int nbr_b;
        NBR_CELL(i, j, k, b, bf_i, bf_j, bf_k, nbr_b);
    }
    else
    {
        bf_i = i;
        bf_j = j;
        bf_k = k;
    }
}

/**
 * \brief Given the indices of two cells, return true if the cells are merged.
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] (nbr_i,nbr_j,nbr_k): indices of the neighboring cell.
 *
 * \return true if the cells are merged, false otherwise.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELLS_ARE_MERGED(const int i, const int j, const int k,
                      const int nbr_i, const int nbr_j, const int nbr_k)
{
    return ((i == nbr_i) && (j == nbr_j) && (k == nbr_k));
}
/**
 * \brief Given the indices of two cells and the element type, return true if the cells are merged.
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell.
 * \param[in] (nbr_i,nbr_j,nbr_k): indices of the neighboring cell.
 *
 * \return true if the cells are merged, false otherwise.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELLS_ARE_MERGED(const int i, const int j, const int k, const short etype,
                      const int nbr_i, const int nbr_j, const int nbr_k)
{
    int bf_i, bf_j, bf_k;
    BF_CELL(i, j, k, etype, bf_i, bf_j, bf_k);
    return ((bf_i == nbr_i) && (bf_j == nbr_j) && (bf_k == nbr_k));
}
/**
 * \brief Given the indices of two cells and the element type, return true if the cells are merged.
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] (nbr_i,nbr_j,nbr_k): indices of the neighboring cell.
 * \param[in] nbr_etype: type of the neighboring cell.
 *
 * \return true if the cells are merged, false otherwise.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELLS_ARE_MERGED(const int i, const int j, const int k,
                      const int nbr_i, const int nbr_j, const int nbr_k, const short nbr_etype)
{
    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
    BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
    return ((i == nbr_BF_i) && (j == nbr_BF_j) && (k == nbr_BF_k));
}
/**
 * \brief Given the indices of two cells and the element type, return true if the cells are merged.
 *
 * \param[in] (i,j,k): indices of the cell.
 * \param[in] etype: type of the cell.
 * \param[in] (nbr_i,nbr_j,nbr_k): indices of the neighboring cell.
 * \param[in] nbr_etype: type of the neighboring cell.
 *
 * \return true if the cells are merged, false otherwise.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CELLS_ARE_MERGED(const int i, const int j, const int k, const short etype,
                      const int nbr_i, const int nbr_j, const int nbr_k, const short nbr_etype)
{
    int bf_i, bf_j, bf_k;
    int nbr_BF_i, nbr_BF_j, nbr_BF_k;
    BF_CELL(i, j, k, etype, bf_i, bf_j, bf_k);
    BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
    return ((bf_i == nbr_BF_i) && (bf_j == nbr_BF_j) && (bf_k == nbr_BF_k));
}
// ====================================================================


// ELEMENT BOUNDARY TYPE ==============================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_TYPE(const int dom)
{
    return (__DG_ELM_BOU_TYPE_N_COMP_PER_DOM__*dom);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_VALID(const short ebtype)
{
    return (ebtype != __DG_ELM_BOU_TYPE_PARTIAL_INVALID__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_EMPTY(const short ebtype)
{
    return (ebtype == __DG_ELM_BOU_TYPE_EMPTY__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_POS_WALL(const short ebtype)
{
    return (ebtype == __DG_ELM_BOU_TYPE_POS_WALL__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_NEG_WALL(const short ebtype)
{
    return (ebtype == __DG_ELM_BOU_TYPE_NEG_WALL__);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_WALL(const short ebtype)
{
    return ((ebtype == __DG_ELM_BOU_TYPE_POS_WALL__) || (ebtype == __DG_ELM_BOU_TYPE_NEG_WALL__));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_NOT_WALL(const short ebtype)
{
    return ((ebtype != __DG_ELM_BOU_TYPE_POS_WALL__) && (ebtype != __DG_ELM_BOU_TYPE_NEG_WALL__));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_INTERNAL(const short ebtype)
{
    const bool eb_is_valid = ELM_BOU_IS_VALID(ebtype);
    const bool eb_is_not_wall = ELM_BOU_IS_NOT_WALL(ebtype);

    return (eb_is_valid && eb_is_not_wall);
}
// ====================================================================


// ELEMENT ID =========================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CELL_NBR_POS_X(const int dom, const int b)
{
    return (dom*__DG_CELL_NBR_INFO_N_COMP_PER_DOM__+b);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_N_NBRS(const int dom)
{
    return (dom*__DG_ELM_NBR_INFO_N_COMP_PER_DOM__+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_NBR_POS_X(const int dom, const int n)
{
    return (dom*__DG_ELM_NBR_INFO_N_COMP_PER_DOM__+2*n+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_NBR_N_DOF(const int dom, const int n)
{
    return (dom*__DG_ELM_NBR_INFO_N_COMP_PER_DOM__+2*n+2);
}
// ====================================================================


// CSR MATRIX FORMAT ==================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CSR_DOM_INFO_POS_X(const int dom)
{
    return (dom*__DG_CSR_DOM_INFO_N_COMP_PER_DOM__+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CSR_DOM_INFO_N_DOF(const int dom)
{
    return (dom*__DG_CSR_DOM_INFO_N_COMP_PER_DOM__+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CSR_DOM_INFO_NNZ_OFFSET(const int dom)
{
    return (dom*__DG_CSR_DOM_INFO_N_COMP_PER_DOM__+2);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CSR_BOU_INFO_NNZ_OFFSET(const int dom, const int mp)
{
    return (dom*__DG_CSR_BOU_INFO_N_COMP_PER_DOM__+mp);
}
// ====================================================================


// AMR ================================================================
/**
 * \brief Given the fine cell indices and the refinement ratio, eval the coarse cell indices.
 *
 * \param[in] (fi,fj,fk): indices of the fine cell.
 * \param[in] rr: refinement ratio.
 * \param[in] (ci,cj,ck): indices of the coarse cell corresponding to the fine cell.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void FINE_TO_COARSE(const int fi, const int fj, const int fk, const IntVect & rr,
                    int & ci, int & cj, int & ck)
{
#if (AMREX_SPACEDIM == 1)
    const int ti = (fi%rr[0] >= 0) ? fi%rr[0] : (fi%rr[0]+rr[0]);
    ci = (fi-ti)/rr[0];
    cj = fj;
    ck = fk;
#endif
#if (AMREX_SPACEDIM == 2)
    const int ti = (fi%rr[0] >= 0) ? fi%rr[0] : (fi%rr[0]+rr[0]);
    const int tj = (fj%rr[1] >= 0) ? fj%rr[1] : (fj%rr[1]+rr[1]);
    ci = (fi-ti)/rr[0];
    cj = (fj-tj)/rr[1];
    ck = fk;
#endif
#if (AMREX_SPACEDIM == 3)
    const int ti = (fi%rr[0] >= 0) ? fi%rr[0] : (fi%rr[0]+rr[0]);
    const int tj = (fj%rr[1] >= 0) ? fj%rr[1] : (fj%rr[1]+rr[1]);
    const int tk = (fk%rr[2] >= 0) ? fk%rr[2] : (fk%rr[2]+rr[2]);
    ci = (fi-ti)/rr[0];
    cj = (fj-tj)/rr[1];
    ck = (fk-tk)/rr[2];
#endif
}
// ====================================================================
// ####################################################################



// MESH CLASS #########################################################
/**
 * \brief Structured mesh parameters class.
 *
*/
struct MeshParameters
{
    // DATA MEMBERS ===================================================
    bool embedded_geometry_is_active;
    std::string embedded_geometry_defined_by;
    bool box_pruning_is_active;
    
    int projected_level_set_p;
    int quadrature_order_projected_level_set;
    
    int quadrature_order_regular_elements, quadrature_order_cut_elements;

    int post_processing_grid_order;
    // ================================================================


    // CONSTRUCTOR ====================================================
    MeshParameters()
    :
    embedded_geometry_is_active{false},
    embedded_geometry_defined_by{""},
    box_pruning_is_active{false},
    projected_level_set_p{-1},
    quadrature_order_projected_level_set{-1},
    quadrature_order_regular_elements{-1},
    quadrature_order_cut_elements{-1},
    post_processing_grid_order{1}
    {}
    // ================================================================
};

/**
 * \brief Mesh class.
 *
*/
struct Mesh
{
    // DATA MEMBERS ===================================================
    static const int dom_data_n_grow = 1;
    static const int bou_data_n_grow = 0;

    MeshParameters params;

    // BOXARRAY AND DISTRIBUTION MAPPING
    BoxArray ba;
    DistributionMapping dm;

    // CELL TYPE / CELL BOUNDARY TYPE / CELL INFO
    sMultiFab cell_type;
    Array<sMultiFab, AMREX_SPACEDIM> cell_bou_type;
    MultiFab cell_vf;

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
    void read_input_file();
    // ================================================================


    // READERS ========================================================
    bool uses_embedded_geometries() const;
    bool uses_level_set() const;
    bool uses_projected_level_set() const;
    bool uses_box_pruning() const;
    // ================================================================


    // INITIALIZE THE MULTIFABS =======================================
    /**
     * \brief Initialize the multifabs that will contain the mesh information.
     *
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                 processors.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void init_multifabs(const BoxArray & ba_, const DistributionMapping & dm_, const IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();
        
        // COMPONENTS OF THE CELL / CELL BOU TYPE MULTIFABS
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
        const BoxArray fc_ba[1] = {convert(ba_, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(ba_, IntVect(1,0)),
                                   convert(ba_, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(ba_, IntVect(1,0,0)),
                                   convert(ba_, IntVect(0,1,0)),
                                   convert(ba_, IntVect(0,0,1))};
#endif
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        // CELL / CELL BOU TYPE
        this->cell_type.define(ba_, dm_, cell_type_n_comp, this->dom_data_n_grow);
        this->cell_type = __DG_CELL_TYPE_UNDEFINED__;
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->cell_bou_type[dim].define(fc_ba[dim], dm_, cell_bou_type_n_comp, this->bou_data_n_grow);
            this->cell_bou_type[dim] = __DG_CELL_BOU_TYPE_UNDEFINED__;
        }

        // CELL VOLUME FRACTION
        this->cell_vf.define(ba_, dm_, n_domains, this->dom_data_n_grow);
        this->cell_vf = 0.0;
        
        // QUADRATURE INFO
        this->cell_dom_quad_info.define(ba_, dm_, cell_dom_quad_info_n_comp, this->dom_data_n_grow);
        this->cell_dom_quad_info = -1L;

        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->cell_bou_quad_info[dim].define(fc_ba[dim], dm_, cell_bou_quad_info_n_comp, this->bou_data_n_grow);
            this->cell_bou_quad_info[dim] = -1L;
        }

        this->cell_int_bou_quad_info.define(ba_, dm_, cell_int_bou_quad_info_n_comp, this->dom_data_n_grow);
        this->cell_int_bou_quad_info = -1L;
        // ------------------------------------------------------------
    }

    /**
     * \brief Initialize the multifabs that will contain the level set functions.
     *
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[inout] L: a MultiFab object that will be initialized to contain the dG coefficients of
     *                  the projected level set functions.
     *
    */
    void init_projected_level_set_multifabs(const BoxArray & ba_, const DistributionMapping & dm_, MultiFab & L) const
    {
        // ORDER
        const int p = this->params.projected_level_set_p;
        const int Np = AMREX_D_TERM((1+p),*(1+p),*(1+p));
        const int L_n_comp = Np;

        // INITIALIZATION
        L.define(ba_, dm_, L_n_comp, this->dom_data_n_grow);
        L = 0.0;
    }
    // ================================================================


    // GHOST CELLS ====================================================
    void label_ghost_cells(const Geometry & geom, const int n_domains);
    // ================================================================


    // LEVEL SET FUNCTIONS ============================================
    /**
     * \brief Project level set over the grid.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[inout] L: a MultiFab object that will contain the dG coefficients of the projected level
     *                  set function.
     * \param[in] level_set: level set object.
     *
    */
    template <typename LEVEL_SET>
    void project_space_level_set(const Real t, const Geometry & geom, MultiFab & L, LEVEL_SET & level_set) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // ORDER
        const int p = this->params.projected_level_set_p;
        const int Np = AMREX_D_TERM((1+p),*(1+p),*(1+p));

        // QUADRATURE
        const int qo = this->params.quadrature_order_projected_level_set;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        StandardRectangle<AMREX_SPACEDIM> std_elm;
        int dom_Nq;
        // ------------------------------------------------------------

        // INITIALIZE THE QUADRATURE FOR THE STANDARD ELEMENT ---------
        std_elm.set_quadrature(&dx[0], qo);
        std_elm.set_mass_matrix(p);

        dom_Nq = std_elm.dom_Nq;
        // ------------------------------------------------------------

        // INITIALIZE MULTIFAB ----------------------------------------
        L = 0.0;
        // ------------------------------------------------------------

        // EVAL THE PROJECTION ----------------------------------------
        AMREX_D_TERM
        (
            Real const * dom_XI1q_ptr = std_elm.dom_XI1q.data();,
            Real const * dom_XI2q_ptr = std_elm.dom_XI2q.data();,
            Real const * dom_XI3q_ptr = std_elm.dom_XI3q.data();
        )
        Real const * dom_Wq_ptr = std_elm.dom_Wq.data();
        Real const * iMM_ptr = std_elm.iMM.data();

        for (MFIter mfi(L); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            Array4<Real> const & L_fab = L.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real integrand;
                Real phi;
                
                // BASIS FUNCTIONS
                space_cell_bf_level_set bf(&prob_lo[0], &dx[0], i, j, k, p);

                // EVAL THE INTEGRAL
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    AMREX_D_TERM
                    (
                        x[0] = dom_XI1q_ptr[q]+cell_lo[0];,
                        x[1] = dom_XI2q_ptr[q]+cell_lo[1];,
                        x[2] = dom_XI3q_ptr[q]+cell_lo[2];
                    )
                    w = dom_Wq_ptr[q];

                    // Level set at x
                    phi = level_set.eval(t, x);

                    // Basis functions at x
                    bf.eval_bf_only_table(x);

                    for (int rs = 0; rs < Np; ++rs)
                    {
                        integrand = 0.0;
                        for (int cs = 0; cs < Np; ++cs)
                        {
                            integrand += iMM_ptr[rs+cs*Np]*bf.BF[cs];
                        }
                        integrand *= phi;

                        // Sum integral contribution
                        L_fab(i,j,k,rs) += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE FROM SCRATCH ==============================================
    /**
     * \brief Make the data structures that will store the mesh info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     * \param[in] level_set: level set object.
     *
     * Note: If any change is made to this function. The same change must be applied to:
     *       make_from_scratch_by_projected_level_set
     *
    */
    template <typename IBVP, typename LEVEL_SET>
    void make_from_scratch_by_level_set(const Real t, const Geometry & geom, const BoxArray & ba_, const DistributionMapping & dm_,
                                        IBVP & ibvp, LEVEL_SET & level_set)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
        const Real atol = 1.0e-12;

        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // QUADRATURE
        const int qo_reg = this->params.quadrature_order_regular_elements;
        const int qo_cut = this->params.quadrature_order_cut_elements+1;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        StandardRectangle<AMREX_SPACEDIM> std_elm;
        long cell_dom_quad_mem_pos;
        Array<long, AMREX_SPACEDIM> cell_bou_quad_mem_pos;
        long cell_int_bou_quad_mem_pos;
        // ------------------------------------------------------------

        // SET TIME FOR THE LEVEL SET ---------------------------------
        level_set.set_time(t);
        // ------------------------------------------------------------

        // PRUNE BOX ARRAY AND DISTRIBUTION MAPPING (IF REQUESTED) ----
        if (this->uses_box_pruning())
        {
            sMultiFab dummy(ba_, dm_, 1, this->dom_data_n_grow);
            
            BoxList new_bl;

            for (MFIter mfi(dummy); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.fabbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);
                bool bx_is_empty;

                bx_is_empty = true;
                for (int dom = 0; dom < n_domains; ++dom)
                {
                    level_set.set_domain(dom);

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
                        AlgoimQuadRule quad = AlgoimQuadGen(level_set, cell_lo, cell_hi, -1, -1, qo);

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
                    new_bl.push_back(mfi.validbox());
                }
            }
            
            // SHARE THE INFO
            AllGatherBoxes(new_bl.data());

            // NEW BOXARRAY AND DISTRIBUTION MAPPING
            this->ba = BoxArray(new_bl);
            this->dm.define(this->ba);
        }
        else
        {
            this->ba = ba_;
            this->dm = dm_;
        }

        /* DEBUG
        {
            const int n_procs = ParallelDescriptor::NProcs();
            for (int r = 0; r < n_procs; ++r)
            {
            Print(r) << "this->ba: " << this->ba << std::endl;
            Print(r) << "this->dm: " << this->dm << std::endl;
            ParallelDescriptor::Barrier();
            }
            ParallelDescriptor::Barrier();
            exit(-1);
        }
        */
        // ------------------------------------------------------------

        // INIT THE MULTIFAB ------------------------------------------
        this->init_multifabs(this->ba, this->dm, ibvp);
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
            Array4<Real> const & cell_vf_fab = this->cell_vf.array(mfi);
            Array4<long> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
            Array4<long> const & cell_int_bou_quad_info_fab = this->cell_int_bou_quad_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            {
                level_set.set_domain(dom);

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
                        AlgoimQuadRule quad = AlgoimQuadGen(level_set, cell_lo, cell_hi, -1, -1, qo_cut);

                        // Compute the volume fraction
                        cell_volume = quad([](const auto & /*x*/) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;

                        // EMPTY CELL
                        if (volume_fraction < atol)
                        {
                            cell_is_cut = false;

                            // CELL TYPE
                            cell_type_fab(i,j,k,CELL_TYPE(dom)) = __DG_CELL_TYPE_EMPTY__;

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = 0.0;

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

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = volume_fraction;

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

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = volume_fraction;

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
                        AlgoimQuadRule quad = AlgoimQuadGen(level_set, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo_cut);

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

                            un = level_set.grad(quad.nodes[q].x);
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
                    else
                    {
                        // QUADRATURE INFO
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom)) = 0L;
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom)) = -1L;
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
                    level_set.set_domain(dom);

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
                            AlgoimQuadRule quad = AlgoimQuadGen(level_set, cell_lo, cell_hi, dir, 0, qo_cut);

                            // Compute the surface fraction
#if (AMREX_SPACEDIM == 1)
                            surface_fraction = (level_set.eval(cell_lo) < 0.0) ? 1.0 : 0.0;
#else
                            boundary_surface = quad([](const auto & /*x*/) {return 1.0;});
                            surface_fraction = boundary_surface/entire_boundary_surface;
#endif

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

#ifdef AMREX_USE_GPU
        // COPY TO DEVICE ---------------------------------------------
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
        // ------------------------------------------------------------
#endif
        // LABEL GHOST CELLS ------------------------------------------
        this->label_ghost_cells(geom, n_domains);
        // ------------------------------------------------------------
    }

    /**
     * \brief Make the data structures that will store the mesh info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] L: MultiFab containing the coefficients of the projected level set function.
     * \param[in] ibvp: initial boundary value problem object.
     *
     * Note: If any change is made to this function. The same change must be applied to:
     *       make_from_scratch_by_level_set
     *
    */
    template <typename IBVP>
    void make_from_scratch_by_projected_level_set(const Real t, const Geometry & geom, const BoxArray & ba_, const DistributionMapping & dm_,
                                                  const MultiFab & L, IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
        const Real atol = 1.0e-12;

        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // ORDER
        const int p = this->params.projected_level_set_p;

        // QUADRATURE
        const int qo_reg = this->params.quadrature_order_regular_elements;
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
            sMultiFab dummy(ba_, dm_, 1, this->dom_data_n_grow);
            
            BoxList new_bl;

            for (MFIter mfi(dummy); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.fabbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);
                bool bx_is_empty;

                Array4<Real const> const & L_fab = L.array(mfi);

                bx_is_empty = true;
                for (int dom = 0; dom < n_domains; ++dom)
                {
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

                        // LEVEL SET FUNCTION
                        space_cell_bfx_algoim_level_set phi(&prob_lo[0], &dx[0], i, j, k, p, L_fab);
                        phi.set_sign(1-2*dom);

                        // ALGOIM QUADRATURE RULE
                        const int qo = 2;
                        AlgoimQuadRule quad = AlgoimQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                        cell_volume = quad([](const auto & x) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;
                        if (volume_fraction > atol)
                        {
                            bx_is_empty = false;
                        }
                    }
                }

                if (!bx_is_empty)
                {
                    new_bl.push_back(mfi.validbox());
                }
            }
            
            // SHARE THE INFO
            AllGatherBoxes(new_bl.data());

            // NEW BOXARRAY AND DISTRIBUTION MAPPING
            this->ba = BoxArray(new_bl);
            this->dm.define(this->ba);
        }
        else
        {
            this->ba = ba_;
            this->dm = dm_;
        }

        /* DEBUG
        {
            const int n_procs = ParallelDescriptor::NProcs();
            for (int r = 0; r < n_procs; ++r)
            {
            Print(r) << "this->ba: " << this->ba << std::endl;
            Print(r) << "this->dm: " << this->dm << std::endl;
            ParallelDescriptor::Barrier();
            }
            ParallelDescriptor::Barrier();
            exit(-1);
        }
        */
        // ------------------------------------------------------------

        // INIT THE MULTIFAB ------------------------------------------
        this->init_multifabs(this->ba, this->dm, ibvp);
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

            Array4<Real const> const & L_fab = L.array(mfi);

            Array4<short> const & cell_type_fab = this->cell_type.array(mfi);
            Array4<Real> const & cell_vf_fab = this->cell_vf.array(mfi);
            Array4<long> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
            Array4<long> const & cell_int_bou_quad_info_fab = this->cell_int_bou_quad_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            {
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

                    // LEVEL SET FUNCTION
                    space_cell_bfx_algoim_level_set phi(&prob_lo[0], &dx[0], i, j, k, p, L_fab);
                    phi.set_sign(1-2*dom);

                    // DOMAIN QUADRATURE
                    {
                        // Algoim data structure
                        AlgoimQuadRule quad = AlgoimQuadGen(phi, cell_lo, cell_hi, -1, -1, qo_cut);

                        // Compute the volume fraction
                        cell_volume = quad([](const auto & x) {return 1.0;});
                        volume_fraction = cell_volume/entire_cell_volume;

                        // EMPTY CELL
                        if (volume_fraction < atol)
                        {
                            cell_is_cut = false;

                            // CELL TYPE
                            cell_type_fab(i,j,k,CELL_TYPE(dom)) = __DG_CELL_TYPE_EMPTY__;

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = 0.0;

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

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = volume_fraction;

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

                            // CELL VOLUME FRACTION
                            cell_vf_fab(i,j,k,dom) = volume_fraction;

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
                        AlgoimQuadRule quad = AlgoimQuadGen(phi, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo_cut);

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
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+0] = un[0];,
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+1] = un[1];,
                                this->cell_int_bou_quad_host_mem[pos+AMREX_SPACEDIM+1+2] = un[2];
                            )
                        }

                        // Update memory (nodes coordinates + weights +
                        // unit normal components)
                        cell_int_bou_quad_mem_pos += (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq;
                    }
                    else
                    {
                        // QUADRATURE INFO
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom)) = 0L;
                        cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom)) = -1L;
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

                Array4<Real const> const & L_fab = L.array(mfi);

                Array4<short> const & cell_bou_type_fab = this->cell_bou_type[dir].array(mfi);
                Array4<long> const & cell_bou_quad_info_fab = this->cell_bou_quad_info[dir].array(mfi);

                for (int dom = 0; dom < n_domains; ++dom)
                {
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

                        // LEVEL SET FUNCTION
                        space_cell_bfx_algoim_level_set phi(&prob_lo[0], &dx[0], i, j, k, p, L_fab);
                        phi.set_sign(1-2*dom);

                        // BOUNDARY QUADRATURE
                        {
                            // Algoim data structure
                            AlgoimQuadRule quad = AlgoimQuadGen(phi, cell_lo, cell_hi, dir, 0, qo_cut);

                            // Compute the surface fraction
#if (AMREX_SPACEDIM == 1)
                            {
                                const GpuArray<Real, AMREX_SPACEDIM> & tmp = {cell_lo[0]};
                                surface_fraction = (phi(tmp) < 0.0) ? 1.0 : 0.0;
                            }
#else
                            boundary_surface = quad([](const auto & x) {return 1.0;});
                            surface_fraction = boundary_surface/entire_boundary_surface;
#endif

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

#ifdef AMREX_USE_GPU
        // COPY TO DEVICE ---------------------------------------------
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
        // ------------------------------------------------------------
#endif
        // LABEL GHOST CELLS ------------------------------------------
        this->label_ghost_cells(geom, n_domains);
        // ------------------------------------------------------------
    }
    
    /**
     * \brief Make the data structures that will store the mesh info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void make_from_scratch(const Real t, const Geometry & geom, const BoxArray & ba_, const DistributionMapping & dm_, IBVP & ibvp)
    {
        {
            std::string msg;
            msg  = "\n";
            msg +=  "WARNING: AMReX_dG_Mesh.H - Mesh::make_from_scratch\n";
            msg += "| Calling make_from_scratch is deprecated.\n";
            Warning(msg);
        }

        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();
        // ------------------------------------------------------------

        // MAKE MESH --------------------------------------------------
        if (this->uses_embedded_geometries())
        {
            if (this->uses_level_set())
            {
                this->make_from_scratch_by_level_set(t, geom, ba_, dm_, ibvp, ibvp.level_set);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: AMReX_dG_Mesh.H - Mesh::make_from_scratch\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+this->params.embedded_geometry_defined_by+".\n";
                Abort(msg);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Mesh.H - Mesh::make_from_scratch\n";
            msg += "| At this moment embedded geometries must be active.\n";
            Abort(msg);
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // REMAKE =========================================================
    /**
     * \brief Remake the data structures that will store the mesh info.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] ba_: amrex BoxArray object; contains the block-structured grids.
     * \param[in] dm_: amrex DistributionMapping object; contains the distribution of the grids among the
     *                processors.
     * \param[in] ibvp: initial boundary value problem object.
     * \param[in] level_set: level set object.
     *
    */
    template <typename IBVP, typename LEVEL_SET>
    void remake_by_level_set(const Real t, const Geometry & geom, const BoxArray & ba_, const DistributionMapping & dm_, IBVP & ibvp, LEVEL_SET & level_set)
    {
        this->make_from_scratch_by_level_set(t, geom, ba_, dm_, ibvp, level_set);
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


    // CHECK QUADRATURE RULES =========================================
    void check_quadrature_rules(const Geometry & geom,
                                const int n_domains,
                                const iMultiFab & mask,
                                const Real * exact_volume,
                                const Real * exact_surface) const;
    void check_quadrature_rules(const Geometry & geom,
                                const int n_domains,
                                const Real * exact_volume,
                                const Real * exact_surface) const;
    // ================================================================

};
// ####################################################################


} // namespace dG
} // namespace amrex

#endif