//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_ImplicitMesh.H
 * \brief Contains constants, data structures and functions to manage the implicitly-defined mesh.
*/

#ifndef AMREX_DG_IMPLICIT_MESH_H_
#define AMREX_DG_IMPLICIT_MESH_H_

#include <AMReX_iMultiFab.H>
#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>

#include <AMReX_DG_StandardElement.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_BFX.H>

#define __i__ 221
#define __j__ 101
#define __k__ 14

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
void BF_CELL(const int i, const int j, const int k, const short etype,
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
             const int i, const int j, const int k, const short etype,
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
             const int i, const int j, const int k, const short etype,
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
int ELM_BOU_CENTROID_X1(const int dom)
{
    return (__DG_ELM_BOU_INFO_N_COMP_PER_DOM__*dom+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_CENTROID_X2(const int dom)
{
    return (__DG_ELM_BOU_INFO_N_COMP_PER_DOM__*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_CENTROID_X3(const int dom)
{
    return (__DG_ELM_BOU_INFO_N_COMP_PER_DOM__*dom+2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_CENTROID_X1(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_CENTROID_X2(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_CENTROID_X3(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+2);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_UN1(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+AMREX_SPACEDIM+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_UN2(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+AMREX_SPACEDIM+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_INT_BOU_UN3(const int dom)
{
    return (__DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*dom+AMREX_SPACEDIM+2);
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
int ELM_BOU_QUAD_NQ(const int dom, const int b)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom+1-(b%2));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_BOU_QUAD_POS(const int dom, const int b)
{
    return (__DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*dom+1-(b%2));
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ELM_FV_INFO_POS(const int dom)
{
    return (__DG_ELM_FV_INFO_N_COMP_PER_DOM__*dom);
}

/**
 * \brief Given a point x, find the nearest on the zero level of the level set function.
 *
 * \param[in] prob_lo: pointer to memory containing the lower ends of the grid.
 * \param[in] dx: pointer to memory containing the grid spacing.
 * \param[in] phi: object that evaluates the level set function.
 * \param[in] x: pointer to memory containing the location of the query point.
 * \param[inout] (n_i,n_j,n_k): On entry, indices of the cell containing x; on exit, indices of the
 *                              cell containing nx.
 * \param[out] nx: pointer to memory that, on exit, will contain the location of the nearest query point.
 * \param[in] tol: geometric tolerance used to stop the iterative search.
 * \param[in] it_max: maximum number of iterations.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void NEAREST_POINT(const Real * prob_lo, const Real * dx,
                   DG_PHI_space_BFX & phi,
                   const Real * x, int & n_i, int & n_j, int & n_k,
                   Real * cell_lo, Real * cell_hi, Real * nx,
                   const Real tol, const int it_max)
{
    int it;
    Real err;
    Real PHI, grad_PHI[AMREX_SPACEDIM];
    Real d1[AMREX_SPACEDIM], d2[AMREX_SPACEDIM], tmp, tmp2;

    AMREX_D_TERM
    (
        nx[0] = x[0];,
        nx[1] = x[1];,
        nx[2] = x[2];
    )
    it = 0;
    err = 1.0;
    while ((it < it_max) && (err > tol))
    {
        AMREX_D_TERM
        (
            cell_lo[0] = prob_lo[0]+n_i*dx[0];,
            cell_lo[1] = prob_lo[1]+n_j*dx[1];,
            cell_lo[2] = prob_lo[2]+n_k*dx[2];
        )
        AMREX_D_TERM
        (
            cell_hi[0] = cell_lo[0]+dx[0];,
            cell_hi[1] = cell_lo[1]+dx[1];,
            cell_hi[2] = cell_lo[2]+dx[2];
        )

        phi.set_fab_ptrs(n_i, n_j, n_k);
        phi.set_support(cell_lo, cell_hi);
        
        phi.eval_BF_table(nx);
        PHI = phi.eval_unsafe();
        phi.eval_grad_unsafe(grad_PHI);

        // d1
        AMREX_D_TERM
        (
            tmp  = grad_PHI[0]*grad_PHI[0];,
            tmp += grad_PHI[1]*grad_PHI[1];,
            tmp += grad_PHI[2]*grad_PHI[2];
        )
        tmp = -PHI/tmp;
        AMREX_D_TERM
        (
            d1[0] = tmp*grad_PHI[0];,
            d1[1] = tmp*grad_PHI[1];,
            d1[2] = tmp*grad_PHI[2];
        )

        // d2
        AMREX_D_TERM
        (
            tmp  = grad_PHI[0]*grad_PHI[0];,
            tmp += grad_PHI[1]*grad_PHI[1];,
            tmp += grad_PHI[2]*grad_PHI[2];
        )
        AMREX_D_TERM
        (
            tmp2  = (x[0]-nx[0])*grad_PHI[0];,
            tmp2 += (x[1]-nx[1])*grad_PHI[1];,
            tmp2 += (x[2]-nx[2])*grad_PHI[2];
        )
        tmp = tmp2/tmp;
        AMREX_D_TERM
        (
            d2[0] = (x[0]-nx[0])-tmp*grad_PHI[0];,
            d2[1] = (x[1]-nx[1])-tmp*grad_PHI[1];,
            d2[2] = (x[2]-nx[2])-tmp*grad_PHI[2];
        )

        // nx^{k+1}
        AMREX_D_TERM
        (
            nx[0] += d1[0]+d2[0];,
            nx[1] += d1[1]+d2[1];,
            nx[2] += d1[2]+d2[2];
        )
#if (AMREX_SPACEDIM == 1)
        n_i = (int) std::floor((nx[0]-prob_lo[0])/dx[0]);

        err  = d1[0]*d1[0];
        err += d2[0]*d2[0];
#endif
#if (AMREX_SPACEDIM == 2)
        n_i = (int) std::floor((nx[0]-prob_lo[0])/dx[0]);
        n_j = (int) std::floor((nx[1]-prob_lo[1])/dx[1]);

        err  = d1[0]*d1[0]+d1[1]*d1[1];
        err += d2[0]*d2[0]+d2[1]*d2[1];
#endif
#if (AMREX_SPACEDIM == 3)
        n_i = (int) std::floor((nx[0]-prob_lo[0])/dx[0]);
        n_j = (int) std::floor((nx[1]-prob_lo[1])/dx[1]);
        n_k = (int) std::floor((nx[2]-prob_lo[2])/dx[2]);

        err  = d1[0]*d1[0]+d1[1]*d1[1]+d1[2]*d1[2];
        err += d2[0]*d2[0]+d2[1]*d2[1]+d2[2]*d2[2];
#endif

        err = std::sqrt(err);

        it += 1;
    }

    AMREX_D_TERM
    (
        cell_lo[0] = prob_lo[0]+n_i*dx[0];,
        cell_lo[1] = prob_lo[1]+n_j*dx[1];,
        cell_lo[2] = prob_lo[2]+n_k*dx[2];
    )
    AMREX_D_TERM
    (
        cell_hi[0] = cell_lo[0]+dx[0];,
        cell_hi[1] = cell_lo[1]+dx[1];,
        cell_hi[2] = cell_lo[2]+dx[2];
    )

    phi.set_fab_ptrs(n_i, n_j, n_k);
    phi.set_support(cell_lo, cell_hi);
}

/**
 * \brief Given the etype, return the corresponding description as a string.
 *
 * \param[in] etype: element type.
 *
 * \return a string describing the element type
 *
*/
AMREX_FORCE_INLINE
std::string ELM_TYPE_DESCRIPTION(const short etype)
{
    const bool elm_is_undefined = (etype == __DG_ELM_TYPE_UNDEFINED__);
    const bool elm_is_empty = (etype == __DG_ELM_TYPE_EMPTY__);
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
    const bool elm_is_partial = (etype == __DG_ELM_TYPE_PARTIAL__);
    const bool elm_is_ghost = (etype/100 == 1);
    const bool elm_is_not_ghost = !elm_is_ghost;
    const bool elm_is_valid = (elm_is_entire || elm_is_large) && elm_is_not_ghost;
    const bool elm_is_extended = elm_is_valid && (etype/10 == 1);
    const int merged_b = (elm_is_small) ? (etype/10) : -1;

    std::string etype_description = std::to_string((int)etype)+": ";
    if (elm_is_undefined) etype_description += "Undefined";
    if (elm_is_empty) etype_description += "Empty";
    if (elm_is_entire) etype_description += "Entire";
    if (elm_is_large) etype_description += "Large";
    if (elm_is_small) etype_description += "Small - Merged through boundary: "+std::to_string(merged_b);
    if (elm_is_partial) etype_description += "Partial";
    if (elm_is_valid) etype_description += " - Valid";
    if (elm_is_extended) etype_description += " - Extended";
    if (elm_is_ghost) etype_description += " (Ghost)";

    return etype_description;
}

/**
 * \brief Given the etype, return true if the element is entire, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is entire, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_ENTIRE(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);

    return elm_is_entire;
}

/**
 * \brief Given the etype, return true if the element is partial, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is partial, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_PARTIAL(const short etype)
{
    const bool elm_is_partial = (etype == __DG_ELM_TYPE_PARTIAL__);

    return elm_is_partial;
}

/**
 * \brief Given the etype, return true if the element is large, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is large, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_LARGE(const short etype)
{
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);

    return elm_is_large;
}

/**
 * \brief Given the etype, return true if the element is small, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is small, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_SMALL(const short etype)
{
    const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);

    return elm_is_small;
}

/**
 * \brief Given the etype, return true if the element is large or small, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is large or small, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_SMALL_OR_LARGE(const short etype)
{
    const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);

    return elm_is_small || elm_is_large;
}

/**
 * \brief Given the etype, return true if the element is large, small or extended, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is large, small or extended, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_SMALL_LARGE_OR_EXTENDED(const short etype)
{
    const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_extended = (elm_is_entire || elm_is_large) && (etype/10 == 1);

    return elm_is_small || elm_is_large || elm_is_extended;
}

/**
 * \brief Given the etype, return true if the element is empty, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is empty, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_EMPTY(const short etype)
{
    const bool elm_is_empty = (etype == __DG_ELM_TYPE_EMPTY__);

    return elm_is_empty;
}

/**
 * \brief Given the etype, return true if the element is not empty, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is not empty, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_NOT_EMPTY(const short etype)
{
    const bool elm_is_not_empty = (etype != __DG_ELM_TYPE_EMPTY__);

    return elm_is_not_empty;
}

/**
 * \brief Given the etype, return true if the element is entire or large, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is entire or large, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_VALID(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_not_ghost = (etype/100 != 1);
    const bool elm_is_valid = (elm_is_entire || elm_is_large) && elm_is_not_ghost;

    return elm_is_valid;
}

/**
 * \brief Given the etype, return true if the element is extended, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is extended, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_EXTENDED(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_extended = (elm_is_entire || elm_is_large) && (etype/10 == 1);

    return elm_is_extended;
}

/**
 * \brief Given the etype, return true if the element is large or extended, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is large or extended, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_LARGE_OR_EXTENDED(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
    const bool elm_is_extended = (elm_is_entire || elm_is_large) && (etype/10 == 1);

    return elm_is_large || elm_is_extended;
}

/**
 * \brief Given the etype, return true if the element is entire and non-extended, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element is entire and non-extended, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_STANDARD_CELL(const short etype)
{
    const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
    const bool elm_is_extended = elm_is_entire && (etype/10 == 1);

    return (elm_is_entire && (!elm_is_extended));
}

/**
 * \brief Given the etype, return true if the element stems from a ghost cell, false otherwise.
 *
 * \param[in] etype: element type.
 *
 * \return true if the element stems from a ghost cell, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_IS_GHOST(const short etype)
{
    const bool elm_is_ghost = (etype/100 == 1);

    return elm_is_ghost;
}

/**
 * \brief Given the bou_type, return true if the element's boundary is not empty, false otherwise.
 *
 * \param[in] bou_type: element's boundary type.
 *
 * \return true if the element's boundary is not empty, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_NOT_EMPTY(const short bou_type)
{
    const bool elm_bou_is_not_empty = (bou_type != __DG_ELM_BOU_TYPE_EMPTY__);

    return elm_bou_is_not_empty;
}

/**
 * \brief Given the bou_type, return true if the element's boundary is valid, false otherwise.
 *
 * \param[in] bou_type: element's boundary type.
 *
 * \return true if the element's boundary is valid, false otherwise.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ELM_BOU_IS_VALID(const short bou_type)
{
    const bool elm_bou_is_entire = (bou_type == __DG_ELM_BOU_TYPE_ENTIRE__);
    const bool elm_bou_is_partial = (bou_type == __DG_ELM_BOU_TYPE_PARTIAL__);

    return (elm_bou_is_entire || elm_bou_is_partial);
}

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
    return (val == __DG_CELL_UNMASKED__);
}

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
// ####################################################################



// ####################################################################
// AUXILIARY FUNCTIONS ################################################
// ####################################################################
void InterpolateLevelsets(const int N_PHI,
                          const Geometry & c_geom,
                          const StandardRectangle<AMREX_SPACEDIM> & c_std_elem,
                          const MultiFab & c_PHI,
                          const IntVect rr,
                          const Geometry & f_geom,
                          const StandardRectangle<AMREX_SPACEDIM> & f_std_elem,
                          MultiFab & f_PHI);
// ####################################################################
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

    // LEVEL SETS
    StandardRectangle<AMREX_SPACEDIM> ls_std_elem;

    MultiFab PHI;

    // IMPLICIT MESH INFO AND DATA STRUCTURES
    StandardRectangle<AMREX_SPACEDIM> im_std_elem;

    int space_q_ie;
    Real volume_fraction_threshold;

    Gpu::ManagedVector<int> Dom2Phi;

    shortMultiFab eType;
    Array<shortMultiFab, AMREX_SPACEDIM> eBouType;

    MultiFab eInfo;
    Array<MultiFab, AMREX_SPACEDIM> eBouInfo;
    MultiFab eIntBouInfo;

    iMultiFab eDomQuad_Nq;
    longMultiFab eDomQuad_pos;
    Array<iMultiFab, AMREX_SPACEDIM> eBouQuad_Nq;
    Array<longMultiFab, AMREX_SPACEDIM> eBouQuad_pos;
    iMultiFab eIntBouQuad_Nq;
    longMultiFab eIntBouQuad_pos;

    longMultiFab eFVInfo_pos;

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
    void define(const Geometry & input_geom, const BoxArray & input_ba, const DistributionMapping & input_dm,
                const int grid_space_p, const int grid_space_q,
                const int implicit_mesh_space_q, const int implicit_mesh_space_q_ie, const Real implicit_mesh_VFT)
    {
        // GEOMETRY ===================================================
        this->geom = input_geom;
        this->rel_tol = (this->abs_tol)*(*std::min_element(this->geom.CellSize(),this->geom.CellSize()+AMREX_SPACEDIM));
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        this->cc_ba = input_ba;

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

        this->dm = input_dm;
        // ============================================================

        // LEVEL SETS STANDARD ELEMENT ================================
        this->ls_std_elem.define(this->geom.CellSize(), grid_space_p, grid_space_q);
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->im_std_elem.define(this->geom.CellSize(), 0, implicit_mesh_space_q);

        this->space_q_ie = implicit_mesh_space_q_ie;

        this->volume_fraction_threshold = implicit_mesh_VFT;

        this->n_cut_elms = 0;
        this->n_cut_elm_bou = 0;
        this->n_entire_extended_elms = 0;

        this->mem_for_eDomQuad = 0L;
        this->mem_for_eBouQuad = 0L;
        this->mem_for_eIntBouQuad = 0L;
        // ============================================================
    }
    
    void define(const Geometry & input_geom, const BoxArray & input_ba, const DistributionMapping & input_dm, const InputReader & inputs)
    {
        this->define(input_geom, input_ba, input_dm,
                     inputs.grid.space_p, inputs.grid.space_q,
                     inputs.implicit_mesh.space_q, inputs.implicit_mesh.space_q_ie, inputs.implicit_mesh.VFT);
    }

    void define(const InputReader & inputs)
    {
        // VARIABLES ==================================================
        RealBox rbx;
        Box ibx;
        Geometry input_geom;
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

        input_geom.define(ibx, &rbx, inputs.space.coord_sys, inputs.space.is_periodic.data());
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        BoxArray input_ba(ibx);
        input_ba.maxSize(inputs.grid.max_box_size);

        DistributionMapping input_dm(input_ba);
        // ============================================================

        this->define(input_geom, input_ba, input_dm, inputs);
    }
    // ################################################################

    // MAKE THE IMPLICIT MESH #########################################
    /**
     * \brief Eval and store the implicitly-defined mesh using N_PHI level sets and N_DOM domains.
     *
     * \param[in] N_PHI: number of level sets that will be projected.
     * \param[in] N_DOM: number of domains that will be considered.
     * \param[in] im_flag: implicit mesh flag. If set to false, all cells are considered as entire
     *                     cells.
    */
    void Make(const int N_PHI, const int N_DOM, const bool im_flag = true)
    {
        // PROFILING ------------------------------------------------------
        BL_PROFILE("Make(const int, const int, const bool)");
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

        // LEVEL SET APPROXIMATION ORDER
        const int p = this->ls_std_elem.p;

        // STANDARD ELEMENT QUADRATURE INFO
        const int std_elem_dom_Nq = this->im_std_elem.dom_Nq;
        const int std_elem_bou_Nq = this->im_std_elem.bou_Nq;
    
        // MULTIFAB INFO
        const int eType_n_comp = __DG_ELM_TYPE_N_COMP_PER_DOM__*N_DOM;
        const int eBouType_n_comp = __DG_ELM_BOU_TYPE_N_COMP_PER_DOM__*N_DOM;

        const int eInfo_n_comp = __DG_ELM_INFO_N_COMP_PER_DOM__*N_DOM;
        const int eBouInfo_n_comp = __DG_ELM_BOU_INFO_N_COMP_PER_DOM__*N_DOM;
        //const int eIntBouInfo_n_comp = __DG_ELM_INT_BOU_INFO_N_COMP_PER_DOM__*N_DOM;

        const int eDomQuad_Nq_n_comp = __DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eDomQuad_pos_n_comp = __DG_ELM_DOM_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eBouQuad_Nq_n_comp = __DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eBouQuad_pos_n_comp = __DG_ELM_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eIntBouQuad_Nq_n_comp = __DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;
        const int eIntBouQuad_pos_n_comp = __DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM__*N_DOM;

        const int eFVInfo_pos_n_comp = __DG_ELM_FV_INFO_N_COMP_PER_DOM__*N_DOM;

        // DOMAIN-TO-LEVELSETS CORRESPONDENCE
        int const * Dom2Phi_ptr = this->Dom2Phi.data();

        // IMPLICIT MESH VOLUME FRACTION THRESHOLD
        const Real vft = this->volume_fraction_threshold;

        // IMPLICIT MESH QUADRATURE ORDER
        const int qo = this->space_q_ie+1;

        // NEAREST POINT SEARCH
        //const int it_max = 10;
        //const Real rtol = atol*AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);
        // ============================================================
        
        // VARIABLES ==================================================
        long eDomQuad_mem_pos, eBouQuad_mem_pos, eIntBouQuad_mem_pos;

        charMultiFab merging_helper;
        // ============================================================

        // PRUNING OF THE BOXES =======================================
        // REMOVE ALL BOXES WHOSE ELEMENTS ARE EMPTY
        // (THE ELEMENTS MUST BE EMPTY FOR EACH AND EVERY DOMAIN)
        // ============================================================
        if (im_flag && false)
        {
            BoxList bl;
            //int n_empty_boxes;

            for (MFIter mfi(this->PHI); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.fabbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<Real const> const & PHI_fab = this->PHI.array(mfi);

                bool bx_is_empty;

                bx_is_empty = true;
                for (int dom = 0; dom < N_DOM; ++dom)
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

                    // LEVELSETS
                    algoim_DG_phi_space_BFX phi(p, cell_lo, cell_hi, PHI_fab, i, j, k);
                    phi.set_u(Dom2Phi_ptr[2*dom]);
                    phi.set_sign(Dom2Phi_ptr[2*dom+1]);

                    iQuadRule quad = iQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                    // Compute the volume fraction
#if __cplusplus < 201402L
                    cell_volume = quad([](const Algoim::TinyVector<Real, AMREX_SPACEDIM> & /*x*/) {return 1.0;});
#else
                    cell_volume = quad([](const auto & /*x*/) {return 1.0;});
#endif
                    volume_fraction = cell_volume/entire_cell_volume;

                    if (volume_fraction > atol)
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

            // CREATE THE NEW BOX ARRAY AND DISTRIBUTION MAPPING
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

            // UPDATE THE LEVEL SET MULTIFAB
            {
                const int ngr = this->dom_data_ngr;
                const int n_comp = (this->ls_std_elem.Np)*N_PHI;

                MultiFab real_mf;
                
                real_mf.define(new_cc_ba, new_dm, n_comp, ngr);
                real_mf.ParallelCopy(this->PHI, 0, 0, n_comp, ngr, ngr, this->geom.periodicity());
                this->PHI.define(new_cc_ba, new_dm, n_comp, ngr);
                MultiFab::Copy(this->PHI, real_mf, 0, 0, n_comp, ngr);
            }

            // UPDATE IMPLICIT-MESH BOX ARRAY AND DISTRIBUTION MAPPING
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

        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->eBouInfo[dim].define(this->fc_ba[dim], this->dm, eBouInfo_n_comp, this->bou_data_ngr);
            this->eBouInfo[dim] = 0.0;
        }

        //this->eIntBouInfo.define(this->cc_ba, this->dm, eIntBouInfo_n_comp, this->dom_data_ngr);
        //this->eIntBouInfo = 0.0;

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

        this->eFVInfo_pos.define(this->cc_ba, this->dm, eFVInfo_pos_n_comp, 0);
        this->eFVInfo_pos = -1L;

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
            const int dom_Nq = std_elem_dom_Nq;

            // Resize memory
            this->eDomQuad_host_mem.resize((AMREX_SPACEDIM+1)*dom_Nq);

            // Store data
            for (int q = 0; q < dom_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+0] = this->im_std_elem.dom_XI1q[q];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+1] = this->im_std_elem.dom_XI2q[q];,
                    this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+2] = this->im_std_elem.dom_XI3q[q];
                )
                this->eDomQuad_host_mem[(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = this->im_std_elem.dom_Wq[q];
            }

            // Update memory (nodes coordinates + weights)
            eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
        }

        // BOUNDARY QUADRATURE POINTS
        {
            // QUADRATURE INFO
            const int bou_Nq = std_elem_bou_Nq;

            // Resize memory
            this->eBouQuad_host_mem.resize((AMREX_SPACEDIM+1)*bou_Nq*__DG_STD_ELEM_N_SPACE_BOUNDARIES__);

            // Store data
            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
            for (int q = 0; q < bou_Nq; ++q)
            {
                AMREX_D_TERM
                (
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+0] = this->im_std_elem.bou_XI1q[bou_Nq*b+q];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+1] = this->im_std_elem.bou_XI2q[bou_Nq*b+q];,
                    this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+2] = this->im_std_elem.bou_XI3q[bou_Nq*b+q];
                )
                this->eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*q+AMREX_SPACEDIM] = this->im_std_elem.bou_Wq[bou_Nq*b+q];
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
            //Array4<Real> const & eIntBouInfo_fab = this->eIntBouInfo.array(mfi);
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

                // LEVELSETS
                algoim_DG_phi_space_BFX phi(p, cell_lo, cell_hi, PHI_fab, i, j, k);
                phi.set_u(Dom2Phi_ptr[2*dom]);
                phi.set_sign(Dom2Phi_ptr[2*dom+1]);

                // ELEMENTS' QUADRATURE POINTS
                {
                    // Algoim data structure
                    iQuadRule quad;
                    if (im_flag)
                    {
                        quad = iQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                        // Compute the volume fraction
#if __cplusplus < 201402L
                        cell_volume = quad([](const Algoim::TinyVector<Real, AMREX_SPACEDIM> & /*x*/) {return 1.0;});
#else
                        cell_volume = quad([](const auto & /*x*/) {return 1.0;});
#endif
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
                    else if (std::abs(volume_fraction-1.0) < atol)
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
                        // STORE CENTROID
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) = cell_m[0];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) = cell_m[1];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) = cell_m[2];
                        )
                    }
                    // ------------------------------------------------
                    // ENTIRE ELEMENTS --------------------------------
                    else if (cell_is_entire)
                    {
                        // STORE CENTROID
                        AMREX_D_TERM
                        (
                            eInfo_fab(i,j,k,ELM_CENTROID_X1(dom)) = cell_m[0];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X2(dom)) = cell_m[1];,
                            eInfo_fab(i,j,k,ELM_CENTROID_X3(dom)) = cell_m[2];
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
                
                // INTERNAL BOUNDARY CENTROID (?)
                /*
                if (cell_is_cut)
                {
                    iQuadRule quad = iQuadGen(phi, cell_lo, cell_hi, AMREX_SPACEDIM, -1, 1);

                    GpuArray<Real, AMREX_SPACEDIM> un;
                    Real tmp;

                    const int q = 0;

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
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X1(dom)) = quad.nodes[q].x[0];,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X2(dom)) = quad.nodes[q].x[1];,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X3(dom)) = quad.nodes[q].x[2];
                    )
                    AMREX_D_TERM
                    (
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN1(dom)) = un[0];,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN2(dom)) = un[1];,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN3(dom)) = un[2];
                    )
                }
                else
                {
                    AMREX_D_TERM
                    (
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X1(dom)) = 0.0;,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X2(dom)) = 0.0;,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_CENTROID_X3(dom)) = 0.0;
                    )
                    AMREX_D_TERM
                    (
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN1(dom)) = 0.0;,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN2(dom)) = 0.0;,
                        eIntBouInfo_fab(i,j,k,ELM_INT_BOU_UN3(dom)) = 0.0;
                    )
                }
                */
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
                Array4<Real> const & eBouInfo_fab = this->eBouInfo[dir].array(mfi);
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
                    bool face_is_empty, face_is_entire, face_is_cut;
                    Real boundary_surface, surface_fraction;

                    // LEVELSETS
                    algoim_DG_phi_space_BFX phi(p, cell_lo, cell_hi, PHI_fab, i, j, k);
                    phi.set_u(Dom2Phi_ptr[2*dom]);
                    phi.set_sign(Dom2Phi_ptr[2*dom+1]);

                    // BOUNDARY QUADRATURE POINTS
                    {
                        // Algoim data structure
                        iQuadRule quad;
                        if (im_flag)
                        {
                            quad = iQuadGen(phi, cell_lo, cell_hi, b/2, b%2, qo);

                            // Compute the surface fraction
#if __cplusplus < 201402L
                            boundary_surface = quad([](const Algoim::TinyVector<Real, AMREX_SPACEDIM> & /*x*/) {return 1.0;});
#else
                            boundary_surface = quad([](const auto & /*x*/) {return 1.0;});
#endif
                            surface_fraction = boundary_surface/entire_boundary_surface;
                        }
                        else
                        {
                            surface_fraction = 1.0;
                        }

                        // EMPTY BOUNDARY -----------------------------
                        if (surface_fraction < atol)
                        {
                            face_is_empty = true;
                            face_is_entire = false;
                            face_is_cut = false;

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
                            face_is_empty = false;
                            face_is_entire = true;
                            face_is_cut = false;

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
                            face_is_empty = false;
                            face_is_entire = false;
                            face_is_cut = true;

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
                                    this->eBouQuad_host_mem[pos+0] = (quad.nodes[q].x[0]-cell_lo[0]);,
                                    this->eBouQuad_host_mem[pos+1] = (quad.nodes[q].x[1]-cell_lo[1]);,
                                    this->eBouQuad_host_mem[pos+2] = (quad.nodes[q].x[2]-cell_lo[2]);
                                )
                                this->eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[q].w;
                            }

                            // Update memory (nodes coordinates + weights)
                            eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;
                        }
                        // --------------------------------------------
                    }

                    // BOUNDARY CENTROID
                    {
                        // EMPTY BOUNDARY -----------------------------
                        if (face_is_empty) 
                        {
                            // STORE CENTROID
                            AMREX_D_TERM
                            (
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X1(dom)) = cell_lo[0]+0.5*(dx[0]-delta[0]);,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X2(dom)) = cell_lo[1]+0.5*(dx[1]-delta[1]);,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X3(dom)) = cell_lo[2]+0.5*(dx[2]-delta[2]);
                            )
                        }
                        // --------------------------------------------
                        // ENTIRE BOUNDARY ----------------------------
                        else if (face_is_entire)
                        {
                            // STORE CENTROID
                            AMREX_D_TERM
                            (
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X1(dom)) = cell_lo[0]+0.5*(dx[0]-delta[0]);,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X2(dom)) = cell_lo[1]+0.5*(dx[1]-delta[1]);,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X3(dom)) = cell_lo[2]+0.5*(dx[2]-delta[2]);
                            )
                        }
                        // --------------------------------------------
                        // PARTIAL BOUNDARY ---------------------------
                        else if (face_is_cut)
                        {
                            const int bou_Nq = eBouQuad_Nq_fab(i,j,k,ELM_BOU_PLUS_QUAD_NQ(dom));
                            const long pos0 = eBouQuad_pos_fab(i,j,k,ELM_BOU_PLUS_QUAD_POS(dom));

                            // VARIABLES
                            Real x[AMREX_SPACEDIM], w, S[AMREX_SPACEDIM], face_measure;

                            // EVAL THE STATIC MOMENTS
                            AMREX_D_TERM
                            (
                                S[0] = 0.0;,
                                S[1] = 0.0;,
                                S[2] = 0.0;
                            )
                            face_measure = 0.0;

                            for (int q = 0; q < bou_Nq; ++q)
                            {
                                const long pos = pos0+(AMREX_SPACEDIM+1)*q;
                                AMREX_D_TERM
                                (
                                    x[0] = this->eBouQuad_host_mem[pos+0]+cell_lo[0];,
                                    x[1] = this->eBouQuad_host_mem[pos+1]+cell_lo[1];,
                                    x[2] = this->eBouQuad_host_mem[pos+2]+cell_lo[2];
                                )
                                w = this->eBouQuad_host_mem[pos+AMREX_SPACEDIM];
                                AMREX_D_TERM
                                (
                                    S[0] += x[0]*w;,
                                    S[1] += x[1]*w;,
                                    S[2] += x[2]*w;
                                )
                                face_measure += w;
                            }

                            // EVAL AND STORE CENTROID
                            AMREX_D_TERM
                            (
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X1(dom)) += S[0]/face_measure;,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X2(dom)) += S[1]/face_measure;,
                                eBouInfo_fab(i,j,k,ELM_BOU_CENTROID_X3(dom)) += S[2]/face_measure;
                            )
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
        merging_helper.define(this->cc_ba, this->dm, 1, this->dom_data_ngr);
        merging_helper = 0;

        // MARK THOSE ELEMENTS WHICH CAN BE EXTENDED
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            Array4<char> const & helper_fab = merging_helper.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                helper_fab(i,j,k,dom) = 1;
            });
            Gpu::synchronize();
        }
        merging_helper.FillBoundary(this->geom.periodicity());

        // MARK THE GHOST CELLS
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            
            Array4<char const> const & helper_fab = merging_helper.array(mfi);
            Array4<short> const & eType_fab = this->eType.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                if (helper_fab(i,j,k,dom) == 0)
                {
                    const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                    const bool elm_is_entire = ELM_IS_ENTIRE(etype);
                    const bool elm_is_partial = ELM_IS_PARTIAL(etype);

                    if (elm_is_entire)
                    {
                        eType_fab(i,j,k,ELM_TYPE(dom)) = 100+__DG_ELM_TYPE_ENTIRE__;
                    }
                    else if (elm_is_partial)
                    {
                        eType_fab(i,j,k,ELM_TYPE(dom)) = 100+__DG_ELM_TYPE_LARGE__;
                    }

                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());

        // SETUP MERGING INFO
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short> const & eType_fab = this->eType.array(mfi);
            AMREX_D_TERM
            (
                Array4<short> const & eBouType_fab_0 = this->eBouType[0].array(mfi);,
                Array4<short> const & eBouType_fab_1 = this->eBouType[1].array(mfi);,
                Array4<short> const & eBouType_fab_2 = this->eBouType[2].array(mfi);
            )
            
            Array4<Real const> const & eInfo_fab = this->eInfo.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const bool elm_is_partial = ELM_IS_PARTIAL(eType_fab(i,j,k,ELM_TYPE(dom)));
                
                const Real vf = eInfo_fab(i,j,k,ELM_VOLUME_FRACTION(dom));

                // LOCAL VARIABLES
                int nbr_i, nbr_j, nbr_k, nbr_b;
                Real nbr_vf, best_nbr_vf;
                bool elm_has_been_merged, nbr_is_not_ghost;

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

                        //nbr_is_valid = domain.contains(nbr_i, nbr_j, nbr_k);
                        nbr_is_not_ghost = (!ELM_IS_GHOST(eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom))));

                        nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,ELM_VOLUME_FRACTION(dom));

                        if ((nbr_vf > best_nbr_vf) && nbr_is_not_ghost)
                        {
                            eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                            elm_has_been_merged = true;
                            best_nbr_vf = nbr_vf;
                        }
                    }

/*
if (i == 40 && j == 36 && k == 27)
{
const int r = ParallelDescriptor::MyProc();

const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

Print(r) << "HELLO " << std::endl;
Print(r) << "etype: " << ELM_TYPE_DESCRIPTION(etype) << std::endl;

BF_CELL(i, j, k, etype, nbr_i, nbr_j, nbr_k);

Print(r) << "nbr: (" << nbr_i << "," << nbr_j << "," << nbr_k << ")" << std::endl;

Print(r) << "domain.contains(nbr): " << domain.contains(nbr_i, nbr_j, nbr_k) << std::endl;
Print(r) << "fabbox.contains(nbr): " << fbx.contains(nbr_i, nbr_j, nbr_k) << std::endl;

Print(r) << "fbx: " << fbx << std::endl;

exit(-1);
}
*/

/*
if (j > i)
{
    const int b = 2;
    eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
}
else
{
    const int b = 0;
    eType_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
}
*/

                    // MARK THE BOUNDARY AS INVALID
                    if (elm_has_been_merged)
                    {
                        const int b = (eType_fab(i,j,k,ELM_TYPE(dom))/10);
                        int fi, fj, fk;

                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        if ((b == 0) || (b == 1))
                        {
                            eBouType_fab_0(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                        }
#if (AMREX_SPACEDIM > 1)
                        if ((b == 2) || (b == 3))
                        {
                            eBouType_fab_1(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                        }
#endif
#if (AMREX_SPACEDIM > 2)
                        if ((b == 4) || (b == 5))
                        {
                            eBouType_fab_2(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                        }
#endif

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
                            if (bx.contains(nbr_i,nbr_j,nbr_k))
                            {
                                printf("| Volume fraction (nbr = %d): %f\n", b, nbr_vf);
                            }
                            else
                            {
                                printf("| Volume fraction (nbr = %d -- in fabbox): %f\n", b, nbr_vf);
                            }
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

        // MARK THE EXTENDED ELEMENT AND THE DISAPPEARED BOUNDARIES AS
        // INVALID (FROM EXTENDED ELEMENTS VIEWPOINT)
        // This is needed because merging can happen across the boxes
        // and, up to now, the disappeared boundaries have been marked
        // only on the boxes containing the small elements.
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short> const & eType_fab = this->eType.array(mfi);
            AMREX_D_TERM
            (
                Array4<short> const & eBouType_fab_0 = this->eBouType[0].array(mfi);,
                Array4<short> const & eBouType_fab_1 = this->eBouType[1].array(mfi);,
                Array4<short> const & eBouType_fab_2 = this->eBouType[2].array(mfi);
            )

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);

                if (elm_is_valid)
                {
                    // Search among the neighbors
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        int nbr_i, nbr_j, nbr_k, nbr_b;

                        // Neighboring cell id and boundary
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        const bool nbr_is_small = ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                        const bool cells_are_merged = (nbr_merged_b == nbr_b);

                        if (cells_are_merged)
                        {
                            // MARK THE ELEMENT AS EXTENDED
                            eType_fab(i,j,k,ELM_TYPE(dom)) = etype+10;

                            // MARK THE BOUNDARY AS INVALID
                            int fi, fj, fk;
                            GRID_FACE(i, j, k, b, fi, fj, fk);
                            
                            if ((b == 0) || (b == 1))
                            {
                                eBouType_fab_0(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                            }
#if (AMREX_SPACEDIM > 1)
                            if ((b == 2) || (b == 3))
                            {
                                eBouType_fab_1(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                            }
#endif
#if (AMREX_SPACEDIM > 2)
                            if ((b == 4) || (b == 5))
                            {
                                eBouType_fab_2(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                            }
#endif
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        this->eType.FillBoundary(this->geom.periodicity());
        // ============================================================

        // MARK GHOST CELLS AS ENTIRE OR LARGE CELLS ==================
        /*
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            Array4<short> const & eType_fab = this->eType.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const bool elm_is_ghost = ELM_IS_GHOST(eType_fab(i,j,k,ELM_TYPE(dom)));
                const bool elm_is_small = ELM_IS_SMALL(eType_fab(i,j,k,ELM_TYPE(dom)));

                if (elm_is_ghost && elm_is_small)
                {
                    if (elm_is_entire)
                    {
                        eType_fab(i,j,k,ELM_TYPE(dom)) = 100+__DG_ELM_TYPE_ENTIRE__;
                    }
                    else if (elm_is_small_or_large)
                    {
                        eType_fab(i,j,k,ELM_TYPE(dom)) = 100+__DG_ELM_TYPE_LARGE__;
                    }
                }
            });
            Gpu::synchronize();
        }
        */
        // ============================================================

        // CHECK THE CONSISTENCY OF THE MERGING INFORMATION ===========
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Box & fbx = mfi.fabbox();

            Array4<short const> const & eType_fab = this->eType.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);

                if (elm_is_valid)
                {
                    // LOCAL VARIABLES
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    short nbr_etype;
                    int nbr_BF_i, nbr_BF_j, nbr_BF_k;

                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);

                        if (!fbx.contains(nbr_BF_i, nbr_BF_j, nbr_BF_k))
                        {
#if (AMREX_USE_CUDA)
                            Abort("ERROR: IBVP_ImplicitMesh.H - ImplicitMesh::Make: Bad merging!");
#else
                            std::string msg;
                            msg  = "\n";
                            msg += "ERROR: IBVP_utils.H - ImplicitMesh::Make\n";
                            msg += "| The merging connects cells in diffent fab boxes.\n";
                            msg += "| cell: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                            msg += "| b: "+std::to_string(b)+"\n";
                            msg += "| nbr_cell   : "+std::to_string(nbr_i)+","+std::to_string(nbr_j)+","+std::to_string(nbr_k)+"\n";
                            msg += "| nbr_BF_cell: "+std::to_string(nbr_BF_i)+","+std::to_string(nbr_BF_j)+","+std::to_string(nbr_BF_k)+"\n";
                            Abort(msg);
#endif
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        // ============================================================

        // UPDATE THE CENTROIDS OF THE EXTENDED ELEMENTS ==============
        for (MFIter mfi(this->eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = this->eType.array(mfi);

            Array4<Real> const & eInfo_fab = this->eInfo.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);

                // LOCAL VARIABLES
                short nbr_etype;
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                bool nbr_is_small;

                // Cycle over the boundaries and add the small element
                // contribution to the centroid.
                if (elm_is_extended)
                {
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                        nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        if (nbr_merged_b == nbr_b)
                        {
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
                }
            });
            Gpu::synchronize();
        }
        this->eInfo.FillBoundary();
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
        const int PHI_n_comp = (this->ls_std_elem.Np)*N_PHI;
        const int PHI_ngr = this->dom_data_ngr;
        // ============================================================
        
        // PROJECT THE LEVEL SETS OVER THE GRID =======================
        // INIT THE MULTIFAB THAT WILL CONTAIN THE PROJECTED LEVELSETS
        this->PHI.define(this->cc_ba, this->dm, PHI_n_comp, PHI_ngr);

        // EVAL THE PROJECTION
        ProjectLevelsetsOverGrid(t, this->geom, this->ls_std_elem, N_PHI, this->PHI, IBVP);
        // ============================================================

        // SET THE CORRESPONDENCE BETWEEN DOMAINS AND LEVELSETS =======
        this->Dom2Phi.resize(2*N_DOM);
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            int phi_info[2];
            IBVP.F_DOM2PHI(dom, phi_info);

            this->Dom2Phi[2*dom] = phi_info[0];
            this->Dom2Phi[2*dom+1] = phi_info[1];
        }
        // ============================================================

        // MAKE THE IMPLICIT MESH USING THE PROJECTED LEVEL SETS ======
        this->Make(N_PHI, N_DOM, im_flag);
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

    // MAKE THE IMPLICIT MESH FROM A COARSE ONE #######################
    /**
     * \brief Make the implicit mesh for a specified ba and dm and an input coarse one.
    */
    template <typename IBVP_CLASS>
    void MakeFromCoarse(const Real /*t*/,
                        const BoxArray & input_ba, const DistributionMapping & input_dm,
                        const IntVect & c_rr, const ImplicitMesh & c_mesh,
                        const int N_PHI, const int N_DOM, const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        // PARAMETERS =================================================
        const int PHI_n_comp = (this->ls_std_elem.Np)*N_PHI;
        const int PHI_ngr = this->dom_data_ngr;
        // ============================================================

        // VARIABLES ==================================================
        this->PHI.define(input_ba, input_dm, PHI_n_comp, PHI_ngr);
        this->PHI = 0.0;
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        this->cc_ba = input_ba;

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

        this->dm = input_dm;
        // ============================================================

        // INIT THE INTERPOLATION OPERATOR FOR THE STANDARD ELEMENT ===
        this->ls_std_elem.InitInterpolation(c_mesh.ls_std_elem, c_rr);
        // ============================================================

        // LEVEL SET FUNCTIONS ========================================
        InterpolateLevelsets(N_PHI, c_mesh.geom, c_mesh.ls_std_elem, c_mesh.PHI, c_rr,
                             this->geom, this->ls_std_elem, this->PHI);

        // SET THE CORRESPONDENCE BETWEEN DOMAINS AND LEVELSETS -------
        this->Dom2Phi.resize(2*N_DOM);
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            int phi_info[2];
            IBVP.F_DOM2PHI(dom, phi_info);

            this->Dom2Phi[2*dom] = phi_info[0];
            this->Dom2Phi[2*dom+1] = phi_info[1];
        }
        // ------------------------------------------------------------

        // MAKE THE IMPLICIT MESH USING THE NEW LEVEL SETS
        this->Make(N_PHI, N_DOM, im_flag);
        // -----------------------------------------------
        // ============================================================
    }

    /**
     * \brief Make the implicit mesh for a specified ba and dm and an input coarse one.
    */
    template <typename IBVP_CLASS>
    void MakeFromCoarse(const Real t,
              const BoxArray & input_ba, const DistributionMapping & input_dm,
              const IntVect & c_rr, const ImplicitMesh & c_mesh,
              const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        const int N_PHI = 1;
        const int N_DOM = 1;
        this->MakeFromCoarse(t, input_ba, input_dm, c_rr, c_mesh, N_PHI, N_DOM, IBVP, im_flag);
    }
    // ################################################################

    // REMAKE THE IMPLICIT MESH #######################################
    /**
     * \brief Remake the implicit mesh for a specified ba and dm.
    */
    template <typename IBVP_CLASS>
    void Remake(const Real /*t*/,
                const BoxArray & input_ba, const DistributionMapping & input_dm,
                const IntVect & c_rr, const ImplicitMesh & c_mesh,
                const int N_PHI, const int N_DOM, const IBVP_CLASS & /*IBVP*/, const bool im_flag = true)
    {
        // PARAMETERS =================================================
        const int n_comp = this->PHI.n_comp;
        const IntVect n_grow = this->PHI.n_grow;
        // ============================================================

        // VARIABLES ==================================================
        MultiFab new_PHI(input_ba, input_dm, n_comp, n_grow);
        new_PHI = 0.0;
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        this->cc_ba = input_ba;

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

        this->dm = input_dm;
        // ============================================================

        // INIT THE INTERPOLATION OPERATOR FOR THE STANDARD ELEMENT ===
        this->ls_std_elem.InitInterpolation(c_mesh.ls_std_elem, c_rr);
        // ============================================================

        // LEVEL SET FUNCTIONS ========================================
        if (new_PHI.getBDKey() != this->PHI.getBDKey())
        {
            InterpolateLevelsets(N_PHI, c_mesh.geom, c_mesh.ls_std_elem, c_mesh.PHI, c_rr,
                                 this->geom, this->ls_std_elem, new_PHI);

            new_PHI.ParallelCopy(this->PHI, 0, 0, n_comp, 0, 0, this->geom.periodicity());
            new_PHI.FillBoundary(this->geom.periodicity());

            std::swap(this->PHI, new_PHI);

            // MAKE THE IMPLICIT MESH USING THE NEW LEVEL SETS
            this->Make(N_PHI, N_DOM, im_flag);
            // -----------------------------------------------
        }
        // ============================================================
    }

    /**
     * \brief Remake the implicit mesh for a specified ba and dm.
    */
    template <typename IBVP_CLASS>
    void Remake(const Real t,
                const BoxArray & input_ba, const DistributionMapping & input_dm,
                const IntVect & c_rr, const ImplicitMesh & c_mesh,
                const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        const int N_PHI = 1;
        const int N_DOM = 1;
        this->Remake(t, input_ba, input_dm, c_rr, c_mesh, N_PHI, N_DOM, IBVP, im_flag);
    }
    // ################################################################

    // COMPUTE GEOMETRICAL PROPERTIES #################################
    /**
     * \brief Compute the volume and surface of each domain.
     *
     * \param[in] N_DOM: number of domains to be considered.
     * \param[in] tags: a iMultiFab object that contains a single value for each cell.
     * \param[out] computed_volume: computed_volume[dom] will contain the computed volume of the dom-th
     *                              domain.
     * \param[out] computed_volume_div: computed_volume[dom] will contain the computed volume of the
     *                                  dom-th domain using the boundary integration points.
     * \param[out] computed_surface: computed_surface[dom] will contain the computed surface of the
     *                               dom-th domain.
     *
    */
    void EvalVolumesAndSurfaces(const int N_DOM,
                                const iMultiFab & tags,
                                Real * computed_volume,
                                Real * computed_volume_div,
                                Real * computed_surface) const
    {
        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->geom.ProbLoArray();

        // QUAD CHECK MULTIFAB
        const int n_comp_per_dom = 3;
        // ============================================================

        // VARIABLES ==================================================
#ifdef AMREX_USE_CUDA
        Real const * eDomQuad_mem_ptr = this->eDomQuad_device_mem.data();
        Real const * eBouQuad_mem_ptr = this->eBouQuad_device_mem.data();
        Real const * eIntBouQuad_mem_ptr = this->eIntBouQuad_device_mem.data();
#else
        Real const * eDomQuad_mem_ptr = this->eDomQuad_host_mem.data();
        Real const * eBouQuad_mem_ptr = this->eBouQuad_host_mem.data();
        Real const * eIntBouQuad_mem_ptr = this->eIntBouQuad_host_mem.data();
#endif

        MultiFab quad_check;
        // ============================================================

        // INITIALIZATION =============================================
        quad_check.define(this->cc_ba, this->dm, n_comp_per_dom*N_DOM, this->dom_data_ngr);
        quad_check = 0.0;
        // ============================================================

        // COMPUTE VOLUME/SURFACE OF EACH DOMAIN ======================
        for (MFIter mfi(quad_check); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eDomQuad_Nq_fab = this->eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = this->eDomQuad_pos.array(mfi);
            AMREX_D_TERM
            (
                Array4<int const> const & eBouQuad_Nq_fab_0 = this->eBouQuad_Nq[0].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_0 = this->eBouQuad_pos[0].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_1 = this->eBouQuad_Nq[1].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_1 = this->eBouQuad_pos[1].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_2 = this->eBouQuad_Nq[2].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_2 = this->eBouQuad_pos[2].array(mfi);
            )
            Array4<int const> const & eIntBouQuad_Nq_fab = this->eIntBouQuad_Nq.array(mfi);
            Array4<long const> const & eIntBouQuad_pos_fab = this->eIntBouQuad_pos.array(mfi);

            Array4<int const> const & tags_fab = tags.array(mfi);

            Array4<Real> const & quad_check_fab = quad_check.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // TAG
                if (CELL_IS_NOT_MASKED(tags_fab(i,j,k)))
                {
                    // LOCAL PARAMETERS
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};

                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                    Real integrand;
                    Real elm_volume, elm_volume_div, elm_surface;

                    // INITIALIZATION
                    elm_volume = 0.0;
                    elm_volume_div = 0.0;
                    elm_surface = 0.0;

                    // DOMAIN QUADRATURE
                    {
                        const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                        const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                        const Real * xptr = &eDomQuad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < dom_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0]+cell_lo[0];,
                                x[1] = xptr[x_pos+1]+cell_lo[1];,
                                x[2] = xptr[x_pos+2]+cell_lo[2];
                            )
                            w = xptr[x_pos+AMREX_SPACEDIM];

                            integrand = 1.0;
                            elm_volume += integrand*w;
                        }
                    }

                    // BOUNDARY QUADRATURE
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        const int dir = b/2;
                        int fi, fj, fk;
                        int bou_Nq;
                        long pos;

                        // WHICH FACE
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        // QUADRATURE INFO
                        if (dir == 0)
                        {
                            bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_0(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                        }
#if (AMREX_SPACEDIM > 1)
                        else if (dir == 1)
                        {
                            bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_1(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                        }
#endif
#if (AMREX_SPACEDIM > 2)
                        else
                        {
                            bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,ELM_BOU_QUAD_NQ(dom, b));
                            pos = eBouQuad_pos_fab_2(fi,fj,fk,ELM_BOU_QUAD_POS(dom, b));
                        }
#endif
                        const Real * xptr = &eBouQuad_mem_ptr[pos];

                        // UNIT NORMAL
                        AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                        un[dir] = 2.0*(b%2)-1.0;

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0]+cell_lo[0];,
                                x[1] = xptr[x_pos+1]+cell_lo[1];,
                                x[2] = xptr[x_pos+2]+cell_lo[2];
                            )
                            w = xptr[x_pos+AMREX_SPACEDIM];
                            
                            integrand = x[0]*un[0];
                            elm_volume_div += integrand*w;
                        }
                    }

                    // INTERNAL BOUNDARY QUADRATURE
                    {
                        const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,ELM_INT_BOU_QUAD_NQ(dom));
                        const long pos = eIntBouQuad_pos_fab(i,j,k,ELM_INT_BOU_QUAD_POS(dom));
                        const Real * xptr = &eIntBouQuad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0];,
                                x[1] = xptr[x_pos+1];,
                                x[2] = xptr[x_pos+2];
                            )
                            w = xptr[x_pos+AMREX_SPACEDIM];

                            // UNIT NORMAL
                            AMREX_D_TERM
                            (
                                un[0] = xptr[x_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = xptr[x_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = xptr[x_pos+AMREX_SPACEDIM+1+2];
                            )

                            integrand = x[0]*un[0];
                            elm_volume_div += integrand*w;

                            integrand = 1.0;
                            elm_surface += integrand*w;
                        }
                    }

                    // STORE COMPUTED VALUES
                    quad_check_fab(i,j,k,n_comp_per_dom*dom) = elm_volume;
                    quad_check_fab(i,j,k,n_comp_per_dom*dom+1) = elm_volume_div;
                    quad_check_fab(i,j,k,n_comp_per_dom*dom+2) = elm_surface;
                }
            });
            Gpu::synchronize();
        }
        // ============================================================

        // SUM CONTRIBUTION ===========================================
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            computed_volume[dom] = quad_check.sum(n_comp_per_dom*dom);
            computed_volume_div[dom] = quad_check.sum(n_comp_per_dom*dom+1);
            computed_surface[dom] = quad_check.sum(n_comp_per_dom*dom+2);
        }
        // ============================================================
    }

    /**
     * \brief Compute the volume and surface of each domain.
     *
     * \param[in] N_DOM: number of domains to be considered.
     * \param[out] computed_volume: computed_volume[dom] will contain the computed volume of the dom-th
     *                              domain.
     * \param[out] computed_volume_div: computed_volume[dom] will contain the computed volume of the
     *                                  dom-th domain using the boundary integration points.
     * \param[out] computed_surface: computed_surface[dom] will contain the computed surface of the
     *                               dom-th domain.
     *
    */
    void EvalVolumesAndSurfaces(const int N_DOM,
                                Real * computed_volume,
                                Real * computed_volume_div,
                                Real * computed_surface) const
    {
        iMultiFab mask(this->PHI.boxarray, this->PHI.distributionMap, 1, this->PHI.n_grow);
        mask = __DG_CELL_UNMASKED__;

        this->EvalVolumesAndSurfaces(N_DOM, mask, computed_volume, computed_volume_div, computed_surface);
    }
    // ################################################################

    // CHECK THE QUADRATURE RULES #####################################
    /**
     * \brief Check the quadrature rules by comparing the domains' volume and surface with exact values.
     *
     * \param[in] N_DOM: number of domains to be considered.
     * \param[in] exact_volume: exact_volume[dom] should contain the exact value of the volume of the
     *                          dom-th domain.
     * \param[in] exact_surface: exact_surface[dom] should contain the exact value of the surface of the
     *                           dom-th domain.
     *
    */
    void CheckQuadratureRules(const int N_DOM,
                              const Real * exact_volume,
                              const Real * exact_surface) const
    {
        // VARIABLES ==================================================
        Vector<Real> computed_volume(N_DOM);
        Vector<Real> computed_volume_div(N_DOM);
        Vector<Real> computed_surface(N_DOM);
        // ============================================================

        // COMPUTE VALUES ==============================================
        this->EvalVolumesAndSurfaces(N_DOM,
                                     computed_volume.data(),
                                     computed_volume_div.data(),
                                     computed_surface.data());
        // ============================================================

        // PRINT REPORT ===============================================
        Print() << "IMPLICIT-MESH QUADRATURE REPORT:" << std::endl;
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            Real err;

            Print() << "| Domain " << dom << std::endl;
            
            Print() << "|  Volume: " << std::endl;
            
            Print() << "|  - Reference: " << exact_volume[dom] << std::endl;

            err = 100.0*std::abs(computed_volume[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via dom. quadrature: " << computed_volume[dom] << ", error (%): " << err << std::endl;
            
            err = 100.0*std::abs(computed_volume_div[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via bou. quadrature: " << computed_volume_div[dom] << ", error (%): " << err << std::endl;

            Print() << "|  Surface: " << std::endl;

            if (exact_surface[dom] == 0.0)
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = std::abs(computed_surface[dom]-exact_surface[dom]);
                Print() << "|  - Computed: " << computed_surface[dom] << ", error: " << err << std::endl;
            }
            else
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = 100.0*std::abs(computed_surface[dom]-exact_surface[dom])/exact_surface[dom];
                Print() << "|  - Computed: " << computed_surface[dom] << ", error (%): " << err << std::endl;
            }
        }
        Print() << std::endl;
        // ============================================================
    }

    /**
     * \brief Check the quadrature rules by comparing the domains' volume and surface with exact values.
     *
     * \param[in] exact_volume: exact value of the volume of the domain.
     * \param[in] exact_surface: exact value of the surface of the domain.
     *
    */
    void CheckQuadratureRules(const Real exact_volume,
                              const Real exact_surface) const
    {
        const int N_DOM = 1;
        
        this->CheckQuadratureRules(N_DOM, &exact_volume, &exact_surface);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################
} // namespace DG
} // namespace amrex

#endif