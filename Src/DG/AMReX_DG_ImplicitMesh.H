//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_ImplicitMesh.H
 * \brief Contains constants, data structures and functions to handle the implicitly defined-mesh.
*/

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
// ####################################################################


} // namespace DG
} // namespace amrex

#endif