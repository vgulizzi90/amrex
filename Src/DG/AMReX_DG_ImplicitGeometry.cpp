// AMReX_DG_ImplicitGeometry.cpp

#include <AMReX_DG_ImplicitGeometry.H>

namespace amrex
{
namespace DG
{

// AUXILIARY FUNCTIONS ################################################
// ELEMENT INFORMATION ================================================
void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<int const> const & eType_fab)
{
    // PARAMETERS
    const int etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
    const bool elm_is_undefined = (etype == DG_ELM_TYPE_UNDEFINED);
    const bool elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
    const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
    const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
    const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
    const bool elm_is_partial = (etype == DG_ELM_TYPE_PARTIAL);
    const bool elm_is_valid = elm_is_entire || elm_is_large;
    const bool elm_is_extended = elm_is_valid && (etype/10 == 1);
    const int merged_b = (elm_is_small) ? (etype/10) : -1;

    // VARIABLES
    Real cell_lo[AMREX_SPACEDIM], cell_hi[AMREX_SPACEDIM];
    int BF_i, BF_j, BF_k;
    Real BF_lo[AMREX_SPACEDIM];
    Real BF_hi[AMREX_SPACEDIM];
    std::string etype_description("");

    if (elm_is_undefined) etype_description = "Undefined";
    if (elm_is_empty) etype_description = "Empty";
    if (elm_is_entire) etype_description = "Entire";
    if (elm_is_large) etype_description = "Large";
    if (elm_is_small) etype_description = "Small";
    if (elm_is_partial) etype_description = "Partial";
    if (elm_is_valid) etype_description += " - Valid";
    if (elm_is_extended) etype_description += " - Extended";

    AMREX_D_TERM
    (
        cell_lo[0] = prob_lo[0]+i*dx[0];
        cell_hi[0] = prob_lo[0]+(i+1)*dx[0];,
        cell_lo[1] = prob_lo[1]+j*dx[1];
        cell_hi[1] = prob_lo[1]+(j+1)*dx[1];,
        cell_lo[2] = prob_lo[2]+k*dx[2];
        cell_hi[2] = prob_lo[2]+(k+1)*dx[2];
    )

    if (elm_is_small)
    {
        BF_CELL(prob_lo, dx,
                i, j, k, etype,
                BF_i, BF_j, BF_k,
                BF_lo, BF_hi);
    }

    // REPORT
    Print() << std::endl;
    Print() << " Element (" << i << "," << j << "," << k << "," << dom << ") info:" << std::endl;
#if (AMREX_SPACEDIM == 1)
    Print() << " | cell_lo: " << cell_lo[0] << std::endl;
    Print() << " | cell_hi: " << cell_hi[0] << std::endl;
#endif
#if (AMREX_SPACEDIM == 2)
    Print() << " | cell_lo: " << cell_lo[0] << "," << cell_lo[1] << std::endl;
    Print() << " | cell_hi: " << cell_hi[0] << "," << cell_hi[1] << std::endl;
#endif
#if (AMREX_SPACEDIM == 3)
    Print() << " | cell_lo: " << cell_lo[0] << "," << cell_lo[1] << "," << cell_lo[2] << std::endl;
    Print() << " | cell_hi: " << cell_hi[0] << "," << cell_hi[1] << "," << cell_hi[2] << std::endl;
#endif
    Print() << " | etype: " << etype_description << std::endl;
    if (elm_is_small)
        Print() << " | merging element (through boundary " << merged_b << "): (" << BF_i << "," << BF_j << "," << BF_k << "," << dom << ")" << std::endl;
}

void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<int const> const & eType_fab, Array4<Real const> const & eInfo_fab)
{
    // PARAMETERS
    const Real vf = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom);
    const Real v = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1);

    PrintElementInfo(prob_lo, dx, i, j, k, dom, eType_fab);
    Print() << " | volume fraction: " << vf << std::endl;
    Print() << " | volume: " << v << std::endl;
}
// ====================================================================

// GRID INFO ==========================================================
AMREX_GPU_HOST_DEVICE
void NBR_CELL(const int & i, const int & j, const int & k, const int & b,
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

AMREX_GPU_HOST_DEVICE
void GRID_FACE(const int & i, const int & j, const int & k, const int & b,
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

AMREX_GPU_HOST_DEVICE
void FACE_2_NBRS(const int & i, const int & j, const int & k, const int & dir,
                 int & min_i, int & min_j, int & min_k,
                 int & pls_i, int & pls_j, int & pls_k)
{
    // Given an edge identified by the face-centered indexing tuple
    // (i,j,k), the neighboring elements that share that face are
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
    //            (elm^-)  (i,j,k)  (elm^+)
    //                        |
    //                    <---|
    //          un = {-1,0,0} |
    //       -----------------+-----------------
    //                        |

    // dir = 0 <-> min_i = i-1, min_j = j,   min_k = k
    // dir = 1 <-> min_i = i,   min_j = j-1, min_k = k
    // dir = 2 <-> min_i = i,   min_j = j,   min_k = k-1
    // dir = 0 <-> pls_i = i,   pls_j = j,   pls_k = k
    // dir = 1 <-> pls_i = i,   pls_j = j,   pls_k = k
    // dir = 2 <-> pls_i = i,   pls_j = j,   pls_k = k
    min_i = i-(dir-1)*(dir-2)/2;
    min_j = j-(2-dir)*dir;
    min_k = k-(dir-1)*dir/2;
    pls_i = i;
    pls_j = j;
    pls_k = k;
}

AMREX_GPU_HOST_DEVICE
void BF_CELL(const int & i, const int & j, const int & k, const int & etype,
             int & BF_i, int & BF_j, int & BF_k)
{
    const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);

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

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real* prob_lo, const Real* dx,
             const int & i, const int & j, const int & k, const int & etype,
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

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real * prob_lo, const Real * dx,
             const int & i, const int & j, const int & k, const int & etype,
             Real * BF_lo, Real * BF_hi)
{
    int BF_i, BF_j, BF_k;
    BF_CELL(prob_lo, dx, i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
}
// ====================================================================
// ####################################################################


} // namespace DG
} // namespace amrex