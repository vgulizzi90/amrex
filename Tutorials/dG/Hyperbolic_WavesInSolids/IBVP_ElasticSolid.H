//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_ElasticSolid.H
 * \brief Contains base constants and data structures for IBVP of elastic solids.
*/

#include "../BaseSolidMechanics/IBVP_LinearElasticity.H"

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define NV AMREX_SPACEDIM
#define NS AMREX_D_PICK(1, 3, 6)
#define N_VS (NV+NS)

// SOLUTION FIELDS ORDER
#if (AMREX_SPACEDIM == 1)
#define V1 0
#define E11 1
#define S11 1
#endif
#if (AMREX_SPACEDIM == 2)
#define V1 0
#define V2 1
#define E11 2
#define E22 3
#define E12 4
#define S11 2
#define S22 3
#define S12 4
#endif
#if (AMREX_SPACEDIM == 3)
#define V1 0
#define V2 1
#define V3 2
#define E11 3
#define E22 4
#define E33 5
#define E23 6
#define E13 7
#define E12 8
#define S11 3
#define S22 4
#define S33 5
#define S23 6
#define S13 7
#define S12 8
#endif

// ORDERING OF THE ELASTIC COEFFICIENTS
#if (AMREX_SPACEDIM == 2)
#define C11 0
#define C21 1
#define C61 2

#define C22 3
#define C62 4

#define C66 5
#endif
#if (AMREX_SPACEDIM == 3)
#define C11 0
#define C21 1
#define C31 2
#define C41 3
#define C51 4
#define C61 5

#define C22 6
#define C32 7
#define C42 8
#define C52 9
#define C62 10

#define C33 11
#define C43 12
#define C53 13
#define C63 14

#define C44 15
#define C54 16
#define C64 17

#define C55 18
#define C65 19

#define C66 20
#endif
// ####################################################################



namespace elastic_solid
{
// PDE OPERATORS ######################################################
/**
 * \brief Fill memory pointed to by An with coefficients of the hyperbolic operator An.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[out] An: pointer to memory that will contain the hyperbolic operator An.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_An_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un, amrex::Real * An)
{
    // PARAMETERS
    const amrex::Real inv_rho = 1.0/rho;

    std::fill(An, An+N_VS*N_VS, 0.0);
#if (AMREX_SPACEDIM == 2)
    An[0+S11*N_VS] = -inv_rho*un[0]; An[0+S12*N_VS] = -inv_rho*un[1];
    An[1+S22*N_VS] = -inv_rho*un[1]; An[1+S12*N_VS] = -inv_rho*un[0];
    
    An[2+V1*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]); An[2+V2*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]);
    An[3+V1*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]); An[3+V2*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]);
    An[4+V1*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]); An[4+V2*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    An[V1+S11*N_VS] = -inv_rho*un[0]; An[V1+S12*N_VS] = -inv_rho*un[1]; An[V1+S13*N_VS] = -inv_rho*un[2];
    An[V2+S12*N_VS] = -inv_rho*un[0]; An[V2+S22*N_VS] = -inv_rho*un[1]; An[V2+S23*N_VS] = -inv_rho*un[2];
    An[V3+S13*N_VS] = -inv_rho*un[0]; An[V3+S23*N_VS] = -inv_rho*un[1]; An[V3+S33*N_VS] = -inv_rho*un[2];

    An[S11+V1*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2]); An[S11+V2*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2]); An[S11+V3*N_VS] = -(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2]);
    An[S22+V1*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2]); An[S22+V2*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2]); An[S22+V3*N_VS] = -(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2]);
    An[S33+V1*N_VS] = -(c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2]); An[S33+V2*N_VS] = -(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2]); An[S33+V3*N_VS] = -(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2]);
    An[S23+V1*N_VS] = -(c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2]); An[S23+V2*N_VS] = -(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2]); An[S23+V3*N_VS] = -(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2]);
    An[S13+V1*N_VS] = -(c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2]); An[S13+V2*N_VS] = -(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2]); An[S13+V3*N_VS] = -(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2]);
    An[S12+V1*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2]); An[S12+V2*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2]); An[S12+V3*N_VS] = -(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2]);
#endif
}

/**
 * \brief Fill memory pointed to by F1, F2, F3 with the hyperbolic fluxes.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_F_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * U, AMREX_D_DECL(amrex::Real * F1, amrex::Real * F2, amrex::Real * F3))
{
    // PARAMETERS
    const amrex::Real inv_rho = 1.0/rho;

    AMREX_D_TERM
    (
        F1[V1] = -inv_rho*U[S11];,
        F1[V2] = -inv_rho*U[S12];,
        F1[V3] = -inv_rho*U[S13];
    )

#if (AMREX_SPACEDIM == 2)
    F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2];
    F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2];
    F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
    F1[S11] = -c[C11]*U[V1]-c[C61]*U[V2]-c[C51]*U[V3];
    F1[S22] = -c[C21]*U[V1]-c[C62]*U[V2]-c[C52]*U[V3];
    F1[S33] = -c[C31]*U[V1]-c[C63]*U[V2]-c[C53]*U[V3];
    F1[S23] = -c[C41]*U[V1]-c[C64]*U[V2]-c[C54]*U[V3];
    F1[S13] = -c[C51]*U[V1]-c[C65]*U[V2]-c[C55]*U[V3];
    F1[S12] = -c[C61]*U[V1]-c[C66]*U[V2]-c[C65]*U[V3];
#endif

#if (AMREX_SPACEDIM > 1)
    AMREX_D_TERM
    (
        F2[V1] = -inv_rho*U[S12];,
        F2[V2] = -inv_rho*U[S22];,
        F2[V3] = -inv_rho*U[S23];
    )

#if (AMREX_SPACEDIM == 2)
    F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2];
    F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2];
    F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
    F2[S11] = -c[C61]*U[V1]-c[C21]*U[V2]-c[C41]*U[V3];
    F2[S22] = -c[C62]*U[V1]-c[C22]*U[V2]-c[C42]*U[V3];
    F2[S33] = -c[C63]*U[V1]-c[C32]*U[V2]-c[C43]*U[V3];
    F2[S23] = -c[C64]*U[V1]-c[C42]*U[V2]-c[C44]*U[V3];
    F2[S13] = -c[C65]*U[V1]-c[C52]*U[V2]-c[C54]*U[V3];
    F2[S12] = -c[C66]*U[V1]-c[C62]*U[V2]-c[C64]*U[V3];
#endif
#endif

#if (AMREX_SPACEDIM > 2)
    F3[V1] = -inv_rho*U[S13];
    F3[V2] = -inv_rho*U[S23];
    F3[V3] = -inv_rho*U[S33];

    F3[S11] = -c[C51]*U[V1]-c[C41]*U[V2]-c[C31]*U[V3];
    F3[S22] = -c[C52]*U[V1]-c[C42]*U[V2]-c[C32]*U[V3];
    F3[S33] = -c[C53]*U[V1]-c[C43]*U[V2]-c[C33]*U[V3];
    F3[S23] = -c[C54]*U[V1]-c[C44]*U[V2]-c[C43]*U[V3];
    F3[S13] = -c[C55]*U[V1]-c[C54]*U[V2]-c[C53]*U[V3];
    F3[S12] = -c[C65]*U[V1]-c[C64]*U[V2]-c[C63]*U[V3];
#endif
}

/**
 * \brief Eval the matrix (1/rho)*InT*cSE*In.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] un: pointer to memory containing the unit normal vector.
 * \param[out] InTcIn: pointer to memory that will be filled with the computed matrix.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_InTcIn_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                           amrex::Real * InTcIn)
{
    // PARAMETERS
    const amrex::Real inv_rho = 1.0/rho;
    const amrex::Real un11 = un[0]*un[0]*inv_rho;
#if (AMREX_SPACEDIM > 1)
    const amrex::Real un12 = un[0]*un[1]*inv_rho;
    const amrex::Real un22 = un[1]*un[1]*inv_rho;
#endif
#if (AMREX_SPACEDIM > 2)
    const amrex::Real un23 = un[1]*un[2]*inv_rho;
    const amrex::Real un13 = un[0]*un[2]*inv_rho;
    const amrex::Real un33 = un[2]*un[2]*inv_rho;
#endif

#if (AMREX_SPACEDIM == 2)
    InTcIn[0+0*AMREX_SPACEDIM] = c[C11]*un11+(c[C61]+c[C61])*un12+c[C66]*un22;
    InTcIn[1+0*AMREX_SPACEDIM] = c[C61]*un11+(c[C21]+c[C66])*un12+c[C62]*un22;
    InTcIn[1+1*AMREX_SPACEDIM] = c[C66]*un11+(c[C62]+c[C62])*un12+c[C22]*un22;

    // Symmetric components
    InTcIn[0+1*AMREX_SPACEDIM] = InTcIn[1+0*AMREX_SPACEDIM];
#endif
#if (AMREX_SPACEDIM == 3)
    InTcIn[0+0*AMREX_SPACEDIM] = c[C11]*un11+c[C66]*un22+c[C55]*un33+(c[C61]+c[C61])*un12+(c[C51]+c[C51])*un13+(c[C65]+c[C65])*un23;
    InTcIn[1+0*AMREX_SPACEDIM] = c[C61]*un11+c[C62]*un22+c[C54]*un33+(c[C21]+c[C66])*un12+(c[C41]+c[C65])*un13+(c[C52]+c[C64])*un23;
    InTcIn[2+0*AMREX_SPACEDIM] = c[C51]*un11+c[C64]*un22+c[C53]*un33+(c[C41]+c[C65])*un12+(c[C31]+c[C55])*un13+(c[C63]+c[C54])*un23;
    InTcIn[1+1*AMREX_SPACEDIM] = c[C66]*un11+c[C22]*un22+c[C44]*un33+(c[C62]+c[C62])*un12+(c[C64]+c[C64])*un13+(c[C42]+c[C42])*un23;
    InTcIn[2+1*AMREX_SPACEDIM] = c[C65]*un11+c[C42]*un22+c[C43]*un33+(c[C52]+c[C64])*un12+(c[C63]+c[C54])*un13+(c[C32]+c[C44])*un23;
    InTcIn[2+2*AMREX_SPACEDIM] = c[C55]*un11+c[C44]*un22+c[C33]*un33+(c[C54]+c[C54])*un12+(c[C53]+c[C53])*un13+(c[C43]+c[C43])*un23;

    // Symmetric components
    InTcIn[0+1*AMREX_SPACEDIM] = InTcIn[1+0*AMREX_SPACEDIM];
    InTcIn[0+2*AMREX_SPACEDIM] = InTcIn[2+0*AMREX_SPACEDIM];
    InTcIn[1+2*AMREX_SPACEDIM] = InTcIn[2+1*AMREX_SPACEDIM];
#endif
}

/**
 * \brief Computes the eigenvalues and eigenvectors of the matrix (1/rho)*InT*cSE*In.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] un: pointer to memory containing the unit normal vector.
 * \param[out] w: pointer to memory that will be filled with eigenvalues.
 * \param[out] v: pointer to memory that will be filled with eigenvectors.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_eig_InTcIn_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                               amrex::Real * w, amrex::Real * v)
{
    amrex::Real InTcIn[AMREX_SPACEDIM*AMREX_SPACEDIM];

    eval_InTcIn_compact_c(rho, c, un, InTcIn);

#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dsy2ev(InTcIn, w, v);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dsy3ev(InTcIn, w, v);
#endif
}
// ####################################################################



// RIEMANN PROBLEM ####################################################
/**
 * \brief Eval the numerical flux at a grid's boundary.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
 *                 unit normal.
 * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
 *                 unit normal.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
 *
 * Ref:
 * Title: Systematic development of upwind numerical fluxes for the space discontinuous Galerkin
 *        method applied to elastic wave propagation in anisotropic and heterogeneous media with
 *        physical interfaces
 * Authors: Tie, B and Mouronval, A-S
 * Journal (Year): Computer Methods in Applied Mechanics and Engineering (2020)
 *
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn)
{
    // PARAMETERS -----------------------------------------------------
    const amrex::Real inv_rho = 1.0/rho;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real lamk, zk, lk[N_VS], ak[AMREX_SPACEDIM];
    amrex::Real * vk;
    // ----------------------------------------------------------------

    // EIGENVALUES AND EIGENVECTORS OF THE CHRISTOFFEL SYMBOL ---------
    eval_eig_InTcIn_compact_c(rho, c, un, w, v);
    // ----------------------------------------------------------------
    
    // COEFFICIENTS OF THE RIGHT EIGENVECTORS -------------------------
    // Temporarily use NFn as an auxiliary variable where the jump
    // of the conserved variables is stored.
    AMREX_D_TERM
    (
        NFn[V1] = m_U[V1]-p_U[V1];,
        NFn[V2] = m_U[V2]-p_U[V2];,
        NFn[V3] = m_U[V3]-p_U[V3];
    )
#if (AMREX_SPACEDIM == 2)
    NFn[S11] = m_U[S11]-p_U[S11];
    NFn[S22] = m_U[S22]-p_U[S22];
    NFn[S12] = m_U[S12]-p_U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
    NFn[S11] = m_U[S11]-p_U[S11];
    NFn[S22] = m_U[S22]-p_U[S22];
    NFn[S33] = m_U[S33]-p_U[S33];
    NFn[S23] = m_U[S23]-p_U[S23];
    NFn[S13] = m_U[S13]-p_U[S13];
    NFn[S12] = m_U[S12]-p_U[S12];
#endif

    for (int k = 0; k < AMREX_SPACEDIM; ++k)
    {
        lamk = -std::sqrt(w[k]);
        zk = rho*lamk;
        vk = &v[k*AMREX_SPACEDIM];

        AMREX_D_TERM
        (
            lk[V1] = 0.5*std::sqrt(2.0)*vk[0];,
            lk[V2] = 0.5*std::sqrt(2.0)*vk[1];,
            lk[V3] = 0.5*std::sqrt(2.0)*vk[2];
        )
#if (AMREX_SPACEDIM == 2)
        lk[S11] = un[0]*lk[V1];
        lk[S22] = un[1]*lk[V2];
        lk[S12] = un[0]*lk[V2]+un[1]*lk[V1];

        lk[S11] *= -1.0/zk;
        lk[S22] *= -1.0/zk;
        lk[S12] *= -1.0/zk;
#endif
#if (AMREX_SPACEDIM == 3)
        lk[S11] = un[0]*lk[V1];
        lk[S22] = un[1]*lk[V2];
        lk[S33] = un[2]*lk[V3];
        lk[S23] = un[2]*lk[V2]+un[1]*lk[V3];
        lk[S13] = un[0]*lk[V3]+un[2]*lk[V1];
        lk[S12] = un[0]*lk[V2]+un[1]*lk[V1];

        lk[S11] *= -1.0/zk;
        lk[S22] *= -1.0/zk;
        lk[S33] *= -1.0/zk;
        lk[S23] *= -1.0/zk;
        lk[S13] *= -1.0/zk;
        lk[S12] *= -1.0/zk;
#endif

        AMREX_D_TERM
        (
            ak[k]  = lk[V1]*NFn[V1];,
            ak[k] += lk[V2]*NFn[V2];,
            ak[k] += lk[V3]*NFn[V3];
        )
#if (AMREX_SPACEDIM == 2)
        ak[k] += lk[S11]*NFn[S11];
        ak[k] += lk[S22]*NFn[S22];
        ak[k] += lk[S12]*NFn[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        ak[k] += lk[S11]*NFn[S11];
        ak[k] += lk[S22]*NFn[S22];
        ak[k] += lk[S33]*NFn[S33];
        ak[k] += lk[S23]*NFn[S23];
        ak[k] += lk[S13]*NFn[S13];
        ak[k] += lk[S12]*NFn[S12];
#endif
    }
    // ----------------------------------------------------------------

    // NUMERICAL FLUX -------------------------------------------------
#if (AMREX_SPACEDIM == 2)
    NFn[V1] = -inv_rho*(m_U[S11]*un[0]+m_U[S12]*un[1]);
    NFn[V2] = -inv_rho*(m_U[S12]*un[0]+m_U[S22]*un[1]);

    NFn[S11] = -(c[C11]*m_U[V1]+c[C61]*m_U[V2])*un[0]-(c[C61]*m_U[V1]+c[C21]*m_U[V2])*un[1];
    NFn[S22] = -(c[C21]*m_U[V1]+c[C62]*m_U[V2])*un[0]-(c[C62]*m_U[V1]+c[C22]*m_U[V2])*un[1];
    NFn[S12] = -(c[C61]*m_U[V1]+c[C66]*m_U[V2])*un[0]-(c[C66]*m_U[V1]+c[C62]*m_U[V2])*un[1];
#endif
#if (AMREX_SPACEDIM == 3)
    NFn[V1] = -inv_rho*(m_U[S11]*un[0]+m_U[S12]*un[1]+m_U[S13]*un[2]);
    NFn[V2] = -inv_rho*(m_U[S12]*un[0]+m_U[S22]*un[1]+m_U[S23]*un[2]);
    NFn[V3] = -inv_rho*(m_U[S13]*un[0]+m_U[S23]*un[1]+m_U[S33]*un[2]);

    NFn[S11] = -(c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2])*m_U[V1]-(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2])*m_U[V2]-(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2])*m_U[V3];
    NFn[S22] = -(c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2])*m_U[V1]-(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2])*m_U[V2]-(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2])*m_U[V3];
    NFn[S33] = -(c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2])*m_U[V1]-(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2])*m_U[V2]-(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2])*m_U[V3];
    NFn[S23] = -(c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2])*m_U[V1]-(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2])*m_U[V2]-(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2])*m_U[V3];
    NFn[S13] = -(c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2])*m_U[V1]-(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2])*m_U[V2]-(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2])*m_U[V3];
    NFn[S12] = -(c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2])*m_U[V1]-(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2])*m_U[V2]-(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2])*m_U[V3];
#endif

    // Use lk to store the right eigenvectors
    for (int k = 0; k < AMREX_SPACEDIM; ++k)
    {
        lamk = -std::sqrt(w[k]);
        zk = rho*lamk;
        vk = &v[k*AMREX_SPACEDIM];

        AMREX_D_TERM
        (
            lk[V1] = 0.5*std::sqrt(2.0)*vk[0];,
            lk[V2] = 0.5*std::sqrt(2.0)*vk[1];,
            lk[V3] = 0.5*std::sqrt(2.0)*vk[2];
        )
#if (AMREX_SPACEDIM == 2)
        lk[S11] = (c[C11]*un[0]+c[C61]*un[1])*lk[V1]+(c[C61]*un[0]+c[C21]*un[1])*lk[V2];
        lk[S22] = (c[C21]*un[0]+c[C62]*un[1])*lk[V1]+(c[C62]*un[0]+c[C22]*un[1])*lk[V2];
        lk[S12] = (c[C61]*un[0]+c[C66]*un[1])*lk[V1]+(c[C66]*un[0]+c[C62]*un[1])*lk[V2];

        lk[S11] *= -rho/zk;
        lk[S22] *= -rho/zk;
        lk[S12] *= -rho/zk;
#endif
#if (AMREX_SPACEDIM == 3)
        lk[S11] = (c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2])*lk[V1]+(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2])*lk[V2]+(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2])*lk[V3];
        lk[S22] = (c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2])*lk[V1]+(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2])*lk[V2]+(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2])*lk[V3];
        lk[S33] = (c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2])*lk[V1]+(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2])*lk[V2]+(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2])*lk[V3];
        lk[S23] = (c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2])*lk[V1]+(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2])*lk[V2]+(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2])*lk[V3];
        lk[S13] = (c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2])*lk[V1]+(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2])*lk[V2]+(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2])*lk[V3];
        lk[S12] = (c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2])*lk[V1]+(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2])*lk[V2]+(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2])*lk[V3];
        
        lk[S11] *= -rho/zk;
        lk[S22] *= -rho/zk;
        lk[S33] *= -rho/zk;
        lk[S23] *= -rho/zk;
        lk[S13] *= -rho/zk;
        lk[S12] *= -rho/zk;
#endif

        AMREX_D_TERM
        (
            NFn[V1] -= ak[k]*lamk*lk[V1];,
            NFn[V2] -= ak[k]*lamk*lk[V2];,
            NFn[V3] -= ak[k]*lamk*lk[V3];
        )
#if (AMREX_SPACEDIM == 2)
        NFn[S11] -= ak[k]*lamk*lk[S11];
        NFn[S22] -= ak[k]*lamk*lk[S22];
        NFn[S12] -= ak[k]*lamk*lk[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        NFn[S11] -= ak[k]*lamk*lk[S11];
        NFn[S22] -= ak[k]*lamk*lk[S22];
        NFn[S33] -= ak[k]*lamk*lk[S33];
        NFn[S23] -= ak[k]*lamk*lk[S23];
        NFn[S13] -= ak[k]*lamk*lk[S13];
        NFn[S12] -= ak[k]*lamk*lk[S12];
#endif
    }
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace elastic_solid