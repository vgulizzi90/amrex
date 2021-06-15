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

// ORDERING OF THE ELASTIC COEFFICIENTS STORED IN COMPACT FORM
#if (AMREX_SPACEDIM == 1)
#define C11 0
#endif
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
// AUXILIARY FUNCTIONS ################################################
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Ricker_wavelet(const amrex::Real a1, const amrex::Real f, const amrex::Real t0, const amrex::Real t)
{
    const amrex::Real a2 = -M_PI*M_PI*f*f;
    const amrex::Real tau = a2*(t-t0)*(t-t0);
    const amrex::Real res = a1*(0.5+tau)*std::exp(tau);
    return res;
}
// ####################################################################



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
#if (AMREX_SPACEDIM == 1)
    An[E11+V1*N_VS] = -inv_rho*un[0];
    
    An[V1+E11*N_VS] = -c[C11]*un[0];
#endif
#if (AMREX_SPACEDIM == 2)
    An[E11+V1*N_VS] = -inv_rho*un[0]; An[E12+V1*N_VS] = -inv_rho*un[1];
    An[E22+V2*N_VS] = -inv_rho*un[1]; An[E12+V2*N_VS] = -inv_rho*un[0];
    
    An[V1+E11*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]); An[V2+E11*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]);
    An[V1+E22*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]); An[V2+E22*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]);
    An[V1+E12*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]); An[V2+E12*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    An[E11+V1*N_VS] = -inv_rho*un[0]; An[E12+V1*N_VS] = -inv_rho*un[1]; An[E13+V1*N_VS] = -inv_rho*un[2];
    An[E12+V2*N_VS] = -inv_rho*un[0]; An[E22+V2*N_VS] = -inv_rho*un[1]; An[E23+V2*N_VS] = -inv_rho*un[2];
    An[E13+V3*N_VS] = -inv_rho*un[0]; An[E23+V3*N_VS] = -inv_rho*un[1]; An[E33+V3*N_VS] = -inv_rho*un[2];

    An[V1+E11*N_VS] = -(c[C11]*un[0]+c[C61]*un[1]+c[C51]*un[2]); An[V2+E11*N_VS] = -(c[C61]*un[0]+c[C21]*un[1]+c[C41]*un[2]); An[V3+E11*N_VS] = -(c[C51]*un[0]+c[C41]*un[1]+c[C31]*un[2]);
    An[V1+E22*N_VS] = -(c[C21]*un[0]+c[C62]*un[1]+c[C52]*un[2]); An[V2+E22*N_VS] = -(c[C62]*un[0]+c[C22]*un[1]+c[C42]*un[2]); An[V3+E22*N_VS] = -(c[C52]*un[0]+c[C42]*un[1]+c[C32]*un[2]);
    An[V1+E33*N_VS] = -(c[C31]*un[0]+c[C63]*un[1]+c[C53]*un[2]); An[V2+E33*N_VS] = -(c[C63]*un[0]+c[C32]*un[1]+c[C43]*un[2]); An[V3+E33*N_VS] = -(c[C53]*un[0]+c[C43]*un[1]+c[C33]*un[2]);
    An[V1+E23*N_VS] = -(c[C41]*un[0]+c[C64]*un[1]+c[C54]*un[2]); An[V2+E23*N_VS] = -(c[C64]*un[0]+c[C42]*un[1]+c[C44]*un[2]); An[V3+E23*N_VS] = -(c[C54]*un[0]+c[C44]*un[1]+c[C43]*un[2]);
    An[V1+E13*N_VS] = -(c[C51]*un[0]+c[C65]*un[1]+c[C55]*un[2]); An[V2+E13*N_VS] = -(c[C65]*un[0]+c[C52]*un[1]+c[C54]*un[2]); An[V3+E13*N_VS] = -(c[C55]*un[0]+c[C54]*un[1]+c[C53]*un[2]);
    An[V1+E12*N_VS] = -(c[C61]*un[0]+c[C66]*un[1]+c[C65]*un[2]); An[V2+E12*N_VS] = -(c[C66]*un[0]+c[C62]*un[1]+c[C64]*un[2]); An[V3+E12*N_VS] = -(c[C65]*un[0]+c[C64]*un[1]+c[C63]*un[2]);
#endif
}

/**
 * \brief Fill memory pointed to by AnU with the product of the hyperbolic operator An and U.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] AnU: pointer to memory that will contain AnU.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_AnU_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un, const amrex::Real * U, amrex::Real * AnU)
{
    // PARAMETERS
    const amrex::Real inv_rho = 1.0/rho;
    AMREX_D_TERM
    (
        const amrex::Real n1 = un[0];,
        const amrex::Real n2 = un[1];,
        const amrex::Real n3 = un[2];
    )

#if (AMREX_SPACEDIM == 1)
    AnU[V1] = -(n1*c[C11])*U[E11];
    AnU[E11] = -n1*inv_rho*U[V1];
#endif
#if (AMREX_SPACEDIM == 2)
    AnU[V1] = -(n1*c[C11]+n2*c[C61])*U[E11]-(n1*c[C61]+n2*c[C66])*U[E12]-(n1*c[C21]+n2*c[C62])*U[E22];
    AnU[V2] = -(n2*c[C21]+n1*c[C61])*U[E11]-(n2*c[C62]+n1*c[C66])*U[E12]-(n2*c[C22]+n1*c[C62])*U[E22];
    AnU[E11] = -n1*inv_rho*U[V1];
    AnU[E22] = -n2*inv_rho*U[V2]; 
    AnU[E12] = -n2*inv_rho*U[V1]-n1*inv_rho*U[V2];
#endif
#if (AMREX_SPACEDIM == 3)
    AnU[V1] = -(n1*c[C11]+n3*c[C51]+n2*c[C61])*U[E11]-(n1*c[C61]+n3*c[C65]+n2*c[C66])*U[E12]-(n1*c[C51]+n3*c[C55]+n2*c[C65])*U[E13]-(n1*c[C21]+n3*c[C52]+n2*c[C62])*U[E22]-(n1*c[C41]+n3*c[C54]+n2*c[C64])*U[E23]-(n1*c[C31]+n3*c[C53]+n2*c[C63])*U[E33];
    AnU[V2] = -(n2*c[C21]+n3*c[C41]+n1*c[C61])*U[E11]-(n2*c[C62]+n3*c[C64]+n1*c[C66])*U[E12]-(n2*c[C52]+n3*c[C54]+n1*c[C65])*U[E13]-(n2*c[C22]+n3*c[C42]+n1*c[C62])*U[E22]-(n2*c[C42]+n3*c[C44]+n1*c[C64])*U[E23]-(n2*c[C32]+n3*c[C43]+n1*c[C63])*U[E33];
    AnU[V3] = -(n3*c[C31]+n2*c[C41]+n1*c[C51])*U[E11]-(n3*c[C63]+n2*c[C64]+n1*c[C65])*U[E12]-(n3*c[C53]+n2*c[C54]+n1*c[C55])*U[E13]-(n3*c[C32]+n2*c[C42]+n1*c[C52])*U[E22]-(n3*c[C43]+n2*c[C44]+n1*c[C54])*U[E23]-(n3*c[C33]+n2*c[C43]+n1*c[C53])*U[E33];
    AnU[E11] = -n1*inv_rho*U[V1];
    AnU[E22] = -n2*inv_rho*U[V2];
    AnU[E33] = -n3*inv_rho*U[V3];
    AnU[E23] = -n3*inv_rho*U[V2]-n2*inv_rho*U[V3];
    AnU[E13] = -n3*inv_rho*U[V1]-n1*inv_rho*U[V3];
    AnU[E12] = -n2*inv_rho*U[V1]-n1*inv_rho*U[V2];
#endif
}

/**
 * \brief Fill memory pointed to by F1 with the hyperbolic flux along x1 direction.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[out] F1: pointer to memory that will contain the flux.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_F1_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * U, amrex::Real * F1)
{
    // PARAMETERS
    const amrex::Real inv_rho = 1.0/rho;
amrex::Print() << "eval_F1_compact_c" << std::endl; exit(-1);
#if (AMREX_SPACEDIM == 2)
    F1[V1] = -c[C11]*U[E11]-c[C21]*U[E22]-c[C61]*U[E12];
    F1[V2] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C66]*U[E12];
    F1[E11] = -inv_rho*U[V1];
    F1[E22] = 0.0;
    F1[E12] = -inv_rho*U[V2];
#endif

#if (AMREX_SPACEDIM == 3)
    F1[V1] = -c[C11]*U[E11]-c[C21]*U[E22]-c[C31]*U[E33]-c[C41]*U[E23]-c[C51]*U[E13]-c[C61]*U[E12];
    F1[V2] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C63]*U[E33]-c[C64]*U[E23]-c[C65]*U[E13]-c[C66]*U[E12];
    F1[V3] = -c[C51]*U[E11]-c[C52]*U[E22]-c[C53]*U[E33]-c[C54]*U[E23]-c[C55]*U[E13]-c[C65]*U[E12];
    F1[E11] = -inv_rho*U[V1];
    F1[E22] = 0.0;
    F1[E33] = 0.0;
    F1[E23] = 0.0;
    F1[E13] = -inv_rho*U[V3];
    F1[E12] = -inv_rho*U[V2];
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

#if (AMREX_SPACEDIM == 1)
    F1[V1] = -c[C11]*U[E11];
    F1[E11] = -inv_rho*U[V1];
#endif

#if (AMREX_SPACEDIM == 2)
    F1[V1] = -c[C11]*U[E11]-c[C21]*U[E22]-c[C61]*U[E12];
    F1[V2] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C66]*U[E12];
    F1[E11] = -inv_rho*U[V1];
    F1[E22] = 0.0;
    F1[E12] = -inv_rho*U[V2];
    
    F2[V1] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C66]*U[E12];
    F2[V2] = -c[C21]*U[E11]-c[C22]*U[E22]-c[C62]*U[E12];
    F2[E11] = 0.0;
    F2[E22] = -inv_rho*U[V2];
    F2[E12] = -inv_rho*U[V1];
#endif

#if (AMREX_SPACEDIM == 3)
    F1[V1] = -c[C11]*U[E11]-c[C21]*U[E22]-c[C31]*U[E33]-c[C41]*U[E23]-c[C51]*U[E13]-c[C61]*U[E12];
    F1[V2] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C63]*U[E33]-c[C64]*U[E23]-c[C65]*U[E13]-c[C66]*U[E12];
    F1[V3] = -c[C51]*U[E11]-c[C52]*U[E22]-c[C53]*U[E33]-c[C54]*U[E23]-c[C55]*U[E13]-c[C65]*U[E12];
    F1[E11] = -inv_rho*U[V1];
    F1[E22] = 0.0;
    F1[E33] = 0.0;
    F1[E23] = 0.0;
    F1[E13] = -inv_rho*U[V3];
    F1[E12] = -inv_rho*U[V2];

    F2[V1] = -c[C61]*U[E11]-c[C62]*U[E22]-c[C63]*U[E33]-c[C64]*U[E23]-c[C65]*U[E13]-c[C66]*U[E12];
    F2[V2] = -c[C21]*U[E11]-c[C22]*U[E22]-c[C32]*U[E33]-c[C42]*U[E23]-c[C52]*U[E13]-c[C62]*U[E12];
    F2[V3] = -c[C41]*U[E11]-c[C42]*U[E22]-c[C43]*U[E33]-c[C44]*U[E23]-c[C54]*U[E13]-c[C64]*U[E12];
    F2[E11] = 0.0;
    F2[E22] = -inv_rho*U[V2];
    F2[E33] = 0.0;
    F2[E23] = -inv_rho*U[V3];
    F2[E13] = 0.0;
    F2[E12] = -inv_rho*U[V1];

    F3[V1] = -c[C51]*U[E11]-c[C52]*U[E22]-c[C53]*U[E33]-c[C54]*U[E23]-c[C55]*U[E13]-c[C65]*U[E12];
    F3[V2] = -c[C41]*U[E11]-c[C42]*U[E22]-c[C43]*U[E33]-c[C44]*U[E23]-c[C54]*U[E13]-c[C64]*U[E12];
    F3[V3] = -c[C31]*U[E11]-c[C32]*U[E22]-c[C33]*U[E33]-c[C43]*U[E23]-c[C53]*U[E13]-c[C63]*U[E12];
    F3[E11] = 0.0;
    F3[E22] = 0.0;
    F3[E33] = -inv_rho*U[V3];
    F3[E23] = -inv_rho*U[V2];
    F3[E13] = -inv_rho*U[V1];
    F3[E12] = 0.0;
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

#if (AMREX_SPACEDIM == 1)
    InTcIn[0] = c[C11]*un11;
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
#if (AMREX_SPACEDIM == 1)
    w[0] = InTcIn[0];
    v[0] = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dsy2ev(InTcIn, w, v);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dsy3ev(InTcIn, w, v);
#endif
}
// ####################################################################



// DERIVED VARIABLES ##################################################
/**
 * \brief Fill memory pointed to by S with stress tensor components in Voigt notation.
 *
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[out] S: pointer to memory that will contain the stress tensor components.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_stress_compact_c(const amrex::Real * c, const amrex::Real * U, amrex::Real * S)
{
amrex::Print() << "eval_stress_compact_c" << std::endl; exit(-1);
#if (AMREX_SPACEDIM == 2)
    S[0] = c[C11]*U[E11]+c[C21]*U[E22]+c[C61]*U[E12];
    S[1] = c[C21]*U[E11]+c[C22]*U[E22]+c[C62]*U[E12];
    S[2] = c[C61]*U[E11]+c[C62]*U[E22]+c[C66]*U[E12];
#endif
#if (AMREX_SPACEDIM == 3)
    S[0] = c[C11]*U[E11]+c[C21]*U[E22]+c[C31]*U[E33]+c[C41]*U[E23]+c[C51]*U[E13]+c[C61]*U[E12];
    S[1] = c[C21]*U[E11]+c[C22]*U[E22]+c[C32]*U[E33]+c[C42]*U[E23]+c[C52]*U[E13]+c[C62]*U[E12];
    S[2] = c[C31]*U[E11]+c[C32]*U[E22]+c[C33]*U[E33]+c[C43]*U[E23]+c[C53]*U[E13]+c[C63]*U[E12];
    S[3] = c[C41]*U[E11]+c[C42]*U[E22]+c[C43]*U[E33]+c[C44]*U[E23]+c[C54]*U[E13]+c[C64]*U[E12];
    S[4] = c[C51]*U[E11]+c[C52]*U[E22]+c[C53]*U[E33]+c[C54]*U[E23]+c[C55]*U[E13]+c[C65]*U[E12];
    S[5] = c[C61]*U[E11]+c[C62]*U[E22]+c[C63]*U[E33]+c[C64]*U[E23]+c[C65]*U[E13]+c[C66]*U[E12];
#endif
}

/**
 * \brief Return total energy.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] U: pointer to memory containing the conserved variables.
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_total_energy_compact_c(const amrex::Real rho, const amrex::Real * c, const amrex::Real * U)
{
    const amrex::Real * m = U;
    const amrex::Real * E = &U[E11];
    amrex::Real S[NS];
    amrex::Real res;

    eval_stress_compact_c(c, U, S);

#if (AMREX_SPACEDIM == 1)
    res  = (m[0]*m[0])/rho;
    res += E[0]*S[0];
#endif
#if (AMREX_SPACEDIM == 2)
    res  = (m[0]*m[0]+m[1]*m[1])/rho;
    res += E[0]*S[0]+E[1]*S[1]+E[2]*S[2];
#endif
#if (AMREX_SPACEDIM == 3)
    res  = (m[0]*m[0]+m[1]*m[1]+m[2]*m[2])/rho;
    res += E[0]*S[0]+E[1]*S[1]+E[2]*S[2]+E[3]*S[3]+E[4]*S[4]+E[5]*S[5];
#endif
    res *= 0.5;

    return res;
}
// ####################################################################



// RIEMANN PROBLEM ####################################################
/**
 * \brief Solve the Riemann problem between two adjacent states.
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
 * Title: An exact Riemann solver for wave propagation in arbitrary anisotropic elastic media with
 *        fluid coupling
 * Authors: Qiwei Zhan et al
 * Journal (Year): Computer Methods in Applied Mechanics and Engineering (2018)
 *
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn)
{
    // PARAMETERS -----------------------------------------------------
    const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM], iv[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real Z[AMREX_SPACEDIM*AMREX_SPACEDIM], iZ[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real dFn[N_VS];
    // ----------------------------------------------------------------

    // EIGENVALUES AND VECTORS OF THE TRANSFORMED CHRISTOFFEL SYMBOL --
    eval_eig_InTcIn_compact_c(rho, c, un, w, v);
#if (AMREX_SPACEDIM == 1)
    iv[0] = 1.0/v[0];
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dge2i(v, iv);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dge3i(v, iv);
#endif
    // ----------------------------------------------------------------

    // IMPEDANCE MATRIX -----------------------------------------------
#if (AMREX_SPACEDIM == 1)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real v11 = v[0];
        const amrex::Real iv11 = iv[0];

        Z[0] = rho*(iv11*v11*w1);
        
        iZ[0] = 1.0/Z[0];
    }
#endif
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real v11 = v[0+0*2];
        const amrex::Real v12 = v[0+1*2];
        const amrex::Real v21 = v[1+0*2];
        const amrex::Real v22 = v[1+1*2];
        const amrex::Real iv11 = iv[0+0*2];
        const amrex::Real iv12 = iv[0+1*2];
        const amrex::Real iv21 = iv[1+0*2];
        const amrex::Real iv22 = iv[1+1*2];

        Z[0+0*2] = rho*(iv11*v11*w1+iv21*v12*w2); Z[0+1*2] = rho*(iv12*v11*w1+iv22*v12*w2);
        Z[1+0*2] = rho*(iv11*v21*w1+iv21*v22*w2); Z[1+1*2] = rho*(iv12*v21*w1+iv22*v22*w2);
        
        amrex::dG::linalg::dge2i(Z, iZ);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real w3 = std::sqrt(w[2]);
        const amrex::Real v11 = v[0+0*3];
        const amrex::Real v12 = v[0+1*3];
        const amrex::Real v13 = v[0+2*3];
        const amrex::Real v21 = v[1+0*3];
        const amrex::Real v22 = v[1+1*3];
        const amrex::Real v23 = v[1+2*3];
        const amrex::Real v31 = v[2+0*3];
        const amrex::Real v32 = v[2+1*3];
        const amrex::Real v33 = v[2+2*3];
        const amrex::Real iv11 = iv[0+0*3];
        const amrex::Real iv12 = iv[0+1*3];
        const amrex::Real iv13 = iv[0+2*3];
        const amrex::Real iv21 = iv[1+0*3];
        const amrex::Real iv22 = iv[1+1*3];
        const amrex::Real iv23 = iv[1+2*3];
        const amrex::Real iv31 = iv[2+0*3];
        const amrex::Real iv32 = iv[2+1*3];
        const amrex::Real iv33 = iv[2+2*3];

        Z[0+0*3] = rho*(iv11*v11*w1+iv21*v12*w2+iv31*v13*w3); Z[0+1*3] = rho*(iv12*v11*w1+iv22*v12*w2+iv32*v13*w3); Z[0+2*3] = rho*(iv13*v11*w1+iv23*v12*w2+iv33*v13*w3);
        Z[1+0*3] = rho*(iv11*v21*w1+iv21*v22*w2+iv31*v23*w3); Z[1+1*3] = rho*(iv12*v21*w1+iv22*v22*w2+iv32*v23*w3); Z[1+2*3] = rho*(iv13*v21*w1+iv23*v22*w2+iv33*v23*w3);
        Z[2+0*3] = rho*(iv11*v31*w1+iv21*v32*w2+iv31*v33*w3); Z[2+1*3] = rho*(iv12*v31*w1+iv22*v32*w2+iv32*v33*w3); Z[2+2*3] = rho*(iv13*v31*w1+iv23*v32*w2+iv33*v33*w3);

        amrex::dG::linalg::dge3i(Z, iZ);
    }
#endif

    /* DEBUG
    amrex::Print() << "un: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
    amrex::Print() << "w: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, w);
    amrex::Print() << "v: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
    amrex::Print() << "iv: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iv);
    amrex::Print() << "Z: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, Z);
    amrex::Print() << "iZ: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iZ);
    */
    // ----------------------------------------------------------------

    // AUXILIARY MATRIX Iun.T*Iun -------------------------------------
    // We use v and iv as temporary variables
#if (AMREX_SPACEDIM == 1)
    v[0] = 1.0;
    iv[0] = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
    v[0+0*2] = 1.0;   v[0+1*2] = n1*n2;
    v[1+0*2] = n1*n2; v[1+1*2] = 1.0;
    amrex::dG::linalg::dge2i(v, iv);
#endif
#if (AMREX_SPACEDIM == 3)
    v[0+0*3] = 1.0;   v[0+1*3] = n1*n2; v[0+2*3] = n1*n3;
    v[1+0*3] = n1*n2; v[1+1*3] = 1.0;   v[1+2*3] = n2*n3;
    v[2+0*3] = n1*n3; v[2+1*3] = n2*n3; v[2+2*3] = 1.0;
    amrex::dG::linalg::dge3i(v, iv);
#endif

    /* DEBUG
    amrex::Print() << "Isq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
    amrex::Print() << "iIsq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iv);
    */
    // ----------------------------------------------------------------

    // FLUXES ---------------------------------------------------------
    eval_AnU_compact_c(rho, c, un, m_U, NFn);
    eval_AnU_compact_c(rho, c, un, p_U, dFn);

    /* DEBUG
    amrex::Print() << "m_U: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, m_U);
    amrex::Print() << "p_U: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, p_U);
    amrex::Print() << "m_Fn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    amrex::Print() << "p_Fn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */

#if (AMREX_SPACEDIM == 1)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[E11] -= NFn[E11];

    // Use w as a temporary variable
    w[V1] = n1*dFn[E11];
#endif
#if (AMREX_SPACEDIM == 2)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E12] -= NFn[E12];

    // Use w as a temporary variable
    w[V1] = n1*dFn[E11]+n2*dFn[E12];
    w[V2] = n1*dFn[E12]+n2*dFn[E22];
#endif
#if (AMREX_SPACEDIM == 3)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[V3] -= NFn[V3];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E33] -= NFn[E33];
    dFn[E23] -= NFn[E23];
    dFn[E13] -= NFn[E13];
    dFn[E12] -= NFn[E12];

    // Use w as a temporary variable
    w[V1] = n1*dFn[E11]+n2*dFn[E12]+n3*dFn[E13];
    w[V2] = n1*dFn[E12]+n2*dFn[E22]+n3*dFn[E23];
    w[V3] = n1*dFn[E13]+n2*dFn[E23]+n3*dFn[E33];
#endif

    /* DEBUG
    amrex::Print() << "dFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */
    // ----------------------------------------------------------------

    // NUMERICAL FLUX -------------------------------------------------
#if (AMREX_SPACEDIM == 1)
    {
        const amrex::Real w1 = w[V1];
        const amrex::Real z11 = Z[0];
        const amrex::Real iv11 = iv[0];
        const amrex::Real iz11 = iZ[0];

        NFn[V1] += 0.5*(dFn[V1]+w1*(iv11*z11));
        NFn[E11] += 0.5*(iz11*n1*dFn[V1]+iv11*n1*w1);
    }
#endif
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = w[V1];
        const amrex::Real w2 = w[V2];
        const amrex::Real z11 = Z[0+0*2];
        const amrex::Real z12 = Z[0+1*2];
        const amrex::Real z21 = Z[1+0*2];
        const amrex::Real z22 = Z[1+1*2];
        const amrex::Real iv11 = iv[0+0*2];
        const amrex::Real iv12 = iv[0+1*2];
        const amrex::Real iv21 = iv[1+0*2];
        const amrex::Real iv22 = iv[1+1*2];
        const amrex::Real iz11 = iZ[0+0*2];
        const amrex::Real iz12 = iZ[0+1*2];
        const amrex::Real iz21 = iZ[1+0*2];
        const amrex::Real iz22 = iZ[1+1*2];

        NFn[V1] += 0.5*(dFn[V1]+w1*(iv11*z11+iv21*z12)+w2*(iv12*z11+iv22*z12));
        NFn[V2] += 0.5*(dFn[V2]+w1*(iv11*z21+iv21*z22)+w2*(iv12*z21+iv22*z22));
        NFn[E11] += 0.5*(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]+          iv11*n1*w1+          iv12*n1*w2);
        NFn[E22] += 0.5*(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]+          iv21*n2*w1+          iv22*n2*w2);
        NFn[E12] += 0.5*((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]+(iv21*n1+iv11*n2)*w1+(iv22*n1+iv12*n2)*w2);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = w[V1];
        const amrex::Real w2 = w[V2];
        const amrex::Real w3 = w[V3];
        const amrex::Real z11 = Z[0+0*3];
        const amrex::Real z12 = Z[0+1*3];
        const amrex::Real z13 = Z[0+2*3];
        const amrex::Real z21 = Z[1+0*3];
        const amrex::Real z22 = Z[1+1*3];
        const amrex::Real z23 = Z[1+2*3];
        const amrex::Real z31 = Z[2+0*3];
        const amrex::Real z32 = Z[2+1*3];
        const amrex::Real z33 = Z[2+2*3];
        const amrex::Real iv11 = iv[0+0*3];
        const amrex::Real iv12 = iv[0+1*3];
        const amrex::Real iv13 = iv[0+2*3];
        const amrex::Real iv21 = iv[1+0*3];
        const amrex::Real iv22 = iv[1+1*3];
        const amrex::Real iv23 = iv[1+2*3];
        const amrex::Real iv31 = iv[2+0*3];
        const amrex::Real iv32 = iv[2+1*3];
        const amrex::Real iv33 = iv[2+2*3];
        const amrex::Real iz11 = iZ[0+0*3];
        const amrex::Real iz12 = iZ[0+1*3];
        const amrex::Real iz13 = iZ[0+2*3];
        const amrex::Real iz21 = iZ[1+0*3];
        const amrex::Real iz22 = iZ[1+1*3];
        const amrex::Real iz23 = iZ[1+2*3];
        const amrex::Real iz31 = iZ[2+0*3];
        const amrex::Real iz32 = iZ[2+1*3];
        const amrex::Real iz33 = iZ[2+2*3];

        NFn[V1] += 0.5*(dFn[V1]+w1*(iv11*z11+iv21*z12+iv31*z13)+w2*(iv12*z11+iv22*z12+iv32*z13)+w3*(iv13*z11+iv23*z12+iv33*z13));
        NFn[V2] += 0.5*(dFn[V2]+w1*(iv11*z21+iv21*z22+iv31*z23)+w2*(iv12*z21+iv22*z22+iv32*z23)+w3*(iv13*z21+iv23*z22+iv33*z23));
        NFn[V3] += 0.5*(dFn[V3]+w1*(iv11*z31+iv21*z32+iv31*z33)+w2*(iv12*z31+iv22*z32+iv32*z33)+w3*(iv13*z31+iv23*z32+iv33*z33));
        NFn[E11] += 0.5*(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]          +iz13*n1*dFn[V3]+          iv11*n1*w1          +iv12*n1*w2          +iv13*n1*w3);
        NFn[E22] += 0.5*(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]          +iz23*n2*dFn[V3]+          iv21*n2*w1          +iv22*n2*w2          +iv23*n2*w3);
        NFn[E33] += 0.5*(          iz31*n3*dFn[V1]+          iz32*n3*dFn[V2]          +iz33*n3*dFn[V3]+          iv31*n3*w1          +iv32*n3*w2          +iv33*n3*w3);
        NFn[E23] += 0.5*((iz31*n2+iz21*n3)*dFn[V1]+(iz32*n2+iz22*n3)*dFn[V2]+(iz33*n2+iz23*n3)*dFn[V3]+(iv31*n2+iv21*n3)*w1+(iv32*n2+iv22*n3)*w2+(iv33*n2+iv23*n3)*w3);
        NFn[E13] += 0.5*((iz31*n1+iz11*n3)*dFn[V1]+(iz32*n1+iz12*n3)*dFn[V2]+(iz33*n1+iz13*n3)*dFn[V3]+(iv31*n1+iv11*n3)*w1+(iv32*n1+iv12*n3)*w2+(iv33*n1+iv13*n3)*w3);
        NFn[E12] += 0.5*((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]+(iz23*n1+iz13*n2)*dFn[V3]+(iv21*n1+iv11*n2)*w1+(iv22*n1+iv12*n2)*w2+(iv23*n1+iv13*n2)*w3);
    }
#endif

    /* DEBUG
    amrex::Print() << "NFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    */
    // ----------------------------------------------------------------
}

/**
 * \brief Solve the Riemann problem between two adjacent states.
 *
 * \param[in] m_rho: density of the material on the negative side of the unit normal.
 * \param[in] m_c: pointer to memory containing the coefficients of the stiffness matrix of the material
 *               on the negative side of the unit normal.
 * \param[in] p_rho: density of the material on the positive side of the unit normal.
 * \param[in] p_c: pointer to memory containing the coefficients of the stiffness matrix of the
 *                 material on the positive side of the unit normal.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
 *                 unit normal.
 * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
 *                 unit normal.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
 *
 * Ref:
 * Title: An exact Riemann solver for wave propagation in arbitrary anisotropic elastic media with
 *        fluid coupling
 * Authors: Qiwei Zhan et al
 * Journal (Year): Computer Methods in Applied Mechanics and Engineering (2018)
 *
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver(const amrex::Real m_rho, const amrex::Real * m_c, const amrex::Real p_rho, const amrex::Real * p_c, const amrex::Real * un,
                             const amrex::Real * m_U, const amrex::Real * p_U, amrex::Real * NFn)
{
    // PARAMETERS -----------------------------------------------------
    const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::Real m_w[AMREX_SPACEDIM], m_v[AMREX_SPACEDIM*AMREX_SPACEDIM], m_iv[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real p_w[AMREX_SPACEDIM], p_v[AMREX_SPACEDIM*AMREX_SPACEDIM], p_iv[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real m_Z[AMREX_SPACEDIM*AMREX_SPACEDIM], p_Z[AMREX_SPACEDIM*AMREX_SPACEDIM], sZ[AMREX_SPACEDIM*AMREX_SPACEDIM], iZ[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real dFn[N_VS];
    // ----------------------------------------------------------------

    // EIGENVALUES AND VECTORS OF THE TRANSFORMED CHRISTOFFEL SYMBOL --
    eval_eig_InTcIn_compact_c(m_rho, m_c, un, m_w, m_v);
    eval_eig_InTcIn_compact_c(p_rho, p_c, un, p_w, p_v);
#if (AMREX_SPACEDIM == 1)
    m_iv[0] = 1.0/m_v[0];
    p_iv[0] = 1.0/p_v[0];
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dge2i(m_v, m_iv);
    amrex::dG::linalg::dge2i(p_v, p_iv);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dge3i(m_v, m_iv);
    amrex::dG::linalg::dge3i(p_v, p_iv);
#endif
    // ----------------------------------------------------------------

    // IMPEDANCE MATRIX -----------------------------------------------
#if (AMREX_SPACEDIM == 1)
    {
        const amrex::Real w1 = std::sqrt(m_w[0]);
        const amrex::Real v11 = m_v[0];
        const amrex::Real iv11 = m_iv[0];

        m_Z[0] = m_rho*(iv11*v11*w1);
    }
    {
        const amrex::Real w1 = std::sqrt(p_w[0]);
        const amrex::Real v11 = p_v[0];
        const amrex::Real iv11 = p_iv[0];

        p_Z[0] = p_rho*(iv11*v11*w1);
    }

    iZ[0] = 1.0/(m_Z[0]+p_Z[0]);
#endif
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = std::sqrt(m_w[0]);
        const amrex::Real w2 = std::sqrt(m_w[1]);
        const amrex::Real v11 = m_v[0+0*2];
        const amrex::Real v12 = m_v[0+1*2];
        const amrex::Real v21 = m_v[1+0*2];
        const amrex::Real v22 = m_v[1+1*2];
        const amrex::Real iv11 = m_iv[0+0*2];
        const amrex::Real iv12 = m_iv[0+1*2];
        const amrex::Real iv21 = m_iv[1+0*2];
        const amrex::Real iv22 = m_iv[1+1*2];

        m_Z[0+0*2] = m_rho*(iv11*v11*w1+iv21*v12*w2); m_Z[0+1*2] = m_rho*(iv12*v11*w1+iv22*v12*w2);
        m_Z[1+0*2] = m_rho*(iv11*v21*w1+iv21*v22*w2); m_Z[1+1*2] = m_rho*(iv12*v21*w1+iv22*v22*w2);
    }
    {
        const amrex::Real w1 = std::sqrt(p_w[0]);
        const amrex::Real w2 = std::sqrt(p_w[1]);
        const amrex::Real v11 = p_v[0+0*2];
        const amrex::Real v12 = p_v[0+1*2];
        const amrex::Real v21 = p_v[1+0*2];
        const amrex::Real v22 = p_v[1+1*2];
        const amrex::Real iv11 = p_iv[0+0*2];
        const amrex::Real iv12 = p_iv[0+1*2];
        const amrex::Real iv21 = p_iv[1+0*2];
        const amrex::Real iv22 = p_iv[1+1*2];

        p_Z[0+0*2] = p_rho*(iv11*v11*w1+iv21*v12*w2); p_Z[0+1*2] = p_rho*(iv12*v11*w1+iv22*v12*w2);
        p_Z[1+0*2] = p_rho*(iv11*v21*w1+iv21*v22*w2); p_Z[1+1*2] = p_rho*(iv12*v21*w1+iv22*v22*w2);
    }

    for (int k = 0; k < AMREX_SPACEDIM*AMREX_SPACEDIM; ++k)
    {
        sZ[k] = m_Z[k]+p_Z[k];
    }
    amrex::dG::linalg::dge2i(sZ, iZ);
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = std::sqrt(m_w[0]);
        const amrex::Real w2 = std::sqrt(m_w[1]);
        const amrex::Real w3 = std::sqrt(m_w[2]);
        const amrex::Real v11 = m_v[0+0*3];
        const amrex::Real v12 = m_v[0+1*3];
        const amrex::Real v13 = m_v[0+2*3];
        const amrex::Real v21 = m_v[1+0*3];
        const amrex::Real v22 = m_v[1+1*3];
        const amrex::Real v23 = m_v[1+2*3];
        const amrex::Real v31 = m_v[2+0*3];
        const amrex::Real v32 = m_v[2+1*3];
        const amrex::Real v33 = m_v[2+2*3];
        const amrex::Real iv11 = m_iv[0+0*3];
        const amrex::Real iv12 = m_iv[0+1*3];
        const amrex::Real iv13 = m_iv[0+2*3];
        const amrex::Real iv21 = m_iv[1+0*3];
        const amrex::Real iv22 = m_iv[1+1*3];
        const amrex::Real iv23 = m_iv[1+2*3];
        const amrex::Real iv31 = m_iv[2+0*3];
        const amrex::Real iv32 = m_iv[2+1*3];
        const amrex::Real iv33 = m_iv[2+2*3];

        m_Z[0+0*3] = m_rho*(iv11*v11*w1+iv21*v12*w2+iv31*v13*w3); m_Z[0+1*3] = m_rho*(iv12*v11*w1+iv22*v12*w2+iv32*v13*w3); m_Z[0+2*3] = m_rho*(iv13*v11*w1+iv23*v12*w2+iv33*v13*w3);
        m_Z[1+0*3] = m_rho*(iv11*v21*w1+iv21*v22*w2+iv31*v23*w3); m_Z[1+1*3] = m_rho*(iv12*v21*w1+iv22*v22*w2+iv32*v23*w3); m_Z[1+2*3] = m_rho*(iv13*v21*w1+iv23*v22*w2+iv33*v23*w3);
        m_Z[2+0*3] = m_rho*(iv11*v31*w1+iv21*v32*w2+iv31*v33*w3); m_Z[2+1*3] = m_rho*(iv12*v31*w1+iv22*v32*w2+iv32*v33*w3); m_Z[2+2*3] = m_rho*(iv13*v31*w1+iv23*v32*w2+iv33*v33*w3);
    }
    {
        const amrex::Real w1 = std::sqrt(p_w[0]);
        const amrex::Real w2 = std::sqrt(p_w[1]);
        const amrex::Real w3 = std::sqrt(p_w[2]);
        const amrex::Real v11 = p_v[0+0*3];
        const amrex::Real v12 = p_v[0+1*3];
        const amrex::Real v13 = p_v[0+2*3];
        const amrex::Real v21 = p_v[1+0*3];
        const amrex::Real v22 = p_v[1+1*3];
        const amrex::Real v23 = p_v[1+2*3];
        const amrex::Real v31 = p_v[2+0*3];
        const amrex::Real v32 = p_v[2+1*3];
        const amrex::Real v33 = p_v[2+2*3];
        const amrex::Real iv11 = p_iv[0+0*3];
        const amrex::Real iv12 = p_iv[0+1*3];
        const amrex::Real iv13 = p_iv[0+2*3];
        const amrex::Real iv21 = p_iv[1+0*3];
        const amrex::Real iv22 = p_iv[1+1*3];
        const amrex::Real iv23 = p_iv[1+2*3];
        const amrex::Real iv31 = p_iv[2+0*3];
        const amrex::Real iv32 = p_iv[2+1*3];
        const amrex::Real iv33 = p_iv[2+2*3];

        p_Z[0+0*3] = p_rho*(iv11*v11*w1+iv21*v12*w2+iv31*v13*w3); p_Z[0+1*3] = p_rho*(iv12*v11*w1+iv22*v12*w2+iv32*v13*w3); p_Z[0+2*3] = p_rho*(iv13*v11*w1+iv23*v12*w2+iv33*v13*w3);
        p_Z[1+0*3] = p_rho*(iv11*v21*w1+iv21*v22*w2+iv31*v23*w3); p_Z[1+1*3] = p_rho*(iv12*v21*w1+iv22*v22*w2+iv32*v23*w3); p_Z[1+2*3] = p_rho*(iv13*v21*w1+iv23*v22*w2+iv33*v23*w3);
        p_Z[2+0*3] = p_rho*(iv11*v31*w1+iv21*v32*w2+iv31*v33*w3); p_Z[2+1*3] = p_rho*(iv12*v31*w1+iv22*v32*w2+iv32*v33*w3); p_Z[2+2*3] = p_rho*(iv13*v31*w1+iv23*v32*w2+iv33*v33*w3);
    }

    for (int k = 0; k < AMREX_SPACEDIM*AMREX_SPACEDIM; ++k)
    {
        sZ[k] = m_Z[k]+p_Z[k];
    }
    amrex::dG::linalg::dge3i(sZ, iZ);
#endif

    /* DEBUG
    amrex::Print() << "un: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
    amrex::Print() << "m_w: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, m_w);
    amrex::Print() << "m_v: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, m_v);
    amrex::Print() << "m_iv: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, m_iv);
    amrex::Print() << "m_Z: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, m_Z);
    amrex::Print() << "p_w: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, p_w);
    amrex::Print() << "p_v: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, p_v);
    amrex::Print() << "p_iv: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, p_iv);
    amrex::Print() << "p_Z: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, p_Z);
    amrex::Print() << "iZ: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iZ);
    */
    // ----------------------------------------------------------------

    // AUXILIARY MATRIX Iun.T*Iun -------------------------------------
    // We use m_v and m_iv as temporary variables
#if (AMREX_SPACEDIM == 1)
    m_v[0] = 1.0;
    m_iv[0] = 1.0;
#endif
#if (AMREX_SPACEDIM == 2)
    m_v[0+0*2] = 1.0;   m_v[0+1*2] = n1*n2;
    m_v[1+0*2] = n1*n2; m_v[1+1*2] = 1.0;
    amrex::dG::linalg::dge2i(m_v, m_iv);
#endif
#if (AMREX_SPACEDIM == 3)
    m_v[0+0*3] = 1.0;   m_v[0+1*3] = n1*n2; m_v[0+2*3] = n1*n3;
    m_v[1+0*3] = n1*n2; m_v[1+1*3] = 1.0;   m_v[1+2*3] = n2*n3;
    m_v[2+0*3] = n1*n3; m_v[2+1*3] = n2*n3; m_v[2+2*3] = 1.0;
    amrex::dG::linalg::dge3i(m_v, m_iv);
#endif

    /* DEBUG
    amrex::Print() << "Isq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, m_v);
    amrex::Print() << "iIsq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, m_iv);
    */
    // ----------------------------------------------------------------

    // FLUXES ---------------------------------------------------------
    eval_AnU_compact_c(m_rho, m_c, un, m_U, NFn);
    eval_AnU_compact_c(m_rho, m_c, un, p_U, dFn);

    /* DEBUG
    amrex::Print() << "m_U: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, m_U);
    amrex::Print() << "p_U: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, p_U);
    amrex::Print() << "m_Fn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    amrex::Print() << "p_Fn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */

#if (AMREX_SPACEDIM == 1)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[E11] -= NFn[E11];

    // Use m_w as a temporary variable
    m_w[V1] = n1*dFn[E11];
#endif
#if (AMREX_SPACEDIM == 2)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E12] -= NFn[E12];

    // Use m_w as a temporary variable
    m_w[V1] = n1*dFn[E11]+n2*dFn[E12];
    m_w[V2] = n1*dFn[E12]+n2*dFn[E22];
#endif
#if (AMREX_SPACEDIM == 3)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[V3] -= NFn[V3];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E33] -= NFn[E33];
    dFn[E23] -= NFn[E23];
    dFn[E13] -= NFn[E13];
    dFn[E12] -= NFn[E12];

    // Use m_w as a temporary variable
    m_w[V1] = n1*dFn[E11]+n2*dFn[E12]+n3*dFn[E13];
    m_w[V2] = n1*dFn[E12]+n2*dFn[E22]+n3*dFn[E23];
    m_w[V3] = n1*dFn[E13]+n2*dFn[E23]+n3*dFn[E33];
#endif

    /* DEBUG
    amrex::Print() << "dFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */
    // ----------------------------------------------------------------

    // NUMERICAL FLUX -------------------------------------------------
    {
        amrex::Real t1DxD[AMREX_SPACEDIM*AMREX_SPACEDIM], t2DxD[AMREX_SPACEDIM*AMREX_SPACEDIM];
        amrex::Real tmp3[AMREX_SPACEDIM];

        amrex::dG::linalg::dgemm('N', 'N', AMREX_SPACEDIM, AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, m_Z, AMREX_SPACEDIM, iZ, AMREX_SPACEDIM, 0.0, t1DxD, AMREX_SPACEDIM);
        amrex::dG::linalg::dgemv('N', AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t1DxD, AMREX_SPACEDIM, dFn, 1, 1.0, NFn, 1);

        amrex::dG::linalg::dgemm('N', 'N', AMREX_SPACEDIM, AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t1DxD, AMREX_SPACEDIM, p_Z, AMREX_SPACEDIM, 0.0, t2DxD, AMREX_SPACEDIM);
        amrex::dG::linalg::dgemm('N', 'N', AMREX_SPACEDIM, AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t2DxD, AMREX_SPACEDIM, m_iv, AMREX_SPACEDIM, 0.0, t1DxD, AMREX_SPACEDIM);
        amrex::dG::linalg::dgemv('N', AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t1DxD, AMREX_SPACEDIM, m_w, 1, 1.0, NFn, 1);

        amrex::dG::linalg::dgemv('N', AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, iZ, AMREX_SPACEDIM, dFn, 1, 0.0, tmp3, 1);

        amrex::dG::linalg::dgemm('N', 'N', AMREX_SPACEDIM, AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, iZ, AMREX_SPACEDIM, p_Z, AMREX_SPACEDIM, 0.0, t2DxD, AMREX_SPACEDIM);
        amrex::dG::linalg::dgemm('N', 'N', AMREX_SPACEDIM, AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t2DxD, AMREX_SPACEDIM, m_iv, AMREX_SPACEDIM, 0.0, t1DxD, AMREX_SPACEDIM);
        amrex::dG::linalg::dgemv('N', AMREX_SPACEDIM, AMREX_SPACEDIM, 1.0, t1DxD, AMREX_SPACEDIM, m_w, 1, 1.0, tmp3, 1);

#if (AMREX_SPACEDIM == 1)
        NFn[E11] += n1*tmp3[V1];
#endif
#if (AMREX_SPACEDIM == 2)
        NFn[E11] += n1*tmp3[V1];
        NFn[E22] += n2*tmp3[V2];
        NFn[E12] += n1*tmp3[V2]+n2*tmp3[V1];
#endif
#if (AMREX_SPACEDIM == 3)
        NFn[E11] += n1*tmp3[V1];
        NFn[E22] += n2*tmp3[V2];
        NFn[E33] += n3*tmp3[V3];
        NFn[E23] += tmp3[V3]*n2+tmp3[V2]*n3;
        NFn[E13] += tmp3[V3]*n1+tmp3[V1]*n3;
        NFn[E12] += tmp3[V2]*n1+tmp3[V1]*n2;
#endif
    }

    /* DEBUG
    amrex::Print() << "NFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    exit(-1);
    */
    // ----------------------------------------------------------------
}

/**
 * \brief Solve the Riemann problem to enforce prescribed tractions BCs.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[in] U: pointer to memory containing the conserved variables.
 * \param[in] T: pointer to memory containing the prescribed tractions.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
 *
 * Ref:
 * Title: An exact Riemann solver for wave propagation in arbitrary anisotropic elastic media with
 *        fluid coupling
 * Authors: Qiwei Zhan et al
 * Journal (Year): Computer Methods in Applied Mechanics and Engineering (2018)
 *
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver_prescribed_tractions(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                                                  const amrex::Real * U, const amrex::Real * T, amrex::Real * NFn)
{
    // PARAMETERS -----------------------------------------------------
    const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM], iv[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real Z[AMREX_SPACEDIM*AMREX_SPACEDIM], iZ[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real dFn[N_VS];
    // ----------------------------------------------------------------

    // EIGENVALUES AND VECTORS OF THE TRANSFORMED CHRISTOFFEL SYMBOL --
    eval_eig_InTcIn_compact_c(rho, c, un, w, v);
#if (AMREX_SPACEDIM == 1)
    iv[0] = 1.0/v[0];
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dge2i(v, iv);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dge3i(v, iv);
#endif
    // ----------------------------------------------------------------

    // IMPEDANCE MATRIX -----------------------------------------------
#if (AMREX_SPACEDIM == 1)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real v11 = v[0];
        const amrex::Real iv11 = iv[0];

        Z[0] = rho*(iv11*v11*w1);
        iZ[0] = 1.0/Z[0];
    }
#endif
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real v11 = v[0+0*2];
        const amrex::Real v12 = v[0+1*2];
        const amrex::Real v21 = v[1+0*2];
        const amrex::Real v22 = v[1+1*2];
        const amrex::Real iv11 = iv[0+0*2];
        const amrex::Real iv12 = iv[0+1*2];
        const amrex::Real iv21 = iv[1+0*2];
        const amrex::Real iv22 = iv[1+1*2];

        Z[0+0*2] = rho*(iv11*v11*w1+iv21*v12*w2); Z[0+1*2] = rho*(iv12*v11*w1+iv22*v12*w2);
        Z[1+0*2] = rho*(iv11*v21*w1+iv21*v22*w2); Z[1+1*2] = rho*(iv12*v21*w1+iv22*v22*w2);
        
        amrex::dG::linalg::dge2i(Z, iZ);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real w3 = std::sqrt(w[2]);
        const amrex::Real v11 = v[0+0*3];
        const amrex::Real v12 = v[0+1*3];
        const amrex::Real v13 = v[0+2*3];
        const amrex::Real v21 = v[1+0*3];
        const amrex::Real v22 = v[1+1*3];
        const amrex::Real v23 = v[1+2*3];
        const amrex::Real v31 = v[2+0*3];
        const amrex::Real v32 = v[2+1*3];
        const amrex::Real v33 = v[2+2*3];
        const amrex::Real iv11 = iv[0+0*3];
        const amrex::Real iv12 = iv[0+1*3];
        const amrex::Real iv13 = iv[0+2*3];
        const amrex::Real iv21 = iv[1+0*3];
        const amrex::Real iv22 = iv[1+1*3];
        const amrex::Real iv23 = iv[1+2*3];
        const amrex::Real iv31 = iv[2+0*3];
        const amrex::Real iv32 = iv[2+1*3];
        const amrex::Real iv33 = iv[2+2*3];

        Z[0+0*3] = rho*(iv11*v11*w1+iv21*v12*w2+iv31*v13*w3); Z[0+1*3] = rho*(iv12*v11*w1+iv22*v12*w2+iv32*v13*w3); Z[0+2*3] = rho*(iv13*v11*w1+iv23*v12*w2+iv33*v13*w3);
        Z[1+0*3] = rho*(iv11*v21*w1+iv21*v22*w2+iv31*v23*w3); Z[1+1*3] = rho*(iv12*v21*w1+iv22*v22*w2+iv32*v23*w3); Z[1+2*3] = rho*(iv13*v21*w1+iv23*v22*w2+iv33*v23*w3);
        Z[2+0*3] = rho*(iv11*v31*w1+iv21*v32*w2+iv31*v33*w3); Z[2+1*3] = rho*(iv12*v31*w1+iv22*v32*w2+iv32*v33*w3); Z[2+2*3] = rho*(iv13*v31*w1+iv23*v32*w2+iv33*v33*w3);

        amrex::dG::linalg::dge3i(Z, iZ);
    }
#endif

    /* DEBUG
    amrex::Print() << "un: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
    amrex::Print() << "w: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, w);
    amrex::Print() << "v: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
    amrex::Print() << "iv: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iv);
    amrex::Print() << "Z: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, Z);
    amrex::Print() << "iZ: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iZ);
    exit(-1);
    */
    // ----------------------------------------------------------------

    // FLUXES ---------------------------------------------------------
    eval_AnU_compact_c(rho, c, un, U, dFn);

#if (AMREX_SPACEDIM == 1)
    // Jump in the flux
    dFn[V1] -= (-T[V1]);
#endif
#if (AMREX_SPACEDIM == 2)
    // Jump in the flux
    dFn[V1] -= (-T[V1]);
    dFn[V2] -= (-T[V2]);
#endif
#if (AMREX_SPACEDIM == 3)
    // Jump in the flux
    dFn[V1] -= (-T[V1]);
    dFn[V2] -= (-T[V2]);
    dFn[V3] -= (-T[V3]);
#endif

    /* DEBUG
    amrex::Print() << "dFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */
    // ----------------------------------------------------------------

    // NUMERICAL FLUX -------------------------------------------------
#if (AMREX_SPACEDIM == 1)
    {
        const amrex::Real iz11 = iZ[0];

        NFn[V1] = -T[V1];
        NFn[E11] = dFn[E11]-(iz11*n1*dFn[V1]);
    }
#endif
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real iz11 = iZ[0+0*2];
        const amrex::Real iz12 = iZ[0+1*2];
        const amrex::Real iz21 = iZ[1+0*2];
        const amrex::Real iz22 = iZ[1+1*2];

        NFn[V1] = -T[V1];
        NFn[V2] = -T[V2];
        NFn[E11] = dFn[E11]-(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]);
        NFn[E22] = dFn[E22]-(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]);
        NFn[E12] = dFn[E12]-((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real iz11 = iZ[0+0*3];
        const amrex::Real iz12 = iZ[0+1*3];
        const amrex::Real iz13 = iZ[0+2*3];
        const amrex::Real iz21 = iZ[1+0*3];
        const amrex::Real iz22 = iZ[1+1*3];
        const amrex::Real iz23 = iZ[1+2*3];
        const amrex::Real iz31 = iZ[2+0*3];
        const amrex::Real iz32 = iZ[2+1*3];
        const amrex::Real iz33 = iZ[2+2*3];

        NFn[V1] = -T[V1];
        NFn[V2] = -T[V2];
        NFn[V3] = -T[V3];
        NFn[E11] = dFn[E11]-(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]          +iz13*n1*dFn[V3]);
        NFn[E22] = dFn[E22]-(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]          +iz23*n2*dFn[V3]);
        NFn[E33] = dFn[E33]-(          iz31*n3*dFn[V1]+          iz32*n3*dFn[V2]          +iz33*n3*dFn[V3]);
        NFn[E23] = dFn[E23]-((iz31*n2+iz21*n3)*dFn[V1]+(iz32*n2+iz22*n3)*dFn[V2]+(iz33*n2+iz23*n3)*dFn[V3]);
        NFn[E13] = dFn[E13]-((iz31*n1+iz11*n3)*dFn[V1]+(iz32*n1+iz12*n3)*dFn[V2]+(iz33*n1+iz13*n3)*dFn[V3]);
        NFn[E12] = dFn[E12]-((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]+(iz23*n1+iz13*n2)*dFn[V3]);
    }
#endif

    /* DEBUG
    amrex::Print() << "NFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    exit(-1);
    */
    // ----------------------------------------------------------------
}

/**
 * \brief Solve the Riemann problem to enforce absorbing boundary.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
 * \param[out] NFn: pointer to memory that will contain the numerical flux.
 *
 * Ref:
 * Title: An exact Riemann solver for wave propagation in arbitrary anisotropic elastic media with
 *        fluid coupling
 * Authors: Qiwei Zhan et al
 * Journal (Year): Computer Methods in Applied Mechanics and Engineering (2018)
 *
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_NFn_Riemann_solver_absorbing_boundary(const amrex::Real rho, const amrex::Real * c, const amrex::Real * un,
                                                const amrex::Real * U, amrex::Real * NFn)
{
    // PARAMETERS -----------------------------------------------------
    const amrex::Real AMREX_D_DECL(n1 = un[0], n2 = un[1], n3 = un[2]);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM], iv[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real Z[AMREX_SPACEDIM*AMREX_SPACEDIM], iZ[AMREX_SPACEDIM*AMREX_SPACEDIM];
    amrex::Real dFn[N_VS];
    // ----------------------------------------------------------------
amrex::Print() << "eval_NFn_Riemann_solver_absorbing_boundary" << std::endl; exit(-1);
    // EIGENVALUES AND VECTORS OF THE TRANSFORMED CHRISTOFFEL SYMBOL --
    eval_eig_InTcIn_compact_c(rho, c, un, w, v);
#if (AMREX_SPACEDIM == 2)
    amrex::dG::linalg::dge2i(v, iv);
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::dG::linalg::dge3i(v, iv);
#endif
    // ----------------------------------------------------------------

    // IMPEDANCE MATRIX -----------------------------------------------
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real v11 = v[0+0*2];
        const amrex::Real v12 = v[0+1*2];
        const amrex::Real v21 = v[1+0*2];
        const amrex::Real v22 = v[1+1*2];
        const amrex::Real iv11 = iv[0+0*2];
        const amrex::Real iv12 = iv[0+1*2];
        const amrex::Real iv21 = iv[1+0*2];
        const amrex::Real iv22 = iv[1+1*2];

        Z[0+0*2] = rho*(iv11*v11*w1+iv21*v12*w2); Z[0+1*2] = rho*(iv12*v11*w1+iv22*v12*w2);
        Z[1+0*2] = rho*(iv11*v21*w1+iv21*v22*w2); Z[1+1*2] = rho*(iv12*v21*w1+iv22*v22*w2);
        
        amrex::dG::linalg::dge2i(Z, iZ);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = std::sqrt(w[0]);
        const amrex::Real w2 = std::sqrt(w[1]);
        const amrex::Real w3 = std::sqrt(w[2]);
        const amrex::Real v11 = v[0+0*3];
        const amrex::Real v12 = v[0+1*3];
        const amrex::Real v13 = v[0+2*3];
        const amrex::Real v21 = v[1+0*3];
        const amrex::Real v22 = v[1+1*3];
        const amrex::Real v23 = v[1+2*3];
        const amrex::Real v31 = v[2+0*3];
        const amrex::Real v32 = v[2+1*3];
        const amrex::Real v33 = v[2+2*3];
        const amrex::Real iv11 = iv[0+0*3];
        const amrex::Real iv12 = iv[0+1*3];
        const amrex::Real iv13 = iv[0+2*3];
        const amrex::Real iv21 = iv[1+0*3];
        const amrex::Real iv22 = iv[1+1*3];
        const amrex::Real iv23 = iv[1+2*3];
        const amrex::Real iv31 = iv[2+0*3];
        const amrex::Real iv32 = iv[2+1*3];
        const amrex::Real iv33 = iv[2+2*3];

        Z[0+0*3] = rho*(iv11*v11*w1+iv21*v12*w2+iv31*v13*w3); Z[0+1*3] = rho*(iv12*v11*w1+iv22*v12*w2+iv32*v13*w3); Z[0+2*3] = rho*(iv13*v11*w1+iv23*v12*w2+iv33*v13*w3);
        Z[1+0*3] = rho*(iv11*v21*w1+iv21*v22*w2+iv31*v23*w3); Z[1+1*3] = rho*(iv12*v21*w1+iv22*v22*w2+iv32*v23*w3); Z[1+2*3] = rho*(iv13*v21*w1+iv23*v22*w2+iv33*v23*w3);
        Z[2+0*3] = rho*(iv11*v31*w1+iv21*v32*w2+iv31*v33*w3); Z[2+1*3] = rho*(iv12*v31*w1+iv22*v32*w2+iv32*v33*w3); Z[2+2*3] = rho*(iv13*v31*w1+iv23*v32*w2+iv33*v33*w3);

        amrex::dG::linalg::dge3i(Z, iZ);
    }
#endif

    /* DEBUG
    amrex::Print() << "un: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
    amrex::Print() << "w: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, w);
    amrex::Print() << "v: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
    amrex::Print() << "iv: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iv);
    amrex::Print() << "Z: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, Z);
    amrex::Print() << "iZ: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iZ);
    */
    // ----------------------------------------------------------------

    // AUXILIARY MATRIX Iun.T*Iun -------------------------------------
    // We use v and iv as temporary variables
#if (AMREX_SPACEDIM == 2)
    v[0+0*2] = 1.0;   v[0+1*2] = n1*n2;
    v[1+0*2] = n1*n2; v[1+1*2] = 1.0;
    amrex::dG::linalg::dge2i(v, iv);
#endif
#if (AMREX_SPACEDIM == 3)
    v[0+0*3] = 1.0;   v[0+1*3] = n1*n2; v[0+2*3] = n1*n3;
    v[1+0*3] = n1*n2; v[1+1*3] = 1.0;   v[1+2*3] = n2*n3;
    v[2+0*3] = n1*n3; v[2+1*3] = n2*n3; v[2+2*3] = 1.0;
    amrex::dG::linalg::dge3i(v, iv);
#endif

    /* DEBUG
    amrex::Print() << "Isq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
    amrex::Print() << "iIsq: " << std::endl;
    amrex::dG::io::print_real_array_2d(AMREX_SPACEDIM, AMREX_SPACEDIM, iv);
    */
    // ----------------------------------------------------------------

    // FLUXES ---------------------------------------------------------
    eval_AnU_compact_c(rho, c, un, U, NFn);
#if (AMREX_SPACEDIM == 2)
    dFn[V1] = 0.0;
    dFn[V2] = 0.0;
    dFn[E11] = 0.0;
    dFn[E22] = 0.0;
    dFn[E12] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
    dFn[V1] = 0.0;
    dFn[V2] = 0.0;
    dFn[V3] = 0.0;
    dFn[E11] = 0.0;
    dFn[E22] = 0.0;
    dFn[E33] = 0.0;
    dFn[E23] = 0.0;
    dFn[E13] = 0.0;
    dFn[E12] = 0.0;
#endif

    /* DEBUG
    amrex::Print() << "U: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, U);
    amrex::Print() << "Fn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    */

#if (AMREX_SPACEDIM == 2)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E12] -= NFn[E12];

    // Use w as a temporary variable
    w[V1] = n1*dFn[E11]+n2*dFn[E12];
    w[V2] = n1*dFn[E12]+n2*dFn[E22];
#endif
#if (AMREX_SPACEDIM == 3)
    // Jump in the flux
    dFn[V1] -= NFn[V1];
    dFn[V2] -= NFn[V2];
    dFn[V3] -= NFn[V3];
    dFn[E11] -= NFn[E11];
    dFn[E22] -= NFn[E22];
    dFn[E33] -= NFn[E33];
    dFn[E23] -= NFn[E23];
    dFn[E13] -= NFn[E13];
    dFn[E12] -= NFn[E12];

    // Use w as a temporary variable
    w[V1] = n1*dFn[E11]+n2*dFn[E12]+n3*dFn[E13];
    w[V2] = n1*dFn[E12]+n2*dFn[E22]+n3*dFn[E23];
    w[V3] = n1*dFn[E13]+n2*dFn[E23]+n3*dFn[E33];
#endif

    /* DEBUG
    amrex::Print() << "dFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, dFn);
    */
    // ----------------------------------------------------------------

    // NUMERICAL FLUX -------------------------------------------------
#if (AMREX_SPACEDIM == 2)
    {
        const amrex::Real w1 = w[V1];
        const amrex::Real w2 = w[V2];
        const amrex::Real z11 = Z[0+0*2];
        const amrex::Real z12 = Z[0+1*2];
        const amrex::Real z21 = Z[1+0*2];
        const amrex::Real z22 = Z[1+1*2];
        const amrex::Real iv11 = iv[0+0*2];
        const amrex::Real iv12 = iv[0+1*2];
        const amrex::Real iv21 = iv[1+0*2];
        const amrex::Real iv22 = iv[1+1*2];
        const amrex::Real iz11 = iZ[0+0*2];
        const amrex::Real iz12 = iZ[0+1*2];
        const amrex::Real iz21 = iZ[1+0*2];
        const amrex::Real iz22 = iZ[1+1*2];

        NFn[V1] += 0.5*(dFn[V1]+w1*(iv11*z11+iv21*z12)+w2*(iv12*z11+iv22*z12));
        NFn[V2] += 0.5*(dFn[V2]+w1*(iv11*z21+iv21*z22)+w2*(iv12*z21+iv22*z22));
        NFn[E11] += 0.5*(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]+          iv11*n1*w1+          iv12*n1*w2);
        NFn[E22] += 0.5*(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]+          iv21*n2*w1+          iv22*n2*w2);
        NFn[E12] += 0.5*((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]+(iv21*n1+iv11*n2)*w1+(iv22*n1+iv12*n2)*w2);
    }
#endif
#if (AMREX_SPACEDIM == 3)
    {
        const amrex::Real w1 = w[V1];
        const amrex::Real w2 = w[V2];
        const amrex::Real w3 = w[V3];
        const amrex::Real z11 = Z[0+0*3];
        const amrex::Real z12 = Z[0+1*3];
        const amrex::Real z13 = Z[0+2*3];
        const amrex::Real z21 = Z[1+0*3];
        const amrex::Real z22 = Z[1+1*3];
        const amrex::Real z23 = Z[1+2*3];
        const amrex::Real z31 = Z[2+0*3];
        const amrex::Real z32 = Z[2+1*3];
        const amrex::Real z33 = Z[2+2*3];
        const amrex::Real iv11 = iv[0+0*3];
        const amrex::Real iv12 = iv[0+1*3];
        const amrex::Real iv13 = iv[0+2*3];
        const amrex::Real iv21 = iv[1+0*3];
        const amrex::Real iv22 = iv[1+1*3];
        const amrex::Real iv23 = iv[1+2*3];
        const amrex::Real iv31 = iv[2+0*3];
        const amrex::Real iv32 = iv[2+1*3];
        const amrex::Real iv33 = iv[2+2*3];
        const amrex::Real iz11 = iZ[0+0*3];
        const amrex::Real iz12 = iZ[0+1*3];
        const amrex::Real iz13 = iZ[0+2*3];
        const amrex::Real iz21 = iZ[1+0*3];
        const amrex::Real iz22 = iZ[1+1*3];
        const amrex::Real iz23 = iZ[1+2*3];
        const amrex::Real iz31 = iZ[2+0*3];
        const amrex::Real iz32 = iZ[2+1*3];
        const amrex::Real iz33 = iZ[2+2*3];

        NFn[V1] += 0.5*(dFn[V1]+w1*(iv11*z11+iv21*z12+iv31*z13)+w2*(iv12*z11+iv22*z12+iv32*z13)+w3*(iv13*z11+iv23*z12+iv33*z13));
        NFn[V2] += 0.5*(dFn[V2]+w1*(iv11*z21+iv21*z22+iv31*z23)+w2*(iv12*z21+iv22*z22+iv32*z23)+w3*(iv13*z21+iv23*z22+iv33*z23));
        NFn[V3] += 0.5*(dFn[V3]+w1*(iv11*z31+iv21*z32+iv31*z33)+w2*(iv12*z31+iv22*z32+iv32*z33)+w3*(iv13*z31+iv23*z32+iv33*z33));
        NFn[E11] += 0.5*(          iz11*n1*dFn[V1]+          iz12*n1*dFn[V2]          +iz13*n1*dFn[V3]+          iv11*n1*w1          +iv12*n1*w2          +iv13*n1*w3);
        NFn[E22] += 0.5*(          iz21*n2*dFn[V1]+          iz22*n2*dFn[V2]          +iz23*n2*dFn[V3]+          iv21*n2*w1          +iv22*n2*w2          +iv23*n2*w3);
        NFn[E33] += 0.5*(          iz31*n3*dFn[V1]+          iz32*n3*dFn[V2]          +iz33*n3*dFn[V3]+          iv31*n3*w1          +iv32*n3*w2          +iv33*n3*w3);
        NFn[E23] += 0.5*((iz31*n2+iz21*n3)*dFn[V1]+(iz32*n2+iz22*n3)*dFn[V2]+(iz33*n2+iz23*n3)*dFn[V3]+(iv31*n2+iv21*n3)*w1+(iv32*n2+iv22*n3)*w2+(iv33*n2+iv23*n3)*w3);
        NFn[E13] += 0.5*((iz31*n1+iz11*n3)*dFn[V1]+(iz32*n1+iz12*n3)*dFn[V2]+(iz33*n1+iz13*n3)*dFn[V3]+(iv31*n1+iv11*n3)*w1+(iv32*n1+iv12*n3)*w2+(iv33*n1+iv13*n3)*w3);
        NFn[E12] += 0.5*((iz21*n1+iz11*n2)*dFn[V1]+(iz22*n1+iz12*n2)*dFn[V2]+(iz23*n1+iz13*n2)*dFn[V3]+(iv21*n1+iv11*n2)*w1+(iv22*n1+iv12*n2)*w2+(iv23*n1+iv13*n2)*w3);
    }
#endif

    /* DEBUG
    amrex::Print() << "NFn: " << std::endl;
    amrex::dG::io::print_real_array_2d(1, N_VS, NFn);
    exit(-1);
    */
    // ----------------------------------------------------------------
}
// ####################################################################



// AMR TAGGING ########################################################
/**
 * \brief Tag cells according to user-defined condition.
 *
 * \param[in] rho: density.
 * \param[in] c: pointer to memory containing the coefficients of the stiffness matrix stored in a
 *               compact fashion.
 * \param[in] geom: amrex geometry object.
 * \param[in] mesh: Mesh object; contains information about embedded geometries.
 * \param[in] solution: Solution object.
 * \param[in] X: a MultiFabs containing the solution dG coefficients.
 * \param[in] thr: threshold parameter that triggers the tagging.
 * \param[inout] tags: the TagBoxArray containing the tags of the cells.
 * \param[in] tagged: the value to be assigned to the tagged cells.
 *
*/
void tag_cells_single_phase_domain(const amrex::Real rho, const amrex::Real * c,
                                   const amrex::Geometry & geom, const amrex::dG::Mesh & mesh, const amrex::dG::Solution & solution,
                                   const amrex::MultiFab & X, const amrex::Real thr,
                                   amrex::TagBoxArray & tags, const char tagged)
{
    // PARAMETERS -----------------------------------------------------
    // GRID
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    // ORDER
    const int sp = solution.params.space_p;

    // DOMAINS
    const int dom = 0;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    amrex::MultiFab sensor(X.boxarray, X.distributionMap, 2, 1);
    // ----------------------------------------------------------------

    // INITIALIZATION -------------------------------------------------
    sensor = 0.0;
    // ----------------------------------------------------------------

    // EVAL THE SENSOR ------------------------------------------------
#ifdef AMREX_USE_GPU
    const amrex::Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
    const amrex::Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif
    // ----------------------------------------------------------------

    // TAG THE CELLS --------------------------------------------------
    for (amrex::MFIter mfi(sensor); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);

        amrex::Array4<char const> const & tags_fab = tags.array(mfi);
        
        amrex::Array4<amrex::Real> const & sensor_fab = sensor.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_NOT_EMPTY(etype) && (tags_fab(i,j,k) != tagged))
            {
                // LOCAL PARAMETERS
                const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                          prob_lo[1]+j*dx[1],
                                                                          prob_lo[2]+k*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,amrex::dG::CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,amrex::dG::CELL_DOM_QUAD_POS(dom));
                const amrex::Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                amrex::Real x[AMREX_SPACEDIM], w;
                amrex::Real U[N_VS];
                amrex::Real val, norm;
                amrex::Real integrand_val, integrand_norm;

                // UNKNOWN FIELDS
                amrex::dG::space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);

                // EVAL THE INTEGRAL
                val = norm = 0.0;
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // Solution at (t, x)
                    bfu.eval(x, 0, N_VS, U);

                    integrand_val = eval_total_energy_compact_c(rho, c, U);
                    integrand_norm = 1.0;

                    val += integrand_val*w;
                    norm += integrand_norm*w;
                }

                sensor_fab(i,j,k,0) = norm;
                sensor_fab(i,j,k,1) = val;
            }
        });
        amrex::Gpu::synchronize();
    }
    sensor.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------

    // TAG THE CELLS --------------------------------------------------
    for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const amrex::Box & bx = mfi.validbox();

        amrex::Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
        amrex::Array4<amrex::Real const> const & sensor_fab = sensor.array(mfi);

        amrex::Array4<char> const & tags_fab = tags.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // ELEMENT TYPE
            const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

            if (amrex::dG::ELM_IS_VALID(etype) && (tags_fab(i,j,k) != tagged))
            {
                // LOCAL VARIABLES
                amrex::Real val, norm;

                norm = sensor_fab(i,j,k,0);
                val = sensor_fab(i,j,k,1);

                // Handle extended elements
                for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                {
                    const int nbr_i = i+amrex::dG::tables::extended_stencil_i[n];
                    const int nbr_j = j+amrex::dG::tables::extended_stencil_j[n];
                    const int nbr_k = k+amrex::dG::tables::extended_stencil_k[n];
                    const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,amrex::dG::ELM_TYPE(dom));
                    
                    if (amrex::dG::CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                    {
                        norm += sensor_fab(nbr_i,nbr_j,nbr_k,0);
                        val += sensor_fab(nbr_i,nbr_j,nbr_k,1);
                    }
                }

                // Tag cells
                val = val/norm;
                if (val > thr)
                {
                    tags_fab(i,j,k) = tagged;
                }
            }
        });
        amrex::Gpu::synchronize();
        
    }
    tags.FillBoundary(geom.periodicity());
    // ----------------------------------------------------------------
}
// ####################################################################

} // namespace elastic_solid