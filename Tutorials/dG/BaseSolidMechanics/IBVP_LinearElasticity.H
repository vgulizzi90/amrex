//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_LinearElasticity.H
 * \brief Contains base functions for linear elasticity.
*/

// BLAS/LAPACK ROUTINES ###############################################
extern "C"
{
    
    int dpotrf_(char *, int *, double *, int *, int *);
    int dpotri_(char *, int *, double *, int *, int *);
    int dgetrf_(int *, int *, double *, int *, int *, int *);
    int dgetrs_(char *, int *, int *, double *, int *, int *, double *, int *, int *);
    int dgeev_(char *, char *, int *, double *, int *, double *, double *, double *, int *, double *, int *, double *, int *, int *);
}
// ####################################################################


namespace linear_elasticity
{
// CHANGE OF REFERENCE SYSTEM #########################################
/**
 * \brief Transform vector from global reference system to unit normal reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] v: pointer to memory containing the vector components in the global reference system.
 *               system.
 * \param[out] vn: pointer to memory that will contain the vector components in the unit normal
 *                 reference system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void v_to_vn(const amrex::Real * un, const amrex::Real * v, amrex::Real * vn)
{
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "linear_elasticity::v_to_vn - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];

        vn[0] =  c*v[0]+s*v[1];
        vn[1] = -s*v[0]+c*v[1];
        
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

amrex::Print() << "linear_elasticity::v_to_vn - DIM = 3" << std::endl;
exit(-1);
        
#endif
}

/**
 * \brief Transform vector from unit normal reference system to global reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] vn: pointer to memory containing the vector components in the unit normal reference
 *               system.
 * \param[out] v: pointer to memory that will contain the vector components in the global reference
 *                 system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void vn_to_v(const amrex::Real * un, const amrex::Real * vn, amrex::Real * v)
{
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "linear_elasticity::vn_to_v - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];

        v[0] = c*vn[0]-s*vn[1];
        v[1] = s*vn[0]+c*vn[1];
        
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

amrex::Print() << "linear_elasticity::vn_to_v - DIM = 3" << std::endl;
exit(-1);
        
#endif
}

/**
 * \brief Transform stress from global reference system to unit normal reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] S: pointer to memory containing the stress components in the global reference system.
 * \param[out] Sn: pointer to memory that will contain the stress components in the unit normal
 *                 reference system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void S_to_Sn(const amrex::Real * un, const amrex::Real * S, amrex::Real * Sn)
{
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "linear_elasticity::S_to_Sn - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];
        const amrex::Real cs = c*s;

        Sn[0] = c*c*S[0]+s*s*S[1]   +2.0*cs*S[2];
        Sn[1] = s*s*S[0]+c*c*S[1]   -2.0*cs*S[2];
        Sn[2] = -cs*S[0]+ cs*S[1]+(c*c-s*s)*S[2];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        Sn[0] =  cph*cph*sth*sth*S[0]+sph*sph*sth*sth*S[1]+cth*cth*S[2]      +2.0*cth*sph*sth*S[3]      +2.0*cph*cth*sth*S[4]  +2.0*cph*sph*sth*sth*S[5];
        Sn[1] =          sph*sph*S[0]        +cph*cph*S[1]                                                                             -2.0*cph*sph*S[5];
        Sn[2] =  cph*cph*cth*cth*S[0]+cth*cth*sph*sph*S[1]+sth*sth*S[2]      -2.0*cth*sph*sth*S[3]      -2.0*cph*cth*sth*S[4]  +2.0*cph*cth*cth*sph*S[5];
        Sn[3] =      cph*cth*sph*S[0]    -cph*cth*sph*S[1]                           +cph*sth*S[3]              -sph*sth*S[4]-(cph*cph-sph*sph)*cth*S[5];
        Sn[4] = -cph*cph*cth*sth*S[0]-cth*sph*sph*sth*S[1]+cth*sth*S[2]-(cth*cth-sth*sth)*sph*S[3]-(cth*cth-sth*sth)*cph*S[4]  -2.0*cph*cth*sph*sth*S[5];
        Sn[5] =     -cph*sph*sth*S[0]    +cph*sph*sth*S[1]                           +cph*cth*S[3]              -cth*sph*S[4]+(cph*cph-sph*sph)*sth*S[5];
#endif
}

/**
 * \brief Transform stress from unit normal reference system to global reference system.
 * 
 * \param[in] un: pointer to memory containing the unit normal.
 * \param[in] Sn: pointer to memory containing the stress components in the unit normal reference
 *                system.
 * \param[out] S: pointer to memory that will contain the stress components in the global reference
 *                system.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Sn_to_S(const amrex::Real * un, const amrex::Real * Sn, amrex::Real * S)
{
#if (AMREX_SPACEDIM == 1)
amrex::Print() << "linear_elasticity::Sn_to_S - DIM = 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
        const amrex::Real c = un[0], s = un[1];
        const amrex::Real cs = c*s;

        S[0] = c*c*Sn[0]+s*s*Sn[1]   -2.0*cs*Sn[2];
        S[1] = s*s*Sn[0]+c*c*Sn[1]   +2.0*cs*Sn[2];
        S[2] =  cs*Sn[0] -cs*Sn[1]+(c*c-s*s)*Sn[2];
#endif
#if (AMREX_SPACEDIM == 3)
        // We assume un be written as
        // un = n1 = cos(phi)*sin(theta)
        //      n2   sin(phi)*sin(theta)
        //      n3   cos(theta)
        const amrex::Real theta = std::acos(un[2]);
        const amrex::Real phi = std::atan2(un[1], un[0]);
        const amrex::Real cth = std::cos(theta);
        const amrex::Real sth = std::sin(theta);
        const amrex::Real cph = std::cos(phi);
        const amrex::Real sph = std::sin(phi);

        S[0] = cph*cph*sth*sth*Sn[0]+sph*sph*Sn[1]+cph*cph*cth*cth*Sn[2]      +2.0*cph*cth*sph*Sn[3]  -2.0*cph*cph*cth*sth*Sn[4]      -2.0*cph*sph*sth*Sn[5];
        S[1] = sph*sph*sth*sth*Sn[0]+cph*cph*Sn[1]+cth*cth*sph*sph*Sn[2]      -2.0*cph*cth*sph*Sn[3]  -2.0*cth*sph*sph*sth*Sn[4]      +2.0*cph*sph*sth*Sn[5];
        S[2] =         cth*cth*Sn[0]                      +sth*sth*Sn[2]                                      +2.0*cth*sth*Sn[4];
        S[3] =     cth*sph*sth*Sn[0]                  -cth*sph*sth*Sn[2]              +cph*sth*Sn[3]+(sth*sth-cth*cth)*sph*Sn[4]              +cph*cth*Sn[5];
        S[4] =     cph*cth*sth*Sn[0]                  -cph*cth*sth*Sn[2]              -sph*sth*Sn[3]+(sth*sth-cth*cth)*cph*Sn[4]              -cth*sph*Sn[5];
        S[5] = cph*sph*sth*sth*Sn[0]-cph*sph*Sn[1]+cph*cth*cth*sph*Sn[2]+(sph*sph-cph*cph)*cth*Sn[3]  -2.0*cph*cth*sph*sth*Sn[4]+(cph*cph-sph*sph)*sth*Sn[5];
#endif
}
// ####################################################################



// COMPLIANCE/STIFFNESS MATRICES ######################################
/**
 * \brief Fill memory pointed to by s with coefficients of the elasticity compliance matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] s: pointer to memory that will contain the coefficients of the elasticity compliance
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_compliance_matrix_2d_isotropic(const amrex::Real E, const amrex::Real nu, amrex::Real * s)
{
    const amrex::Real sii = 1.0/E;
    const amrex::Real sij = -nu/E;
    const amrex::Real invG = 2.0*(1.0+nu)*sii;

    std::fill(s, s+3*3, 0.0);
    s[0+0*3] = sii; s[0+1*3] = sij;
    s[1+0*3] = sij; s[1+1*3] = sii;
    s[2+2*3] = invG;

    //amrex::Print() << "get_compliance_matrix_2d_isotropic - s: " << std::endl;
    //amrex::dG::io::print_real_array_2d(3, 3, s);
}

/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_2d_isotropic(const amrex::Real E, const amrex::Real nu, amrex::Real * c)
{
    char uplo = 'U';
    int n = 3;
    int info;

    get_compliance_matrix_2d_isotropic(E, nu, c);

    dpotrf_(&uplo, &n, c, &n, &info);
    dpotri_(&uplo, &n, c, &n, &info);
    for (int i = 1; i < 3; ++i)
    for (int j = 0; j < i; ++j)
    {
        c[i+j*3] = c[j+i*3];
    }
}

/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] (c11, c22, c12, c66): Elastic constants
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_2d_orthotropic(const amrex::Real c11, const amrex::Real c22, const amrex::Real c12, const amrex::Real c66, amrex::Real * c)
{
    std::fill(c, c+3*3, 0.0);
    c[0+0*3] = c11; c[0+1*3] = c12;
    c[1+0*3] = c12; c[1+1*3] = c22;
    c[2+2*3] = c66;

    //amrex::Print() << "get_stiffness_matrix_2d_orthotropic - c: " << std::endl;
    //amrex::dG::io::print_real_array_2d(3, 3, c);
}

/**
 * \brief Fill memory pointed to by s with coefficients of the elasticity compliance matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] s: pointer to memory that will contain the coefficients of the elasticity compliance
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_compliance_matrix_3d_isotropic(const amrex::Real E, const amrex::Real nu, amrex::Real * s)
{
    const amrex::Real sii = 1.0/E;
    const amrex::Real sij = -nu/E;
    const amrex::Real invG = 2.0*(1.0+nu)*sii;

    std::fill(s, s+6*6, 0.0);
    s[0+0*6] = sii; s[0+1*6] = sij; s[0+2*6] = sij;
    s[1+0*6] = sij; s[1+1*6] = sii; s[1+2*6] = sij;
    s[2+0*6] = sij; s[2+1*6] = sij; s[2+2*6] = sii;
    s[3+3*6] = invG;
    s[4+4*6] = invG;
    s[5+5*6] = invG;

    amrex::Print() << "get_compliance_matrix_3d_isotropic - s: " << std::endl;
    amrex::dG::io::print_real_array_2d(6, 6, s);
}

/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_3d_isotropic(const amrex::Real E, const amrex::Real nu, amrex::Real * c)
{
    char uplo = 'U';
    int n = 6;
    int info;

    get_compliance_matrix_3d_isotropic(E, nu, c);

    dpotrf_(&uplo, &n, c, &n, &info);
    dpotri_(&uplo, &n, c, &n, &info);
    for (int i = 1; i < 6; ++i)
    for (int j = 0; j < i; ++j)
    {
        c[i+j*6] = c[j+i*6];
    }
}

/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] (c11, c22, c33, c23, c13, c12, c44, c55, c66): Elastic constants
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_3d_orthotropic(const amrex::Real c11, const amrex::Real c22, const amrex::Real c33,
                                         const amrex::Real c23, const amrex::Real c13, const amrex::Real c12,
                                         const amrex::Real c44, const amrex::Real c55, const amrex::Real c66,
                                         amrex::Real * c)
{
    std::fill(c, c+6*6, 0.0);
    c[0+0*6] = c11; c[0+1*6] = c12; c[0+2*6] = c13;
    c[1+0*6] = c12; c[1+1*6] = c22; c[1+2*6] = c23;
    c[2+0*6] = c13; c[2+1*6] = c23; c[2+2*6] = c33;
    c[3+3*6] = c44;
    c[4+4*6] = c55;
    c[5+5*6] = c66;
}

/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_2d_isotropic_plane_strain(const amrex::Real E, const amrex::Real nu, amrex::Real * c)
{
    amrex::Real c3D[6*6];

    get_stiffness_matrix_3d_isotropic(E, nu, c3D);

    std::fill(c, c+3*3, 0.0);
    c[0+0*3] = c3D[0+0*6]; c[0+1*3] = c3D[0+1*6]; c[0+2*3] = c3D[0+5*6];
    c[1+0*3] = c3D[1+0*6]; c[1+1*3] = c3D[1+1*6]; c[1+2*3] = c3D[1+5*6];
    c[2+0*3] = c3D[5+0*6]; c[2+1*3] = c3D[5+1*6]; c[2+2*3] = c3D[5+5*6];
}


/**
 * \brief Fill memory pointed to by c with coefficients of the elasticity stiffness matrix.
 *
 * \param[in] E: Young's modulus.
 * \param[in] nu: Poisson's ratio.
 * \param[out] c: pointer to memory that will contain the coefficients of the elasticity stiffness
 *                matrix.
 *
*/
AMREX_FORCE_INLINE
void get_stiffness_matrix_2d_isotropic_plane_stress(const amrex::Real E, const amrex::Real nu, amrex::Real * c)
{
    amrex::Real c3D[6*6];
    char uplo = 'U';
    int n = 6;
    int info;

    get_compliance_matrix_3d_isotropic(E, nu, c3D);
    c3D[0+2*6] = 0.0;
    c3D[1+2*6] = 0.0;
    c3D[2+0*6] = 0.0;
    c3D[2+1*6] = 0.0;

    dpotrf_(&uplo, &n, c3D, &n, &info);
    dpotri_(&uplo, &n, c3D, &n, &info);
    for (int i = 1; i < 6; ++i)
    for (int j = 0; j < i; ++j)
    {
        c3D[i+j*6] = c3D[j+i*6];
    }

    std::fill(c, c+3*3, 0.0);
    c[0+0*3] = c3D[0+0*6]; c[0+1*3] = c3D[0+1*6]; c[0+2*3] = c3D[0+5*6];
    c[1+0*3] = c3D[1+0*6]; c[1+1*3] = c3D[1+1*6]; c[1+2*3] = c3D[1+5*6];
    c[2+0*3] = c3D[5+0*6]; c[2+1*3] = c3D[5+1*6]; c[2+2*3] = c3D[5+5*6];
}
// ####################################################################



// WAVE SPEED #########################################################
// ####################################################################

} // namespace linear_elasticity