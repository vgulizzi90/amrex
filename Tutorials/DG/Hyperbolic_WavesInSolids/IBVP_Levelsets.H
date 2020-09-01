/* TYPICAL LEVEL SET FUNCTIONS */


// SELECT THE WHOLE GRID ##############################################
AMREX_GPU_HOST_DEVICE
amrex::Real F_PHI_WHOLE_GRID(const amrex::Real t, const amrex::Real * x)
{
    return -1.0;
}
// ####################################################################



// 1D/2D/3D SPHERE ####################################################
AMREX_GPU_HOST_DEVICE
amrex::Real F_PHI_SPHERE(const amrex::Real t, const amrex::Real * x,
                         const amrex::Real R, const amrex::Real * O)
{
    amrex::Real phi;
    
    phi = -R*R;
    AMREX_D_TERM
    (
        phi += (x[0]-O[0])*(x[0]-O[0]);,
        phi += (x[1]-O[1])*(x[1]-O[1]);,
        phi += (x[2]-O[2])*(x[2]-O[2]);
    )

    return phi;
}
// ####################################################################



// IMPLICIT REGIONS THAT ARE PERIODIC IN [0,1]^d INTERVAL #############
AMREX_GPU_HOST_DEVICE
amrex::Real F_PHI_1_CC(const amrex::Real t, const amrex::Real * x,
                      const amrex::Real a)
{
    const amrex::Real phi = std::cos(2.0*M_PI*x[0])*std::cos(2.0*M_PI*x[1])-a;

    return phi;
}

AMREX_GPU_HOST_DEVICE
amrex::Real F_PHI_1_CCC(const amrex::Real t, const amrex::Real * x,
                       const amrex::Real a)
{
    const amrex::Real phi = -std::cos(2.0*M_PI*x[0])*std::cos(2.0*M_PI*x[1])*std::cos(2.0*M_PI*x[2])-a;

    return phi;
}
// ####################################################################