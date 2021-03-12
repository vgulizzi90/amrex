//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_BodyInFreeStream.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_Base.H"

#define N_PARAMS 8

#define RHO_INF 1.0
#define P_INF 1.0

// ####################################################################
// AUXILIARY FUNCTIONS FOR NACA AIRFOILS ##############################
// ####################################################################
#define NACA_ZERO_TE_THICKNESS

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real NACA4_thickness(const amrex::Real t, const amrex::Real x)
{
    const amrex::Real a05 = 0.2969;
    const amrex::Real a1 = -0.126;
    const amrex::Real a2 = -0.3516;
    const amrex::Real a3 = 0.2843;
#ifdef NACA_ZERO_TE_THICKNESS
    const amrex::Real a4 = -0.1036;
#else
    const amrex::Real a4 = -0.1015;
#endif
    const amrex::Real x05 = std::sqrt(x);
    const amrex::Real x2 = x*x;
    const amrex::Real x3 = x2*x;
    const amrex::Real x4 = x3*x;
    const amrex::Real res = 5.0*t*(a05*x05+a1*x+a2*x2+a3*x3+a4*x4);
    return res;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real NACA4_thickness_derivative(const amrex::Real t, const amrex::Real x)
{
    const amrex::Real a05 = 0.2969;
    const amrex::Real a1 = -0.126;
    const amrex::Real a2 = -0.3516;
    const amrex::Real a3 = 0.2843;
#ifdef NACA_ZERO_TE_THICKNESS
    const amrex::Real a4 = -0.1036;
#else
    const amrex::Real a4 = -0.1015;
#endif
    const amrex::Real x05 = std::sqrt(x);
    const amrex::Real x2 = x*x;
    const amrex::Real x3 = x2*x;
    const amrex::Real res = 5.0*t*(a05*0.5/x05+a1+a2*2.0*x+a3*3.0*x2+a4*4.0*x3);
    return res;
}
// ####################################################################
// ####################################################################


// ####################################################################
// IDEAL GAS: BODY-IN-FREE-STREAM PROBLEM #############################
// ####################################################################
/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct IDEAL_GAS
:
public IDEAL_GAS_BASE
{
    // DATA MEMBERS ===================================================
    amrex::Real params[N_PARAMS];

    amrex::DG::nurbs::curve<22, 18, 3> airfoil;
    // ================================================================
    // NOTE: params must store:
    //       - params[0]: free-stream Mach number
    //       - params[1]: angle of attack
    //       - params[2]: angle of sideslip (used in 3D only)
    //
    //       if (int_params[0] == -1)
    //       - params[AMREX_SPACEDIM:]: not used
    //       if (int_params[0] == 0 and AMREX_SPACEDIM == 2)
    //       - params[2:3]: axes of the ellipse
    //       - params[4:5]: center of the ellipse
    //       if int_params[0] == 0 and AMREX_SPACEDIM == 3
    //       - params[2:4]: axes of the ellipsoid
    //       - params[5:7]: center of the ellipsoid
    // ================================================================


    // CONSTRUCTOR ====================================================
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS(const amrex::Vector<int> & input_int_params, const amrex::Vector<amrex::Real> & input_params)
    :
    IDEAL_GAS_BASE(input_int_params, input_params)
    {
        std::fill(this->params, this->params+N_PARAMS, 0.0);
        std::copy(input_params.begin()+1, input_params.end(), this->params);

        if (this->eb_flag == 40012)
        {
#ifdef NACA_ZERO_TE_THICKNESS
            const amrex::Real U[23] = {0.0, 0.0, 0.0, 0.0,
                                       0.2977087208638509, 0.3843938377101769, 0.4382379884096275, 0.4694002989606703, 0.48606644816551575,
                                       0.4943673550816257, 0.49816789947059603, 0.49999999999999983, 0.5018321005294037, 0.505632644918374,
                                       0.5139335518344841, 0.5305997010393295, 0.5617620115903723, 0.615606162289823, 0.7022912791361491,
                                       1.0, 1.0, 1.0, 1.0};
            const amrex::Real P[38] = { 1.000000000000e+00,  0.0,
                                        7.992224113906e-01,  2.632817543000e-02,
                                        5.400813127019e-01,  5.802806536048e-02,
                                        2.413898421319e-01,  6.131522034020e-02,
                                        1.253653766155e-01,  5.226196458736e-02,
                                        5.685791305132e-02,  3.900686578504e-02,
                                        2.134249877609e-02,  2.554892455391e-02,
                                        4.949854501821e-03,  1.378114993833e-02,
                                        4.555147521226e-04,  5.075536309260e-03,
                                       -1.481628667543e-04,  0.0,
                                        4.555147521226e-04, -5.075536309260e-03,
                                        4.949854501821e-03, -1.378114993833e-02,
                                        2.134249877609e-02, -2.554892455391e-02,
                                        5.685791305132e-02, -3.900686578504e-02,
                                        1.253653766154e-01, -5.226196458736e-02,
                                        2.413898421319e-01, -6.131522034020e-02,
                                        5.400813127019e-01, -5.802806536048e-02,
                                        7.992224113906e-01, -2.632817543000e-02,
                                        1.000000000000e+00,  0.0};
#else
            const amrex::Real U[23] = {0.0, 0.0, 0.0, 0.0,
                                       0.2976842268420151, 0.3843788790980685, 0.43823001726419275, 0.46939636558055553, 0.4860646589065395,
                                       0.49436663185643037, 0.4981676642320874, 0.5, 0.5018323357679125, 0.5056333681435696, 0.5139353410934604,
                                       0.5306036344194445, 0.5617699827358071, 0.6156211209019314, 0.7023157731579848,
                                       1.0, 1.0, 1.0, 1.0};
            const amrex::Real P[38] = { 1.000000000000e+00, 1.260000000000e-03,
                                        7.991439120221e-01, 2.662457739804e-02,
                                        5.400351935227e-01, 5.806503621175e-02,
                                        2.413886217786e-01, 6.131697921476e-02,
                                        1.253654544341e-01, 5.226207235022e-02,
                                        5.685794099719e-02, 3.900687707546e-02,
                                        2.134250149417e-02, 2.554892638723e-02,
                                        4.949854573638e-03, 1.378115007480e-02,
                                        4.555147524017e-04, 5.075536311605e-03,
                                        -1.481628667550e-04, 0.0,
                                        4.555147524016e-04, -5.075536311605e-03,
                                        4.949854573638e-03, -1.378115007480e-02,
                                        2.134250149417e-02, -2.554892638723e-02,
                                        5.685794099719e-02, -3.900687707546e-02,
                                        1.253654544341e-01, -5.226207235022e-02,
                                        2.413886217786e-01, -6.131697921476e-02,
                                        5.400351935227e-01, -5.806503621175e-02,
                                        7.991439120221e-01, -2.662457739804e-02,
                                        1.000000000000e+00, -1.260000000000e-03};
#endif
            const amrex::Real W[19] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0};
            this->airfoil.set(U, P, W);
        }
    }
    // ================================================================
    // NOTE: input_int_params[0] must contain an integer describing the
    //       embedded geometry; similarly, input_params[0] must contain
    //       the specific heats ratio. They are stored when the
    //       constructor IDEAL_GAS_BASE is called.
    // ================================================================


    // LEVEL SET FUNCTIONS ============================================
    /**
     * \brief Fill PHI with values of the level sets at (t,x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] PHI: pointer to memory that will be filled with the value of the level sets at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_PHI(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * PHI) const
    {
        if (this->eb_flag == -1)
        {
            PHI[0] = -1.0;
        }
        else if (this->eb_flag == 0)
        {
            const amrex::Real a[AMREX_SPACEDIM] = {AMREX_D_DECL(this->params[2],
                                                                this->params[3],
                                                                this->params[4])};
            const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(this->params[2+AMREX_SPACEDIM],
                                                                 this->params[3+AMREX_SPACEDIM],
                                                                 this->params[4+AMREX_SPACEDIM])};
            const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL((x[0]-xc[0])/a[0],
                                                                (x[1]-xc[1])/a[1],
                                                                (x[2]-xc[2])/a[2])};

            PHI[0] = 1.0-AMREX_D_TERM(y[0]*y[0],-y[1]*y[1],-y[2]*y[2]);
        }
        else if (this->eb_flag == 40012)
        {
            const amrex::Real max_th = 0.12;
            const amrex::Real * xLE = &this->params[2];
            const amrex::Real xTE[2] = {1.0, NACA4_thickness(max_th, 1.0)};
            const amrex::Real c = this->params[4];
            const amrex::Real alpha = this->params[5]*M_PI/180.0;
            const amrex::Real ca = std::cos(alpha);
            const amrex::Real sa = std::sin(alpha);
            const amrex::Real dx[2] = {x[0]-xLE[0], x[1]-xLE[1]};
            const amrex::Real xr[2] = {dx[0]*ca-dx[1]*sa, dx[0]*sa+dx[1]*ca};
            const amrex::Real xs[2] = {xr[0]/c, xr[1]/c};
            const amrex::Real thTE = xTE[1];
            const amrex::Real phTE = -std::atan(NACA4_thickness_derivative(max_th, xTE[0]));
            const amrex::Real rTE = thTE/std::cos(phTE);
            const amrex::Real cTE = xTE[0]-rTE*std::sin(phTE);

            if ((xs[0]-cTE) >= std::tan(phTE)*std::abs(xs[1]))
            {
#ifdef NACA_ZERO_TE_THICKNESS
                const amrex::Real dxs[2] = {xs[0]-cTE, std::abs(xs[1])};
                const amrex::Real dtp = dxs[0]*std::sin(phTE)+dxs[1]*std::cos(phTE);
                const amrex::Real d = -dtp;
#else
                const amrex::Real dxs[2] = {xs[0]-cTE, xs[1]};
                const amrex::Real rxs = std::sqrt(dxs[0]*dxs[0]+dxs[1]*dxs[1]);
                const amrex::Real d = rTE-rxs;
#endif

                PHI[0] = d;
            }
            else
            {
                const amrex::Real tol = 1.0e-12*c;
                amrex::Real d2, u, dkx[AMREX_SPACEDIM*2], tmp, un[2];
                d2 = this->airfoil.distance_from(xs, tol, &u);
                this->airfoil.eval_der(u, 1, dkx);
                tmp = 1.0/std::sqrt(dkx[AMREX_SPACEDIM]*dkx[AMREX_SPACEDIM]+dkx[1+AMREX_SPACEDIM]*dkx[1+AMREX_SPACEDIM]);
                un[0] = +dkx[1+AMREX_SPACEDIM]*tmp;
                un[1] = -dkx[0+AMREX_SPACEDIM]*tmp;
                
                PHI[0] = std::sqrt(d2);
                if (((xs[0]-dkx[0])*un[0]+(xs[1]-dkx[1])*un[1]) > 0.0)
                {
                    PHI[0] = -PHI[0];
                }

/*
if (PHI[0] > c && xs[0] > 1.0)
{
amrex::Print() << "xs: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xs);
amrex::Print() << "xu: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, dkx);
amrex::Print() << "un: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
amrex::Print() << "((dkx[0]-xs[0])*un[0]+(dkx[1]-xs[1])*un[1]): " << ((dkx[0]-xs[0])*un[0]+(dkx[1]-xs[1])*un[1]) << std::endl;
amrex::Print() << "u: " << u << std::endl;
amrex::Print() << "d2: " << d2 << std::endl;
amrex::Print() << "PHI[0]: " << PHI[0] << std::endl;
exit(-1);
}
*/
            }
        }
        else
        {
amrex::Print() << "IDEAL_GAS.F_PHI" << std::endl;
exit(-1);
        }
    }

    /**
     * \brief Fill phi_info with information about which domain is associated to which levelset.
     * 
     * \param[in] dom: domain index.
     * \param[out] phi_info: pointer to memory that will be filled domain info. More specifically:
     *
     *             phi_info[0] = ph means that the dom-th domain is defined by the ph-th levelset;
     *             phi_info[1] = -1 means that the dom-th domain is defined where the ph-th levelset is negative;
     *             phi_info[1] = +1 means that the dom-th domain is defined where the ph-th levelset is positive;
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_DOM2PHI(const int /*dom*/, int * phi_info) const
    {
        phi_info[0] = 0;
        phi_info[1] = -1;
    }

    /**
     * \brief Returns the domain index dom where the unknown solution field ru is defined.
     * 
     * \param[in] ru: index of the unknown solution field.
     *
     * \return the index of the domain where the unknown solution field ru is defined.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int F_SOL2DOM(const int /*ru*/) const
    {
        return 0;
    }

    /**
     * \brief Returns the index nbr_dom of the domain that is neighbor with the current domain.
     * 
     * \param[in] dom: index of the current domain.
     *
     * \return the index nbr_dom of the domain that is neighbor with domain dom.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int F_DOM2NBRDOM(const int /*dom*/) const
    {
        return -1;
    }
    // ================================================================


    // INITIAL CONDITIONS =============================================
    AMREX_GPU_HOST_DEVICE
    void F_U_inf(amrex::Real * U) const
    {
        const amrex::Real g = this->gamma;

        const amrex::Real M_inf = this->params[0];
        const amrex::Real alpha = this->params[1]*M_PI/180.0;
        const amrex::Real ca = std::cos(alpha);
        const amrex::Real sa = std::sin(alpha);
        const amrex::Real a_inf = std::sqrt(g*P_INF/RHO_INF);
        const amrex::Real V_inf = a_inf*M_inf;

        // STATES
#if (AMREX_SPACEDIM == 2)
        const amrex::Real u1 = V_inf*ca;
        const amrex::Real u2 = V_inf*sa;
#endif
#if (AMREX_SPACEDIM == 3)
#endif
        
        U[RHO] = RHO_INF;
        AMREX_D_TERM
        (
            U[RHOV1] = RHO_INF*u1;,
            U[RHOV2] = RHO_INF*u2;,
            U[RHOV3] = RHO_INF*u3;
        )
        U[TE] = P_INF/(g-1.0)+0.5*RHO_INF*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
    }
    
    /**
     * \brief Fill U with initial conditions of the unknown solution fields.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * /*x*/, amrex::Real * U) const
    {
        // STATES
        this->F_U_inf(U);
    }
    // ================================================================


    // NUMERICAL FLUXES ===============================================
    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int dom,
                  const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        amrex::Real ghost_U[DG_N_SOL];

        this->F_U_inf(ghost_U);
        this->F_NF_ICS(dom, t, x, un, U, ghost_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int dom,
                      const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        this->F_NF_WALL(dom, t, x, un, U, NFn);
    }
    // ================================================================


    // OUTPUT FIELDS ==================================================
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void F_FIELDS_NAMES(amrex::Vector<std::string> & fields_names) const
    {
        fields_names = {"density",
                        AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                        "pressure",
                        "mach",
                        "total_energy",
                        "internal_energy",
                        "entropy"};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0, 0};
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_FIELDS(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                  const amrex::Real * U,
                  amrex::Real * F) const
    {
        const amrex::Real g = this->gamma;

        // PRIMARY VARIABLES
        amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
        this->eval_primary_variables(U, rho, AMREX_D_DECL(u1, u2, u3), p);

        // SOUND SPEED AND MACH NUMBER
        const amrex::Real a = std::sqrt(g*p/rho);
        const amrex::Real M = std::sqrt(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3))/a;

        // OUTPUT FIELDS
        F[RHO] = rho;
        AMREX_D_TERM
        (
            F[RHOV1] = u1;,
            F[RHOV2] = u2;,
            F[RHOV3] = u3;
        )
        F[TE] = p;
        F[TE+1] = M;
        F[TE+2] = U[TE];
        F[TE+3] = p/(rho*(g-1.0));
        F[TE+4] = p/(std::pow(rho, g));
    }
    // ================================================================
};
// ####################################################################
// ####################################################################



// ####################################################################
// IDEAL GAS: AMR #####################################################
// ####################################################################
/**
 * \brief This struct contains constants and methods for adaptive mesh refinement. 
 *
*/
struct AMR_IDEAL_GAS
:
public amrex::DG::AMR::Base
{
    // DATA MEMBERS ===================================================
    IDEAL_GAS IG;
    amrex::Vector<amrex::DG::MatrixFactory *> mfs;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR_IDEAL_GAS()
    :
    Base(),
    IG(this->inputs.problem.int_params, this->inputs.problem.params)
    {
        const int n_levels = this->max_level+1;

        this->mfs.resize(n_levels);
    }

    virtual ~AMR_IDEAL_GAS()
    {
        for (int lev = 0; lev <= this->max_level; ++lev)
        {
            if (this->mfs[lev] != nullptr)
            {
                delete this->mfs[lev];
            }
        }
    }
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int ngr = 1;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        amrex::DG::InputReader level_inputs(false);
        bool im_flag;
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
        // ------------------------------------------------------------

        // SET THE IMPLICIT-MESH FLAG ---------------------------------
        im_flag = true;
        if (this->IG.eb_flag == 40012)
        {
            im_flag = false;
            if (lev > 2)
            {
                im_flag = true;
            }
        }
        // ------------------------------------------------------------

        // IMPLICIT MESH ----------------------------------------------
        this->meshes[lev] = new amrex::DG::ImplicitMesh();
        this->meshes[lev]->define(this->Geom(lev), ba, dm, level_inputs);
        this->meshes[lev]->MakeFromScratch(this->IG, im_flag);
        // ------------------------------------------------------------
        
        // MATRIX FACTORY ---------------------------------------------
        this->matfactories[lev] = new amrex::DG::MatrixFactory();
        this->matfactories[lev]->define(this->Geom(lev), this->meshes[lev]->cc_ba, this->meshes[lev]->dm, level_inputs);
        this->matfactories[lev]->EvalMassMatrices(*this->meshes[lev]);

        if (lev > 0)
        {
            this->matfactories[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                               this->refRatio(lev-1),
                                                               *this->meshes[lev-1],
                                                               *this->matfactories[lev-1]);
        }

        if (lev != this->max_level)
        {
            this->mfs[lev] = new amrex::DG::MatrixFactory();
            this->mfs[lev]->define(this->Geom(lev), level_inputs.dG.space_p-1, level_inputs.dG.space_p);
            this->mfs[lev]->EvalMassMatrices(*this->meshes[lev]);
            this->mfs[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                      amrex::IntVect::TheUnitVector(),
                                                      *this->meshes[lev],
                                                      *this->matfactories[lev]);
        }
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
        this->masks[lev] = __DG_CELL_UNMASKED__;
        // ------------------------------------------------------------

        // SOLUTION MULTIFAB ------------------------------------------
        const int p = level_inputs.dG.space_p;
        const int X_n_comp = DG_N_SOL*(AMREX_D_TERM((1+p),*(1+p),*(1+p)));
        
        this->Xs[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, ngr);
        this->Xs[lev] = 0.0;
        // ------------------------------------------------------------

        // SOLUTION INITIAL CONDITIONS --------------------------------
        const bool include_ghost_cells = ((lev == this->max_level) && (level_inputs.dG.space_p == 0));

        amrex::DG::ProjectInitialConditions(*this->meshes[lev],
                                            *this->matfactories[lev],
                                            DG_N_SOL, this->Xs[lev],
                                            this->IG,
                                            include_ghost_cells);

        if (this->Xs[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: IBVP_BodyInFreeStream.H - MakeNewLevelFromScratch\n";
            msg += "| Xs["+std::to_string(lev)+"] contains nans.\n";
            amrex::Abort(msg);
        }
        // ------------------------------------------------------------

        // SET BOX ARRAY AND DISTRIBUTION MAP -------------------------
        this->SetBoxArray(lev, this->meshes[lev]->cc_ba);
        this->SetDistributionMap(lev, this->meshes[lev]->dm);
        // ------------------------------------------------------------
    }
    // ================================================================


    // TAG CELL FOR REFINEMENT ========================================
    virtual void ErrorEst(int lev, amrex::TagBoxArray & tags, amrex::Real time, int /*ngr*/) override
    {
        // PROFILING ------------------------------------------------------
        BL_PROFILE("ErrorEst(int, amrex::TagBoxArray &, ....)");
        // ----------------------------------------------------------------

        // QUICK RETURN IF WE REACHED THE MAXIMUM LEVEL ---------------
        if (lev >= this->max_level) return;
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->meshes[lev]->geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->meshes[lev]->geom.ProbLoArray();
        const int dom = 0;
        const char untagged = amrex::TagBox::CLEAR;
        const char tagged = amrex::TagBox::SET;
        // ------------------------------------------------------------

        // ELLIPSE ----------------------------------------------------
        tags.setVal(untagged);
        
        if (this->IG.eb_flag == 0)
        {
            const amrex::Real a[AMREX_SPACEDIM] = {AMREX_D_DECL(this->IG.params[2],
                                                                this->IG.params[3],
                                                                this->IG.params[4])};
            const amrex::Real c[AMREX_SPACEDIM] = {AMREX_D_DECL(this->IG.params[2+AMREX_SPACEDIM],
                                                                this->IG.params[3+AMREX_SPACEDIM],
                                                                this->IG.params[4+AMREX_SPACEDIM])};

            const amrex::Real delta = std::sqrt(dx[0]*dx[1]);

            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();

                amrex::Array4<short const> const & eType_fab = this->meshes[lev]->eType.array(mfi);
                amrex::Array4<char> const & tags_fab = tags.array(mfi);
                
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                    if (amrex::DG::ELM_IS_NOT_EMPTY(etype))
                    {
                        // LOCAL PARAMETERS
                        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                             prob_lo[1]+(j+0.5)*dx[1],
                                                                             prob_lo[2]+(k+0.5)*dx[2])};
                        const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL((xc[0]-c[0])/a[0],
                                                                            (xc[1]-c[1])/a[1],
                                                                            (xc[2]-c[2])/a[2])};

                        if (AMREX_D_TERM(y[0]*y[0],+y[1]*y[1],+y[2]*y[2])-1.0 < 2.0*delta)
                        {
                            tags_fab(i,j,k) = tagged;
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
            tags.FillBoundary(this->meshes[lev]->geom.periodicity());
        }
        // ------------------------------------------------------------

        // NACA0012 ---------------------------------------------------
        if (this->IG.eb_flag == 40012)
        {
            const amrex::Real max_th = 0.12;
            const amrex::GpuArray<amrex::Real, 2> xLE = {this->IG.params[2], this->IG.params[3]};
            const amrex::Real c = this->IG.params[4];
            const amrex::Real alpha = this->IG.params[5]*M_PI/180.0;
            const amrex::Real ca = std::cos(alpha);
            const amrex::Real sa = std::sin(alpha);
            const amrex::GpuArray<amrex::Real, 2> xTE = {xLE[0]+c*ca, xLE[1]-c*sa};

            const amrex::Real delta = std::sqrt(dx[0]*dx[1]);

            for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();

                amrex::Array4<short const> const & eType_fab = this->meshes[lev]->eType.array(mfi);
                amrex::Array4<char> const & tags_fab = tags.array(mfi);
                
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                    if (amrex::DG::ELM_IS_NOT_EMPTY(etype) && (tags_fab(i,j,k) != tagged))
                    {
                        // LOCAL PARAMETERS
                        const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                             prob_lo[1]+(j+0.5)*dx[1],
                                                                             prob_lo[2]+(k+0.5)*dx[2])};
                        const amrex::Real xm[AMREX_SPACEDIM] = {xc[0]-0.5*(xLE[0]+xTE[0]), xc[1]-0.5*(xLE[1]+xTE[1])};
                        const amrex::Real d2 = xm[0]*xm[0]+xm[1]*xm[1];
                        
                        if (lev == 0)
                        {
                            if (d2 < 9.0*c*c)
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                        else if (lev == 1)
                        {
                            if (d2 < 6.25*c*c)
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                        else if (lev == 2)
                        {
                            const amrex::Real xr[AMREX_SPACEDIM] = {(xc[0]-xLE[0])*ca-(xc[1]-xLE[1])*sa, (xc[0]-xLE[0])*sa+(xc[1]-xLE[1])*ca};
                            const amrex::Real xs[AMREX_SPACEDIM] = {xr[0]/c, xr[1]/c};

                            if ((xs[0] > -delta) && (xs[0] < 1.0+delta) && (xs[1] > -max_th-delta) && (xs[1] < +max_th+delta))
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                        else if (lev == 3)
                        {
                            const amrex::Real dxLE[AMREX_SPACEDIM] = {xc[0]-xLE[0], xc[1]-xLE[1]};
                            const amrex::Real dxTE[AMREX_SPACEDIM] = {xc[0]-xTE[0], xc[1]-xTE[1]};
                            const amrex::Real d2LE = dxLE[0]*dxLE[0]+dxLE[1]*dxLE[1];
                            const amrex::Real d2TE = dxTE[0]*dxTE[0]+dxTE[1]*dxTE[1];
                            
                            if ((d2LE < 0.0001*c*c) || (d2TE < 0.0001*c*c))
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
            tags.FillBoundary(this->meshes[lev]->geom.periodicity());
        }
        // ------------------------------------------------------------

        // EVAL TROUBLED CELLS ----------------------------------------
        /**/
        this->IG.F_UPDATE_TROUBLED(time,
                                   *this->meshes[lev],
                                   *this->matfactories[lev],
                                   *this->mfs[lev],
                                   this->Xs[lev],
                                   tags,
                                   tagged);
        /**/
        
        /**/
        if ((lev+1) != this->max_level)
        {
            this->IG.F_UPDATE_TAGS(lev,
                                   time,
                                   *this->meshes[lev],
                                   *this->matfactories[lev],
                                   *this->mfs[lev],
                                   this->Xs[lev],
                                   tags,
                                   tagged);
        }
        /**/
        // ------------------------------------------------------------
    }
    // ================================================================

    // MAKE NEW LEVEL FROM A COARSE LEVEL =============================
    virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int ngr = 1;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        amrex::DG::InputReader level_inputs(false);
        bool im_flag;
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
        // ------------------------------------------------------------

        // SET THE IMPLICIT-MESH FLAG ---------------------------------
        im_flag = true;
        if (this->IG.eb_flag == 40012)
        {
            im_flag = false;
            if (lev > 2)
            {
                im_flag = true;
            }
        }
        // ------------------------------------------------------------

        // LEVEL == 0
        if (lev == 0)
        {
amrex::Print() << "AMR_IDEAL_GAS.MakeNewLevelFromCoarse - lev == 0" << std::endl;
exit(-1);
        }
        // LEVEL > 0
        else
        {
            // IMPLICIT MESH ------------------------------------------
            if (this->meshes[lev] == nullptr)
            {
                this->meshes[lev] = new amrex::DG::ImplicitMesh();
            }
            this->meshes[lev]->define(this->Geom(lev), ba, dm, level_inputs);
            this->meshes[lev]->MakeFromCoarse(time, ba, dm, this->refRatio(lev-1), *this->meshes[lev-1], this->IG, im_flag);
            // --------------------------------------------------------

            // MATRIX FACTORY -----------------------------------------
            if (this->matfactories[lev] == nullptr)
            {
                this->matfactories[lev] = new amrex::DG::MatrixFactory();
            }
            this->matfactories[lev]->define(this->Geom(lev), this->meshes[lev]->cc_ba, this->meshes[lev]->dm, level_inputs);
            this->matfactories[lev]->EvalMassMatrices(*this->meshes[lev]);

            if (lev > 0)
            {
                this->matfactories[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                                    this->refRatio(lev-1),
                                                                    *this->meshes[lev-1],
                                                                    *this->matfactories[lev-1]);
            }

            if (lev != this->max_level)
            {
                if (this->mfs[lev] == nullptr)
                {
                    this->mfs[lev] = new amrex::DG::MatrixFactory();
                }
                this->mfs[lev]->define(this->Geom(lev), level_inputs.dG.space_p-1, level_inputs.dG.space_p);
                this->mfs[lev]->EvalMassMatrices(*this->meshes[lev]);
                this->mfs[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                          amrex::IntVect::TheUnitVector(),
                                                          *this->meshes[lev],
                                                          *this->matfactories[lev]);
            }
            // --------------------------------------------------------

            // MASKS --------------------------------------------------
            this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
            this->masks[lev] = __DG_CELL_UNMASKED__;
            // --------------------------------------------------------

            // SOLUTION MULTIFAB --------------------------------------
            const int p = level_inputs.dG.space_p;
            const int X_n_comp = DG_N_SOL*(AMREX_D_TERM((1+p),*(1+p),*(1+p)));

            this->Xs[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, ngr);
            this->Xs[lev] = 0.0;

            const amrex::Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, 0);

            const bool include_ghost_cells = ((lev == this->max_level) && (level_inputs.dG.space_p == 0));
            
            amrex::DG::AMR::Interpolate(DG_N_SOL, Sol2Dom,
                                        *this->meshes[lev-1], *this->matfactories[lev-1], this->Xs[lev-1],
                                        this->refRatio(lev-1),
                                        *this->meshes[lev], *this->matfactories[lev], this->Xs[lev],
                                        include_ghost_cells);
            // --------------------------------------------------------

            // SET BOX ARRAY AND DISTRIBUTION MAP ---------------------
            this->SetBoxArray(lev, this->meshes[lev]->cc_ba);
            this->SetDistributionMap(lev, this->meshes[lev]->dm);
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // REMAKE AN EXISTING LEVEL =======================================
    virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int ngr = 1;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        amrex::DG::InputReader level_inputs(false);
        bool im_flag;
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
        // ------------------------------------------------------------

        // SET THE IMPLICIT-MESH FLAG ---------------------------------
        im_flag = true;
        if (this->IG.eb_flag == 40012)
        {
            im_flag = false;
            if (lev > 2)
            {
                im_flag = true;
            }
        }
        // ------------------------------------------------------------

        // LEVEL == 0
        if (lev == 0)
        {
amrex::Print() << "AMR_IDEAL_GAS.RemakeLevel - lev == 0" << std::endl;
exit(-1);
        }
        // LEVEL > 0
        else
        {
            // IMPLICIT MESH ------------------------------------------
            this->meshes[lev]->Remake(time, ba, dm, this->refRatio(lev-1), *this->meshes[lev-1], this->IG, im_flag);
            // --------------------------------------------------------

            // MATRIX FACTORY -----------------------------------------
            this->matfactories[lev]->EvalMassMatrices(*this->meshes[lev]);

            this->matfactories[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                               this->refRatio(lev-1),
                                                               *this->meshes[lev-1],
                                                               *this->matfactories[lev-1]);

            if (lev != this->max_level)
            {
                this->mfs[lev]->EvalMassMatrices(*this->meshes[lev]);
                this->mfs[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                          amrex::IntVect::TheUnitVector(),
                                                          *this->meshes[lev],
                                                          *this->matfactories[lev]);
            }
            // --------------------------------------------------------

            // MASKS --------------------------------------------------
            this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
            this->masks[lev] = __DG_CELL_UNMASKED__;
            // --------------------------------------------------------

            // SOLUTION MULTIFAB --------------------------------------
            const int X_n_comp = this->Xs[lev].n_comp;
            const amrex::IntVect X_n_grow = this->Xs[lev].n_grow;

            amrex::MultiFab new_X(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, X_n_grow);
            new_X = 0.0;

            const amrex::Gpu::ManagedVector<int> Sol2Dom(DG_N_SOL, 0);

            const bool include_ghost_cells = ((lev == this->max_level) && (level_inputs.dG.space_p == 0));

            amrex::DG::AMR::Interpolate(DG_N_SOL, Sol2Dom,
                                        *this->meshes[lev-1], *this->matfactories[lev-1], this->Xs[lev-1],
                                        this->refRatio(lev-1),
                                        *this->meshes[lev], *this->matfactories[lev], new_X,
                                        include_ghost_cells);
                        
            new_X.ParallelCopy(this->Xs[lev], 0, 0, X_n_comp, 0, 0, this->Geom(lev).periodicity());
            new_X.FillBoundary(this->Geom(lev).periodicity());

            std::swap(this->Xs[lev], new_X);
            // --------------------------------------------------------

            // SET BOX ARRAY AND DISTRIBUTION MAP ---------------------
            this->SetBoxArray(lev, this->meshes[lev]->cc_ba);
            this->SetDistributionMap(lev, this->meshes[lev]->dm);
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // DELETE LEVEL DATA ==============================================
    virtual void ClearLevel(int /*lev*/) override
    {}
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize AMR data.
     * 
    */
    void Init()
    {
        if (this->inputs.restart > 0)
        {
            // INIT THE LEVELS
            for (int lev = 0; lev <= this->max_level; ++lev)
            {
                this->meshes[lev] = new amrex::DG::ImplicitMesh();
                this->matfactories[lev] = new amrex::DG::MatrixFactory();
                if (lev != this->max_level)
                {
                    this->mfs[lev] = new amrex::DG::MatrixFactory();
                }
            }

            // LOAD THE LEVELSET FUNCTIONS AND SOLUTION MULTIFABS
            this->InitMultiFabsFromCheckpointFile();

            // MAKE THE LEVELS
            const int N_PHI = 1;
            const int N_DOM = 1;
            const int ngr = 1;
            int phi_info[2];
            bool im_flag;

            for (int lev = 0; lev <= this->max_level; ++lev)
            {
                amrex::DG::InputReader level_inputs(false);
                this->ExtractLevelInputs(lev, level_inputs);

                im_flag = true;
                if (this->IG.eb_flag == 40012)
                {
                    im_flag = false;
                    if (lev > 2)
                    {
                        im_flag = true;
                    }
                }

                this->meshes[lev]->define(this->Geom(lev), this->grids[lev], this->dmap[lev], level_inputs);
                this->IG.F_DOM2PHI(0, phi_info);
                this->meshes[lev]->Dom2Phi = {phi_info[0], phi_info[1]};
                this->meshes[lev]->Make(N_PHI, N_DOM, im_flag);

                this->matfactories[lev]->define(this->Geom(lev), this->meshes[lev]->cc_ba, this->meshes[lev]->dm, level_inputs);
                this->matfactories[lev]->EvalMassMatrices(*this->meshes[lev]);
                if (lev > 0)
                {
                    this->matfactories[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                                       this->refRatio(lev-1),
                                                                       *this->meshes[lev-1],
                                                                       *this->matfactories[lev-1]);
                }
                if (lev != this->max_level)
                {
                    this->mfs[lev]->define(this->Geom(lev), level_inputs.dG.space_p-1, level_inputs.dG.space_p);
                    this->mfs[lev]->EvalMassMatrices(*this->meshes[lev]);
                    this->mfs[lev]->EvalInterpolationOperator(*this->meshes[lev],
                                                              amrex::IntVect::TheUnitVector(),
                                                              *this->meshes[lev],
                                                              *this->matfactories[lev]);
                }

                this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
                this->masks[lev] = __DG_CELL_UNMASKED__;
            }
        }
        else
        {
            const amrex::Real t = 0.0;
            this->InitFromScratch(t);
        }

        // UPDATE MASKS ----
        this->UpdateMasks();
        // -----------------
    }
    // ================================================================
};
// ####################################################################
// ####################################################################