//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_BodyInFreeStream.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_Base.H"

#define N_PARAMS 4

// ####################################################################
// IDEAL GAS: SUPERSONIC VORTEX PROBLEM ###############################
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
    // ================================================================
    // NOTE: params must store:
    //       - params[0]: rho_inner, i.e. density at the inner radius
    //                    of the circular annulus;
    //       - params[1]: Mach_inner, i.e. Mach number at the inner
    //                    radius of the circular annulus;
    //       - params[2-3]: r_inner, r_outer, i.e. inner and outer
    //                      radii of the circular annulus;
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
        const amrex::Real ri = this->params[2];
        const amrex::Real ro = this->params[3];
        const amrex::Real ri2 = ri*ri;
        const amrex::Real ro2 = ro*ro;
        const amrex::Real ra2 = 0.25*(ri2+ro2+2.0*ri*ro);

        const amrex::Real x2 = x[0]*x[0];
        const amrex::Real y2 = x[1]*x[1];
        const amrex::Real r2 = x2+y2;

        if (r2 < ra2)
        {
            PHI[0] = ri2-r2;
        }
        else
        {
            PHI[0] = r2-ro2;
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


    // EXACT SOLUTION =================================================
    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * U) const
    {
        // PARAMETERS
        const amrex::Real g = this->gamma;
        const amrex::Real g2 = 0.5*(g-1.0);
        const amrex::Real ig = 1.0/(g-1.0);
        const amrex::Real rho_i = this->params[0];
        const amrex::Real M_i = this->params[1];

        const amrex::Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]);
        const amrex::Real cth = x[0]/r;
        const amrex::Real sth = x[1]/r;
        const amrex::Real ri = this->params[2];
        const amrex::Real rr2 = (ri/r)*(ri/r);

        const amrex::Real rho = rho_i*std::pow(1.0+g2*M_i*M_i*(1.0-rr2), ig);
        const amrex::Real a_i = 1.0;
        const amrex::Real uth = a_i*M_i*(ri/r);
        const amrex::Real u = -uth*sth;
        const amrex::Real v = uth*cth;
        const amrex::Real p = std::pow(rho, g)/g;

        U[RHO] = rho;
        U[RHOV1] = rho*u;
        U[RHOV2] = rho*v;
        U[TE] = ig*p+0.5*rho*(u*u+v*v);

        /* Constant initial state
        U[RHO] = 1.0;
        U[RHOV1] = 0.0;
        U[RHOV2] = 0.0;
        U[TE] = ig;
        */
    }
    // ================================================================
    // To be used in paraview:
    // density = (1.0+0.5*(1.4-1)*2.25*2.25*(1.0-1/(coordsX*coordsX+coordsY*coordsY)))^(1/(1.4-1))
    // ================================================================


    // INITIAL CONDITIONS =============================================
    /**
     * \brief Fill U with initial conditions of the unknown solution fields.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * x, amrex::Real * U) const
    {
        this->F_EXACT(0.0, x, U);
    }
    // ================================================================


    // ERROR ==========================================================
    /**
     * \brief Compute the error and a normalization value at (t,x).
     * 
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables.
     * \param[out] err_x: error at (t,x).
     * \param[out] norm_x: normalization value at (t,x).
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void F_ERROR(const int /*dom*/,
                 const amrex::Real t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x, amrex::Real & norm_x) const
    {
        // VARIABLES
        amrex::Real ref_U[DG_N_SOL];
        
        // EXACT SOLUTION
        this->F_EXACT(t, x, ref_U);

        norm_x = 1.0;

        err_x = std::abs(ref_U[RHO]-U[RHO]);
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
        // EXACT SOLUTION
        amrex::Real ref_U[DG_N_SOL];
        this->F_EXACT(t, x, ref_U);

        // INFLOW
        if (un[1] < -0.5)
        {
            this->F_NF_INFLOW(dom, t, x, un, U, ref_U, NFn);
        }
        // OUTFLOW
        else if (un[0] < -0.5)
        {
            this->F_NF_OUTFLOW(dom, t, x, un, U, ref_U, NFn);
        }
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
                        "velocity_x", "velocity_y",
                        "pressure",
                        "mach",
                        "total_energy",
                        "internal_energy"};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0, 0, 0, 0, 0, 0, 0};
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
        F[RHOV1] = u1;
        F[RHOV2] = u2;
        F[TE] = p;
        F[TE+1] = M;
        F[TE+2] = U[TE];
        F[TE+3] = p/(rho*(g-1.0));
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
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR_IDEAL_GAS()
    :
    Base(),
    IG(this->inputs.problem.int_params, this->inputs.problem.params)
    {}
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int ngr = 1;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        amrex::DG::InputReader level_inputs(false);
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
        // ------------------------------------------------------------

        // IMPLICIT MESH ----------------------------------------------
        this->meshes[lev] = new amrex::DG::ImplicitMesh();
        this->meshes[lev]->define(this->Geom(lev), ba, dm, level_inputs);
        this->meshes[lev]->MakeFromScratch(this->IG);
        // ------------------------------------------------------------
        
        // MATRIX FACTORY ---------------------------------------------
        this->matfactories[lev] = new amrex::DG::MatrixFactory();
        this->matfactories[lev]->define(this->Geom(lev), ba, dm, level_inputs);
        this->matfactories[lev]->EvalMassMatrices(*this->meshes[lev]);
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
        this->masks[lev] = 0;
        // ------------------------------------------------------------

        // SOLUTION MULTIFAB ------------------------------------------
        const int p = level_inputs.dG.space_p;
        const int X_n_comp = DG_N_SOL*(AMREX_D_PICK(1+p, (1+p)*(1+p), (1+p)*(1+p)*(1+p)));
        
        this->Xs[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, ngr);
        this->Xs[lev] = 0.0;
        // ------------------------------------------------------------

        // SOLUTION INITIAL CONDITIONS --------------------------------
        amrex::DG::ProjectInitialConditions(*this->meshes[lev],
                                            *this->matfactories[lev],
                                            DG_N_SOL, this->Xs[lev],
                                            this->IG);
        // ------------------------------------------------------------
    }
    // ================================================================


    // TAG CELL FOR REFINEMENT ========================================
    virtual void ErrorEst(int /*lev*/, amrex::TagBoxArray & /*tags*/, amrex::Real /*time*/, int /*ngr*/) override
    {
        return;
    }
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize AMR data.
     * 
    */
    void Init()
    {
        // PARAMETERS -------------------------------------------------
        const amrex::Real t = 0.0;
        // ------------------------------------------------------------

        // CALL PARENT CLASS METHOD -----------------------------------
        this->InitFromScratch(t);
        // -----------------------------------------------------------

        // UPDATE MASKS ----
        this->UpdateMasks();
        // -----------------
    }
    // ================================================================
};
// ####################################################################
// ####################################################################