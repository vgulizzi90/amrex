//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_BodyInFreeStream.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_Base.H"

#define N_PARAMS 9

// ####################################################################
// IDEAL GAS: SOD'S TUBE PROBLEM ######################################
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
    //       - params[0]: x1s, the shock location;
    //       - params[1]: Ms, the Mach number of the shock;
    //       - params[2]: rho0, density ahead of the shock;
    //       - params[3]: u0, velocity ahead of the shock;
    //       - params[4]: p0, pressure ahead of the shock;
    //
    //       if (this->eb_flag == -1)
    //       - params[5-8]: not used;
    //       if (this->eb_flag == 0 and AMREX_SPACEDIM == 2)
    //       - params[5]: radius of the embedded circle;
    //       - params[6-7]: coordinates of the center of the embedded
    //                      circle;
    //       - params[8]: not used;
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
        if (this->eb_flag == -1)
        {
            PHI[0] = -1.0;
        }
        else if (this->eb_flag == 0)
        {
            const amrex::Real r = this->params[5];
            const amrex::Real * xc = &this->params[6];

            PHI[0] = r*r-(AMREX_D_TERM((x[0]-xc[0])*(x[0]-xc[0]),+(x[1]-xc[1])*(x[1]-xc[1]),+(x[2]-xc[2])*(x[2]-xc[2])));
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
        const amrex::Real g = this->gamma;
        const amrex::Real g2 = 2.0*g/(g+1.0);
        const amrex::Real gr = (g-1.0)/(g+1.0);
        const amrex::Real x1s = this->params[0];
        const amrex::Real Ms = this->params[1];
        const amrex::Real rho0 = this->params[2];
        const amrex::Real u0 = this->params[3];
        const amrex::Real p0 = this->params[4];
        const amrex::Real a0 = std::sqrt(g*p0/rho0);

        const amrex::Real rhoL = rho0*(g+1.0)*(Ms*Ms/(2.0+(g-1.0)*Ms*Ms));
        const amrex::Real pL = p0*(1.0+g2*(Ms*Ms-1.0));
        const amrex::Real uL = (a0/g)*(pL/p0-1.0)*std::sqrt(g2/(pL/p0+gr));
        //const amrex::Real aL = std::sqrt(g*pL/rhoL);

        {
            if (x[0] > x1s)
            {
                U[RHO] = rho0;
                AMREX_D_TERM
                (
                    U[RHOV1] = rho0*u0;,
                    U[RHOV2] = 0.0;,
                    U[RHOV3] = 0.0;
                )
                U[TE] = p0/(g-1.0)+0.5*rho0*u0*u0;
            }
            else
            {
                U[RHO] = rhoL;
                AMREX_D_TERM
                (
                    U[RHOV1] = rhoL*uL;,
                    U[RHOV2] = 0.0;,
                    U[RHOV3] = 0.0;
                )
                U[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
            }
        }

        /* Constant state
        U[RHO] = rhoL;
        AMREX_D_TERM
        (
            U[RHOV1] = 0.0;,
            U[RHOV2] = 0.0;,
            U[RHOV3] = 0.0;
        )
        U[TE] = pL/(g-1.0);
        */
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
        if (un[0] < -0.5)
        {
            this->F_NF_INFLOW(dom, t, x, un, U, U, NFn);
        }
        else
        {
            this->F_NF_WALL(dom, t, x, un, U, NFn);
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
        if ((this->eb_flag == -1) || (this->eb_flag == 0))
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy"};
        }
        else
        {
amrex::Print() << "IDEAL_GAS.F_FIELDS_NAMES" << std::endl;
exit(-1);
        }
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        if ((this->eb_flag == -1) || (this->eb_flag == 0))
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0};
        }
        else
        {
amrex::Print() << "IDEAL_GAS.F_FIELDS_DOMAINS" << std::endl;
exit(-1);
        }
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
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
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
            msg += "ERROR: IBVP_ShockInteraction.H - MakeNewLevelFromScratch\n";
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
        const int dom = 0;
        const char untagged = amrex::TagBox::CLEAR;
        const char tagged = amrex::TagBox::SET;
        // ------------------------------------------------------------

        // EVAL TROUBLED CELLS ----------------------------------------
        this->IG.F_EVAL_TROUBLED(time,
                                 *this->meshes[lev],
                                 *this->matfactories[lev],
                                 *this->mfs[lev],
                                 this->Xs[lev],
                                 tags,
                                 tagged,
                                 untagged);
        
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
    
    virtual void ManualTagsPlacement(int lev, amrex::TagBoxArray & tags, const amrex::Vector<amrex::IntVect> & bf_lev) override
    {
        // QUICK RETURN IF WE REACHED THE MAXIMUM LEVEL ---------------
        if (lev >= this->max_level) return;
        // ------------------------------------------------------------

        // MAKE TAGGING CONSISTENT BY FURTHER TAGGING
        // i) Small cells merged with tagged extended
        // ii) Extended cells merging with tagged small cells 
        this->FixTagging(*this->meshes[lev], tags, amrex::TagBox::BUF);
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
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
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
            this->meshes[lev]->MakeFromCoarse(time, ba, dm, this->refRatio(lev-1), *this->meshes[lev-1], this->IG);
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
        // ------------------------------------------------------------
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
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
            this->meshes[lev]->Remake(time, ba, dm, this->refRatio(lev-1), *this->meshes[lev-1], this->IG);
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
    {
        // IMPLICIT MESH ----------------------------------------------
        //delete this->meshes[lev];
        //this->meshes[lev] = nullptr;
        // ------------------------------------------------------------
        
        // MATRIX FACTORY ---------------------------------------------
        //delete this->matfactories[lev];
        //this->matfactories[lev] = nullptr;

        //delete this->mfs[lev];
        //this->mfs[lev] = nullptr;
        // ------------------------------------------------------------
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