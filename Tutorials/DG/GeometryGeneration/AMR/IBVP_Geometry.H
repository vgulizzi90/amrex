//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Geometry.H
 * \brief Contains constants and data structures for geometry generation.
*/

#define N_INT_PARAMS 10
#define N_PARAMS 10
#define N_SOL 1

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct GEOMETRY_GENERATOR
{
    // DATA MEMBERS ===================================================
    int eb_flag;
    int int_params[N_INT_PARAMS];
    amrex::Real params[N_PARAMS];
    // ================================================================

    // CONSTRUCTOR ====================================================
    /**
     * \brief This method initializes the IBVP.
    */
    GEOMETRY_GENERATOR(const amrex::Vector<int> & input_int_params, const amrex::Vector<amrex::Real> & input_params)
    {
        this->eb_flag = input_int_params[0];
        
        std::fill(this->int_params, this->int_params+N_INT_PARAMS, 0);
        std::copy(input_int_params.begin()+1, input_int_params.end(), this->int_params);

        std::fill(this->params, this->params+N_PARAMS, 0);
        std::copy(input_params.begin(), input_params.end(), this->params);
    }
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
        else if (this->eb_flag == 102)
        {
            const amrex::Real * a = this->params;
            PHI[0] = a[0]          +a[1]*x[0]          +a[2]*x[0]*x[0]+
                     a[3]*x[1]     +a[4]*x[0]*x[1]     +a[5]*x[0]*x[0]     *x[1]+
                     a[6]*x[1]*x[1]+a[7]*x[0]*x[1]*x[1]+a[8]*x[0]*x[0]*x[1]*x[1];
        }
        else
        {
amrex::Abort("GEOMETRY_GENERATOR.F_PHI");
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
    AMREX_FORCE_INLINE
    amrex::Real phi_shock(const amrex::Real * x) const
    {
        return x[0]-0.5-0.25*(x[1]-0.8)*(x[1]-0.8);
    }
    
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
        U[0] = -0.5*std::sin(M_PI*(x[0]+0.5*x[1]));

        if (this->phi_shock(x) > 0.0)
        {
            U[0] += 0.5;
        }
    }
    // ================================================================

    // SLOPE RECONSTRUCTION FOR LEVELS USING FINITE-VOLUME SCHEMES ====
    void F_R_SLOPES(const amrex::Real /*t*/,
                    const amrex::DG::ImplicitMesh & /*mesh*/,
                    const int /*N_DOM*/,
                    const int /*N_SOL*/,
                    const amrex::MultiFab & /*X*/,
                    amrex::MultiFab & dX) const
    {
        dX = 0.0;
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
        fields_names = {"phi"};
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
        fields_domains = {0};
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
        F[0] = U[0];
    }
    // ================================================================
};


/**
 * \brief This struct contains constants and methods for adaptive mesh refinement. 
 *
*/
struct AMR_GEOMETRY
:
public amrex::DG::AMR::Base
{
    // DATA MEMBERS ===================================================
    GEOMETRY_GENERATOR GG;
    // ================================================================

    // CONSTRUCTOR ====================================================
    AMR_GEOMETRY()
    :
    Base(),
    GG(this->inputs.problem.int_params, this->inputs.problem.params)
    {

    }

    virtual ~AMR_GEOMETRY()
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
        // In the constructor, false means that the input file will not
        // be read.
        // ------------------------------------------------------------

        // INPUTS AT CURRENT LEVEL ------------------------------------
        this->ExtractLevelInputs(lev, level_inputs);
        // ------------------------------------------------------------

        // IMPLICIT MESH ----------------------------------------------
        this->meshes[lev] = new amrex::DG::ImplicitMesh();
        this->meshes[lev]->define(this->Geom(lev), ba, dm, level_inputs);
        this->meshes[lev]->MakeFromScratch(this->GG);
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
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, ngr);
        this->masks[lev] = __DG_CELL_UNMASKED__;
        // ------------------------------------------------------------

        // SOLUTION MULTIFAB ------------------------------------------
        const int p = level_inputs.dG.space_p;
        const int X_n_comp = N_SOL*(AMREX_D_TERM((1+p),*(1+p),*(1+p)));

        this->Xs[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, ngr);
        this->Xs[lev] = 0.0;
        // ------------------------------------------------------------

        // SOLUTION INITIAL CONDITIONS --------------------------------
        const bool include_ghost_cells = ((lev == this->max_level) && (level_inputs.dG.space_p == 0));

        amrex::DG::ProjectInitialConditions(*this->meshes[lev],
                                            *this->matfactories[lev],
                                            N_SOL, this->Xs[lev],
                                            this->GG,
                                            include_ghost_cells);
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
        // PROFILING --------------------------------------------------
        BL_PROFILE("ErrorEst(int, amrex::TagBoxArray &, ....)");
        // ------------------------------------------------------------

        // QUICK RETURN IF WE REACHED THE MAXIMUM LEVEL ---------------
        if (lev >= this->max_level) return;
        // ------------------------------------------------------------

        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->meshes[lev]->geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->meshes[lev]->geom.ProbLoArray();
        const int dom = 0;
        const char tagged = amrex::TagBox::SET;

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

                    const amrex::Real phi = this->GG.phi_shock(xc);
                    
                    if (std::abs(phi) < dx[0])
                    {
                        tags_fab(i,j,k) = tagged;
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
        tags.FillBoundary(this->meshes[lev]->geom.periodicity());
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