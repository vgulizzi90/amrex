//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Geometry.H
 * \brief Contains constants and data structures for geometry generation.
*/

#define N_INT_PARAMS 1
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_thickness(const amrex::Real t, const amrex::Real x) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.126;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1015;
        const amrex::Real x05 = std::sqrt(x);
        const amrex::Real x2 = x*x;
        const amrex::Real x3 = x2*x;
        const amrex::Real x4 = x3*x;
        const amrex::Real res = 5.0*t*(a05*x05+a1*x+a2*x2+a3*x3+a4*x4);
        return res;
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_thickness_derivative(const amrex::Real t, const amrex::Real x) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.126;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1015;
        const amrex::Real x05 = std::sqrt(x);
        const amrex::Real x2 = x*x;
        const amrex::Real x3 = x2*x;
        const amrex::Real res = 5.0*t*(a05*0.5/x05+a1+a2*2.0*x+a3*3.0*x2+a4*4.0*x3);
        return res;
    }
    
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
        else if (this->eb_flag == 40012)
        {
            const amrex::Real max_th = 0.12;
            const amrex::Real * LE = this->params;
            const amrex::Real c = this->params[2];
            const amrex::Real alpha = this->params[3]*M_PI/180.0;
            const amrex::Real ca = std::cos(alpha);
            const amrex::Real sa = std::sin(alpha);
            const amrex::Real dx[AMREX_SPACEDIM] = {x[0]-LE[0], x[1]-LE[1]};
            const amrex::Real xr[AMREX_SPACEDIM] = {dx[0]*ca-dx[1]*sa, dx[0]*sa+dx[1]*ca};
            const amrex::Real xs[AMREX_SPACEDIM] = {xr[0]/c, xr[1]/c};
            const amrex::Real xTE = 1.0;
            const amrex::Real thTE = this->NACA4_thickness(max_th, xTE);
            const amrex::Real phTE = -std::atan(this->NACA4_thickness_derivative(max_th, xTE));
            const amrex::Real rTE = thTE/std::cos(phTE);
            const amrex::Real cTE = xTE-rTE*std::sin(phTE);

            if ((xs[0]-cTE) >= std::tan(phTE)*std::abs(xs[1]))
            {
                const amrex::Real dxs[AMREX_SPACEDIM] = {xs[0]-cTE, xs[1]};
                const amrex::Real rxs = std::sqrt(dxs[0]*dxs[0]+dxs[1]*dxs[1]);
                const amrex::Real d = rTE-rxs;

                PHI[0] = d;

/*
amrex::Print() << "1 xs: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xs);
amrex::Print() << "1 dxs: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, dxs);
amrex::Print() << "1 rxs: " << rxs << std::endl;
amrex::Print() << "1 rTE: " << rTE << std::endl;
amrex::Print() << "1 d: " << d << std::endl;
exit(-1);
*/
            }
            else
            {
                amrex::DG::nurbs::curve<22, 18, 3> curve;
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
                const amrex::Real W[19] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                           1.0, 1.0, 1.0};
                curve.set(U, P, W);

                const amrex::Real tol = 1.0e-12*c;
                amrex::Real d2, u, dkx[AMREX_SPACEDIM*2], tmp, un[2];
                d2 = curve.distance_from(xs, tol, &u);
                curve.eval_der(u, 1, dkx);
                tmp = 1.0/std::sqrt(dkx[AMREX_SPACEDIM]*dkx[AMREX_SPACEDIM]+dkx[1+AMREX_SPACEDIM]*dkx[1+AMREX_SPACEDIM]);
                un[0] = -dkx[1+AMREX_SPACEDIM]*tmp;
                un[1] = +dkx[0+AMREX_SPACEDIM]*tmp;
                
                PHI[0] = std::sqrt(d2);
                if (((dkx[0]-xs[0])*un[0]+(dkx[1]-xs[1])*un[1]) > 0.0)
                {
                    PHI[0] = -PHI[0];
                }
/*
amrex::Print() << "2 xs: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xs);
amrex::Print() << "dkx: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, dkx);
amrex::Print() << "PHI: " << PHI[0] << std::endl;
exit(-1);
*/
            }
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
        const int mask_n_grow = 1;
        const int X_n_grow = 1;
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
        this->masks[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, 1, mask_n_grow);
        this->masks[lev] = __DG_CELL_UNMASKED__;
        // ------------------------------------------------------------

        // SOLUTION MULTIFAB ------------------------------------------
        const int p = level_inputs.dG.space_p;
        const int X_n_comp = N_SOL*(AMREX_D_TERM((1+p),*(1+p),*(1+p)));

        this->Xs[lev].define(this->meshes[lev]->cc_ba, this->meshes[lev]->dm, X_n_comp, X_n_grow);
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

        // PARAMETERS -------------------------------------------------
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->meshes[lev]->geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->meshes[lev]->geom.ProbLoArray();
        const int dom = 0;
        const char tagged = amrex::TagBox::SET;
        // ------------------------------------------------------------

        // SIMPLE EB WITH AMR TEST ------------------------------------
        if (this->GG.eb_flag == 102)
        {
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
        // ------------------------------------------------------------

        // NACA 0012 AIRFOIL ------------------------------------------
        if (this->GG.eb_flag == 40012)
        {
            const amrex::Real max_th = 0.12;
            const amrex::Real LE[AMREX_SPACEDIM] = {this->GG.params[0], this->GG.params[1]};
            const amrex::Real c = this->GG.params[2];
            const amrex::Real alpha = this->GG.params[3]*M_PI/180.0;
            const amrex::Real ca = std::cos(alpha);
            const amrex::Real sa = std::sin(alpha);
            const amrex::Real TE[AMREX_SPACEDIM] = {LE[0]+c*ca, LE[1]-c*sa};

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
                        const amrex::Real xx[AMREX_SPACEDIM] = {xc[0]-LE[0], xc[1]-LE[1]};
                        const amrex::Real xr[AMREX_SPACEDIM] = {xx[0]*ca-xx[1]*sa, xx[0]*sa+xx[1]*ca};
                        const amrex::Real xs[AMREX_SPACEDIM] = {xr[0]/c, xr[1]/c};
                        
                        if (lev == 0)
                        {
                            if ((xs[0] > 0.0) && (xs[0] < 1.0) && (std::abs(xs[1]) < (c*max_th+delta)))
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                        else if (lev == 1)
                        {
                            const amrex::Real dxLE[AMREX_SPACEDIM] = {xc[0]-LE[0], xc[1]-LE[1]};
                            const amrex::Real dxTE[AMREX_SPACEDIM] = {xc[0]-TE[0], xc[1]-TE[1]};
                            const amrex::Real dLE = std::sqrt(dxLE[0]*dxLE[0]+dxLE[1]*dxLE[1]);
                            const amrex::Real dTE = std::sqrt(dxTE[0]*dxTE[0]+dxTE[1]*dxTE[1]);

                            if ((dLE < delta) || (dTE < 2.0*delta))
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                        else if (lev == 2)
                        {
                            const amrex::Real dxTE[AMREX_SPACEDIM] = {xc[0]-TE[0], xc[1]-TE[1]};
                            const amrex::Real dTE = std::sqrt(dxTE[0]*dxTE[0]+dxTE[1]*dxTE[1]);

                            if (dTE < 2.0*delta)
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