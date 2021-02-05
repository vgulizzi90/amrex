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
        // STATES
        const amrex::Real g = this->gamma;
        AMREX_D_TERM
        (
            const amrex::Real u1 = 0.0;,
            const amrex::Real u2 = 0.0;,
            const amrex::Real u3 = 0.0;
        )
        
        U[RHO] = RHO_INF;
        AMREX_D_TERM
        (
            U[RHOV1] = RHO_INF*u1;,
            U[RHOV2] = RHO_INF*u2;,
            U[RHOV3] = RHO_INF*u3;
        )
        U[TE] = P_INF/(g-1.0)+0.5*RHO_INF*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
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
amrex::Print() << "IDEAL_GAS_BASE.F_NF_BCS" << std::endl;
exit(-1);
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
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR_IDEAL_GAS()
    :
    Base(),
    IG(this->inputs.problem.int_params, this->inputs.problem.params)
    {}
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
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

        // MODIFY COARSER LEVEL'S MASK --------------------------------
        if (lev > 0)
        {  
            const amrex::MultiFab & cX = this->Xs[lev-1];
            const amrex::MultiFab & fX = this->Xs[lev];
            const amrex::DG::ImplicitMesh & c_mesh = *this->meshes[lev-1];
            
            this->masks[lev-1] = amrex::makeFineMask(cX, fX, cX.n_grow, this->ref_ratio[lev-1],
                                                     c_mesh.geom.periodicity(),
                                                     __DG_CELL_UNMASKED__, __DG_CELL_MASKED__);
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // TAG CELL FOR REFINEMENT ========================================
    virtual void ErrorEst(int lev, amrex::TagBoxArray & tags, amrex::Real time, int /*ngr*/) override
    {
        // QUICK RETURN IF WE REACHED THE MAXIMUM LEVEL ---------------
        if (lev > this->max_level) return;
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const amrex::DG::ImplicitMesh & mesh = *this->meshes[lev];
        const amrex::DG::MatrixFactory & matfactory = *this->matfactories[lev];
        const amrex::MultiFab & X = this->Xs[lev];

        // GRID
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

        // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
        amrex::Real const * eBouQuad_mem_ptr = mesh.eBouQuad_device_mem.data();
        amrex::Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_device_mem.data();
#else
        amrex::Real const * eBouQuad_mem_ptr = mesh.eBouQuad_host_mem.data();
        amrex::Real const * eIntBouQuad_mem_ptr = mesh.eIntBouQuad_host_mem.data();
#endif

        // DG
        const int sp = matfactory.std_elem.p;

        // DOMAIN
        const int dom = 0;
        
        const int tagval = amrex::TagBox::SET;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------

        // TAG CELLS
        for (amrex::MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();

            amrex::Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            AMREX_D_TERM
            (
                amrex::Array4<int const> const & eBouQuad_Nq_fab_0 = mesh.eBouQuad_Nq[0].array(mfi);
                amrex::Array4<long const> const & eBouQuad_pos_fab_0 = mesh.eBouQuad_pos[0].array(mfi);,
                amrex::Array4<int const> const & eBouQuad_Nq_fab_1 = mesh.eBouQuad_Nq[1].array(mfi);
                amrex::Array4<long const> const & eBouQuad_pos_fab_1 = mesh.eBouQuad_pos[1].array(mfi);,
                amrex::Array4<int const> const & eBouQuad_Nq_fab_2 = mesh.eBouQuad_Nq[2].array(mfi);
                amrex::Array4<long const> const & eBouQuad_pos_fab_2 = mesh.eBouQuad_pos[2].array(mfi);
            )
            amrex::Array4<int const> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
            amrex::Array4<long const> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

            amrex::Array4<amrex::Real const> const & X_fab = X.array(mfi);

            amrex::Array4<char> const & tags_fab = tags.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,amrex::DG::ELM_TYPE(dom));

                if (amrex::DG::ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};

                    // LOCAL VARIABLES
                    int BF_i, BF_j, BF_k;
                    amrex::Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
                    amrex::Real x[AMREX_SPACEDIM];
                    amrex::Real SOL[DG_N_SOL];

                    // SUPPORT OF THE BASIS FUNCTIONS
                    amrex::DG::BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                    // SOLUTION
                    amrex::DG::DG_SOL_space_BFX sol(sp, BF_lo, BF_hi, X_fab, BF_i, BF_j, BF_k);

                    // CONDITION #1: POSITIVE DENSITY AND PRESSURE ----
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        int bou_Nq;
                        const amrex::Real * xptr;

                        // FACE QUADRATURE INFO
                        {
                            const int dir = b/2;
                            int fi, fj, fk;
                            long pos;
                            amrex::DG::GRID_FACE(i, j, k, b, fi, fj, fk);

                            if (dir == 0)
                            {
                                bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_NQ(dom, b));
                                pos = eBouQuad_pos_fab_0(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_POS(dom, b));
                            }
#if (AMREX_SPACEDIM > 1)
                            else if (dir == 1)
                            {
                                bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_NQ(dom, b));
                                pos = eBouQuad_pos_fab_1(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_POS(dom, b));
                            }
#endif
#if (AMREX_SPACEDIM > 2)
                            else
                            {
                                bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_NQ(dom, b));
                                pos = eBouQuad_pos_fab_2(fi,fj,fk,amrex::DG::ELM_BOU_QUAD_POS(dom, b));
                            }
#endif
                            xptr = &eBouQuad_mem_ptr[pos];
                        }

                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            // QUADRATURE POINT
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0]+cell_lo[0];,
                                x[1] = xptr[x_pos+1]+cell_lo[1];,
                                x[2] = xptr[x_pos+2]+cell_lo[2];
                            )

                            // EVAL SOLUTION
                            sol.eval(x, 0, DG_N_SOL, SOL);

                            {
                                amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
                                this->IG.eval_primary_variables(SOL, rho, AMREX_D_DECL(u1, u2, u3), p, false);

                                if ((rho < 0.0) || (p < 0.0))
                                {
                                    tags_fab(i,j,k) = tagval;
                                }
                            }
                        }
                    }

                    // ZERO-LEVEL OF THE LEVEL SET (IF ANY)
                    {
                        const int bou_Nq = eIntBouQuad_Nq_fab(i,j,k,amrex::DG::ELM_INT_BOU_QUAD_NQ(dom));
                        const long pos = eIntBouQuad_pos_fab(i,j,k,amrex::DG::ELM_INT_BOU_QUAD_POS(dom));
                        const amrex::Real * xptr = &eIntBouQuad_mem_ptr[pos];

                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            // QUADRATURE POINT
                            const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                            AMREX_D_TERM
                            (
                                x[0] = xptr[x_pos+0];,
                                x[1] = xptr[x_pos+1];,
                                x[2] = xptr[x_pos+2];
                            )

                            // EVAL SOLUTION
                            sol.eval(x, 0, DG_N_SOL, SOL);

                            {
                                amrex::Real rho, AMREX_D_DECL(u1, u2, u3), p;
                                this->IG.eval_primary_variables(SOL, rho, AMREX_D_DECL(u1, u2, u3), p, false);
                                if ((rho < 0.0) || (p < 0.0))
                                {
                                    tags_fab(i,j,k) = tagval;
                                }
                            }
                        }
                    }
                    // ------------------------------------------------
                }
            });
            amrex::Gpu::synchronize();
        }
        tags.FillBoundary(mesh.geom.periodicity());

        // CELLS EXTENDING ONTO TAGGED SMALL CELLS GET TAGGED TOO -----
        this->TagExtendedCells(mesh, tags);
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
    }
    // ================================================================
};
// ####################################################################
// ####################################################################