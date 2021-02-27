//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_BodyInFreeStream.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_Base.H"

#define N_PARAMS 10

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
    //       - params[0]: rhoL, i.e. density at the left of the shock;
    //       - params[1]: uL, i.e. velocity at the left of the shock;
    //       - params[2]: pL, i.e. pressure at the left of the shock;
    //       - params[3]: shock location;
    //       - params[4]: rhoR, i.e. density at the right of the shock;
    //       - params[5]: uR, i.e. velocity at the right of the shock;
    //       - params[6]: pR, i.e. pressure at the right of the shock;
    //
    //       if (this->eb_flag == -1)
    //       - params[7-9]: not used;
    //       if (this->eb_flag == 1 and AMREX_SPACEDIM == 2)
    //       - params[7]: diameter of the embedded Sod's tube;
    //       - params[8]: inclination of the tube;
    //       - params[9]: not used;
    //       if (this->eb_flag == 1 and AMREX_SPACEDIM == 3)
    //       - params[7]: diameter of the embedded Sod's tube;
    //       - params[8]: azimuth of the tube;
    //       - params[9]: elevation of the tube;
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
        else if (this->eb_flag == 1)
        {
            const amrex::Real diam = this->params[7];

#if (AMREX_SPACEDIM == 2)
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
            const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
            const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1];
            const amrex::Real y[2] = {0.5+s*xi[0], 0.5+s*xi[1]};
            const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]);
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
            const amrex::Real phi = (this->params[9])*M_PI/180.0;
            const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
            const amrex::Real s = (x[0]-0.5)*xi[0]+(x[1]-0.5)*xi[1]+(x[2]-0.5)*xi[2];
            const amrex::Real y[3] = {0.5+s*xi[0], 0.5+s*xi[1], 0.5+s*xi[2]};
            const amrex::Real d2 = (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]);
#endif
            const amrex::Real res = d2-0.25*diam*diam;
        
            PHI[0] = res;
        }
        else
        {
amrex::Abort("IDEAL_GAS.F_PHI");
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
        const amrex::Real rhoL = this->params[0];
        const amrex::Real uL = this->params[1];
        const amrex::Real pL = this->params[2];
        const amrex::Real xis = this->params[3];
        const amrex::Real rhoR = this->params[4];
        const amrex::Real uR = this->params[5];
        const amrex::Real pR = this->params[6];

        if (this->eb_flag == -1)
        {
            if (x[0] > xis)
            {
                U[RHO] = rhoR;
                AMREX_D_TERM
                (
                    U[RHOV1] = rhoR*uR;,
                    U[RHOV2] = 0.0;,
                    U[RHOV3] = 0.0;
                )
                U[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
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

            /* Smooth shock
            const amrex::Real beta = 10.0;
            const amrex::Real fs = std::atan(beta*(x[0]-xis))/M_PI+0.5;

            U[RHO] = (rhoR-rhoL)*fs+rhoL;
            AMREX_D_TERM
            (
                U[RHOV1] = (rhoR*uR-rhoL*uL)*fs+rhoL*uL;,
                U[RHOV2] = 0.0;,
                U[RHOV3] = 0.0;
            )
            U[TE] = (pR/(g-1.0)+0.5*rhoR*uR*uR-(pL/(g-1.0)+0.5*rhoL*uL*uL))*fs+(pL/(g-1.0)+0.5*rhoL*uL*uL);
            */
        }
        else if (this->eb_flag == 1)
        {
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
            const amrex::Real cth = std::cos(theta);
            const amrex::Real sth = std::sin(theta);
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = (this->params[9])*M_PI/180.0;
            const amrex::Real cph = std::cos(phi);
            const amrex::Real sph = std::sin(phi);
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth, sth, 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi_dir[AMREX_SPACEDIM] = {AMREX_D_DECL(cth*cph, sth*cph, sph)};
#endif
            const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-0.5, x[1]-0.5, x[2]-0.5)};
            const amrex::Real xi = AMREX_D_TERM(xi_dir[0]*y[0],+xi_dir[1]*y[1],+xi_dir[2]*y[2]);

            amrex::Real Uxi[DG_N_SOL];

            if (xi > xis)
            {
                Uxi[RHO] = rhoR;
                AMREX_D_TERM
                (
                    Uxi[RHOV1] = rhoR*uR;,
                    Uxi[RHOV2] = 0.0;,
                    Uxi[RHOV3] = 0.0;
                )
                Uxi[TE] = pR/(g-1.0)+0.5*rhoR*uR*uR;
            }
            else
            {
                Uxi[RHO] = rhoL;
                AMREX_D_TERM
                (
                    Uxi[RHOV1] = rhoL*uL;,
                    Uxi[RHOV2] = 0.0;,
                    Uxi[RHOV3] = 0.0;
                )
                Uxi[TE] = pL/(g-1.0)+0.5*rhoL*uL*uL;
            }

            /* Smooth shock
            const amrex::Real beta = 100.0;
            const amrex::Real fs = std::atan(beta*(xi-xis))/M_PI+0.5;

            Uxi[RHO] = (rhoR-rhoL)*fs+rhoL;
            AMREX_D_TERM
            (
                Uxi[RHOV1] = (rhoR*uR-rhoL*uL)*fs+rhoL*uL;,
                Uxi[RHOV2] = 0.0;,
                Uxi[RHOV3] = 0.0;
            )
            Uxi[TE] = (pR/(g-1.0)+0.5*rhoR*uR*uR-(pL/(g-1.0)+0.5*rhoL*uL*uL))*fs+(pL/(g-1.0)+0.5*rhoL*uL*uL);
            */

            this->eval_Un2U(xi_dir, Uxi, U);
        }
        else
        {
amrex::Abort("IDEAL_GAS.F_SOL0");
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
        this->F_NF_WALL(dom, t, x, un, U, NFn);
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
        if (this->eb_flag == -1)
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy"};
        }
        else if (this->eb_flag == 1)
        {
            fields_names = {"density",
                            AMREX_D_DECL("velocity_x", "velocity_y", "velocity_z"),
                            "pressure",
                            "mach",
                            "total_energy",
                            "internal_energy",
                             AMREX_D_DECL("velocity_xi", "velocity_eta", "velocity_zeta")};
        }
        else
        {
amrex::Abort("IDEAL_GAS.F_FIELDS_NAMES");
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
        if (this->eb_flag == -1)
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0};
        }
        else if (this->eb_flag == 1)
        {
            fields_domains = {0, AMREX_D_DECL(0, 0, 0), 0, 0, 0, 0, AMREX_D_DECL(0, 0, 0)};
        }
        else
        {
amrex::Abort("IDEAL_GAS.F_FIELDS_DOMAINS");
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

        /*
        if (std::abs(u1) > 1.0e10)
        {
            amrex::Print() << "what?!?" << std::endl;
            amrex::Print() << "U: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, U);
            exit(-1);
        }
        */

        // VELOCITY COMPONENTS ALONG THE TUBE'S CENTERLINE
        if (this->eb_flag == 1)
        {
            const amrex::Real theta = (this->params[8])*M_PI/180.0;
#if (AMREX_SPACEDIM == 3)
            const amrex::Real phi = (this->params[9])*M_PI/180.0;
#endif
#if (AMREX_SPACEDIM == 2)
            const amrex::Real xi[2] = {std::cos(theta), std::sin(theta)};
#endif
#if (AMREX_SPACEDIM == 3)
            const amrex::Real xi[3] = {std::cos(theta)*std::cos(phi), std::sin(theta)*std::cos(phi), std::sin(phi)};
#endif
            amrex::Real Un[DG_N_SOL];
            this->eval_U2Un(xi, U, Un);

            AMREX_D_TERM
            (
                F[TE+4] = Un[RHOV1]/Un[RHO];,
                F[TE+5] = Un[RHOV2]/Un[RHO];,
                F[TE+6] = Un[RHOV3]/Un[RHO];
            )
        }
    }
    // ================================================================
};
// ####################################################################
// ####################################################################



// ####################################################################
// IDEAL GAS: AMR #####################################################
// ####################################################################
/**
 * \brief Data structure used for subcell slope limiting. 
 *
*/
struct SUBCELL_SLOPE_LIMITER
{
    // DATA MEMBERS ===================================================
    amrex::IntVect rr;
    amrex::DG::ImplicitMesh mesh;
    amrex::DG::MatrixFactory matfactory;
    amrex::MultiFab X;
    amrex::iMultiFab mask;
    // ================================================================

    // UPDATE MASK FROM PARENT MESH MASK ==============================
    void UpdateMask(const amrex::iMultiFab & parent_mask)
    {
        // PARAMETERS ------------------------------------------------
        const amrex::IntVect rr_ = this->rr;

        // PARALLEL COPY?
        const bool parallel_copy_is_needed = (!amrex::isMFIterSafe(this->mask, parent_mask));
        // -----------------------------------------------------------

        // VARIABLES -------------------------------------------------
        amrex::iMultiFab safe_parent_mask;
        const amrex::iMultiFab * safe_parent_mask_ptr;
        // -----------------------------------------------------------

        // SAFE COARSE DATA ------------------------------------------
        if (parallel_copy_is_needed)
        {
            const amrex::BoxArray safe_ba = amrex::coarsen(this->mesh.cc_ba, rr_);
            const amrex::DistributionMapping & f_dm = this->mesh.dm;
            const int n_comp = parent_mask.n_comp;
            const amrex::IntVect n_grow = parent_mask.n_grow;

            safe_parent_mask.define(safe_ba, f_dm, n_comp, n_grow);
            safe_parent_mask.ParallelCopy(parent_mask, 0, 0, n_comp, n_grow, n_grow, this->mesh.geom.periodicity());

            safe_parent_mask_ptr = &safe_parent_mask;
        }
        else
        {
            safe_parent_mask_ptr = &parent_mask;
        }
        // -----------------------------------------------------------
        
        // UPDATE THE MASK -------------------------------------------
        for (amrex::MFIter mfi(this->mask); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();
            
            amrex::Array4<int const> const & parent_mask_fab = safe_parent_mask_ptr->array(mfi);
            
            amrex::Array4<int> const & mask_fab = this->mask.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int f_i, int f_j, int f_k) noexcept
            {
                int c_i, c_j, c_k;
                amrex::DG::FINE_TO_COARSE(f_i, f_j, f_k, rr_, c_i, c_j, c_k);
                
                if (amrex::DG::CELL_IS_MASKED(parent_mask_fab(c_i,c_j,c_k)))
                {
                    mask_fab(f_i,f_j,f_k) = __DG_CELL_UNMASKED__;
                }
            });
            amrex::Gpu::synchronize();
        }
        this->mask.FillBoundary(this->mesh.geom.periodicity());
        // -----------------------------------------------------------
    }

    void EvalMask(const amrex::iMultiFab & parent_mask)
    {
        this->mask = __DG_CELL_MASKED__;
        this->UpdateMask(parent_mask);
    }
    // ================================================================

    // MAKE GRID ======================================================
    void Remake(const amrex::DG::ImplicitMesh & parent_mesh,
                const amrex::DG::MatrixFactory & parent_matfactory,
                const amrex::MultiFab & parent_X,
                const amrex::iMultiFab & parent_mask)
    {
        // PARAMETERS -------------------------------------------------
        const int N_PHI = 1;
        const int N_DOM = 1;
        const int PHI_n_comp = parent_mesh.PHI.n_comp;
        const amrex::IntVect PHI_n_grow = parent_mesh.PHI.n_grow;
        const int X_n_comp = DG_N_SOL;
        const amrex::IntVect X_n_grow = parent_X.n_grow;
        const int mask_n_comp = 1;
        const amrex::IntVect mask_n_grow = amrex::IntVect::TheUnitVector();
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Geometry sub_geom;
        amrex::BoxArray sub_ba;
        amrex::DistributionMapping sub_dm;
        // ------------------------------------------------------------

        // GEOMETRY, BOXARRAY AND DISTRIBUTION MAPPING ----------------
        sub_geom = parent_mesh.geom;
        sub_geom.refine(this->rr);
        
        amrex::BoxList bl;
        bl.clear();
        for (amrex::MFIter mfi(parent_mask); mfi.isValid(); ++mfi)
        {
            const amrex::Box & bx = mfi.validbox();
            const amrex::Dim3 lo = lbound(bx);
            const amrex::Dim3 hi = ubound(bx);
            
            bool bx_has_been_added;
            bx_has_been_added = false;

            amrex::Array4<int const> const & parent_mask_fab = parent_mask.array(mfi);

            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                if (amrex::DG::CELL_IS_MASKED(parent_mask_fab(i,j,k)))
                {
                    if (!bx_has_been_added)
                    {
                        bl.push_back(bx);
                        bx_has_been_added = true;
                    }
                }
            }
        }

        //sub_ba = parent_mesh.cc_ba;
        sub_ba = amrex::BoxArray(bl);
        sub_ba.refine(this->rr);
        
        sub_dm.define(sub_ba);
        // ------------------------------------------------------------

        // INIT MESH DATA STRUCTURE -----------------------------------
        this->mesh.define(sub_geom, sub_ba, sub_dm,
                          parent_mesh.ls_std_elem.p, parent_mesh.ls_std_elem.q,
                          parent_mesh.im_std_elem.q, parent_mesh.space_q_ie,
                          parent_mesh.volume_fraction_threshold);

        this->mesh.ls_std_elem.InitInterpolation(parent_mesh.ls_std_elem, this->rr);
        // ------------------------------------------------------------

        // LEVEL SET FUNCTIONS ----------------------------------------
        this->mesh.PHI.define(sub_ba, sub_dm, PHI_n_comp, PHI_n_grow);
        this->mesh.PHI = 0.0;

        amrex::DG::InterpolateLevelsets(N_PHI, parent_mesh.geom, parent_mesh.ls_std_elem, parent_mesh.PHI, this->rr,
                                        this->mesh.geom, this->mesh.ls_std_elem, this->mesh.PHI);

        this->mesh.Dom2Phi.resize(parent_mesh.Dom2Phi.size());
        std::copy(parent_mesh.Dom2Phi.begin(), parent_mesh.Dom2Phi.end(), this->mesh.Dom2Phi.begin());
        // ------------------------------------------------------------

        // MAKE THE IMPLICIT MESH -------------------------------------
        this->mesh.Make(N_PHI, N_DOM);
        // ------------------------------------------------------------

        // MATRIX FACTORY ---------------------------------------------
        this->matfactory.define(sub_geom, 0, 1);
        this->matfactory.EvalMassMatrices(this->mesh);
        this->matfactory.EvalInterpolationOperator(this->mesh,
                                                   this->rr,
                                                   parent_mesh,
                                                   parent_matfactory);
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->mask.define(sub_ba, sub_dm, mask_n_comp, mask_n_grow);
        this->mask = __DG_CELL_MASKED__;
        this->UpdateMask(parent_mask);
        // ------------------------------------------------------------

        // SOLUTION MULTIFAB ------------------------------------------
        this->X.define(sub_ba, sub_dm, X_n_comp, X_n_grow);
        this->X = 0.0;
        // ------------------------------------------------------------
    }
    
    template <typename IBVP_CLASS>
    void MakeFromScratch(const amrex::IntVect parent_rr,
                         const amrex::DG::ImplicitMesh & parent_mesh,
                         const amrex::DG::MatrixFactory & parent_matfactory,
                         const amrex::MultiFab & parent_X,
                         const amrex::iMultiFab & parent_mask,
                         const IBVP_CLASS & IBVP)
    {
        // REFINEMENT RATIO -------------------------------------------
        this->rr = parent_rr;
        // ------------------------------------------------------------

        this->Remake(parent_mesh, parent_matfactory, parent_X, parent_mask);

        // SOLUTION INITIAL CONDITIONS --------------------------------
        amrex::DG::ProjectInitialConditions(this->mesh,
                                            this->matfactory,
                                            DG_N_SOL, this->X,
                                            IBVP,
                                            true);
        // ------------------------------------------------------------
        // The last input denotes that the ghost cells will be filled
        // too.
        // ------------------------------------------------------------
    }
    // ================================================================
};

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
    //SUBCELL_SLOPE_LIMITER limiter;
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
        // ------------------------------------------------------------

        // SET BOX ARRAY AND DISTRIBUTION MAP -------------------------
        this->SetBoxArray(lev, this->meshes[lev]->cc_ba);
        this->SetDistributionMap(lev, this->meshes[lev]->dm);
        // ------------------------------------------------------------

        // SUB CELL LIMITER -------------------------------------------
        /*
        if (this->inputs.use_sub_cell_limiter && (lev == this->max_level))
        {
            // TAG CELLS THAT ARE CONSIDERED TROUBLED
            this->IG.F_UPDATE_TROUBLED(0.0,
                                       *this->meshes[lev],
                                       *this->matfactories[lev],
                                       this->Xs[lev],
                                       this->masks[lev]);
            // MAKE TAGGING CONSISTENT
            // i) Small cells merged with tagged extended
            // ii) Extended cells merging with tagged small cells 
            this->FixTagging(*this->meshes[lev], this->masks[lev], __DG_CELL_MASKED__);

            // MAKE A SUBGRID
            if (this->masks[lev].max(0) == 1)
            {
                const amrex::IntVect rr = {AMREX_D_DECL(2*p+1, 2*p+1, 2*p+1)};
                this->limiter.MakeFromScratch(rr,
                                              *this->meshes[lev],
                                              *this->matfactories[lev],
                                              this->Xs[lev],
                                              this->masks[lev],
                                              this->IG);
            }
        }
        */
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

        // MAKE SURE TO TRACK THE SHOCK AT t = 0 ----------------------
        /**/
        {
            const int eb_flag = this->IG.eb_flag;
            bool cond;
            int dir;
            amrex::Real xs;

            if (eb_flag == -1)
            {
                cond = true;
                xs = this->IG.params[3];
            }
            else if (eb_flag == 1)
            {
                const amrex::Real theta = (this->IG.params[8])*M_PI/180.0;
                const amrex::Real cth = std::cos(theta);
                const amrex::Real sth = std::sin(theta);

#if (AMREX_SPACEDIM == 2)
                if (std::abs(std::abs(cth)-1.0) < 1.0e-3)
                {
                    cond = true;
                    dir = 0;
                }
                else if (std::abs(std::abs(sth)-1.0) < 1.0e-3)
                {
                    cond = true;
                    dir = 1;
                }
                else
                {
                    cond = false;
                    dir = -1;
                }
#endif

#if (AMREX_SPACEDIM == 3)
                const amrex::Real phi = (this->IG.params[9])*M_PI/180.0;
                const amrex::Real cph = std::cos(phi);
                const amrex::Real sph = std::sin(phi);

                if ((std::abs(std::abs(cth)-1.0) < 1.0e-3) && (std::abs(std::abs(cph)-1.0) < 1.0e-3))
                {
                    cond = true;
                    dir = 0;
                }
                else if ((std::abs(std::abs(sth)-1.0) < 1.0e-3) && (std::abs(std::abs(cph)-1.0) < 1.0e-3))
                {
                    cond = true;
                    dir = 1;
                }
                else if (std::abs(std::abs(sph)-1.0) < 1.0e-3)
                {
                    cond = true;
                    dir = 2;
                }
                else
                {
                    cond = false;
                    dir = -1;
                }
#endif
                xs = this->IG.params[3]+0.5;
            }
            else
            {
                cond = false;
            }

            if ((time < 1.0e-12) && cond)
            {
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->meshes[lev]->geom.CellSizeArray();
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->meshes[lev]->geom.ProbLoArray();
                
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

                            if (std::abs(xc[dir]-xs) < dx[dir])
                            {
                                tags_fab(i,j,k) = tagged;
                            }
                        }
                    });
                    amrex::Gpu::synchronize();
                }
                tags.FillBoundary(this->meshes[lev]->geom.periodicity());
            }
        }
        /**/
        // ------------------------------------------------------------

        // TAG AN ADDITIONAL LAYER OF CELLS ---------------------------
        //this->GrowTagging(*this->meshes[lev], tags, tagged, 1);
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
amrex::Abort("AMR_IDEAL_GAS.MakeNewLevelFromCoarse - lev == 0");
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
amrex::Abort("AMR_IDEAL_GAS.RemakeLevel - lev == 0");
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