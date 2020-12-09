//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SinglePhase.H
 * \brief Contains constants and data structures for single-phase elastic wave propagation problems.
*/

#include "../IBVP_Base.H"

#define DG_N_SOL N_VS
#define DG_MAX_N_PARAMS 5

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct ELASTIC_SOLID
:
public ELASTIC_SOLID_BASE<1>
{
    // DATA MEMBERS ###################################################
    // VARIABLES
    int ls_flag;
    amrex::Array<amrex::Real, DG_MAX_N_PARAMS> ls_params;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    ELASTIC_SOLID(const amrex::Vector<int> & int_params,
                  const amrex::Vector<amrex::Real> & params)
    :
    ELASTIC_SOLID_BASE<1>({int_params}, {params})
    {
        const int mat_type = int_params[0];
        this->ls_flag = int_params[1];

        // CHECK THAT THE LEVEL SET FUNCTION FLAG IS CORRECT
        if (this->ls_flag == -1)
        {
            // Do nothing
        }
        else if (this->ls_flag == 0)
        {
            if (mat_type == MAT_TYPE_ISOTROPIC)
            {
                this->ls_params[0] = params[3];
                AMREX_D_TERM
                (
                    this->ls_params[1] = params[4];,
                    this->ls_params[2] = params[5];,
                    this->ls_params[3] = params[6];
                )
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: IBVP_SinglePhase.H\n";
                msg += "| Unexpected material type flag: "+std::to_string(mat_type)+".\n";
                amrex::Abort(msg);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: IBVP_SinglePhase.H\n";
            msg += "| Unexpected flag for the geometry description.\n";
            msg += "| this->ls_flag = "+std::to_string(this->ls_flag)+".\n";
            amrex::Abort(msg);
        }
    }
    // ################################################################

    // LEVEL SET FUNCTIONS ############################################
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
        if (this->ls_flag == -1)
        {
            PHI[0] = -1.0;
        }
        else if (this->ls_flag == 0)
        {
            const amrex::Real y[AMREX_SPACEDIM] = {AMREX_D_DECL(x[0]-std::floor(x[0]), x[1]-std::floor(x[1]), x[2]-std::floor(x[2]))};
            const amrex::Real r = this->ls_params[0];
            const amrex::Real xc[AMREX_SPACEDIM] = {AMREX_D_DECL(this->ls_params[1]-y[0], this->ls_params[2]-y[1], this->ls_params[3]-y[2])};

            PHI[0] = r*r-(AMREX_D_TERM(xc[0]*xc[0], +xc[1]*xc[1], +xc[2]*xc[2]));
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
    // ################################################################

    // EXACT SOLUTION/INITIAL CONDITIONS ##############################
    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real t, const amrex::Real * x, amrex::Real * U) const
    {
        // PARAMETERS
        const int dom = 0;
        const amrex::Real density = this->rho[dom];
        const amrex::Real * c = &this->cSE[dom][0];

#if (AMREX_SPACEDIM == 2)
        const amrex::Real theta = M_PI/6.0;
        const amrex::Real dir[AMREX_SPACEDIM] = {AMREX_D_DECL(std::cos(theta), std::sin(theta), 0.0)};
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real phi = M_PI/6.0;
        const amrex::Real theta = M_PI/6.0;
        const amrex::Real dir[AMREX_SPACEDIM] = {AMREX_D_DECL(std::cos(phi)*std::sin(theta), std::sin(phi)*std::sin(theta), std::cos(theta))};
#endif

        // VARIABLES
        int k;
        amrex::Real w[AMREX_SPACEDIM], v[AMREX_SPACEDIM*AMREX_SPACEDIM];
        amrex::Real z[AMREX_SPACEDIM];
        amrex::Real zk, vk[AMREX_SPACEDIM];
        amrex::Real f;

        // EIGENVALUES AND EIGENVECTORS OF THE CHRISTOFFEL SYMBOL =====
        this->eig_InTcIn(0, dir, w, v);

        // Acoustic impedance
        AMREX_D_TERM
        (
            z[0] = density*std::sqrt(w[0]);,
            z[1] = density*std::sqrt(w[1]);,
            z[2] = density*std::sqrt(w[2]);
        )
        // ============================================================

        // EVAL COEFFICIENTS ==========================================
        k = 0;
        zk = z[k];
        for (int ik = 1; ik < AMREX_SPACEDIM; ++ik)
        {
            if (z[ik] > zk)
            {
                k = ik;
                zk = z[ik];
            }
        }

        AMREX_D_TERM
        (
            vk[0] = 0.5*std::sqrt(2.0)*v[0+k*AMREX_SPACEDIM];,
            vk[1] = 0.5*std::sqrt(2.0)*v[1+k*AMREX_SPACEDIM];,
            vk[2] = 0.5*std::sqrt(2.0)*v[2+k*AMREX_SPACEDIM];
        )

#if (AMREX_SPACEDIM == 2)
        U[V1] = vk[0];
        U[V2] = vk[1];

        U[S11] = (c[C11]*dir[0]+c[C61]*dir[1])*vk[0]+(c[C61]*dir[0]+c[C21]*dir[1])*vk[1];
        U[S22] = (c[C21]*dir[0]+c[C62]*dir[1])*vk[0]+(c[C62]*dir[0]+c[C22]*dir[1])*vk[1];
        U[S12] = (c[C61]*dir[0]+c[C66]*dir[1])*vk[0]+(c[C66]*dir[0]+c[C62]*dir[1])*vk[1];

        U[S11] *= -density/zk;
        U[S22] *= -density/zk;
        U[S12] *= -density/zk;
#endif
#if (AMREX_SPACEDIM == 3)
        U[V1] = vk[0];
        U[V2] = vk[1];
        U[V3] = vk[2];

        U[S11] = (c[C11]*dir[0]+c[C61]*dir[1]+c[C51]*dir[2])*vk[0]+(c[C61]*dir[0]+c[C21]*dir[1]+c[C41]*dir[2])*vk[1]+(c[C51]*dir[0]+c[C41]*dir[1]+c[C31]*dir[2])*vk[2];
        U[S22] = (c[C21]*dir[0]+c[C62]*dir[1]+c[C52]*dir[2])*vk[0]+(c[C62]*dir[0]+c[C22]*dir[1]+c[C42]*dir[2])*vk[1]+(c[C52]*dir[0]+c[C42]*dir[1]+c[C32]*dir[2])*vk[2];
        U[S33] = (c[C31]*dir[0]+c[C63]*dir[1]+c[C53]*dir[2])*vk[0]+(c[C63]*dir[0]+c[C32]*dir[1]+c[C43]*dir[2])*vk[1]+(c[C53]*dir[0]+c[C43]*dir[1]+c[C33]*dir[2])*vk[2];
        U[S23] = (c[C41]*dir[0]+c[C64]*dir[1]+c[C54]*dir[2])*vk[0]+(c[C64]*dir[0]+c[C42]*dir[1]+c[C44]*dir[2])*vk[1]+(c[C54]*dir[0]+c[C44]*dir[1]+c[C43]*dir[2])*vk[2];
        U[S13] = (c[C51]*dir[0]+c[C65]*dir[1]+c[C55]*dir[2])*vk[0]+(c[C65]*dir[0]+c[C52]*dir[1]+c[C54]*dir[2])*vk[1]+(c[C55]*dir[0]+c[C54]*dir[1]+c[C53]*dir[2])*vk[2];
        U[S12] = (c[C61]*dir[0]+c[C66]*dir[1]+c[C65]*dir[2])*vk[0]+(c[C66]*dir[0]+c[C62]*dir[1]+c[C64]*dir[2])*vk[1]+(c[C65]*dir[0]+c[C64]*dir[1]+c[C63]*dir[2])*vk[2];

        U[S11] *= -density/zk;
        U[S22] *= -density/zk;
        U[S33] *= -density/zk;
        U[S23] *= -density/zk;
        U[S13] *= -density/zk;
        U[S12] *= -density/zk;
#endif
        // ============================================================

/*
amrex::Print() << "w: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, w);
amrex::Print() << "z: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, z);
amrex::Print() << "v: " << std::endl; amrex::DG::IO::PrintRealArray2D(AMREX_SPACEDIM, AMREX_SPACEDIM, v);
amrex::Print() << "k: " << k << std::endl;
amrex::Print() << "U0: " << std::endl; amrex::DG::IO::PrintRealArray2D(DG_N_SOL, 1, U);
exit(-1);
*/
        // MULTIPLY BY FUNCTION OF SPACE AND TIME =====================
        zk = zk/density;
        f = (AMREX_D_TERM(x[0]*dir[0], +x[1]*dir[1], +x[2]*dir[2]))-zk*t;
        f = std::sin(2.0*M_PI*f);

        AMREX_D_TERM
        (
            U[V1] *= f;,
            U[V2] *= f;,
            U[V3] *= f;
        )

#if (AMREX_SPACEDIM == 2)
        U[S11] *= f;
        U[S22] *= f;
        U[S12] *= f;
#endif
#if (AMREX_SPACEDIM == 3)
        U[S11] *= f;
        U[S22] *= f;
        U[S33] *= f;
        U[S23] *= f;
        U[S13] *= f;
        U[S12] *= f;
#endif
        // ============================================================
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
        this->F_EXACT(0.0, x, U);
    }
    // ################################################################

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_U memory to enforce boundary conditions.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const int /*dom*/,
               const amrex::Real t, const amrex::Real * x, const amrex::Real * /*un*/,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
        this->F_EXACT(t, x, bcs_U);

        // PRESCRIBED VELOCITY
        AMREX_D_TERM
        (
            bcs_U[V1] = 2.0*bcs_U[V1]-U[V1];,
            bcs_U[V2] = 2.0*bcs_U[V2]-U[V2];,
            bcs_U[V3] = 2.0*bcs_U[V3]-U[V3];
        )
#if (AMREX_SPACEDIM == 2)
        bcs_U[S11] = U[S11];
        bcs_U[S22] = U[S22];
        bcs_U[S12] = U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        bcs_U[S11] = U[S11];
        bcs_U[S22] = U[S22];
        bcs_U[S33] = U[S33];
        bcs_U[S23] = U[S23];
        bcs_U[S13] = U[S13];
        bcs_U[S12] = U[S12];
#endif
    }

    /**
     * \brief Fill bcs_U memory to enforce boundary conditions for the level set function.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS(const int /*dom*/,
                   const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                   const amrex::Real * U,
                   amrex::Real * bcs_U) const
    {
        amrex::Real Un[DG_N_SOL], bcs_Un[DG_N_SOL];

        this->F_EXACT(t, x, bcs_U);

        this->eval_U2Un(un, U, Un);
        this->eval_U2Un(un, bcs_U, bcs_Un);

        // PRESCRIBED TRACTIONS
        AMREX_D_TERM
        (
            bcs_Un[V1] = Un[V1];,
            bcs_Un[V2] = Un[V2];,
            bcs_Un[V3] = Un[V3];
        )
#if (AMREX_SPACEDIM == 2)
        bcs_Un[S11] = 2.0*bcs_Un[S11]-Un[S11];
        bcs_Un[S22] = Un[S22];
        bcs_Un[S12] = 2.0*bcs_Un[S12]-Un[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        bcs_Un[S11] = 2.0*bcs_Un[S11]-Un[S11];
        bcs_Un[S22] = Un[S22];
        bcs_Un[S33] = Un[S33];
        bcs_Un[S23] = Un[S23];
        bcs_Un[S13] = 2.0*bcs_Un[S13]-Un[S13];
        bcs_Un[S12] = 2.0*bcs_Un[S12]-Un[S12];
#endif
        this->eval_Un2U(un, bcs_Un, bcs_U);

/*
amrex::Print() << "un: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, un);
amrex::Print() << "bcs_Un: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, bcs_Un);
this->eval_U2Un(un, bcs_Un, Un);
this->eval_Un2U(un, Un, bcs_Un);
amrex::Print() << "bcs_Un: "; amrex::DG::IO::PrintRealArray2D(1, DG_N_SOL, bcs_Un);
exit(-1);
*/
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    // NOTE: The function void F_NF_ICS is found in "../IBVP_Base.H".
    // ################################################################
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
        amrex::Real bcs_U[DG_N_SOL];
        this->F_BCS(dom, t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }

    /**
     * \brief Eval the numerical flux at the boundary of the level set function.
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
        amrex::Real bcs_U[DG_N_SOL];
        this->F_PHI_BCS(dom, t, x, un, U, bcs_U);
        this->F_NF_ICS(dom, t, x, un, U, bcs_U, NFn);
    }
    // ################################################################

    // ERROR ##########################################################
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
    void F_ERROR(const int dom,
                 const amrex::Real t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x, amrex::Real & norm_x) const
    {
        // VARIABLES
        amrex::Real Ue[DG_N_SOL];
        
        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

        norm_x = this->eval_total_energy(dom, Ue);
        //norm_x = Ue[V1]*Ue[V1];
        //norm_x = 1.0;

        AMREX_D_TERM
        (
            Ue[V1] -= U[V1];,
            Ue[V2] -= U[V2];,
            Ue[V3] -= U[V3];
        )
#if (AMREX_SPACEDIM == 2)
        Ue[S11] -= U[S11];
        Ue[S22] -= U[S22];
        Ue[S12] -= U[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        Ue[S11] -= U[S11];
        Ue[S22] -= U[S22];
        Ue[S33] -= U[S33];
        Ue[S23] -= U[S23];
        Ue[S13] -= U[S13];
        Ue[S12] -= U[S12];
#endif
        err_x = this->eval_total_energy(dom, Ue);
        //err_x = Ue[V1]*Ue[V1];
        //err_x = x[0];
    }
    // ################################################################

    // OUTPUT FIELDS ##################################################
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void F_FIELDS_NAMES(amrex::Vector<std::string> & fields_names) const
    {
#if (AMREX_SPACEDIM == 1)
        fields_names = {"velocity_x",
                        "stress_xx"};
#endif
#if (AMREX_SPACEDIM == 2)
        fields_names = {"velocity_x", "velocity_y",
                        "stress_xx", "stress_yy", "stress_xy",
                        "err_velocity_x", "err_velocity_y",
                        "err_stress_xx", "err_stress_yy", "err_stress_xy"};
#endif
#if (AMREX_SPACEDIM == 3)
        fields_names = {"velocity_x", "velocity_y", "velocity_z",
                        "stress_xx", "stress_yy", "stress_zz", "stress_yz", "stress_xz", "stress_xy",
                        "err_velocity_x", "err_velocity_y", "err_velocity_z",
                        "err_stress_xx", "err_stress_yy", "err_stress_zz", "err_stress_yz", "err_stress_xz", "err_stress_xy"};
#endif
    }

    /**
     * \brief Set the domains where the output fields are defined.
     * 
     * \param[out] fields_domains: vector of ints containing the domains where the output fields are
     *                             defined.
    */
    void F_FIELDS_DOMAINS(amrex::Vector<int> & fields_domains) const
    {
#if (AMREX_SPACEDIM == 1)
        fields_domains = {0, 0};
#endif
#if (AMREX_SPACEDIM == 2)
        fields_domains = {0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0};
#endif
#if (AMREX_SPACEDIM == 3)
        fields_domains = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
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
    void F_FIELDS(const amrex::Real t, const amrex::Real * x,
                  const amrex::Real * U,
                  amrex::Real * F) const
    {
        // VARIABLES
        amrex::Real Ue[DG_N_SOL];
        
        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

#if (AMREX_SPACEDIM == 1)
        F[0] = U[V1];
        F[1] = U[S11];
#endif
#if (AMREX_SPACEDIM == 2)
        F[0] = U[ V1];
        F[1] = U[ V2];
        F[2] = U[S11];
        F[3] = U[S22];
        F[4] = U[S12];

        F[0+N_VS] = U[ V1]-Ue[ V1];
        F[1+N_VS] = U[ V2]-Ue[ V2];
        F[2+N_VS] = U[S11]-Ue[S11];
        F[3+N_VS] = U[S22]-Ue[S22];
        F[4+N_VS] = U[S12]-Ue[S12];
#endif
#if (AMREX_SPACEDIM == 3)
        F[0] = U[ V1];
        F[1] = U[ V2];
        F[2] = U[ V3];
        F[3] = U[S11];
        F[4] = U[S22];
        F[5] = U[S33];
        F[6] = U[S23];
        F[7] = U[S13];
        F[8] = U[S12];

        F[0+N_VS] = U[ V1]-Ue[ V1];
        F[1+N_VS] = U[ V2]-Ue[ V2];
        F[2+N_VS] = U[ V3]-Ue[ V3];
        F[3+N_VS] = U[S11]-Ue[S11];
        F[4+N_VS] = U[S22]-Ue[S22];
        F[5+N_VS] = U[S33]-Ue[S33];
        F[6+N_VS] = U[S23]-Ue[S23];
        F[7+N_VS] = U[S13]-Ue[S13];
        F[8+N_VS] = U[S12]-Ue[S12];
#endif
    }
    // ################################################################
};