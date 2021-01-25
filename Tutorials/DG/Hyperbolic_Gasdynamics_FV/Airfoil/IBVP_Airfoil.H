//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_Airfoil.H
 * \brief Contains constants and data structures for the airfoil problem.
*/

#include "../IBVP_Base.H"

#define RHO_INF 1.0
#define P_INF 1.0

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct IDEAL_GAS
:
public IDEAL_GAS_BASE
{
    // DATA MEMBERS ###################################################
    int int_params[1];
    amrex::Real params[5];
    // ################################################################
    // NOTE: int_params must store:
    //       - int_params[0]:
    //       params must store:
    //       - params[0]: free-stream Mach number
    //       - params[1]: angle of attack
    //
    //       if int_params[0] == 0
    //       - params[2:3]: axes of the ellipse
    //       - params[4]: not used
    //       
    //       if int_params[0] == 4
    //       - params[2]: maximum camber in percentage of the cord
    //       - params[3]: position of maximum camber in tenths of the
    //                    cord
    //       - params[4]: maximum thickness in percentage of the cord
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
    */
    IDEAL_GAS(const amrex::Vector<int> & int_params_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(int_params_, params_)
    {
        std::fill(this->params, this->params+5, 0.0);
        std::copy(params_.begin()+1, params_.end(), this->params);
    }
    // ################################################################
    // NOTE: params_[0] must contain the specific heats ratio and is
    //       stored when the constructor IDEAL_GAS_BASE is called.
    // ################################################################

    // LEVEL SET FUNCTIONS ############################################
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_yc(const amrex::Real c, const amrex::Real p, const amrex::Real xi) const
    {
        const amrex::Real yc = (xi < p) ? (c/(p*p)*(2.0*p*xi-xi*xi)) : (c/((1.0-p)*(1.0-p))*(1.0-2.0*p+2.0*p*xi-xi*xi));
        return yc;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_dyc(const amrex::Real c, const amrex::Real p, const amrex::Real xi) const
    {
        const amrex::Real dyc = (xi < p) ? ((2.0*c*(p-xi))/(p*p)) : ((2.0*c*(p-xi))/((p-1.0)*p-1.0));
        return dyc;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_d2yc(const amrex::Real c, const amrex::Real p, const amrex::Real xi) const
    {
        const amrex::Real d2yc = (xi < p) ? (-2.0*c/(p*p)) : (-2.0*c/((p-1.0)*(p-1.0)));
        return d2yc;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_yt(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real y0 = 5.0*th*(a05*std::sqrt(xi)+a1*xi+a2*xi*xi+a3*xi*xi*xi+a4*xi*xi*xi*xi);
        const amrex::Real yc = this->NACA4_yc(c, p, xi);
        return yc+y0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_dyt(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real dy0 = 5.0*th*(a1+a05/(2.0*std::sqrt(xi))+xi*(2.0*a2+3.0*a3*xi+4.0*a4*xi*xi));
        const amrex::Real dyc = this->NACA4_dyc(c, p, xi);
        return dyc+dy0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_d2yt(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real d2y0 = 5.0*th*(2.0*a2-a05/(4.0*xi*std::sqrt(xi))+6.0*xi*(a3+2.0*a4*xi));
        const amrex::Real d2yc = this->NACA4_d2yc(c, p, xi);
        return d2yc+d2y0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_yb(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real y0 = 5.0*th*(a05*std::sqrt(xi)+a1*xi+a2*xi*xi+a3*xi*xi*xi+a4*xi*xi*xi*xi);
        const amrex::Real yc = this->NACA4_yc(c, p, xi);
        return yc-y0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_dyb(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real dy0 = 5.0*th*(a1+a05/(2.0*std::sqrt(xi))+xi*(2.0*a2+3.0*a3*xi+4.0*a4*xi*xi));
        const amrex::Real dyc = this->NACA4_dyc(c, p, xi);
        return dyc-dy0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_d2yb(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;
        const amrex::Real d2y0 = 5.0*th*(2.0*a2-a05/(4.0*xi*std::sqrt(xi))+6.0*xi*(a3+2.0*a4*xi));
        const amrex::Real d2yc = this->NACA4_d2yc(c, p, xi);
        return d2yc-d2y0;
    }
    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_solve(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real * x, const amrex::Real a0, const amrex::Real b0, const int flag) const
    {
        const amrex::Real tol = 1.0e-12;
        amrex::Real a, b, fa, fb, dfa, dfb;
        amrex::Real err, xi, fxi, dfxi;
        int it;

        a = a0;
        fa = (flag == -1) ? (this->NACA4_yb(c, p, th, a)) : (this->NACA4_yt(c, p, th, a));
        dfa = (flag == -1) ? (this->NACA4_dyb(c, p, th, a)) : (this->NACA4_dyt(c, p, th, a));
        fa = (a-x[0])+(fa-x[1])*dfa;

        if (std::abs(fa) < tol)
        {
            xi = a;
            return xi;
        }

        b = b0;
        fb = (flag == -1) ? (this->NACA4_yb(c, p, th, b)) : (this->NACA4_yt(c, p, th, b));
        dfb = (flag == -1) ? (this->NACA4_dyb(c, p, th, b)) : (this->NACA4_dyt(c, p, th, b));
        fb = (b-x[0])+(fb-x[1])*dfb;

        if (std::abs(fb) < tol)
        {
            xi = b;
            return xi;
        }

        if (fa*fb > 0.0)
        {
            amrex::Print() << "NACA4_solve - cannot apply bisection method" << std::endl;
            exit(-1);
        }

        err = 1.0;
        it = 0;
        while ((err > tol) && (it < 100))
        {
            xi = 0.5*(a+b);
            fxi = (flag == -1) ? (this->NACA4_yb(c, p, th, xi)) : (this->NACA4_yt(c, p, th, xi));
            dfxi = (flag == -1) ? (this->NACA4_dyb(c, p, th, xi)) : (this->NACA4_dyt(c, p, th, xi));
            fxi = (xi-x[0])+(fxi-x[1])*dfxi;

//amrex::Print() << "fxi: " << fxi << std::endl;

            if (fxi*fa > 0.0)
            {
                a = xi;
                fa = (flag == -1) ? (this->NACA4_yb(c, p, th, a)) : (this->NACA4_yt(c, p, th, a));
                dfa = (flag == -1) ? (this->NACA4_dyb(c, p, th, a)) : (this->NACA4_dyt(c, p, th, a));
                fa = (a-x[0])+(fa-x[1])*dfa;
            }
            else
            {
                b = xi;
            }

            err = std::min(0.5*std::abs(a-b), std::abs(fxi));
            it += 1;
        }

        if (err > tol)
        {
            amrex::Print() << "NACA4_solve - convergence not reached" << std::endl;
            exit(-1);
        }

        return xi;
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
        else if (this->eb_flag == 0)
        {
            const amrex::Real a = this->params[2];
            const amrex::Real b = this->params[3];

            PHI[0] = 1.0-((x[0]-1.5)*(x[0]-1.5)/(a*a))-((x[1]-0.0)*(x[1]-0.0)/(b*b));
        }
        else if (this->eb_flag == 4)
        {
            const amrex::Real c = this->params[2]*0.01;
            const amrex::Real p = this->params[3]*0.1;
            const amrex::Real th = this->params[4]*0.01;
            
            const amrex::Real eps = 10.0*std::numeric_limits<amrex::Real>::epsilon();
            bool inside, L0, R1, R2, R3;
            amrex::Real tmp;
            amrex::Real y1, y2, xi, eta;

            // DETERMINE WHICH REGION WE ARE IN -----------------------
            inside = ((x[0] > 0.0) && (x[0] < 1.0));
            if (inside)
            {
                y1 = this->NACA4_yb(c, p, th, x[0]);
                y2 = this->NACA4_yt(c, p, th, x[0]);
                inside = inside && (x[1] > y1) && (x[1] < y2);
            }

            L0 = (x[1] == 0.0) && (x[0] <= 0.0);

            tmp = (x[1] >= 0.0) ? this->NACA4_dyt(c, p, th, 1.0) : this->NACA4_dyb(c, p, th, 1.0);
            if (x[0] >= (1.0-tmp*x[1]))
            {
                R1 = false;
                R2 = false;
                R3 = true;
            }
            else
            {
                R3 = false;
                if (x[0] >= 0.0)
                {
                    tmp = this->NACA4_yc(c, p, x[0]);
                    R1 = (x[1] > tmp);
                    R2 = (x[1] <= tmp);
                }
                else if (x[0] < 0.0)
                {
                    R1 = (x[1] > 0.0);
                    R2 = (x[1] < 0.0);
                }
                else
                {
                    R1 = false;
                    R2 = false;
                }
            }
            // --------------------------------------------------------

            // R1 -----------------------------------------------------
            if (R1 && !R2 && !R3 && !L0)
            {
                const int n_xi = 100;
                amrex::Real ds[100], xi_a, xi_b;
                
                for (int k = 0; k < n_xi; ++k)
                {
                    tmp = eps+(1.0-eps)*(1.0*k)/(n_xi-1.0)*(1.0*k)/(n_xi-1.0);
                    y1 = this->NACA4_yt(c, p, th, tmp);

                    ds[k] = (tmp-x[0])*(tmp-x[0])+(y1-x[1])*(y1-x[1]);
                }

                tmp = ds[0];
                xi_a = 0.0;
                xi_b = eps+(1.0-eps)*1.0/(n_xi-1.0)*1.0/(n_xi-1.0);
                for (int k = 1; k < (n_xi-1); ++k)
                {
                    if (ds[k] < tmp)
                    {
                        tmp = ds[k];
                        xi_a = eps+(1.0-eps)*(1.0*(k-1))/(n_xi-1.0)*(1.0*(k-1))/(n_xi-1.0);
                        xi_b = eps+(1.0-eps)*(1.0*(k+1))/(n_xi-1.0)*(1.0*(k+1))/(n_xi-1.0);
                    }
                }

                xi = this->NACA4_solve(c, p, th, x, xi_a, xi_b, +1);
                eta = this->NACA4_yt(c, p, th, xi);
            }
            // --------------------------------------------------------
            // R2 -----------------------------------------------------
            else if (!R1 && R2 && !R3 && !L0)
            {
                const int n_xi = 100;
                amrex::Real ds[100], xi_a, xi_b;
                
                for (int k = 0; k < n_xi; ++k)
                {
                    tmp = eps+(1.0-eps)*(1.0*k)/(n_xi-1.0)*(1.0*k)/(n_xi-1.0);
                    y1 = this->NACA4_yb(c, p, th, tmp);

                    ds[k] = (tmp-x[0])*(tmp-x[0])+(y1-x[1])*(y1-x[1]);
                }

                tmp = ds[0];
                xi_a = 0.0;
                xi_b = eps+(1.0-eps)*1.0/(n_xi-1.0)*1.0/(n_xi-1.0);
                for (int k = 1; k < (n_xi-1); ++k)
                {
                    if (ds[k] < tmp)
                    {
                        tmp = ds[k];
                        xi_a = eps+(1.0-eps)*(1.0*(k-1))/(n_xi-1.0)*(1.0*(k-1))/(n_xi-1.0);
                        xi_b = eps+(1.0-eps)*(1.0*(k+1))/(n_xi-1.0)*(1.0*(k+1))/(n_xi-1.0);
                    }
                }

                xi = this->NACA4_solve(c, p, th, x, xi_a, xi_b, -1);
                eta = this->NACA4_yb(c, p, th, xi);
            }
            // --------------------------------------------------------
            // R3 -----------------------------------------------------
            else if (!R1 && !R2 && R3 && !L0)
            {
                xi = 1.0;
                eta = 0.0;
            }
            // --------------------------------------------------------
            // L0 -----------------------------------------------------
            // --------------------------------------------------------
            else
            {
amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
amrex::Print() << "R1: " << R1 << std::endl;
amrex::Print() << "R2: " << R2 << std::endl;
amrex::Print() << "R3: " << R3 << std::endl;
amrex::Print() << "L0: " << L0 << std::endl;
amrex::Print() << "inside: " << inside << std::endl;
amrex::Print() << "IDEAL_GAS.F_PHI" << std::endl;
exit(-1);
            }

            PHI[0] = -std::sqrt((x[0]-xi)*(x[0]-xi)+(x[1]-eta)*(x[1]-eta));
            if (inside) PHI[0] *= -1.0;
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
    // ################################################################

    // INITIAL CONDITIONS #############################################
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
        const amrex::Real M_inf = this->params[0];
        const amrex::Real alpha = this->params[1]*M_PI/180.0;
        const amrex::Real a_inf = std::sqrt(g*P_INF/RHO_INF);
        const amrex::Real V_inf = M_inf*a_inf;
        //const amrex::Real tau = 10.0;
        //const amrex::Real ft = 0.5*(1.0-std::cos(M_PI*t/tau));
        const amrex::Real u1 = 0.0*V_inf*std::cos(alpha);
        const amrex::Real u2 = 0.0*V_inf*std::sin(alpha);
        
        U[RHO] = RHO_INF;
        AMREX_D_TERM
        (
            U[RHOV1] = RHO_INF*u1;,
            U[RHOV2] = RHO_INF*u2;,
            U[RHOV3] = 0.0;
        )
        U[TE] = P_INF/(g-1.0)+0.5*RHO_INF*(AMREX_D_TERM(u1*u1,+u2*u2,+u3*u3));
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
               const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
               const amrex::Real * U,
               amrex::Real * bcs_U) const
    {
amrex::Print() << "IDEAL_GAS.F_BCS" << std::endl;
exit(-1);
    }
    
    /**
     * \brief Fill bcs_U memory to enforce boundary conditions on boundaries of the levelsets.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] bcs_U: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                   const amrex::Real * U,
                   amrex::Real * bcs_U) const
    {
amrex::Print() << "IDEAL_GAS.F_PHI_BCS" << std::endl;
exit(-1);
    }

    /**
     * \brief Fill bcs_P memory to enforce boundary conditions.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables at (t,x).
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS_P(const int /*dom*/,
                 const amrex::Real t, const amrex::Real * /*x*/, const amrex::Real * un,
                 const amrex::Real * P,
                 amrex::Real * bcs_P) const
    {
        const amrex::Real g = this->gamma;
        const amrex::Real M_inf = this->params[0];
        const amrex::Real alpha = this->params[1]*M_PI/180.0;
        const amrex::Real a_inf = std::sqrt(g*P_INF/RHO_INF);
        const amrex::Real V_inf = M_inf*a_inf;
        const amrex::Real tau = 20.0;
        const amrex::Real ft = 0.5*(1.0-std::cos(M_PI*t/tau));
        const amrex::Real u1_inf = ft*V_inf*std::cos(alpha);
        const amrex::Real u2_inf = ft*V_inf*std::sin(alpha);
        const amrex::Real u = P[RHOV1]*un[0]+P[RHOV2]*un[1];
        const amrex::Real a = std::sqrt(g*P[TE]/P[RHO]);

        if ((un[0] > 0.5) || (un[1] > 0.5))
        {
            bcs_P[RHO] = P[RHO];
            AMREX_D_TERM
            (
                bcs_P[RHOV1] = P[RHOV1];,
                bcs_P[RHOV2] = P[RHOV2];,
                bcs_P[RHOV3] = P[RHOV3];
            )
            bcs_P[TE] = P[TE];
        }
        else
        {
            this->bcs_prescribed_velocity_P(RHO_INF, AMREX_D_DECL(u1_inf, u2_inf, u3_inf),un, P, bcs_P);
        }

        /*
        if ((u-a) >= 0.0)
        {
            bcs_P[RHO] = P[RHO];
            AMREX_D_TERM
            (
                bcs_P[RHOV1] = P[RHOV1];,
                bcs_P[RHOV2] = P[RHOV2];,
                bcs_P[RHOV3] = P[RHOV3];
            )
            bcs_P[TE] = P[TE];
        }
        else if (((u-a) < 0.0) && (u >= 0.0))
        {
            bcs_P[RHO] = P[RHO];
            AMREX_D_TERM
            (
                bcs_P[RHOV1] = P[RHOV1];,
                bcs_P[RHOV2] = P[RHOV2];,
                bcs_P[RHOV3] = P[RHOV3];
            )
            bcs_P[TE] = P_INF;
        }
        else if (((u-a) < 0.0) && (u < 0.0) && (u+a >= 0.0))
        {
            bcs_P[RHO] = P[RHO];
            AMREX_D_TERM
            (
                bcs_P[RHOV1] = u1_inf;,
                bcs_P[RHOV2] = u2_inf;,
                bcs_P[RHOV3] = u3_inf;
            )
            bcs_P[TE] = P_INF;
        }
        else
        {
            bcs_P[RHO] = RHO_INF;
            AMREX_D_TERM
            (
                bcs_P[RHOV1] = u1_inf;,
                bcs_P[RHOV2] = u2_inf;,
                bcs_P[RHOV3] = u3_inf;
            )
            bcs_P[TE] = P_INF;
        }
        */
    }

    /**
     * \brief Fill bcs_P memory to enforce boundary conditions on boundaries of the levelsets.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] P: pointer to memory that stores the PRIMARY variables at (t,x).
     * \param[out] bcs_P: pointer to memory that will be filled with values suitably chosen to
     *                    enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI_BCS_P(const int /*dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                     const amrex::Real * P,
                     amrex::Real * bcs_P) const
    {
        this->bcs_wall(un, P, bcs_P);
    }
    // ################################################################

    // NUMERICAL FLUXES ###############################################
    // NOTE: The function void F_NF_ICS_P is found in "../IBVP_Base.H".
    // ################################################################
    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS_P(const int dom,
                    const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                    const amrex::Real * P,
                    amrex::Real * NFn) const
    {
        amrex::Real bcs_P[DG_N_SOL];
        this->F_BCS_P(dom, t, x, un, P, bcs_P);
        this->F_NF_ICS_P(dom, t, x, un, P, bcs_P, NFn);
    }

    /**
     * \brief Eval the numerical flux at a levelset's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] P: pointer to memory that stores the PRIMARY variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS_P(const int dom,
                        const amrex::Real t, const amrex::Real * x, const amrex::Real * un,
                        const amrex::Real * P,
                        amrex::Real * NFn) const
    {
        amrex::Real bcs_P[DG_N_SOL];
        this->F_PHI_BCS_P(dom, t, x, un, P, bcs_P);
        this->F_NF_ICS_P(dom, t, x, un, P, bcs_P, NFn);
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
        const amrex::Real theta = (this->params[8])*M_PI/180.0;

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
    // ################################################################
};