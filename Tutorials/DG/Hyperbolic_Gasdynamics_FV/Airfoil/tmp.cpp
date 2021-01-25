/**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_y(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi, const amrex::Real * x) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;

        const bool cond = (x[0] > 0.0) ? (x[1] > this->NACA4_yc(c, p, x[1])) : (x[1] > 0.0);

        const amrex::Real yc = this->NACA4_yc(c, p, xi);
        const amrex::Real y0 = 5.0*th*(a05*std::sqrt(xi)+a1*xi+a2*xi*xi+a3*xi*xi*xi+a4*xi*xi*xi*xi);
        const amrex::Real y = (cond) ? (y0+yc) : (-y0+yc);
        return y;
    }

    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_dy(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi, const amrex::Real * x) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;

        const bool cond = (x[0] > 0.0) ? (x[1] > this->NACA4_yc(c, p, x[1])) : (x[1] > 0.0);
        
        const amrex::Real dyc = (xi < p) ? ((2.0*c*(p-xi))/(p*p)) : ((2.0*c*(p-xi))/((p-1.0)*p-1.0));
        const amrex::Real dy0 = 5.0*th*(a1+a05/(2.0*std::sqrt(xi))+xi*(2.0*a2+3.0*a3*xi+4.0*a4*xi*xi));
        const amrex::Real dy = (cond) ? (dy0+dyc) : (-dy0+dyc);
        return dy;
    }

    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_ddy(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi, const amrex::Real * x) const
    {
        const amrex::Real a05 = 0.2969;
        const amrex::Real a1 = -0.1260;
        const amrex::Real a2 = -0.3516;
        const amrex::Real a3 = 0.2843;
        const amrex::Real a4 = -0.1036;

        const bool cond = (x[0] > 0.0) ? (x[1] > this->NACA4_yc(c, p, x[1])) : (x[1] > 0.0);
        
        const amrex::Real ddyc = (xi < p) ? (-2.0*c/(p*p)) : (-2.0*c/((p-1.0)*(p-1.0)));
        const amrex::Real ddy0 = 5.0*th*(2.0*a2-a05/(4.0*xi*std::sqrt(xi))+6.0*xi*(a3+2.0*a4*xi));
        const amrex::Real ddy = (cond) ? (ddy0+ddyc) : (-ddy0+ddyc);
        return ddy;
    }

    /**
     * \brief Auxiliary functions
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real NACA4_solve(const amrex::Real c, const amrex::Real p, const amrex::Real th, const amrex::Real xi0, const amrex::Real * x) const
    {
        const amrex::Real tol = 1.0e-8;
        const int n_max = 200;
        amrex::Real err, derr, y, dy, ddy, dxi, xi;
        int n;

        xi = xi0;
        y = this->NACA4_y(c, p, th, xi, x);
        dy = this->NACA4_dy(c, p, th, xi, x);

        err = xi-x[0]+(y-x[1])*dy;
        n = 0;
        while ((std::abs(err) > tol) && (n < n_max))
        {
            ddy = this->NACA4_ddy(c, p, th, xi, x);
            derr = 1.0+dy*dy+(y-x[1])*ddy;

            dxi = -err/derr;
            xi = std::max(tol, xi+dxi);

            y = this->NACA4_y(c, p, th, xi, x);
            dy = this->NACA4_dy(c, p, th, xi, x);

            err = xi-x[0]+(y-x[1])*dy;

            n += 1;

            amrex::Print() << "err(" << xi << "): " << err << std::endl;
        }

        if (err > tol)
        {
amrex::Print() << "Convergence not reached" << std::endl;
exit(-1);
        }

        return xi;
    }








    /*
                xi0 = 1.0e-8;
                y = this->NACA4_y(c, p, th, xi0, x);
                eq = this->NACA4_dy(c, p, th, xi0, x);
                eq = (xi0-x[0])+(y-x[1])*eq;
                sgn = (eq > 0.0) ? 1.0 : -1.0;

                while ((eq*sgn > 0.0) && (xi0 < 1.0))
                {
                    xi0 += 1.0/nxi;
                    y = this->NACA4_y(c, p, th, xi0, x);
                    eq = this->NACA4_dy(c, p, th, xi0, x);
                    eq = (xi0-x[0])+(y-x[1])*eq;
                }
                xi0 -= 0.5/nxi;

                xi = this->NACA4_solve(c, p, th, xi0, x);
                xi = std::min(xi, 1.0);

                y = this->NACA4_y(c, p, th, xi, x);

                PHI[0] = -std::sqrt((x[0]-xi)*(x[0]-xi)+(x[1]-y)*(x[1]-y));
                */


                amrex::Real d, err, a, b, xi, eta;
        int l, n;

        a = 0.0;
        b = 1.0;
        xi = 0.0;
        eta = (flag == -1) ? this->NACA4_yb(c, p, th, xi) : this->NACA4_yt(c, p, th, xi);

        l = 0;
        while (err > tol && l < n_levels)
        {
            n = 0;
            while (err > tol && n < n_nodes)
            {

                n += 1;
            }

            l += 1;
        }


        d = std::sqrt(x[0]*x[0]+x[1]*x[1]);
        err = 1.0;

        xi = 10.0*std::numeric_limits<amrex::Real>::epsilon();




const int n_xi = 10;
            amrex::Real xis[10];
            int k;




        const amrex::Real eps = 10.0*std::numeric_limits<amrex::Real>::epsilon();
            const int n_xi = 10;
            int k;

            const amrex::Real c = this->params[2]*0.01;
            const amrex::Real p = this->params[3]*0.1;
            const amrex::Real th = this->params[4]*0.01;
            amrex::Real xia, xib, da, db;
            amrex::Real tmp, xi, eta;
            bool inside;

            inside = ((x[0] > 0.0) && (x[0] < 1.0));
            inside = inside && (x[1] < (this->NACA4_yt(c, p, th, x[0]))) && (x[1] > (this->NACA4_yb(c, p, th, x[0])));

            if (inside)
            {
                tmp = this->NACA4_yc(c, p, x[0]);

                if (x[1] > tmp)
                {
                    k = 0;
                    xia = eps;
                    eta = this->NACA4_yt(c, p, th, xia);
                    da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                    xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                    eta = this->NACA4_yt(c, p, th, xib);
                    db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));
                    
                    while ((db < da) && (k < (n_xi-1)))
                    {
                        xia = eps+((1.0-eps)*k)/(n_xi-1);
                        eta = this->NACA4_yt(c, p, th, xia);
                        da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                        xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                        eta = this->NACA4_yt(c, p, th, xib);
                        db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                        k += 1;
                    }

                    xi = this->NACA4_solve(c, p, th, x, xia, xib, +1);
                    eta = this->NACA4_dyt(c, p, th, xi);
                }
                else
                {
                    k = 0;
                    xia = eps;
                    eta = this->NACA4_yb(c, p, th, xia);
                    da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                    xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                    eta = this->NACA4_yb(c, p, th, xib);
                    db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                    while ((db < da) && (k < (n_xi-1)))
                    {
                        xia = eps+((1.0-eps)*k)/(n_xi-1);
                        eta = this->NACA4_yb(c, p, th, xia);
                        da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                        xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                        eta = this->NACA4_yb(c, p, th, xib);
                        db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                        k += 1;
                    }

                    xi = this->NACA4_solve(c, p, th, x, xia, xib, -1);
                    eta = this->NACA4_dyb(c, p, th, xi);
                }

                PHI[0] = std::sqrt((x[0]-xi)*(x[0]-xi)+(x[1]-eta)*(x[1]-eta));
            }
            else
            {
                tmp = (x[1] >= 0.0) ? this->NACA4_dyt(c, p, th, 1.0) : this->NACA4_dyb(c, p, th, 1.0);

                if (x[0] >= (1.0-tmp)*x[1])
                {
                    PHI[0] = -std::sqrt((x[0]-1.0)*(x[0]-1.0)+x[1]*x[1]);
                }
                else if ((x[1] == 0.0) && (x[0] <= 0.0))
                {
                    PHI[0] = -std::min(std::abs(x[0]), std::abs(x[0]-1.0));
                }
                else
                {
                    if (x[1] > 0.0)
                    {
                        k = 0;
                        xia = eps;
                        eta = this->NACA4_yt(c, p, th, xia);
                        da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                        xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                        eta = this->NACA4_yt(c, p, th, xib);
                        db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));
                        
                        while ((db < da) && (k < (n_xi-1)))
                        {
                            xia = eps+((1.0-eps)*k)/(n_xi-1);
                            eta = this->NACA4_yt(c, p, th, xia);
                            da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                            xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                            eta = this->NACA4_yt(c, p, th, xib);
                            db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                            k += 1;
                        }

                        xi = this->NACA4_solve(c, p, th, x, xia, xib, +1);
                        eta = this->NACA4_dyt(c, p, th, xi);
                    }
                    else
                    {
                        k = 0;
                        xia = eps;
                        eta = this->NACA4_yb(c, p, th, xia);
                        da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                        xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                        eta = this->NACA4_yb(c, p, th, xib);
                        db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                        while ((db < da) && (k < (n_xi-1)))
                        {
                            xia = eps+((1.0-eps)*k)/(n_xi-1);
                            eta = this->NACA4_yb(c, p, th, xia);
                            da = std::sqrt((x[0]-xia)*(x[0]-xia)+(x[1]-eta)*(x[1]-eta));

                            xib = eps+((1.0-eps)*(k+1))/(n_xi-1);
                            eta = this->NACA4_yb(c, p, th, xib);
                            db = std::sqrt((x[0]-xib)*(x[0]-xib)+(x[1]-eta)*(x[1]-eta));

                            k += 1;
                        }

                        xi = this->NACA4_solve(c, p, th, x, xia, xib, -1);
                        eta = this->NACA4_dyb(c, p, th, xi);
                    }

                    PHI[0] = -std::sqrt((x[0]-xi)*(x[0]-xi)+(x[1]-eta)*(x[1]-eta));
                }

            }








            amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
amrex::Print() << "ds: "; amrex::DG::IO::PrintRealArray2D(1, n_xi, ds);

amrex::Print() << "xi_a: " << xi_a << std::endl;
amrex::Print() << "xi_b: " << xi_b << std::endl;
amrex::Print() << "xi: " << xi << std::endl;
amrex::Print() << "eta: " << eta << std::endl;


amrex::Print() << "IDEAL_GAS.F_PHI" << std::endl;
exit(-1);


if (k_min == 0)
                {
                    xi_a = eps;
                }
                else
                {
                    xi_a = (1.0*(k_min-1))/(n_xi-1.0);
                    xi_a = xi_a*xi_a*xi_a*xi_a;
                    xi_a = eps+(1.0-eps)*xi_a;
                }

                if (k_min == (n_xi-1))
                {
                    xi_b = 1.0;
                }
                else
                {
                    xi_b = (1.0*(k_min+1))/(n_xi-1.0);
                    xi_b = xi_b*xi_b*xi_b*xi_b;
                    xi_b = eps+(1.0-eps)*xi_b;
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

amrex::Print() << "fa: " << fa << std::endl;
amrex::Print() << "fb: " << fb << std::endl;

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