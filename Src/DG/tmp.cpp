        // SAMPLE THE CURVE AND FIND A STARTING POINT -----------------
        {
            Real u0;

            u = 0.0;
            while (u < 1.0)
            {
                this->eval_der(u, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (std::abs(dp[1]) < tol)
                {
                    return dp[0];
                }
                
                if (dp[0] < res)
                {
                    res = dp[0];
                    u0 = u;
                }
                
                const Real r = 1.0/curve_curvature(dkx);
                const Real * dt = &dkx[1*AMREX_SPACEDIM];
                const Real dt_mag = std::sqrt(AMREX_D_TERM(dt[0]*dt[0],+dt[1]*dt[1],+dt[2]*dt[2]));
                const Real du = amrex::max(0.01, amrex::min(0.1, r*(M_PI/18.0)/dt_mag));

                u = amrex::min(1.0, u+du);
            }

            u = u0;
        }
        // ------------------------------------------------------------

        // FIND AN INTERVAL FOR THE NEWTON-RAPHSON SEARCH -------------
        {
            
        }
        // ------------------------------------------------------------

        // START NEWTON-RAPHSON SEARCH --------------------------------
        {
            const int it_max = 20;
            
            int it;
            Real err;

            this->eval_der(u, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            
            err = std::abs(dp[1]);
            if (err < tol)
            {
                return dp[0];
            }

            while ((it < it_max) && (err > tol))
            {

            }
        }
        // ------------------------------------------------------------
        
        
        
        
        
        
        
        // VARIABLES --------------------------------------------------
        Real ulo, uhi;
        Real res;
        Real dp[3], dkx[AMREX_SPACEDIM*3];
        // ------------------------------------------------------------

        // FIND A GOOD INTERVAL BEFORE STARTING THE NR SEARCH ---------
        {
            const int it_max = 4;
            const int nu = 10;

            bool good_interval;
            int it;
            Real uLO, uHI, u, tmp2[2];

            uLO = 0.0;
            uHI = 1.0;

            good_interval = false;
            it = 0;
            while ((it < it_max) && !good_interval)
            {
                Real du = (uHI-uLO)/(nu-1);

                {
                    int iu = 0;

                    u = uLO+(iu+0.5)*du;
                    this->eval_der(u, 2, dkx);
                    curve_sqrdist_up_to_second_der(dkx, x, dp);
                    if (std::abs(dp[1]) < tol)
                    {
                        return dp[0];
                    }
                    res = dp[0];
                    ulo = uLO+iu*du;
                    uhi = uLO+(iu+1)*du;
                }
                for (int iu = 1; iu < (nu-1); ++iu)
                {
                    u = uLO+(iu+0.5)*du;

                    this->eval_der(u, 2, dkx);
                    curve_sqrdist_up_to_second_der(dkx, x, dp);
                    if (std::abs(dp[1]) < tol)
                    {
                        return dp[0];
                    }
                    if (dp[0] < res)
                    {
                        res = dp[0];
                        ulo = uLO+iu*du;
                        uhi = uLO+(iu+1)*du;
                    }
                }

                this->eval_der(ulo, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (std::abs(dp[1]) < tol)
                {
                    return dp[0];
                }
                tmp2[0] = dp[1];
                this->eval_der(uhi, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (std::abs(dp[1]) < tol)
                {
                    return dp[0];
                }
                tmp2[1] = dp[1];

                if (tmp2[0]*tmp2[1] < 0.0)
                {
                    good_interval = true;
                }
                else
                {
                    uLO = ulo;
                    uHI = uhi;
                    it += 1;
                }
            }
        }
        // ------------------------------------------------------------

        // START NEWTON-RAPHSON SEARCH --------------------------------
        {
            const int it_max = 20;
            
            Real dplo, dphi, u, du, err;
            int it;
            
            this->eval_der(ulo, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            dplo = dp[1];
            this->eval_der(uhi, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            dphi = dp[1];

            if ((dplo > 0.0 && dphi > 0.0) || (dplo < 0.0 && dphi < 0.0))
            {
#ifdef AMREX_USE_GPU
                Abort("Cannot compute the distance.");
#else
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_NURBS.cpp - rational_Bezier::curve::distance_from\n";
                msg += "| Cannot compute the distance.\n";
                msg += "| dplo: "+std::to_string(dplo)+"\n";
                msg += "| dphi: "+std::to_string(dphi)+"\n";
                //Warning(msg);

Print() << "P2: " << std::endl;
IO::PrintRealArray2D(AMREX_SPACEDIM, N+1, this->P);
IO::PrintRealArray2D(1, N+1, this->W);
Print() << "x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
Print() << "int u : " << ulo << " " << uhi << std::endl;
Print() << "res: " << res << std::endl;
exit(-1);

#endif
            }

            if (dphi < 0.0)
            {
                const Real tmp = uhi;
                uhi = ulo;
                ulo = tmp;
            }

            u = 0.5*(ulo+uhi);
            du = uhi-ulo;
            this->eval_der(u, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            if (std::abs(dp[1]) < tol)
            {
                return dp[0];
            }

            it = 0;
            err = std::abs(dp[1]);
            while ((it < it_max) && (err > tol))
            {
                if (((dp[2]*(u-ulo)-dp[1])*(dp[2]*(u-uhi)-dp[1]) < 0.0) && (std::abs(dp[1]) < 0.5*std::abs(du*dp[2])))
                {
                    du = -dp[1]/dp[2];
                    u += du; 
                }
                else
                {
                    du = 0.5*(uhi-ulo);
                    u = ulo+du;
                }

                this->eval_der(u, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (std::abs(dp[1]) < tol)
                {
                    return dp[0];
                }

                err = std::abs(dp[1]);

                if (dp[1] < 0.0)
                {
                    ulo = u;
                }
                else
                {
                    uhi = u;
                }

                it += 1;
            }

            this->eval_der(u, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            res = dp[0];
        }
        // ------------------------------------------------------------







AMREX_GPU_HOST_DEVICE
    Real distance_from_old(const Real * x, const Real tol = 1.0e-8) const
    {
        // CHECK WHETHER ONE OF THE ENDPOINTS IS THE CLOSEST ----------
        {
            bool all_points_are_behind;
            const Real d2 = end_points_distance_from(N, this->P, x, all_points_are_behind);

            if (all_points_are_behind)
            {
                return d2;
            }
        }
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        int a, b, nb;
        rational_Bezier::curve<D> rbz;
        rational_Bezier::curve<D-2> aux_rbz;
        Real alphas[D], alpha;

        Real res;
        // ------------------------------------------------------------

        // INITIALIZE THE DISTANCE ------------------------------------
        res = sqrdist(this->P, x);
        // ------------------------------------------------------------
        

        // COMPUTE THE DISTANCE FROM THE POINTS -----------------------
        a = D;
        b = D+1;
        nb = 0;

        for (int i = 0; i <= D; ++i)
        {
            AMREX_D_TERM
            (
                rbz.P[0+i*AMREX_SPACEDIM] = this->P[0+i*AMREX_SPACEDIM];,
                rbz.P[1+i*AMREX_SPACEDIM] = this->P[1+i*AMREX_SPACEDIM];,
                rbz.P[2+i*AMREX_SPACEDIM] = this->P[2+i*AMREX_SPACEDIM];
            )
            rbz.W[i] = this->W[i];
        }

        while (b < M)
        {
            int ii = b;
            while ((b < M) && (this->U[b+1] == this->U[b]))
            {
                ++b;
            }
            int mult = b-ii+1;

            if (mult < D)
            {
                Real numer = this->U[b]-this->U[a];

                for (int j = D; j > mult; --j)
                {
Print() << "j-mult-1: " << j-mult-1 << std::endl;

                    alphas[j-mult-1] = numer/(this->U[a+j]-this->U[a]);
                }

                int r = D-mult;
                for (int j = 1; j <= r; ++j)
                {
                    int save = r-j;
                    int s = mult+j;
                    for (int k = D; k >= s; --k)
                    {
                        alpha = alphas[k-s];
                        AMREX_D_TERM
                        (
                            rbz.P[0+k*AMREX_SPACEDIM] = alpha*rbz.P[0+k*AMREX_SPACEDIM]+(1.0-alpha)*rbz.P[0+(k-1)*AMREX_SPACEDIM];,
                            rbz.P[1+k*AMREX_SPACEDIM] = alpha*rbz.P[1+k*AMREX_SPACEDIM]+(1.0-alpha)*rbz.P[1+(k-1)*AMREX_SPACEDIM];,
                            rbz.P[2+k*AMREX_SPACEDIM] = alpha*rbz.P[2+k*AMREX_SPACEDIM]+(1.0-alpha)*rbz.P[2+(k-1)*AMREX_SPACEDIM];
                        )
                        rbz.W[k] = alpha*rbz.W[k]+(1.0-alpha)*rbz.W[k-1];
                    }

                    if (b < M)
                    {
                        AMREX_D_TERM
                        (
                            aux_rbz.P[0+save*AMREX_SPACEDIM] = this->P[0+D*AMREX_SPACEDIM];,
                            aux_rbz.P[1+save*AMREX_SPACEDIM] = this->P[1+D*AMREX_SPACEDIM];,
                            aux_rbz.P[2+save*AMREX_SPACEDIM] = this->P[2+D*AMREX_SPACEDIM];
                        )
                        aux_rbz.W[save] = this->W[D];
                    }
                }
            }

            /*
            Print() << "Rational Bezier curve: " << std::endl;
            Print() << "rbz.P: " << std::endl;
            IO::PrintRealArray2D(AMREX_SPACEDIM, D+1, rbz.P);
            IO::PrintRealArray2D(1, D+1, rbz.W);
            Print() << "distance: " << rbz.distance_from(x, tol) << std::endl;
            */

            // Compute distance
            res = amrex::min(res, rbz.distance_from(x, tol));

for (int i = 0; i < D-1; ++i)
{
    AMREX_D_TERM
    (
        rbz.P[0+i*AMREX_SPACEDIM] = aux_rbz.P[0+i*AMREX_SPACEDIM];,
        rbz.P[1+i*AMREX_SPACEDIM] = aux_rbz.P[1+i*AMREX_SPACEDIM];,
        rbz.P[2+i*AMREX_SPACEDIM] = aux_rbz.P[2+i*AMREX_SPACEDIM];
    )
    rbz.W[i] = aux_rbz.W[i];
}

            if (b < M)
            {
                for (int i = D-mult; i <= D; ++i)
                {
                    AMREX_D_TERM
                    (
                        rbz.P[0+i*AMREX_SPACEDIM] = this->P[0+(i+b-D)*AMREX_SPACEDIM];,
                        rbz.P[1+i*AMREX_SPACEDIM] = this->P[1+(i+b-D)*AMREX_SPACEDIM];,
                        rbz.P[2+i*AMREX_SPACEDIM] = this->P[2+(i+b-D)*AMREX_SPACEDIM];
                    )
                    rbz.W[i] = this->W[i+b-D];
                }
                a = b;
                b += 1;
            }
        }
        // ------------------------------------------------------------

        return res;
    }






    const amrex::Real * LE = this->params;
            const amrex::Real c = this->params[2];
            const amrex::Real xs[AMREX_SPACEDIM] = {(x[0]-LE[0])/c, (x[1]-LE[1])/c};

            const amrex::Real xTE = 1.0;
            const amrex::Real thTE = this->NACA4_thickness(0.12, xTE);
            const amrex::Real phTE = -std::atan(this->NACA4_thickness_derivative(0.12, xTE));
            const amrex::Real xLE = 0.0025;
            
            amrex::DG::nurbs::curve<5, 2, 2> curve_TE;
            amrex::DG::nurbs::curve<5, 2, 2> curve_LE;
            amrex::DG::nurbs::curve<19, 15, 3> curve_top;
            amrex::DG::nurbs::curve<19, 15, 3> curve_bottom;

            // TRAILING EDGE
            {
                const amrex::Real PTE[AMREX_SPACEDIM] = {xTE+thTE/std::tan(phTE), 0.0};
                const amrex::Real U[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
                const amrex::Real P[6] = {xTE, -thTE,
                                          PTE[0], PTE[1],
                                          xTE, +thTE};
                const amrex::Real W[3] = {1.0, std::sin(phTE), 1.0};
                
                curve_TE.set(U, P, W);
                
                /*
                const int nu = 10;
                for (int iu = 0; iu < nu; ++iu)
                {
                    const amrex::Real u = (1.0*iu)/(nu-1);
                    amrex::Real xx[AMREX_SPACEDIM];
                    curve_TE.eval(u, xx);
                    amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xx);
                }
                */
            }

            // LEADING EDGE
            {
                const amrex::Real thLE = this->NACA4_thickness(0.12, xLE);
                const amrex::Real phLE = std::atan(this->NACA4_thickness_derivative(0.12, xLE));
                const amrex::Real PLE[AMREX_SPACEDIM] = {-(thLE/std::tan(phLE)-xLE), 0.0};
                const amrex::Real U[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
                const amrex::Real P[6] = {xLE, thLE,
                                          PLE[0], PLE[1],
                                          xLE, -thLE};
                const amrex::Real W[3] = {1.0, std::sin(phLE), 1.0};
                
                curve_LE.set(U, P, W);

                /*
                const int nu = 10;
                for (int iu = 0; iu < nu; ++iu)
                {
                    const amrex::Real u = (1.0*iu)/(nu-1);
                    amrex::Real xx[AMREX_SPACEDIM];
                    curve_LE.eval(u, xx);
                    amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xx);
                }
                */
            }

            // TOP
            {
                const amrex::Real U[20] = {0.0, 0.0, 0.0, 0.0,
                                           0.025754532469464426, 0.049694506145328256, 0.08163785307378141, 0.12183765509467155, 
                                           0.17047357246983033, 0.22767466267090952, 0.29354456136020435, 0.36817025905048073,
                                           0.4516232599636818, 0.5439594844406513, 0.6452223428310663, 0.755452426041811,
                                           1.0, 1.0, 1.0, 1.0};
                const amrex::Real P[32] = {0.002500000000, 0.008716684163,
                                           0.007163369564, 0.016197037987,
                                           0.021777069026, 0.025465120029,
                                           0.047304023825, 0.035313893286,
                                           0.078818724642, 0.043310317965,
                                           0.118886124946, 0.050191425226,
                                           0.167821636357, 0.055539554113,
                                           0.225550738143, 0.059033777696,
                                           0.292132840525, 0.060396757760,
                                           0.367555632362, 0.059437571880,
                                           0.451828366715, 0.056047942169,
                                           0.544953180430, 0.050195056431,
                                           0.646939994627, 0.041883526517,
                                           0.799800790932, 0.026981867341,
                                           0.918433713697, 0.012656181473,
                                           1.000000000000, 0.001260000000};
                const amrex::Real W[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
                curve_top.set(U, P, W);

                /*
                const int nu = 10;
                for (int iu = 0; iu < nu; ++iu)
                {
                    const amrex::Real u = (1.0*iu)/(nu-1);
                    amrex::Real xx[AMREX_SPACEDIM];
                    curve_top.eval(u, xx);
                    amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xx);
                }
                */
            }
            // BOTTOM
            {
                const amrex::Real U[20] = {0.0, 0.0, 0.0, 0.0,
                                           0.025754532469464426, 0.049694506145328256, 0.08163785307378141, 0.12183765509467155, 
                                           0.17047357246983033, 0.22767466267090952, 0.29354456136020435, 0.36817025905048073,
                                           0.4516232599636818, 0.5439594844406513, 0.6452223428310663, 0.755452426041811,
                                           1.0, 1.0, 1.0, 1.0};
                const amrex::Real P[32] = {0.002500000000, -0.008716684163,
                                           0.007163369564, -0.016197037987,
                                           0.021777069026, -0.025465120029,
                                           0.047304023825, -0.035313893286,
                                           0.078818724642, -0.043310317965,
                                           0.118886124946, -0.050191425226,
                                           0.167821636357, -0.055539554113,
                                           0.225550738143, -0.059033777696,
                                           0.292132840525, -0.060396757760,
                                           0.367555632362, -0.059437571880,
                                           0.451828366715, -0.056047942169,
                                           0.544953180430, -0.050195056431,
                                           0.646939994627, -0.041883526517,
                                           0.799800790932, -0.026981867341,
                                           0.918433713697, -0.012656181473,
                                           1.000000000000, -0.001260000000};
                const amrex::Real W[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
                curve_bottom.set(U, P, W);

                /*
                const int nu = 10;
                for (int iu = 0; iu < nu; ++iu)
                {
                    const amrex::Real u = (1.0*iu)/(nu-1);
                    amrex::Real xx[AMREX_SPACEDIM];
                    curve_bottom.eval(u, xx);
                    amrex::Print() << "x: "; amrex::DG::IO::PrintRealArray2D(1, AMREX_SPACEDIM, xx);
                }
                */
            }


//const amrex::Real y[2] = {0.75, 0.001};
//d2 = curve_top.distance_from(y, tol);

            {
                const amrex::Real tol = 1.0e-8*c;
                amrex::Real d2;
                amrex::Real y[AMREX_SPACEDIM];
                bool cond;

                d2 = curve_top.distance_from(xs, tol);
                d2 = amrex::min(d2, curve_LE.distance_from(xs, tol));
                d2 = amrex::min(d2, curve_TE.distance_from(xs, tol));
                d2 = amrex::min(d2, curve_bottom.distance_from(xs, tol));

                cond = false;
                curve_TE.eval(0.5, y);
                if ((xs[0] >= 0.0) && (xs[0] <= y[0]))
                {
                    const amrex::Real th = this->NACA4_thickness(0.12, xs[0]);
                    const amrex::Real r = thTE/std::cos(phTE);
                    const amrex::Real dy[AMREX_SPACEDIM] = {xs[0]-(xTE-th*std::tan(phTE)), xs[1]};

                    cond = ((xs[0] < xTE) && (std::abs(xs[1]) < th));
                    cond = cond || ((xs[0] >= xTE) && ((dy[0]*dy[0]+dy[1]*dy[1]) < r*r));
                }

                if (cond)
                {
                    PHI[0] = +d2;
                }
                else
                {
                    PHI[0] = -d2;
                }
            }







            {
                const amrex::Real PTE[AMREX_SPACEDIM] = {xTE+thTE/std::tan(phTE), 0.0};
                const amrex::Real U[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
                const amrex::Real P[6] = {xTE, -thTE,
                                            PTE[0], PTE[1],
                                            xTE, +thTE};
                const amrex::Real W[3] = {1.0, std::sin(phTE), 1.0};
                
                curve_TE.set(U, P, W);
            }