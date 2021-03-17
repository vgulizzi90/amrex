//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_NURBS.H
 * \brief Header file for some auxiliary input/output routines.
*/

#ifndef AMREX_DG_NURBS_H_
#define AMREX_DG_NURBS_H_

#define __DG_NURBS_MAX_p__ 5

namespace amrex
{
namespace DG
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real norm2(const Real * A)
{
    const Real res = AMREX_D_TERM(A[0]*A[0],+A[1]*A[1],+A[2]*A[2]);
    return std::sqrt(res);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cross(const Real * A, const Real * B, Real * C)
{
    C[0] = A[1]*B[2]-A[2]*B[1];
    C[1] = A[2]*B[0]-A[0]*B[2];
    C[2] = A[0]*B[1]-A[1]*B[0];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real sqrdist(const Real * A, const Real * B)
{
    Real res = AMREX_D_TERM((A[0]-B[0])*(A[0]-B[0]),+(A[1]-B[1])*(A[1]-B[1]),+(A[2]-B[2])*(A[2]-B[2]));
    return res;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void curve_sqrdist_up_to_second_der(const Real * dkA, const Real * B, Real * d2p)
{
    d2p[0] = AMREX_D_TERM( (dkA[0]-B[0])*(dkA[0]-B[0]),
                          +(dkA[1]-B[1])*(dkA[1]-B[1]),
                          +(dkA[2]-B[2])*(dkA[2]-B[2]));
    d2p[1] = AMREX_D_TERM( (dkA[0]-B[0])*dkA[0+AMREX_SPACEDIM],
                          +(dkA[1]-B[1])*dkA[1+AMREX_SPACEDIM],
                          +(dkA[2]-B[2])*dkA[2+AMREX_SPACEDIM]);
    d2p[2] = AMREX_D_TERM( (dkA[0]-B[0])*dkA[0+2*AMREX_SPACEDIM]+dkA[0+AMREX_SPACEDIM]*dkA[0+AMREX_SPACEDIM],
                          +(dkA[1]-B[1])*dkA[1+2*AMREX_SPACEDIM]+dkA[1+AMREX_SPACEDIM]*dkA[1+AMREX_SPACEDIM],
                          +(dkA[2]-B[2])*dkA[2+2*AMREX_SPACEDIM]+dkA[2+AMREX_SPACEDIM]*dkA[2+AMREX_SPACEDIM]);

    d2p[1] *= 2.0;
    d2p[2] *= 2.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real curve_curvature(const Real * dkx)
{
    const Real * dx = &dkx[1*AMREX_SPACEDIM];
    const Real * d2x = &dkx[2*AMREX_SPACEDIM];
    
#if (AMREX_SPACEDIM == 2)
    Real k;
    k = (dx[0]*dx[0]+dx[1]*dx[1]);
    k = std::sqrt(k*k*k);
    k = std::abs(dx[0]*d2x[1]-dx[1]*d2x[0])/k;
#endif
#if (AMREX_SPACEDIM == 3)
    Real k;
    Abort("ERROR: AMReX_DG_NURBS.H - curve_curvature ");
#endif

    return k;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real end_points_distance_from(const int N, const Real * P, const Real * x, bool & all_points_are_behind)
{
    const Real * P0 = P;
    const Real * PN = &P[N*AMREX_SPACEDIM];
    const Real d0 = sqrdist(P0, x);
    const Real dN = sqrdist(PN, x);

    Real v[AMREX_SPACEDIM], vi[AMREX_SPACEDIM], tmp;
    int i;

    if (d0 < dN)
    {   
        AMREX_D_TERM
        (
            v[0] = P[0]-x[0];,
            v[1] = P[1]-x[1];,
            v[2] = P[2]-x[2];
        )
        
        i = 1;
        all_points_are_behind = true;
        while ((i <= N) && all_points_are_behind)
        {
            AMREX_D_TERM
            (
                vi[0] = P[0+i*AMREX_SPACEDIM]-P[0];,
                vi[1] = P[1+i*AMREX_SPACEDIM]-P[1];,
                vi[2] = P[2+i*AMREX_SPACEDIM]-P[2];
            )

            tmp = AMREX_D_TERM(v[0]*vi[0],+v[1]*vi[1],+v[2]*vi[2]);
            
            if (tmp < 0.0)
            {
                all_points_are_behind = false;
            }
            else
            {
                i += 1;
            }
        }

        if (all_points_are_behind)
        {
            tmp = d0;
        }
    }
    else
    {
        AMREX_D_TERM
        (
            v[0] = PN[0]-x[0];,
            v[1] = PN[1]-x[1];,
            v[2] = PN[2]-x[2];
        )

        i = N-1;
        all_points_are_behind = true;
        while ((i >= 0) && all_points_are_behind)
        {
            AMREX_D_TERM
            (
                vi[0] = P[0+i*AMREX_SPACEDIM]-P[0];,
                vi[1] = P[1+i*AMREX_SPACEDIM]-P[1];,
                vi[2] = P[2+i*AMREX_SPACEDIM]-P[2];
            )

            tmp = AMREX_D_TERM(v[0]*vi[0],+v[1]*vi[1],+v[2]*vi[2]);
            
            if (tmp < 0.0)
            {
                all_points_are_behind = false;
            }
            else
            {
                i -= 1;
            }
        }

        if (all_points_are_behind)
        {
            tmp = dN;
        }
    }
    
    return tmp;
}

namespace Bezier
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void Bernstein_table(const int n, const Real u, Real * B)
{
    const Real u1 = 1.0-u;
    Real saved, temp;

    B[0] = 1.0;

    for (int j = 1; j <= n; ++j)
    {
        saved = 0.0;
        for (int k = 0; k < j; ++k)
        {
            temp = B[k];
            B[k] = saved+u1*temp;
            saved = u*temp;
        }
        B[j] = saved;
    }
}

template <int N>
struct curve
{
    // DATA MEMBERS ===================================================
    Real P[AMREX_SPACEDIM*(1+N)];
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    curve(const Real * P_)
    {
        const int s = AMREX_SPACEDIM*(1+N);
        for (int k = 0; k < s; ++k)
        {
            this->P[k] = P_[k];
        }
    }
    // ================================================================

    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE
    void eval(const Real u, Real * x) const
    {
        Real B[1+N];
        Bernstein_table(N, u, B);
        
        AMREX_D_TERM
        (
            x[0] = 0.0;,
            x[1] = 0.0;,
            x[2] = 0.0;
        )
        for (int k = 0; k <= N; ++k)
        {
            AMREX_D_TERM
            (
                x[0] += B[k]*this->P[0+k*AMREX_SPACEDIM];,
                x[1] += B[k]*this->P[1+k*AMREX_SPACEDIM];,
                x[2] += B[k]*this->P[2+k*AMREX_SPACEDIM];
            )
        }
    }
    // ================================================================
};

} // namespace Bezier


namespace rational_Bezier
{

template <int N>
struct curve
{
    // DATA MEMBERS ===================================================
    Real P[AMREX_SPACEDIM*(1+N)], W[1+N];
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    curve()
    {}

    AMREX_GPU_HOST_DEVICE
    curve(const Real * P_, const Real * W_)
    {
        const int s = AMREX_SPACEDIM*(1+N);
        for (int k = 0; k < s; ++k)
        {
            this->P[k] = P_[k];
        }
        for (int k = 0; k <= N; ++k)
        {
            this->W[k] = W_[k];
        }
    }
    // ================================================================

    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE
    void eval(const Real u, Real * x) const
    {
        Real B[1+N], w;
        Bezier::Bernstein_table(N, u, B);
        
        AMREX_D_TERM
        (
            x[0] = 0.0;,
            x[1] = 0.0;,
            x[2] = 0.0;
        )
        w = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            AMREX_D_TERM
            (
                x[0] += B[k]*this->W[k]*this->P[0+k*AMREX_SPACEDIM];,
                x[1] += B[k]*this->W[k]*this->P[1+k*AMREX_SPACEDIM];,
                x[2] += B[k]*this->W[k]*this->P[2+k*AMREX_SPACEDIM];
            )
            w += this->W[k]*B[k];
        }
        w = 1.0/w;
        AMREX_D_TERM
        (
            x[0] *= w;,
            x[1] *= w;,
            x[2] *= w;
        )
    }

    AMREX_GPU_HOST_DEVICE
    void eval_up_to_second_der(const Real u, Real * dkx) const
    {
        Real B[1+N], B1[N], B2[N-1];
        Real dkw[3], dkA[AMREX_SPACEDIM*3];
        Bezier::Bernstein_table(N, u, B);
        Bezier::Bernstein_table(N-1, u, B1);
        Bezier::Bernstein_table(N-2, u, B2);
        
        // 0-TH DERIVATIVE
        AMREX_D_TERM
        (
            dkA[0] = 0.0;,
            dkA[1] = 0.0;,
            dkA[2] = 0.0;
        )
        dkw[0] = 0.0;
        for (int k = 0; k <= N; ++k)
        {
            AMREX_D_TERM
            (
                dkA[0] += B[k]*this->W[k]*this->P[0+k*AMREX_SPACEDIM];,
                dkA[1] += B[k]*this->W[k]*this->P[1+k*AMREX_SPACEDIM];,
                dkA[2] += B[k]*this->W[k]*this->P[2+k*AMREX_SPACEDIM];
            )
            dkw[0] += this->W[k]*B[k];
        }
        AMREX_D_TERM
        (
            dkx[0] = dkA[0]/dkw[0];,
            dkx[1] = dkA[1]/dkw[0];,
            dkx[2] = dkA[2]/dkw[0];
        )

        // 1-ST DERIVATIVE
        {
            int j = 1;

            AMREX_D_TERM
            (
                dkA[0+j*AMREX_SPACEDIM] = 0.0;,
                dkA[1+j*AMREX_SPACEDIM] = 0.0;,
                dkA[2+j*AMREX_SPACEDIM] = 0.0;
            )
            dkw[j] = 0.0;

            for (int k = 0; k <= (N-1); ++k)
            {
                AMREX_D_TERM
                (
                    dkA[0+j*AMREX_SPACEDIM] += B1[k]*(this->W[k+1]*this->P[0+(k+1)*AMREX_SPACEDIM]-this->W[k]*this->P[0+k*AMREX_SPACEDIM]);,
                    dkA[1+j*AMREX_SPACEDIM] += B1[k]*(this->W[k+1]*this->P[1+(k+1)*AMREX_SPACEDIM]-this->W[k]*this->P[1+k*AMREX_SPACEDIM]);,
                    dkA[2+j*AMREX_SPACEDIM] += B1[k]*(this->W[k+1]*this->P[2+(k+1)*AMREX_SPACEDIM]-this->W[k]*this->P[2+k*AMREX_SPACEDIM]);
                )
                dkw[j] += B1[k]*(this->W[k+1]-this->W[k]);
            }

            AMREX_D_TERM
            (
                dkA[0+j*AMREX_SPACEDIM] *= N;,
                dkA[1+j*AMREX_SPACEDIM] *= N;,
                dkA[2+j*AMREX_SPACEDIM] *= N;
            )
            dkw[j] *= N;

            AMREX_D_TERM
            (
                dkx[0+j*AMREX_SPACEDIM] = (dkA[0+j*AMREX_SPACEDIM]-dkw[j]*dkx[0])/dkw[0];,
                dkx[1+j*AMREX_SPACEDIM] = (dkA[1+j*AMREX_SPACEDIM]-dkw[j]*dkx[1])/dkw[0];,
                dkx[2+j*AMREX_SPACEDIM] = (dkA[2+j*AMREX_SPACEDIM]-dkw[j]*dkx[2])/dkw[0];
            )
        }

        // 2-ND DERIVATIVE
        {
            int j = 2;

            AMREX_D_TERM
            (
                dkA[0+j*AMREX_SPACEDIM] = 0.0;,
                dkA[1+j*AMREX_SPACEDIM] = 0.0;,
                dkA[2+j*AMREX_SPACEDIM] = 0.0;
            )
            dkw[j] = 0.0;

            for (int k = 0; k <= (N-2); ++k)
            {
                AMREX_D_TERM
                (
                    dkA[0+j*AMREX_SPACEDIM] += B2[k]*(this->W[k+2]*this->P[0+(k+2)*AMREX_SPACEDIM]-2.0*this->W[k+1]*this->P[0+(k+1)*AMREX_SPACEDIM]+this->W[k]*this->P[0+k*AMREX_SPACEDIM]);,
                    dkA[1+j*AMREX_SPACEDIM] += B2[k]*(this->W[k+2]*this->P[1+(k+2)*AMREX_SPACEDIM]-2.0*this->W[k+1]*this->P[1+(k+1)*AMREX_SPACEDIM]+this->W[k]*this->P[1+k*AMREX_SPACEDIM]);,
                    dkA[2+j*AMREX_SPACEDIM] += B2[k]*(this->W[k+2]*this->P[2+(k+2)*AMREX_SPACEDIM]-2.0*this->W[k+1]*this->P[2+(k+1)*AMREX_SPACEDIM]+this->W[k]*this->P[2+k*AMREX_SPACEDIM]);
                )
                dkw[j] += B2[k]*(this->W[k+2]-2.0*this->W[k+1]+this->W[k]);
            }

            AMREX_D_TERM
            (
                dkA[0+j*AMREX_SPACEDIM] *= N*(N-1);,
                dkA[1+j*AMREX_SPACEDIM] *= N*(N-1);,
                dkA[2+j*AMREX_SPACEDIM] *= N*(N-1);
            )
            dkw[j] *= N*(N-1);

            AMREX_D_TERM
            (
                dkx[0+j*AMREX_SPACEDIM] = (dkA[0+j*AMREX_SPACEDIM]-dkw[j]*dkx[0]-2.0*dkw[j-1]*dkx[0+(j-1)*AMREX_SPACEDIM])/dkw[0];,
                dkx[1+j*AMREX_SPACEDIM] = (dkA[1+j*AMREX_SPACEDIM]-dkw[j]*dkx[1]-2.0*dkw[j-1]*dkx[1+(j-1)*AMREX_SPACEDIM])/dkw[0];,
                dkx[2+j*AMREX_SPACEDIM] = (dkA[2+j*AMREX_SPACEDIM]-dkw[j]*dkx[2]-2.0*dkw[j-1]*dkx[2+(j-1)*AMREX_SPACEDIM])/dkw[0];
            )
        }
    }
    // ================================================================

    // COMPUTE THE DISTANCE FROM A POINT ==============================
    AMREX_GPU_HOST_DEVICE
    Real distance_from(const Real * x, const Real tol = 1.0e-8) const
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
        
        // PARAMETERS -------------------------------------------------
        const int it_max = 20;
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        Real ulo, uhi, u, du, dp[3];
        Real dkx[AMREX_SPACEDIM*3];
        
        int it;
        Real err;
        // ------------------------------------------------------------

        // START NEWTON-RAPHSON SEARCH --------------------------------
        {
            Real dplo, dphi;

            ulo = 0.0;
            this->eval_up_to_second_der(ulo, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            dplo = dp[1];
            uhi = 1.0;
            this->eval_up_to_second_der(uhi, dkx);
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
                Abort(msg);
#endif
            }

            if (dplo == 0.0)
            {
                this->eval_up_to_second_der(ulo, dkx);
                return sqrdist(dkx, x);
            }
            if (dphi == 0.0)
            {
                this->eval_up_to_second_der(uhi, dkx);
                return sqrdist(dkx, x);
            }

            if (dphi < 0.0)
            {
                const Real tmp = uhi;
                uhi = ulo;
                ulo = tmp;
            }
        }

        u = 0.5*(ulo+uhi);
        du = uhi-ulo;
        this->eval_up_to_second_der(u, dkx);
        curve_sqrdist_up_to_second_der(dkx, x, dp);

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

            this->eval_up_to_second_der(u, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);

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

        //u = 0.5*(ulo+uhi);
        this->eval_up_to_second_der(u, dkx);
        curve_sqrdist_up_to_second_der(dkx, x, dp);

        //Print() << "x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
        //Print() << "it: " << it << std::endl;
        //Print() << "u: " << u << std::endl;
        //Print() << "dp[1]: " << dp[1] << std::endl;
        //Print() << "xlo: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, dkxlo);
        //Print() << "xhi: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, dkxhi);
        // ------------------------------------------------------------

        return dp[0];
    }
    // ================================================================
};

} // namespace rational_Bezier

namespace bsplines
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int find_span(const int n, const int p, const Real u, const Real * U)
{
    if (u == U[n+1]) return n;

    int low, high, mid;
    
    low = p;
    high = n+1;
    mid = (int) std::floor(0.5*(low+high));
    while (u < U[mid] || u >= U[mid+1])
    {
        if (u < U[mid])
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
        mid = (int) std::floor(0.5*(low+high));
    }
    return mid;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void basis_functions_table(const int i, const Real u, const int p, const Real * U, Real * N)
{
    Real left[1+__DG_NURBS_MAX_p__], right[1+__DG_NURBS_MAX_p__], saved, temp;

    for (int j = 0; j <= p; ++j)
    {
        left[j] = 0.0;
        right[j] = 0.0;
    }

    N[0] = 1.0;
    for (int j = 1; j <= p; ++j)
    {
        left[j] = u-U[i+1-j];
        right[j] = U[i+j]-u;
        saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            temp = N[r]/(right[r+1]+left[j-r]);
            N[r] = saved+right[r+1]*temp;
            saved = left[j-r]*temp;
        }
        N[j] = saved;
    }
}

/**
 * \brief Compute the derivatives of the Bsplines basis functions up to order n.
 *
 * Source: The NURBS book - Piegl and Tiller - pag 72
 *
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void basis_functions_der_table(const int i, const Real u, const int p, const int n, const Real * U, Real * dN)
{
    Real left[1+__DG_NURBS_MAX_p__], right[1+__DG_NURBS_MAX_p__];
    Real saved, temp, d;
    Real ndu[1+__DG_NURBS_MAX_p__][1+__DG_NURBS_MAX_p__], a[2][1+__DG_NURBS_MAX_p__];
    int s1, s2, rk, pk, j1, j2;

    for (int j = 0; j <= p; ++j)
    {
        left[j] = 0.0;
        right[j] = 0.0;
    }

    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j)
    {
        left[j] = u-U[i+1-j];
        right[j] = U[i+j]-u;
        saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            ndu[j][r] = right[r+1]+left[j-r];
            temp = ndu[r][j-1]/ndu[j][r];

            ndu[r][j] = saved+right[r+1]*temp;
            saved = left[j-r]*temp;
        }
        ndu[j][j] = saved;
    }

    for (int j = 0; j <= p; ++j)
    {
        //dN[0][j] = ndu[j][p];
        dN[j] = ndu[j][p];
    }

    for (int r = 0; r <= p; ++r)
    {
        s1 = 0;
        s2 = 1;
        a[0][0] = 1.0;

        for (int k = 1; k <= n; ++k)
        {
            d = 0.0;
            rk = r-k;
            pk = p-k;

            if (r >= k)
            {
                a[s2][0] = a[s1][0]/ndu[pk+1][rk];
                d = a[s2][0]*ndu[rk][pk];
            }
            if (rk >= -1)
            {
                j1 = 1;
            }
            else
            {
                j1 = -rk;
            }
            if (r-1 <= pk)
            {
                j2 = k-1;
            }
            else
            {
                j2 = p-r;
            }

            for (int j = j1; j <= j2; ++j)
            {
                a[s2][j] = (a[s1][j]-a[s1][j-1])/ndu[pk+1][rk+j];
                d += a[s2][j]*ndu[rk+j][pk];
            }

            if (r <= pk)
            {
                a[s2][k] = -a[s1][k-1]/ndu[pk+1][r];
                d += a[s2][k]*ndu[r][pk];
            }
            //dN[k][r] = d;
            dN[r+k*(1+p)] = d;

            int itmp = s1;
            s1 = s2;
            s2 = itmp;
        }
    }

    {
        int r = p;
        
        for (int k = 1; k <= n; ++k)
        {
            for (int j = 0; j <= p; ++j)
            {
                //dN[k][j] *= r;
                dN[j+k*(1+p)] *= r;
            }
            r *= (p-k);
        }
    }
}

template <int M, int N, int D>
struct curve
{
    // DATA MEMBERS ===================================================
    Real U[1+M], P[AMREX_SPACEDIM*(1+N)];
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    curve()
    {
        assert(N == M-D-1);
    }

    AMREX_GPU_HOST_DEVICE
    curve(const Real * U_, const Real * P_)
    {
        assert(N == M-D-1);

        this->set(U_, P_);
    }
    // ================================================================

    // AUXILIARY ======================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set(const Real * U_, const Real * P_)
    {
        for (int k = 0; k <= M; ++k)
        {
            this->U[k] = U_[k];
        }

        for (int k = 0; k <= N; ++k)
        {
            AMREX_D_TERM
            (
                this->P[0+k*AMREX_SPACEDIM] = P_[0+k*AMREX_SPACEDIM];,
                this->P[1+k*AMREX_SPACEDIM] = P_[1+k*AMREX_SPACEDIM];,
                this->P[2+k*AMREX_SPACEDIM] = P_[2+k*AMREX_SPACEDIM];
            )
        }
    }
    // ================================================================

    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE
    void eval(const Real u, Real * x) const
    {
        const int span = find_span(N, D, u, this->U);

        Real BF[1+D];
        
        basis_functions_table(span, u, D, this->U, BF);

        AMREX_D_TERM
        (
            x[0] = 0.0;,
            x[1] = 0.0;,
            x[2] = 0.0;
        )
        for (int k = 0; k <= D; ++k)
        {
            AMREX_D_TERM
            (
                x[0] += BF[k]*this->P[0+(k+span-D)*AMREX_SPACEDIM];,
                x[1] += BF[k]*this->P[1+(k+span-D)*AMREX_SPACEDIM];,
                x[2] += BF[k]*this->P[2+(k+span-D)*AMREX_SPACEDIM];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void eval_der(const Real u, const int d, Real * dkx) const
    {
        const int du = amrex::min(d, D);
        const int span = find_span(N, D, u, this->U);

        Real dBF[(1+D)*(1+D)];

        for (int k = D+1; k <= du; ++k)
        {
            AMREX_D_TERM
            (
                dkx[0+k*AMREX_SPACEDIM] = 0.0;,
                dkx[1+k*AMREX_SPACEDIM] = 0.0;,
                dkx[2+k*AMREX_SPACEDIM] = 0.0;
            )
        }

        basis_functions_der_table(span, u, D, du, this->U, dBF);

        for (int k = 0; k <= du; ++k)
        {
            AMREX_D_TERM
            (
                dkx[0+k*AMREX_SPACEDIM] = 0.0;,
                dkx[1+k*AMREX_SPACEDIM] = 0.0;,
                dkx[2+k*AMREX_SPACEDIM] = 0.0;
            )
            for (int j = 0; j <= D; ++j)
            {
                AMREX_D_TERM
                (
                    dkx[0+k*AMREX_SPACEDIM] += dBF[j+k*du]*this->P[0+(j+span-D)*AMREX_SPACEDIM];,
                    dkx[1+k*AMREX_SPACEDIM] += dBF[j+k*du]*this->P[1+(j+span-D)*AMREX_SPACEDIM];,
                    dkx[2+k*AMREX_SPACEDIM] += dBF[j+k*du]*this->P[2+(j+span-D)*AMREX_SPACEDIM];
                )
            }
        }
    }
    // ================================================================

};

template <int M1, int M2, int N1, int N2, int D1, int D2>
struct surface
{
    // DATA MEMBERS ===================================================
    Real U1[1+M1], U2[1+M2], P[AMREX_SPACEDIM*(1+N1)*(1+N2)];
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    surface()
    {
        assert(N1 == M1-D1-1);
        assert(N2 == M2-D2-1);
    }

    AMREX_GPU_HOST_DEVICE
    surface(const Real * U1_, const Real * U2_, const Real * P_)
    {
        assert(N1 == M1-D1-1);
        assert(N2 == M2-D2-1);

        this->set(U1_, U2_, P_);
    }
    // ================================================================

    // AUXILIARY ======================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set(const Real * U1_, const Real * U2_, const Real * P_)
    {
        for (int k = 0; k <= M1; ++k)
        {
            this->U1[k] = U1_[k];
        }
        for (int k = 0; k <= M2; ++k)
        {
            this->U2[k] = U2_[k];
        }

        for (int k = 0; k < (1+N1)*(1+N2); ++k)
        {
            AMREX_D_TERM
            (
                this->P[0+k*AMREX_SPACEDIM] = P_[0+k*AMREX_SPACEDIM];,
                this->P[1+k*AMREX_SPACEDIM] = P_[1+k*AMREX_SPACEDIM];,
                this->P[2+k*AMREX_SPACEDIM] = P_[2+k*AMREX_SPACEDIM];
            )
        }
    }
    // ================================================================

    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE
    void eval(const Real u1, const Real u2, Real * x) const
    {
        const int span1 = find_span(N1, D1, u1, this->U1);
        const int span2 = find_span(N2, D2, u2, this->U2);

        Real BF1[1+D1], BF2[1+D2], tmp[AMREX_SPACEDIM];
        int iu1, iu2;
        
        basis_functions_table(span1, u1, D1, this->U1, BF1);
        basis_functions_table(span2, u2, D2, this->U2, BF2);

        iu1 = span1-D1;
        iu2 = span2-D2;

        AMREX_D_TERM
        (
            x[0] = 0.0;,
            x[1] = 0.0;,
            x[2] = 0.0;
        )
        for (int k2 = 0; k2 <= D2; ++k2)
        {
            AMREX_D_TERM
            (
                tmp[0] = 0.0;,
                tmp[1] = 0.0;,
                tmp[2] = 0.0;
            )
            for (int k1 = 0; k1 <= D1; ++k1)
            {
                AMREX_D_TERM
                (
                    tmp[0] += BF1[k1]*this->P[0+(k1+iu1+(1+N1)*(k2+iu2))*AMREX_SPACEDIM];,
                    tmp[1] += BF1[k1]*this->P[1+(k1+iu1+(1+N1)*(k2+iu2))*AMREX_SPACEDIM];,
                    tmp[2] += BF1[k1]*this->P[2+(k1+iu1+(1+N1)*(k2+iu2))*AMREX_SPACEDIM];
                )
            }
            AMREX_D_TERM
            (
                x[0] += BF2[k2]*tmp[0];,
                x[1] += BF2[k2]*tmp[1];,
                x[2] += BF2[k2]*tmp[2];
            )
        }
    }

    AMREX_GPU_HOST_DEVICE
    void eval_der(const Real u1, const Real u2, const int d, Real * dkx) const
    {
        const int du1 = amrex::min(d, D1);
        const int du2 = amrex::min(d, D2);
        const int span1 = find_span(N1, D1, u1, this->U1);
        const int span2 = find_span(N2, D2, u2, this->U2);

        Real dBF1[(1+D1)*(1+D1)], dBF2[(1+D2)*(1+D2)], tmp[(1+D2)*AMREX_SPACEDIM];
        int iu1, iu2, dd;

        for (int k1 = D1+1; k1 <= d; ++k1)
        for (int k2 = 0; k2 <= d-k1; ++k2)
        {
            AMREX_D_TERM
            (
                dkx[0+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                dkx[1+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                dkx[2+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;
            )
        }
        for (int k2 = D2+1; k2 <= d; ++k2)
        for (int k1 = 0; k1 <= d-k2; ++k1)
        {
            AMREX_D_TERM
            (
                dkx[0+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                dkx[1+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                dkx[2+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;
            )
        }

        basis_functions_der_table(span1, u1, D1, du1, this->U1, dBF1);
        basis_functions_der_table(span2, u2, D2, du2, this->U2, dBF2);

        iu1 = span1-D1;
        iu2 = span2-D2;

        for (int k1 = 0; k1 <= du1; ++k1)
        {
            for (int j2 = 0; j2 <= D2; ++j2)
            {
                AMREX_D_TERM
                (
                    tmp[0+j2*AMREX_SPACEDIM] = 0.0;,
                    tmp[1+j2*AMREX_SPACEDIM] = 0.0;,
                    tmp[2+j2*AMREX_SPACEDIM] = 0.0;
                )
                for (int j1 = 0; j1 <= D1; ++j1)
                {
                    AMREX_D_TERM
                    (
                        tmp[0+j2*AMREX_SPACEDIM] += dBF1[j1+k1*du1]*this->P[0+(j1+iu1+(1+N1)*(j2+iu2))*AMREX_SPACEDIM];,
                        tmp[1+j2*AMREX_SPACEDIM] += dBF1[j1+k1*du1]*this->P[1+(j1+iu1+(1+N1)*(j2+iu2))*AMREX_SPACEDIM];,
                        tmp[2+j2*AMREX_SPACEDIM] += dBF1[j1+k1*du1]*this->P[2+(j1+iu1+(1+N1)*(j2+iu2))*AMREX_SPACEDIM];
                    )
                }
            }

            dd = amrex::min(d-k1, du2);

            for (int k2 = 0; k2 <= dd; ++k2)
            {
                AMREX_D_TERM
                (
                    dkx[0+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                    dkx[1+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;,
                    dkx[2+(k1+(1+d)*k2)*AMREX_SPACEDIM] = 0.0;
                )
                for (int j2 = 0; j2 <= D2; ++j2)
                {
                    AMREX_D_TERM
                    (
                        dkx[0+(k1+(1+d)*k2)*AMREX_SPACEDIM] += dBF2[j2+k2*du2]*tmp[0+j2*AMREX_SPACEDIM];,
                        dkx[1+(k1+(1+d)*k2)*AMREX_SPACEDIM] += dBF2[j2+k2*du2]*tmp[1+j2*AMREX_SPACEDIM];,
                        dkx[2+(k1+(1+d)*k2)*AMREX_SPACEDIM] += dBF2[j2+k2*du2]*tmp[2+j2*AMREX_SPACEDIM];
                    )
                }
            }
        }
    }
    // ================================================================

};

} // namespace bsplines

namespace nurbs
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int binomial(const int n, const int k)
{
    int res = 1;
    if (k > n)
    {
        return 0;
    }
    for (int i = 1; i <= k; ++i)
    {
        res *= (n+1-i);
        res /= i;
    }
    return res;
}


template <int M, int N, int D>
struct curve
:
public bsplines::curve<M, N, D>
{
    // DATA MEMBERS ===================================================
    Real W[1+N];
    // ================================================================

    // CONSTRUCTORS ===================================================
    AMREX_GPU_HOST_DEVICE
    curve()
    :
    bsplines::curve<M, N, D>()
    {}
    
    AMREX_GPU_HOST_DEVICE
    curve(const Real * U_, const Real * P_, const Real * W_)
    :
    bsplines::curve<M, N, D>()
    {
        this->set(U_, P_, W_);
    }
    // ================================================================

    // AUXILIARY ======================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void set(const Real * U_, const Real * P_, const Real * W_)
    {
        for (int k = 0; k <= M; ++k)
        {
            this->U[k] = U_[k];
        }

        for (int k = 0; k <= N; ++k)
        {
            AMREX_D_TERM
            (
                this->P[0+k*AMREX_SPACEDIM] = P_[0+k*AMREX_SPACEDIM];,
                this->P[1+k*AMREX_SPACEDIM] = P_[1+k*AMREX_SPACEDIM];,
                this->P[2+k*AMREX_SPACEDIM] = P_[2+k*AMREX_SPACEDIM];
            )
        }

        for (int k = 0; k <= N; ++k)
        {
            this->W[k] = W_[k];
        }
    }
    // ================================================================

    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE
    void eval(const Real u, Real * x) const
    {
        const int span = bsplines::find_span(N, D, u, this->U);

        Real BF[1+D];
        Real w;
        
        bsplines::basis_functions_table(span, u, D, this->U, BF);

        AMREX_D_TERM
        (
            x[0] = 0.0;,
            x[1] = 0.0;,
            x[2] = 0.0;
        )
        w = 0.0;
        for (int k = 0; k <= D; ++k)
        {
            AMREX_D_TERM
            (
                x[0] += BF[k]*this->P[0+(k+span-D)*AMREX_SPACEDIM]*this->W[k+span-D];,
                x[1] += BF[k]*this->P[1+(k+span-D)*AMREX_SPACEDIM]*this->W[k+span-D];,
                x[2] += BF[k]*this->P[2+(k+span-D)*AMREX_SPACEDIM]*this->W[k+span-D];
            )
            w += BF[k]*this->W[k+span-D];
        }
        w = 1.0/w;
        AMREX_D_TERM
        (
            x[0] *= w;,
            x[1] *= w;,
            x[2] *= w;
        )
    }

    AMREX_GPU_HOST_DEVICE
    void eval_der(const Real u, const int d, Real * dkx) const
    {
        const int du = amrex::min(d, D);
        const int span = bsplines::find_span(N, D, u, this->U);

        Real dBF[(1+D)*(1+D)];
        Real dkA[AMREX_SPACEDIM*(1+D)], dkw[1+D];
        Real v[AMREX_SPACEDIM];

        bsplines::basis_functions_der_table(span, u, D, du, this->U, dBF);

        /*
        {
            Print() << "dBF(" << u << "): " << std::endl;
            IO::PrintRealArray2D(1+D, 1+D, dBF);
            //exit(-1);
        }
        */

        for (int k = D+1; k <= d; ++k)
        {
            AMREX_D_TERM
            (
                dkA[0+k*AMREX_SPACEDIM] = 0.0;,
                dkA[1+k*AMREX_SPACEDIM] = 0.0;,
                dkA[2+k*AMREX_SPACEDIM] = 0.0;
            )
            dkw[k] = 0.0;
        }
        for (int k = 0; k <= du; ++k)
        {
            AMREX_D_TERM
            (
                dkA[0+k*AMREX_SPACEDIM] = 0.0;,
                dkA[1+k*AMREX_SPACEDIM] = 0.0;,
                dkA[2+k*AMREX_SPACEDIM] = 0.0;
            )
            dkw[k] = 0.0;
            for (int j = 0; j <= D; ++j)
            {
                AMREX_D_TERM
                (
                    dkA[0+k*AMREX_SPACEDIM] += dBF[j+k*(1+D)]*this->W[j+span-D]*this->P[0+(j+span-D)*AMREX_SPACEDIM];,
                    dkA[1+k*AMREX_SPACEDIM] += dBF[j+k*(1+D)]*this->W[j+span-D]*this->P[1+(j+span-D)*AMREX_SPACEDIM];,
                    dkA[2+k*AMREX_SPACEDIM] += dBF[j+k*(1+D)]*this->W[j+span-D]*this->P[2+(j+span-D)*AMREX_SPACEDIM];
                )
                dkw[k] += dBF[j+k*(1+D)]*this->W[j+span-D];
            }
        }

        /*
        {
            Print() << "dkA/dkw: " << std::endl;
            IO::PrintRealArray2D(AMREX_SPACEDIM, 1+D, dkA);
            IO::PrintRealArray2D(1, 1+D, dkw);
            //exit(-1);
        }
        */

        for (int k = 0; k <= d; ++k)
        {
            AMREX_D_TERM
            (
                v[0] = dkA[0+k*AMREX_SPACEDIM];,
                v[1] = dkA[1+k*AMREX_SPACEDIM];,
                v[2] = dkA[2+k*AMREX_SPACEDIM];
            )
            for (int i = 1; i <= k; ++i)
            {
                const int bn = binomial(k, i);

                AMREX_D_TERM
                (
                    v[0] -= bn*dkw[i]*dkx[0+(k-i)*AMREX_SPACEDIM];,
                    v[1] -= bn*dkw[i]*dkx[1+(k-i)*AMREX_SPACEDIM];,
                    v[2] -= bn*dkw[i]*dkx[2+(k-i)*AMREX_SPACEDIM];
                )
            }
            AMREX_D_TERM
            (
                dkx[0+k*AMREX_SPACEDIM] = v[0]/dkw[0];,
                dkx[1+k*AMREX_SPACEDIM] = v[1]/dkw[0];,
                dkx[2+k*AMREX_SPACEDIM] = v[2]/dkw[0];
            )
        }

        /*
        {
            Print() << "dkx: " << std::endl;
            IO::PrintRealArray2D(AMREX_SPACEDIM, d+1, dkx);
            if (u > 0.05) exit(-1);
        }
        */
    }
    // ================================================================

    // COMPUTE THE DISTANCE FROM A POINT ==============================
    AMREX_GPU_HOST_DEVICE
    Real distance_from(const Real * x, const Real tol = 1.0e-8, Real * u_ptr = nullptr) const
    {
        // CHECK WHETHER ONE OF THE ENDPOINTS IS THE CLOSEST ----------
        /*
        {
            bool all_points_are_behind;
            const Real d2 = end_points_distance_from(N, this->P, x, all_points_are_behind);

            if (all_points_are_behind)
            {
                return d2;
            }
        }
        */
        // ------------------------------------------------------------

        // PARAMETERS -------------------------------------------------
        const Real u_tol = 1.0e-8;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        Real u, dp[3], dkx[AMREX_SPACEDIM*3];
        Real res;
        // ------------------------------------------------------------
//Print() << "x: "; IO::PrintRealArray2D(1, AMREX_SPACEDIM, x);
        // SAMPLE THE CURVE -------------------------------------------
        {
            const int nu = (N+1)*D;
            const Real du = 1.0/(nu-1);
            Real uu;

            {
                u = u_tol;
                this->eval_der(u, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                res = dp[0];
                if (u_ptr != nullptr)
                {
                    *u_ptr = u;
                }

//Print() << "dp(" << u << "): " << dp[0] << std::endl;

            }
            for (int iu = 1; iu < nu; ++iu)
            {
                uu = (1.0-2.0*u_tol)*iu*du+u_tol;
                this->eval_der(uu, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (dp[0] < res)
                {
                    u = uu;
                    res = dp[0];
                    if (u_ptr != nullptr)
                    {
                        *u_ptr = u;
                    }
                }
//Print() << "dp(" << uu << "): " << dp[0] << std::endl;
            }
        }
        // ------------------------------------------------------------

        // PERFORM THE ITERATIVE SEARCH -------------------------------
        {
            const int it_max = 20;
            int it;
            Real err, du;

            this->eval_der(u, 2, dkx);
            curve_sqrdist_up_to_second_der(dkx, x, dp);
            if (dp[0] < res)
            {
                res = dp[0];
                if (u_ptr != nullptr)
                {
                    *u_ptr = u;
                }
            }
            err = std::abs(dp[1]);

//Print() << "u, err: " << u << " " << err << std::endl;
            
            it = 0;
            while ((it < it_max) && (err > tol))
            {
                if (std::abs(dp[2]) < 1.0e-4)
                {
                    const Real * dx = &dkx[AMREX_SPACEDIM];
                    const Real dx2 = AMREX_D_TERM(dx[0]*dx[0],+dx[1]*dx[1],+dx[2]*dx[2]);
                    du = -0.5*dp[1]/dx2;
                }
                else
                {
                    du = -dp[1]/dp[2];
                }
                
                u += du;
                u = amrex::max(0.0, amrex::min(u, 1.0));
                
                this->eval_der(u, 2, dkx);
                curve_sqrdist_up_to_second_der(dkx, x, dp);
                if (dp[0] < res)
                {
                    res = dp[0];
                    if (u_ptr != nullptr)
                    {
                        *u_ptr = u;
                    }
                }
                err = std::abs(dp[1]);

//Print() << "du, u, err: " << du << " " << u << " " << err << std::endl;

                it += 1;
            }

//exit(-1);
        }
        // ------------------------------------------------------------

        return res;
    }
    // ================================================================

};

} // namespace nurbs

} // namespace DG
} // namespace amrex

#endif
