#ifndef ALGOIM_BOUNDINGBOX_HPP
#define AlGOIM_BOUNDINGBOX_HPP

// Algoim::BoundingBox<T,N>

#include "algoim_real.hpp"
#include "algoim_tinyvector.hpp"

namespace Algoim
{
    // BoundingBox<T,N> describes the extent of a hyperrectangle, i.e., the set of points
    // x = (x(0), ..., x(i), ..., x(N-1)) such that xmin(i) <= x(i) <= xmax(i) for all i.
    template<typename T, int N>
    struct BoundingBox
    {
        TinyVector<TinyVector<T,N>,2> range;

        BoundingBox(const Real* xlo, const Real* xhi)
        {
            for (int n = 0; n < N; ++n) range[0][n] = xlo[n], range[1][n] = xhi[n];
        }
        BoundingBox(const TinyVector<T,N>& min, const TinyVector<T,N>& max)
        {
            for (int n = 0; n < N; ++n)
            {
                range[0][n] = min[n];
                range[1][n] = max[n];
            }
        }

        inline const TinyVector<T,N>& operator() (int i) const
        {
            return range[i];
        }

        inline const TinyVector<T,N>& min() const
        {
            return range[0];
        }

        inline T& min(int i)
        {
            return range[0][i];
        }

        inline T min(int i) const
        {
            return range[0][i];
        }

        inline const TinyVector<T,N>& max() const
        {
            return range[1];
        }

        inline T& max(int i)
        {
            return range[1][i];
        }

        inline T max(int i) const
        {
            return range[1][i];
        }

        inline TinyVector<T,N> extent() const
        {
            TinyVector<Real,N> res;
            for (int n = 0; n < N; ++n) res(n) = range[1][n]-range[0][n];
            return res;
        }

        inline T extent(int i) const
        {
            return range[1][i]-range[0][i];
        }

        inline TinyVector<Real,N> midpoint() const
        {
            TinyVector<Real,N> res;
            for (int n = 0; n < N; ++n) res[n] = 0.5*(range[0][n]+range[1][n]);
            return res;
        }

        inline Real midpoint(int i) const
        {
            return (range[0][i]+range[1][i])*0.5;
        }

        inline bool operator==(const BoundingBox& x) const
        {
            for (int dim = 0; dim < N; ++dim)
                if (range[0][dim] != x.range[0][dim] || range[1][dim] != x.range[1][dim])
                    return false;
            return true;
        }
    };
} // namespace Algoim

#endif
