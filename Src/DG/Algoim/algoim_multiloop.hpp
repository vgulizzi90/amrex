#ifndef ALGOIM_MULTILOOP_HPP
#define ALGOIM_MULTILOOP_HPP

// Algoim::MultiLoop for writing N-dimensional nested for loops

#include "algoim_tinyvector.hpp"

namespace Algoim
{
    // MultiLoop is essentially an N-dimensional iterator for iterating over the logical grid coordinates
    // of a Cartesian grid with indices min(0) <= i < max(0), min(1) <= j < max(1), min(2) <= k < max(2), etc.
    template<int N>
    class MultiLoop
    {
        TinyVector<int,N> i;
        /*const*/ TinyVector<int,N> min, max;
        bool valid;
    public:
        MultiLoop(const int& min_, const int& max_)
        {
            for (int dim = 0; dim < N; ++dim)
            {
                i[dim] = min_;
                min[dim] = min_;
                max[dim] = max_;
            }

            valid = true;
            for (int dim = 0; dim < N; ++dim)
                valid &= min[dim] <= i[dim] && i[dim] < max[dim];
        }

        MultiLoop(const TinyVector<int,N>& min_, const TinyVector<int,N>& max_)
            : i(min_), min(min_), max(max_)
        {
            valid = true;
            for (int dim = 0; dim < N; ++dim)
                valid &= min[dim] <= i[dim] && i[dim] < max[dim];
        }

        inline MultiLoop& operator++()
        {
            for (int dim = N - 1; dim >= 0; --dim)
            {
                if (++i[dim] < max[dim])
                    return *this;
                i[dim] = min[dim];
            }
            valid = false;
            return *this;
        }

        inline const TinyVector<int,N>& operator()() const
        {
            return i;
        }

        inline operator bool() const
        {
            return valid;
        }

        inline int operator()(int index) const
        {
            return i[index];
        }

        inline TinyVector<int,N> extent() const
        {
            TinyVector<int,N> res;
            for (int dim = 0; dim < N; ++dim) res[dim] = max[dim]-min[dim];

            return res;
        }
    };
} // namespace Algoim

#endif
