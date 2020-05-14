#ifndef ALGOIM_TINYVECTOR_HPP
#define ALGOIM_TINYVECTOR_HPP

#include <iostream>
#include <iomanip>

namespace Algoim
{

template<typename T, size_t N>
using TinyVector = amrex::GpuArray<T, N>;

/*
template<typename T, int N>
struct TinyVector
{
    T data_[N];

    // OPERATORS ======================================================
    inline
    const T& operator[](unsigned i) const {return data_[i];}

    inline
    T& operator[](unsigned i) {return data_[i];}
    // ================================================================
};
*/

}

#endif