//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_StandardElements.H
 * \brief Constants and data structures of standard elements for discontinuous Galerkin methods.
*/


#ifndef AMREX_DG_STANDARD_ELEMENTS_H_
#define AMREX_DG_STANDARD_ELEMENTS_H_

#include <AMReX_Gpu.H>

#include <algoim_quad.hpp>

#include <AMReX_dG_Tables.H>
#include <AMReX_dG_Utils.H>
#include <AMReX_dG_BasisFunctions.H>
#include <AMReX_dG_LinAlg.H>

namespace amrex
{
namespace dG
{

// STANDARD RECTANGLE #################################################
/**
 * \brief This struct contains info about the standard hyper-rectangular element.
 *
 * This struct contains constants and methods related to a hyper-rectangle with sizes
 * [0,dx_1] x [0,dx_2] x ... x [0,dx_DIM].
 *
*/
template <int DIM>
struct StandardRectangle
{
    // DATA MEMBERS ===================================================
    // SIZE AND ORDER
    Real dx[DIM];
    int p, Np;
    int q, dom_Nq, bou_Nq;
    int g, dom_Ng, bou_Ng;

    // QUADRATURE ABSCISSAE AND WEIGHTS
    Gpu::ManagedVector<Real> dom_XI1q;
    Gpu::ManagedVector<Real> dom_XI2q;
    Gpu::ManagedVector<Real> dom_XI3q;
    Gpu::ManagedVector<Real> dom_Wq;

    Array<Gpu::ManagedVector<Real>, DIM> bou_XI1q;
    Array<Gpu::ManagedVector<Real>, DIM> bou_XI2q;
    Array<Gpu::ManagedVector<Real>, DIM> bou_XI3q;
    Array<Gpu::ManagedVector<Real>, DIM> bou_Wq;

    // GRID POINTS
    Gpu::ManagedVector<Real> dom_XI1g;
    Gpu::ManagedVector<Real> dom_XI2g;
    Gpu::ManagedVector<Real> dom_XI3g;
    Array<Gpu::ManagedVector<Real>, DIM> bou_XI1g;
    Array<Gpu::ManagedVector<Real>, DIM> bou_XI2g;
    Array<Gpu::ManagedVector<Real>, DIM> bou_XI3g;

    // MASS MATRIX AND ITS CHOLESKY DECOMPOSITION AND INVERSE
    const char MM_uplo = 'U';
    Gpu::ManagedVector<Real> MM, MMCh, iMM;

    // INTERPOLATION
    IntVect rr;
    Gpu::ManagedVector<Real> IBfTBc;
    Gpu::ManagedVector<Real> I;
    // ================================================================


    // CONSTRUCTOR ====================================================
    StandardRectangle()
    :
    rr{IntVect::TheZeroVector()}
    {}
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief This method evaluates the quadrature data of the standard element.
     * 
     * \param[in] dx_: pointer to memory containing the sizes of the element.
     * \param[in] q_: order of the quadrature.
    */
    void set_quadrature(const Real * dx_, const int q_)
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        Vector<Real> Xq_1d(1+q_);
        Vector<Real> Wq_1d(1+q_);
        // ------------------------------------------------------------

        // CHECKS -----------------------------------------------------
        if (q_ < 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_StandardElements.H - StandardRectangle::set_quadrature\n";
            msg += "| The order of the quadrature must be positive.\n";
            msg += "| Input order q = "+std::to_string(q_)+"\n";
            Abort(msg);   
        }
        // ------------------------------------------------------------

        // ELEMENT SIZE -----------------------------------------------
        for (int d = 0; d < DIM; ++d)
        {
            this->dx[d] = dx_[d];
        }
        // ------------------------------------------------------------

        // ORDER ------------------------------------------------------
        this->q = q_;
        
        if (DIM == 1)
        {
            this->dom_Nq = 1+q_;
            this->bou_Nq = 1;
        }
        else if (DIM == 2)
        {
            this->dom_Nq = (1+q_)*(1+q_);
            this->bou_Nq = 1+q_;
        }
        else if (DIM == 3)
        {
            this->dom_Nq = (1+q_)*(1+q_)*(1+q_);
            this->bou_Nq = (1+q_)*(1+q_);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_quadrature\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // 1D QUADRATURE ABSCISSAE AND WEIGHTS ------------------------
        // NOTE: These are referred to the [0,+1] interval.
        // ------------------------------------------------------------
        for (int q1 = 0; q1 < (1+q_); ++q1)
        {
            Xq_1d[q1] = Algoim::GaussQuad::x(1+q_, q1);
            Wq_1d[q1] = Algoim::GaussQuad::w(1+q_, q1);
        }
        // ------------------------------------------------------------

        // ALLOCATE MEMORY FOR QUADRATURE ABSCISSAE AND WEIGHTS -------
        if (DIM == 1)
        {
            this->dom_XI1q.resize(this->dom_Nq);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1q[dim].resize(this->bou_Nq);
            }
        }
        else if (DIM == 2)
        {
            this->dom_XI1q.resize(this->dom_Nq);
            this->dom_XI2q.resize(this->dom_Nq);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1q[dim].resize(this->bou_Nq);
                this->bou_XI2q[dim].resize(this->bou_Nq);
            }
        }
        else if (DIM == 3)
        {
            this->dom_XI1q.resize(this->dom_Nq);
            this->dom_XI2q.resize(this->dom_Nq);
            this->dom_XI3q.resize(this->dom_Nq);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1q[dim].resize(this->bou_Nq);
                this->bou_XI2q[dim].resize(this->bou_Nq);
                this->bou_XI3q[dim].resize(this->bou_Nq);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_quadrature\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }

        this->dom_Wq.resize(this->dom_Nq);
        for (int dim = 0; dim < DIM; ++dim)
        {
            this->bou_Wq[dim].resize(this->bou_Nq);
        }
        // ------------------------------------------------------------

        // STORE QUADRATURE ABSCISSAE AND WEIGHTS ---------------------
        if (DIM == 1)
        {
            const Real aJ = dx_[0];

            int q1, pos;
            int dir;

            // DOMAIN
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1;
                this->dom_XI1q[pos] = Xq_1d[q1]*dx_[0];
                this->dom_Wq[pos] = Wq_1d[q1]*aJ;
            }

            // BOUNDARY WITH UNIT NORMAL {-1}
            dir = 0;
            this->bou_XI1q[dir][0] = 0.0;
            this->bou_Wq[dir][0] = +1.0;
        }
        else if (DIM == 2)
        {
            const Real aJ = dx_[0]*dx_[1];
            const Real bou_aJ[2] = {dx_[1],
                                    dx_[0]};

            int q1, q2, pos;
            int dir;

            // DOMAIN
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_);
                this->dom_XI1q[pos] = Xq_1d[q1]*dx_[0];
                this->dom_XI2q[pos] = Xq_1d[q2]*dx_[1];
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*aJ;
            }

            // BOUNDARY WITH UNIT NORMAL {-1,0}
            dir = 0;
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2;
                this->bou_XI1q[dir][pos] = 0.0;
                this->bou_XI2q[dir][pos] = Xq_1d[q2]*dx_[1];
                this->bou_Wq[dir][pos] = Wq_1d[q2]*bou_aJ[dir];
            }

            // BOUNDARY WITH UNIT NORMAL {0,-1}
            dir = 1;
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1;
                this->bou_XI1q[dir][pos] = Xq_1d[q1]*dx_[0];
                this->bou_XI2q[dir][pos] = 0.0;
                this->bou_Wq[dir][pos] = Wq_1d[q1]*bou_aJ[dir];
            }
        }
        else if (DIM == 3)
        {
            const Real aJ = dx_[0]*dx_[1]*dx_[2];
            const Real bou_aJ[3] = {dx_[1]*dx_[2],
                                    dx_[0]*dx_[2],
                                    dx_[0]*dx_[1]};

            int q1, q2, q3, pos;
            int dir;

            // DOMAIN
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_)+q3*(1+q_)*(1+q_);
                this->dom_XI1q[pos] = Xq_1d[q1]*dx_[0];
                this->dom_XI2q[pos] = Xq_1d[q2]*dx_[1];
                this->dom_XI3q[pos] = Xq_1d[q3]*dx_[2];
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*Wq_1d[q3]*aJ;
            }

            // BOUNDARY WITH UNIT NORMAL {-1,0,0}
            dir = 0;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2+q3*(1+q_);
                this->bou_XI1q[dir][pos] = 0.0;
                this->bou_XI2q[dir][pos] = Xq_1d[q2]*dx_[1];
                this->bou_XI3q[dir][pos] = Xq_1d[q3]*dx_[2];
                this->bou_Wq[dir][pos] = Wq_1d[q2]*Wq_1d[q3]*bou_aJ[dir];
            }

            // BOUNDARY WITH UNIT NORMAL {0,-1,0}
            dir = 1;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q3*(1+q_);
                this->bou_XI1q[dir][pos] = Xq_1d[q1]*dx_[0];
                this->bou_XI2q[dir][pos] = 0.0;
                this->bou_XI3q[dir][pos] = Xq_1d[q3]*dx_[2];
                this->bou_Wq[dir][pos] = Wq_1d[q1]*Wq_1d[q3]*bou_aJ[dir];
            }

            // BOUNDARY WITH UNIT NORMAL {0,0,-1}
            dir = 2;
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_);
                this->bou_XI1q[dir][pos] = Xq_1d[q1]*dx_[0];
                this->bou_XI2q[dir][pos] = Xq_1d[q2]*dx_[1];
                this->bou_XI3q[dir][pos] = 0.0;
                this->bou_Wq[dir][pos] = Wq_1d[q1]*Wq_1d[q2]*bou_aJ[dir];
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_quadrature\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }

        /*
        Print() << "Xq_1d: " << std::endl;
        io::print_real_array_2d(1, 1+q_, Xq_1d.data());
        Print() << "Wq_1d: " << std::endl;
        io::print_real_array_2d(1, 1+q_, Wq_1d.data());
        Print() << "dx_: " << std::endl;
        io::print_real_array_2d(1, AMREX_SPACEDIM, dx_);
        Print() << "dom_XI1q: " << std::endl;
        io::print_real_array_2d(1, this->dom_Nq, this->dom_XI1q.data());
        Print() << "dom_XI2q: " << std::endl;
        io::print_real_array_2d(1, this->dom_Nq, this->dom_XI2q.data());
        Print() << "dom_Wq: " << std::endl;
        io::print_real_array_2d(1, this->dom_Nq, this->dom_Wq.data());
        exit(-1);
        */
        // ------------------------------------------------------------
    }

    /**
     * \brief This method evaluates the grid data of the standard element.
     * 
     * \param[in] dx_: pointer to memory containing the sizes of the element.
     * \param[in] g_: order of the grid.
    */
    void set_grid(const Real * dx_, const int g_)
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        Vector<Real> Xg_1d(1+g_);
        // ------------------------------------------------------------

        // CHECKS -----------------------------------------------------
        if (g_ < 1)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_StandardElements.H - StandardRectangle::set_grid\n";
            msg += "| The order of the grid must be greater than 0.\n";
            msg += "| Input order g = "+std::to_string(g_)+"\n";
            Abort(msg);   
        }
        // ------------------------------------------------------------

        // ELEMENT SIZE -----------------------------------------------
        for (int d = 0; d < DIM; ++d)
        {
            this->dx[d] = dx_[d];
        }
        // ------------------------------------------------------------

        // ORDER ------------------------------------------------------
        this->g = g_;
        
        if (DIM == 1)
        {
            this->dom_Ng = 1+g_;
            this->bou_Ng = 1;
        }
        else if (DIM == 2)
        {
            this->dom_Ng = (1+g_)*(1+g_);
            this->bou_Ng = 1+g_;
        }
        else if (DIM == 3)
        {
            this->dom_Ng = (1+g_)*(1+g_)*(1+g_);
            this->bou_Ng = (1+g_)*(1+g_);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_grid\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // 1D GRID ABSCISSAE ------------------------------------------
        // NOTE: These are referred to the [0,+1] interval.
        // ------------------------------------------------------------
        const Real * gau_lob_x_ptr;

        if      (g_ == 0) {gau_lob_x_ptr = tables::gau_lob_1_x;}
        else if (g_ == 1) {gau_lob_x_ptr = tables::gau_lob_2_x;}
        else if (g_ == 2) {gau_lob_x_ptr = tables::gau_lob_3_x;}
        else if (g_ == 3) {gau_lob_x_ptr = tables::gau_lob_4_x;}
        else if (g_ == 4) {gau_lob_x_ptr = tables::gau_lob_5_x;}
        else if (g_ == 5) {gau_lob_x_ptr = tables::gau_lob_6_x;}
        else if (g_ == 6) {gau_lob_x_ptr = tables::gau_lob_7_x;}
        else if (g_ == 7) {gau_lob_x_ptr = tables::gau_lob_8_x;}
        else if (g_ == 8) {gau_lob_x_ptr = tables::gau_lob_9_x;}
        else if (g_ == 9) {gau_lob_x_ptr = tables::gau_lob_10_x;}
        else if (g_ == 10) {gau_lob_x_ptr = tables::gau_lob_11_x;}
        else if (g_ == 11) {gau_lob_x_ptr = tables::gau_lob_12_x;}
        else if (g_ == 12) {gau_lob_x_ptr = tables::gau_lob_13_x;}
        else if (g_ == 13) {gau_lob_x_ptr = tables::gau_lob_14_x;}
        else if (g_ == 14) {gau_lob_x_ptr = tables::gau_lob_15_x;}
        else if (g_ == 15) {gau_lob_x_ptr = tables::gau_lob_16_x;}
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_grid\n";
            msg += "| Grid order = "+std::to_string(g_)+" not implemented yet\n";
            Abort(msg);
        }
        
        for (int g1 = 0; g1 < (1+g_); ++g1)
        {
            Xg_1d[g1] = (gau_lob_x_ptr[g1]+1.0)*0.5;
        }
        // ------------------------------------------------------------

        // ALLOCATE MEMORY FOR GRID POINTS ABSCISSAE ------------------
        if (DIM == 1)
        {
            this->dom_XI1g.resize(this->dom_Ng);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1g[dim].resize(this->bou_Ng);
            }
        }
        else if (DIM == 2)
        {
            this->dom_XI1g.resize(this->dom_Ng);
            this->dom_XI2g.resize(this->dom_Ng);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1g[dim].resize(this->bou_Ng);
                this->bou_XI2g[dim].resize(this->bou_Ng);
            }
        }
        else if (DIM == 3)
        {
            this->dom_XI1g.resize(this->dom_Ng);
            this->dom_XI2g.resize(this->dom_Ng);
            this->dom_XI3g.resize(this->dom_Ng);
            for (int dim = 0; dim < DIM; ++dim)
            {
                this->bou_XI1g[dim].resize(this->bou_Ng);
                this->bou_XI2g[dim].resize(this->bou_Ng);
                this->bou_XI3g[dim].resize(this->bou_Ng);
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_grid\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // STORE GRID POINTS ABSCISSAE --------------------------------
        if (DIM == 1)
        {
            int g1, pos;
            int dir;

            // DOMAIN
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1;
                this->dom_XI1g[pos] = Xg_1d[g1]*dx_[0];
            }

            // BOUNDARY WITH UNIT NORMAL {-1}
            dir = 0;
            this->bou_XI1g[dir][0] = 0.0;
        }
        else if (DIM == 2)
        {
            int g1, g2, pos;
            int dir;

            // DOMAIN
            for (g2 = 0; g2 < (1+g_); ++g2)
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1+g2*(1+g_);
                this->dom_XI1g[pos] = Xg_1d[g1]*dx_[0];
                this->dom_XI2g[pos] = Xg_1d[g2]*dx_[1];
            }

            // BOUNDARY WITH UNIT NORMAL {-1,0}
            dir = 0;
            for (g2 = 0; g2 < (1+g_); ++g2)
            {
                pos = g2;
                this->bou_XI1g[dir][pos] = 0.0;
                this->bou_XI2g[dir][pos] = Xg_1d[g2]*dx_[1];
            }

            // BOUNDARY WITH UNIT NORMAL {0,-1}
            dir = 1;
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1;
                this->bou_XI1g[dir][pos] = Xg_1d[g1]*dx_[0];
                this->bou_XI2g[dir][pos] = 0.0;
            }
        }
        else if (DIM == 3)
        {
            int g1, g2, g3, pos;
            int dir;

            // DOMAIN
            for (g3 = 0; g3 < (1+g_); ++g3)
            for (g2 = 0; g2 < (1+g_); ++g2)
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1+g2*(1+g_)+g3*(1+g_)*(1+g_);
                this->dom_XI1g[pos] = Xg_1d[g1]*dx_[0];
                this->dom_XI2g[pos] = Xg_1d[g2]*dx_[1];
                this->dom_XI3g[pos] = Xg_1d[g3]*dx_[2];
            }

            // BOUNDARY WITH UNIT NORMAL {-1,0,0}
            dir = 0;
            for (g3 = 0; g3 < (1+g_); ++g3)
            for (g2 = 0; g2 < (1+g_); ++g2)
            {
                pos = g2+g3*(1+g_);
                this->bou_XI1g[dir][pos] = 0.0;
                this->bou_XI2g[dir][pos] = Xg_1d[g2]*dx_[1];
                this->bou_XI3g[dir][pos] = Xg_1d[g3]*dx_[2];
            }

            // BOUNDARY WITH UNIT NORMAL {0,-1,0}
            dir = 1;
            for (g3 = 0; g3 < (1+g_); ++g3)
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1+g3*(1+g_);
                this->bou_XI1g[dir][pos] = Xg_1d[g1]*dx_[0];
                this->bou_XI2g[dir][pos] = 0.0;
                this->bou_XI3g[dir][pos] = Xg_1d[g3]*dx_[2];
            }

            // BOUNDARY WITH UNIT NORMAL {0,0,-1}
            dir = 2;
            for (g2 = 0; g2 < (1+g_); ++g2)
            for (g1 = 0; g1 < (1+g_); ++g1)
            {
                pos = g1+g2*(1+g_);
                this->bou_XI1g[dir][pos] = Xg_1d[g1]*dx_[0];
                this->bou_XI2g[dir][pos] = Xg_1d[g2]*dx_[1];
                this->bou_XI3g[dir][pos] = 0.0;
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_grid\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }

        /*
        Print() << "Xg_1d: " << std::endl;
        io::print_real_array_2d(1, 1+g_, Xg_1d.data());
        Print() << "dx_: " << std::endl;
        io::print_real_array_2d(1, AMREX_SPACEDIM, dx_);
        Print() << "dom_XI1g: " << std::endl;
        io::print_real_array_2d(1, this->dom_Ng, this->dom_XI1g.data());
        Print() << "dom_XI2g: " << std::endl;
        io::print_real_array_2d(1, this->dom_Ng, this->dom_XI2g.data());
        exit(-1);
        */
        // ------------------------------------------------------------
    }

    /**
     * \brief This method evaluates the mass matrix of the standard element.
     * 
     * \param[in] p_: order of the basis functions.
    */
    void set_mass_matrix(const int p_)
    {
        // PARAMETERS -------------------------------------------------
        const Real zero[3] = {0.0, 0.0, 0.0};
        const int Nq_ = this->dom_Nq;
        const int size_1d = Nq_*(1+p_);
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        Vector<Real> P1(size_1d), P2(size_1d), P3(size_1d);
        
        int Np_, size_d;
        Vector<Real> BF;

        int size_MM;
        // ------------------------------------------------------------

        // CHECKS -----------------------------------------------------
        if (p_ < 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_StandardElements.H - StandardRectangle::set_mass_matrix\n";
            msg += "| The order of the basis functions must be positive.\n";
            msg += "| Input order p = "+std::to_string(p_)+"\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // ORDER ------------------------------------------------------
        if (DIM == 1)
        {
            Np_ = 1+p_;
        }
        else if (DIM == 2)
        {
            Np_ = (1+p_)*(1+p_);
        }
        else if (DIM == 3)
        {
            Np_ = (1+p_)*(1+p_)*(1+p_);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_mass_matrix\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        size_d = Nq_*Np_;
        size_MM = Np_*Np_;

        this->p = p_;
        this->Np = Np_;

        this->MM.resize(size_MM);
        this->MMCh.resize(size_MM);
        this->iMM.resize(size_MM);
        // ------------------------------------------------------------

        // EVAL THE BASIS FUNCTIONS -----------------------------------
        BF.resize(size_d);

        if (DIM == 1)
        {
            DG_space_1d_BF_Table_NxL(Nq_, p_,
                                     this->dom_XI1q.data(), 0.0, this->dx[0],
                                     BF.data());
        }
        else if (DIM == 2)
        {
            DG_space_2d_BF_Table_NxL(Nq_, p_,
                                     this->dom_XI1q.data(), this->dom_XI2q.data(), zero, this->dx,
                                     P1.data(), P2.data(),
                                     BF.data());
        }
        else if (DIM == 3)
        {
            DG_space_3d_BF_Table_NxL(Nq_, p_,
                                     this->dom_XI1q.data(), this->dom_XI2q.data(), this->dom_XI3q.data(), zero, this->dx,
                                     P1.data(), P2.data(), P3.data(),
                                     BF.data());
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_mass_matrix\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        // ------------------------------------------------------------

        // MASS MATRIX AND ITS CHOLESKY DECOMPOSITION AND INVERSE -----
        std::fill(this->MM.begin(), this->MM.end(), 0.0);
        std::fill(this->MMCh.begin(), this->MMCh.end(), 0.0);
        std::fill(this->iMM.begin(), this->iMM.end(), 0.0);

        // MASS MATRIX
        for (int c = 0; c < Np_; ++c)
        for (int r = 0; r < Np_; ++r)
        for (int k = 0; k < Nq_; ++k)
        {
            const int pos = r+c*Np_;
            const int pos_r = k+r*Nq_;
            const int pos_c = k+c*Nq_;

            this->MM[pos] += BF[pos_r]*BF[pos_c]*this->dom_Wq[k];
        }

        // CHOLESKY DECOMPOSITION
        int info;
        std::copy(this->MM.begin(), this->MM.end(), this->MMCh.begin());
        linalg::dpotf2(this->MM_uplo, Np_, this->MMCh.data(), Np_,
                       info);

        if (info != 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElements.H - StandardRectangle::set_mass_matrix\n";
            msg += "| Something went wrong with the evaluation of the Cholesky\n";
            msg += "| decomposition of the mass matrix.\n";
            Abort(msg);
        }

        // INVERSE
        for (int r = 0; r < Np_; ++r)
        {
            this->iMM[r+r*Np_] = 1.0;
        }

        linalg::dpotrs(this->MM_uplo, Np_, Np_, this->MMCh.data(), Np_,
                       this->iMM.data(), Np_,
                       info);

        if (info != 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::set_mass_matrix\n";
            msg += "| Something went wrong with the evaluation of the inverse of the mass\n";
            msg += "| matrix.\n";
            Abort(msg);
        }

        /*
        Print() << "dom_Wq: " << std::endl;
        io::print_reals(Nq_, this->dom_Wq.data());
        Print() << "MM: " << std::endl;
        io::print_real_array_2d(Np_, Np_, this->MM.data());
        Print() << "MMCh: " << std::endl;
        io::print_real_array_2d(Np_, Np_, this->MMCh.data());
        Print() << "iMM: " << std::endl;
        io::print_real_array_2d(Np_, Np_, this->iMM.data());
        exit(-1);
        */
        // ------------------------------------------------------------
    }
    // ================================================================
};
// ####################################################################

} // namespace dG
} // namespace amrex

#endif