//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_StandardElement.H
 * \brief Constants and data structures of standard elements for discontinuous Galerkin methods.
*/

#include <AMReX_GpuContainers.H>

#include <AMReX_DG_IO.H>
#include <AMReX_DG_Quadrature.H>
#include <AMReX_DG_BasisFunctions.H>
#include <AMReX_DG_LinAlg.H>

#ifndef AMREX_DG_STANDARD_ELEMENTS_H_
#define AMREX_DG_STANDARD_ELEMENTS_H_

namespace amrex
{
namespace DG
{

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
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int n_bou = 2*AMREX_SPACEDIM;

    // VARIABLES
    Real dx[DIM];
    int p, Np;
    int q, dom_Nq, bou_Nq;
    int g, dom_Ng, bou_Ng;

    // QUADRATURE ABSCISSAE AND WEIGHTS
    Gpu::ManagedVector<Real> dom_XI1q;
    Gpu::ManagedVector<Real> dom_XI2q;
    Gpu::ManagedVector<Real> dom_XI3q;
    Gpu::ManagedVector<Real> dom_Wq;

    Gpu::ManagedVector<Real> bou_XI1q;
    Gpu::ManagedVector<Real> bou_XI2q;
    Gpu::ManagedVector<Real> bou_XI3q;
    Gpu::ManagedVector<Real> bou_Wq;

    // BASIS FUNCTIONS
    Gpu::ManagedVector<Real> dom_BF;
    Gpu::ManagedVector<Real> dom_dBFdXI1;
    Gpu::ManagedVector<Real> dom_dBFdXI2;
    Gpu::ManagedVector<Real> dom_dBFdXI3;
    
    Gpu::ManagedVector<Real> bou_BF;
    Gpu::ManagedVector<Real> bou_dBFdXI1;
    Gpu::ManagedVector<Real> bou_dBFdXI2;
    Gpu::ManagedVector<Real> bou_dBFdXI3;

    // MASS MATRIX AND ITS CHOLESKY DECOMPOSITION AND INVERSE
    const char MM_uplo = 'U';
    Gpu::ManagedVector<Real> MM, MMCh, iMM;

    // GRID POINTS
    Gpu::ManagedVector<Real> dom_XI1g;
    Gpu::ManagedVector<Real> dom_XI2g;
    Gpu::ManagedVector<Real> dom_XI3g;
    Gpu::ManagedVector<Real> bou_XI1g;
    Gpu::ManagedVector<Real> bou_XI2g;
    Gpu::ManagedVector<Real> bou_XI3g;

    // SLOpe OPERATOR
    Gpu::ManagedVector<Real> SLOpe;
    // ################################################################

    // CONSTRUCTOR ####################################################
    StandardRectangle()
    {}

    StandardRectangle(const Real * dx_, const int p_, const int q_)
    {
        this->define(dx_, p_, q_);
    }
    // ################################################################

    // INITIALIZATION #################################################
    /**
     * \brief This method initializes the basic data members of the standard element.
     * 
     * \param[in] dx_: pointer to memory containing the sizes of the element.
     * \param[in] p_: order of the basis functions.
     * \param[in] q_: order of the quadrature.
    */
    void define(const Real * dx_, const int p_, const int q_)
    {
        // PARAMETERS =================================================
        const int g_ = std::max(1, q_);
        // ============================================================

        // QUICK CHECKS ===============================================
        if (p_ < 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
            msg += "| The order of the basis functions must be positive.\n";
            msg += "| Input order p = "+std::to_string(p_)+"\n";
            Abort(msg);
        }

        if (q_ < 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
            msg += "| The order of the quadrature must be positive.\n";
            msg += "| Input order q = "+std::to_string(q_)+"\n";
            Abort(msg);   
        }
        // ============================================================

        // ELEMENT SIZE ===============================================
        for (int d = 0; d < DIM; ++d)
        {
            this->dx[d] = dx_[d];
        }
        // ============================================================

        // ORDER ======================================================
        this->p = p_;
        this->Np = AMREX_D_PICK(1+p_, (1+p_)*(1+p_), (1+p_)*(1+p_)*(1+p_));
        this->q = q_;
        this->dom_Nq = AMREX_D_PICK(1+q_, (1+q_)*(1+q_), (1+q_)*(1+q_)*(1+q_));
        this->bou_Nq = AMREX_D_PICK(1, 1+q_, (1+q_)*(1+q_));
        this->g = g_;
        this->dom_Ng = AMREX_D_PICK(1+g_, (1+g_)*(1+g_), (1+g_)*(1+g_)*(1+g_));
        this->bou_Ng = AMREX_D_PICK(1, 1+g_, (1+g_)*(1+g_));
        // ============================================================

        // POINTERS TO 1D QUADRATURE ABSCISSAE AND WEIGHTS ============
        // NOTE: Remember that these are referred to the [0,+1]
        //       interval.
        // ============================================================
        Vector<Real> Xq_1d;
        Vector<Real> Wq_1d;

        Xq_1d.resize(1+q_);
        Wq_1d.resize(1+q_);

        for (int q1 = 0; q1 < (1+q_); ++q1)
        {
            Xq_1d[q1] = Algoim::GaussQuad::x(1+q_,q1);
            Wq_1d[q1] = Algoim::GaussQuad::w(1+q_,q1);
        }
        // ============================================================

        // ALLOCATE MEMORY FOR QUADRATURE ABSCISSAE AND WEIGHTS =======
        // DOMAIN
        if (DIM > 0) this->dom_XI1q.resize(this->dom_Nq);
        if (DIM > 1) this->dom_XI2q.resize(this->dom_Nq);
        if (DIM > 2) this->dom_XI3q.resize(this->dom_Nq);
        this->dom_Wq.resize(this->dom_Nq);

        // BOUNDARIES
        if (DIM > 0) this->bou_XI1q.resize((this->bou_Nq)*(this->n_bou));
        if (DIM > 1) this->bou_XI2q.resize((this->bou_Nq)*(this->n_bou));
        if (DIM > 2) this->bou_XI3q.resize((this->bou_Nq)*(this->n_bou));
        this->bou_Wq.resize((this->bou_Nq)*(this->n_bou));
        // ============================================================

        // STORE QUADRATURE ABSCISSAE AND WEIGHTS =====================
        // NOTE: Remember that we are storing the quadrature abscissae
        //       and weights as referred to the interval 
        //       [0,dx_1] x [0,dx_2] x ... x [0,dx_DIM].
        // ============================================================
        if (DIM == 1)
        {
            const Real aJ = (this->dx[0]);

            int q1, pos;
            int b;

            // DOMAIN
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1;
                this->dom_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->dom_Wq[pos] = Wq_1d[q1]*aJ;
            }

            // BOUNDARIES
            // Boundary with unit normal {-1}
            b = 0;
            this->bou_XI1q[b] = 0.0;
            this->bou_Wq[b] = +1.0;

            // Boundary with unit normal {+1}
            b = 1;
            this->bou_XI1q[b] = this->dx[0];
            this->bou_Wq[b] = +1.0;
        }
        else if (DIM == 2)
        {
            const Real aJ = (this->dx[0]*this->dx[1]);
            const Real bou_aJ[2] = {(this->dx[1]),
                                    (this->dx[0])};

            int q1, q2, pos;
            int b;

            // DOMAIN
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_);
                this->dom_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->dom_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*aJ;
            }

            // BOUNDARIES
            // Boundary with normal {-1,0}
            b = 0;
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2+b*(this->bou_Nq);
                this->bou_XI1q[pos] = 0.0;
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_Wq[pos] = Wq_1d[q2]*bou_aJ[b/2];
            }

            // Boundary with normal {+1,0}
            b = 1;
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2+b*(this->bou_Nq);
                this->bou_XI1q[pos] = this->dx[0];
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_Wq[pos] = Wq_1d[q2]*bou_aJ[b/2];
            }

            // Boundary with normal {0,-1}
            b = 2;
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = 0.0;
                this->bou_Wq[pos] = Wq_1d[q1]*bou_aJ[b/2];
            }

            // Boundary with normal {0,+1}
            b = 3;
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = this->dx[1];
                this->bou_Wq[pos] = Wq_1d[q1]*bou_aJ[b/2];
            }
        }
        else if (DIM == 3)
        {
            const Real aJ = (this->dx[0]*this->dx[1]*this->dx[2]);
            const Real bou_aJ[3] = {(this->dx[1]*this->dx[2]),
                                    (this->dx[0]*this->dx[2]),
                                    (this->dx[0]*this->dx[1])};
            
            int q1, q2, q3, pos;
            int b;

            // DOMAIN
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_)+q3*(1+q_)*(1+q_);
                this->dom_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->dom_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->dom_XI3q[pos] = Xq_1d[q3]*(this->dx[2]);
                this->dom_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*Wq_1d[q3]*aJ;
            }

            // BOUNDARIES
            // Boundary with normal {-1,0,0}
            b = 0;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2+q3*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = 0.0;
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_XI3q[pos] = Xq_1d[q3]*(this->dx[2]);
                this->bou_Wq[pos] = Wq_1d[q2]*Wq_1d[q3]*bou_aJ[b/2];
            }

            // Boundary with normal {+1,0,0}
            b = 1;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q2 = 0; q2 < (1+q_); ++q2)
            {
                pos = q2+q3*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = this->dx[0];
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_XI3q[pos] = Xq_1d[q3]*(this->dx[2]);
                this->bou_Wq[pos] = Wq_1d[q2]*Wq_1d[q3]*bou_aJ[b/2];
            }

            // Boundary with normal {0,-1,0}
            b = 2;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q3*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = 0.0;
                this->bou_XI3q[pos] = Xq_1d[q3]*(this->dx[2]);
                this->bou_Wq[pos] = Wq_1d[q1]*Wq_1d[q3]*bou_aJ[b/2];
            }

            // Boundary with normal {0,+1,0}
            b = 3;
            for (q3 = 0; q3 < (1+q_); ++q3)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q3*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = this->dx[1];
                this->bou_XI3q[pos] = Xq_1d[q3]*(this->dx[2]);
                this->bou_Wq[pos] = Wq_1d[q1]*Wq_1d[q3]*bou_aJ[b/2];
            }

            // Boundary with normal {0,0,-1}
            b = 4;
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_XI3q[pos] = 0.0;
                this->bou_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*bou_aJ[b/2];
            }

            // Boundary with normal {0,0,+1}
            b = 5;
            for (q2 = 0; q2 < (1+q_); ++q2)
            for (q1 = 0; q1 < (1+q_); ++q1)
            {
                pos = q1+q2*(1+q_)+b*(this->bou_Nq);
                this->bou_XI1q[pos] = Xq_1d[q1]*(this->dx[0]);
                this->bou_XI2q[pos] = Xq_1d[q2]*(this->dx[1]);
                this->bou_XI3q[pos] = this->dx[2];
                this->bou_Wq[pos] = Wq_1d[q1]*Wq_1d[q2]*bou_aJ[b/2];
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
            msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
            Abort(msg);
        }
        // ============================================================

        // ALLOCATE MEMORY FOR BASIS FUNCTIONS ========================
        // DOMAIN
        {
            int size = (this->dom_Nq)*(this->Np);

            this->dom_BF.resize(size);
            if (DIM > 0) this->dom_dBFdXI1.resize(size);
            if (DIM > 1) this->dom_dBFdXI2.resize(size);
            if (DIM > 2) this->dom_dBFdXI3.resize(size);
        }

        // BOUNDARIES
        {
            int size = (this->bou_Nq)*(this->Np)*(this->n_bou);

            this->bou_BF.resize(size);
            if (DIM > 0) this->bou_dBFdXI1.resize(size);
            if (DIM > 1) this->bou_dBFdXI2.resize(size);
            if (DIM > 2) this->bou_dBFdXI3.resize(size);
        }
        // ============================================================

        // STORE THE BASIS FUNCTIONS ==================================
        {
            const Real zero[3] = {0.0, 0.0, 0.0};
            const int size_1d = (this->dom_Nq)*(1+p_);

            Vector<Real> P1(size_1d), P2(size_1d), P3(size_1d);
            Vector<Real> dP1(size_1d), dP2(size_1d), dP3(size_1d);
            int pos, pos2;

            if (DIM == 1)
            {
                // DOMAIN
                DG_space_1d_BF_Table_NxL(this->dom_Nq, p_,
                                         this->dom_XI1q.data(), 0.0, this->dx[0],
                                         this->dom_BF.data(), this->dom_dBFdXI1.data());

                // BOUNDARIES
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_1d_BF_Table_NxL(this->bou_Nq, p_,
                                             &this->bou_XI1q[pos], 0.0, this->dx[0],
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2]);
                }
            }
            else if (DIM == 2)
            {
                // DOMAIN
                DG_space_2d_BF_Table_NxL(this->dom_Nq, p_,
                                         this->dom_XI1q.data(), this->dom_XI2q.data(), zero, this->dx,
                                         P1.data(), dP1.data(), P2.data(), dP2.data(),
                                         this->dom_BF.data(), this->dom_dBFdXI1.data(), this->dom_dBFdXI2.data());

                // BOUNDARIES
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_2d_BF_Table_NxL(this->bou_Nq, p_,
                                             &this->bou_XI1q[pos], &this->bou_XI2q[pos], zero, this->dx,
                                             P1.data(), dP1.data(), P2.data(), dP2.data(),
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2], &this->bou_dBFdXI2[pos2]);
                }
                
            }
            else if (DIM == 3)
            {
                // DOMAIN
                DG_space_3d_BF_Table_NxL(this->dom_Nq, p_,
                                         this->dom_XI1q.data(), this->dom_XI2q.data(), this->dom_XI3q.data(), zero, this->dx,
                                         P1.data(), dP1.data(), P2.data(), dP2.data(), P3.data(), dP3.data(),
                                         this->dom_BF.data(), this->dom_dBFdXI1.data(), this->dom_dBFdXI2.data(), this->dom_dBFdXI3.data());

                // BOUNDARIES
                for (int b = 0; b < (this->n_bou); ++b)
                {
                    pos = b*(this->bou_Nq);
                    pos2 = b*(this->bou_Nq)*(this->Np);

                    DG_space_3d_BF_Table_NxL(this->bou_Nq, p_,
                                             &this->bou_XI1q[pos], &this->bou_XI2q[pos], &this->bou_XI3q[pos], zero, this->dx,
                                             P1.data(), dP1.data(), P2.data(), dP2.data(), P3.data(), dP3.data(),
                                             &this->bou_BF[pos2], &this->bou_dBFdXI1[pos2], &this->bou_dBFdXI2[pos2], &this->bou_dBFdXI3[pos2]);
                }
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
                msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
                Abort(msg);
            }
        }
        // ============================================================

        // MASS MATRIX AND ITS CHOLESKY DECOMPOSITION AND INVERSE =====
        {
            int size = (this->Np)*(this->Np);
            int pos, posr, posc;

            this->MM.resize(size);
            this->MMCh.resize(size);
            this->iMM.resize(size);

            std::fill(this->MM.begin(), this->MM.end(), 0.0);
            std::fill(this->MMCh.begin(), this->MMCh.end(), 0.0);
            std::fill(this->iMM.begin(), this->iMM.end(), 0.0);

            // MASS MATRIX
            for (int c = 0; c < (this->Np); ++c)
            for (int r = 0; r < (this->Np); ++r)
            for (int k = 0; k < (this->dom_Nq); ++k)
            {
                pos = r+c*(this->Np);
                posr = k+r*(this->dom_Nq);
                posc = k+c*(this->dom_Nq);

                this->MM[pos] += this->dom_BF[posr]*this->dom_BF[posc]*this->dom_Wq[k];
            }

            // CHOLESKY DECOMPOSITION
            int info;
            std::copy(this->MM.begin(), this->MM.end(), this->MMCh.begin());
            linalg::dpotf2(this->MM_uplo, this->Np, this->MMCh.data(), this->Np, info);

            if (info != 0)
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
                msg += "| Something went wrong with the evaluation of the Cholesky\n";
                msg += "| decomposition of the mass matrix.\n";
                Abort(msg);
            }

            // INVERSE
            for (int r = 0; r < (this->Np); ++r)
            {
                this->iMM[r+r*(this->Np)] = 1.0;
            }

            linalg::dpotrs(this->MM_uplo, this->Np, this->Np, this->MMCh.data(), this->Np,
                           this->iMM.data(), this->Np,
                           info);

            if (info != 0)
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
                msg += "| Something went wrong with the evaluation of the inverse of the mass\n";
                msg += "| matrix.\n";
                Abort(msg);
            }

            /*
            Print() << "MM: " << std::endl;
            IO::PrintRealArray2D(this->Np, this->Np, this->MM.data());
            Print() << "MMCh: " << std::endl;
            IO::PrintRealArray2D(this->Np, this->Np, this->MMCh.data());
            Print() << "iMM: " << std::endl;
            IO::PrintRealArray2D(this->Np, this->Np, this->iMM.data());
            exit(-1);
            */
        }
        // ============================================================

        // ALLOCATE MEMORY FOR GRID POINTS ============================
        // DOMAIN
        if (DIM > 0) this->dom_XI1g.resize(this->dom_Ng);
        if (DIM > 1) this->dom_XI2g.resize(this->dom_Ng);
        if (DIM > 2) this->dom_XI3g.resize(this->dom_Ng);

        // BOUNDARIES
        if (DIM > 0) this->bou_XI1g.resize((this->bou_Ng)*(this->n_bou));
        if (DIM > 1) this->bou_XI2g.resize((this->bou_Ng)*(this->n_bou));
        if (DIM > 2) this->bou_XI3g.resize((this->bou_Ng)*(this->n_bou));
        // ============================================================

        // POINTERS TO 1D GRID POINTS =================================
        // NOTE: Remember that these are referred to the [-1,+1]
        //       interval.
        // ============================================================
        const Real * Xg_1d;

        if      (g_ == 0) {Xg_1d = gau_lob_1_x;}
        else if (g_ == 1) {Xg_1d = gau_lob_2_x;}
        else if (g_ == 2) {Xg_1d = gau_lob_3_x;}
        else if (g_ == 3) {Xg_1d = gau_lob_4_x;}
        else if (g_ == 4) {Xg_1d = gau_lob_5_x;}
        else if (g_ == 5) {Xg_1d = gau_lob_6_x;}
        else if (g_ == 6) {Xg_1d = gau_lob_7_x;}
        else if (g_ == 7) {Xg_1d = gau_lob_8_x;}
        else if (g_ == 8) {Xg_1d = gau_lob_9_x;}
        else if (g_ == 9) {Xg_1d = gau_lob_10_x;}
        else if (g_ == 10) {Xg_1d = gau_lob_11_x;}
        else if (g_ == 11) {Xg_1d = gau_lob_12_x;}
        else if (g_ == 12) {Xg_1d = gau_lob_13_x;}
        else if (g_ == 13) {Xg_1d = gau_lob_14_x;}
        else if (g_ == 14) {Xg_1d = gau_lob_15_x;}
        else if (g_ == 15) {Xg_1d = gau_lob_16_x;}
        else
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
            msg += "| Grid order = "+std::to_string(g_)+" not implemented yet\n";
            Abort(msg);
        }
        // ============================================================

        // STORE GRID POINTS ==========================================
        {
            if (DIM == 1)
            {
                int g1, pos;
                int b;

                // DOMAIN
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1;
                    this->dom_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                }

                // BOUNDARIES
                // Boundary with unit normal {-1}
                b = 0;
                this->bou_XI1g[b] = 0.0;

                // Boundary with unit normal {+1}
                b = 1;
                this->bou_XI1g[b] = this->dx[0];

            }
            else if (DIM == 2)
            {
                int g1, g2, pos;
                int b;

                // DOMAIN
                for (g2 = 0; g2 < (1+g_); ++g2)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g2*(1+g_);
                    this->dom_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->dom_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                }

                // BOUNDARIES
                // Boundary with normal {-1,0}
                b = 0;
                for (g2 = 0; g2 < (1+g_); ++g2)
                {
                    pos = g2+b*(this->bou_Nq);
                    this->bou_XI1g[pos] = 0.0;
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                }

                // Boundary with normal {+1,0}
                b = 1;
                for (g2 = 0; g2 < (1+g_); ++g2)
                {
                    pos = g2+b*(this->bou_Nq);
                    this->bou_XI1g[pos] = this->dx[0];
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                }

                // Boundary with normal {0,-1}
                b = 2;
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+b*(this->bou_Nq);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = 0.0;
                }

                // Boundary with normal {0,+1}
                b = 3;
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+b*(this->bou_Nq);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = this->dx[1];
                }
                
            }
            else if (DIM == 3)
            {
                int g1, g2, g3, pos;
                int b;

                // DOMAIN
                for (g3 = 0; g3 < (1+g_); ++g3)
                for (g2 = 0; g2 < (1+g_); ++g2)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g2*(1+g_)+g3*(1+g_)*(1+g_);
                    this->dom_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->dom_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                    this->dom_XI3g[pos] = (Xg_1d[g3]+1.0)*0.5*(this->dx[2]);
                }

                // BOUNDARIES
                // Boundary with normal {-1,0,0}
                b = 0;
                for (g3 = 0; g3 < (1+g_); ++g3)
                for (g2 = 0; g2 < (1+g_); ++g2)
                {
                    pos = g2+g3*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = 0.0;
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                    this->bou_XI3g[pos] = (Xg_1d[g3]+1.0)*0.5*(this->dx[2]);
                }

                // Boundary with normal {+1,0,0}
                b = 1;
                for (g3 = 0; g3 < (1+g_); ++g3)
                for (g2 = 0; g2 < (1+g_); ++g2)
                {
                    pos = g2+g3*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = this->dx[0];
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                    this->bou_XI3g[pos] = (Xg_1d[g3]+1.0)*0.5*(this->dx[2]);
                }

                // Boundary with normal {0,-1,0}
                b = 2;
                for (g3 = 0; g3 < (1+g_); ++g3)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g3*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = 0.0;
                    this->bou_XI3g[pos] = (Xg_1d[g3]+1.0)*0.5*(this->dx[2]);
                }

                // Boundary with normal {0,+1,0}
                b = 3;
                for (g3 = 0; g3 < (1+g_); ++g3)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g3*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = this->dx[1];
                    this->bou_XI3g[pos] = (Xg_1d[g3]+1.0)*0.5*(this->dx[2]);
                }

                // Boundary with normal {0,0,-1}
                b = 4;
                for (g2 = 0; g2 < (1+g_); ++g2)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g2*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                    this->bou_XI3g[pos] = 0.0;
                }

                // Boundary with normal {0,0,+1}
                b = 5;
                for (g2 = 0; g2 < (1+g_); ++g2)
                for (g1 = 0; g1 < (1+g_); ++g1)
                {
                    pos = g1+g2*(1+g_)+b*(this->bou_Ng);
                    this->bou_XI1g[pos] = (Xg_1d[g1]+1.0)*0.5*(this->dx[0]);
                    this->bou_XI2g[pos] = (Xg_1d[g2]+1.0)*0.5*(this->dx[1]);
                    this->bou_XI3g[pos] = this->dx[2];
                }
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::define\n";
                msg += "| Dimension = "+std::to_string(DIM)+" not implemented yet\n";
                Abort(msg);
            }
        }
        // ============================================================
    }

    /**
     * \brief This method initializes all the data members of the 0th-order standard element.
     *
     * The method calls the define(dx_, p_, q_) method with p_ = 0 and q_ = 0.
     * 
     * \param[in] dx_: pointer to memory containing the sizes of the element.
    */
    void define(const Real * dx_)
    {
        const int p_ = 0;
        const int q_ = 0;
        this->define(dx_, p_, q_);
    }
    // ################################################################

    // AUXILIARY SLOPE OPERATOR #######################################
    /**
     * \brief This method initializes the SLOpe operator.
     *
     * The SLOpe operator allows to express a function of the form:
     *
     * u(x) = u(x0)+grad_u dot (x-x0)
     *
     * in terms of the dG basis functions.
     *
    */
    void InitSLOpe()
    {
        // PARAMETERS =================================================
        const int size = (this->Np)*(1+AMREX_SPACEDIM);
        // ============================================================
        
        // VARIABLES ==================================================
        int info;
        // ============================================================

        // ALLOCATE MEMORY ============================================
        this->SLOpe.resize(size);
        std::fill(this->SLOpe.begin(), this->SLOpe.end(), 0.0);
        // ============================================================

        // EVAL THE AUXILIARY SLOPE LIMITER OPERATOR ==================
        for (int r = 0; r < (this->Np); ++r)
        for (int k = 0; k < (this->dom_Nq); ++k)
        {
            const int posr = k+r*(this->dom_Nq);

            // Average
            this->SLOpe[r] += this->dom_BF[posr]*this->dom_Wq[k];

            AMREX_D_TERM
            (
                // d()/dx1
                this->SLOpe[r+1*(this->Np)] += this->dom_BF[posr]*(this->dom_XI1q[k]-0.5*this->dx[0])*this->dom_Wq[k];,
                // d()/dx2
                this->SLOpe[r+2*(this->Np)] += this->dom_BF[posr]*(this->dom_XI2q[k]-0.5*this->dx[1])*this->dom_Wq[k];,
                // d()/dx3
                this->SLOpe[r+3*(this->Np)] += this->dom_BF[posr]*(this->dom_XI3q[k]-0.5*this->dx[2])*this->dom_Wq[k];
            )
        }

        // MULTIPLY BY THE INVERSE MASS MATRIX
        linalg::dpotrs(this->MM_uplo, this->Np, 1+AMREX_SPACEDIM,
                       this->MMCh.data(), this->Np,
                       this->SLOpe.data(), this->Np,
                       info);

        if (info != 0)
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_DG_StandardElement.H - StandardRectangle::InitSLOpe\n";
            msg += "| Something went wrong with the evaluation of the inverse of the mass\n";
            msg += "| matrix.\n";
            Abort(msg);
        }

        /*
        Print() << "SLOpe: " << std::endl;
        IO::PrintRealArray2D(this->Np, 1+AMREX_SPACEDIM, this->SLOpe.data());
        */
        // ============================================================
    }
    // ################################################################
};

} // namespace DG
} // namespace amrex

#endif