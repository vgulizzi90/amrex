// AMReX_DG_MatrixFactory.H

#ifndef BL_DG_MATRIX_FACTORY_H_
#define BL_DG_MATRIX_FACTORY_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>

namespace amrex
{
namespace DG
{

// ####################################################################
// DG MATRIX FACTORY CLASS ############################################
// ####################################################################
template<int N_PHI, int N_DOM>
struct MatrixFactory
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;
    static const int bou_data_ngr = 0;
    
    static const int Dom_BF_pos_n_comp = DG_DOM_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int Tri_BF_pos_n_comp = DG_TRI_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int IntBou_BF_pos_n_comp = DG_INT_BOU_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int Bou_BF_pos_n_comp = DG_BOU_BF_INFO_N_COMP_PER_DOM*N_DOM;
    static const int MMCh_pos_n_comp = N_DOM;
    static const int Dom_I_pos_n_comp = DG_DOM_INTERPOLATION_N_COMP_PER_DOM*N_DOM;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    UnitCube<AMREX_SPACEDIM> cube;

    // IMPLICIT MESH
    longMultiFab space_MMCh_pos;
    longMultiFab space_Dom_I_pos;

    long mem_for_space_MMCh;
    long mem_for_space_Dom_I;

    Gpu::HostVector<Real> space_MMCh_host_mem;
    Gpu::HostVector<Real> space_Dom_I_host_mem;

    Gpu::DeviceVector<Real> space_MMCh_device_mem;
    Gpu::DeviceVector<Real> space_Dom_I_device_mem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    MatrixFactory()
    {}
    // ################################################################

    // DESTRUCTOR #####################################################
    ~MatrixFactory()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void DefineUnitCube(const int space_p, const int space_q,
                        const int time_p)
    {
        this->cube.Define(space_p, space_q);

        // SOME CHECKS
        BL_ASSERT(space_p >= 0);
        BL_ASSERT(space_p <= DG_SOL_MAX_SPACE_p);

        if ((space_p != DG_SOL_MAX_SPACE_p) && (false))
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::DefineUnitCube" << std::endl;
            Print() << "| DG_SOL_MAX_SPACE_p = " << DG_SOL_MAX_SPACE_p << std::endl;
            Print() << "| input space p = " << space_p << std::endl;
            Print() << "| Consider setting DG_SOL_MAX_SPACE_p = " << space_p << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }

        BL_ASSERT(time_p >= 0);
        BL_ASSERT(time_p <= DG_SOL_MAX_TIME_p);

        if ((time_p != DG_SOL_MAX_TIME_p) && (false))
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::DefineUnitCube" << std::endl;
            Print() << "| DG_SOL_MAX_TIME_p = " << DG_SOL_MAX_TIME_p << std::endl;
            Print() << "| input time p = " << time_p << std::endl;
            Print() << "| Consider setting DG_SOL_MAX_TIME_p = " << time_p << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }
    }

    void DefineImplicitMeshInfo(const BoxArray & cc_ba, const DistributionMapping & dm)
    {
        // ============================================================
        // WARNING: Remember that on the MultiFabs
        //          - this->space_MMCh_pos
        //          - this->space_Dom_I_pos
        //          the FillBoundary method must NEVER be called.
        // ============================================================
        // DOMAIN (CELL-CENTERED) INFO --------------------------------
        {
            this->space_MMCh_pos.define(cc_ba, dm, this->MMCh_pos_n_comp, this->dom_data_ngr);
            this->space_MMCh_pos = -1L;

            this->space_Dom_I_pos.define(cc_ba, dm, this->Dom_I_pos_n_comp, this->dom_data_ngr);
            this->space_Dom_I_pos = -1L;
        }
        // ------------------------------------------------------------

        this->mem_for_space_MMCh = 0L;
        this->mem_for_space_Dom_I = 0L;
        // ============================================================
    }
    
    void Define(const BoxArray & cc_ba, const DistributionMapping & dm,
                const InputReader & inputs, const int d)
    {
        // UNIT CUBE [-1,1]^AMREX_SPACEDIM ============================
        this->DefineUnitCube(inputs.dG[d].space_p, inputs.dG[d].space_q,
                             inputs.dG[d].time_p);
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->DefineImplicitMeshInfo(cc_ba, dm);
        // ============================================================
    }
    // ################################################################

    // INITIALIZATION FOR A SLOPE LIMITER #############################
    void DefineForSlopeLimiter(const MatrixFactory<N_PHI, N_DOM> & coarse,
                               const IntVect & rr,
                               const BoxArray & cc_ba, const DistributionMapping & dm,
                               const InputReader & inputs, const int d)
    {
        // UNIT CUBE [-1,1]^AMREX_SPACEDIM ============================
        // We are assuming that our solution will be constant over each
        // element. Thus p = 0.
        // ============================================================
        const int space_p = 0;
        const int time_p = 0;

        this->DefineUnitCube(space_p, inputs.dG[d].space_q, time_p);

        // WE ALSO NEED THE INTERPOLATION OPERATOR
        this->cube.InitInterpolation(coarse.cube, rr);
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->DefineImplicitMeshInfo(cc_ba, dm);
        // ============================================================
    }
    // ################################################################

};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif