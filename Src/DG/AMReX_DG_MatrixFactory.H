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
    
    static const int MMCh_pos_n_comp = N_DOM;
    static const int Avg_pos_n_comp = N_DOM;
    static const int Dom_I_pos_n_comp = DG_DOM_INTERPOLATION_N_COMP_PER_DOM*N_DOM;
    static const int SLOpe_pos_n_comp = N_DOM;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    UnitCube<AMREX_SPACEDIM> cube;

    // IMPLICIT MESH
    longMultiFab space_MMCh_pos;
    longMultiFab space_Avg_pos;
    longMultiFab space_Dom_I_pos;
    longMultiFab space_SLOpe_pos;

    long mem_for_space_MMCh;
    long mem_for_space_Avg;
    long mem_for_space_Dom_I;
    long mem_for_space_SLOpe;

    Gpu::HostVector<Real> space_MMCh_host_mem;
    Gpu::HostVector<Real> space_Avg_host_mem;
    Gpu::HostVector<Real> space_Dom_I_host_mem;
    Gpu::HostVector<Real> space_SLOpe_host_mem;

    Gpu::DeviceVector<Real> space_MMCh_device_mem;
    Gpu::DeviceVector<Real> space_Avg_device_mem;
    Gpu::DeviceVector<Real> space_Dom_I_device_mem;
    Gpu::DeviceVector<Real> space_SLOpe_device_mem;
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
    void InitUnitCube(const int space_p, const int space_q,
                      const int time_p)
    {
        this->cube.Init(space_p, space_q);

        // SOME CHECKS
        BL_ASSERT(space_p >= 0);
        BL_ASSERT(space_p <= DG_SOL_MAX_SPACE_p);

        if ((space_p != DG_SOL_MAX_SPACE_p) && (false))
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::InitUnitCube" << std::endl;
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
            Print() << "WARNING: AMReX_DG_MatrixFactory.H - MatrixFactory::InitUnitCube" << std::endl;
            Print() << "| DG_SOL_MAX_TIME_p = " << DG_SOL_MAX_TIME_p << std::endl;
            Print() << "| input time p = " << time_p << std::endl;
            Print() << "| Consider setting DG_SOL_MAX_TIME_p = " << time_p << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }
    }

    void Init(const InputReader & inputs, const int d)
    {
        // UNIT CUBE [-1,1]^AMREX_SPACEDIM ============================
        this->InitUnitCube(inputs.dG[d].space_p, inputs.dG[d].space_q,
                           inputs.dG[d].time_p);
        // ============================================================
    }

    void DefineImplicitMeshInfo(const BoxArray & cc_ba, const DistributionMapping & dm)
    {
        // ============================================================
        // WARNING: Remember that on the MultiFabs
        //          - this->space_MMCh_pos
        //          - this->space_Avg_pos
        //          - this->space_Dom_I_pos
        //          the FillBoundary method must NEVER be called.
        // ============================================================
        // DOMAIN (CELL-CENTERED) INFO --------------------------------
        {
            this->space_MMCh_pos.define(cc_ba, dm, this->MMCh_pos_n_comp, this->dom_data_ngr);
            this->space_MMCh_pos = -1L;

            this->space_Avg_pos.define(cc_ba, dm, this->Avg_pos_n_comp, this->dom_data_ngr);
            this->space_Avg_pos = -1L;

            this->space_Dom_I_pos.define(cc_ba, dm, this->Dom_I_pos_n_comp, this->dom_data_ngr);
            this->space_Dom_I_pos = -1L;

            this->space_SLOpe_pos.define(cc_ba, dm, this->SLOpe_pos_n_comp, this->dom_data_ngr);
            this->space_SLOpe_pos = -1L;
        }
        // ------------------------------------------------------------

        this->mem_for_space_MMCh = 0L;
        this->mem_for_space_Avg = 0L;
        this->mem_for_space_Dom_I = 0L;
        this->mem_for_space_SLOpe = 0L;
        // ============================================================
    }
    
    void Define(const BoxArray & cc_ba, const DistributionMapping & dm)
    {
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