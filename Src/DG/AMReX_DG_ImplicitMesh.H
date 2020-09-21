// AMReX_DG_ImplicitMesh.H

#ifndef BL_DG_IMPLICIT_MESH_H_
#define BL_DG_IMPLICIT_MESH_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_VTK_utils.H>

namespace amrex
{
namespace DG
{
// AUXILIARY FUNCTIONS ################################################
// ELEMENT INFORMATION ================================================
void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<short const> const & etype_fab);
void PrintElementInfo(const Real * prob_lo, const Real * dx,
                      const int & i, const int & j, const int & k, const int & dom,
                      Array4<short const> const & etype_fab, Array4<Real const> const & einfo_fab);
//PrintElementInfo(&prob_lo[0], &dx[0], i, j, k, dom, eType_fab, eInfo_fab);
// ====================================================================

// GRID INFORMATION ===================================================
AMREX_GPU_HOST_DEVICE
void NBR_CELL(const int & i, const int & j, const int & k, const int & b,
              int & nbr_i, int & nbr_j, int & nbr_k, int & nbr_b);

AMREX_GPU_HOST_DEVICE
void GRID_FACE(const int & i, const int & j, const int & k, const int & b,
               int & fi, int & fj, int & fk);

AMREX_GPU_HOST_DEVICE
void FACE_2_NBRS(const int & i, const int & j, const int & k, const int & dir,
                 int & min_i, int & min_j, int & min_k,
                 int & pls_i, int & pls_j, int & pls_k);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const int & i, const int & j, const int & k, const int & etype,
             int & BF_i, int & BF_j, int & BF_k);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real* prob_lo, const Real* dx,
             const int & i, const int & j, const int & k, const int & etype,
             int & BF_i, int & BF_j, int & BF_k,
             Real * BF_lo, Real * BF_hi);

AMREX_GPU_HOST_DEVICE
void BF_CELL(const Real * prob_lo, const Real * dx,
             const int & i, const int & j, const int & k, const int & etype,
             Real * BF_lo, Real * BF_hi);

AMREX_GPU_HOST_DEVICE
void COARSE_TO_FINE_GRID_BOUNDS(const int i, const int j, const int k, const int b,
                                const IntVect & rr,
                                Dim3 & lo, Dim3 & hi);
// ====================================================================

// ALGOIM RELATED FUNCTIONS ===========================================
template<typename F>
iQuadRule iQuadGen(const F & phi, const Real * lo, const Real * hi, const int & dim, const int & side, const int & qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real,AMREX_SPACEDIM>(lo, hi), dim, side, qo);
}

template<typename F>
iTriRule iTriGen(const F & phi, const Real * lo, const Real * hi, const int & dim, const int & side, const int & qo)
{
    return Algoim::quadGen<AMREX_SPACEDIM>(phi, Algoim::BoundingBox<Real,AMREX_SPACEDIM>(lo, hi), dim, side, qo, [] (int p, int i) {return 1.0e-12+i*(1.0-2.0e-12)/(p-1);}, [] (int p, int i) {return (1.0-2.0e-12)/(p-1);});
}
// ====================================================================
// ####################################################################



// ####################################################################
// IMPLICIT MESH CLASS ################################################
// ####################################################################
template<int N_PHI, int N_DOM>
struct ImplicitMesh
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;
    static const int bou_data_ngr = 0;

    static const int eType_n_comp = DG_ELM_TYPE_N_COMP_PER_DOM*N_DOM;
    static const int eInfo_n_comp = DG_ELM_INFO_N_COMP_PER_DOM*N_DOM;
    static const int eColor_n_comp = DG_ELM_COLOR_N_COMP_PER_DOM*N_DOM;
    static const int eBouType_n_comp = DG_ELM_BOU_TYPE_N_COMP_PER_DOM*N_DOM;
    static const int eDomQuad_Nq_n_comp = DG_ELM_DOM_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eBouQuad_Nq_n_comp = DG_ELM_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eIntBouQuad_Nq_n_comp = DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eQuery_Npt_n_comp = DG_ELM_TRI_N_COMP_PER_DOM*N_DOM;
    static const int eDomQuad_pos_n_comp = DG_ELM_DOM_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eBouQuad_pos_n_comp = DG_ELM_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eIntBouQuad_pos_n_comp = DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*N_DOM;
    static const int eQuery_pos_n_comp = DG_ELM_TRI_N_COMP_PER_DOM*N_DOM;

    // GRID INFO
    RealBox rbx;
    Box ibx;
    Geometry geom;
    GpuArray<Real, AMREX_SPACEDIM> dx, prob_lo;
    GpuArray<int, AMREX_SPACEDIM> is_periodic;
    GpuArray<Real, AMREX_SPACEDIM> J, iJ;
    Real aJ, iaJ;
    GpuArray<Real, AMREX_SPACEDIM> bou_aJ;
    Real rtol, atol;

    // BOX ARRAY AND DISTRIBUTION MAPPING
    BoxArray cc_ba;
    Array<BoxArray, AMREX_SPACEDIM> fc_ba;
    DistributionMapping dm;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    UnitCube<AMREX_SPACEDIM> cube;

    // LEVEL SETS
    MultiFab PHI;

    // IMPLICIT MESH
    int space_q_im, time_q_im;
    Real volume_fraction_threshold;

    shortMultiFab eType;
    MultiFab eInfo;
    iMultiFab eColor;
    iMultiFab eDomQuad_Nq, eQuery_Npt;
    longMultiFab eDomQuad_pos, eQuery_pos;
    Array<shortMultiFab, AMREX_SPACEDIM> eBouType;
    Array<iMultiFab, AMREX_SPACEDIM> eBouQuad_Nq;
    Array<longMultiFab, AMREX_SPACEDIM> eBouQuad_pos;
    iMultiFab eIntBouQuad_Nq;
    longMultiFab eIntBouQuad_pos;

    int n_cut_elms, n_cut_elm_bou, n_entire_extended_elms;
    long mem_for_eDomQuad, mem_for_eBouQuad, mem_for_eIntBouQuad, mem_for_eQuery;

    Gpu::HostVector<Real> eDomQuad_host_mem;
    Gpu::HostVector<Real> eBouQuad_host_mem;
    Gpu::HostVector<Real> eIntBouQuad_host_mem;
    Gpu::HostVector<Real> eQuery_host_mem;
    
    Gpu::DeviceVector<Real> eDomQuad_device_mem;
    Gpu::DeviceVector<Real> eBouQuad_device_mem;
    Gpu::DeviceVector<Real> eIntBouQuad_device_mem;
    Gpu::DeviceVector<Real> eQuery_device_mem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    ImplicitMesh()
    {}
    // ################################################################

    // DESTRUCTOR #####################################################
    ~ImplicitMesh()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void DefineGridInfo()
    {
        this->dx = this->geom.CellSizeArray();
        this->prob_lo = this->geom.ProbLoArray();
        this->is_periodic = this->geom.isPeriodicArray();
        this->J = {AMREX_D_DECL(0.5*this->dx[0], 0.5*this->dx[1], 0.5*this->dx[2])};
        this->iJ = {AMREX_D_DECL(2.0/this->dx[0], 2.0/this->dx[1], 2.0/this->dx[2])};
        this->aJ = AMREX_D_PICK(this->J[0], (this->J[0])*(this->J[1]), (this->J[0])*(this->J[1])*(this->J[2]));
        this->iaJ = 1.0/this->aJ;
#if (AMREX_SPACEDIM == 1)
        this->bou_aJ = {1.0};
        this->rtol = 1.0e-12*this->dx[0];
#endif
#if (AMREX_SPACEDIM == 2)
        this->bou_aJ = {this->J[1], this->J[0]};
        this->rtol = 1.0e-12*fmin(this->dx[0], this->dx[1]);
#endif
#if (AMREX_SPACEDIM == 3)
        this->bou_aJ = {this->J[1]*this->J[2], this->J[0]*this->J[2], this->J[0]*this->J[1]};
        this->rtol = 1.0e-12*fmin(fmin(this->dx[0], this->dx[1]), this->dx[2]);
#endif
        this->atol = 1.0e-12;
    }

    void DefineUnitCube(const int phi_space_p, const int space_q)
    {
        this->cube.Define(phi_space_p, space_q);

        // SOME CHECKS
        BL_ASSERT(phi_space_p >= 1);
        BL_ASSERT(phi_space_p <= DG_PHI_MAX_SPACE_p);
        if ((phi_space_p != DG_PHI_MAX_SPACE_p) && (false))
        {
            Print() << std::endl;
            Print() << "WARNING: AMReX_DG_ImplicitMesh.H - ImplicitMesh::Define" << std::endl;
            Print() << "| DG_PHI_MAX_SPACE_p = " << DG_PHI_MAX_SPACE_p << std::endl;
            Print() << "| input phi space p = " << phi_space_p << std::endl;
            Print() << "| Consider setting DG_PHI_MAX_SPACE_p = " << phi_space_p << " and recompiling the code." <<std::endl;
            Print() << std::endl;
        }

        BL_ASSERT(space_q >= 1);
    }

    void DefineLevelSetsMultiFabs()
    {
        const int n_comp = (this->cube.Np)*N_PHI;

        this->PHI.define(this->cc_ba, this->dm, n_comp, this->dom_data_ngr);
        this->PHI = 0.0;
    }

    void DefineImplicitMeshInfo(const int space_q_im_, const int time_q_im_, const Real VF)
    {
        // ============================================================
        // WARNING: Remember that on the MultiFabs
        //          - this->eDomQuad_pos
        //          - this->eBouQuad_pos[dim]
        //          - this->eTri_pos
        //          the FillBoundary method must NEVER be called.
        // ============================================================
        this->space_q_im = space_q_im_;
        this->time_q_im = time_q_im_;
        this->volume_fraction_threshold = VF;
        this->n_cut_elms = 0;
        this->n_cut_elm_bou = 0;
        this->n_entire_extended_elms = 0;

        // DOMAIN (CELL-CENTERED) INFO --------------------------------
        {
            this->eType.define(this->cc_ba, this->dm, this->eType_n_comp, this->dom_data_ngr);
            this->eType = DG_ELM_TYPE_UNDEFINED;
            
            this->eInfo.define(this->cc_ba, this->dm, this->eInfo_n_comp, this->dom_data_ngr);
            this->eInfo = 0.0;

            this->eColor.define(this->cc_ba, this->dm, this->eColor_n_comp, this->dom_data_ngr);
            this->eColor = DG_ELM_COLOR_UNDEFINED;
            
            this->eDomQuad_Nq.define(this->cc_ba, this->dm, this->eDomQuad_Nq_n_comp, this->dom_data_ngr);
            this->eDomQuad_Nq = -1;
            
            this->eQuery_Npt.define(this->cc_ba, this->dm, this->eQuery_Npt_n_comp, this->dom_data_ngr);
            this->eQuery_Npt = -1;
            
            this->eDomQuad_pos.define(this->cc_ba, this->dm, this->eDomQuad_pos_n_comp, this->dom_data_ngr);
            this->eDomQuad_pos = -1L;
            
            this->eQuery_pos.define(this->cc_ba, this->dm, this->eQuery_pos_n_comp, this->dom_data_ngr);
            this->eQuery_pos = -1L;

            this->eIntBouQuad_Nq.define(this->cc_ba, this->dm, this->eIntBouQuad_Nq_n_comp, this->dom_data_ngr);
            this->eIntBouQuad_Nq = -1;

            this->eIntBouQuad_pos.define(this->cc_ba, this->dm, this->eIntBouQuad_pos_n_comp, this->dom_data_ngr);
            this->eIntBouQuad_pos = -1L;
        }
        // ------------------------------------------------------------

        // BOUNDARY (FACE-CENTERED) INFO ------------------------------
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->eBouType[dim].define(this->fc_ba[dim], this->dm, this->eBouType_n_comp, this->bou_data_ngr);
            this->eBouType[dim] = DG_ELM_BOU_TYPE_UNDEFINED;

            this->eBouQuad_Nq[dim].define(this->fc_ba[dim], this->dm, this->eBouQuad_Nq_n_comp, this->bou_data_ngr);
            this->eBouQuad_Nq[dim] = -1;

            this->eBouQuad_pos[dim].define(this->fc_ba[dim], this->dm, this->eBouQuad_pos_n_comp, this->bou_data_ngr);
            this->eBouQuad_pos[dim] = -1L;
        }
        // ------------------------------------------------------------

        this->mem_for_eDomQuad = 0L;
        this->mem_for_eBouQuad = 0L;
        this->mem_for_eIntBouQuad = 0L;
        this->mem_for_eQuery = 0L;
        // ============================================================
    }

    void Define(const InputReader & inputs, const int d)
    {
        // GRID INFO ==================================================
        this->rbx.setLo(inputs.space[d].prob_lo.data());
        this->rbx.setHi(inputs.space[d].prob_hi.data());
        
        this->ibx.setSmall({AMREX_D_DECL(0, 0, 0)});
        this->ibx.setBig(inputs.mesh[d].n_cells-1);
        
        this->geom.define(this->ibx, &this->rbx,
                          inputs.space[d].coord_sys,
                          inputs.space[d].is_periodic.data());

        this->DefineGridInfo();
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        this->cc_ba.define(this->ibx);
        this->cc_ba.maxSize(inputs.mesh[d].max_grid_size);

#if (AMREX_SPACEDIM == 1)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1));
#endif
#if (AMREX_SPACEDIM == 2)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1));
#endif
#if (AMREX_SPACEDIM == 3)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1,0));
        this->fc_ba[2] = convert(this->cc_ba, IntVect(0,0,1));
#endif

        this->dm.define(this->cc_ba);
        // ============================================================

        // UNIT CUBE [-1,1]^AMREX_SPACEDIM ============================
        this->DefineUnitCube(inputs.dG[d].phi_space_p, inputs.dG[d].space_q);
        // ============================================================

        // LEVEL SETS =================================================
        this->DefineLevelSetsMultiFabs();
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->DefineImplicitMeshInfo(inputs.dG[d].space_q_im, inputs.dG[d].time_q_im,
                                     inputs.dG[d].VF);
        // ============================================================
    }
    // ################################################################

    // INITIALIZATION FOR A SLOPE LIMITER #############################
    void DefineForSlopeLimiter(const ImplicitMesh<N_PHI, N_DOM> & coarse,
                               const IntVect & rr,
                               const InputReader & inputs, const int d)
    {
        // GRID INFO ==================================================
        this->rbx = coarse.rbx;

        this->ibx = coarse.ibx;
        this->ibx.refine(rr);

        this->geom.define(this->ibx, &this->rbx,
                          inputs.space[d].coord_sys,
                          inputs.space[d].is_periodic.data());

        this->DefineGridInfo();
        // ============================================================

        // BOX ARRAY AND DISTRIBUTION MAPPING =========================
        this->cc_ba = coarse.cc_ba;
        this->cc_ba.refine(rr);

#if (AMREX_SPACEDIM == 1)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1));
#endif
#if (AMREX_SPACEDIM == 2)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1));
#endif
#if (AMREX_SPACEDIM == 3)
        this->fc_ba[0] = convert(this->cc_ba, IntVect(1,0,0));
        this->fc_ba[1] = convert(this->cc_ba, IntVect(0,1,0));
        this->fc_ba[2] = convert(this->cc_ba, IntVect(0,0,1));
#endif

        this->dm = coarse.dm;
        // ============================================================

        // UNIT CUBE [-1,1]^AMREX_SPACEDIM ============================
        this->DefineUnitCube(inputs.dG[d].phi_space_p, inputs.dG[d].space_q);

        // WE ALSO NEED THE INTERPOLATION OPERATOR
        this->cube.InitInterpolation(coarse.cube, rr);
        // ============================================================

        // LEVEL SETS =================================================
        this->DefineLevelSetsMultiFabs();
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->DefineImplicitMeshInfo(inputs.dG[d].space_q_im, inputs.dG[d].time_q_im,
                                     inputs.dG[d].VF);
        // ============================================================
    }
    // ################################################################

    // CHECK QUADRATURE RULES #########################################
    template<typename IBVP_CLASS>
    void CheckQuadratureRules(const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int QUAD_CHECK_N_COMP_PER_DOM = 3;
        // ============================================================

        // VARIABLES ==================================================
        Real exact_volume[N_DOM], exact_surface[N_DOM];
        Real volume[N_DOM], volume_from_bou_quadrature[N_DOM], surface[N_DOM];
        Real error_in_volume[N_DOM], error_in_volume_from_bou_quadrature[N_DOM], error_in_surface[N_DOM];

#ifdef AMREX_USE_CUDA
        Real * eDomQuad_mem_ptr = this->eDomQuad_device_mem.data();
        Real * eBouQuad_mem_ptr = this->eBouQuad_device_mem.data();
        Real * eIntBouQuad_mem_ptr = this->eIntBouQuad_device_mem.data();
#else
        Real * eDomQuad_mem_ptr = this->eDomQuad_host_mem.data();
        Real * eBouQuad_mem_ptr = this->eBouQuad_host_mem.data();
        Real * eIntBouQuad_mem_ptr = this->eIntBouQuad_host_mem.data();
#endif

        MultiFab quad_check;
        // ============================================================

        // INITIALIZATION =============================================
        quad_check.define(this->cc_ba, this->dm, QUAD_CHECK_N_COMP_PER_DOM*N_DOM, this->dom_data_ngr);
        quad_check = 0.0;
        // ============================================================

        // COMPUTE VOLUME/SURFACE OF EACH DOMAIN ======================
        for (MFIter mfi(quad_check); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & eDomQuad_Nq_fab = this->eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = this->eDomQuad_pos.array(mfi);
            AMREX_D_TERM
            (
                Array4<int const> const & eBouQuad_Nq_fab_0 = this->eBouQuad_Nq[0].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_0 = this->eBouQuad_pos[0].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_1 = this->eBouQuad_Nq[1].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_1 = this->eBouQuad_pos[1].array(mfi);,
                Array4<int const> const & eBouQuad_Nq_fab_2 = this->eBouQuad_Nq[2].array(mfi);
                Array4<long const> const & eBouQuad_pos_fab_2 = this->eBouQuad_pos[2].array(mfi);
            )
            Array4<int const> const & eIntBouQuad_Nq_fab = this->eIntBouQuad_Nq.array(mfi);
            Array4<long const> const & eIntBouQuad_pos_fab = this->eIntBouQuad_pos.array(mfi);
            Array4<Real> const & dst_fab = quad_check.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                  prob_lo[1]+(j+0.5)*dx[1],
                                                                  prob_lo[2]+(k+0.5)*dx[2])};

                // LOCAL VARIABLES
                int dom_Nq, bou_Nq;
                int fi, fj, fk;
                long pos, qi_pos;
                const Real * xptr;
                Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                Real integrand;
                Real cell_volume, cell_volume_from_bou_quadrature, cell_surface;

                // INITIALIZATION
                dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom) = 0.0;
                dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) = 0.0;
                dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) = 0.0;
                cell_volume = 0.0;
                cell_volume_from_bou_quadrature = 0.0;
                cell_surface = 0.0;

                // DOMAIN INTEGRAL
                {
                    // QUADRATURE INFO
                    dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    xptr = &eDomQuad_mem_ptr[pos];

                    // EVAL THE INTEGRAL
                    for (int qi = 0; qi < dom_Nq; ++qi)
                    {
                        qi_pos = (AMREX_SPACEDIM+1)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                        )
                        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

                        integrand = 1.0;
                        dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom) += integrand*w;

                        integrand = 1.0;
                        cell_volume += integrand*w;
                    }
                }

                // BOUNDARY INTEGRAL
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    GRID_FACE(i, j, k, b, fi, fj, fk);

                    // QUADRATURE INFO
                    if (b/2 == 0)
                    {
                        bou_Nq = eBouQuad_Nq_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        pos = eBouQuad_pos_fab_0(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                    }
#if (AMREX_SPACEDIM > 1)
                    else if (b/2 == 1)
                    {
                        bou_Nq = eBouQuad_Nq_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        pos = eBouQuad_pos_fab_1(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                    }
#endif
#if (AMREX_SPACEDIM > 2)
                    else
                    {
                        bou_Nq = eBouQuad_Nq_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                        pos = eBouQuad_pos_fab_2(fi,fj,fk,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1-(b%2));
                    }
#endif
                    xptr = &eBouQuad_mem_ptr[pos];

                    // UNIT NORMAL
                    AMREX_D_TERM(un[0] = 0.0;, un[1] = 0.0;, un[2] = 0.0;)
                    un[b/2] = 2.0*(b%2)-1.0;

                    // EVAL THE INTEGRAL
                    for (int qi = 0; qi < bou_Nq; ++qi)
                    {
                        qi_pos = (AMREX_SPACEDIM+1)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                        )
                        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*bou_aJ[b/2];

                        integrand = x[1]*un[1];
                        dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) += integrand*w;

                        integrand = 0.0;
                        dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) += integrand*w;

                        integrand = x[2]*un[2];
                        cell_volume_from_bou_quadrature += integrand*w;
                        
                        integrand = 0.0;
                        cell_surface += integrand*w;
                    }
                }

                // INTERNAL BOUNDARY INTEGRAL
                {
                    // QUADRATURE INFO
                    bou_Nq = eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    pos = eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom);
                    xptr = &eIntBouQuad_mem_ptr[pos];

                    // COMPUTE THE INTEGRAL
                    for (int qi = 0; qi < bou_Nq; ++qi)
                    {
                        qi_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0];,
                            x[1] = xptr[qi_pos+1];,
                            x[2] = xptr[qi_pos+2];
                        )
                        w = xptr[qi_pos+AMREX_SPACEDIM];

                        // UNIT NORMAL
                        AMREX_D_TERM
                        (
                            un[0] = xptr[qi_pos+AMREX_SPACEDIM+1+0];,
                            un[1] = xptr[qi_pos+AMREX_SPACEDIM+1+1];,
                            un[2] = xptr[qi_pos+AMREX_SPACEDIM+1+2];
                        )

                        integrand = x[1]*un[1];
                        dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+1) += integrand*w;

                        integrand = 1.0;
                        dst_fab(i,j,k,QUAD_CHECK_N_COMP_PER_DOM*dom+2) += integrand*w;

                        integrand = x[1]*un[1];
                        cell_volume_from_bou_quadrature += integrand*w;

                        integrand = 1.0;
                        cell_surface += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }

        for (int dom = 0; dom < N_DOM; ++dom)
        {
            volume[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom);
            volume_from_bou_quadrature[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom+1);
            surface[dom] = quad_check.sum(QUAD_CHECK_N_COMP_PER_DOM*dom+2);
        }
        // ============================================================

        // COMPARE WITH RESULTS FROM IBVP CLASS =======================
        for (int dom = 0; dom < N_DOM; ++dom)
        {
            exact_volume[dom] = IBVP.F_EXACT_VOLUME(dom);
            exact_surface[dom] = IBVP.F_EXACT_SURFACE(dom);
        }

        for (int dom = 0; dom < N_DOM; ++dom)
        {
            error_in_volume[dom] = std::abs(volume[dom]-exact_volume[dom])/exact_volume[dom]*100.0;
            error_in_volume_from_bou_quadrature[dom] = std::abs(volume_from_bou_quadrature[dom]-exact_volume[dom])/exact_volume[dom]*100.0;
            
            if (exact_surface[dom] < this->rtol)
            {
                error_in_surface[dom] = std::abs(surface[dom]-exact_surface[dom]);
            }
            else
            {
                error_in_surface[dom] = std::abs(surface[dom]-exact_surface[dom])/exact_surface[dom]*100.0;
            }
        }
        // ============================================================

        // PRINT THE REPORT ===========================================
        Print() << "IMPLICIT MESH QUADRATURE REPORT:" << std::endl;

        if (ParallelDescriptor::IOProcessor())
        {
            for (int dom = 0; dom < N_DOM; ++dom)
            {
                Print() << "| Domain " << dom << std::endl;
                printf("| - volume (from domain quadrature)  : %.5f ( %.5f %% error)\n", volume[dom], error_in_volume[dom]);
                printf("| - volume (from boundary quadrature): %.5f ( %.5f %% error)\n", volume_from_bou_quadrature[dom], error_in_volume_from_bou_quadrature[dom]);
                if (exact_surface[dom] < this->rtol)
                {
                    printf("| - surface                          : %.5f ( %.5f error)\n", surface[dom], error_in_surface[dom]);
                }
                else
                {
                    printf("| - surface                          : %.5f ( %.5f %% error)\n", surface[dom], error_in_surface[dom]);
                }
            }
        }

        ParallelDescriptor::Barrier();
        Print() << std::endl;
        // ============================================================
    }
    // ################################################################
};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif
