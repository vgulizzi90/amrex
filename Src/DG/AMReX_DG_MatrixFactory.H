//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_MatrixFactory.H
 * \brief Contains operators to manage dG methods over implicitly-defined meshes.
*/

#ifndef AMREX_DG_MATRIX_FACTORY_H_
#define AMREX_DG_MATRIX_FACTORY_H_

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_ImplicitMesh.H>

namespace amrex
{
namespace DG
{

// ####################################################################
// MATRIX FACTORY CLASS ###############################################
// ####################################################################
/**
 * \brief Matrix factory class.
 *
*/
struct MatrixFactory
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;

    // GRID
    StandardRectangle<AMREX_SPACEDIM> std_elem;

    // CHOLESKY DECOMPOSITION OF THE ELEMENTS' MASS MATRICES
    longMultiFab eMMCh_pos;
    long mem_for_eMMCh;
    Gpu::HostVector<Real> eMMCh_host_mem;
    Gpu::DeviceVector<Real> eMMCh_device_mem;

    // INTERPOLATION OPERATOR
    longMultiFab space_Dom_I_pos;
    long mem_for_space_Dom_I;
    Gpu::HostVector<Real> space_Dom_I_host_mem;
    Gpu::DeviceVector<Real> space_Dom_I_device_mem;
    // ################################################################

    // CONSTRUCTOR ####################################################
    MatrixFactory()
    {}
    
    MatrixFactory(const InputReader & inputs)
    {
        this->define(inputs);
    }
    // ################################################################

    // DESTRUCTOR #####################################################
    ~MatrixFactory()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void define(const amrex::Geometry & input_geom, const int input_dG_space_p, const int input_dG_space_q)
    {
        // STANDARD ELEMENT ===========================================
        this->std_elem.define(input_geom.CellSize(), input_dG_space_p, input_dG_space_q);
        // ============================================================   
    }

    void define(const amrex::Geometry & input_geom, const amrex::BoxArray & /*input_ba*/, const amrex::DistributionMapping & /*input_dm*/, const InputReader & inputs)
    {
        this->define(input_geom, inputs.dG.space_p, inputs.dG.space_p+1);
    }

    void define(const InputReader & inputs)
    {
        // VARIABLES ==================================================
        RealBox rbx;
        Box ibx;
        Geometry input_geom;
        // ============================================================

        // GEOMETRY ===================================================
        rbx.setLo(inputs.space.lo.data());
        rbx.setHi(inputs.space.hi.data());
        
        AMREX_D_TERM
        (
            ibx.setSmall(0, 0);,
            ibx.setSmall(1, 0);,
            ibx.setSmall(2, 0);
        )
        ibx.setBig(inputs.grid.n_cells-1);

        input_geom.define(ibx, &rbx, inputs.space.coord_sys, inputs.space.is_periodic.data());
        // ============================================================

        this->define(input_geom, inputs.dG.space_p, inputs.dG.space_p+1);
    }
    // ################################################################

    // EVAL AND STORE THE CHOLESKY DECOMPOSITION OF THE MESH ELEMENTS'
    // MASS MATRICES ##################################################
    /**
     * \brief Eval and store the Cholesky decomposition of the implicitly-defined mesh elements.
     *
     * \param[in] N_DOM: number of domains that will be considered.
    */
    void EvalMassMatrices(const int N_DOM, const ImplicitMesh & mesh)
    {
        // PROFILING ------------------------------------------------------
        BL_PROFILE("EvalMassMatrices(const int, const ImplicitMesh &, const IBVP_CLASS &)");
        // ----------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

        // STANDARD ELEMENT
        const int p = this->std_elem.p;
        const int sNp = this->std_elem.Np;

        // IMPLICIT MESH
        const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
    
        // MULTIFAB INFO
        const int eMMCh_pos_n_comp = N_DOM;
        // ============================================================

        // VARIABLES ==================================================
        long eMMCh_mem_pos;
        // ============================================================

        // INITIALIZATION =============================================
        this->eMMCh_pos.define(mesh.cc_ba, mesh.dm, eMMCh_pos_n_comp, this->dom_data_ngr);
        this->eMMCh_pos = -1L;

        eMMCh_mem_pos = 0L;
        // ============================================================

        // THE FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO =
        {
            // Resize memory
            this->eMMCh_host_mem.resize(sNp*sNp);

            // Store data
            std::copy(this->std_elem.MMCh.begin(), this->std_elem.MMCh.end(), this->eMMCh_host_mem.data());

            // Update memory
            eMMCh_mem_pos += sNp*sNp;
        }
        // ============================================================

        // EVAL AND STORE THE MASS MATRICES ===========================
        for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

            Array4<long> const & eMMCh_pos_fab = this->eMMCh_pos.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_empty = (etype == __DG_ELM_TYPE_EMPTY__);
                const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool elm_is_small = (etype%10 == __DG_ELM_TYPE_SMALL__);
                const bool elm_is_partial = (elm_is_large || elm_is_small);
                const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real MM[__DG_SOL_MAX_SPACE_Np__*__DG_SOL_MAX_SPACE_Np__];
                Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

                // Get the support of the basis functions. It is
                // recalled that if the element is small, the support
                // of the basis functions is given by the element that
                // is merging into the small current element.
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

                // BASIS FUNCTIONS
                DG_SOL_space_BF bf(p, BF_lo, BF_hi);

                // EMPTY ELEMENTS -------------------------------------
                if (elm_is_empty)
                {
                    eMMCh_pos_fab(i,j,k,dom) = -1L;
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (elm_is_entire)
                {
                    // NOTE: If the element is extended, its mass
                    //       matrix must be supplemented by the
                    //       contribution of the small elements.
                    if (elm_is_entire_extended)
                    {
                        // Store memory offset
                        eMMCh_pos_fab(i,j,k,dom) = eMMCh_mem_pos;

                        // Expand memory
                        this->eMMCh_host_mem.resize(eMMCh_mem_pos+sNp*sNp);

                        // Store data
                        std::copy(this->std_elem.MM.begin(), this->std_elem.MM.end(), &this->eMMCh_host_mem[eMMCh_mem_pos]);

                        // Update memory
                        eMMCh_mem_pos += sNp*sNp;
                    }
                    else
                    {
                        eMMCh_pos_fab(i,j,k,dom) = 0L;
                    }
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else if (elm_is_partial)
                {
                    // QUADRATURE INFO
                    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                    const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                    const Real * xptr = &eDomQuad_mem_ptr[pos];
                    long x_pos;

                    // Store memory offset
                    eMMCh_pos_fab(i,j,k,dom) = eMMCh_mem_pos;

                    // Expand memory
                    this->eMMCh_host_mem.resize(eMMCh_mem_pos+sNp*sNp);

                    // Store data
                    std::fill(MM, MM+sNp*sNp, 0.0);

                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0]+cell_lo[0];,
                            x[1] = xptr[x_pos+1]+cell_lo[1];,
                            x[2] = xptr[x_pos+2]+cell_lo[2];
                        )
                        w = xptr[x_pos+AMREX_SPACEDIM];

                        bf.eval_BF_only_table(x);

                        for (int cs = 0; cs < sNp; ++cs)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            MM[rs+cs*sNp] += bf.BF[rs]*bf.BF[cs]*w;
                        }
                    }

                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        this->eMMCh_host_mem[eMMCh_mem_pos+rs] = MM[rs];
                    }

                    // Update memory
                    eMMCh_mem_pos += sNp*sNp;
                }
                // ----------------------------------------------------
            }
        }
        // ============================================================

        // STORE MEMORY INFO ==========================================
        this->mem_for_eMMCh = eMMCh_mem_pos;
        // ============================================================

        // COPY MEMORY TO THE DEVICE ==================================
#ifdef AMREX_USE_CUDA
        this->eMMCh_device_mem.resize(this->eMMCh_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->eMMCh_host_mem.begin(), this->eMMCh_host_mem.end(), this->eMMCh_device_mem.begin());
#endif
        // ============================================================

        // EVAL CHOLESKY DECOMPOSITION ================================
        // Note that, before evaluating the Cholesky decomposition for 
        // the extended elements, their mass matrices are supplemented
        // with the contribution of the small elements.
        // ============================================================
#ifdef AMREX_USE_CUDA
        Real * eMMCh_mem_ptr = this->eMMCh_device_mem.data();
#else
        Real * eMMCh_mem_ptr = this->eMMCh_host_mem.data();
#endif

        for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<long const> const & eMMCh_pos_fab = this->eMMCh_pos.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_entire = (etype%10 == __DG_ELM_TYPE_ENTIRE__);
                const bool elm_is_large = (etype%10 == __DG_ELM_TYPE_LARGE__);
                const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);
                const bool elm_is_special = elm_is_large || elm_is_entire_extended;
                const bool elm_is_valid = elm_is_entire || elm_is_large;
                const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

                const long pos = eMMCh_pos_fab(i,j,k,dom);

                // LOCAL VARIABLES
                short nbr_etype;
                int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
                bool nbr_is_small;
                long nbr_pos;
                Real MM[__DG_SOL_MAX_SPACE_Np__*__DG_SOL_MAX_SPACE_Np__];
                int info;

                // RETRIEVE THE DG MATRICES FOR VALID SPECIAL ELEMENTS
                if (elm_is_special)
                {
                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        // Mass matrix
                        MM[rs] = eMMCh_mem_ptr[pos+rs];
                    }
                }

                // Cycle over the boundaries, check whether the element
                // is extended and add the contribution
                if (elm_is_extended)
                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                {
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                    nbr_is_small = (nbr_etype%10 == __DG_ELM_TYPE_SMALL__);
                    nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                    if (nbr_merged_b == nbr_b)
                    {
                        nbr_pos = eMMCh_pos_fab(nbr_i,nbr_j,nbr_k,dom);

                        for (int rs = 0; rs < (sNp*sNp); ++rs)
                        {
                            // Mass matrix
                            MM[rs] += eMMCh_mem_ptr[nbr_pos+rs];
                        }
                    }
                }

                // CHOLESKY DECOMPOSITION OF MASS MATRIX FOR VALID
                // SPECIAL ELEMENTS
                if (elm_is_special)
                {
                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        // Mass matrix
                        eMMCh_mem_ptr[pos+rs] = MM[rs];
                    }

                    linalg::dpotf2('U', sNp, &eMMCh_mem_ptr[pos], sNp, info);
                }
            });
            Gpu::synchronize();
        }

        // HANDLE GHOST ELEMENTS
        for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<long const> const & eMMCh_pos_fab = this->eMMCh_pos.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_ghost = ELM_IS_GHOST(etype);
                const bool elm_is_large = ELM_IS_LARGE(etype);

                if (elm_is_ghost && elm_is_large)
                {
                    const long pos = eMMCh_pos_fab(i,j,k,dom);
                    int info;

                    // CHOLESKY DECOMPOSITION
                    linalg::dpotf2('U', sNp, &eMMCh_mem_ptr[pos], sNp, info);
                }
            });
            Gpu::synchronize();
        }
        // ================================================================
    }

    /**
     * \brief Eval and store the Cholesky decomposition of the implicitly-defined mesh elements.
     *
     * \param[in] N_DOM: number of domains that will be considered.
    */
    void EvalMassMatrices(const ImplicitMesh & mesh)
    {
        this->EvalMassMatrices(1, mesh);
    }
    // ################################################################

    // INTERPOLATION OPERATOR #########################################
    /**
     * \brief Eval and store the interpolation operator with respect to a coarser mesh.
     *
     * \param[in] N_DOM: number of domains that will be considered.
     * \param[in] mesh: data structure containing the mesh.
     * \param[in] rr: refinement ratio.
     * \param[in] c_eType: shortMultiFab structure containing the merging information.
     * \param[in] c_sp: order of the basis functions at the coarse level.
    */
    void EvalInterpolationOperator(const int N_DOM,
                                   const ImplicitMesh & mesh,
                                   const IntVect & rr,
                                   const shortMultiFab & c_eType,
                                   const int c_sp)
    {
        // PROFILING --------------------------------------------------
        BL_PROFILE("EvalInterpolationOperator(const int, const ImplicitMesh &, const IntVect &, ....)");
        // ------------------------------------------------------------

        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> f_dx = mesh.geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.geom.ProbLoArray();

        // GRID (COARSE MESH)
        const GpuArray<Real, AMREX_SPACEDIM> c_dx = {AMREX_D_DECL(f_dx[0]*rr[0], f_dx[1]*rr[1], f_dx[2]*rr[2])};

        // STANDARD ELEMENT
        const int f_sp = this->std_elem.p;
        const int f_sNp = this->std_elem.Np;

        // STANDARD ELEMENT (COARSE MESH)
        const int c_sNp = AMREX_D_TERM((1+c_sp),*(1+c_sp),*(1+c_sp));

        // IMPLICIT MESH
        const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();

        // MULTIFAB INFO
        const int space_Dom_I_pos_n_comp = N_DOM;
        // ============================================================

        // VARIABLES ==================================================
        long space_Dom_I_mem_pos;
        // ============================================================

        // INITIALIZATION =============================================
        this->space_Dom_I_pos.define(mesh.cc_ba, mesh.dm, space_Dom_I_pos_n_comp, this->dom_data_ngr);
        this->space_Dom_I_pos = -1L;

        space_Dom_I_mem_pos = 0L;
        // ============================================================

        // THE FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO =
        {
            const int size = f_sNp*c_sNp*AMREX_D_TERM(rr[0],*rr[1],*rr[2]);

            // Resize memory
            this->space_Dom_I_host_mem.resize(size);

            // Store data
            for (int k = 0; k < size; ++k)
            {
                this->space_Dom_I_host_mem[k] = this->std_elem.IBfTBc[k];
            }

            // Update memory
            space_Dom_I_mem_pos += size;
        }
        // ============================================================

        // EVAL AND STORE THE INTERPOLATION OPERATOR ==================
        for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            // FINE MESH
            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
            Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

            // COARSE MESH
            Array4<short const> const & c_eType_fab = c_eType.array(mfi);

            // INTERPOLATION OPERATOR
            Array4<long> const & space_Dom_I_pos_fab = this->space_Dom_I_pos.array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real f_cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*f_dx[0],
                                                                     prob_lo[1]+j*f_dx[1],
                                                                     prob_lo[2]+k*f_dx[2])};

                // LOCAL VARIABLES
                Real f_BF_lo[AMREX_SPACEDIM], f_BF_hi[AMREX_SPACEDIM];
                Real c_BF_lo[AMREX_SPACEDIM], c_BF_hi[AMREX_SPACEDIM];
                Real x[AMREX_SPACEDIM], w;
                Real I[__DG_SOL_MAX_SPACE_Np__*__DG_SOL_MAX_SPACE_Np__];

                // INDICES OF THE COARSE CELL
#if (AMREX_SPACEDIM == 1)
Print() << "EvalInterpolationOperator: AMREX_SPACEDIM == 1" << std::endl;
exit(-1);
#endif
#if (AMREX_SPACEDIM == 2)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int c_i = (i-pi)/rr[0];
                const int c_j = (j-pj)/rr[1];
                const int c_k = k;
                const int I_pos = (pi+pj*rr[0])*f_sNp*c_sNp;
#endif
#if (AMREX_SPACEDIM == 3)
                const int pi = (i%rr[0] >= 0) ? i%rr[0] : (i%rr[0]+rr[0]);
                const int pj = (j%rr[1] >= 0) ? j%rr[1] : (j%rr[1]+rr[1]);
                const int pk = (k%rr[2] >= 0) ? k%rr[2] : (k%rr[2]+rr[2]);
                const int c_i = (i-pi)/rr[0];
                const int c_j = (j-pj)/rr[1];
                const int c_k = (k-pk)/rr[2];
                const int I_pos = (pi+pj*rr[0]+pk*rr[0]*rr[1])*f_sNp*c_sNp;
#endif

                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));
                const short c_etype = c_eType_fab(c_i,c_j,c_k,ELM_TYPE(dom));

                // SUPPORT OF THE BASIS FUNCTIONS
                BF_CELL(&prob_lo[0], &f_dx[0], i, j, k, etype, f_BF_lo, f_BF_hi);
                BF_CELL(&prob_lo[0], &c_dx[0], c_i, c_j, c_k, c_etype, c_BF_lo, c_BF_hi);

                // BASIS FUNCTIONS
                DG_SOL_space_BF f_bf(f_sp, f_BF_lo, f_BF_hi);
                DG_SOL_space_BF c_bf(c_sp, c_BF_lo, c_BF_hi);

                // EMPTY ELEMENTS -------------------------------------
                if (ELM_IS_EMPTY(etype))
                {
                    // Store memory offset
                    space_Dom_I_pos_fab(i,j,k,dom) = -1L;
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (ELM_IS_ENTIRE(etype))
                {
                    // NOTE: If the element is extended, we must 
                    //       provide the contribution of the small
                    //       elements.
                    if (ELM_IS_EXTENDED(etype))
                    {
                        // Store memory offset
                        space_Dom_I_pos_fab(i,j,k,dom) = space_Dom_I_mem_pos;

                        // Expand memory
                        this->space_Dom_I_host_mem.resize(space_Dom_I_mem_pos+f_sNp*c_sNp);

                        // Store data
                        std::copy(&this->std_elem.IBfTBc[I_pos], &this->std_elem.IBfTBc[I_pos+f_sNp*c_sNp], &this->space_Dom_I_host_mem[space_Dom_I_mem_pos]);

                        // Update memory
                        space_Dom_I_mem_pos += f_sNp*c_sNp;
                    }
                    else
                    {
                        space_Dom_I_pos_fab(i,j,k,dom) = I_pos;
                    }
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else if (ELM_IS_LARGE(etype) || ELM_IS_SMALL(etype))
                {
                    // QUADRATURE INFO
                    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,ELM_DOM_QUAD_NQ(dom));
                    const long pos = eDomQuad_pos_fab(i,j,k,ELM_DOM_QUAD_POS(dom));
                    const Real * xptr = &eDomQuad_mem_ptr[pos];
                    long x_pos;

                    // Store memory offset
                    space_Dom_I_pos_fab(i,j,k,dom) = space_Dom_I_mem_pos;

                    // Expand memory
                    this->space_Dom_I_host_mem.resize(space_Dom_I_mem_pos+f_sNp*c_sNp);

                    // Store data
                    std::fill(I, I+f_sNp*c_sNp, 0.0);

                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[x_pos+0]+f_cell_lo[0];,
                            x[1] = xptr[x_pos+1]+f_cell_lo[1];,
                            x[2] = xptr[x_pos+2]+f_cell_lo[2];
                        )
                        w = xptr[x_pos+AMREX_SPACEDIM];

                        f_bf.eval_BF_only_table(x);
                        c_bf.eval_BF_only_table(x);

                        for (int cs = 0; cs < c_sNp; ++cs)
                        for (int rs = 0; rs < f_sNp; ++rs)
                        {
                            I[rs+cs*f_sNp] += f_bf.BF[rs]*c_bf.BF[cs]*w;
                        }
                    }

                    for (int rs = 0; rs < (f_sNp*c_sNp); ++rs)
                    {
                        this->space_Dom_I_host_mem[space_Dom_I_mem_pos+rs] = I[rs];
                    }

                    // Update memory
                    space_Dom_I_mem_pos += f_sNp*c_sNp;
                }
                // ----------------------------------------------------
            }
        }

        // STORE MEMORY INFO
        this->mem_for_space_Dom_I = space_Dom_I_mem_pos;
        // ============================================================

        // COPY MEMORY TO THE DEVICE ==================================
#ifdef AMREX_USE_GPU
        this->space_Dom_I_device_mem.resize(this->space_Dom_I_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->space_Dom_I_host_mem.begin(), this->space_Dom_I_host_mem.end(), this->space_Dom_I_device_mem.begin());
#endif
        // ============================================================

        // ADD THE CONTRIBUTION FOR THE EXTENDED ELEMENTS =============
        /*
#ifdef AMREX_USE_GPU
        Real * space_Dom_I_mem_ptr = this->space_Dom_I_device_mem.data();
#else
        Real * space_Dom_I_mem_ptr = this->space_Dom_I_host_mem.data();
#endif

        for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & eType_fab = mesh.eType.array(mfi);

            Array4<long const> const & space_Dom_I_pos_fab = this->space_Dom_I_pos.array(mfi);

            ParallelFor(bx, N_DOM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = eType_fab(i,j,k,ELM_TYPE(dom));

                if (ELM_IS_EXTENDED(etype))
                {
                    const long pos = space_Dom_I_pos_fab(i,j,k,dom);

                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    long nbr_pos;

                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        const bool nbr_is_small = ELM_IS_SMALL(nbr_etype);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                        if (nbr_merged_b == nbr_b)
                        {
                            nbr_pos = space_Dom_I_pos_fab(nbr_i,nbr_j,nbr_k,dom);

                            for (int rs = 0; rs < (f_sNp*c_sNp); ++rs)
                            {
                                space_Dom_I_mem_ptr[pos+rs] += space_Dom_I_mem_ptr[nbr_pos+rs];
                            }
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        */
        // ============================================================
    }
    
    /**
     * \brief Eval and store the interpolation operator with respect to a coarser mesh.
     *
     * \param[in] N_DOM: number of domains that will be considered.
     * \param[in] mesh: data structure containing the mesh.
     * \param[in] rr: refinement ratio.
     * \param[in] c_mesh: data structure containing the coarser mesh.
     * \param[in] c_matfactory: data structure containing the coarser matrix factory.
    */
    void EvalInterpolationOperator(const int N_DOM,
                                   const ImplicitMesh & mesh,
                                   const IntVect & rr,
                                   const ImplicitMesh & c_mesh,
                                   const MatrixFactory & c_matfactory)
    {
        // CONSISTENCY CHECK ==========================================
        {
            const bool bad_rr[AMREX_SPACEDIM] = 
            {
                AMREX_D_DECL
                (
                    std::abs(this->std_elem.dx[0]*rr[0]-c_matfactory.std_elem.dx[0]) > mesh.rel_tol,
                    std::abs(this->std_elem.dx[1]*rr[1]-c_matfactory.std_elem.dx[1]) > mesh.rel_tol,
                    std::abs(this->std_elem.dx[2]*rr[2]-c_matfactory.std_elem.dx[2]) > mesh.rel_tol
                )
            };

            if (AMREX_D_TERM(bad_rr[0], || bad_rr[1], || bad_rr[2]))
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_DG_MatrixFactory.H - MatrixFactory::EvalInterpolationOperator\n";
                msg += "| The input coarse ImplicitMesh is not consistent with the input fine ImplicitMesh:\n";
                msg += "| Input coarse dx = "+AMREX_D_TERM(std::to_string(c_matfactory.std_elem.dx[0]),+","+
                                                           std::to_string(c_matfactory.std_elem.dx[1]),+","+
                                                           std::to_string(c_matfactory.std_elem.dx[2]))+"\n";
                msg += "| This fine dx = "+AMREX_D_TERM(std::to_string(this->std_elem.dx[0]),+","+
                                                        std::to_string(this->std_elem.dx[1]),+","+
                                                        std::to_string(this->std_elem.dx[2]))+"\n";
                msg += "| Refinement ratios = "+AMREX_D_TERM(std::to_string(rr[0]),+","+
                                                             std::to_string(rr[1]),+","+
                                                             std::to_string(rr[2]))+"\n";
                Abort(msg);
            }
        }
        // ============================================================

        // INIT THE INTERPOLATION OPERATOR FOR THE STANDARD ELEMENT ===
        this->std_elem.InitInterpolation(c_matfactory.std_elem, rr);
        // ============================================================

        // CALL THE MAIN FUNCTION =====================================
        if (isMFIterSafe(mesh.eType, c_mesh.eType))
        {
            EvalInterpolationOperator(N_DOM, mesh, rr, c_mesh.eType, c_matfactory.std_elem.p);
        }
        else
        {
            const BoxArray safe_c_ba = coarsen(mesh.cc_ba, rr);
            const DistributionMapping & f_dm = mesh.dm;
            const int eType_n_comp = c_mesh.eType.n_comp;
            const IntVect eType_n_grow = c_mesh.eType.n_grow;
            shortMultiFab safe_c_eType(safe_c_ba, f_dm, eType_n_comp, eType_n_grow);
            
            safe_c_eType.ParallelCopy(c_mesh.eType, 0, 0, eType_n_comp, eType_n_grow, eType_n_grow, c_mesh.geom.periodicity());

            EvalInterpolationOperator(N_DOM, mesh, rr, safe_c_eType, c_matfactory.std_elem.p);
        }
        // ============================================================
    }

    /**
     * \brief Eval and store the interpolation operator with respect to a coarser mesh.
     *
     * \param[in] mesh: data structure containing the mesh.
     * \param[in] rr: refinement ratio.
     * \param[in] c_mesh: data structure containing the coarser mesh.
     * \param[in] c_matfactory: data structure containing the coarser matrix factory.
    */
    void EvalInterpolationOperator(const ImplicitMesh & mesh,
                                   const IntVect & rr,
                                   const ImplicitMesh & c_mesh,
                                   const MatrixFactory & c_matfactory)
    {
        const int N_DOM = 1;
        this->EvalInterpolationOperator(N_DOM, mesh, rr, c_mesh, c_matfactory);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif