//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_DG_MatrixFactory.H
 * \brief Contains operators to manage dG methods over implicitly-defined meshes.
*/

#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_ImplicitMesh.H>

#ifndef AMREX_DG_MATRIX_FACTORY_H_
#define AMREX_DG_MATRIX_FACTORY_H_

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
    void define(const InputReader & inputs)
    {
        // VARIABLES ==================================================
        RealBox rbx;
        Box ibx;
        Geometry geom;
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

        geom.define(ibx, &rbx, inputs.space.coord_sys, inputs.space.is_periodic.data());
        // ============================================================

        // STANDARD ELEMENT ===========================================
        this->std_elem.define(geom.CellSize(), inputs.dG.space_p, inputs.dG.space_p+1);
        // ============================================================
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
                // -----------------------------------------------------
                // PARTIAL ELEMENTS ------------------------------------
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
};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif