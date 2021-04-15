//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Mesh.cpp
 * \brief Contains implementations of functions for mesh objects.
*/

#include <AMReX_dG_Mesh.H>

namespace amrex
{
namespace dG
{


// STRUCTURED MESH CLASS ##############################################
    // INITIALIZATION =================================================
    /**
     * \brief Read structured mesh parameters from input file.
    */
    void StructuredMesh::read_input_file(const ParmParse & pp)
    {
        // VARIABLES --------------------------------------------------
        int tmp_int;
        // ------------------------------------------------------------

        // READ PARAMETERS --------------------------------------------
        tmp_int = 0;
        pp.query("embedded_geometry_is_active", tmp_int);
        this->params.embedded_geometry_is_active = (tmp_int != 0);

        if (this->params.embedded_geometry_is_active)
        {
            pp.get("embedded_geometry_defined_by", this->params.embedded_geometry_defined_by);
        }

        pp.query("box_pruning_is_active", tmp_int);
        this->params.box_pruning_is_active = (tmp_int != 0);

        pp.query("quadrature_order_regular_elements", this->params.quadrature_order_regular_elements);
        pp.query("quadrature_order_cut_elements", this->params.quadrature_order_cut_elements);
        // ------------------------------------------------------------
    }
    // ================================================================


    // READERS ========================================================
    /**
     * \brief Return true if the embedded geometries is active.
     *
     * \return params.embedded_geometry_is_active.
     *
    */
    bool StructuredMesh::uses_embedded_geometries() const
    {
        return (this->params.embedded_geometry_is_active);
    }

    /**
     * \brief Return true if the embedded geometries is defined by a level set function.
     *
     * \return true if params.embedded_geometry_defined_by == "levelset", false otherwise.
     *
    */
    bool StructuredMesh::uses_levelset() const
    {
        return (this->params.embedded_geometry_defined_by.compare("levelset") == 0);
    }

    /**
     * \brief Return true if the box pruning is active.
     *
     * \return params.box_pruning_is_active.
     *
    */
    bool StructuredMesh::uses_box_pruning() const
    {
        return (this->params.box_pruning_is_active);
    }
    // ================================================================


    // COMPUTE GEOMETRICAL PROPERTIES =================================
    /**
     * \brief Compute the volume and surface of each domain.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] n_domains: number of domains to be considered.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[out] computed_volume: computed_volume[dom] will contain the computed volume of the dom-th
     *                              domain.
     * \param[out] computed_volume_div: computed_volume[dom] will contain the computed volume of the
     *                                  dom-th domain using the boundary integration points.
     * \param[out] computed_surface: computed_surface[dom] will contain the computed surface of the
     *                               dom-th domain.
     *
    */
    void StructuredMesh::eval_volumes_and_surfaces(const Geometry & geom,
                                                   const int n_domains,
                                                   const iMultiFab & mask,
                                                   Real * computed_volume,
                                                   Real * computed_volume_div,
                                                   Real * computed_surface) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // QUADRATURE CHECK MULTIFAB
        const int n_comp_per_dom = 3;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = this->cell_dom_quad_dev_mem.data();
        Array<Real const *, AMREX_SPACEDIM> cell_bou_quad_mem_ptr = {AMREX_D_DECL(this->cell_bou_quad_dev_mem[0].data(),
                                                                                  this->cell_bou_quad_dev_mem[1].data(),
                                                                                  this->cell_bou_quad_dev_mem[2].data())};
        Real const * cell_int_bou_quad_mem_ptr = this->cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = this->cell_dom_quad_host_mem.data();
        Array<Real const *, AMREX_SPACEDIM> cell_bou_quad_mem_ptr = {AMREX_D_DECL(this->cell_bou_quad_host_mem[0].data(),
                                                                                  this->cell_bou_quad_host_mem[1].data(),
                                                                                  this->cell_bou_quad_host_mem[2].data())};
        Real const * cell_int_bou_quad_mem_ptr = this->cell_int_bou_quad_dev_mem.data();
#endif

        MultiFab dom_quad;
        Array<MultiFab, AMREX_SPACEDIM> bou_quad;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        dom_quad.define(this->cell_type.boxarray, this->cell_type.distributionMap, n_comp_per_dom*n_domains, this->dom_data_n_grow);
        dom_quad = 0.0;
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            bou_quad[dim].define(this->cell_bou_type[dim].boxarray, this->cell_bou_type[dim].distributionMap, n_comp_per_dom*n_domains, this->bou_data_n_grow);
            bou_quad[dim] = 0.0;
        }
        // ------------------------------------------------------------

        // COMPUTE VOLUME/SURFACE OF EACH DOMAIN ----------------------
        for (MFIter mfi(dom_quad); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            Array4<Real> const & dom_quad_fab = dom_quad.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)))
                {
                    // LOCAL PARAMETERS
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    
                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                    Real integrand;
                    Real cell_volume;

                    // INITIALIZATION
                    cell_volume = 0.0;
                    
                    // DOMAIN QUADRATURE
                    {
                        const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                        const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                        const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < dom_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                                x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                                x[2] = x_ptr[x_pos+2]+cell_lo[2];
                            )
                            w = x_ptr[x_pos+AMREX_SPACEDIM];

                            integrand = 1.0;
                            cell_volume += integrand*w;
                        }
                    }

                    // STORE COMPUTED VALUES
                    dom_quad_fab(i,j,k,n_comp_per_dom*dom) = cell_volume;
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // SUM CONTRIBUTION -------------------------------------------
        for (int dom = 0; dom < n_domains; ++dom)
        {
            computed_volume[dom] = dom_quad.sum(n_comp_per_dom*dom);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
// ####################################################################


} // namespace dG
} // namespace amrex