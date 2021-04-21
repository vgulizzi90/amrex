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


// MESH CLASS #########################################################
    // INITIALIZATION =================================================
    /**
     * \brief Read structured mesh parameters from input file.
    */
    void Mesh::read_input_file(const ParmParse & pp)
    {
        // VARIABLES --------------------------------------------------
        int tmp_int;
        // ------------------------------------------------------------

        // READ PARAMETERS --------------------------------------------
        tmp_int = 0;
        pp.query("embedded_geometry_is_active", tmp_int);
        this->params.embedded_geometry_is_active = (tmp_int > 0);

        if (this->params.embedded_geometry_is_active)
        {
            pp.get("embedded_geometry_defined_by", this->params.embedded_geometry_defined_by);
        }

        pp.query("box_pruning_is_active", tmp_int);
        this->params.box_pruning_is_active = (tmp_int > 0);

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
    bool Mesh::uses_embedded_geometries() const
    {
        return (this->params.embedded_geometry_is_active);
    }

    /**
     * \brief Return true if the embedded geometries is defined by a level set function.
     *
     * \return true if params.embedded_geometry_defined_by == "level_set", false otherwise.
     *
    */
    bool Mesh::uses_level_set() const
    {
        return (this->params.embedded_geometry_defined_by.compare("level_set") == 0);
    }

    /**
     * \brief Return true if the box pruning is active.
     *
     * \return params.box_pruning_is_active.
     *
    */
    bool Mesh::uses_box_pruning() const
    {
        return (this->params.box_pruning_is_active);
    }
    // ================================================================


    // GHOST CELLS ====================================================
    /**
     * \brief Label those cells that live in the ghost rows.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] n_domains: number of domains to be considered.
     *
    */
    void Mesh::label_ghost_cells(const Geometry & geom, const int n_domains)
    {
        // VARIABLES --------------------------------------------------
        cMultiFab ghostbuster(this->cell_type.boxarray, this->cell_type.distributionMap, 1, this->dom_data_n_grow);
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        ghostbuster = 1;
        // ------------------------------------------------------------

        // MARK THE CELLS IN THE VALID BOXES --------------------------
        for (MFIter mfi(ghostbuster); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            Array4<char> const & ghostbuster_fab = ghostbuster.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                ghostbuster_fab(i,j,k) = 0;
            });
            Gpu::synchronize();
        }
        ghostbuster.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // LABEL THE GHOST CELLS --------------------------------------
        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();
            Array4<char const> const & ghostbuster_fab = ghostbuster.array(mfi);
            Array4<short> const & cell_type_fab = this->cell_type.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                if (ghostbuster_fab(i,j,k) == 1)
                {
                    cell_type_fab(i,j,k,CELL_TYPE(dom)) += __DG_CELL_GHOST__;
                }
            });
            Gpu::synchronize();
        }
        this->cell_type.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
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
    void Mesh::eval_volumes_and_surfaces(const Geometry & geom,
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
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        MultiFab dom_quad, int_bou_quad, dom_quad_div;
        Array<MultiFab, AMREX_SPACEDIM> bou_quad;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        dom_quad.define(this->cell_type.boxarray, this->cell_type.distributionMap, n_domains, this->dom_data_n_grow);
        dom_quad = 0.0;
        int_bou_quad.define(this->cell_type.boxarray, this->cell_type.distributionMap, n_domains, this->dom_data_n_grow);
        int_bou_quad = 0.0;

        dom_quad_div.define(this->cell_type.boxarray, this->cell_type.distributionMap, n_domains, this->dom_data_n_grow);
        dom_quad_div = 0.0;
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            bou_quad[dim].define(this->cell_bou_type[dim].boxarray, this->cell_bou_type[dim].distributionMap, n_domains, this->bou_data_n_grow);
            bou_quad[dim] = 0.0;
        }
        // ------------------------------------------------------------

        // COMPUTE VOLUME/SURFACE OF EACH DOMAIN ----------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = this->cell_dom_quad_dev_mem.data();
        Real const * cell_int_bou_quad_mem_ptr = this->cell_int_bou_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = this->cell_dom_quad_host_mem.data();
        Real const * cell_int_bou_quad_mem_ptr = this->cell_int_bou_quad_host_mem.data();
#endif
        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = this->cell_dom_quad_info.array(mfi);
            Array4<long const> const & cell_int_bou_quad_info_fab = this->cell_int_bou_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            Array4<Real> const & dom_quad_fab = dom_quad.array(mfi);
            Array4<Real> const & int_bou_quad_fab = int_bou_quad.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)))
                {
                    // LOCAL PARAMETERS
                    /*
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    */
                    
                    // LOCAL VARIABLES
                    Real /*x[AMREX_SPACEDIM],*/ w;
                    Real integrand;
                    Real cell_volume, cell_surface;

                    // INITIALIZATION
                    cell_volume = 0.0;
                    cell_surface = 0.0;
                    
                    // DOMAIN QUADRATURE
                    {
                        const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                        const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                        const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < dom_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            /*
                            AMREX_D_TERM
                            (
                                x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                                x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                                x[2] = x_ptr[x_pos+2]+cell_lo[2];
                            )
                            */
                            w = x_ptr[x_pos+AMREX_SPACEDIM];

                            integrand = 1.0;
                            cell_volume += integrand*w;
                        }
                    }

                    // INTERNAL BOUNDARY QUADRATURE
                    {
                        const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                        const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                        const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                            /*
                            AMREX_D_TERM
                            (
                                x[0] = x_ptr[x_pos+0];,
                                x[1] = x_ptr[x_pos+1];,
                                x[2] = x_ptr[x_pos+2];
                            )
                            */
                            w = x_ptr[x_pos+AMREX_SPACEDIM];

                            integrand = 1.0;
                            cell_surface += integrand*w;
                        }
                    }

                    // STORE COMPUTED VALUES
                    dom_quad_fab(i,j,k,dom) = cell_volume;
                    int_bou_quad_fab(i,j,k,dom) = cell_surface;
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // COMPUTE VOLUME OF EACH DOMAIN (USING DIVERGENCE THEOREM) ---
        // INTERNAL BOUNDARY
        for (MFIter mfi(this->cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_int_bou_quad_info_fab = this->cell_int_bou_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);

            Array4<Real> const & dom_quad_div_fab = dom_quad_div.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)))
                {
                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], w, un[AMREX_SPACEDIM];
                    Real integrand;
                    Real cell_volume_div;

                    // INITIALIZATION
                    cell_volume_div = 0.0;

                    // INTERNAL BOUNDARY QUADRATURE
                    {
                        const int bou_Nq = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_NQ(dom));
                        const long pos = cell_int_bou_quad_info_fab(i,j,k,CELL_INT_BOU_QUAD_POS(dom));
                        const Real * x_ptr = &cell_int_bou_quad_mem_ptr[pos];

                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*q;
                            AMREX_D_TERM
                            (
                                x[0] = x_ptr[x_pos+0];,
                                x[1] = x_ptr[x_pos+1];,
                                x[2] = x_ptr[x_pos+2];
                            )
                            w = x_ptr[x_pos+AMREX_SPACEDIM];
                            AMREX_D_TERM
                            (
                                un[0] = x_ptr[x_pos+AMREX_SPACEDIM+1+0];,
                                un[1] = x_ptr[x_pos+AMREX_SPACEDIM+1+1];,
                                un[2] = x_ptr[x_pos+AMREX_SPACEDIM+1+2];
                            )

                            integrand = x[0]*un[0];
                            cell_volume_div += integrand*w;
                        }
                    }

                    // STORE COMPUTED VALUES
                    dom_quad_div_fab(i,j,k,dom) = cell_volume_div;
                }
            });
            Gpu::synchronize();
        }

        // CELL BOUNDARIES
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            // UNIT NORMAL
            GpuArray<Real, AMREX_SPACEDIM> un = {AMREX_D_DECL(0.0, 0.0, 0.0)};
            un[dir] = +1.0;

#ifdef AMREX_USE_GPU
            Real const * cell_bou_quad_mem_ptr = this->cell_bou_quad_dev_mem[dir].data();
#else
            Real const * cell_bou_quad_mem_ptr = this->cell_bou_quad_host_mem[dir].data();
#endif
            for (MFIter mfi(this->cell_bou_type[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                Array4<long const> const & cell_bou_quad_info_fab = this->cell_bou_quad_info[dir].array(mfi);
                Array4<short const> const & cell_type_fab = this->cell_type.array(mfi);
                
                Array4<Real> const & bou_quad_fab = bou_quad[dir].array(mfi);

                ParallelFor(bx, n_domains,
                [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
                {
                    // LOCAL PARAMETERS
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;
                    bool m_ghost, p_ghost;
                    Real x[AMREX_SPACEDIM], w;
                    Real integrand;
                    Real cell_volume_div;

                    // INITIALIZATION
                    cell_volume_div = 0.0;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    m_ghost = CELL_IS_GHOST(cell_type_fab(mi,mj,mk,CELL_TYPE(dom)));
                    p_ghost = CELL_IS_GHOST(cell_type_fab(pi,pj,pk,CELL_TYPE(dom)));

                    if (m_ghost ^ p_ghost)
                    {
                        // EVAL THE INTEGRAL
                        for (int q = 0; q < bou_Nq; ++q)
                        {
                            const long x_pos = (AMREX_SPACEDIM+1)*q;
                            AMREX_D_TERM
                            (
                                x[0] = x_ptr[x_pos+0]+face_lo[0];,
                                x[1] = x_ptr[x_pos+1]+face_lo[1];,
                                x[2] = x_ptr[x_pos+2]+face_lo[2];
                            )
                            w = x_ptr[x_pos+AMREX_SPACEDIM];

                            integrand = x[0]*un[0];
                            cell_volume_div += integrand*w;
                        }

                        if (m_ghost)
                        {
                            cell_volume_div *= -1.0;
                        }
                    }

                    // STORE COMPUTED VALUES
                    bou_quad_fab(fi,fj,fk,dom) = cell_volume_div;
                });
                Gpu::synchronize();
            }
        }
        // ------------------------------------------------------------

        // SUM CONTRIBUTION -------------------------------------------
        for (int dom = 0; dom < n_domains; ++dom)
        {
            computed_volume[dom] = dom_quad.sum(dom);
            computed_volume_div[dom] = dom_quad_div.sum(dom);
            computed_surface[dom] = int_bou_quad.sum(dom);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
// ####################################################################


} // namespace dG
} // namespace amrex