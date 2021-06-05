#define __DG_N_NBRS_N_COMP_PER_DOM__ 2

#define __DG_NNZ_N_COMP_PER_DOM__ 3


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int N_NBRS_VALID_ID(const int dom)
{
    return (dom*__DG_N_NBRS_N_COMP_PER_DOM__+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int N_NBRS_GREATER_ID(const int dom)
{
    return (dom*__DG_N_NBRS_N_COMP_PER_DOM__+1);
}

// NON-ZEROS
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int NNZ(const int dom)
{
    return (dom*__DG_NNZ_N_COMP_PER_DOM__+0);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int NNZ_OD_VALID_NBR_ID(const int dom)
{
    return (dom*__DG_NNZ_N_COMP_PER_DOM__+1);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int NNZ_OD_GREATER_NBR_ID(const int dom)
{
    return (dom*__DG_NNZ_N_COMP_PER_DOM__+2);
}


/**
     * \brief Construct an iMultiFab with the number of neighbors for each element.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void get_n_nbrs(const Geometry & geom, const Mesh & mesh, const iMultiFab & mask, iMultiFab & n_nbrs, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        const int n_domains = ibvp.get_number_of_domains();

        const int n_nbrs_n_comp = n_domains*__DG_N_NBRS_N_COMP_PER_DOM__;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        iMultiFab nbr_ids;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        n_nbrs.define(mesh.ba, mesh.dm, n_nbrs_n_comp, mesh.dom_data_n_grow);
        n_nbrs = 0;
        // ------------------------------------------------------------

        // GATHER NEIGHBORS IDS ---------------------------------------
        this->get_nbr_ids(geom, mesh, nbr_ids, ibvp);
        // ------------------------------------------------------------

        // COUNT THE NUMBER OF NEIGHBORS ------------------------------
        for (MFIter mfi(this->elm_id); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int const> const & nbr_ids_fab = nbr_ids.array(mfi);

            Array4<int> const & n_nbrs_fab = n_nbrs.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);

                // LOCAL VARIABLES
                int n_nbrs_valid_id;
                int n_nbrs_greater_id;

                // INIT NUMBER OF NEIGHBORS
                n_nbrs_valid_id = 0;
                n_nbrs_greater_id = 0;
/*
Print() << "id(" << i << "," << j << "," << k << ") " << id << std::endl;
*/
                // DIRECT NEIGHBORS
                if (elm_is_valid)
                {
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        const int n_id = nbr_ids_fab(i,j,k,NBR_ID(dom, b));
/*
if (i == 3 && j == 1)
{
Print() << "n_id(" << i << "," << j << "," << k << "," << b << ") " << n_id << std::endl;
}
*/
                        if ((n_id != id) && (n_id >= 0))
                        {
                            n_nbrs_valid_id += 1;
                        }
                        if (n_id > id)
                        {
                            n_nbrs_greater_id += 1;
                        }
                    }
                }

                // NEIGHBORS OF MERGING CELLS
                if (elm_is_extended)
                {
                    for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                    {
                        const int nbr_i = i+tables::extended_stencil_i[n];
                        const int nbr_j = j+tables::extended_stencil_j[n];
                        const int nbr_k = k+tables::extended_stencil_k[n];
                        const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        
                        if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                        {
                            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                            {
                                const int n_id = nbr_ids_fab(nbr_i,nbr_j,nbr_k,NBR_ID(dom, b));
/*
if (i == 3 && j == 1)
{
Print() << "n_id(" << nbr_i << "," << nbr_j << "," << nbr_k << "," << b << ") " << n_id << std::endl;
}
*/

                                if ((n_id != id) && (n_id >= 0))
                                {
                                    n_nbrs_valid_id += 1;
                                }
                                if (n_id > id)
                                {
                                    n_nbrs_greater_id += 1;
                                }
                            }
                        }
                    }
                }
/*
for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
{
    Print() << " + n_id(" << 2 << "," << 1 << "," << 0 << "," << b << ") " << nbr_ids_fab(2,1,0,NBR_ID(0, b)) << std::endl;
}
Print() << "n_nbrs_valid_id: " << n_nbrs_valid_id << ", n_nbrs_greater_id: " << n_nbrs_greater_id << std::endl << std::endl;
*/

                // NEIGHBORS ON OTHER PHASE
                //...

                // STORE INFO
                n_nbrs_fab(i,j,k,N_NBRS_VALID_ID(dom)) = n_nbrs_valid_id;
                n_nbrs_fab(i,j,k,N_NBRS_GREATER_ID(dom)) = n_nbrs_greater_id;
            }
        }
        n_nbrs.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }



    /**
     * \brief Construct an iMultiFab with the number of non-zeros for each element.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void get_nnz(const Geometry & geom, const Mesh & mesh, const iMultiFab & mask, iMultiFab & nnz, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

        const int nnz_n_comp = n_domains*__DG_NNZ_N_COMP_PER_DOM__;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        iMultiFab nbr_ids;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        nnz.define(mesh.ba, mesh.dm, nnz_n_comp, mesh.dom_data_n_grow);
        nnz = 0;
        // ------------------------------------------------------------

        // GATHER NEIGHBORS IDS ---------------------------------------
        this->get_nbr_ids(geom, mesh, nbr_ids, ibvp);
        // ------------------------------------------------------------

        // COUNT THE NUMBER OF NEIGHBORS ------------------------------
        for (MFIter mfi(this->elm_id); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int const> const & nbr_ids_fab = nbr_ids.array(mfi);

            Array4<int> const & nnz_fab = nnz.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);

                // LOCAL VARIABLES
                int nnz_od_valid_nbr_id;
                int nnz_od_greater_nbr_id;

                // UNKNOWN FIELDS
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int elm_Nup = Nu*sNp;
                const int nbr_Nup = elm_Nup;

                // INIT NUMBER OF NEIGHBORS
                nnz_od_valid_nbr_id = 0;
                nnz_od_greater_nbr_id = 0;

                // SELF CONTRIBUTION
                // We store just the number of columns
                if (cell_is_not_masked && elm_is_valid)
                {
                    nnz_fab(i,j,k,NNZ(dom)) = elm_Nup;
                }
                else
                {
                    nnz_fab(i,j,k,NNZ(dom)) = 0;
                }

                // DIRECT NEIGHBORS
                // We store just the number of columns
                if (cell_is_not_masked && elm_is_valid)
                {
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        const int n_id = nbr_ids_fab(i,j,k,NBR_ID(dom, b));

                        if ((n_id != id) && (n_id >= 0))
                        {
                            nnz_od_valid_nbr_id += nbr_Nup;
                        }
                        if (n_id > id)
                        {
                            nnz_od_greater_nbr_id += nbr_Nup;
                        }
                    }
                }

                // NEIGHBORS OF MERGING CELLS
                // We store just the number of columns
                if (cell_is_not_masked && elm_is_extended)
                {
                    for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                    {
                        const int nbr_i = i+tables::extended_stencil_i[n];
                        const int nbr_j = j+tables::extended_stencil_j[n];
                        const int nbr_k = k+tables::extended_stencil_k[n];
                        const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        
                        if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                        {
                            for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                            {
                                const int n_id = nbr_ids_fab(nbr_i,nbr_j,nbr_k,NBR_ID(dom, b));

                                if ((n_id != id) && (n_id >= 0))
                                {
                                    nnz_od_valid_nbr_id += nbr_Nup;
                                }
                                if (n_id > id)
                                {
                                    nnz_od_greater_nbr_id += nbr_Nup;
                                }
                            }
                        }
                    }
                }

                // NEIGHBORS ON OTHER PHASE
                //...

                // STORE INFO
                nnz_fab(i,j,k,NNZ_OD_VALID_NBR_ID(dom)) = nnz_od_valid_nbr_id;
                nnz_fab(i,j,k,NNZ_OD_GREATER_NBR_ID(dom)) = nnz_od_greater_nbr_id;
            }
        }
        nnz.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }







    /**
     * \brief Construct an iMultiFab with the ids of the neighbors of each cell.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void get_cell_nbr_ids(const Geometry & geom, const Mesh & mesh, iMultiFab & cell_nbr_ids, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        const int n_domains = ibvp.get_number_of_domains();

        const int cell_nbr_ids_n_comp = n_domains*__DG_CELL_NBR_ID_N_COMP_PER_DOM__;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        cell_nbr_ids.define(mesh.ba, mesh.dm, cell_nbr_ids_n_comp, mesh.dom_data_n_grow);
        // ------------------------------------------------------------

        // GET IDS OF THE NEIGHBORS OF EACH CELL ----------------------
        for (MFIter mfi(cell_nbr_ids); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int> const & cell_nbr_ids_fab = cell_nbr_ids.array(mfi);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                Array4<short const> const & elm_bou_type_fab = this->elm_bou_type[dir].array(mfi);

                ParallelFor(bx, n_domains,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
                {
                    for (int b = 2*dir; b < 2*(dir+1); ++b)
                    {
                        // NEIGHBOR CELL
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        // GRID FACE
                        int fi, fj, fk;
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        // ELEMENT BOUNDARY TYPE
                        const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                        if (ELM_BOU_IS_VALID(ebtype))
                        {
                            cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b)) = elm_id_fab(nbr_i,nbr_j,nbr_k,dom);
                        }
                        else
                        {
                            cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b)) = -1;
                        }
                        /* DEBUG
                        if (i == 2 && j == 1)
                        {
                        const int id = elm_id_fab(i,j,k,dom);
                        Print() << "1 - id(" << i << "," << j << "," << k << ") " << id << ", ebtype: (" << fi << "," << fj << "," << fk << "): " << ebtype << " is valid? " << (ELM_BOU_IS_VALID(ebtype) ? "yes" : "no") << ", cell_nbr_ids_fab: " << cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom,b)) << std::endl;
                        }
                        */
                    }

                });
                Gpu::synchronize();
            }
        }
        cell_nbr_ids.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }

    /**
     * \brief Construct an iMultiFab with the ids and dofs of the neighbors of each element.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void get_elm_nbr_info(const Geometry & geom, const Mesh & mesh, const iMultiFab & mask, iMultiFab & elm_nbr_info, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

        const int elm_nbr_info_n_comp = __DG_ELM_NBR_INFO_N_COMP_PER_DOM__*n_domains;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        iMultiFab cell_nbr_ids;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        elm_nbr_info.define(mesh.ba, mesh.dm, elm_nbr_info_n_comp, mesh.dom_data_n_grow);
        elm_nbr_info = -1;
        // ------------------------------------------------------------

        // GATHER NEIGHBORS IDS FROM THE CELLS' POINT OF VIEW ---------
        this->get_cell_nbr_ids(geom, mesh, cell_nbr_ids, ibvp);
        // ------------------------------------------------------------

        // STORE NEIGHBORS FROM THE ELEMENTS' POINT OF VIEW -----------
        for (MFIter mfi(this->elm_id); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int const> const & cell_nbr_ids_fab = cell_nbr_ids.array(mfi);

            Array4<int> const & elm_nbr_info_fab = elm_nbr_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);

                // LOCAL VARIABLES
                bool add_nbr_id;
                int elm_n_nbrs;
                int elm_nbrs[__DG_ELM_MAX_N_NBR_PER_DOM__];

                // UNKNOWN FIELDS
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int elm_n_dof = Nu*sNp;

                // INITIALIZATION
                elm_n_nbrs = 0;
                std::fill(elm_nbrs, elm_nbrs+__DG_ELM_MAX_N_NBR_PER_DOM__, 0);
                
                if (cell_is_not_masked && elm_is_valid)
                {
                    // DIRECT NEIGHBORS
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        const int n_id = cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b));

                        add_nbr_id =               ((n_id != id) && (n_id >= 0));
                        add_nbr_id = add_nbr_id && (std::count(elm_nbrs, elm_nbrs+elm_n_nbrs, n_id) == 0);

                        if (add_nbr_id)
                        {
                            elm_nbrs[elm_n_nbrs] = n_id;
                            elm_n_nbrs += 1;
                        }
                    }

                    // MERGING CELLS NBRS
                    if (elm_is_extended)
                    {
                        for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                        {
                            const int nbr_i = i+tables::extended_stencil_i[n];
                            const int nbr_j = j+tables::extended_stencil_j[n];
                            const int nbr_k = k+tables::extended_stencil_k[n];
                            const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                            
                            if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                            {
                                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                                {
                                    const int n_id = cell_nbr_ids_fab(nbr_i,nbr_j,nbr_k,CELL_NBR_ID(dom, b));

                                    add_nbr_id =               ((n_id != id) && (n_id >= 0));
                                    add_nbr_id = add_nbr_id && (std::count(elm_nbrs, elm_nbrs+elm_n_nbrs, n_id) == 0);

                                    if (add_nbr_id)
                                    {
                                        elm_nbrs[elm_n_nbrs] = n_id;
                                        elm_n_nbrs += 1;
                                    }
                                }
                            }
                        }

                        // NEIGHBORS ON OTHER PHASE
                        //...
                    }

                    // SORT THE NEIGHBORS
                    // Here we assume all elements have the same number
                    // of degrees of freedom. See also below.
                    std::sort(elm_nbrs, elm_nbrs+elm_n_nbrs);

                    /* DEBUG
                    Print() << "elm_nbrs(" << i << "," << j << "," << k << ") (elm_n_nbrs = " << elm_n_nbrs << "): "; io::print_int_array_2d(1, elm_n_nbrs, elm_nbrs);
                    */

                    // STORE INFO
                    elm_nbr_info_fab(i,j,k,ELM_N_NBRS(dom)) = elm_n_nbrs;
                    for (int n = 0; n < elm_n_nbrs; ++n)
                    {
                        elm_nbr_info_fab(i,j,k,ELM_NBR_ID(dom, n)) = elm_nbrs[n];
                        
                        // We assume all neighbors have the same number
                        // of degrees of freedom.
                        elm_nbr_info_fab(i,j,k,ELM_NBR_N_DOF(dom, n)) = elm_n_dof;
                        
                        //Print() << " - ELM_NBR_ID(dom, " << n << "): " << ELM_NBR_ID(dom, n) << ", ELM_NBR_N_DOF(dom, " << n << "): " << ELM_NBR_N_DOF(dom, n) << std::endl;
                    }
                }
            }
        }
        elm_nbr_info.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // COPY INFO TO SMALL ELEMENTS --------------------------------
        for (MFIter mfi(this->elm_id); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);

            Array4<int> const & elm_nbr_info_fab = elm_nbr_info.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_small = ELM_IS_SMALL(etype);

                if (elm_is_small)
                {
                    int bf_i, bf_j, bf_k, bf_elm_n_nbrs;
                    BF_CELL(i, j, k, etype, bf_i, bf_j, bf_k);

                    for (int n = dom*__DG_ELM_NBR_INFO_N_COMP_PER_DOM__; n < (dom+1)*__DG_ELM_NBR_INFO_N_COMP_PER_DOM__; ++n)
                    {
                        elm_nbr_info_fab(i,j,k,n) = elm_nbr_info_fab(bf_i,bf_j,bf_k,n);
                    }
                }
            });
            Gpu::synchronize();
        }
        elm_nbr_info.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }

    /**
     * \brief Add the non-zeros to a real symmetric matrix stored in CSR format.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[inout] csr: .
     * \param[inout] dom_row: .
     * \param[inout] bou_nnz_offset: .
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename CSR, typename IBVP>
    void add_nnz_to_real_sym_csr_matrix(const Geometry & geom, const Mesh & mesh, const iMultiFab & mask,
                                        CSR & csr, iMultiFab & csr_dom_info, Array<iMultiFab, AMREX_SPACEDIM> & csr_bou_info,
                                        const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // FACE-CENTERED BOX ARRAYS
#if (AMREX_SPACEDIM == 1)
        const BoxArray fc_ba[1] = {convert(mesh.ba, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(mesh.ba, IntVect(1,0)),
                                   convert(mesh.ba, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(mesh.ba, IntVect(1,0,0)),
                                   convert(mesh.ba, IntVect(0,1,0)),
                                   convert(mesh.ba, IntVect(0,0,1))};
#endif

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

        // CSR INFO
        const int csr_dom_info_n_comp = __DG_CSR_DOM_INFO_N_COMP_PER_DOM__*n_domains;
        const int csr_bou_info_n_comp = __DG_CSR_BOU_INFO_N_COMP_PER_DOM__*n_domains;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        iMultiFab elm_nbr_info;

        int rA;
        // ------------------------------------------------------------

        // GATHER NEIGHBORS INFO --------------------------------------
        this->get_elm_nbr_info(geom, mesh, mask, elm_nbr_info, ibvp);
        // ------------------------------------------------------------

        // INIT THE CSR INFO MULTIFABS --------------------------------
        csr_dom_info.define(mesh.ba, mesh.dm, csr_dom_info_n_comp, this->dom_data_n_grow);
        csr_dom_info = -1;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            csr_bou_info[dir].define(fc_ba[dir], mesh.dm, csr_bou_info_n_comp, this->bou_data_n_grow);
            csr_bou_info[dir] = -1;
        }
        // ------------------------------------------------------------

        // LAST ROW INFO ----------------------------------------------
        rA = csr.ia.size()-1;
        // ------------------------------------------------------------

        // SET THE LOCATION OF THE ELEMENT IN THE ENTIRE SYSTEM -------
        for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int const> const & elm_nbr_info_fab = elm_nbr_info.array(mfi);

            Array4<int> const & csr_dom_info_fab = csr_dom_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);
                const int elm_n_nbrs = elm_nbr_info_fab(i,j,k,ELM_N_NBRS(dom));   
            }
        }
        csr_dom_info.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
exit(-1);
        // ADD THE NON-ZEROS ------------------------------------------
        for (MFIter mfi(csr_dom_info); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int const> const & elm_nbr_info_fab = elm_nbr_info.array(mfi);

            Array4<int> const & csr_dom_info_fab = csr_dom_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const bool cell_is_not_masked = CELL_IS_NOT_MASKED(mask_fab(i,j,k));
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);
                const int elm_n_nbrs = elm_nbr_info_fab(i,j,k,ELM_N_NBRS(dom));

                if (cell_is_not_masked && elm_is_valid)
                {
                    // LOCAL VARIABLES
                    int elm_nbrs_id[__DG_ELM_MAX_N_NBR_PER_DOM__];
                    int elm_nbrs_n_dof[__DG_ELM_MAX_N_NBR_PER_DOM__];
                    int n_id;
                    bool found;
                    int row_nnz;

                    // UNKNOWN FIELDS
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                    const int Nu = u_hi-u_lo;
                    const int elm_Nup = Nu*sNp;

                    // GATHER INFO
                    for (int n = 0; n < elm_n_nbrs; ++n)
                    {
                        elm_nbrs_id[n] = elm_nbr_info_fab(i,j,k,ELM_NBR_ID(dom, n));
                        elm_nbrs_n_dof[n] = elm_nbr_info_fab(i,j,k,ELM_NBR_N_DOF(dom, n));
                    }

                    /* DEBUG
                    Print() << "elm_nbrs_id(" << i << "," << j << "," << k << "): "; io::print_int_array_2d(1, elm_n_nbrs, elm_nbrs_id);
                    Print() << "elm_nbrs_n_dof(" << i << "," << j << "," << k << "): "; io::print_int_array_2d(1, elm_n_nbrs, elm_nbrs_n_dof);
                    */

                    // UPDATE CSR INFO
                    csr_dom_info_fab(i,j,k,CSR_DOM_INFO_ROW(dom)) = rA;
                    csr_dom_info_fab(i,j,k,CSR_DOM_INFO_NNZ_OFFSET(dom)) = csr.ia[rA];

                    for (int r = 0; r < elm_Nup; ++r)
                    {
                        row_nnz = csr.ia[rA];
                        
                        // Because of symmetry, we subtract r
                        csr.ja.resize(row_nnz+elm_Nup-r);
                        csr.A.resize(row_nnz+elm_Nup-r);
                        for (int c = r; c < elm_Nup; ++c)
                        {
                            csr.ja[row_nnz+c] = rA+c;
                            csr.A[row_nnz+c] = 0.0;
                        }
                        row_nnz += elm_Nup-r;
                        
                        // Because of symmetry, we consider neighbors
                        // with id greater than current element
                        for (int n = 0; n < elm_n_nbrs; ++n)
                        {
                            if (elm_nbrs_id[n] > id)
                            {
                                csr.ja.resize(row_nnz+elm_nbrs_n_dof[n]);
                                csr.A.resize(row_nnz+elm_nbrs_n_dof[n]);
                                for (int c = 0; c < elm_nbrs_n_dof[n]; ++c)
                                {
                                    csr.ja[row_nnz+c] = 0;
                                    csr.A[row_nnz+c] = 0.0;
                                }
                                row_nnz += elm_nbrs_n_dof[n];
                            }
                        }

                        csr.ia.push_back(row_nnz);

                        // NEXT ROW
                        rA += 1;
                    }
                }
            }
        }
        csr_dom_info.FillBoundary(geom.periodicity());

        // NEW NUMBER OF NON-ZEROS
        csr.nnz = csr.ia.back();

        // JA AND A
        //csr.ja.resize(csr.nnz);
        //csr.A.resize(csr.nnz);

        // B AND X
        csr.B.resize(rA);
        csr.X.resize(rA);
        // ------------------------------------------------------------

        // COPY INFO TO SMALL ELEMENTS --------------------------------
        for (MFIter mfi(this->elm_id); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);

            Array4<int> const & csr_dom_info_fab = csr_dom_info.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_small = ELM_IS_SMALL(etype);

                if (elm_is_small)
                {
                    int bf_i, bf_j, bf_k, bf_elm_n_nbrs;
                    BF_CELL(i, j, k, etype, bf_i, bf_j, bf_k);

                    csr_dom_info_fab(i,j,k,CSR_DOM_INFO_ROW(dom)) = csr_dom_info_fab(bf_i,bf_j,bf_k,CSR_DOM_INFO_ROW(dom));
                    csr_dom_info_fab(i,j,k,CSR_DOM_INFO_NNZ_OFFSET(dom)) = csr_dom_info_fab(bf_i,bf_j,bf_k,CSR_DOM_INFO_NNZ_OFFSET(dom));
                }
            });
            Gpu::synchronize();
        }
        csr_dom_info.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // STORE THE CSR BOUNDARY INFO --------------------------------
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            for (MFIter mfi(csr_bou_info[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                Array4<int const> const & mask_fab = mask.array(mfi);
                Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
                Array4<short const> const & elm_bou_type_fab = this->elm_bou_type[dir].array(mfi);
                Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
                Array4<int const> const & elm_nbr_info_fab = elm_nbr_info.array(mfi);

                Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
                Array4<int> const & csr_bou_info_fab = csr_bou_info[dir].array(mfi);

                for (int dom = 0; dom < n_domains; ++dom)
                for (int fk = lo.z; fk <= hi.z; ++fk)
                for (int fj = lo.y; fj <= hi.y; ++fj)
                for (int fi = lo.x; fi <= hi.x; ++fi)
                {
                    // ELEMENT BOUNDARY TYPE
                    const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                    if (ELM_BOU_IS_VALID(ebtype) && ELM_BOU_IS_NOT_WALL(ebtype))
                    {
                        // LOCAL VARIABLES
                        int mi, mj, mk, pi, pj, pk;
                        bool m_cell_is_masked, p_cell_is_masked;
                        short m_etype, p_etype;
                        int m_id, p_id, i, j, k;
                        int elm_n_nbrs, elm_n_dof;
                        int id, n_id, n, nnz_offset, r;
                        bool found;

                        // UNKNOWN FIELDS
                        int u_lo, u_hi;
                        ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                        const int Nu = u_hi-u_lo;
                        const int elm_Nup = Nu*sNp;

                        elm_n_dof = elm_Nup;

                        // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                        FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                        m_cell_is_masked = CELL_IS_MASKED(mask_fab(mi,mj,mk));
                        p_cell_is_masked = CELL_IS_MASKED(mask_fab(pi,pj,pk));
                        m_etype = elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                        p_etype = elm_type_fab(pi,pj,pk,ELM_TYPE(dom));
                        m_id = elm_id_fab(mi,mj,mk,dom);
                        p_id = elm_id_fab(pi,pj,pk,dom);
                        
                        /* DEBUG 
                        Print() << "face" << dir << "(" << fi << "," << fj << "," << fk << "," << dom << ") shared by: " << m_id << " and " << p_id << std::endl;
                        */

                        if (m_id < p_id)
                        {
                            id = m_id;
                            n_id = p_id;
                            i = mi;
                            j = mj;
                            k = mk;
                        }
                        else if (m_id > p_id)
                        {
                            id = p_id;
                            n_id = m_id;
                            i = pi;
                            j = pj;
                            k = pk;
                        }
                        else if (m_id == p_id)
                        {
                            std::string msg;
                            msg  = "\n";
                            msg +=  "ERROR: AMReX_dG_Solution.H - add_nnz_to_real_sym_csr_matrix\n";
                            msg += "| Elements that are neighbor with themselves are currently not supported.\n";
                            Abort(msg);
                        }

                        // GATHER INFO
                        elm_n_nbrs = elm_nbr_info_fab(i,j,k,ELM_N_NBRS(dom));
                        n = 0;
                        nnz_offset = elm_n_dof;
                        found = false;
                        while ((n < elm_n_nbrs) && (!found))
                        {
                            if (elm_nbr_info_fab(i,j,k,ELM_NBR_ID(dom, n)) == n_id)
                            {
                                found = true;
                            }
                            else
                            {
                                n += 1;
                                nnz_offset += elm_nbr_info_fab(i,j,k,ELM_NBR_N_DOF(dom, n));
                            }
                        }
                        if (!found)
                        {
                            /* DEBUG
                            int elm_nbrs_id[__DG_ELM_MAX_N_NBR_PER_DOM__];
                            for (int nn = 0; nn < elm_n_nbrs; ++nn)
                            {
                                elm_nbrs_id[nn] = elm_nbr_info_fab(i,j,k,ELM_NBR_ID(dom, nn));
                            }
                            Print() << "elm_nbrs_id(" << i << "," << j << "," << k << "): "; io::print_int_array_2d(1, elm_n_nbrs, elm_nbrs_id);
                            */

                            std::string msg;
                            msg  = "\n";
                            msg +=  "ERROR: AMReX_dG_Solution.H - add_nnz_to_real_sym_csr_matrix\n";
                            msg += "| Inconsistent neighbors information along direction: "+std::to_string(dir)+" for face: "+std::to_string(fi)+", "+std::to_string(fj)+", "+std::to_string(fk)+".\n";
                            Abort(msg);
                        }

                        // UPDATE CSR INFO
                        r = csr_dom_info_fab(i,j,k,CSR_DOM_INFO_ROW(dom));
                        csr_bou_info_fab(fi,fj,fk,CSR_BOU_INFO_NNZ_OFFSET(dom)) = csr.ia[r]+nnz_offset;  

                        /* DEBUG
                        Print() << "face" << dir << "(" << fi << "," << fj << "," << fk << "," << dom << ") row: " << r << ", csr_bou_info: " << csr.ia[r]+nnz_offset << std::endl;
                        */

                    }
                }
            }
        }
        // ------------------------------------------------------------

        // CSR INTERNAL BOUNDARY INFO ---------------------------------
        //...
        // ------------------------------------------------------------
    }








    /**
     * \brief Construct an iMultiFab with the ids of the neighbors of each cell.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[out] cell_nbr_ids: a iMultiFab that will contain the output information. 
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void get_cell_nbr_ids(const Geometry & geom, const Mesh & mesh, iMultiFab & cell_nbr_ids, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        const int n_domains = ibvp.get_number_of_domains();

        const int cell_nbr_ids_n_comp = n_domains*__DG_CELL_NBR_ID_N_COMP_PER_DOM__;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        cell_nbr_ids.define(mesh.ba, mesh.dm, cell_nbr_ids_n_comp, mesh.dom_data_n_grow);
        // ------------------------------------------------------------

        // GET IDS OF THE NEIGHBORS OF EACH CELL ----------------------
        for (MFIter mfi(cell_nbr_ids); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            Array4<int> const & cell_nbr_ids_fab = cell_nbr_ids.array(mfi);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
            {
                Array4<short const> const & elm_bou_type_fab = this->elm_bou_type[dir].array(mfi);

                ParallelFor(bx, n_domains,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
                {
                    for (int b = 2*dir; b < 2*(dir+1); ++b)
                    {
                        // NEIGHBOR CELL
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        // GRID FACE
                        int fi, fj, fk;
                        GRID_FACE(i, j, k, b, fi, fj, fk);

                        // ELEMENT BOUNDARY TYPE
                        const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                        if (ELM_BOU_IS_VALID(ebtype))
                        {
                            cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b)) = elm_id_fab(nbr_i,nbr_j,nbr_k,dom);
                        }
                        else
                        {
                            cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b)) = -1;
                        }
                        /* DEBUG
                        if (i == 2 && j == 1)
                        {
                        const int id = elm_id_fab(i,j,k,dom);
                        Print() << "1 - id(" << i << "," << j << "," << k << ") " << id << ", ebtype: (" << fi << "," << fj << "," << fk << "): " << ebtype
                                << " is valid? " << (ELM_BOU_IS_VALID(ebtype) ? "yes" : "no")
                                << ", cell_nbr_ids_fab: " << cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom,b)) << std::endl;
                        }
                        */
                    }

                });
                Gpu::synchronize();
            }
        }
        cell_nbr_ids.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }

    /**
     * \brief Construct an iMultiFab with the ids, posX, dofs of the neighbors of each element.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] csr_dom_info: a iMultiFab containing the dof position offset.
     * \param[out] nbr_info: a iMultiFab that will contain the output information.
     * \param[in] ibvp: initial boundary value problem object.
     *
     * This routine is intended for a single-level application.
     *
    */
    template <typename IBVP>
    void get_system_nbr_info(const Geometry & geom, const Mesh & mesh, const iMultiFab & csr_dom_info, iMultiFab & nbr_info, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

        const int nbr_info_n_comp = __DG_ELM_NBR_INFO_N_COMP_PER_DOM__*n_domains;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        iMultiFab cell_nbr_ids;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        nbr_info.define(mesh.ba, mesh.dm, nbr_info_n_comp, mesh.dom_data_n_grow);
        nbr_info = -1;
        // ------------------------------------------------------------

        // GATHER NEIGHBORS IDS FROM THE CELLS' POINT OF VIEW ---------
        this->get_cell_nbr_ids(geom, mesh, cell_nbr_ids, ibvp);
        // ------------------------------------------------------------

        // STORE NEIGHBOR INFORMATION ---------------------------------
        for (MFIter mfi(nbr_info); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<int const> const & elm_id_fab = this->elm_id.array(mfi);
            
            Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
            Array4<int const> const & cell_nbr_ids_fab = cell_nbr_ids.array(mfi);

            Array4<int> const & nbr_info_fab = nbr_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const int id = elm_id_fab(i,j,k,dom);

                // LOCAL VARIABLES
                bool add_nbr_id;
                int elm_n_nbrs;
                int elm_nbrs[__DG_ELM_MAX_N_NBR_PER_DOM__];

                // UNKNOWN FIELDS
                int u_lo, u_hi;
                ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);
                const int Nu = u_hi-u_lo;
                const int elm_n_dof = Nu*sNp;

                // INITIALIZATION
                elm_n_nbrs = 0;
                std::fill(elm_nbrs, elm_nbrs+__DG_ELM_MAX_N_NBR_PER_DOM__, 0);
                
                if (elm_is_valid)
                {
                    // DIRECT NEIGHBORS
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        const int n_id = cell_nbr_ids_fab(i,j,k,CELL_NBR_ID(dom, b));

                        add_nbr_id =               ((n_id != id) && (n_id >= 0));
                        add_nbr_id = add_nbr_id && (std::count(elm_nbrs, elm_nbrs+elm_n_nbrs, n_id) == 0);

                        if (add_nbr_id)
                        {
                            elm_nbrs[elm_n_nbrs] = n_id;
                            elm_n_nbrs += 1;
                        }
                    }

                    // MERGING CELLS NBRS
                    if (elm_is_extended)
                    {
                        for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                        {
                            const int nbr_i = i+tables::extended_stencil_i[n];
                            const int nbr_j = j+tables::extended_stencil_j[n];
                            const int nbr_k = k+tables::extended_stencil_k[n];
                            const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                            
                            if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                            {
                                for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                                {
                                    const int n_id = cell_nbr_ids_fab(nbr_i,nbr_j,nbr_k,CELL_NBR_ID(dom, b));

                                    add_nbr_id =               ((n_id != id) && (n_id >= 0));
                                    add_nbr_id = add_nbr_id && (std::count(elm_nbrs, elm_nbrs+elm_n_nbrs, n_id) == 0);

                                    if (add_nbr_id)
                                    {
                                        elm_nbrs[elm_n_nbrs] = n_id;
                                        elm_n_nbrs += 1;
                                    }
                                }
                            }
                        }

                        // NEIGHBORS ON OTHER PHASE
                        //...
                    }

                    // SORT THE NEIGHBORS
                    // Here we assume all elements have the same number
                    // of degrees of freedom. See also below.
                    std::sort(elm_nbrs, elm_nbrs+elm_n_nbrs);

                    /* DEBUG
                    Print() << "elm_nbrs(" << i << "," << j << "," << k << ") (elm_n_nbrs = " << elm_n_nbrs << "): "; io::print_int_array_2d(1, elm_n_nbrs, elm_nbrs);
                    */

                    // STORE INFO
                    nbr_info_fab(i,j,k,ELM_N_NBRS(dom)) = elm_n_nbrs;
                    for (int n = 0; n < elm_n_nbrs; ++n)
                    {
                        nbr_info_fab(i,j,k,ELM_NBR_ID(dom, n)) = elm_nbrs[n];
                        
                        // We assume all neighbors have the same number
                        // of degrees of freedom.
                        nbr_info_fab(i,j,k,ELM_NBR_N_DOF(dom, n)) = elm_n_dof;
                        
                        //Print() << " - ELM_NBR_ID(dom, " << n << "): " << ELM_NBR_ID(dom, n) << ", ELM_NBR_N_DOF(dom, " << n << "): " << ELM_NBR_N_DOF(dom, n) << std::endl;
                    }
                }
            }
        }
        nbr_info.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }






    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
#ifdef AMREX_USE_GPU
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_dev_mem[dir].data();
        int const * ia_mem_ptr = ia.data();
        Real * J_mem_ptr = J.data();
#else
        Real const * cell_bou_quad_mem_ptr = mesh.cell_bou_quad_host_mem[dir].data();
        int const * ia_mem_ptr = ia.data();
        Real * J_mem_ptr = J.data();
#endif

        for (MFIter mfi(csr_bou_info[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<long const> const & cell_bou_quad_info_fab = mesh.cell_bou_quad_info[dir].array(mfi);
            Array4<short const> const & elm_type_fab = solution.elm_type.array(mfi);
            Array4<short const> const & elm_bou_type_fab = solution.elm_bou_type[dir].array(mfi);
            Array4<int const> const & csr_dom_info_fab = csr_dom_info.array(mfi);
            Array4<int const> const & csr_bou_info_fab = csr_bou_info[dir].array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // ELEMENT BOUNDARY TYPE
                const short ebtype = elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom));

                if (ELM_BOU_IS_VALID(ebtype))
                {
                    // LOCAL PARAMETERS
                    const int ff = (dir == 0) ? fi : ((dir == 1) ? fj : fk);
                    const Real face_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+fi*dx[0],
                                                                       prob_lo[1]+fj*dx[1],
                                                                       prob_lo[2]+fk*dx[2])};

                    // QUADRATURE INFO
                    const int bou_Nq = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_NQ(dom));
                    const long pos = cell_bou_quad_info_fab(fi,fj,fk,CELL_BOU_QUAD_POS(dom));
                    const Real * x_ptr = &cell_bou_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    int mi, mj, mk, pi, pj, pk;
                    short m_etype, p_etype;
                    int m_posX, p_posX;

                    // NEIGHBOR CELLS SHARING THE FACE fi,fj,fk
                    FACE_TO_NBRS(fi, fj, fk, dir, mi, mj, mk, pi, pj, pk);
                    m_etype = elm_type_fab(mi,mj,mk,ELM_TYPE(dom));
                    p_etype = elm_type_fab(pi,pj,pk,ELM_TYPE(dom));
                    m_posX = csr_dom_info_fab(mi,mj,mk,CSR_DOM_INFO_POS_X(dom));
                    p_posX = csr_dom_info_fab(pi,pj,pk,CSR_DOM_INFO_POS_X(dom));

                    // UNKNOWN FIELDS
                    space_elm_bfx m_bfu(&prob_lo[0], &dx[0], mi, mj, mk, m_etype, sp, X_fab);
                    space_elm_bfx p_bfu(&prob_lo[0], &dx[0], pi, pj, pk, p_etype, sp, X_fab);
                    int u_lo, u_hi;
                    ibvp.domain_unknown_fields_index_bounds(dom, u_lo, u_hi);

                    // BOUNDARY CONDITIONS
                    if (ELM_BOU_IS_NEG_WALL(ebtype))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = -1.0;

                        // EVAL THE INTEGRAL
                        JAC_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, p_bfu, u_lo, u_hi, fi, fj, fk, ia_mem_ptr, idx, J_mem_ptr, p_posX, ibvp);
                    }
                    // BOUNDARY CONDITIONS
                    else if (ELM_BOU_IS_POS_WALL(ebtype))
                    {
                        // OUTER UNIT NORMAL (Note the sign)
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

                        // EVAL THE INTEGRAL
                        JAC_BCS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, u_lo, u_hi, fi, fj, fk, ia_mem_ptr, idx, J_mem_ptr, m_posX, ibvp);
                    }
                    // INTRAPHASE CONDITIONS
                    else
                    {
                        // UNIT NORMAL
                        Real un[AMREX_SPACEDIM] = {AMREX_D_DECL(0.0, 0.0, 0.0)};
                        un[dir] = +1.0;

Print() << "fi, fj, fk: " << fi << "," << fj << "," << fk << ", m_posX, p_posX: " << m_posX << ", " << p_posX << std::endl;

                        // EVAL THE INTEGRAL
                        JAC_ICS(dom, t, face_lo, bou_Nq, x_ptr, un, m_bfu, p_bfu, u_lo, u_hi, fi, fj, fk, ia_mem_ptr, idx, J_mem_ptr, m_posX, p_posX, ibvp);
                    }
                }
            }
        }
    }