//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution.H
 * \brief Contains constants, data structures and functions for discontinuous Galerkin solutions.
*/

#ifndef AMREX_DG_SOLUTION_H_
#define AMREX_DG_SOLUTION_H_

#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <AMReX_dG_Mesh.H>
#include <AMReX_dG_BFX.H>

#define __DG_MAX_N_UNKNOWN_FIELDS__ 10
#define __DG_MAX_N_OUTPUT_FIELDS__ 18

namespace amrex
{
namespace dG
{
// AUXILIARY FUNCTIONS ################################################
// ####################################################################



// DG SOLUTION CLASS ##################################################
/**
 * \brief dG solution parameters class.
 *
*/
struct SolutionParameters
{
    // DATA MEMBERS ===================================================
    int time_p, space_p;

    bool FV_is_active;

    bool cell_merging_is_active;
    Real cell_merging_volume_fraction_threshold;
    
    std::string time_integration;
    Real time_integration_CFL;

    int post_processing_grid_order;
    // ================================================================


    // CONSTRUCTOR ====================================================
    SolutionParameters()
    :
    time_p{-1},
    space_p{-1},
    FV_is_active{false},
    cell_merging_is_active{false},
    cell_merging_volume_fraction_threshold{-1.0},
    time_integration{""},
    time_integration_CFL{0.0},
    post_processing_grid_order{1}
    {}
    // ================================================================
};

/**
 * \brief dG solution class.
 *
*/
struct Solution
{
    // DATA MEMBERS ===================================================
    static const int dom_data_n_grow = 1;
    static const int bou_data_n_grow = 0;

    SolutionParameters params;

    // FIELDS TO PARENT DOMAIN
    Gpu::HostVector<int> unknown_field_parent_domain_host_mem;
    Gpu::DeviceVector<int> unknown_field_parent_domain_dev_mem;
    Gpu::HostVector<int> output_field_parent_domain_host_mem;
    Gpu::DeviceVector<int> output_field_parent_domain_dev_mem;

    // ELEMENT TYPE / ELEMENT BOUNDARY TYPE
    sMultiFab elm_type;
    Array<sMultiFab, AMREX_SPACEDIM> elm_bou_type;

    // ELEMENT CENTROID
    MultiFab elm_centroid;

    // MASS MATRIX (STORED AS ITS CHOLESKY DECOMPOSITION)
    lMultiFab elm_MMCh_info;

    Gpu::HostVector<Real> elm_MMCh_host_mem;
    Gpu::DeviceVector<Real> elm_MMCh_dev_mem;

    // INTERPOLATION OPERATORS
    lMultiFab elm_I_info;
    Gpu::HostVector<Real> elm_I_host_mem;
    Gpu::DeviceVector<Real> elm_I_dev_mem;
    // ================================================================


    // CONSTRUCTOR ====================================================
    // ================================================================


    // INITIALIZATION =================================================
    void read_input_file(const ParmParse & pp);
    void read_input_file();
    // ================================================================


    // READERS ========================================================
    bool uses_cell_merging() const;
    bool time_integration_is_explicit_RKdG() const;

    /**
     * \brief Make sure the problem can be handled.
     *
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void check_whether_problem_is_manageable(const IBVP & ibvp) const
    {
        // IBVP PARAMETERS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
        const int n_output_fields = ibvp.get_number_of_output_fields();

        if (n_unknown_fields > __DG_MAX_N_UNKNOWN_FIELDS__)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Solution.H - Solution::check_whether_problem_is_manageable\n";
            msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
            msg += "| n_unknown_fields           : "+std::to_string(n_unknown_fields)+".\n";
            msg += "| __DG_MAX_N_UNKNOWN_FIELDS__: "+std::to_string(__DG_MAX_N_UNKNOWN_FIELDS__)+".\n";
            Abort(msg);
        }
        if (n_output_fields > __DG_MAX_N_OUTPUT_FIELDS__)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Solution.H - Solution::check_whether_problem_is_manageable\n";
            msg += "| The number of unknown fields exceeds the maximum allowed number.\n";
            msg += "| n_output_fields           : "+std::to_string(n_output_fields)+".\n";
            msg += "| __DG_MAX_N_OUTPUT_FIELDS__: "+std::to_string(__DG_MAX_N_OUTPUT_FIELDS__)+".\n";
            Abort(msg);
        }
    }
    // ================================================================


    // FIELDS TO DOMAIN CORRESPONDENCE ================================
    /**
     * \brief Set the correspondence between the unknown/output fields and their parent domains.
     *
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void set_fields_parent_domains(const IBVP & ibvp)
    {
        // IBVP PARAMETERS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
        const int n_output_fields = ibvp.get_number_of_output_fields();

        // UNKNOWN FIELDS TO PARENT DOMAIN
        this->unknown_field_parent_domain_host_mem.resize(n_unknown_fields);
        for (int u = 0; u < n_unknown_fields; ++u)
        {
            this->unknown_field_parent_domain_host_mem[u] = ibvp.unknown_field_parent_domain(u);
        }
#ifdef AMREX_USE_GPU
        this->unknown_field_parent_domain_dev_mem.resize(this->unknown_field_parent_domain_host_mem.size());
        Gpu::copy(Gpu::hostToDevice, this->unknown_field_parent_domain_host_mem.begin(),
                                     this->unknown_field_parent_domain_host_mem.end(),
                                     this->unknown_field_parent_domain_dev_mem.begin());
#endif
        // OUTPUT FIELDS TO PARENT DOMAIN
        this->output_field_parent_domain_host_mem.resize(n_output_fields);
        for (int u = 0; u < n_output_fields; ++u)
        {
            this->output_field_parent_domain_host_mem[u] = ibvp.output_field_parent_domain(u);
        }
#ifdef AMREX_USE_GPU
        this->output_field_parent_domain_dev_mem.resize(this->output_field_parent_domain_host_mem.size());
        Gpu::copy(Gpu::hostToDevice, this->output_field_parent_domain_host_mem.begin(),
                                     this->output_field_parent_domain_host_mem.end(),
                                     this->output_field_parent_domain_dev_mem.begin());
#endif
    }
    // ================================================================


    // INITIALIZE THE MULTIFABS =======================================
    /**
     * \brief Initialize the multifabs that will contain the dG solution in space.
     *
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[inout] X: a MultiFab object that will be initialized to contain the dG coefficients.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void init_space_solution_multifabs(const Mesh & mesh, MultiFab & X, const IBVP & ibvp)
    {
        // UNKNOWN FIELDS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
        const int X_n_comp = sNp*n_unknown_fields;

        // INITIALIZATION
        X.define(mesh.ba, mesh.dm, X_n_comp, this->dom_data_n_grow);
        X = 0.0;
    }
    // ================================================================


    // ELEMENT INFO ===================================================
    /**
     * \brief Set the type of elements' and element boundaries'.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void set_elm_and_elm_bou_type(const Geometry & geom, const Mesh & mesh, const IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // COMPONENTS OF THE ELM / ELM BOU TYPE MULTIFABS
        // Type
        const int elm_type_n_comp = n_domains*__DG_ELM_TYPE_N_COMP_PER_DOM__;
        const int elm_bou_type_n_comp = n_domains*__DG_ELM_BOU_TYPE_N_COMP_PER_DOM__;

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
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        // ELEMENT / ELEMENT BOU TYPE
        this->elm_type.define(mesh.ba, mesh.dm, elm_type_n_comp, this->dom_data_n_grow);
        this->elm_type = __DG_ELM_TYPE_UNDEFINED__;
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
        {
            this->elm_bou_type[dim].define(fc_ba[dim], mesh.dm, elm_bou_type_n_comp, this->bou_data_n_grow);
            this->elm_bou_type[dim] = __DG_ELM_BOU_TYPE_UNDEFINED__;
        }
        // ------------------------------------------------------------

        // ELEMENT TYPE -----------------------------------------------
        for (MFIter mfi(mesh.cell_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            Array4<short const> const & cell_type_fab = mesh.cell_type.array(mfi);
            Array4<short> const & elm_type_fab = this->elm_type.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                const short ctype = cell_type_fab(i,j,k,CELL_TYPE(dom));
                
                if (CELL_IS_EMPTY(ctype))
                {
                    elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_EMPTY__;
                }
                else if (CELL_IS_ENTIRE(ctype))
                {
                    elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_ENTIRE__;
                }
                else if (CELL_IS_CUT(ctype))
                {
                    elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_PARTIAL__;
                }

                if (CELL_IS_GHOST(ctype))
                {
                    elm_type_fab(i,j,k,ELM_TYPE(dom)) += __DG_CELL_GHOST__;
                }
            });
            Gpu::synchronize();
        }
        this->elm_type.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // ELEMENT BOUNDARY TYPE --------------------------------------
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                Array4<short const> const & cell_bou_type_fab = mesh.cell_bou_type[dir].array(mfi);
                Array4<short> const & elm_bou_type_fab = this->elm_bou_type[dir].array(mfi);

                ParallelFor(bx, n_domains,
                [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
                {
                    const short type = cell_bou_type_fab(fi,fj,fk,CELL_BOU_TYPE(dom));
                
                    if (CELL_BOU_IS_EMPTY(type))
                    {
                        elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_EMPTY__;
                    }
                    else if (CELL_BOU_IS_ENTIRE(type))
                    {
                        elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_ENTIRE__;
                    }
                    else if (CELL_BOU_IS_CUT(type))
                    {
                        elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL__;
                    }
                });
                Gpu::synchronize();
            }
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Set the merging information.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void set_cell_merging(const Geometry & geom, const Mesh & mesh, const IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // VOLUME FRACTION THRESHOLD
        const Real vft = this->params.cell_merging_volume_fraction_threshold;
        // ------------------------------------------------------------

        // SET LARGE VS SMALL ELEMENTS --------------------------------
        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<Real const> const & cell_vf_fab = mesh.cell_vf.array(mfi);
            Array4<short> const & elm_type_fab = this->elm_type.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const bool elm_is_partial = ELM_IS_PARTIAL(elm_type_fab(i,j,k,ELM_TYPE(dom)));

                if (elm_is_partial)
                {
                    const Real vf = cell_vf_fab(i,j,k,dom);
                    
                    //Print() << "vf(" << i << "," << j << ",k): " << vf << std::endl;

                    // CONDITION FOR A LARGE ELEMENT
                    if (vf >= vft)
                    {
                        elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_LARGE__;
                    }
                    // SMALL ELEMENT
                    else
                    {
                        elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__;
                    }
                }
            });
            Gpu::synchronize();
        }
        this->elm_type.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // FIND SUITABLE NEIGHBORS FOR MERGING ------------------------
        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<Real const> const & cell_vf_fab = mesh.cell_vf.array(mfi);
            
            Array4<short> const & elm_type_fab = this->elm_type.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const bool elm_is_small = ELM_IS_SMALL(elm_type_fab(i,j,k,ELM_TYPE(dom)));

                if (elm_is_small)
                {
                    // LOCAL VARIABLES
                    bool elm_has_been_merged;
                    Real best_nbr_vf;
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    bool nbr_is_not_ghost;
                    Real nbr_vf;

                    elm_has_been_merged = false;
                    best_nbr_vf = vft;

                    // SEARCH THE VON NEUMANN NEIGHBORHOOD
                    for (int b = 0; b < __DG_STD_ELEM_N_SPACE_BOUNDARIES__; ++b)
                    {
                        // Neighboring cell
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_is_not_ghost = !ELM_IS_GHOST(elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom)));
                        nbr_vf = cell_vf_fab(nbr_i,nbr_j,nbr_k,dom);

                        if ((nbr_vf > best_nbr_vf) && nbr_is_not_ghost)
                        {
                            elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                            elm_has_been_merged = true;
                            best_nbr_vf = nbr_vf;
                        }
                    }
#if (AMREX_SPACEDIM == 2)
                    // IN 2D, WE EXTEND THE SEARCH TO THE NEIGHBORS
                    // THAT SHARE A CORNER WITH THE CURRENT CELL
                    if (!elm_has_been_merged)
                    {
                        for (int c = 0; c < __DG_STD_ELEM_N_SPACE_CORNERS__; ++c)
                        {
                            const int b = c+__DG_STD_ELEM_N_SPACE_BOUNDARIES__;

                            // Neighboring cell
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_is_not_ghost = !ELM_IS_GHOST(elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom)));
                            nbr_vf = cell_vf_fab(nbr_i,nbr_j,nbr_k,dom);

                            if ((nbr_vf > best_nbr_vf) && nbr_is_not_ghost)
                            {
                                elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                                elm_has_been_merged = true;
                                best_nbr_vf = nbr_vf;
                            }
                        }
                    }
#endif
#if (AMREX_SPACEDIM == 3)
                    // IN 3D, WE EXTEND THE SEARCH TO THE NEIGHBORS
                    // THAT SHARE AN EDGE WITH THE CURRENT CELL
                    if (!elm_has_been_merged)
                    {
                        for (int ed = 0; ed < __DG_STD_ELEM_N_SPACE_EDGES__; ++ed)
                        {
                            const int b = ed+__DG_STD_ELEM_N_SPACE_BOUNDARIES__;

                            // Neighboring cell
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_is_not_ghost = !ELM_IS_GHOST(elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom)));
                            nbr_vf = cell_vf_fab(nbr_i,nbr_j,nbr_k,dom);

                            if ((nbr_vf > best_nbr_vf) && nbr_is_not_ghost)
                            {
                                elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                                elm_has_been_merged = true;
                                best_nbr_vf = nbr_vf;
                            }
                        }
                    }

                    // IN 3D, WE EXTEND THE SEARCH TO THE NEIGHBORS
                    // THAT SHARE A CORNER WITH THE CURRENT CELL
                    if (!elm_has_been_merged)
                    {
                        for (int c = 0; c < __DG_STD_ELEM_N_SPACE_CORNERS__; ++c)
                        {
                            const int b = c+__DG_STD_ELEM_N_SPACE_BOUNDARIES__+__DG_STD_ELEM_N_SPACE_EDGES__;

                            // Neighboring cell
                            NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                            nbr_is_not_ghost = !CELL_IS_GHOST(elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom)));
                            nbr_vf = cell_vf_fab(nbr_i,nbr_j,nbr_k,dom);

                            if ((nbr_vf > best_nbr_vf) && nbr_is_not_ghost)
                            {
                                elm_type_fab(i,j,k,ELM_TYPE(dom)) = __DG_ELM_TYPE_SMALL__+b*10;
                                elm_has_been_merged = true;
                                best_nbr_vf = nbr_vf;
                            }
                        }
                    }
#endif

                    // CHECK
                    if (!elm_has_been_merged)
                    {
#ifdef AMREX_USE_GPU
                        Abort("\nERROR: AMReX_dG_Solution.H - Solution::set_cell_merging\n");
#else
                        std::string msg;
                        msg  = "\n";
                        msg +=  "ERROR: AMReX_dG_Solution.H - Solution::set_cell_merging\n";
                        msg += "| Could not find a neighboring cell suitable for merging:\n";
                        msg += "| Domain: "+std::to_string(dom)+"\n";
                        msg += "| cell indices: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                        msg += "| cell volume fraction: "+std::to_string(cell_vf_fab(i,j,k,dom))+"\n";
                        Abort(msg);
#endif
                    }
                }
            });
            Gpu::synchronize();
        }
        this->elm_type.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // LABEL THE EXTENDED ELEMENTS --------------------------------
        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short> const & elm_type_fab = this->elm_type.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_valid = ELM_IS_VALID(etype);

                if (elm_is_valid)
                {
                    // LOCAL VARIABLES
                    bool elm_is_extended;
                    int n;

                    elm_is_extended = false;
                    n = 0;

                    while (!elm_is_extended && (n < __DG_EXTENDED_STENCIL_N_NBR__))
                    {
                        const int nbr_i = i+tables::extended_stencil_i[n];
                        const int nbr_j = j+tables::extended_stencil_j[n];
                        const int nbr_k = k+tables::extended_stencil_k[n];
                        const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                        if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                        {
                            elm_is_extended = true;
                            elm_type_fab(i,j,k,ELM_TYPE(dom)) += 10;
                        }
                        else
                        {
                            n += 1;
                        }
                    }   
                }
            });
            Gpu::synchronize();
        }
        this->elm_type.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
        
        // LABEL INVALID BOUNDARIES -----------------------------------
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            for (MFIter mfi(mesh.cell_bou_type[dir]); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();

                Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
                Array4<short> const & elm_bou_type_fab = this->elm_bou_type[dir].array(mfi);

                ParallelFor(bx, n_domains,
                [=] AMREX_GPU_DEVICE (int fi, int fj, int fk, int dom) noexcept
                {
                    int i, j, k, nbr_i, nbr_j, nbr_k;
                    short etype, nbr_etype;

                    FACE_TO_NBRS(fi, fj, fk, dir, i, j, k, nbr_i, nbr_j, nbr_k);

                    etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                    nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));

                    if (CELLS_ARE_MERGED(i, j, k, etype, nbr_i, nbr_j, nbr_k, nbr_etype))
                    {
                        elm_bou_type_fab(fi,fj,fk,ELM_BOU_TYPE(dom)) = __DG_ELM_BOU_TYPE_PARTIAL_INVALID__;
                    }
                });
                Gpu::synchronize();
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // ROUTINES RELATED TO EXTENDED / SMALL ELEMENTS ==================
    /**
     * \brief Add the contribution of the small elements to the corresponding merging extended elements.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[inout] X: the MultiFab object whose components are updated.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void add_small_elements_contribution(const Geometry & geom, MultiFab & X, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // NUMBER OF UNKNOWN FIELDS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
        // ------------------------------------------------------------

        // ADD SMALL ELEMENT DATA TO MERGING EXTENDED ELEMENTS --------
#ifdef AMREX_USE_GPU
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_dev_mem.data();
#else
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_host_mem.data();
#endif
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            
            Array4<Real> const & X_fab = X.array(mfi);

            ParallelFor(bx, n_unknown_fields,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = parent_dom_mem_ptr[ru];
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);

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
                            for (int rs = 0; rs < sNp; ++rs)
                            {
                                X_fab(i,j,k,rs+ru*sNp) += X_fab(nbr_i,nbr_j,nbr_k,rs+ru*sNp);
                            }
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
        X.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }

    /**
     * \brief Copy solution from extended elements to their small elements.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[inout] X: the MultiFab object whose components are updated.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void copy_solution_to_small_elements(const Geometry & geom, MultiFab & X, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // NUMBER OF UNKNOWN FIELDS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
        // ------------------------------------------------------------

        // ADD SMALL ELEMENT DATA TO MERGING EXTENDED ELEMENTS --------
#ifdef AMREX_USE_GPU
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_dev_mem.data();
#else
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_host_mem.data();
#endif
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            
            Array4<Real> const & X_fab = X.array(mfi);

            ParallelFor(bx, n_unknown_fields,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = parent_dom_mem_ptr[ru];
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_small = ELM_IS_SMALL(etype);

                // LOCAL VARIABLES
                int b, nbr_i, nbr_j, nbr_k, nbr_b;

                if (elm_is_small)
                {
                    b = etype/10;
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        X_fab(i,j,k,rs+ru*sNp) = X_fab(nbr_i,nbr_j,nbr_k,rs+ru*sNp);
                    }
                }
            });
            Gpu::synchronize();
        }
        X.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE FROM SCRATCH ==============================================
    /**
     * \brief Make the data structures that will store the solution info from scratch.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void make_from_scratch(const Real t, const Geometry & geom, const Mesh & mesh, const IBVP & ibvp)
    {
        // ELEMENT / ELEMENT BOUNDARY TYPE
        this->set_elm_and_elm_bou_type(geom, mesh, ibvp);

        // CELL MERGING
        if (this->uses_cell_merging())
        {
            this->set_cell_merging(geom, mesh, ibvp);
        }

        this->set_fields_parent_domains(ibvp);
        this->check_whether_problem_is_manageable(ibvp);
    }
    // ================================================================


    // REMAKE =========================================================
    /**
     * \brief Remake the data structures that will store the solution info.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void remake(const Real t, const Geometry & geom, const Mesh & mesh, const IBVP & ibvp)
    {
        // ELEMENT / ELEMENT BOUNDARY TYPE
        this->set_elm_and_elm_bou_type(geom, mesh, ibvp);

        // CELL MERGING
        if (this->uses_cell_merging())
        {
            this->set_cell_merging(geom, mesh, ibvp);
        }
    }

    /**
     * \brief Remake the data structures that will store the solution info.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[inout] X: Multi object containing the dG coefficients.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void remake_solution(const Real t, const Geometry & geom, const Mesh & mesh, MultiFab & X, const IBVP & ibvp)
    {
        
    }
    // ================================================================


    // MASS MATRICES ==================================================
    /**
     * \brief Eval and store the Cholesky decomposition of mass matrix for the mesh elements.
     *
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
     *
    */
    template <typename IBVP>
    void eval_space_mass_matrix(const Geometry & geom, const Mesh & mesh, const IBVP & ibvp)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));

        // QUADRATURE
        const int qo_reg = mesh.params.quadrature_order_regular_elements;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        StandardRectangle<AMREX_SPACEDIM> std_elm;
        long elm_MMCh_mem_pos;
        // ------------------------------------------------------------

        // INIT THE MULTIFAB ------------------------------------------
        this->elm_MMCh_info.define(mesh.ba, mesh.dm, n_domains, this->dom_data_n_grow);
        this->elm_MMCh_info = -1L;
        // ------------------------------------------------------------

        // INITIALIZE THE STANDARD ELEMENT ----------------------------
        std_elm.set_quadrature(&dx[0], qo_reg);
        std_elm.set_mass_matrix(sp);
        // ------------------------------------------------------------

        // FIRST BLOCK OF MEMORY STORES THE STANDARD ELEMENT INFO -----
        elm_MMCh_mem_pos = 0L;

        {
            // Resize memory
            this->elm_MMCh_host_mem.resize(sNp*sNp);

            // Store data
            std::copy(std_elm.MMCh.begin(), std_elm.MMCh.end(), this->elm_MMCh_host_mem.data());

            // Update memory
            elm_MMCh_mem_pos += sNp*sNp;
        }
        // ------------------------------------------------------------

        // EVAL (AND TEMPORARILY STORE) THE MASS MATRICES -------------
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();

        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);

            Array4<long> const & elm_MMCh_info_fab = this->elm_MMCh_info.array(mfi);

            for (int dom = 0; dom < n_domains; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real MM[__DG_MAX_SPACE_Np__*__DG_MAX_SPACE_Np__];

                // BASIS FUNCTIONS
                space_elm_bf bf(&prob_lo[0], &dx[0], i, j, k, etype, sp);

                // EMPTY ELEMENT
                if (ELM_IS_EMPTY(etype))
                {
                    // Store memory offset
                    elm_MMCh_info_fab(i,j,k,dom) = -1L;
                }
                // ENTIRE ELEMENT
                else if (ELM_IS_ENTIRE(etype))
                {
                    // For extended elements, we will add the
                    // contribution from the small elements.
                    if (ELM_IS_EXTENDED(etype))
                    {
                        // Store memory offset
                        elm_MMCh_info_fab(i,j,k,dom) = elm_MMCh_mem_pos;

                        // Expand memory
                        this->elm_MMCh_host_mem.resize(elm_MMCh_mem_pos+sNp*sNp);

                        // Store data
                        std::copy(std_elm.MM.begin(), std_elm.MM.end(), &this->elm_MMCh_host_mem[elm_MMCh_mem_pos]);

                        // Update memory
                        elm_MMCh_mem_pos += sNp*sNp;
                    }
                    else
                    {
                        // Store memory offset
                        elm_MMCh_info_fab(i,j,k,dom) = 0L;
                    }
                }
                // LARGE OR SMALL ELEMENT
                else
                {
                    // DOMAIN QUADRATURE INFO
                    const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                    const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                    const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                    // Store memory offset
                    elm_MMCh_info_fab(i,j,k,dom) = elm_MMCh_mem_pos;

                    // Expand memory
                    this->elm_MMCh_host_mem.resize(elm_MMCh_mem_pos+sNp*sNp);

                    // Store data
                    std::fill(MM, MM+sNp*sNp, 0.0);
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

                        bf.eval_bf_only_table(x);

                        for (int cs = 0; cs < sNp; ++cs)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            MM[rs+cs*sNp] += bf.BF[rs]*bf.BF[cs]*w;
                        }
                    }

                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        this->elm_MMCh_host_mem[elm_MMCh_mem_pos+rs] = MM[rs];
                    }

                    // Update memory
                    elm_MMCh_mem_pos += sNp*sNp;
                }
            }
        }
        // ------------------------------------------------------------

#ifdef AMREX_USE_GPU
        // COPY TO DEVICE ---------------------------------------------
        this->elm_MMCh_dev_mem.resize(this->elm_MMCh_host_mem.size());

        Gpu::copy(Gpu::hostToDevice, this->elm_MMCh_host_mem.begin(), this->elm_MMCh_host_mem.end(), this->elm_MMCh_dev_mem.begin());
        // ------------------------------------------------------------
#endif

        // EVAL THE CHOLESKY DECOMPOSITION ----------------------------
#ifdef AMREX_USE_GPU
        Real * elm_MMCh_mem_ptr = this->elm_MMCh_dev_mem.data();
#else
        Real * elm_MMCh_mem_ptr = this->elm_MMCh_host_mem.data();
#endif

        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<long const> const & elm_MMCh_info_fab = this->elm_MMCh_info.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_large = ELM_IS_LARGE(etype);
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const bool elm_is_entire_extended = ELM_IS_ENTIRE(etype) && elm_is_extended;
                const bool elm_is_special = elm_is_large || elm_is_entire_extended;
                const long pos = elm_MMCh_info_fab(i,j,k,dom);

                // LOCAL VARIABLES
                Real * MMCh_ptr = &elm_MMCh_mem_ptr[pos];
                long nbr_pos;
                Real * nbr_MMCh_ptr;
                int info;

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
                            nbr_pos = elm_MMCh_info_fab(nbr_i,nbr_j,nbr_k,dom);
                            nbr_MMCh_ptr = &elm_MMCh_mem_ptr[nbr_pos];

                            for (int rs = 0; rs < (sNp*sNp); ++rs)
                            {
                                MMCh_ptr[rs] += nbr_MMCh_ptr[rs];
                            }
                        }
                    }
                }

                if (elm_is_special)
                {
                    linalg::dpotf2('U', sNp, MMCh_ptr, sNp, info);

                    if (info != 0)
                    {
#ifdef AMREX_USE_GPU
                        Abort("\nERROR: AMReX_dG_Solution.H - Solution::eval_space_mass_matrix\n");
#else
                        std::string msg;
                        msg  = "\n";
                        msg +=  "ERROR: AMReX_dG_Solution.H - Solution::eval_space_mass_matrix\n";
                        msg += "| Something went wrong during the evaluation of the Cholesky decomposition of the mass matrix:\n";
                        msg += "| Domain: "+std::to_string(dom)+"\n";
                        msg += "| cell indices: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                        Abort(msg);
#endif
                    }
                }

            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // GHOST ELEMENTS ---------------------------------------------
        for (MFIter mfi(this->elm_type); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.fabbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<long const> const & elm_MMCh_info_fab = this->elm_MMCh_info.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_ghost = ELM_IS_GHOST(etype);
                const bool elm_is_large = ELM_IS_LARGE(etype);

                if (elm_is_ghost && elm_is_large)
                {
                    const long pos = elm_MMCh_info_fab(i,j,k,dom);
                    int info;

                    // CHOLESKY DECOMPOSITION
                    linalg::dpotf2('U', sNp, &elm_MMCh_mem_ptr[pos], sNp, info);

                    if (info != 0)
                    {
#ifdef AMREX_USE_GPU
                        Abort("\nERROR: AMReX_dG_Solution.H - Solution::eval_space_mass_matrix - 2\n");
#else
                        std::string msg;
                        msg  = "\n";
                        msg +=  "ERROR: AMReX_dG_Solution.H - Solution::eval_space_mass_matrix - 2\n";
                        msg += "| Something went wrong during the evaluation of the Cholesky decomposition of the mass matrix:\n";
                        msg += "| Domain: "+std::to_string(dom)+"\n";
                        msg += "| cell indices: "+std::to_string(i)+","+std::to_string(j)+","+std::to_string(k)+"\n";
                        Abort(msg);
#endif
                    }
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Multiply dG coefficients by the inverse of the mass matrices.
     *
     * \param[in] geom: amrex Geometry object; contains the problem periodicity.
     * \param[inout] X: the MultiFab object whose coefficients are multiplied by the inverse of the space
     *                  mass matrix.
     * \param[in] ibvp: initial boundary value problem object.
     * \param[in] use_fabbox: if true, the loops will include the ghost cells.
     *
    */
    template <typename IBVP>
    void multiply_by_inverse_space_mass_matrix(const Geometry & geom, MultiFab & X, const IBVP & ibvp, const bool use_fabbox = false) const
    {
        // PARAMETERS -------------------------------------------------
        // NUMBER OF UNKNOWN FIELDS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
        // ------------------------------------------------------------

        // PERFORM THE MULTIPLICATION ---------------------------------
#ifdef AMREX_USE_GPU
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_dev_mem.data();
        const Real * elm_MMCh_mem_ptr = this->elm_MMCh_dev_mem.data();
#else
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_host_mem.data();
        const Real * elm_MMCh_mem_ptr = this->elm_MMCh_host_mem.data();
#endif
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = use_fabbox ? mfi.fabbox() : mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<long const> const & elm_MMCh_info_fab = this->elm_MMCh_info.array(mfi);

            Array4<Real> const & X_fab = X.array(mfi);

            ParallelFor(bx, n_unknown_fields,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const int dom = parent_dom_mem_ptr[ru];
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_entire = ELM_IS_ENTIRE(etype);
                const bool elm_is_large = ELM_IS_LARGE(etype);

                // MASS MATRIX
                const long pos = elm_MMCh_info_fab(i,j,k,dom);
                const Real * MMCh_ptr = &elm_MMCh_mem_ptr[pos];

                // LOCAL VARIABLES
                int info;
                Real x[__DG_MAX_SPACE_Np__];

                if (elm_is_entire || elm_is_large)
                {
                    // COPY FROM MEMORY
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        x[rs] = X_fab(i,j,k,rs+ru*sNp);
                    }

                    // USE CHOLESKY DECOMPOSITION
                    linalg::dpotrs('U', sNp, 1, MMCh_ptr, sNp, x, sNp, info);

                    // COPY BACK TO MEMORY
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        X_fab(i,j,k,rs+ru*sNp) = x[rs];
                    }
                }
            });
            Gpu::synchronize();
        }
        X.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }
    // ================================================================


    // PROJECTION OF INITIAL CONDITIONS ===============================
    /**
     * \brief Project initial conditions over the mesh.
     *
     * \param[in] geom: amrex Geometry object; contains dx and background domain lower end.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[inout] X: a MultiFab object that will contain the dG coefficients of the projected initial
     *                  conditions.
     * \param[in] ibvp: initial boundary value problem object.
     * \param[in] use_fabbox: if true, the loops will include the ghost cells.
     *
    */
    template <typename IBVP>
    void project_initial_conditions(const Geometry & geom, const Mesh & mesh, MultiFab & X, const IBVP & ibvp, const bool use_fabbox = false)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // NUMBER OF UNKNOWN FIELDS
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        const int sNp = AMREX_D_TERM((1+sp),*(1+sp),*(1+sp));
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        X = 0.0;
        // ------------------------------------------------------------

        // EVAL THE PROJECTION ----------------------------------------
#ifdef AMREX_USE_GPU
        const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_dev_mem.data();
#else
        const Real * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
        const int * parent_dom_mem_ptr = this->unknown_field_parent_domain_host_mem.data();
#endif
        for (MFIter mfi(X); mfi.isValid(); ++mfi)
        {
            const Box & bx = use_fabbox ? mfi.fabbox() : mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            
            Array4<Real> const & X_fab = X.array(mfi);

            ParallelFor(bx, n_unknown_fields,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
            {
                // LOCAL PARAMETERS
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const int dom = parent_dom_mem_ptr[ru];
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w;
                Real integrand;
                Real U[__DG_MAX_N_UNKNOWN_FIELDS__];

                // BASIS FUNCTIONS
                space_elm_bf bf(&prob_lo[0], &dx[0], i, j, k, etype, sp);

                // EVAL THE INTEGRAL
                for (int q = 0; q < dom_Nq; ++q)
                {
                    // Quadrature point and weight
                    const long x_pos = (AMREX_SPACEDIM+1)*q;
                    AMREX_D_TERM
                    (
                        x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                        x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                        x[2] = x_ptr[x_pos+2]+cell_lo[2];
                    )
                    w = x_ptr[x_pos+AMREX_SPACEDIM];

                    // Initial conditions at x
                    ibvp.eval_ICs(x, U);

                    // Basis functions at x
                    bf.eval_bf_only_table(x);

                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        integrand = bf.BF[rs]*U[ru];
                        X_fab(i,j,k,rs+ru*sNp) += integrand*w;
                    }
                }
            });
            Gpu::synchronize();
        }
        X.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX
        this->add_small_elements_contribution(geom, X, ibvp);
        this->multiply_by_inverse_space_mass_matrix(geom, X, ibvp, use_fabbox);
        this->copy_solution_to_small_elements(geom, X, ibvp);
        // ------------------------------------------------------------
    }
    // ================================================================


    // ELEMENTS CENTROIDS =============================================
    /**
     * \brief Compute the centroids of the elements.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_space_centroids(const Geometry & geom, const Mesh & mesh, const IBVP & ibvp, const bool include_ghost_cells = false)
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
        const Real entire_cell_volume = AMREX_D_TERM(dx[0],*dx[1],*dx[2]);

        // DOMAINS
        const int n_domains = ibvp.get_number_of_domains();
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        this->elm_centroid.define(mesh.ba, mesh.dm, AMREX_SPACEDIM*n_domains, this->dom_data_n_grow);
        this->elm_centroid = 0.0;
        // ------------------------------------------------------------

        // CYCLE OVER THE ELEMENTS ------------------------------------
        // Now, we compute the centroids of empty and (non-extended)
        // large and entire elements. For extended elements and small
        // elements, we compute the static moments.
        // ------------------------------------------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif
        for (MFIter mfi(this->elm_centroid); mfi.isValid(); ++mfi)
        {
            const Box & bx = include_ghost_cells ? mfi.fabbox() : mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
            Array4<Real const> const & cell_vf_fab = mesh.cell_vf.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);

            Array4<Real> const & elm_centroid_fab = this->elm_centroid.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_extended = ELM_IS_EXTENDED(etype);
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(cell_lo[0]+0.5*dx[0],
                                                                  cell_lo[1]+0.5*dx[1],
                                                                  cell_lo[2]+0.5*dx[2])};

                // QUADRATURE INFO
                const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];
                
                // LOCAL VARIABLES
                Real x[AMREX_SPACEDIM], w, S[AMREX_SPACEDIM], iV;

                // CELL MID POINT
                if (ELM_IS_EMPTY(etype) || (ELM_IS_ENTIRE(etype) && !elm_is_extended))
                {
                    AMREX_D_TERM
                    (
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom)) = cell_m[0];,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom)) = cell_m[1];,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom)) = cell_m[2];
                    )
                }
                // NEED TO EVAL THE STATIC MOMENTS
                else
                {
                    AMREX_D_TERM
                    (
                        S[0] = 0.0;,
                        S[1] = 0.0;,
                        S[2] = 0.0;
                    )

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

                        AMREX_D_TERM
                        (
                            S[0] += x[0]*w;,
                            S[1] += x[1]*w;,
                            S[2] += x[2]*w;
                        )
                    }

                    // STORE THE CENTROID FOR NON-EXTENDED LARGE ELEMENTS
                    if (ELM_IS_LARGE(etype) && !elm_is_extended)
                    {
                        iV = 1.0/(cell_vf_fab(i,j,k,dom)*entire_cell_volume);
                        AMREX_D_TERM
                        (
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom)) = S[0]*iV;,
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom)) = S[1]*iV;,
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom)) = S[2]*iV;
                        )
                    }
                    // ELSE, STORE THE STATIC MOMENTS
                    else
                    {
                        AMREX_D_TERM
                        (
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom)) = S[0];,
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom)) = S[1];,
                            elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom)) = S[2];
                        )
                    }
                }
            });
            Gpu::synchronize();
        }
        this->elm_centroid.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // CYCLE AGAIN OVER THE ELEMENTS ------------------------------
        // Next, for extended elements, we sum the contribution of the
        // small elements.
        // ------------------------------------------------------------
        for (MFIter mfi(this->elm_centroid); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<Real const> const & cell_vf_fab = mesh.cell_vf.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);

            Array4<Real> const & elm_centroid_fab = this->elm_centroid.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                // LOCAL VARIABLES
                Real S[AMREX_SPACEDIM], V;
                
                if (ELM_IS_EXTENDED(etype))
                {
                    AMREX_D_TERM
                    (
                        S[0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        S[1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        S[2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )
                    V = cell_vf_fab(i,j,k,dom)*entire_cell_volume;

                    for (int n = 0; n < __DG_EXTENDED_STENCIL_N_NBR__; ++n)
                    {
                        const int nbr_i = i+tables::extended_stencil_i[n];
                        const int nbr_j = j+tables::extended_stencil_j[n];
                        const int nbr_k = k+tables::extended_stencil_k[n];
                        const short nbr_etype = elm_type_fab(nbr_i,nbr_j,nbr_k,ELM_TYPE(dom));
                        
                        if (CELLS_ARE_MERGED(i, j, k, nbr_i, nbr_j, nbr_k, nbr_etype))
                        {
                            AMREX_D_TERM
                            (
                                S[0] += elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                                S[1] += elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                                S[2] += elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                            )
                            V += cell_vf_fab(nbr_i,nbr_j,nbr_k,dom)*entire_cell_volume;
                        }
                    }

                    // STORE THE CENTROID FOR EXTENDED ELEMENTS
                    V = 1.0/V;
                    AMREX_D_TERM
                    (
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom)) = S[0]*V;,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom)) = S[1]*V;,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom)) = S[2]*V;
                    )
                }
            });
            Gpu::synchronize();
        }
        this->elm_centroid.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------

        // REPLACE THE CENTROIDS OF THE SMALL ELEMENTS WITH THAT OF THE
        // EXTENDED ELEMENT THAT EXTENDS INTO THEM --------------------
        for (MFIter mfi(this->elm_centroid); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<Real> const & elm_centroid_fab = this->elm_centroid.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // LOCAL PARAMETERS
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));
                const bool elm_is_small = ELM_IS_SMALL(etype);

                // LOCAL VARIABLES
                int b, nbr_i, nbr_j, nbr_k, nbr_b;

                if (elm_is_small)
                {
                    b = etype/10;
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    AMREX_D_TERM
                    (
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom)) = elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X1(dom));,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom)) = elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X2(dom));,
                        elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom)) = elm_centroid_fab(nbr_i,nbr_j,nbr_k,ELM_CENTROID_X3(dom));
                    )
                }
            });
            Gpu::synchronize();
        }
        this->elm_centroid.FillBoundary(geom.periodicity());
        // ------------------------------------------------------------
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error using the L_inf norm.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] X: a MultiFab object containing the dG coefficients.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[out] err: L_inf-based error.
     * \param[out] err_norm: L_inf-based error normalization.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_error_L_inf(const Real t, const Geometry & geom, const Mesh & mesh, const MultiFab & X, const iMultiFab & mask, Real & err, Real & err_norm, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // DOMAINS / UNKNOWN FIELDS
        const int n_domains = ibvp.get_number_of_domains();
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        MultiFab elm_err;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        elm_err.define(this->elm_type.boxarray, this->elm_type.distributionMap, 2*n_domains, this->dom_data_n_grow);
        elm_err = 0.0;
        // ------------------------------------------------------------

        // EVAL THE ERROR ---------------------------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif
        for (MFIter mfi(elm_err); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);

            Array4<Real> const & elm_err_fab = elm_err.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)) && ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                    const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                    const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];

                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], err_x, err_norm_x, ex, nx;
                    Real U[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);

                    // INITIALIZATION
                    err_x = 0.0;

                    // LOOP OVER THE QUADRATURE POINTS
                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                            x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                            x[2] = x_ptr[x_pos+2]+cell_lo[2];
                        )

                        // EVAL SOLUTION
                        bfu.eval(x, 0, n_unknown_fields, U);

                        // EVAL ERROR
                        ibvp.eval_error_L_inf(dom, t, x, U, ex, nx);
                        err_x = amrex::max(err_x, ex);
                        err_norm_x = amrex::max(err_norm_x, nx);
                    }

                    // STORE ERROR
                    elm_err_fab(i,j,k,2*dom) = err_x;
                    elm_err_fab(i,j,k,2*dom+1) = err_norm_x;
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // GLOBAL ERROR -----------------------------------------------
        err = elm_err.max(0);
        err_norm = elm_err.max(1);
        for (int dom = 1; dom < n_domains; ++dom)
        {
            err = amrex::max(err, elm_err.max(2*dom));
            err_norm = amrex::max(err_norm, elm_err.max(2*dom+1));
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Eval the error using the L_inf norm.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] X: a MultiFab object containing the dG coefficients.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[out] err: L_inf-based error.
     * \param[out] err_norm: L_inf-based error normalization.
     * \param[in] ibvp: initial boundary value problem object.
    */
    template <typename IBVP>
    void eval_FV_error_L_inf(const Real t, const Geometry & geom, const Mesh & mesh, const MultiFab & X, const iMultiFab & mask, Real & err, Real & err_norm, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // DOMAINS / UNKNOWN FIELDS
        const int n_domains = ibvp.get_number_of_domains();
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        MultiFab dX(X.boxarray, X.distributionMap, AMREX_SPACEDIM*n_unknown_fields, 1);
        MultiFab elm_err;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        elm_err.define(this->elm_type.boxarray, this->elm_type.distributionMap, 2*n_domains, this->dom_data_n_grow);
        elm_err = 0.0;
        // ------------------------------------------------------------

        // EVAL THE RECONSTRUCTED SLOPES ------------------------------
        ibvp.eval_FV_slopes(geom, mesh, *this, X, dX);
        // ------------------------------------------------------------

        // EVAL THE ERROR ---------------------------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif
        for (MFIter mfi(elm_err); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<Real const> const & elm_centroid_fab = this->elm_centroid.array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);
            Array4<Real const> const & dX_fab = dX.array(mfi);

            Array4<Real> const & elm_err_fab = elm_err.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)) && ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                    const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                    const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];
                    
                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], err_x, err_norm_x, ex, nx;
                    Real U[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_rx ru(i, j, k, X_fab, dX_fab);
                    AMREX_D_TERM
                    (
                        ru.xc[0] = elm_centroid_fab(i,j,k,ELM_CENTROID_X1(dom));,
                        ru.xc[1] = elm_centroid_fab(i,j,k,ELM_CENTROID_X2(dom));,
                        ru.xc[2] = elm_centroid_fab(i,j,k,ELM_CENTROID_X3(dom));
                    )

                    // INITIALIZATION
                    err_x = 0.0;
                    err_norm_x = 0.0;

                    // LOOP OVER THE QUADRATURE POINTS
                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                            x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                            x[2] = x_ptr[x_pos+2]+cell_lo[2];
                        )

                        // EVAL SOLUTION
                        ru.eval(x, 0, n_unknown_fields, U);

                        // EVAL ERROR
                        ibvp.eval_error_L_inf(dom, t, x, U, ex, nx);
                        err_x = amrex::max(err_x, ex);
                        err_norm_x = amrex::max(err_norm_x, nx);
                    }

                    // STORE ERROR
                    elm_err_fab(i,j,k,2*dom) = err_x;
                    elm_err_fab(i,j,k,2*dom+1) = err_norm_x;
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // GLOBAL ERROR -----------------------------------------------
        err = elm_err.max(0);
        err_norm = elm_err.max(1);
        for (int dom = 1; dom < n_domains; ++dom)
        {
            err = amrex::max(err, elm_err.max(2*dom));
            err_norm = amrex::max(err_norm, elm_err.max(2*dom+1));
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    /**
     * \brief Compute the time step considering an explicit Runge-Kutta scheme.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] X: a MultiFab object containing the dG coefficients.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[in] ibvp: initial boundary value problem object.
     *
     * \return the time step.
    */
    template <typename IBVP>
    Real eval_dt_for_explicit_RKdG(const Real t, const Geometry & geom, const Mesh & mesh, const MultiFab & X, const iMultiFab & mask, const IBVP & ibvp) const
    {
        // PARAMETERS -------------------------------------------------
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

        // DOMAINS / UNKNOWN FIELDS
        const int n_domains = ibvp.get_number_of_domains();
        const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

        // ORDER
        const int sp = this->params.space_p;

        // CFL
        const Real CFL = this->params.time_integration_CFL;

        // MAX REAL
        const Real max_real = std::numeric_limits<Real>::max();
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        MultiFab elm_dt;
        Real dt;
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        elm_dt.define(this->elm_type.boxarray, this->elm_type.distributionMap, n_domains, this->dom_data_n_grow);
        elm_dt = max_real;
        // ------------------------------------------------------------

        // EVAL THE TIME STEP -----------------------------------------
#ifdef AMREX_USE_GPU
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_dev_mem.data();
#else
        Real const * cell_dom_quad_mem_ptr = mesh.cell_dom_quad_host_mem.data();
#endif
        for (MFIter mfi(elm_dt); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();

            Array4<long const> const & cell_dom_quad_info_fab = mesh.cell_dom_quad_info.array(mfi);
            Array4<int const> const & mask_fab = mask.array(mfi);
            Array4<short const> const & elm_type_fab = this->elm_type.array(mfi);
            Array4<Real const> const & X_fab = X.array(mfi);

            Array4<Real> const & elm_dt_fab = elm_dt.array(mfi);

            ParallelFor(bx, n_domains,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
            {
                // ELEMENT TYPE
                const short etype = elm_type_fab(i,j,k,ELM_TYPE(dom));

                if (CELL_IS_NOT_MASKED(mask_fab(i,j,k)) && ELM_IS_NOT_EMPTY(etype))
                {
                    // LOCAL PARAMETERS
                    const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                       prob_lo[1]+j*dx[1],
                                                                       prob_lo[2]+k*dx[2])};
                    const int dom_Nq = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_NQ(dom));
                    const long pos = cell_dom_quad_info_fab(i,j,k,CELL_DOM_QUAD_POS(dom));
                    const Real * x_ptr = &cell_dom_quad_mem_ptr[pos];
                    
                    // LOCAL VARIABLES
                    Real x[AMREX_SPACEDIM], dt_x;
                    Real U[__DG_MAX_N_UNKNOWN_FIELDS__];

                    // UNKNOWN FIELDS
                    space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);

                    // INITIALIZATION
                    dt_x = max_real;

                    // LOOP OVER THE QUADRATURE POINTS
                    for (int q = 0; q < dom_Nq; ++q)
                    {
                        const long x_pos = (AMREX_SPACEDIM+1)*q;
                        AMREX_D_TERM
                        (
                            x[0] = x_ptr[x_pos+0]+cell_lo[0];,
                            x[1] = x_ptr[x_pos+1]+cell_lo[1];,
                            x[2] = x_ptr[x_pos+2]+cell_lo[2];
                        )

                        // EVAL SOLUTION
                        bfu.eval(x, 0, n_unknown_fields, U);

                        // EVAL TIME INCREMENT
                        dt_x = amrex::min(dt_x, ibvp.eval_dt(dom, &dx[0], t, x, U));
                    }

                    // STORE TIME INCREMENT
                    elm_dt_fab(i,j,k,dom) = dt_x;
                }
            });
            Gpu::synchronize();
        }
        // ------------------------------------------------------------

        // GLOBAL MINIMUM TIME INCREMENT ------------------------------
        dt = elm_dt.min(0);
        for (int dom = 1; dom < n_domains; ++dom)
        {
            dt = amrex::min(dt, elm_dt.min(dom));
        }
        dt *= CFL/(1.0+2.0*sp);
        // ------------------------------------------------------------
        return dt;
    }
    
    /**
     * \brief Compute the time increment.
     *
     * \param[in] t: time.
     * \param[in] geom: amrex geometry object.
     * \param[in] mesh: Mesh object; contains information about embedded geometries.
     * \param[in] X: a MultiFab object containing the dG coefficients.
     * \param[in] mask: a iMultiFab object that contains a single value for each cell.
     * \param[in] ibvp: initial boundary value problem object.
     *
     * \return the time increment.
    */
    template <typename IBVP>
    Real eval_dt(const Real t, const Geometry & geom, const Mesh & mesh, const MultiFab & X, const iMultiFab & mask, const IBVP & ibvp) const
    {
        Real dt;

        if (this->time_integration_is_explicit_RKdG())
        {
            dt = this->eval_dt_for_explicit_RKdG(t, geom, mesh, X, mask, ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_Solution.H - Solution::eval_dt\n";
            msg += "| Unexpected input parameters.\n";
            Abort(msg);
        }

        return dt;
    }
    // ================================================================
    
};
// ####################################################################

} // namespace dG
} // namespace amrex

#endif