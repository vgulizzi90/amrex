//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SupersonicVortex.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../IBVP_IdealGas.H"

namespace SupersonicVortex
{

// SUPERSONIC VORTEX: PROBLEM PARAMETERS ##############################
struct ProblemParameters
{
    // DATA MEMBERS ===================================================
    amrex::Real gamma;
    amrex::Real rho_inner, Mach_inner, r_inner, r_outer;
    std::string norm;
    // ================================================================

    // CONSTRUCTOR ====================================================
    ProblemParameters()
    {
        amrex::ParmParse pp;

        pp.get("gamma", this->gamma);
        
        pp.get("rho_inner", this->rho_inner);
        pp.get("Mach_inner", this->Mach_inner);
        pp.get("r_inner", this->r_inner);
        pp.get("r_outer", this->r_outer);
        pp.get("norm", this->norm);
    }
    // ================================================================
};
// ####################################################################



// SUPERSONIC VORTEX: IBVP ############################################
template <int N>
struct Levelset
{
    // DATA MEMBERS ===================================================
    amrex::Real ri, ro;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    Levelset(const amrex::Real r_inner, const amrex::Real r_outer)
    :
    ri{r_inner},
    ro{r_outer},
    dom{0}
    {}
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real ri2 = this->ri*this->ri;
        const amrex::Real ro2 = this->ro*this->ro;
        const T r2 = x[0]*x[0]+x[1]*x[1];
        const T res = (r2-ri2)*(r2-ro2);

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
        const amrex::Real ri2 = this->ri*this->ri;
        const amrex::Real ro2 = this->ro*this->ro;
        const T r2 = x[0]*x[0]+x[1]*x[1];
        const T dr2dx = x[0]+x[0];
        const T dr2dy = x[1]+x[1];
        const T rr = (2.0*r2-ri2+ro2);
        const amrex::GpuArray<T, N> res = {rr*dr2dx, rr*dr2dy};

        return res;
    }
    // ================================================================
};

struct IBVP
{
    // DATA MEMBERS ===================================================
    ProblemParameters problem_params;
    Levelset<AMREX_SPACEDIM> levelset;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    :
    levelset(this->problem_params.r_inner, this->problem_params.r_outer)
    {
    }
    // ================================================================


    // READERS ========================================================
    /**
     * \brief Return the number of considered domains.
     *
    */
    int get_number_of_domains() const
    {
        return 1;
    }
    // ================================================================
};
// ####################################################################



// SUPERSONIC VORTEX: AMR #############################################
struct AMR
:
public amrex::dG::AMR::SinglePatch
{
    // DATA MEMBERS ===================================================
    IBVP ibvp;
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize AMR data.
     * 
    */
    void init()
    {
        // INIT PARENT DATA STRUCTURE ---------------------------------
        amrex::dG::AMR::SinglePatch::init();
        // ------------------------------------------------------------

        // INIT SOLUTION FROM SCRATCH ---------------------------------
        if (this->inputs.restart < 0)
        {
            const amrex::Real t = 0.0;
            this->InitFromScratch(t);
        }
        // ------------------------------------------------------------
        else
        // INIT SOLUTION FROM CHECKPOINT ------------------------------
        {

        }
        // ------------------------------------------------------------

        // UPDATE MASKS -----------------------------------------------
        this->update_masks();
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int mask_n_grow = 1;
        // ------------------------------------------------------------

        // MESH -------------------------------------------------------
        this->meshes[lev]->make_from_scratch(time, this->Geom(lev), ba, dm, this->ibvp);
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(this->meshes[lev]->pruned_ba, this->meshes[lev]->pruned_dm, 1, mask_n_grow);
        this->masks[lev] = 0;
        // ------------------------------------------------------------

        // SET BOX ARRAY AND DISTRIBUTION MAP -------------------------
        this->SetBoxArray(lev, this->meshes[lev]->pruned_ba);
        this->SetDistributionMap(lev, this->meshes[lev]->pruned_dm);
        // ------------------------------------------------------------
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void export_mesh(const int n, const amrex::Real t, const std::string & filename_root)
    {
        this->export_mesh_to_VTK(n, t, filename_root, this->ibvp);
    }
    // ================================================================
};
// ####################################################################

} // SupersonicVortex