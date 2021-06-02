//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_AMR.cpp
 * \brief Contains implementations of functions for AMR operations.
*/

#include <AMReX_ParmParse.H>

#include <AMReX_dG_AMR.H>

namespace amrex
{
namespace dG
{
namespace amr
{

// ADAPTIVE MESH REFINEMENT CLASS #####################################
    // CONSTRUCTOR ====================================================
    SinglePatch::SinglePatch()
    {
        const int n_levels = this->max_level+1;

        this->meshes.resize(n_levels);
        for (int lev = 0; lev < n_levels; ++lev)
        {
            this->meshes[lev] = new Mesh();
        }

        this->solutions.resize(n_levels);
        for (int lev = 0; lev < n_levels; ++lev)
        {
            this->solutions[lev] = new Solution();
        }

        this->masks.resize(n_levels);

        this->linear_direct_solver_finest_level = this->finest_level;
        this->linear_direct_solver = "";
    }
    // ================================================================


    // DESTRUCTOR =====================================================
    SinglePatch::~SinglePatch()
    {
        const int n_levels = this->max_level+1;

        for (int lev = 0; lev < n_levels; ++lev)
        {
            if (this->solutions[lev] != nullptr)
            {
                delete this->solutions[lev];
            }
        }
        
        for (int lev = 0; lev < n_levels; ++lev)
        {
            if (this->meshes[lev] != nullptr)
            {
                delete this->meshes[lev];
            }
        }
    }
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize input data.
     * 
    */
    void SinglePatch::init_inputs()
    {
        // PARAMETERS -------------------------------------------------
        const int n_levels = this->max_level+1;
        // ------------------------------------------------------------
        
        // INIT GENERAL INPUTS ----------------------------------------
        this->inputs.read_input_file();
        // ------------------------------------------------------------

        // INIT MESHES INPUTS -----------------------------------------
        for (int lev = 0; lev < n_levels; ++lev)
        {
            ParmParse pp("mesh["+std::to_string(lev)+"]");
            this->meshes[lev]->read_input_file(pp);
        }
        // ------------------------------------------------------------

        // INIT SOLUTIONS INPUTS --------------------------------------
        for (int lev = 0; lev < n_levels; ++lev)
        {
            ParmParse pp("solution["+std::to_string(lev)+"]");
            this->solutions[lev]->read_input_file(pp);
        }
        // ------------------------------------------------------------

        // LINEAR SOLVER ----------------------------------------------
        {
            ParmParse pp;

            pp.query("linear_direct_solver_finest_level", this->linear_direct_solver_finest_level);
            pp.query("linear_direct_solver", this->linear_direct_solver);
        }

        if (this->linear_direct_solver_is_pardiso())
        {
            ParmParse pp("pardiso");

            pp.get("mtype", this->pardiso.mtype);
        }
        // ------------------------------------------------------------
    }
    
    /**
     * \brief Initialize AMR data.
     * 
    */
    void SinglePatch::init()
    {
        this->init_inputs();
    }
    // ================================================================


    // READERS ========================================================
    bool SinglePatch::advance_in_time_continues(const int n, const Real t) const
    {
        const int N = this->inputs.time.n_steps;
        const Real T = this->inputs.time.T;
        bool cond;

        cond = (n < N);
        cond = cond && (t < T*(1.0-1.0e-12));

        return cond;
    }

    int SinglePatch::get_largest_dG_space_p() const
    {
        int sp;

        sp = 0;
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            sp = amrex::max(sp, this->solutions[lev]->params.space_p);
        }

        return sp;
    }
    // ================================================================


    // CHECK QUADRATURE RULES =========================================
    /**
     * \brief Check the quadrature rules by comparing the domains' volume and surface with exact values.
     *
     * \param[in] n_domains: number of domains to be considered.
     * \param[in] exact_volume: exact_volume[dom] should contain the exact value of the volume of the
     *                          dom-th domain.
     * \param[in] exact_surface: exact_surface[dom] should contain the exact value of the surface of the
     *                           dom-th domain.
     *
    */
    void SinglePatch::check_quadrature_rules(const int n_domains,
                                             const Real * exact_volume,
                                             const Real * exact_surface) const
    {
        // VARIABLES --------------------------------------------------
        Vector<Real> computed_volume(n_domains);
        Vector<Real> computed_volume_div(n_domains);
        Vector<Real> computed_surface(n_domains);
        // ------------------------------------------------------------

        // INITIALIZATION ---------------------------------------------
        for (int dom = 0; dom < n_domains; ++dom)
        {
            computed_volume[dom] = 0.0;
            computed_volume_div[dom] = 0.0;
            computed_surface[dom] = 0.0;
        }
        // ------------------------------------------------------------

        // COMPUTE VALUES ---------------------------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const iMultiFab & mask = this->masks[lev];

            Vector<Real> lev_computed_volume(n_domains);
            Vector<Real> lev_computed_volume_div(n_domains);
            Vector<Real> lev_computed_surface(n_domains);

            this->meshes[lev]->eval_volumes_and_surfaces(this->Geom(lev),
                                                         n_domains,
                                                         mask,
                                                         lev_computed_volume.data(),
                                                         lev_computed_volume_div.data(),
                                                         lev_computed_surface.data());

            for (int dom = 0; dom < n_domains; ++dom)
            {
                computed_volume[dom] += lev_computed_volume[dom];
                computed_volume_div[dom] += lev_computed_volume_div[dom];
                computed_surface[dom] += lev_computed_surface[dom];
            }
        }
        // ------------------------------------------------------------

        // PRINT REPORT -----------------------------------------------
        Print() << "EMBEDDED-MESH QUADRATURE REPORT:" << std::endl;
        for (int dom = 0; dom < n_domains; ++dom)
        {
            Real err;

            Print() << "| Domain " << dom << std::endl;
            
            Print() << "|  Volume: " << std::endl;
            
            Print() << "|  - Reference: " << exact_volume[dom] << std::endl;

            err = 100.0*std::abs(computed_volume[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via dom. quadrature: " << computed_volume[dom] << ", error (%): " << err << std::endl;
            
            err = 100.0*std::abs(computed_volume_div[dom]-exact_volume[dom])/exact_volume[dom];
            Print() << "|  - Computed via bou. quadrature: " << computed_volume_div[dom] << ", error (%): " << err << std::endl;

            Print() << "|  Surface: " << std::endl;

            if (exact_surface[dom] == 0.0)
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = std::abs(computed_surface[dom]-exact_surface[dom]);
                Print() << "|  - Computed: " << computed_surface[dom] << ", error: " << err << std::endl;
            }
            else
            {
                Print() << "|  - Reference: " << exact_surface[dom] << std::endl;

                err = 100.0*std::abs(computed_surface[dom]-exact_surface[dom])/exact_surface[dom];
                Print() << "|  - Computed: " << computed_surface[dom] << ", error (%): " << err << std::endl;
            }
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Check the quadrature rules by comparing the domain's volume and surface with exact values.
     *
     * \param[in] exact_volume: exact value of the volume of the domain.
     * \param[in] exact_surface: exact value of the surface of the domain.
     *
    */
    void SinglePatch::check_quadrature_rules(const Real exact_volume,
                                             const Real exact_surface) const
    {
        const int n_domains = 1;
        this->check_quadrature_rules(n_domains, &exact_volume, &exact_surface);
    }
    // ================================================================


    // LINEAR SOLVER ==================================================
    bool SinglePatch::linear_direct_solver_is_pardiso() const
    {
        return (this->linear_direct_solver.compare("pardiso") == 0);
    }

    bool SinglePatch::linear_direct_solver_first_call() const
    {
        bool first_call;

        first_call = true;

        if (this->linear_direct_solver_is_pardiso())
        {
            first_call = this->pardiso.first_call;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR.cpp - SinglePatch::linear_direct_solver_first_call\n";
            msg += "| Unexpected input parameters.\n";
            msg += "| Linear solver: "+this->linear_direct_solver+".\n";
            Abort(msg);
        }

        return first_call;
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    /**
     * \brief Make output folder path for the current step.
     *
     * \param[in] n: time step index.
     * \param[in] t: time.
    */
    void SinglePatch::make_step_output_folder(const int n, const Real t) const
    {
        // CREATE OUTPUT DIRECTORY ------------------------------------
        if (this->inputs.output_overwrite)
        {
            UtilCreateDirectory(this->inputs.output_folderpath, 0755);
        }
        else
        {
            UtilCreateCleanDirectory(this->inputs.output_folderpath, 0755);
        }
        // ------------------------------------------------------------

        // CREATE LEVEL/STEP DIRECTORIES ------------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            UtilCreateDirectory(this->inputs.get_level_step_folderpath(lev, n), 0755);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
// ####################################################################


} // namespace amr
} // namespace dG
} // namespace amrex