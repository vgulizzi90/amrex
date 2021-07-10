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
            ParmParse pp("amr");

            pp.query("linear_direct_solver_finest_level", this->linear_direct_solver_finest_level);
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

    /**
     * \brief Make checkpoint folder path for the current step.
     *
     * \param[in] n: time step index.
     * \param[in] t: time.
    */
    void SinglePatch::make_step_checkpoint_folder(const int n, const Real t) const
    {
        // CREATE LEVEL/CHECKPOINT DIRECTORIES ------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            UtilCreateDirectory(this->inputs.get_level_checkpoint_folderpath(lev, n), 0755);
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Write the general header file for the checkpoint.
     *
     * \param[in] n: time step index.
     * \param[in] t: time.
     * \param[in] filename_root: root of the general header file for the checkpoint.
    */
    void SinglePatch::write_checkpoint_header_file(const int n, const Real t, const std::string & filename_root) const
    {
        // PARAMETERS -------------------------------------------------
        const std::string step_string = this->inputs.get_step_string(n);
        // ------------------------------------------------------------

        // HEADER FILE ------------------------------------------------
        if (ParallelDescriptor::IOProcessor())
        {
            const std::string header_filepath = io::make_path({this->inputs.output_folderpath, filename_root+"_header_"+step_string+".txt"});
            time_t date_and_time = time(0);
            char * date_and_time_ = ctime(&date_and_time);
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

            std::ofstream fp;
            fp.precision(17);
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

            fp.open(header_filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            if(!fp.good())
            {
                FileOpenFailed(header_filepath);
            }
            
            fp << std::endl << "CHECKPOINT HEADER FILE - " << date_and_time_ << "\n";
            fp << "| Number of levels: " << this->finest_level+1 << std::endl;
            fp << "| time: " << t << std::endl;
            fp.close();
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Write the multifab header file for the checkpoint.
     *
     * \param[in] n: time step index.
     * \param[in] t: time.
     * \param[in] filename_root: root of the multifab header file for the checkpoint.
     * \param[in] X: vector of multifabs containing the solution at each level.
     * \param[in] L: vector of multifabs containing the level set at each level.
    */
    void SinglePatch::write_checkpoint_multifabs_header_file(const int n, const Real /*t*/, const std::string & filename_root,
                                                             const Vector<MultiFab> & X, const Vector<MultiFab> & L) const
    {
        // PARAMETERS -------------------------------------------------
        const std::string step_string = this->inputs.get_step_string(n);
        // ------------------------------------------------------------

        // HEADER FILE FOR MULTIFABS ----------------------------------
        if (ParallelDescriptor::IOProcessor())
        {
            const std::string header_filepath = io::make_path({this->inputs.output_folderpath, filename_root+"_header_multifabs_"+step_string+".txt"});
            VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

            std::ofstream fp;
            fp.precision(17);
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            
            fp.open(header_filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
            if(!fp.good())
            {
                FileOpenFailed(header_filepath);
            }

            fp << "Checkpoint file\n";
            fp << this->finest_level << "\n";
            for (int lev = 0; lev <= this->finest_level; ++lev)
            {
                {
                    this->boxArray(lev).writeOn(fp);
                    fp << '\n';
                    fp << L[lev].n_comp << "\n";
                    fp << L[lev].n_grow << "\n";
                    fp << X[lev].n_comp << "\n";
                    fp << X[lev].n_grow << "\n";
                }
            }
            
            fp.close();
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Write a checkpoint for a solution using a projected level set.
     *
     * \param[in] n: time step index.
     * \param[in] t: time.
     * \param[in] filename_root: root of the filename of the checkpoint files.
     * \param[in] X: vector of multifabs containing the solution at each level.
     * \param[in] L: vector of multifabs containing the level set at each level.
    */
    void SinglePatch::write_checkpoint_using_projected_level_set(const int n, const Real t, const std::string & filename_root,
                                                                 const Vector<MultiFab> & X, const Vector<MultiFab> & L) const
    {
        // HEADER FILE ------------------------------------------------
        this->write_checkpoint_header_file(n, t, filename_root);
        // ------------------------------------------------------------

        // HEADER FILE FOR MULTIFABS ----------------------------------
        this->write_checkpoint_multifabs_header_file(n, t, filename_root, X, L);
        // ------------------------------------------------------------

        // SOLUTION AND LEVEL SET MULTIFABS ---------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_checkpoint_folderpath = this->inputs.get_level_checkpoint_folderpath(lev, n);

            const std::string filepath_X = io::make_path({level_checkpoint_folderpath, filename_root+"_X"});
            const std::string filepath_L = io::make_path({level_checkpoint_folderpath, filename_root+"_L"});

            VisMF::Write(X[lev], filepath_X);
            VisMF::Write(L[lev], filepath_L);
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Read the general header file for the checkpoint.
     *
     * \param[in] restart_n: index of the restart checkpoint.
     * \param[in] filename_root: root of the general header file for the checkpoint.
    */
    void SinglePatch::read_checkpoint_header_file(const int restart_n, const std::string & filename_root)
    {
        // PARAMETERS -------------------------------------------------
        const std::string step_string = this->inputs.get_step_string(restart_n);
        // ------------------------------------------------------------

        // HEADER FILE ------------------------------------------------
        {
            const std::string header_filepath = io::make_path({this->inputs.output_folderpath, filename_root+"_header_"+step_string+".txt"});
            std::string line;
            std::ifstream fp;

            fp.open(header_filepath);
            while (std::getline(fp, line))
            {
                if (line.find("time") != std::string::npos)
                {
                    std::istringstream is(line.substr(line.find(":")+1));
                    is >> this->inputs.restart_time;
                }
            }
            fp.close();
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Read the multifab header file for the checkpoint.
     *
     * \param[in] restart_n: index of the restart checkpoint.
     * \param[in] filename_root: root of the multifab header file for the checkpoint.
     * \param[in] X: vector of multifabs containing the solution at each level.
     * \param[in] L: vector of multifabs containing the level set at each level.
    */
    void SinglePatch::read_checkpoint_multifabs_header_file(const int restart_n, const std::string & filename_root,
                                                            Vector<MultiFab> & X, Vector<MultiFab> & L)
    {
        // PARAMETERS -------------------------------------------------
        const std::string step_string = this->inputs.get_step_string(restart_n);

        constexpr std::streamsize bl_ignore_max{100000};
        // ------------------------------------------------------------

        // HEADER FILE FOR MULTIFABS ----------------------------------
        {
            const std::string header_filepath = io::make_path({this->inputs.output_folderpath, filename_root+"_header_multifabs_"+step_string+".txt"});
            VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

            Vector<char> file_char_ptr;
            ParallelDescriptor::ReadAndBcastFile(header_filepath, file_char_ptr);
            std::string file_char_ptr_str(file_char_ptr.dataPtr());
            std::istringstream is(file_char_ptr_str, std::istringstream::in);

            std::string line;
            int n_comp;
            IntVect n_grow;

            std::getline(is, line);
            is >> this->finest_level;
            is.ignore(bl_ignore_max, '\n');
            for (int lev = 0; lev <= this->max_level; ++lev)
            {
                BoxArray ba;
                ba.readFrom(is);
                is.ignore(bl_ignore_max, '\n');
                DistributionMapping dm(ba, ParallelDescriptor::NProcs());

                this->SetBoxArray(lev, ba);
                this->SetDistributionMap(lev, dm);

                is >> n_comp;
                is >> n_grow;
                
                L[lev].define(this->grids[lev], this->dmap[lev], n_comp, n_grow);
                
                is >> n_comp;
                is >> n_grow;
                
                X[lev].define(this->grids[lev], this->dmap[lev], n_comp, n_grow);
            }
        }
        // ------------------------------------------------------------
    }

    /**
     * \brief Read a checkpoint for a solution using a projected level set.
     *
     * \param[in] restart_n: index of the restart checkpoint.
     * \param[in] filename_root: root of the filename of the checkpoint files.
     * \param[in] X: vector of multifabs containing the solution at each level.
     * \param[in] L: vector of multifabs containing the level set at each level.
    */
    void SinglePatch::read_checkpoint_using_projected_level_set(const int restart_n, const std::string & filename_root,
                                                                Vector<MultiFab> & X, Vector<MultiFab> & L)
    {
        // HEADER FILE ------------------------------------------------
        this->read_checkpoint_header_file(restart_n, filename_root);
        // ------------------------------------------------------------

        // HEADER FILE FOR MULTIFABS ----------------------------------
        this->read_checkpoint_multifabs_header_file(restart_n, filename_root, X, L);
        // ------------------------------------------------------------

        // SOLUTION AND LEVEL SET MULTIFABS ---------------------------
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_checkpoint_folderpath = this->inputs.get_level_checkpoint_folderpath(lev, restart_n);

            const std::string filepath_X = io::make_path({level_checkpoint_folderpath, filename_root+"_X"});
            const std::string filepath_L = io::make_path({level_checkpoint_folderpath, filename_root+"_L"});

            VisMF::Read(X[lev], filepath_X);
            VisMF::Read(L[lev], filepath_L);
        }
        // ------------------------------------------------------------
    }
    // ================================================================
// ####################################################################


} // namespace amr
} // namespace dG
} // namespace amrex