//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Utils.cpp
 * \brief Implementations of the functions contained in AMReX_dG_Utils.H.
*/

#include <AMReX_ParmParse.H>

#include <AMReX_dG_Utils.H>

namespace amrex
{
namespace dG
{
// AUXILIARY UTILITIES ################################################
/**
 * \brief Convert a real containing seconds to a string showing hh:mm:ss.
 *
 * \param[in] s: seconds as a real;
 *
 * \return a string showing hh:mm:ss.
 *
*/
std::string seconds_to_hms(const Real s)
{
    const int int_s = (int) std::round(s);
    const int aux = int_s/60;
    const int seconds = int_s%60;
    const int hours = aux/60;
    const int minutes = aux%60;

    std::string res;
    res = std::to_string(hours)+":";
    res += ((minutes < 10) ? ("0"+std::to_string(minutes)) : std::to_string(minutes))+":";
    res += ((seconds < 10) ? ("0"+std::to_string(seconds)) : std::to_string(seconds));
    
    return res;
}

/**
 * \brief Modify string by converting keywords from input file.
 *
 * \param[in] s: seconds as a real;
 *
 * \return a string.
 *
*/
std::string replace_input_keywords(const std::string & is)
{
    if (is.find("$") == std::string::npos)
    {
        return is;
    }

    ParmParse pp;
    std::string os;
    Vector<std::string> input_key;

    os = is;
    while (os.find("$") != std::string::npos)
    {
        const size_t p1 = os.find("$(");
        const size_t p2 = os.find(")");
        const std::string key = os.substr(p1+2, p2-p1-2);

        if (key.compare("AMREX_SPACEDIM") == 0)
        {
            os.replace(p1, p2-p1+1, std::to_string(AMREX_SPACEDIM));
        }
        else
        {
            pp.getarr(key.c_str(), input_key);

            std::string ns = input_key[0];
            for (int k = 1; k < input_key.size(); ++k)
            {
                ns += "x"+input_key[k];
            }
            os.replace(p1, p2-p1+1, ns);
        }
    }
    
    return os;
}
// ####################################################################



namespace io
{
// SIMPLE INPUT/OUTPUT ROUTINES #######################################
void good_so_far(const int rank, const MPI_Comm comm, const int n, std::ostream & os)
{
    Print(rank, comm, os) << "good so far: " << n << std::endl;
}
void good_so_far(const int n, std::ostream & os)
{
    good_so_far(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), n, os);
}
void good_so_far(std::ostream & os)
{
    good_so_far(0, os);
}
// ####################################################################



// INPUT/OUTPUT ROUTINES FOR FOLDERS ##################################
/**
 * \brief Build a path from a set of input strings.
 *
 * \param[in] list: the set of input strings.
 *
 * \return a string containing the path.
*/
std::string make_path(const std::initializer_list<std::string> list)
{
    int cnt;
    std::string path;

    cnt = 0;
    for (const std::string & str : list)
    {
        if (cnt == 0)
        {
            path += str;
        }
        else
        {
            if (str.front() == OS_SEP)
            {
                path += str;
            }
            else
            {
                path += OS_SEP+str;
            }
        }

        if (str.back() == OS_SEP)
        {
            path.pop_back();
        }

        cnt += 1;
    }

    return path;
}
// ####################################################################



// INPUT/OUTPUT ROUTINES FOR ARRAYS ###################################
/**
 * \brief Print a list of n space-separated reals on process rank of communicator comm.
 *
 * \param[in] rank: process identifier;
 * \param[in] comm: communicator identifier;
 * \param[in] n: number of reals to be printed;
 * \param[in] src: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_reals(const int rank, const MPI_Comm comm, const int n, const Real * src, std::ostream & os)
{
    for (int i = 0; i < (n-1); ++i)
    {
        Print(rank, comm, os) << std::scientific << std::setprecision(11) << std::setw(18) << std::showpos << src[i] << " ";
    }
    Print(rank, comm, os) << std::scientific << std::setprecision(11) << std::setw(18) << std::showpos << src[n-1];
}

/**
 * \brief Print a list of n space-separated reals on process rank of the default communicator.
 *
 * \param[in] rank: process identifier;
 * \param[in] n: number of reals to be printed;
 * \param[in] src pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_reals(const int rank, const int n, const Real * src, std::ostream & os)
{
    print_reals(rank, ParallelContext::CommunicatorSub(), n, src, os);
}

/**
 * \brief Print a list of n space-separated reals on I/O process of the default communicator.
 *
 * \param[in] n: number of reals to be printed;
 * \param[in] src: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_reals(const int n, const Real * src, std::ostream & os)
{
    print_reals(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), n, src, os);
}

/**
 * \brief Print an array or reals with column-major order on process rank of communicator comm.
 *
 * \param[in] rank: process identifier;
 * \param[in] comm: communicator identifier;
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] src: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_real_array_2d(const int rank, const MPI_Comm comm, const int Nr, const int Nc, const Real * src, std::ostream & os)
{
    for (int r = 0; r < Nr; ++r)
    {
        for (int c = 0; c < Nc; ++c)
        {
            Print(rank, comm, os) << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << src[r+c*Nr] << " ";
        }
        Print(rank, comm, os) << std::endl;
    }
}

/**
 * \brief Print an array or reals with column-major order on process rank of the default
 *        communicator.
 *
 * \param[in] rank: process identifier;
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] src: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_real_array_2d(const int rank, const int Nr, const int Nc, const Real * src, std::ostream & os)
{
    print_real_array_2d(rank, ParallelContext::CommunicatorSub(), Nr, Nc, src, os);
}

/**
 * \brief Print an array or reals with column-major order on I/O process of the default
 *        communicator.
 *
 * \param[in] Nr: number of rows;
 * \param[in] Nc: number of columns;
 * \param[in] src: pointer to memory;
 * \param[inout] os: output stream;
 *
*/
void print_real_array_2d(const int Nr, const int Nc, const Real * src, std::ostream & os)
{
    print_real_array_2d(ParallelContext::IOProcessorNumberSub(), ParallelContext::CommunicatorSub(), Nr, Nc, src, os);
}
// ####################################################################

} // namespace io



// TIME KEEPER ########################################################
    // TIC/TOC ========================================================
    /**
     * \brief Add a new time window to be tracked.
    */
    void TimeKeeper::tic()
    {
        this->start_time.push_back(second());
    }

    /**
     * \brief Close the time window.
    */
    void TimeKeeper::toc()
    {
        this->elapsed_time = second()-this->start_time.back();
        this->start_time.pop_back();

        ParallelDescriptor::ReduceRealMax(this->elapsed_time, ParallelDescriptor::IOProcessorNumber());
    }
    // ================================================================
    

    // ELAPSED TIME ===================================================
    /**
     * \brief Return the elapsed time.
    */
    Real TimeKeeper::get_elapsed_time_in_seconds() const
    {
        return this->elapsed_time;
    }

    /**
     * \brief Return the elapsed time as a string in hh:mm:ss format.
    */
    std::string TimeKeeper::get_elapsed_time_in_hms() const
    {
        return seconds_to_hms(this->elapsed_time);
    }
    // ================================================================
// ####################################################################



// INPUT READERS ######################################################
    // READ INPUT FILE ================================================
    void InputReaderBase::read_input_file()
    {
        // PARAMETERS -------------------------------------------------
        const time_t date_and_time = time(0);
        // ------------------------------------------------------------
        
        // VARIABLES --------------------------------------------------
        ParmParse pp;

        int tmp_int;
        // ------------------------------------------------------------

        // AMR REGRID -------------------------------------------------
        pp.query("amr.regrid_int", this->amr_regrid_int);
        // ------------------------------------------------------------

        // WALL TIME --------------------------------------------------
        pp.query("wall_time", this->wall_time);
        pp.query("wall_time_units", this->wall_time_units);

        if (this->wall_time_units.compare("h") == 0)
        {
            this->wall_time_s = this->wall_time*3600.0;
        }
        else if (this->wall_time_units.compare("m") == 0)
        {
            this->wall_time_s = this->wall_time*60.0;
        }
        else
        {
            this->wall_time_s = this->wall_time;
        }
        // ------------------------------------------------------------

        // OUTPUT FILES -----------------------------------------------
        this->output_folderpath = "out"+std::to_string(date_and_time);
        this->checkpoint_filename = "chk"+std::to_string(date_and_time);
        this->plot_filename = "plt"+std::to_string(date_and_time);
        
        pp.query("output_folderpath", this->output_folderpath);
        pp.query("checkpoint_filename", this->checkpoint_filename);
        pp.query("checkpoint_int", this->checkpoint_int);
        pp.query("plot_filename", this->plot_filename);
        pp.query("plot_int", this->plot_int);

        tmp_int = 1;
        pp.query("output_overwrite", tmp_int);
        this->output_overwrite = (tmp_int > 0);

        this->output_folderpath = replace_input_keywords(this->output_folderpath);
        // ------------------------------------------------------------

        // RESTART ----------------------------------------------------
        pp.query("restart", this->restart);
        // ------------------------------------------------------------
    }
    // ================================================================


    // OUTPUT FOLDERS =================================================
    std::string InputReaderBase::get_step_string(const int n) const
    {
        int n_digits;
        n_digits = 1;
        while (std::pow(10, n_digits) <= 1) n_digits += 1;
        const std::string step_string = Concatenate("", n, n_digits);
        
        return step_string;
    }

    std::string InputReaderBase::get_output_folderpath() const
    {
        return this->output_folderpath;
    }

    std::string InputReaderBase::get_output_step_folderpath(const int n) const
    {
        int n_digits;
        n_digits = 1;
        while (std::pow(10, n_digits) <= 1) n_digits += 1;

        const std::string step_folder = Concatenate("Step_", n, n_digits);
        const std::string level_step_folderpath = io::make_path({this->output_folderpath, step_folder});

        return level_step_folderpath;
    }

    void InputReaderBase::make_step_output_folder(const int n) const
    {
        // CREATE OUTPUT DIRECTORY ------------------------------------
        if (this->output_overwrite)
        {
            UtilCreateDirectory(this->output_folderpath, 0755);
        }
        else
        {
            UtilCreateCleanDirectory(this->output_folderpath, 0755);
        }
        // ------------------------------------------------------------

        // CREATE STEP DIRECTORIES ------------------------------------
        UtilCreateDirectory(this->get_output_step_folderpath(n), 0755);
        // ------------------------------------------------------------
    }
    // ================================================================

    
    // READ INPUT FILE ================================================
    void InputReaderSinglePatch::read_input_file()
    {
        // CALL PARENT METHOD -----------------------------------------
        InputReaderBase::read_input_file();
        // ------------------------------------------------------------

        // ANALYSIS TIME ----------------------------------------------
        {
            ParmParse pp("time");

            pp.get("T", this->time.T);
            pp.get("n_steps", this->time.n_steps);
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // READERS ========================================================
    bool InputReaderSinglePatch::plot(const int n, const Real t) const
    {
        bool res;
        res = (n%(this->plot_int) == 0);
        res = res || (n == this->time.n_steps);
        res = res || (std::abs(t/this->time.T-1.0) < 1.0e-12);
        res = res && (this->plot_int > 0);

        return res;
    }
    bool InputReaderSinglePatch::regrid(const int n) const
    {
        bool res;
        res = (n%(this->amr_regrid_int) == 0);
        res = res && (this->amr_regrid_int > 0);

        return res;
    }
    // ================================================================


    // OUTPUT FOLDERS =================================================
    std::string InputReaderSinglePatch::get_step_string(const int n) const
    {
        int n_digits;
        n_digits = 1;
        while (std::pow(10, n_digits) <= this->time.n_steps) n_digits += 1;
        const std::string step_string = Concatenate("", n, n_digits);
        
        return step_string;
    }

    std::string InputReaderSinglePatch::get_level_folderpath(const int lev) const
    {
        const std::string level_folder = "Level_"+std::to_string(lev);
        const std::string level_folderpath = io::make_path({this->output_folderpath, level_folder});

        return level_folderpath;
    }

    std::string InputReaderSinglePatch::get_level_step_folderpath(const int lev, const int n) const
    {
        int n_digits;
        n_digits = 1;
        while (std::pow(10, n_digits) <= this->time.n_steps) n_digits += 1;

        const std::string level_folder = "Level_"+std::to_string(lev);
        const std::string step_folder = Concatenate("Step_", n, n_digits);
        const std::string level_step_folderpath = io::make_path({this->output_folderpath, level_folder, step_folder});

        return level_step_folderpath;
    }
    // ================================================================
// ####################################################################

} // namespace dG
} // namespace amrex