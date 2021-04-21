//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_Solution.cpp
 * \brief Contains implementations of functions for discontinuous Galerkin solution objects.
*/

#include <AMReX_dG_Solution.H>

namespace amrex
{
namespace dG
{

// DG SOLUTION CLASS ##################################################
    // INITIALIZATION =================================================
    void Solution::read_input_file(const ParmParse & pp)
    {
        // VARIABLES --------------------------------------------------
        int tmp_int;
        // ------------------------------------------------------------

        // READ PARAMETERS --------------------------------------------
        pp.query("time_p", this->params.time_p);
        pp.query("space_p", this->params.space_p);

        tmp_int = 0;
        pp.query("FV_is_active", tmp_int);
        this->params.FV_is_active = (tmp_int > 0);

        tmp_int = 0;
        pp.query("cell_merging_is_active", tmp_int);
        this->params.cell_merging_is_active = (tmp_int > 0);

        if (this->params.cell_merging_is_active)
        {
            pp.get("cell_merging_volume_fraction_threshold", this->params.cell_merging_volume_fraction_threshold);
        }

        pp.query("time_integration", this->params.time_integration);
        pp.query("time_integration_CFL", this->params.time_integration_CFL);
        // ------------------------------------------------------------
    }
    // ================================================================


    // READERS ========================================================
    /**
     * \brief Return true if the cell merging is active.
     *
     * \return params.cell_merging_is_active.
     *
    */
    bool Solution::uses_cell_merging() const
    {
        return (this->params.cell_merging_is_active);
    }

    /**
     * \brief Return true if time integration is performed via explicit Runge-Kutta algorithms.
     *
     * \return true if params.time_integration == "explicit_RKdG".
     *
    */
    bool Solution::time_integration_is_explicit_RKdG() const
    {
        return (this->params.time_integration.compare("explicit_RKdG") == 0);
    }
    // ================================================================
// ####################################################################

} // namespace dG
} // namespace amrex