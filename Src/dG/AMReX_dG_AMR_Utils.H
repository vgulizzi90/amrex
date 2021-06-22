//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_AMR_Utils.H
 * \brief Functions for AMR computations.
*/

#ifndef AMREX_DG_AMR_UTILS_H_
#define AMREX_DG_AMR_UTILS_H_

#include <AMReX_dG_StandardElements.H>
#include <AMReX_dG_Solution_RungeKutta.H>
#include <AMReX_dG_AMR.H>

namespace amrex
{
namespace dG
{
// ADVANCE IN TIME: GENERAL ###########################################
// TIME INCREMENT =====================================================
/**
 * \brief Compute the time increment.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFabs containing the solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
Real eval_dt(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & X, const IBVP & ibvp)
{
    const int finest_level = amr.finestLevel();
    Real dt;

    dt = std::numeric_limits<Real>::max();
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        const iMultiFab & mask = amr.masks[lev];
        const Real dt_lev = solution.eval_dt(t, amr.Geom(lev), mesh, X[lev], mask, ibvp);

        dt = amrex::min(dt, dt_lev);
    }

    return dt;
}
// ====================================================================
// ####################################################################


namespace hyperbolic_pde
{
// ADVANCE IN TIME ####################################################
// TIME INCREMENT =====================================================
/**
 * \brief Compute the time increment.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFabs containing the solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
Real eval_dt(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & X, const IBVP & ibvp)
{
    const int finest_level = amr.finestLevel();
    Real dt;

    dt = std::numeric_limits<Real>::max();
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        const iMultiFab & mask = amr.masks[lev];
        const Real dt_lev = solution.eval_dt(t, amr.Geom(lev), mesh, X[lev], mask, ibvp);

        dt = amrex::min(dt, dt_lev);
    }

    return dt;
}
// ====================================================================


// TAKE TIME STEP USING EXPLICIT RUNGE-KUTTA ALGORITHMS ===============
/**
 * \brief Get the order of the Runge-Kutta algorithm to match the space discretization order.
 *
 * \param[in] amr: SinglePath AMR object.
 *
 * \return the time increment.
*/
int get_explicit_RKdG_order(const amr::SinglePatch & amr)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    int RK_order;
    // ----------------------------------------------------------------

    // FIND THE HIGHEST SPACE DISCRETIZATION ORDER --------------------
    RK_order = 1;
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];
        
        int RK_order_lev;

        RK_order_lev = solution.params.space_p+1;
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            RK_order_lev = 2;
        }

        RK_order = amrex::max(RK_order, RK_order_lev);
    }
    // ----------------------------------------------------------------

    return RK_order;
}

/**
 * \brief Eval the time derivatives of the dG coefficients
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFab objects containing the dG coefficients.
 * \param[in] dXdt: a vector of MultiFab objects that will contain the time derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void eval_dXdt_explicit_RKdG(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & X, Vector<MultiFab> & dXdt, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

    // DG ORDER / FX OFFSET
    const int largest_sp = amr.get_largest_dG_space_p();
    const int largest_sNp = AMREX_D_TERM((1+largest_sp),*(1+largest_sp),*(1+largest_sp));
    const int FX_offset = largest_sNp*n_unknown_fields;
    const int FX_n_comp = 2*FX_offset;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_levels);
    Vector<MultiFab> dX(n_levels);
    // ----------------------------------------------------------------

    // INITIALIZE FLUX MULTIFABS --------------------------------------
    // FLUXES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const BoxArray ba = X[lev].boxarray;
#if (AMREX_SPACEDIM == 1)
        const BoxArray fc_ba[1] = {convert(ba, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(ba, IntVect(1,0)),
                                   convert(ba, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(ba, IntVect(1,0,0)),
                                   convert(ba, IntVect(0,1,0)),
                                   convert(ba, IntVect(0,0,1))};
#endif
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            FX[lev][dir].define(fc_ba[dir], X[lev].distributionMap, FX_n_comp, 0);
            FX[lev][dir] = 0.0;
        }
    }

    // SLOPES FOR FINITE VOLUME SCHEMES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            dX[lev].define(X[lev].boxarray, X[lev].distributionMap, AMREX_SPACEDIM*n_unknown_fields, X[lev].n_grow);
            
            ibvp.eval_FV_slopes(amr.Geom(lev), mesh, solution, X[lev], dX[lev]);
        }
    }

    // TIME DERIVATIVES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        dXdt[lev] = 0.0;
    }
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];

        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            if (dX[lev].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_dG_AMR_Utils.H - eval_dXdt_explicit_RKdG\n";
                msg += "| Reconstructed slopes multifab dX contains nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
    // ----------------------------------------------------------------
#endif

    // DOMAIN INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES -----------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            // ...
        }
        // HIGH-ORDER METHODS
        else
        {
            explicit_RK::add_dom_contrib_to_dXdt(t, geom, mesh, solution, X[lev], dXdt[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------

    // BOUNDARY INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES ---------
    // COMPUTE BOUNDARY FLUXES: LEVEL 0
    {
        const int lev = 0;
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        
        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            explicit_RK::eval_FV_bou_fluxes(t, geom, mesh, solution, X[lev], dX[lev], FX[lev], FX_offset, ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::eval_bou_fluxes(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, ibvp);
        }
    }

    // COMPUTE BOUNDARY FLUXES: LEVEL > 0
    for (int lev = 1; lev < n_levels; ++lev)
    {
        const Geometry & c_geom = amr.Geom(lev-1);
        const Mesh & c_mesh = *amr.meshes[lev-1];
        const Solution & c_solution = *amr.solutions[lev-1];
        const MultiFab & c_X = X[lev-1];
        const IntVect rr = amr.refRatio(lev-1);
        const Geometry & f_geom = amr.Geom(lev);
        const Mesh & f_mesh = *amr.meshes[lev];
        const Solution & f_solution = *amr.solutions[lev];
        const MultiFab & f_X = X[lev];
        const MultiFab & f_dX = dX[lev];
        
        // FINITE VOLUME SCHEME
        if ((f_solution.params.space_p == 0) && f_solution.params.FV_is_active)
        {
            explicit_RK::eval_dG_FV_bou_fluxes(t,
                                               c_geom, c_mesh, c_solution, c_X,
                                               rr,
                                               f_geom, f_mesh, f_solution, f_X, f_dX,
                                               FX[lev], FX_offset,
                                               ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::eval_dG_dG_bou_fluxes(t,
                                               c_geom, c_mesh, c_solution, c_X,
                                               rr,
                                               f_geom, f_mesh, f_solution, f_X,
                                               FX[lev], FX_offset,
                                               ibvp);
        }
    }

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN
    for (int lev = 0; lev < n_levels; ++lev)
    {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            if (FX[lev][dir].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_dG_AMR_Utils.H - eval_dXdt_explicit_RKdG\n";
                msg += "| Fluxes in direction "+std::to_string(dir)+" contains nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
#endif

    // COPY FLUXES FROM FINE TO COARSE LEVEL
    for (int lev = (n_levels-1); lev > 0; --lev)
    {
        average_down_faces(GetArrOfConstPtrs(FX[lev]), GetArrOfPtrs(FX[lev-1]), amr.refRatio(lev-1), amr.Geom(lev-1));
    }

    // ACTUALLY ADD THE BOUNDARY INTEGRAL CONTRIBUTION
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            explicit_RK::add_FV_bou_contrib_to_dXdt(t, geom, mesh, solution, X[lev], dX[lev], FX[lev], FX_offset, dXdt[lev], ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::add_bou_contrib_to_dXdt(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, dXdt[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ----
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        Solution & solution = *amr.solutions[lev];
        
        solution.add_small_elements_contribution(geom, dXdt[lev], ibvp);
        solution.multiply_by_inverse_space_mass_matrix(geom, dXdt[lev], ibvp);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Advance in time using explicit Runge-Kutta algorithms.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[inout] X: a vector of MultiFabs that will contain the updated solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void take_time_step_using_explicit_RKdG(const amr::SinglePatch & amr, const Real t, const Real dt, Vector<MultiFab> & X_new, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // RUNGE-KUTTA ORDER
    const int RK_order = get_explicit_RKdG_order(amr);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<MultiFab> X_old(n_levels);
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if (X_new[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+" on entry.\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
#endif

    // SWAP NEW AND OLD -----------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        X_old[lev].define(X_new[lev].boxarray, X_new[lev].distributionMap, X_new[lev].n_comp, X_new[lev].n_grow);
        std::swap(X_old[lev], X_new[lev]);
    }
    // ----------------------------------------------------------------

    // RK1 ------------------------------------------------------------
    if (RK_order == 1)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK1 STEP 1
        // dXdt := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt, ibvp);

        // X_new := X_old+dt*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0, dt, dXdt[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK2 ------------------------------------------------------------
    else if (RK_order == 2)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK2 STEP 1
        // dXdt := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt, ibvp);

        // X1 := X_old+dt*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK2 STEP 2
        // dXdt := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt, ibvp);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 0.5, X_old[lev], 0, 0.5, X1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev], 0.5*dt, dXdt[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK3 ------------------------------------------------------------
    else if (RK_order == 3)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_levels);
        Vector<MultiFab> dXdt2(n_levels);
        Vector<MultiFab> dXdt3(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK3 STEP 1
        // dXdt1 := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt1, ibvp);

        // X1 := X_old+dt*dXdt1
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK3 STEP 2
        // dXdt2 := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt2, ibvp);

        // X1 := X_old+(dt/4)*dXdt1+(dt/4)*dXdt2
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.25*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            MultiFab::Saxpy(X1[lev],                       0.25*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK3 STEP 3
        // dXdt3 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt3, ibvp);

        // X_new := X_old+(dt/6)*dXdt1+(dt/6)*dXdt2+(2*dt/3)*dXdt3
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0,     dt/6.0, dXdt1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                           dt/6.0, dXdt2[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       2.0*dt/3.0, dXdt3[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK4 ------------------------------------------------------------
    else if (RK_order == 4)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_levels);
        Vector<MultiFab> dXdt2(n_levels);
        Vector<MultiFab> dXdt3(n_levels);
        Vector<MultiFab> dXdt4(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt4[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }
        
        // RK4 STEP 1
        // dXdt1 := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt1, ibvp);

        // X1 := X_old+(dt/2)*dXdt1
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.5*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 2
        // dXdt2 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt2, ibvp);

        // X1 := X_old+(dt/2)*dXdt2
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.5*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 3
        // dXdt3 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt3, ibvp);

        // X1 := X_old+dt*dXdt3
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt3[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 4
        // dXdt4 := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt4, ibvp);

        // X_new := X_old+(dt/6)*dXdt1+(dt/3)*dXdt2+(dt/3)*dXdt3+(dt/6)*dXdt4
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0, dt/6.0, dXdt1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/3.0, dXdt2[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/3.0, dXdt3[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/6.0, dXdt4[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if (X_new[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
#endif

    // RESTRICT FINE SOLUTION ONTO THE COARSE LEVEL -------------------
    for (int lev = (n_levels-1); lev > 0; --lev)
    {
        amr_utils::restrict(amr.Geom(lev), *amr.solutions[lev], X_new[lev],
                            amr.refRatio(lev-1),
                            amr.Geom(lev-1), *amr.solutions[lev-1], X_new[lev-1], amr.masks[lev-1],
                            ibvp);
    }
    // ----------------------------------------------------------------
}
// ====================================================================


// TAKE TIME STEP (GENERAL) ===========================================
/**
 * \brief Advance in time.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[inout] X: a vector of MultiFabs that will contain the updated solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void take_time_step(const amr::SinglePatch & amr, const Real t, const Real dt, Vector<MultiFab> & X, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    std::string time_integration;
    // ----------------------------------------------------------------

    // TIME INTEGRATION -----------------------------------------------
    // At this moment, all AMR levels must use the same time
    // integration algorithm.
    // ----------------------------------------------------------------
    {
        const int lev = 0;
        const Solution & solution = *amr.solutions[lev];

        time_integration = solution.params.time_integration;
    }
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];

        if (time_integration.compare(solution.params.time_integration) != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR_Utils.H - take_time_step\n";
            msg += "| At this moment all AMR levels must use the same time integration algorithm.\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------

    // SELECT TIME INTEGRATION ALGORITHM ------------------------------
    {
        const int lev = 0;
        const Solution & solution = *amr.solutions[lev];

        if (solution.time_integration_is_explicit_RKdG())
        {
            take_time_step_using_explicit_RKdG(amr, t, dt, X, ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR_Utils.H - take_time_step\n";
            msg += "| Unexpected time integration algorithm: "+solution.params.time_integration+".\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
}
// ====================================================================
// ####################################################################
} // namespace hyperbolic_pde



namespace parabolic_pde
{
// ADVANCE IN TIME ####################################################
// TAKE TIME STEP USING EXPLICIT RUNGE-KUTTA ALGORITHMS ===============
/**
 * \brief Get the order of the Runge-Kutta algorithm to match the space discretization order.
 *
 * \param[in] amr: SinglePath AMR object.
 *
 * \return the time increment.
*/
int get_explicit_RKdG_order(const amr::SinglePatch & amr)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    int RK_order;
    // ----------------------------------------------------------------

    // FIND THE HIGHEST SPACE DISCRETIZATION ORDER --------------------
    RK_order = 1;
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];
        
        int RK_order_lev;

        RK_order_lev = solution.params.space_p+1;
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            RK_order_lev = 2;
        }

        RK_order = amrex::max(RK_order, RK_order_lev);
    }
    // ----------------------------------------------------------------

    return RK_order;
}

/**
 * \brief Eval the time derivatives of the dG coefficients
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFab objects containing the dG coefficients.
 * \param[in] dXdt: a vector of MultiFab objects that will contain the time derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void eval_dXdt_explicit_RKdG(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & X, Vector<MultiFab> & dXdt, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

    // DG ORDER / FX OFFSET
    const int largest_sp = amr.get_largest_dG_space_p();
    const int largest_sNp = AMREX_D_TERM((1+largest_sp),*(1+largest_sp),*(1+largest_sp));
    const int FX_offset = largest_sNp*n_unknown_fields;
    const int FX_n_comp = 2*FX_offset;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_levels);
    Vector<MultiFab> dX(n_levels);
    // ----------------------------------------------------------------

    // INITIALIZE FLUX MULTIFABS --------------------------------------
    // FLUXES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const BoxArray ba = X[lev].boxarray;
#if (AMREX_SPACEDIM == 1)
        const BoxArray fc_ba[1] = {convert(ba, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(ba, IntVect(1,0)),
                                   convert(ba, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(ba, IntVect(1,0,0)),
                                   convert(ba, IntVect(0,1,0)),
                                   convert(ba, IntVect(0,0,1))};
#endif
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            FX[lev][dir].define(fc_ba[dir], X[lev].distributionMap, FX_n_comp, 0);
            FX[lev][dir] = 0.0;
        }
    }

    // SLOPES FOR FINITE VOLUME SCHEMES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            dX[lev].define(X[lev].boxarray, X[lev].distributionMap, AMREX_SPACEDIM*n_unknown_fields, X[lev].n_grow);
            
            ibvp.eval_FV_slopes(amr.Geom(lev), mesh, solution, X[lev], dX[lev]);
        }
    }

    // TIME DERIVATIVES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        dXdt[lev] = 0.0;
    }
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];

        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            if (dX[lev].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_dG_AMR_Utils.H - eval_dXdt_explicit_RKdG\n";
                msg += "| Reconstructed slopes multifab dX contains nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
    // ----------------------------------------------------------------
#endif

    // DOMAIN INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES -----------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            // ...
        }
        // HIGH-ORDER METHODS
        else
        {
            explicit_RK::add_dom_contrib_to_dXdt(t, geom, mesh, solution, X[lev], dXdt[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------

    // BOUNDARY INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES ---------
    // COMPUTE BOUNDARY FLUXES: LEVEL 0
    {
        const int lev = 0;
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        
        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            explicit_RK::eval_FV_bou_fluxes(t, geom, mesh, solution, X[lev], dX[lev], FX[lev], FX_offset, ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::eval_bou_fluxes(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, ibvp);
        }
    }

    // COMPUTE BOUNDARY FLUXES: LEVEL > 0
    for (int lev = 1; lev < n_levels; ++lev)
    {
        const Geometry & c_geom = amr.Geom(lev-1);
        const Mesh & c_mesh = *amr.meshes[lev-1];
        const Solution & c_solution = *amr.solutions[lev-1];
        const MultiFab & c_X = X[lev-1];
        const IntVect rr = amr.refRatio(lev-1);
        const Geometry & f_geom = amr.Geom(lev);
        const Mesh & f_mesh = *amr.meshes[lev];
        const Solution & f_solution = *amr.solutions[lev];
        const MultiFab & f_X = X[lev];
        const MultiFab & f_dX = dX[lev];
        
        // FINITE VOLUME SCHEME
        if ((f_solution.params.space_p == 0) && f_solution.params.FV_is_active)
        {
            explicit_RK::eval_dG_FV_bou_fluxes(t,
                                               c_geom, c_mesh, c_solution, c_X,
                                               rr,
                                               f_geom, f_mesh, f_solution, f_X, f_dX,
                                               FX[lev], FX_offset,
                                               ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::eval_dG_dG_bou_fluxes(t,
                                               c_geom, c_mesh, c_solution, c_X,
                                               rr,
                                               f_geom, f_mesh, f_solution, f_X,
                                               FX[lev], FX_offset,
                                               ibvp);
        }
    }

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN
    for (int lev = 0; lev < n_levels; ++lev)
    {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            if (FX[lev][dir].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_dG_AMR_Utils.H - eval_dXdt_explicit_RKdG\n";
                msg += "| Fluxes in direction "+std::to_string(dir)+" contains nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
#endif

    // COPY FLUXES FROM FINE TO COARSE LEVEL
    for (int lev = (n_levels-1); lev > 0; --lev)
    {
        average_down_faces(GetArrOfConstPtrs(FX[lev]), GetArrOfPtrs(FX[lev-1]), amr.refRatio(lev-1), amr.Geom(lev-1));
    }

    // ACTUALLY ADD THE BOUNDARY INTEGRAL CONTRIBUTION
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        // FINITE VOLUME SCHEME
        if ((solution.params.space_p == 0) && solution.params.FV_is_active)
        {
            explicit_RK::add_FV_bou_contrib_to_dXdt(t, geom, mesh, solution, X[lev], dX[lev], FX[lev], FX_offset, dXdt[lev], ibvp);
        }
        // DISCONTINUOUS GALERKIN SCHEME
        else
        {
            explicit_RK::add_bou_contrib_to_dXdt(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, dXdt[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ----
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        Solution & solution = *amr.solutions[lev];
        
        solution.add_small_elements_contribution(geom, dXdt[lev], ibvp);
        solution.multiply_by_inverse_space_mass_matrix(geom, dXdt[lev], ibvp);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Advance in time using explicit Runge-Kutta algorithms.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[inout] X: a vector of MultiFabs that will contain the updated solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void take_time_step_using_explicit_RKdG(const amr::SinglePatch & amr, const Real t, const Real dt, Vector<MultiFab> & X_new, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // RUNGE-KUTTA ORDER
    const int RK_order = get_explicit_RKdG_order(amr);
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<MultiFab> X_old(n_levels);
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if (X_new[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+" on entry.\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
#endif

    // SWAP NEW AND OLD -----------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        X_old[lev].define(X_new[lev].boxarray, X_new[lev].distributionMap, X_new[lev].n_comp, X_new[lev].n_grow);
        std::swap(X_old[lev], X_new[lev]);
    }
    // ----------------------------------------------------------------

    // RK1 ------------------------------------------------------------
    if (RK_order == 1)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK1 STEP 1
        // dXdt := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt, ibvp);

        // X_new := X_old+dt*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0, dt, dXdt[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK2 ------------------------------------------------------------
    else if (RK_order == 2)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK2 STEP 1
        // dXdt := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt, ibvp);

        // X1 := X_old+dt*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK2 STEP 2
        // dXdt := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt, ibvp);

        // X_new := (1/2)*X_old+(1/2)*X1+(dt/2)*dXdt
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 0.5, X_old[lev], 0, 0.5, X1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev], 0.5*dt, dXdt[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK3 ------------------------------------------------------------
    else if (RK_order == 3)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_levels);
        Vector<MultiFab> dXdt2(n_levels);
        Vector<MultiFab> dXdt3(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }

        // RK3 STEP 1
        // dXdt1 := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt1, ibvp);

        // X1 := X_old+dt*dXdt1
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK3 STEP 2
        // dXdt2 := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt2, ibvp);

        // X1 := X_old+(dt/4)*dXdt1+(dt/4)*dXdt2
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.25*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            MultiFab::Saxpy(X1[lev],                       0.25*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK3 STEP 3
        // dXdt3 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt3, ibvp);

        // X_new := X_old+(dt/6)*dXdt1+(dt/6)*dXdt2+(2*dt/3)*dXdt3
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0,     dt/6.0, dXdt1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                           dt/6.0, dXdt2[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       2.0*dt/3.0, dXdt3[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    // RK4 ------------------------------------------------------------
    else if (RK_order == 4)
    {
        // NEEDED MULTIFABS
        Vector<MultiFab> dXdt1(n_levels);
        Vector<MultiFab> dXdt2(n_levels);
        Vector<MultiFab> dXdt3(n_levels);
        Vector<MultiFab> dXdt4(n_levels);
        Vector<MultiFab> X1(n_levels);

        for (int lev = 0; lev < n_levels; ++lev)
        {
            const MultiFab & X = X_old[lev];
            
            dXdt1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt2[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt3[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            dXdt4[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
            X1[lev].define(X.boxarray, X.distributionMap, X.n_comp, X.n_grow);
        }
        
        // RK4 STEP 1
        // dXdt1 := dXdt(t, X_old)
        eval_dXdt_explicit_RKdG(amr, t, X_old, dXdt1, ibvp);

        // X1 := X_old+(dt/2)*dXdt1
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.5*dt, dXdt1[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 2
        // dXdt2 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt2, ibvp);

        // X1 := X_old+(dt/2)*dXdt2
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, 0.5*dt, dXdt2[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 3
        // dXdt3 := dXdt(t+dt/2, X1)
        eval_dXdt_explicit_RKdG(amr, t+0.5*dt, X1, dXdt3, ibvp);

        // X1 := X_old+dt*dXdt3
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X1[lev], 1.0, X_old[lev], 0, dt, dXdt3[lev], 0, 0, X1[lev].n_comp, X1[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X1[lev], ibvp);
        }

        // RK4 STEP 4
        // dXdt4 := dXdt(t+dt, X1)
        eval_dXdt_explicit_RKdG(amr, t+dt, X1, dXdt4, ibvp);

        // X_new := X_old+(dt/6)*dXdt1+(dt/3)*dXdt2+(dt/3)*dXdt3+(dt/6)*dXdt4
        for (int lev = 0; lev < n_levels; ++lev)
        {
            MultiFab::LinComb(X_new[lev], 1.0, X_old[lev], 0, dt/6.0, dXdt1[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/3.0, dXdt2[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/3.0, dXdt3[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            MultiFab::Saxpy(X_new[lev],                       dt/6.0, dXdt4[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
            amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        }
    }
    // ----------------------------------------------------------------
    else
    {
        std::string msg;
        msg  = "\n";
        msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
        msg += "| Unexpected order: "+std::to_string(RK_order)+"\n";
        Abort(msg);
    }
    // ----------------------------------------------------------------

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN --------------------------------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        if (X_new[lev].contains_nan())
        {
            std::string msg;
            msg  = "\n";
            msg += "ERROR: AMReX_dG_AMR_Utils.H - take_time_step_using_explicit_RKdG\n";
            msg += "| Solution contains nans at level "+std::to_string(lev)+".\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
#endif

    // RESTRICT FINE SOLUTION ONTO THE COARSE LEVEL -------------------
    for (int lev = (n_levels-1); lev > 0; --lev)
    {
        amr_utils::restrict(amr.Geom(lev), *amr.solutions[lev], X_new[lev],
                            amr.refRatio(lev-1),
                            amr.Geom(lev-1), *amr.solutions[lev-1], X_new[lev-1], amr.masks[lev-1],
                            ibvp);
    }
    // ----------------------------------------------------------------
}
// ====================================================================


// TAKE TIME STEP (GENERAL) ===========================================
/**
 * \brief Advance in time.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[inout] X: a vector of MultiFabs that will contain the updated solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void take_time_step(const amr::SinglePatch & amr, const Real t, const Real dt, Vector<MultiFab> & X, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    std::string time_integration;
    // ----------------------------------------------------------------

    // TIME INTEGRATION -----------------------------------------------
    // At this moment, all AMR levels must use the same time
    // integration algorithm.
    // ----------------------------------------------------------------
    {
        const int lev = 0;
        const Solution & solution = *amr.solutions[lev];

        time_integration = solution.params.time_integration;
    }
    for (int lev = 1; lev <= finest_level; ++lev)
    {
        const Solution & solution = *amr.solutions[lev];

        if (time_integration.compare(solution.params.time_integration) != 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR_Utils.H - take_time_step\n";
            msg += "| At this moment all AMR levels must use the same time integration algorithm.\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------

    // SELECT TIME INTEGRATION ALGORITHM ------------------------------
    {
        const int lev = 0;
        const Solution & solution = *amr.solutions[lev];

        if (solution.time_integration_is_explicit_RKdG())
        {
            take_time_step_using_explicit_RKdG(amr, t, dt, X, ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR_Utils.H - take_time_step\n";
            msg += "| Unexpected time integration algorithm: "+solution.params.time_integration+".\n";
            Abort(msg);
        }
    }
    // ----------------------------------------------------------------
}
// ====================================================================
// ####################################################################
} // parabolic_pde



namespace hyperbolic_2nd_order_pde
{
// ADVANCE IN TIME ####################################################
// TIME INCREMENT =====================================================
/**
 * \brief Compute the time increment.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFabs containing the solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
Real eval_dt(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & X, const IBVP & ibvp)
{
    const int finest_level = amr.finestLevel();
    Real dt;

    dt = std::numeric_limits<Real>::max();
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        const iMultiFab & mask = amr.masks[lev];
        const Real dt_lev = solution.eval_dt(t, amr.Geom(lev), mesh, X[lev], mask, ibvp);

        dt = amrex::min(dt, dt_lev);
    }

    return dt;
}
// ====================================================================


// TAKE TIME STEP USING EXPLICIT CENTRAL DIFFERENCE ===================
/**
 * \brief Eval the time derivatives of the dG coefficients
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] X: a vector of MultiFab objects containing the dG coefficients.
 * \param[in] MCh: a vector of MultiFab objects containing the Cholesky decomposition of the pde
 *                 mass matrices.
 * \param[in] d2Xdt2: a vector of MultiFab objects that will contain the time derivatives of the dG coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void eval_d2Xdt2_explicit_central_difference(const amr::SinglePatch & amr, const Real t, const Vector<MultiFab> & MCh, const Vector<MultiFab> & X, Vector<MultiFab> & d2Xdt2, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // UNKNOWN FIELDS
    const int n_unknown_fields = ibvp.get_number_of_unknown_fields();

    // DG ORDER / FX OFFSET
    const int largest_sp = amr.get_largest_dG_space_p();
    const int largest_sNp = AMREX_D_TERM((1+largest_sp),*(1+largest_sp),*(1+largest_sp));
    const int FX_offset = largest_sNp*n_unknown_fields;
    const int FX_n_comp = 2*FX_offset;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<Array<MultiFab, AMREX_SPACEDIM>> FX(n_levels);
    // ----------------------------------------------------------------

    // INITIALIZE FLUX MULTIFABS --------------------------------------
    // FLUXES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const BoxArray ba = X[lev].boxarray;
#if (AMREX_SPACEDIM == 1)
        const BoxArray fc_ba[1] = {convert(ba, IntVect(1))};
#endif
#if (AMREX_SPACEDIM == 2)
        const BoxArray fc_ba[2] = {convert(ba, IntVect(1,0)),
                                   convert(ba, IntVect(0,1))};
#endif
#if (AMREX_SPACEDIM == 3)
        const BoxArray fc_ba[3] = {convert(ba, IntVect(1,0,0)),
                                   convert(ba, IntVect(0,1,0)),
                                   convert(ba, IntVect(0,0,1))};
#endif
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            FX[lev][dir].define(fc_ba[dir], X[lev].distributionMap, FX_n_comp, 0);
            FX[lev][dir] = 0.0;
        }
    }

    // TIME DERIVATIVES
    for (int lev = 0; lev < n_levels; ++lev)
    {
        d2Xdt2[lev] = 0.0;
    }
    // ----------------------------------------------------------------

    // DOMAIN INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES -----------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        explicit_central_difference::add_dom_contrib_to_dXdt(t, geom, mesh, solution, X[lev], d2Xdt2[lev], ibvp);
    }
    // ----------------------------------------------------------------

    // BOUNDARY INTEGRAL CONTRIBUTION TO THE TIME DERIVATIVES ---------
    // COMPUTE BOUNDARY FLUXES: LEVEL 0
    {
        const int lev = 0;
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];
        
        explicit_central_difference::eval_bou_fluxes(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, ibvp);
    }

    // COMPUTE BOUNDARY FLUXES: LEVEL > 0
    for (int lev = 1; lev < n_levels; ++lev)
    {
        //const Geometry & c_geom = amr.Geom(lev-1);
        //const Mesh & c_mesh = *amr.meshes[lev-1];
        //const Solution & c_solution = *amr.solutions[lev-1];
        //const MultiFab & c_X = X[lev-1];
        //const IntVect rr = amr.refRatio(lev-1);
        //const Geometry & f_geom = amr.Geom(lev);
        //const Mesh & f_mesh = *amr.meshes[lev];
        //const Solution & f_solution = *amr.solutions[lev];
        //const MultiFab & f_X = X[lev];
        
Print() << "COMPUTE BOUNDARY FLUXES: LEVEL > 0" << std::endl;
exit(-1);
    }

#ifdef AMREX_DG_CHECK_FOR_NAN
    // CHECK FOR NAN
    for (int lev = 0; lev < n_levels; ++lev)
    {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            if (FX[lev][dir].contains_nan())
            {
                std::string msg;
                msg  = "\n";
                msg += "ERROR: AMReX_dG_AMR_Utils.H - eval_d2Xdt2_explicit_central_difference\n";
                msg += "| Fluxes in direction "+std::to_string(dir)+" contains nans at level "+std::to_string(lev)+".\n";
                Abort(msg);
            }
        }
    }
#endif

    // COPY FLUXES FROM FINE TO COARSE LEVEL
    for (int lev = (n_levels-1); lev > 0; --lev)
    {
        average_down_faces(GetArrOfConstPtrs(FX[lev]), GetArrOfPtrs(FX[lev-1]), amr.refRatio(lev-1), amr.Geom(lev-1));
    }

    // ACTUALLY ADD THE BOUNDARY INTEGRAL CONTRIBUTION
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        explicit_central_difference::add_bou_contrib_to_dXdt(t, geom, mesh, solution, X[lev], FX[lev], FX_offset, d2Xdt2[lev], ibvp);
    }
    // ----------------------------------------------------------------

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ----
    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        Solution & solution = *amr.solutions[lev];
        
        solution.add_small_elements_contribution(geom, d2Xdt2[lev], ibvp);
        solution.multiply_by_inverse_pde_mass_matrix(geom, MCh[lev], d2Xdt2[lev], ibvp);
    }
    // ----------------------------------------------------------------
}

/**
 * \brief Advance in time using explicit central-difference algorithm.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[in] MCh: a vector of MultiFab objects containing the Cholesky decomposition of the pde
 *                 mass matrices.
 * \param[inout] X_prev: a vector of MultiFabs that will contain X_new.
 * \param[inout] X_new: a vector of MultiFabs that will contain the updated solution coefficients.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename IBVP>
void take_time_step_using_explicit_central_difference(const amr::SinglePatch & amr, const Real t, const Real dt,
                                                      const Vector<MultiFab> & MCh, Vector<MultiFab> & X_prev, Vector<MultiFab> & X_new,
                                                      const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<MultiFab> X_old(n_levels), d2Xdt2(n_levels);
    // ----------------------------------------------------------------

    // INIT MULTIFABS AND SWAP NEW AND OLD ----------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        d2Xdt2[lev].define(X_new[lev].boxarray, X_new[lev].distributionMap, X_new[lev].n_comp, X_new[lev].n_grow);

        X_old[lev].define(X_new[lev].boxarray, X_new[lev].distributionMap, X_new[lev].n_comp, X_new[lev].n_grow);
        std::swap(X_old[lev], X_new[lev]);
    }
    // ----------------------------------------------------------------

    // APPLY CENTRAL DIFFERENCE IN TIME -------------------------------
    eval_d2Xdt2_explicit_central_difference(amr, t, MCh, X_old, d2Xdt2, ibvp);

    for (int lev = 0; lev < n_levels; ++lev)
    {
        // X_new := 2*X_old-X_prev+dt*dt*d2Xdt2
        MultiFab::LinComb(X_new[lev], 2.0, X_old[lev], 0, -1.0, X_prev[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
        MultiFab::Saxpy(X_new[lev], dt*dt, d2Xdt2[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
        amr.solutions[lev]->copy_solution_to_small_elements(amr.Geom(lev), X_new[lev], ibvp);
        
        // X_prev := X_old
        std::swap(X_prev[lev], X_old[lev]);
    }
    // ----------------------------------------------------------------
}
// ====================================================================


// TAKE TIME STEP USING IMPLICIT NEWMARK ==============================
/**
 * \brief Eval the residuals (and Jacobian) for the implicit Newmark algorithm.
 *
 * \param[in] gamma, beta: parameters of the Newmark algorithm.
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[in] MCh: a vector of MultiFab objects containing the Cholesky decomposition of the pde
 *                 mass matrices.
 * \param[in] X_old a vector of MultiFabs containing the old dG coefficients.
 * \param[in] Xd_old a vector of MultiFabs containing the old dG coefficients time derivatives.
 * \param[in] X a vector of MultiFabs containing the current dG coefficients.
 * \param[in] Xd a vector of MultiFabs containing the current dG coefficients time derivatives.
 * \param[in] ls: object containing the data structures used by the linear solver.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename LS, typename IBVP>
void eval_residuals_implicit_Newmark(const Real gamma, const Real beta,
                                     const amr::SinglePatch & amr, const Real t, const Real dt,
                                     const Vector<MultiFab> & MCh, const Vector<MultiFab> & X_old, const Vector<MultiFab> & Xd_old, const Vector<MultiFab> & X, const Vector<MultiFab> & Xd,
                                     LS & ls, const IBVP & ibvp,
                                     const bool update_Jacobian = false)
{
    // INITIALIZATION -------------------------------------------------
    std::fill(ls.B.begin(), ls.B.end(), 0.0);
    if (update_Jacobian)
    {
        std::fill(ls.A.begin(), ls.A.end(), 0.0);
    }
    // ----------------------------------------------------------------

    if (amr.linear_direct_solver_finest_level == 0)
    {
        const int lev = 0;
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        if (ls.matrix_is_real_symmetric())
        {
            implicit_Newmark::eval_residuals_sym(gamma, beta, t, dt,
                                                 geom, mesh, solution, amr.masks[lev],
                                                 MCh[lev], X_old[lev], Xd_old[lev], X[lev], Xd[lev],
                                                 ls.B, ls.ia, ls.A,
                                                 ls.dom_info[lev], ls.bou_info[lev],
                                                 ibvp,
                                                 update_Jacobian);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: AMReX_dG_AMR_Utils.H - hyperbolic_2nd_order_pde::eval_residuals_implicit_Newmark\n";
            msg += "| At the moment, we support symmetric matrix type only.\n";
            Abort(msg);
        }
    }
    else
    {
        std::string msg;
        msg  = "\n";
        msg +=  "ERROR: AMReX_dG_AMR_Utils.H - hyperbolic_2nd_order_pde::eval_residuals_implicit_Newmark\n";
        msg += "| Evaluating the Jacobian for multilevel AMR is not supported at the moment.\n";
        msg += "| linear_direct_solver_finest_level: "+std::to_string(amr.linear_direct_solver_finest_level)+".\n";
        Abort(msg);
    }
}

/**
 * \brief Advance in time using the implicit Newmark algorithm.
 *
 * \param[in] amr: SinglePath AMR object.
 * \param[in] t: time.
 * \param[in] dt: time increment.
 * \param[in] MCh: a vector of MultiFab objects containing the Cholesky decomposition of the pde
 *                 mass matrices.
 * \param[inout] X_new: a vector of MultiFabs that will contain the updated dG coefficients.
 * \param[inout] Xd_new: a vector of MultiFabs that will contain the updated dG coefficients'
 *                       time derivatives.
 * \param[in] ibvp: initial boundary value problem object.
 *
 * \return the time increment.
*/
template <typename LS, typename IBVP>
void take_time_step_using_implicit_Newmark(const amr::SinglePatch & amr, const Real t, const Real dt,
                                           const Vector<MultiFab> & MCh, Vector<MultiFab> & X_new, Vector<MultiFab> & Xd_new,
                                           LS & ls, const IBVP & ibvp)
{
    // PARAMETERS -----------------------------------------------------
    const int finest_level = amr.finestLevel();
    const int n_levels = finest_level+1;

    // NEWMARK ALGORITHM PARAMETERS
    // Second-order accurate scheme but no numerical dissipation
    //const Real gamma = 0.5;
    //const Real beta = 0.25;
    // First-order accurate scheme with numerical dissipation
    const Real gamma = 5.0/6.0;
    const Real beta = 0.25*(gamma+0.5)*(gamma+0.5);

    // NEWTON-RAPHSON
    const Real tol = 1.0e-12;
    const int it_max = 100;
    // ----------------------------------------------------------------

    // VARIABLES ------------------------------------------------------
    Vector<MultiFab> X_old(n_levels), Xd_old(n_levels);
    Real err;
    int it;
    Vector<Real> X(ls.n);
    // ----------------------------------------------------------------

    // INIT MULTIFABS AND SWAP NEW AND OLD ----------------------------
    for (int lev = 0; lev < n_levels; ++lev)
    {
        X_old[lev].define(X_new[lev].boxarray, X_new[lev].distributionMap, X_new[lev].n_comp, X_new[lev].n_grow);
        Xd_old[lev].define(Xd_new[lev].boxarray, Xd_new[lev].distributionMap, Xd_new[lev].n_comp, Xd_new[lev].n_grow);

        MultiFab::Copy(X_old[lev], X_new[lev], 0, 0, X_new[lev].n_comp, X_new[lev].n_grow);
        MultiFab::Copy(Xd_old[lev], Xd_new[lev], 0, 0, Xd_new[lev].n_comp, Xd_new[lev].n_grow);
    }
    // ----------------------------------------------------------------

    // LINEAR SOLVER FIRST CALL ---------------------------------------
    if (ls.first_call)
    {
        eval_residuals_implicit_Newmark(gamma, beta, amr, t, dt, MCh, X_old, Xd_old, X_new, Xd_new, ls, ibvp, true);
        ls.factorize();

        /* DEBUG
        ls.print_to_file_sparse("system");
        exit(-1);
        */
    }
    else
    {
        eval_residuals_implicit_Newmark(gamma, beta, amr, t, dt, MCh, X_old, Xd_old, X_new, Xd_new, ls, ibvp, false);
    }
    // ----------------------------------------------------------------

    // LINEAR SOLVE ---------------------------------------------------
    {
        // Compute: dX := J^{-1}*R
        ls.solve();

        // X := X-dX
        amr.subtract_solution_of_linear_solver(ls, X_new, ibvp);
    }
    // ----------------------------------------------------------------

    // APPLY NEWTON-RAPHSON -------------------------------------------
    /*
    err = 1.0;
    it = 0;
    while ((it < it_max) && (err > tol))
    {
        // CALL SOLVER
        // Compute: dX := J^{-1}*R
        ls.solve();

        // X := X-dX
        amr.subtract_solution_of_linear_solver(ls, X_new, ibvp);

        eval_residuals_implicit_Newmark(gamma, beta, amr, t, dt, MCh, X_old, Xd_old, X_new, Xd_new, ls, ibvp, false);
        
        err = linalg::norm2(ls.B.size(), ls.B.data());

        //Print() << "| Residuals (it = " << it << "): " << err << std::endl;

        it += 1;
    }
    */
    // ----------------------------------------------------------------

    // SEND A WARNING IF MAXIMUM NUMBER OF ITERATIONS IS REACHED ------
    if (it == it_max)
    {
        std::string msg;
        msg  = "\n";
        msg +=  "WARNING: AMReX_dG_AMR_Utils.H - hyperbolic_2nd_order_pde::eval_residuals_implicit_Newmark\n";
        msg += "| Reached maximum number of iterations.\n";
        msg += "| it_max: "+std::to_string(it_max)+".\n";
        msg += "| residuals norm: "+std::to_string(err)+".\n";
        Abort(msg);
    }
    // ----------------------------------------------------------------

    // UPDATE VELOCITY ------------------------------------------------
    std::fill(ls.B.begin(), ls.B.end(), 0.0);

    for (int lev = 0; lev < n_levels; ++lev)
    {
        const Geometry & geom = amr.Geom(lev);
        const Mesh & mesh = *amr.meshes[lev];
        const Solution & solution = *amr.solutions[lev];

        implicit_Newmark::eval_Xd_sym(gamma, t, dt,
                                      geom, mesh, solution, amr.masks[lev],
                                      MCh[lev], X_old[lev], Xd_old[lev], X_new[lev], Xd_new[lev],
                                      ls.B,
                                      ls.dom_info[lev], ls.bou_info[lev],
                                      ibvp);
    }
    // ----------------------------------------------------------------
}
// ====================================================================
// #################################################################### 
} // hyperbolic_2nd_order_pde

} // namespace dG
} // namespace amrex

#endif