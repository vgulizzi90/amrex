//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file AMReX_dG_IBVP_Base.H
 * \brief Base IBVP class used to derive user-defined IBVP classes.
*/

#ifndef AMREX_DG_IBVP_BASE_H_
#define AMREX_DG_IBVP_BASE_H_

#include <AMReX_REAL.H>

namespace amrex
{
namespace dG
{

struct IBVP_Base
{
    // CONSTRUCTOR ====================================================
    IBVP_Base()
    {}
    // ================================================================


    // SOLUTION =======================================================
    /**
     * \brief Return the number of considered domains.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_domains() const
    {
        Abort("IBVP_Base - get_number_of_domains: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        Abort("IBVP_Base - get_number_of_unknown_fields: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int /*dom*/, int & /*u_lo*/, int & /*u_hi*/) const
    {
        Abort("IBVP_Base - domain_unknown_fields_index_bounds: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int unknown_field_parent_domain(const int /*u*/) const
    {
        Abort("IBVP_Base - unknown_field_parent_domain: We should not end up in here. The user must implement this function.");
        return -1;
    }
    
    /**
     * \brief Return the domain that is neighbor with domain dom.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_nbr_dom(const int /*dom*/) const
    {
        Abort("IBVP_Base - get_nbr_dom: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the solution at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_exact_solution(const Real /*t*/, const Real * /*x*/, Real * /*U*/) const
    {
        Abort("IBVP_Base - eval_exact_solution: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the initial conditions of
     *                the solution at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs(const Real * /*x*/, Real * /*U*/) const
    {
        Abort("IBVP_Base - eval_ICs: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Fill U_t with initial conditions of the time-derivative of U at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U_t: pointer to memory that will be filled with the value of the initial conditions of
     *                  the time-derivatives of the solution at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs_der(const Real * /*x*/, Real * /*U_t*/) const
    {
        Abort("IBVP_Base - eval_ICs_der: We should not end up in here. The user must implement this function.");
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error at (t, x) to be used for the L_inf norm error.
     *
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the value of the solution at (t, x).
     * \param[out] err: error at (t, x).
     * \param[out] err_norm: error normalization at (t, x).
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_error_L_inf(const int /*dom*/,
                          const Real /*t*/, const Real * /*x*/,
                          const Real * /*U*/,
                          Real & /*err*/, Real & /*err_norm*/) const
    {
        Abort("IBVP_Base - eval_error_L_inf: We should not end up in here. The user must implement this function.");
    }
    // ================================================================


    // TIME INCREMENT =================================================
    /**
     * \brief Eval the time increment based on solution at (t, x).
     *
     * \param[in] dom: domain index.
     * \param[in] dx: pointer to memory containing the grid sizes.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the value of the solution at (t, x).
     *
     * \return: time increment based on solution at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real eval_dt(const int /*dom*/,
                        const Real * /*dx*/,
                        const Real /*t*/, const Real * /*x*/,
                        const Real * /*U*/) const
    {
        Abort("IBVP_Base - eval_dt: We should not end up in here. The user must implement this function.");
        return 0.0;
    }
    // ================================================================


    // GOVERNING EQUATIONS ============================================
    /**
     * \brief Eval the mass matrix for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] M: pointer to memory that will contain the mass matrix.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_M(const int /*dom*/,
                const Real /*t*/, const Real * /*x*/,
                Real * /*M*/) const
    {
        Abort("IBVP_Base - eval_M: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the fluxes for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_F(const int /*dom*/,
                const Real /*t*/, const Real * /*x*/,
                const Real * /*U*/, AMREX_D_DECL(Real * /*F1*/, Real * /*F2*/, Real * /*F3*/)) const
    {
        Abort("IBVP_Base - eval_F: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the fluxes/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] F1, F2, F3: pointer to memory that will contain the fluxes.
     * \param[out] R: pointer to memory that will contain the reactive term.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_FR(const int /*dom*/,
                const Real /*t*/, const Real * /*x*/,
                const Real * /*U*/, AMREX_D_DECL(Real * /*F1*/, Real * /*F2*/, Real * /*F3*/), Real * /*R*/) const
    {
        Abort("IBVP_Base - eval_FR: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the diffusive/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the solution.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the space-derivatives of the solution.
     * \param[out] D1, D2, D3: pointer to memory that will contain the diffusive terms.
     * \param[out] R: pointer to memory that will contain the reactive term.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_DR(const int /*dom*/,
                 const Real /*t*/, const Real * /*x*/,
                 const Real * /*U*/, AMREX_D_DECL(const Real * /*U_1*/, const Real * /*U_2*/, const Real * /*U_3*/),
                 AMREX_D_DECL(Real * /*D1*/, Real * /*D2*/, Real * /*D3*/), Real * /*R*/) const
    {
        Abort("IBVP_Base - eval_DR: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the gradients of diffusive/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the solution.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the space-derivatives of the solution.
     * \param[out] dDidU_j: pointer to memory that will contain the derivative of the diffusive term
     *                      Di with respect to U_j.
     * \param[out] dDidU: pointer to memory that will contain the derivative of the diffusive term Di
     *                    with respect to U.
     * \param[out] dRdU_j: pointer to memory that will contain the derivative of the reactive term R
     *                     with respect to U_j.
     * \param[out] dRdU: pointer to memory that will contain the derivative of the reactive term R
     *                   with respect to U.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_dDRdU(const int /*dom*/,
                    const Real /*t*/, const Real * /*x*/,
                    const Real * /*U*/, AMREX_D_DECL(const Real * /*U_1*/, const Real * /*U_2*/, const Real * /*U_3*/),
                    Real * /*dD1dU_1*/, Real * /*dD1dU_2*/, Real * /*dD2dU_1*/, Real * /*dD2dU_2*/, Real * /*dD1dU*/, Real * /*dD2dU*/,
                    Real * /*dRdU_1*/, Real * /*dRdU_2*/, Real * /*dRdU*/) const
    {
        Abort("IBVP_Base - eval_dDRdU: We should not end up in here. The user must implement this function.");
    }
    // ================================================================


    // DG =============================================================
    /**
     * \brief Eval the numerical flux at the intraphase boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] m_U: pointer to memory containing the solution on the negative side of the unit
     *                 normal.
     * \param[in] p_U: pointer to memory containing the solution on the positive side of the unit
     *                 normal.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_intraphase(const int /*dom*/,
                             const Real /*t*/, const Real * /*x*/, const Real * /*un*/,
                             const Real * /*m_U*/, const Real * /*p_U*/, Real * /*NFn*/) const
    {
        Abort("IBVP_Base - eval_NFn_intraphase: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_BCs(const int /*dom*/,
                      const Real /*t*/, const Real * /*x*/, const Real * /*un*/,
                      const Real * /*U*/, Real * /*NFn*/) const
    {
        Abort("IBVP_Base - eval_NFn_BCs: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the solution.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_EB(const int /*dom*/,
                     const Real /*t*/, const Real * /*x*/, const Real * /*un*/,
                     const Real * /*U*/, Real * /*NFn*/) const
    {
        Abort("IBVP_Base - eval_NFn_EB: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] nbr_dom: index of the domain that is neighbor with dom.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] m_U: pointer to memory containing the solution on the negative side of the unit
     *                 normal.
     * \param[in] p_U: pointer to memory containing the solution on the positive side of the unit
     *                 normal.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NFn_EB(const int /*dom*/, const int /*nbr_dom*/,
                     const Real /*t*/, const Real * /*x*/, const Real * /*un*/,
                     const Real * /*m_U*/, const Real * /*p_U*/, Real * /*NFn*/) const
    {
        Abort("IBVP_Base - eval_NFn_EB: We should not end up in here. The user must implement this function.");
    }
    // ================================================================


    // FV =============================================================
    /**
     * \brief Reconstruct and limit the slopes for FV schemes.
     *
     * \param[in] geom: amrex geometry object.
     * \param[in] solution: Solution object.
     * \param[in] mesh: Mesh object.
     * \param[in] X: a MultiFabs containing the solution states.
     * \param[out] dX: a MultiFabs that will contain the solution slopes.
     *
    */
    void eval_FV_slopes(const amrex::Geometry & /*geom*/,
                        const amrex::dG::Mesh & /*mesh*/,
                        const amrex::dG::Solution & /*solution*/,
                        const amrex::MultiFab & /*X*/,
                        amrex::MultiFab & /*dX*/) const
    {
        Abort("IBVP_Base - eval_FV_slopes: We should not end up in here. The user must implement this function.");
    }
    // ================================================================


    // OUTPUT FIELDS ==================================================
    /**
     * \brief Return the number of output fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_output_fields() const
    {
        Abort("IBVP_Base - get_number_of_output_fields: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int /*u*/) const
    {
        Abort("IBVP_Base - output_field_parent_domain: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & /*output_fields_names*/) const
    {
        Abort("IBVP_Base - get_output_fields_names: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Return the number of output fields for the shell problem.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_shell_output_fields() const
    {
        Abort("IBVP_Base - get_number_of_shell_output_fields: We should not end up in here. The user must implement this function.");
        return -1;
    }

    /**
     * \brief Set the names of the fields to be output for the shell problem.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_shell_output_fields_names(amrex::Vector<std::string> & /*output_fields_names*/) const
    {
        Abort("IBVP_Base - get_shell_output_fields_names: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the solution at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const Real /*t*/, const Real * /*x*/,
                            const Real * /*U*/,
                            Real * /*F*/) const
    {
        Abort("IBVP_Base - eval_output_fields: We should not end up in here. The user must implement this function.");
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the solution at (t,x).
     * \param[in] U_1, U_2, U_3: pointer to memory containing the space-derivatives of the solution at
     *                           (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const Real /*t*/, const Real * /*x*/,
                            const Real * /*U*/, AMREX_D_DECL(const Real * /*U_1*/, const Real * /*U_2*/, const Real * /*U_3*/),
                            Real * /*F*/) const
    {
        Abort("IBVP_Base - eval_output_fields: We should not end up in here. The user must implement this function.");
    }
    // ================================================================

};

} // namespace dG
} // namespace amrex

#endif