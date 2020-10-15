//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_SodsTube.H
 * \brief Contains constants and data structures for the Sod's tube problem with ideal gases.
*/

#include "../IBVP_Base.H"

/**
 * \brief This struct contains constants and methods for the considered IBVP. 
*/
struct IDEAL_GAS
:
public IDEAL_GAS_BASE
{
    // DATA MEMBERS ###################################################
    amrex::Real params[9];
    // ################################################################
    // NOTE: params must store:
    //       - params[0]: rhoL, i.e. density at the left of the shock;
    //       - params[1]: uL, i.e. velocity at the left of the shock;
    //       - params[2]: pL, i.e. pressure at the left of the shock;
    //       - params[3]: shock location;
    //       - params[4]: rhoR, i.e. density at the right of the shock;
    //       - params[5]: uR, i.e. velocity at the right of the shock;
    //       - params[6]: pR, i.e. pressure at the right of the shock;
    //       - params[7]: diameter of the tube;
    //       - params[8]: inclination of the tube;
    // ################################################################

    // CONSTRUCTOR ####################################################
    /**
     * \brief This method initializes the IBVP.
     * 
     * \param[in] gamma_: ratio of specific heats.
    */
    IDEAL_GAS(const amrex::Real gamma_, const amrex::Vector<amrex::Real> & params_)
    :
    IDEAL_GAS_BASE(gamma_)
    {
        std::copy(params_.begin(), params_.end(), this->params);
    }
    // ################################################################

    // AUXILIARY FUNCTIONS ############################################
    /**
     * \brief Euclidean norm of a vector in 3D space.
     * 
     * \param[in] v: pointer to memory containing the vector's components.
     *
     * \return the Euclidean norm of the vector.
    */
    AMREX_GPU_HOST_DEVICE
    amrex::Real norm2_3D(const amrex::Real * v) const
    {
        amrex::Real res;
        res =  v[0]*v[0];
        res += v[1]*v[1];
        res += v[2]*v[2];

        return std::sqrt(res);
    }

    /**
     * \brief Dot product of two vectors in 3D space.
     * 
     * \param[in] v1: pointer to memory containing the first vector's components.
     * \param[in] v2: pointer to memory containing the second vector's components.
     *
     * \return the dot product between the two vectors.
    */
    AMREX_GPU_HOST_DEVICE
    amrex::Real dot_3D(const amrex::Real * v1, const amrex::Real * v2) const
    {
        amrex::Real res;
        res =  v1[0]*v2[0];
        res += v1[1]*v2[1];
        res += v1[2]*v2[2];

        return res;
    }

    /**
     * \brief Cross product between two vector in 3D space.
     * 
     * \param[in] v1: pointer to memory containing the first vector's components.
     * \param[in] v2: pointer to memory containing the second vector's components.
     * \param[out] res: pointer to memory that will be filled with the result.
    */
    AMREX_GPU_HOST_DEVICE
    void cross(const amrex::Real * v1, const amrex::Real * v2, amrex::Real * res) const
    {
        res[0] = v1[1]*v2[2]-v1[2]*v2[1];
        res[1] = v1[2]*v2[0]-v1[0]*v2[2];
        res[2] = v1[0]*v2[1]-v1[1]*v2[0];
    }
    // ################################################################

    // LEVEL SET FUNCTIONS ############################################
    /**
     * \brief Fill PHI with values of the level sets at (t,x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] PHI: pointer to memory that will be filled with the value of the level sets at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_PHI(const amrex::Real /*t*/, const amrex::Real * x, amrex::Real * PHI) const
    {
        const amrex::Real diam = this->params[7];
        const amrex::Real theta = (this->params[8])*M_PI/180.0;
        const amrex::Real m = std::tan(theta);
        const amrex::Real q = 0.5*(1.0-m);
        const amrex::Real A[3] = {0.0, q, 0.5};
        const amrex::Real B[3] = {1.0, m+q, 0.5};
        const amrex::Real X[3] = {x[0], x[1], (AMREX_SPACEDIM == 3) ? x[2] : 0.5};
        const amrex::Real AB[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
        const amrex::Real AX[3] = {X[0]-A[0], X[1]-A[1], X[2]-A[2]};
        const amrex::Real BX[3] = {X[0]-B[0], X[1]-B[1], X[2]-B[2]};
        
        amrex::Real tmp[3], res;
        this->cross(BX, AX, tmp);
        res = this->norm2_3D(tmp)/this->norm2_3D(AB)-0.5*diam;
        
        PHI[0] = res;
    }

    /**
     * \brief Fill phi_info with information about which domain is associated to which levelset.
     * 
     * \param[in] dom: domain index.
     * \param[out] phi_info: pointer to memory that will be filled domain info. More specifically:
     *
     *             phi_info[0] = ph means that the dom-th domain is defined by the ph-th levelset;
     *             phi_info[1] = -1 means that the dom-th domain is defined where the ph-th levelset is negative;
     *             phi_info[1] = +1 means that the dom-th domain is defined where the ph-th levelset is positive;
    */
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int /*dom*/, int * phi_info) const
    {
        phi_info[0] = 0;
        phi_info[1] = -1;
    }
    // ################################################################
    
    // INITIAL CONDITIONS #############################################
    /**
     * \brief Fill SOL with initial conditions of the unknown solution fields.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] SOL: pointer to memory that will be filled with the value of the unknown solution
     *                  fields at x.
    */
    AMREX_GPU_HOST_DEVICE
    void F_SOL0(const amrex::Real * x, amrex::Real * SOL) const
    {
        amrex::Print() << "HERE WE ARE - F_SOL0" << std::endl;
        exit(-1);
    }
    // ################################################################

    // BOUNDARY CONDITIONS ############################################
    /**
     * \brief Fill bcs_SOL memory to enforce boundary conditions.
     *
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the boundary outward unit normal.
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] bcs_SOL: pointer to memory that will be filled with values suitably chosen to
     *                      enforce the bcs at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_BCS(const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
               const amrex::Real * SOL,
               amrex::Real * bcs_SOL) const
    {
        amrex::Print() << "HERE WE ARE - F_BCS" << std::endl;
        exit(-1);
    }
    // ################################################################

    // OUTPUT FIELDS ##################################################
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void F_FIELDS_NAMES(amrex::Vector<std::string> & fields_names) const
    {
        amrex::Print() << "HERE WE ARE - F_FIELDS_NAMES" << std::endl;
        exit(-1);
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] PHI: pointer to memory that stores the level sets at (t,x).
     * \param[in] SOL: pointer to memory that stores the unknown solution fields at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE
    void F_FIELDS(const amrex::Real /*t*/, const amrex::Real * /*x*/,
                  const amrex::Real * SOL,
                  amrex::Real * F) const
    {
        amrex::Print() << "HERE WE ARE - F_FIELDS" << std::endl;
        exit(-1);
    }
    // ################################################################
};