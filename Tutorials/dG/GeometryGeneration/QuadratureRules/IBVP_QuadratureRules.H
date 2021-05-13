//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_QuadratureRules.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

namespace QuadratureRules
{
// QUADRATURE RULES: LEVEL SETS #######################################
struct CustomLevelSet
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    CustomLevelSet()
    :
    t{0.0},
    dom{0}
    {}
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_time(const amrex::Real t_)
    {
        this->t = t_;
    }
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // EVALUATION =====================================================
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval(const amrex::Real /*t*/, const amrex::Real * x) const
    {
        amrex::Real res;

        res = x[1]-std::cos(2.0*M_PI*(x[0]-1.0/32.0))/8.0-1.0/2.0;

        return res;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval(const amrex::Real * x) const
    {
        return this->eval(0.0, x);
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, AMREX_SPACEDIM> & x) const
    {
        const int sgn = (1-2*this->dom);
        T res;

        res = x[2]-0.6+0.4*x[1]*x[1]*x[1]-0.2*x[0]*x[0];
        res *= sgn;

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, AMREX_SPACEDIM> grad(const amrex::GpuArray<T, AMREX_SPACEDIM> & x) const
    {
        const int sgn = (1-2*this->dom);
        amrex::GpuArray<T, AMREX_SPACEDIM> res;

        AMREX_D_TERM
        (
            res[0] = -0.2*2.0*x[0];,
            res[1] = 0.4*3.0*x[1]*x[1];,
            res[2] = 1.0;
        )

        AMREX_D_TERM
        (
            res[0] *= sgn;,
            res[1] *= sgn;,
            res[2] *= sgn;
        )

        return res;
    }
    // ================================================================
};

struct Circle
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    amrex::Real c[AMREX_SPACEDIM], r;
    // ================================================================


    // CONSTRUCTOR ====================================================
    Circle()
    {
        // VARIABLES
        amrex::ParmParse pp;

        // READ INPUT FILE
        pp.get("x_coord", this->c[0]);
        pp.get("y_coord", this->c[1]);
        pp.get("radius", this->r);

    }
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_time(const amrex::Real t_)
    {
        this->t = t_;
    }
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // EVALUATION =====================================================
    amrex::Real eval(const amrex::Real * x) const
    {
        const int sgn = -(1-2*this->dom);
        const amrex::Real r2 = this->r*this->r;
        const amrex::Real dx[2] = {x[0]-this->c[0], x[1]-this->c[1]};
        const amrex::Real res = (dx[0]*dx[0]+dx[1]*dx[1]-r2)*sgn;

        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, 2> & x) const
    {
        const int sgn = -(1-2*this->dom);
        const amrex::Real r2 = this->r*this->r;
        const T dx[2] = {x[0]-this->c[0], x[1]-this->c[1]};
        const T res = (dx[0]*dx[0]+dx[1]*dx[1]-r2)*sgn;

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, 2> grad(const amrex::GpuArray<T, 2> & x) const
    {
        const int sgn = -(1-2*this->dom);
        const T dx[2] = {x[0]-this->c[0], x[1]-this->c[1]};
        const amrex::GpuArray<T, 2> res = {2.0*dx[0]*sgn, 2.0*dx[1]*sgn};

        return res;
    }
    // ================================================================
};

struct Sphere
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    amrex::Real c[AMREX_SPACEDIM], r;
    // ================================================================


    // CONSTRUCTOR ====================================================
    Sphere()
    {
        // VARIABLES
        amrex::ParmParse pp;

        // READ INPUT FILE
        pp.get("x_coord", this->c[0]);
        pp.get("y_coord", this->c[1]);
        pp.get("z_coord", this->c[2]);
        pp.get("radius", this->r);

    }
    // ================================================================


    // AUXILIARY ROUTINES =============================================
    void set_time(const amrex::Real t_)
    {
        this->t = t_;
    }
    void set_domain(const int dom_)
    {
        this->dom = dom_;
    }
    // ================================================================


    // EVALUATION =====================================================
    amrex::Real eval(const amrex::Real * x) const
    {
        const int sgn = -(1-2*this->dom);
        const amrex::Real r2 = this->r*this->r;
        const amrex::Real dx[3] = {x[0]-this->c[0], x[1]-this->c[1], x[2]-this->c[2]};
        const amrex::Real res = (dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]-r2)*sgn;

        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, 3> & x) const
    {
        const int sgn = -(1-2*this->dom);
        const amrex::Real r2 = this->r*this->r;
        const T dx[3] = {x[0]-this->c[0], x[1]-this->c[1], x[2]-this->c[2]};
        const T res = (dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]-r2)*sgn;

        return res;
    }

    template <typename T>
    amrex::GpuArray<T, 3> grad(const amrex::GpuArray<T, 3> & x) const
    {
        const int sgn = -(1-2*this->dom);
        const T dx[3] = {x[0]-this->c[0], x[1]-this->c[1], x[2]-this->c[2]};
        const amrex::GpuArray<T, 3> res = {2.0*dx[0]*sgn, 2.0*dx[1]*sgn, 2.0*dx[2]*sgn};

        return res;
    }
    // ================================================================
};
// ####################################################################



// QUADRATURE RULES: IBVP #############################################
struct IBVP
{
    // DATA MEMBERS ===================================================
    bool use_two_phases;
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        // VARIABLES
        amrex::ParmParse pp;
        int i;

        // READ INPUT FILE
        i = 0;
        pp.query("use_two_phases", i);
        this->use_two_phases = (i > 0);

    }
    // ================================================================


    // SOLUTION =======================================================
    /**
     * \brief Return the number of considered domains.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_domains() const
    {
        return (this->use_two_phases ? 2 : 1);
    }
    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        return (this->use_two_phases ? 2 : 1);
    }
    /**
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int unknown_field_parent_domain(const int u) const
    {
        if (this->use_two_phases)
        {
            return u;
        }
        else
        {
            return 0;
        }
    }
    /**
     * \brief Return the domain that is neighbor with domain dom.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_nbr_dom(const int dom) const
    {
        if (this->use_two_phases)
        {
            return 1-dom;
        }
        else
        {
            return -1;
        }
    }

    /**
     * \brief Fill U with initial conditions at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs(const amrex::Real * x, amrex::Real * U) const
    {
        U[0] = 1.0-x[0];
        if (this->use_two_phases)
        {
            U[1] = 1.0+x[1];
        }
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
        return (this->use_two_phases ? 2 : 1);
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int u) const
    {
        if (this->use_two_phases)
        {
            return u;
        }
        else
        {
            return 0;
        }
    }
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
        if (this->use_two_phases)
        {
            output_fields_names = {"U1", "U2"};
        }
        else
        {
            output_fields_names = {"U1"};
        }
        
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the unknown fields at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const amrex::Real t, const amrex::Real * x,
                            const amrex::Real * U,
                            amrex::Real * F) const
    {
        // OUTPUT FIELDS
        F[0] = U[0];
        if (this->use_two_phases)
        {
            F[1] = U[1];
        }
    }
    // ================================================================
};
// ####################################################################

} // QuadratureRules