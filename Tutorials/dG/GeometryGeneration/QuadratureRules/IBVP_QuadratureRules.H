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
    amrex::Real eval(const amrex::Real * x) const
    {
        const int sgn = -(1-2*this->dom);
        amrex::Real res;

        res = x[2]-0.6+0.4*x[1]*x[1]*x[1]-0.2*x[0]*x[0];
        res *= sgn;

        return res;
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

        res[0] = -0.2*2.0*x[0];
        res[1] = 0.4*3.0*x[1]*x[1];
        res[2] = 1.0;

        res[0] *= sgn;
        res[1] *= sgn;
        res[2] *= sgn;

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
    // ================================================================
};
// ####################################################################

} // QuadratureRules