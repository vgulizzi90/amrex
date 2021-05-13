//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_MappedGeometry.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

namespace MappedGeometry
{
// MAPPED GEOMETRY: LEVEL SETS ########################################
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
        amrex::Real X[3];

        /*
        X[0] = x[1];
        X[1] = x[2]*std::cos(x[0]);
        X[2] = x[2]*std::sin(x[0]);
        */

        /**/
        const amrex::Real Rc = 1.0;
        X[0] = x[1];
        X[1] = Rc*std::cos(x[0]);
        X[2] = Rc*std::sin(x[0]);
        /**/

        const amrex::Real r = 0.218;
        amrex::Real ls1, ls2;
        amrex::Real res;

        //ls1 = -(std::pow(X[0]-0.5, 6.0)+std::pow(X[2], 6.0)-std::pow(r, 6.0));
        //ls2 = -(std::pow(X[0]+0.5, 6.0)+std::pow(X[2], 6.0)-std::pow(r, 6.0));
        //res = amrex::max(ls1, ls2);

        //res = -1.0;

        ls2 = -(std::pow(X[0]-0.5, 6.0)+std::pow(X[2], 6.0)+std::pow(x[2]-0.25, 6.0)-std::pow(r, 6.0));
        res = ls2;

        return res;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval(const amrex::Real * x) const
    {
        return this->eval(0.0, x);
    }
    // ================================================================
};
// ####################################################################



// MAPPED GEOMETRY: IBVP ##############################################
#define IBVP_MAX_N_PARAMS 10
#define MAPPED_GEOMETRY_FLAG_NONE -1
#define MAPPED_GEOMETRY_FLAG_CUSTOM 100
#define MAPPED_GEOMETRY_FLAG_CIRCLE 0
#define MAPPED_GEOMETRY_FLAG_CYLINDER 1
#define MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL 10

struct IBVP
{
    // DATA MEMBERS ===================================================
    bool use_two_phases;
    int geometry_mapping_flag;
    amrex::Real params[IBVP_MAX_N_PARAMS];
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP()
    {
        // VARIABLES
        amrex::ParmParse pp;
        int i;
        std::string which_mapped_geometry;

        // READ INPUT FILE --------------------------------------------
        // USING TWO PHASES
        i = 0;
        pp.query("use_two_phases", i);
        this->use_two_phases = (i > 0);

        // MAPPED GEOMETRY
        which_mapped_geometry = "";
        pp.query("which_mapped_geometry", which_mapped_geometry);
        if (which_mapped_geometry.compare("custom") == 0)
        {
            this->geometry_mapping_flag = MAPPED_GEOMETRY_FLAG_CUSTOM;
        }
        else if (which_mapped_geometry.compare("circle") == 0)
        {
            this->geometry_mapping_flag = MAPPED_GEOMETRY_FLAG_CIRCLE;
            pp.get("mapped_geometry.x_coord", this->params[0]);
            pp.get("mapped_geometry.y_coord", this->params[1]);
        }
        else if (which_mapped_geometry.compare("cylinder") == 0)
        {
            this->geometry_mapping_flag = MAPPED_GEOMETRY_FLAG_CYLINDER;
        }
        else if (which_mapped_geometry.compare("cylinder_shell") == 0)
        {
            this->geometry_mapping_flag = MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL;
            pp.get("mapped_geometry.radius", this->params[0]);
        }
        else
        {
            this->geometry_mapping_flag = MAPPED_GEOMETRY_FLAG_NONE;
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // GEOMETRY MAPPING ===============================================
    /**
     * \brief Return the mapping of the geometry.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_map(const amrex::Real * x, amrex::Real * X) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_NONE)
        {
            AMREX_D_TERM
            (
                X[0] = x[0];,
                X[1] = x[1];,
                X[2] = x[2];
            )
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            amrex::Real un[3];

            this->eval_shell_map_and_unit_normal(x, X, un);

            X[0] += un[0]*x[2];
            X[1] += un[1]*x[2];
            X[2] += un[2]*x[2];
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CIRCLE)
        {
            const amrex::Real * c = this->params;
            
            X[0] = c[0]+x[0]*std::cos(x[1]);
            X[1] = c[1]+x[0]*std::sin(x[1]);
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER)
        {
            const amrex::Real * c = this->params;
            
            X[0] = x[1];
            X[1] = x[2]*std::cos(x[0]);
            X[2] = x[2]*std::sin(x[0]);
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            const amrex::Real r = this->params[0];
            
            X[0] = x[1];
            X[1] = r*std::cos(x[0]);
            X[2] = r*std::sin(x[0]);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_map\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }

    /**
     * \brief Return the covariant basis of the mapping of the geometry.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_map_covariant_basis(const amrex::Real * x, amrex::Real * gl) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            amrex::Real X[3], X_1[3], X_2[3], un[3], un_1[3], un_2[3];

            this->eval_shell_map_derivatives(x, X, X_1, X_2);
            this->eval_shell_map_unit_normal_derivatives(x, un, un_1, un_2);

            gl[0+0*3] = X_1[0]+x[2]*un_1[0]; gl[0+1*3] = X_2[0]+x[2]*un_2[0]; gl[0+2*3] = un[0];
            gl[1+0*3] = X_1[1]+x[2]*un_1[1]; gl[1+1*3] = X_2[1]+x[2]*un_2[1]; gl[1+2*3] = un[1];
            gl[2+0*3] = X_1[2]+x[2]*un_1[2]; gl[2+1*3] = X_2[2]+x[2]*un_2[2]; gl[2+2*3] = un[2];
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_map_covariant_basis\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }

    /**
     * \brief Return the derivatives of the mapping of the geometry.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_derivatives(const amrex::Real * x, amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            const amrex::Real h = 0.25;
            const amrex::Real a1 = -0.125;
            const amrex::Real a2 = +0.125;

            X[0] = x[0];
            X[1] = x[1];
            X[2] = h+a1*std::sin(M_PI*x[0])+a2*std::sin(M_PI*x[1]);

            X_1[0] = 1.0;
            X_1[1] = 0.0;
            X_1[2] = M_PI*a1*std::cos(M_PI*x[0]);
            X_2[0] = 0.0;
            X_2[1] = 1.0;
            X_2[2] = M_PI*a2*std::cos(M_PI*x[1]);
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            const amrex::Real r = this->params[0];
            
            X[0] = x[1];
            X[1] = r*std::cos(x[0]);
            X[2] = r*std::sin(x[0]);

            X_1[0] = 0.0;
            X_1[1] = -r*std::sin(x[0]);
            X_1[2] = r*std::cos(x[0]);
            X_2[0] = 1.0;
            X_2[1] = 0.0;
            X_2[2] = 0.0;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_shell_map_derivatives\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }

    /**
     * \brief Return the derivatives of the mapping of the geometry.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_derivatives(const amrex::Real * x, amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2,
                                    amrex::Real * X_11, amrex::Real * X_22, amrex::Real * X_12) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            const amrex::Real h = 0.25;
            const amrex::Real a1 = -0.125;
            const amrex::Real a2 = +0.125;

            X[0] = x[0];
            X[1] = x[1];
            X[2] = h+a1*std::sin(M_PI*x[0])+a2*std::sin(M_PI*x[1]);

            X_1[0] = 1.0;
            X_1[1] = 0.0;
            X_1[2] = M_PI*a1*std::cos(M_PI*x[0]);
            X_2[0] = 0.0;
            X_2[1] = 1.0;
            X_2[2] = M_PI*a2*std::cos(M_PI*x[1]);

            X_11[0] = 0.0;
            X_11[1] = 0.0;
            X_11[2] = -M_PI*M_PI*a1*std::sin(M_PI*x[0]);
            X_22[0] = 0.0;
            X_22[1] = 0.0;
            X_22[2] = -M_PI*M_PI*a2*std::sin(M_PI*x[1]);
            X_12[0] = 0.0;
            X_12[1] = 0.0;
            X_12[2] = 0.0;
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            const amrex::Real r = this->params[0];
            
            X[0] = x[1];
            X[1] = r*std::cos(x[0]);
            X[2] = r*std::sin(x[0]);

            X_1[0] = 0.0;
            X_1[1] = -r*std::sin(x[0]);
            X_1[2] = r*std::cos(x[0]);
            X_2[0] = 1.0;
            X_2[1] = 0.0;
            X_2[2] = 0.0;

            X_11[0] = 0;
            X_11[1] = -r*std::cos(x[0]);
            X_11[2] = -r*std::sin(x[0]);
            X_22[0] = 0;
            X_22[1] = 0;
            X_22[2] = 0;
            X_12[0] = 0;
            X_12[1] = 0;
            X_12[2] = 0;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_shell_map_derivatives\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }

    /**
     * \brief Return the derivatives of the mapping of the geometry.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_derivatives(const amrex::Real * x, amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2,
                                    amrex::Real * X_11, amrex::Real * X_22, amrex::Real * X_12,
                                    amrex::Real * X_111, amrex::Real * X_222, amrex::Real * X_112, amrex::Real * X_122) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            const amrex::Real h = 0.25;
            const amrex::Real a1 = -0.125;
            const amrex::Real a2 = +0.125;

            X[0] = x[0];
            X[1] = x[1];
            X[2] = h+a1*std::sin(M_PI*x[0])+a2*std::sin(M_PI*x[1]);

            X_1[0] = 1.0;
            X_1[1] = 0.0;
            X_1[2] = M_PI*a1*std::cos(M_PI*x[0]);
            X_2[0] = 0.0;
            X_2[1] = 1.0;
            X_2[2] = M_PI*a2*std::cos(M_PI*x[1]);

            X_11[0] = 0.0;
            X_11[1] = 0.0;
            X_11[2] = -M_PI*M_PI*a1*std::sin(M_PI*x[0]);
            X_22[0] = 0.0;
            X_22[1] = 0.0;
            X_22[2] = -M_PI*M_PI*a2*std::sin(M_PI*x[1]);
            X_12[0] = 0.0;
            X_12[1] = 0.0;
            X_12[2] = 0.0;

            X_111[0] = 0.0;
            X_111[1] = 0.0;
            X_111[2] = -M_PI*M_PI*M_PI*a1*std::cos(M_PI*x[0]);
            X_222[0] = 0.0;
            X_222[1] = 0.0;
            X_222[2] = -M_PI*M_PI*M_PI*a2*std::cos(M_PI*x[1]);
            X_112[0] = 0.0;
            X_112[1] = 0.0;
            X_112[2] = 0.0;
            X_122[0] = 0.0;
            X_122[1] = 0.0;
            X_122[2] = 0.0;
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            const amrex::Real r = this->params[0];
            
            X[0] = x[1];
            X[1] = r*std::cos(x[0]);
            X[2] = r*std::sin(x[0]);

            X_1[0] = 0.0;
            X_1[1] = -r*std::sin(x[0]);
            X_1[2] = r*std::cos(x[0]);
            X_2[0] = 1.0;
            X_2[1] = 0.0;
            X_2[2] = 0.0;

            X_11[0] = 0.0;
            X_11[1] = -r*std::cos(x[0]);
            X_11[2] = -r*std::sin(x[0]);
            X_22[0] = 0.0;
            X_22[1] = 0.0;
            X_22[2] = 0.0;
            X_12[0] = 0.0;
            X_12[1] = 0.0;
            X_12[2] = 0.0;

            X_111[0] = 0.0;
            X_111[1] = r*std::sin(x[0]);
            X_111[2] = -r*std::cos(x[0]);
            X_222[0] = 0.0;
            X_222[1] = 0.0;
            X_222[2] = 0.0;
            X_112[0] = 0.0;
            X_112[1] = 0.0;
            X_112[2] = 0.0;
            X_122[0] = 0.0;
            X_122[1] = 0.0;
            X_122[2] = 0.0;
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_shell_map_derivatives\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }

    /**
     * \brief Return the unit normal of the mid surface of a shell.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_and_unit_normal(const amrex::Real * x, amrex::Real * X, amrex::Real * un) const
    {
        // VARIABLES
        amrex::Real X_1[3], X_2[3];
        amrex::Real tmp;

        this->eval_shell_map_derivatives(x, X, X_1, X_2);
        
        amrex::dG::linalg::cross3(X_1, X_2, un);
        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(un, un));
        un[0] *= tmp;
        un[1] *= tmp;
        un[2] *= tmp;
    }

    /**
     * \brief Return the unit normal of the mid surface of a shell.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_unit_normal(const amrex::Real * x, amrex::Real * un) const
    {
        // VARIABLES
        amrex::Real X[3];

        this->eval_shell_map_and_unit_normal(x, X, un);
    }

    /**
     * \brief Return the unit normal of the mid surface of a shell.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_map_unit_normal_derivatives(const amrex::Real * x, amrex::Real * un, amrex::Real * un_1, amrex::Real * un_2) const
    {
        // VARIABLES
        amrex::Real X[3], X_1[3], X_2[3], X_11[3], X_22[3], X_12[3];
        amrex::Real M[9], tmp, v_1[3], v_2[3];

        this->eval_shell_map_derivatives(x, X, X_1, X_2, X_11, X_22, X_12);
        
        amrex::dG::linalg::cross3(X_1, X_2, un);
        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(un, un));
        un[0] *= tmp;
        un[1] *= tmp;
        un[2] *= tmp;

        amrex::dG::linalg::cross3(X_11, X_2, M);
        amrex::dG::linalg::cross3(X_1, X_12, &M[3]);
        v_1[0] = M[0]+M[3];
        v_1[1] = M[1]+M[4];
        v_1[2] = M[2]+M[5];

        amrex::dG::linalg::cross3(X_12, X_2, M);
        amrex::dG::linalg::cross3(X_1, X_22, &M[3]);
        v_2[0] = M[0]+M[3];
        v_2[1] = M[1]+M[4];
        v_2[2] = M[2]+M[5];

        M[0+0*3] = 1.0-un[0]*un[0]; M[0+1*3] =    -un[0]*un[1]; M[0+2*3] =    -un[0]*un[2];
        M[1+0*3] =    -un[1]*un[0]; M[1+1*3] = 1.0-un[1]*un[1]; M[1+2*3] =    -un[1]*un[2];
        M[2+0*3] =    -un[2]*un[0]; M[2+1*3] =    -un[2]*un[1]; M[2+2*3] = 1.0-un[2]*un[2];

        un_1[0] = (M[0+0*3]*v_1[0]+M[0+1*3]*v_1[1]+M[0+2*3]*v_1[2])*tmp;
        un_1[1] = (M[1+0*3]*v_1[0]+M[1+1*3]*v_1[1]+M[1+2*3]*v_1[2])*tmp;
        un_1[2] = (M[2+0*3]*v_1[0]+M[2+1*3]*v_1[1]+M[2+2*3]*v_1[2])*tmp;

        un_2[0] = (M[0+0*3]*v_2[0]+M[0+1*3]*v_2[1]+M[0+2*3]*v_2[2])*tmp;
        un_2[1] = (M[1+0*3]*v_2[0]+M[1+1*3]*v_2[1]+M[1+2*3]*v_2[2])*tmp;
        un_2[2] = (M[2+0*3]*v_2[0]+M[2+1*3]*v_2[1]+M[2+2*3]*v_2[2])*tmp;
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
        if (this->use_two_phases)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        if (this->use_two_phases)
        {
            return 2;
        }
        else
        {
            return 1;
        }
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
        int n_output_fields;

        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_NONE)
        {
            if (this->use_two_phases)
            {
                n_output_fields = 2;
            }
            else
            {
                n_output_fields = 1;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            if (this->use_two_phases)
            {
                n_output_fields = 2+3;
            }
            else
            {
                n_output_fields = 1+3;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CIRCLE)
        {
            if (this->use_two_phases)
            {
                n_output_fields = 2+2;
            }
            else
            {
                n_output_fields = 1+2;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER)
        {
            if (this->use_two_phases)
            {
                n_output_fields = 2+3;
            }
            else
            {
                n_output_fields = 1+3;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            if (this->use_two_phases)
            {
                n_output_fields = 2+3+3;
            }
            else
            {
                n_output_fields = 1+3+3;
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.get_number_of_output_fields\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }

        return n_output_fields;
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int u) const
    {
        int dom;

        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_NONE)
        {
            dom = u;
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            if (this->use_two_phases)
            {
                dom = ((u == 1) ? 1 : 0);
            }
            else
            {
                dom = 0;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CIRCLE)
        {
            if (this->use_two_phases)
            {
                dom = ((u == 1) ? 1 : 0);
            }
            else
            {
                dom = 0;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER)
        {
            if (this->use_two_phases)
            {
                dom = ((u == 1) ? 1 : 0);
            }
            else
            {
                dom = 0;
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            if (this->use_two_phases)
            {
                dom = ((u == 1) ? 1 : 0);
            }
            else
            {
                dom = 0;
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.output_field_parent_domain\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }

        return dom;
    }
    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_NONE)
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
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            if (this->use_two_phases)
            {
                output_fields_names = {"U1", "U2",
                                       "X1", "X2", "X3"};
            }
            else
            {
                output_fields_names = {"U1",
                                       "X1", "X2", "X3"};
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CIRCLE)
        {
            if (this->use_two_phases)
            {
                output_fields_names = {"U1", "U2", "X1", "X2"};
            }
            else
            {
                output_fields_names = {"U1", "X1", "X2"};
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER)
        {
            if (this->use_two_phases)
            {
                output_fields_names = {"U1", "U2",
                                       "X1", "X2", "X3"};
            }
            else
            {
                output_fields_names = {"U1",
                                       "X1", "X2", "X3"};
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            if (this->use_two_phases)
            {
                output_fields_names = {"U1", "U2",
                                       "X1", "X2", "X3",
                                       "un1", "un2", "un3"};
            }
            else
            {
                output_fields_names = {"U1",
                                       "X1", "X2", "X3",
                                       "un1", "un2", "un3"};
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.get_output_fields_names\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
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
        // VARIABLES
        amrex::Real X[3];

        // MAP
        this->eval_map(x, X);

        // OUTPUT FIELDS
        if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_NONE)
        {
            if (this->use_two_phases)
            {
                F[0] = U[0];
                F[1] = U[1];
            }
            else
            {
                F[0] = U[0];
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CUSTOM)
        {
            if (this->use_two_phases)
            {
                F[0] = U[0];
                F[1] = U[1];
                F[2] = X[0];
                F[3] = X[1];
                F[4] = X[2];
            }
            else
            {
                F[0] = U[0];
                F[1] = X[0];
                F[2] = X[1];
                F[3] = X[2];
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CIRCLE)
        {
            if (this->use_two_phases)
            {
                F[0] = U[0];
                F[1] = U[1];
                F[2] = X[0];
                F[3] = X[1];
            }
            else
            {
                F[0] = U[0];
                F[1] = X[0];
                F[2] = X[1];
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER)
        {
            if (this->use_two_phases)
            {
                F[0] = U[0];
                F[1] = U[1];
                F[2] = X[0];
                F[3] = X[1];
                F[4] = X[2];
            }
            else
            {
                F[0] = U[0];
                F[1] = X[0];
                F[2] = X[1];
                F[3] = X[2];
            }
        }
        else if (this->geometry_mapping_flag == MAPPED_GEOMETRY_FLAG_CYLINDER_SHELL)
        {
            amrex::Real un[3];
            this->eval_shell_map_unit_normal(x, un);

            if (this->use_two_phases)
            {
                F[0] = U[0];
                F[1] = U[1];
                F[2] = X[0];
                F[3] = X[1];
                F[4] = X[2];
                F[5] = un[0];
                F[6] = un[1];
                F[7] = un[2];
            }
            else
            {
                F[0] = U[0];
                F[1] = X[0];
                F[2] = X[1];
                F[3] = X[2];
                F[4] = un[0];
                F[5] = un[1];
                F[6] = un[2];
            }
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MappedGeometry.H - IBVP.eval_output_fields\n";
            msg += "| Unexpected mapped geometry flag: "+std::to_string(this->geometry_mapping_flag)+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================
};
// ####################################################################

} // QuadratureRules