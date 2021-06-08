//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_MultilayeredPlate.H
 * \brief Contains constants and data structures for the considered IBVP.
*/

#include "../../BaseSolidMechanics/IBVP_LinearElasticity.H"

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define NS 6

// ORDERING OF THE ELASTIC COEFFICIENTS
#define C11 0
#define C21 1
#define C31 2
#define C41 3
#define C51 4
#define C61 5

#define C22 6
#define C32 7
#define C42 8
#define C52 9
#define C62 10

#define C33 11
#define C43 12
#define C53 13
#define C63 14

#define C44 15
#define C54 16
#define C64 17

#define C55 18
#define C65 19

#define C66 20

/*
#define INSERT_HOLE
#define HOLE_RADIUS 0.11
*/
// ####################################################################


namespace multilayered_plate
{
// SINGLE PHASE DOMAIN: LEVEL SET #####################################
template <int N>
struct LevelSet
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    int dom;
    // ================================================================


    // CONSTRUCTOR ====================================================
    LevelSet()
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
    amrex::Real eval(const amrex::Real * x) const
    {
#ifdef INSERT_HOLE
        const amrex::Real xc[2] = {0.5, 0.5};
        const amrex::Real res = HOLE_RADIUS*HOLE_RADIUS-((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
#else
        const amrex::Real res = -1.0;
#endif
        return res;
    }
    // ================================================================


    // ROUTINES REQUIRED BY ALGOIM ====================================
    template <typename T>
    T operator() (const amrex::GpuArray<T, N> & x) const
    {
#ifdef INSERT_HOLE
        const amrex::Real xc[2] = {0.5, 0.5};
        const T res = HOLE_RADIUS*HOLE_RADIUS-((x[0]-xc[0])*(x[0]-xc[0])+(x[1]-xc[1])*(x[1]-xc[1]));
#else
        const T res = (x[0]-x[0]-1.0);
#endif
        return res;
    }

    template <typename T>
    amrex::GpuArray<T, N> grad(const amrex::GpuArray<T, N> & x) const
    {
#ifdef INSERT_HOLE
        const amrex::Real xc[2] = {0.5, 0.5};
        const amrex::GpuArray<T, N> res = {-2.0*(x[0]-xc[0]), -2.0*(x[1]-xc[1])};
#else
        const amrex::GpuArray<T, N> res = {AMREX_D_DECL(x[0]-x[0], x[0]-x[0], x[0]-x[0])};
#endif

        return res;
    }
    // ================================================================
};
// ####################################################################



// SINGLE PHASE DOMAIN: IBVP ##########################################
#define MAX_N_MATERIALS 2
#define MAX_N_LAYERS 8

#define SHELL_THEORY_2D -2
#define SHELL_THEORY_FSDT 0
#define SHELL_THEORY 0

#if (SHELL_THEORY == SHELL_THEORY_2D)
#define SHELL_NU 2
#endif
#if (SHELL_THEORY == SHELL_THEORY_FSDT)
#define SHELL_NU 5
#endif

#define QTH 3

struct Material
{
    amrex::Real density;
    amrex::Real cSE[NS*(NS+1)/2];
};

struct IBVP
{
    // DATA MEMBERS ===================================================
    Material materials[MAX_N_MATERIALS];
    int n_layers;
    int layers_material[MAX_N_LAYERS];
    amrex::Real layers_thickness[MAX_N_LAYERS], layers_x3b[MAX_N_LAYERS];
    amrex::Real layers_orientation[MAX_N_LAYERS];
    int theory[3];

    amrex::Real thickness;
    amrex::Real max_wave_speed;
    // Penalty parameters
    const amrex::Real mu0 = 10.0;
    amrex::Real muD, muI;

    amrex::Real xGth[QTH+1], wGth[QTH+1];

    amrex::Real M[SHELL_NU*SHELL_NU];
    amrex::Real Q11[SHELL_NU*SHELL_NU], Q12[SHELL_NU*SHELL_NU], Q22[SHELL_NU*SHELL_NU];
    amrex::Real R1[SHELL_NU*SHELL_NU], R2[SHELL_NU*SHELL_NU], S[SHELL_NU*SHELL_NU];

    amrex::Real Us[SHELL_NU];
    amrex::Real wA[SHELL_NU], vA[SHELL_NU*SHELL_NU], cvA[SHELL_NU];
    // ================================================================


    // CONSTRUCTOR ====================================================
    IBVP(const amrex::Real * dx)
    {
        // VARIABLES --------------------------------------------------
        amrex::ParmParse pp;

        int n_materials;
        // ------------------------------------------------------------

        // READ MATERIAL PROPERTIES -----------------------------------
        pp.get("n_materials", n_materials);

        if (n_materials <= 0)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
            msg += "| Number of materials must be positive.\n";
            amrex::Abort(msg);
        }
        else if (n_materials > MAX_N_MATERIALS)
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
            msg += "| Number of materials cannot exceed "+std::to_string(MAX_N_MATERIALS)+".\n";
            amrex::Abort(msg);
        }

        for (int m = 0; m < n_materials; ++m)
        {
            const std::string material_string = "material["+std::to_string(m)+"]";
            amrex::ParmParse pp_mat(material_string.c_str());
            
            std::string material_type;
            amrex::Vector<amrex::Real> material_constants;
            int use_plane_stress;
            amrex::Real shear_factor;
            Material & material = this->materials[m];
            amrex::Real cSE[NS*NS];

            pp_mat.get("type", material_type);
            pp_mat.getarr("constants", material_constants);
            use_plane_stress = 0;
            pp_mat.query("use_plane_stress", use_plane_stress);
            shear_factor = 1.0;
            pp_mat.query("shear_factor", shear_factor);

            if (material_type.compare("isotropic") == 0)
            {
                amrex::Real E, nu;

                E = material_constants[1];
                nu = material_constants[2];

                if (use_plane_stress > 0)
                {
                    linear_elasticity::get_stiffness_matrix_3d_isotropic_plane_stress(E, nu, cSE, shear_factor);
                }
                else
                {
                    linear_elasticity::get_stiffness_matrix_3d_isotropic(E, nu, cSE);
                }
            }
            else if (material_type.compare("orthotropic_eng_constants") == 0)
            {
                amrex::Real E1, E2, E3, nu23, nu13, nu12, G23, G13, G12;

                E1 = material_constants[1];
                E2 = material_constants[2];
                E3 = material_constants[3];
                nu23 = material_constants[4];
                nu13 = material_constants[5];
                nu12 = material_constants[6];
                G23 = material_constants[7];
                G13 = material_constants[8];
                G12 = material_constants[9];

                if (use_plane_stress > 0)
                {
                    linear_elasticity::get_stiffness_matrix_3d_orthotropic_eng_constants_plane_stress(E1, E2, E3, nu23, nu13, nu12, G23, G13, G12, cSE, shear_factor);
                }
                else
                {
                    linear_elasticity::get_stiffness_matrix_3d_orthotropic_eng_constants(E1, E2, E3, nu23, nu13, nu12, G23, G13, G12, cSE);
                }
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                msg += "| Unexpected input material type: "+material_type+".\n";
                amrex::Abort(msg);
            }

            // STORE DENSITY
            material.density = material_constants[0];

            // STORE STIFFNESS MATRIX COEFFICIENTS
            material.cSE[C11] = cSE[0+0*NS];
            material.cSE[C21] = cSE[1+0*NS]; material.cSE[C22] = cSE[1+1*NS];
            material.cSE[C31] = cSE[2+0*NS]; material.cSE[C32] = cSE[2+1*NS]; material.cSE[C33] = cSE[2+2*NS];
            material.cSE[C41] = cSE[3+0*NS]; material.cSE[C42] = cSE[3+1*NS]; material.cSE[C43] = cSE[3+2*NS]; material.cSE[C44] = cSE[3+3*NS];
            material.cSE[C51] = cSE[4+0*NS]; material.cSE[C52] = cSE[4+1*NS]; material.cSE[C53] = cSE[4+2*NS]; material.cSE[C54] = cSE[4+3*NS]; material.cSE[C55] = cSE[4+4*NS];
            material.cSE[C61] = cSE[5+0*NS]; material.cSE[C62] = cSE[5+1*NS]; material.cSE[C63] = cSE[5+2*NS]; material.cSE[C64] = cSE[5+3*NS]; material.cSE[C65] = cSE[5+4*NS]; material.cSE[C66] = cSE[5+5*NS];
        }
        // ------------------------------------------------------------

        // READ SHELL PROPERTIES --------------------------------------
        {
            amrex::Vector<int> int_aux;
            amrex::Vector<amrex::Real> int_real;
            int Nu;

            pp.get("shell_n_layers", this->n_layers);
            if (this->n_layers > MAX_N_LAYERS)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                msg += "| Number of layers cannot exceed "+std::to_string(MAX_N_LAYERS)+".\n";
                amrex::Abort(msg);
            }

            pp.getarr("shell_layers_material", int_aux);
            for (int l = 0; l < this->n_layers; ++l)
            {
                this->layers_material[l] = int_aux[l];
            }
            pp.getarr("shell_layers_thickness", int_real);
            for (int l = 0; l < this->n_layers; ++l)
            {
                this->layers_thickness[l] = int_real[l];
            }
            pp.getarr("shell_layers_orientation", int_real);
            for (int l = 0; l < this->n_layers; ++l)
            {
                this->layers_orientation[l] = int_real[l]*M_PI/180.0;
            }

            pp.getarr("shell_theory", int_aux);
            Nu = 0;
            for (int u = 0; u < 3; ++u)
            {
                this->theory[u] = int_aux[u];
                Nu += int_aux[u]+1;
            }

            if (Nu != SHELL_NU)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                msg += "| Shell theory[0]+theory[1]+theory[2]+3 does not match "+std::to_string(SHELL_NU)+".\n";
                msg += "| Shell theory: "+std::to_string(int_aux[0])+" "+std::to_string(int_aux[1])+" "+std::to_string(int_aux[2])+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------

        // EVAL SHELL THICKNESS ---------------------------------------
        {
            this->thickness = 0.0;
            for (int l = 0; l < this->n_layers; ++l)
            {
                this->thickness += this->layers_thickness[l];
            }
            this->layers_x3b[0] = -0.5*this->thickness;
            for (int l = 1; l < this->n_layers; ++l)
            {
                this->layers_x3b[l] = this->layers_x3b[l-1]+this->layers_thickness[l];
            }
        }
        // ------------------------------------------------------------

        // EVAL MAX WAVE SPEED / PENALTY PARAMETER --------------------
        {
            amrex::Vector<int> material_ids;
            for (int l = 0; l < this->n_layers; ++l)
            {
                if (std::count(material_ids.begin(), material_ids.end(), this->layers_material[l]) == 0)
                {
                    material_ids.push_back(this->layers_material[l]);
                }
            }

            this->max_wave_speed = 0.0;
            for (int m = 0; m < material_ids.size(); ++m)
            {
                const Material & material = this->materials[material_ids[m]];
                this->max_wave_speed = amrex::max(this->max_wave_speed, linear_elasticity::eval_max_wave_speed_3d_compact_c(material.density, material.cSE));
            }

            const amrex::Real ws2 = this->max_wave_speed*this->max_wave_speed;
            const amrex::Real h = amrex::min(dx[0], dx[1]);
            
            this->muD = this->mu0*ws2/h;
            this->muI = this->mu0*ws2/h;

            /* DEBUG */
            amrex::Print() << "Max wave speed: " << this->max_wave_speed << std::endl;
            amrex::Print() << "Penalty parameter BCs: " << this->muD << std::endl;
            amrex::Print() << "Penalty parameter ICs: " << this->muI << std::endl;
            /**/
        }
        // ------------------------------------------------------------

        // STORE QUADRATURE ABSCISSAE AND WEIGHTS IN THE [0,1] INTERVAL
        for (int q = 0; q < (1+QTH); ++q)
        {
            this->xGth[q] = Algoim::GaussQuad::x(1+QTH, q);

            this->wGth[q] = Algoim::GaussQuad::w(1+QTH, q);
        }
        // ------------------------------------------------------------

        // EVAL AND STORE GENERALIZED MATRICES ------------------------
        {
            const amrex::Real x[2] = {0.0, 0.0};
            this->shell_theory_eval_mass_matrix(x, this->M);
            this->shell_theory_eval_stiffness_matrices(x, this->Q11, this->Q22, this->Q12, this->R1, this->R2, this->S);
        }
        // ------------------------------------------------------------

        // EVAL DATA STRUCTURES FOR THE EXACT SOLUTION ----------------
        {
            const int m1 = 1;
            const amrex::Real L1 = 1.0;
            const int m2 = 1;
            const amrex::Real L2 = 1.0;

            int info;
            amrex::Real I[SHELL_NU*SHELL_NU], I1[SHELL_NU*SHELL_NU], I2[SHELL_NU*SHELL_NU], I11[SHELL_NU*SHELL_NU], I12[SHELL_NU*SHELL_NU], I22[SHELL_NU*SHELL_NU];
            amrex::Real iM[SHELL_NU*SHELL_NU];
            amrex::Real A[SHELL_NU*SHELL_NU];

            // AUXILIARY MATRICES
            amrex::dG::linalg::eye(SHELL_NU, I);
            amrex::dG::linalg::eye(SHELL_NU, I1);
            amrex::dG::linalg::eye(SHELL_NU, I2);
            amrex::dG::linalg::eye(SHELL_NU, I11);
            amrex::dG::linalg::eye(SHELL_NU, I12);
            amrex::dG::linalg::eye(SHELL_NU, I22);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                if ((u%3) == 0)
                {
                    I1[u+u*SHELL_NU] *= -(m1*M_PI/L1);
                    I2[u+u*SHELL_NU] *= +(m2*M_PI/L2);
                    I12[u+u*SHELL_NU] *= -(m1*M_PI/L1)*(m2*M_PI/L2);
                }
                else if ((u%3) == 1)
                {
                    I1[u+u*SHELL_NU] *= +(m1*M_PI/L1);
                    I2[u+u*SHELL_NU] *= -(m2*M_PI/L2);
                    I12[u+u*SHELL_NU] *= -(m1*M_PI/L1)*(m2*M_PI/L2);
                }
                else if ((u%3) == 2)
                {
                    I1[u+u*SHELL_NU] *= +(m1*M_PI/L1);
                    I2[u+u*SHELL_NU] *= +(m2*M_PI/L2);
                    I12[u+u*SHELL_NU] *= +(m1*M_PI/L1)*(m2*M_PI/L2);
                }

                I11[u+u*SHELL_NU] *= -(m1*M_PI/L1)*(m1*M_PI/L1);
                I22[u+u*SHELL_NU] *= -(m2*M_PI/L2)*(m2*M_PI/L2);
            }

            // FACTORIZATION OF THE MASS MATRIX
            std::copy(this->M, this->M+SHELL_NU*SHELL_NU, iM);
            amrex::dG::linalg::dpotf2('U', SHELL_NU, iM, SHELL_NU, info);

            amrex::Print() << "iM: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, iM);

            if (info != 0)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                msg += "| Something went wrong during the factorization of the mass matrix (dpotf2).\n";
                msg += "| info: "+std::to_string(info)+".\n";
                amrex::Abort(msg);
            }
            
            // STIFFNESS MATRIX
            std::fill(A, A+SHELL_NU*SHELL_NU, 0.0);

            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, 1.0, this->S, SHELL_NU, I, SHELL_NU, 0.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, SHELL_NU, 1.0, this->R1, SHELL_NU, I1, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, SHELL_NU, 1.0, this->R2, SHELL_NU, I2, SHELL_NU, 1.0, A, SHELL_NU);
            
            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->Q11, SHELL_NU, I11, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->Q12, SHELL_NU, I12, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->R1, SHELL_NU, I1, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->Q12, SHELL_NU, I12, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->Q22, SHELL_NU, I22, SHELL_NU, 1.0, A, SHELL_NU);
            amrex::dG::linalg::dgemm('N', 'N', SHELL_NU, SHELL_NU, SHELL_NU, -1.0, this->R2, SHELL_NU, I2, SHELL_NU, 1.0, A, SHELL_NU);

            // PARTICULAR SOLUTION
            {
                const amrex::Real x[2] = {0.0, 0.0};

                const amrex::Real q1 = 0.0, q2 = 0.0, q3 = 1.0;
                const amrex::Real tt[3] = {q1,
                                           q2,
                                           q3};
                const int l = this->n_layers-1;
                const amrex::Real x3b = this->layers_x3b[l];
                const amrex::Real tau = this->layers_thickness[l];
                const amrex::Real x3 = x3b+tau;
                const amrex::Real g = this->shell_map_eval_surface_Jacobian(x, x3, 2);

                amrex::Real iK[SHELL_NU*SHELL_NU], RF[3*SHELL_NU];

                this->shell_theory_eval_RF_operator(x, x3, RF);

                amrex::dG::linalg::dgemv('T', 3, SHELL_NU, g, RF, 3, tt, 1, 0.0, this->Us, 1);

                //amrex::Print() << "Us: " << std::endl;
                //amrex::dG::io::print_real_array_2d(SHELL_NU, 1, this->Us);

                // FACTORIZATION OF THE STIFFNESS MATRIX
                std::copy(A, A+SHELL_NU*SHELL_NU, iK);
                amrex::dG::linalg::dpotf2('U', SHELL_NU, iK, SHELL_NU, info);

                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                    msg += "| Something went wrong during the factorization of the stiffness matrix (dpotf2).\n";
                    msg += "| info: "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

                amrex::Print() << "iK: " << std::endl;
                amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, iK);

                // MULTIPLY BY INVERSE STIFFNESS MATRIX
                // (STATIC SOLUTION)
                amrex::dG::linalg::dpotrs('U', SHELL_NU, 1, iK, SHELL_NU, this->Us, SHELL_NU, info);

                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                    msg += "| Something went wrong during the inversion of the stiffness matrix (dpotrs).\n";
                    msg += "| info: "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

                //amrex::Print() << "Us: " << std::endl;
                //amrex::dG::io::print_real_array_2d(SHELL_NU, 1, this->Us);
            }

            // MULTIPLY BY INVERSE MASS MATRIX
            amrex::dG::linalg::dpotrs('U', SHELL_NU, SHELL_NU, iM, SHELL_NU, A, SHELL_NU, info);

            if (info != 0)
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                msg += "| Something went wrong during the inversion of the mass matrix (dpotrs).\n";
                msg += "| info: "+std::to_string(info)+".\n";
                amrex::Abort(msg);
            }

            // EIGENVALUE PROBLEM
            {
                char jobvl = 'N';
                char jobvr = 'V';
                int n = SHELL_NU;
                amrex::Real A_copy[SHELL_NU*SHELL_NU];
                amrex::Real wA_im[SHELL_NU];
                amrex::Real work[SHELL_NU*SHELL_NU];
                int lwork = SHELL_NU*SHELL_NU;

                std::copy(A, A+SHELL_NU*SHELL_NU, A_copy);
                dgeev_(&jobvl, &jobvr, &n, A_copy, &n, wA, wA_im, nullptr, &n, this->vA, &n, work, &lwork, &info);

                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                    msg += "| Something went wrong during the solution of the eigenproblem (dgeev_).\n";
                    msg += "| info: "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

                // STORE EIGENVALUES
                for (int u = 0; u < SHELL_NU; ++u)
                {
                    this->wA[u] = std::sqrt(this->wA[u]);
                }
            }

            amrex::Print() << "Eigenvalues: " << std::endl;
            amrex::dG::io::print_real_array_2d(1, SHELL_NU, this->wA);
            amrex::Print() << "Right eigenvectors: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, this->vA);
            amrex::Print() << "Period: " << 2.0*M_PI/(*std::min_element(this->wA, this->wA+SHELL_NU)) << std::endl;

            // COEFFICIENTS OF THE EIGENVECTORS
            {
                amrex::Real vA_copy[SHELL_NU*SHELL_NU];
                int ipiv[SHELL_NU];

                std::copy(this->vA, this->vA+SHELL_NU*SHELL_NU, vA_copy);

                amrex::dG::linalg::dgetrf(SHELL_NU, SHELL_NU, vA_copy, SHELL_NU, ipiv, info);

                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                    msg += "| Something went wrong during the factorization of the right eigenvectors matrix (dgetrf).\n";
                    msg += "| info: "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }

                for (int u = 0; u < SHELL_NU; ++u)
                {
                    this->cvA[u] = -this->Us[u];
                }
                amrex::dG::linalg::dgetrs('N', SHELL_NU, 1, vA_copy, SHELL_NU, ipiv, this->cvA, SHELL_NU, info);

                if (info != 0)
                {
                    std::string msg;
                    msg  = "\n";
                    msg +=  "ERROR: IBVP_MultilayeredPlate.H - IBVP::IBVP\n";
                    msg += "| Something went wrong during the inversion of the right eigenvectors matrix (dgetrs).\n";
                    msg += "| info: "+std::to_string(info)+".\n";
                    amrex::Abort(msg);
                }
            }

            /* DEBUG */
            amrex::Print() << "A: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, A);
            amrex::Print() << "cvA: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, this->cvA);
            //exit(-1);
            /**/
        }
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAP ============================================================
    /**
     * \brief Return up to first-order derivatives of the shell map.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map(const amrex::Real * x, amrex::Real * X) const
    {
        X[0] = x[0];
        X[1] = x[1];
        X[2] = 0.0;
    }

    /**
     * \brief Return up to first-order derivatives of the shell map.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_volume(const amrex::Real * x, const amrex::Real x3, amrex::Real * X) const
    {
        amrex::Real un[3], un_1[3], un_2[3];

        this->shell_map(x, X);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2);

        X[0] += x3*un[0];
        X[1] += x3*un[1];
        X[2] += x3*un[2];
    }

    /**
     * \brief Return up to first-order derivatives of the shell map.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_derivatives(const amrex::Real * x,
                                    amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2) const
    {
        this->shell_map(x, X);

        X_1[0] = 1.0;
        X_1[1] = 0.0;
        X_1[2] = 0.0;
        X_2[0] = 0.0;
        X_2[1] = 1.0;
        X_2[2] = 0.0;
    }

    /**
     * \brief Return up to second-order derivatives of the shell map.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_derivatives(const amrex::Real * x,
                                    amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2,
                                    amrex::Real * X_11, amrex::Real * X_22, amrex::Real * X_12) const
    {
        this->shell_map_eval_derivatives(x, X, X_1, X_2);

        X_11[0] = 0.0;
        X_11[1] = 0.0;
        X_11[2] = 0.0;
        X_22[0] = 0.0;
        X_22[1] = 0.0;
        X_22[2] = 0.0;
        X_12[0] = 0.0;
        X_12[1] = 0.0;
        X_12[2] = 0.0;
    }

    /**
     * \brief Return up to third-order derivatives of the shell map.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_derivatives(const amrex::Real * x,
                                    amrex::Real * X,
                                    amrex::Real * X_1, amrex::Real * X_2,
                                    amrex::Real * X_11, amrex::Real * X_22, amrex::Real * X_12,
                                    amrex::Real * X_111, amrex::Real * X_222, amrex::Real * X_112, amrex::Real * X_122) const
    {
        this->shell_map_eval_derivatives(x, X, X_1, X_2, X_11, X_22, X_12);

        X_111[0] = 0.0;
        X_111[1] = 0.0;
        X_111[2] = 0.0;
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

    /**
     * \brief Return up to first-order derivatives of the unit normal of the shell's mid surface.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_unit_normal_derivatives(const amrex::Real * x,
                                                amrex::Real * un,
                                                amrex::Real * un_1, amrex::Real * un_2) const
    {
        // VARIABLES
        amrex::Real X[3], X_1[3], X_2[3], X_11[3], X_22[3], X_12[3];
        amrex::Real MM[9], tmp, v_1[3], v_2[3];

        this->shell_map_eval_derivatives(x, X, X_1, X_2, X_11, X_22, X_12);

        // We use MM as a temporary variable
        amrex::dG::linalg::cross3(X_11, X_2, MM);
        amrex::dG::linalg::cross3(X_1, X_12, &MM[3]);
        v_1[0] = MM[0]+MM[3];
        v_1[1] = MM[1]+MM[4];
        v_1[2] = MM[2]+MM[5];

        amrex::dG::linalg::cross3(X_12, X_2, MM);
        amrex::dG::linalg::cross3(X_1, X_22, &MM[3]);
        v_2[0] = MM[0]+MM[3];
        v_2[1] = MM[1]+MM[4];
        v_2[2] = MM[2]+MM[5];

        amrex::dG::linalg::cross3(X_1, X_2, un);
        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(un, un));
        un[0] *= tmp;
        un[1] *= tmp;
        un[2] *= tmp;

        MM[0+0*3] = (1.0-un[0]*un[0])*tmp; MM[0+1*3] = (   -un[0]*un[1])*tmp; MM[0+2*3] = (   -un[0]*un[2])*tmp;
        MM[1+0*3] = (   -un[1]*un[0])*tmp; MM[1+1*3] = (1.0-un[1]*un[1])*tmp; MM[1+2*3] = (   -un[1]*un[2])*tmp;
        MM[2+0*3] = (   -un[2]*un[0])*tmp; MM[2+1*3] = (   -un[2]*un[1])*tmp; MM[2+2*3] = (1.0-un[2]*un[2])*tmp;

        // un_{,1}
        un_1[0] = MM[0+0*3]*v_1[0]+MM[0+1*3]*v_1[1]+MM[0+2*3]*v_1[2];
        un_1[1] = MM[1+0*3]*v_1[0]+MM[1+1*3]*v_1[1]+MM[1+2*3]*v_1[2];
        un_1[2] = MM[2+0*3]*v_1[0]+MM[2+1*3]*v_1[1]+MM[2+2*3]*v_1[2];

        // un_{,2}
        un_2[0] = MM[0+0*3]*v_2[0]+MM[0+1*3]*v_2[1]+MM[0+2*3]*v_2[2];
        un_2[1] = MM[1+0*3]*v_2[0]+MM[1+1*3]*v_2[1]+MM[1+2*3]*v_2[2];
        un_2[2] = MM[2+0*3]*v_2[0]+MM[2+1*3]*v_2[1]+MM[2+2*3]*v_2[2];
    }

    /**
     * \brief Return up to second-order derivatives of the unit normal of the shell's mid surface.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_unit_normal_derivatives(const amrex::Real * x,
                                                amrex::Real * un,
                                                amrex::Real * un_1, amrex::Real * un_2,
                                                amrex::Real * un_11, amrex::Real * un_22, amrex::Real * un_12) const
    {
        // VARIABLES
        amrex::Real X[3], X_1[3], X_2[3], X_11[3], X_22[3], X_12[3], X_111[3], X_222[3], X_112[3], X_122[3];
        amrex::Real MM[12], tmp, v_1[3], v_2[3], v_11[3], v_22[3], v_12[3];
        amrex::Real d1, d2;
        amrex::Real M11, M21, M31, M22, M32, M33;
        amrex::Real A11, A21, A31, A22, A32, A33;

        this->shell_map_eval_derivatives(x, X, X_1, X_2, X_11, X_22, X_12, X_111, X_222, X_112, X_122);

        // We use MM as a temporary variable
        amrex::dG::linalg::cross3(X_11, X_2, MM);
        amrex::dG::linalg::cross3(X_1, X_12, &MM[3]);
        v_1[0] = MM[0]+MM[3];
        v_1[1] = MM[1]+MM[4];
        v_1[2] = MM[2]+MM[5];

        amrex::dG::linalg::cross3(X_12, X_2, MM);
        amrex::dG::linalg::cross3(X_1, X_22, &MM[3]);
        v_2[0] = MM[0]+MM[3];
        v_2[1] = MM[1]+MM[4];
        v_2[2] = MM[2]+MM[5];

        amrex::dG::linalg::cross3(X_111, X_2, MM);
        amrex::dG::linalg::cross3(X_11, X_12, &MM[3]);
        amrex::dG::linalg::cross3(X_11, X_12, &MM[6]);
        amrex::dG::linalg::cross3(X_1, X_112, &MM[9]);
        v_11[0] = MM[0]+MM[3]+MM[6]+MM[9];
        v_11[1] = MM[1]+MM[4]+MM[7]+MM[10];
        v_11[2] = MM[2]+MM[5]+MM[8]+MM[11];

        amrex::dG::linalg::cross3(X_122, X_2, MM);
        amrex::dG::linalg::cross3(X_12, X_22, &MM[3]);
        amrex::dG::linalg::cross3(X_12, X_22, &MM[6]);
        amrex::dG::linalg::cross3(X_1, X_222, &MM[9]);
        v_22[0] = MM[0]+MM[3]+MM[6]+MM[9];
        v_22[1] = MM[1]+MM[4]+MM[7]+MM[10];
        v_22[2] = MM[2]+MM[5]+MM[8]+MM[11];

        amrex::dG::linalg::cross3(X_112, X_2, MM);
        amrex::dG::linalg::cross3(X_11, X_22, &MM[3]);
        amrex::dG::linalg::cross3(X_12, X_12, &MM[6]);
        amrex::dG::linalg::cross3(X_1, X_122, &MM[9]);
        v_12[0] = MM[0]+MM[3]+MM[6]+MM[9];
        v_12[1] = MM[1]+MM[4]+MM[7]+MM[10];
        v_12[2] = MM[2]+MM[5]+MM[8]+MM[11];

        amrex::dG::linalg::cross3(X_1, X_2, un);
        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(un, un));
        un[0] *= tmp;
        un[1] *= tmp;
        un[2] *= tmp;

        M11 = (1.0-un[0]*un[0])*tmp;
        M21 = (   -un[1]*un[0])*tmp; M22 = (1.0-un[1]*un[1])*tmp;
        M31 = (   -un[2]*un[0])*tmp; M32 = (   -un[2]*un[1])*tmp; M33 = (1.0-un[2]*un[2])*tmp;

        // un_{,1}
        un_1[0] = M11*v_1[0]+M21*v_1[1]+M31*v_1[2];
        un_1[1] = M21*v_1[0]+M22*v_1[1]+M32*v_1[2];
        un_1[2] = M31*v_1[0]+M32*v_1[1]+M33*v_1[2];

        // un_{,2}
        un_2[0] = M11*v_2[0]+M21*v_2[1]+M31*v_2[2];
        un_2[1] = M21*v_2[0]+M22*v_2[1]+M32*v_2[2];
        un_2[2] = M31*v_2[0]+M32*v_2[1]+M33*v_2[2];

        d1 = amrex::dG::linalg::dot3(un, v_1);
        d2 = amrex::dG::linalg::dot3(un, v_2);

        // un_{,11}
        A11 = d1*M11 + 2.0*un[0]*(M11*v_1[0] + M21*v_1[1] + M31*v_1[2]);
        A21 = d1*M21 + M21*un[0]*v_1[0] + M11*un[1]*v_1[0] + M22*un[0]*v_1[1] + M21*un[1]*v_1[1] + M32*un[0]*v_1[2] + M31*un[1]*v_1[2];
        A31 = d1*M31 + M31*un[0]*v_1[0] + M11*un[2]*v_1[0] + M32*un[0]*v_1[1] + M21*un[2]*v_1[1] + M33*un[0]*v_1[2] + M31*un[2]*v_1[2];
        A22 = d1*M22 + 2.0*un[1]*(M21*v_1[0] + M22*v_1[1] + M32*v_1[2]);
        A32 = d1*M32 + M31*un[1]*v_1[0] + M21*un[2]*v_1[0] + M32*un[1]*v_1[1] + M22*un[2]*v_1[1] + M33*un[1]*v_1[2] + M32*un[2]*v_1[2];
        A33 = d1*M33 + 2.0*un[2]*(M31*v_1[0] + M32*v_1[1] + M33*v_1[2]);

        un_11[0] = M11*v_11[0]+M21*v_11[1]+M31*v_11[2]-tmp*(A11*v_1[0]+A21*v_1[1]+A31*v_1[2]);
        un_11[1] = M21*v_11[0]+M22*v_11[1]+M32*v_11[2]-tmp*(A21*v_1[0]+A22*v_1[1]+A32*v_1[2]);
        un_11[2] = M31*v_11[0]+M32*v_11[1]+M33*v_11[2]-tmp*(A31*v_1[0]+A32*v_1[1]+A33*v_1[2]);

        // un_{,22}
        A11 = d2*M11 + 2.0*un[0]*(M11*v_2[0] + M21*v_2[1] + M31*v_2[2]);
        A21 = d2*M21 + M21*un[0]*v_2[0] + M11*un[1]*v_2[0] + M22*un[0]*v_2[1] + M21*un[1]*v_2[1] + M32*un[0]*v_2[2] + M31*un[1]*v_2[2];
        A31 = d2*M31 + M31*un[0]*v_2[0] + M11*un[2]*v_2[0] + M32*un[0]*v_2[1] + M21*un[2]*v_2[1] + M33*un[0]*v_2[2] + M31*un[2]*v_2[2];
        A22 = d2*M22 + 2.0*un[1]*(M21*v_2[0] + M22*v_2[1] + M32*v_2[2]);
        A32 = d2*M32 + M31*un[1]*v_2[0] + M21*un[2]*v_2[0] + M32*un[1]*v_2[1] + M22*un[2]*v_2[1] + M33*un[1]*v_2[2] + M32*un[2]*v_2[2];
        A33 = d2*M33 + 2.0*un[2]*(M31*v_2[0] + M32*v_2[1] + M33*v_2[2]);

        un_22[0] = M11*v_22[0]+M21*v_22[1]+M31*v_22[2]-tmp*(A11*v_2[0]+A21*v_2[1]+A31*v_2[2]);
        un_22[1] = M21*v_22[0]+M22*v_22[1]+M32*v_22[2]-tmp*(A21*v_2[0]+A22*v_2[1]+A32*v_2[2]);
        un_22[2] = M31*v_22[0]+M32*v_22[1]+M33*v_22[2]-tmp*(A31*v_2[0]+A32*v_2[1]+A33*v_2[2]);

        // un_{,12}
        // Aij same as for un_{,22}

        un_12[0] = M11*v_12[0]+M21*v_12[1]+M31*v_12[2]-tmp*(A11*v_1[0]+A21*v_1[1]+A31*v_1[2]);
        un_12[1] = M21*v_12[0]+M22*v_12[1]+M32*v_12[2]-tmp*(A21*v_1[0]+A22*v_1[1]+A32*v_1[2]);
        un_12[2] = M31*v_12[0]+M32*v_12[1]+M33*v_12[2]-tmp*(A31*v_1[0]+A32*v_1[1]+A33*v_1[2]);
    }

    /**
     * \brief Return the covariant basis vectors.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_covariant_basis(const amrex::Real * x, const amrex::Real x3,
                                        amrex::Real * g1, amrex::Real * g2, amrex::Real * g3) const
    {
        amrex::Real X[3], X_1[3], X_2[3];
        amrex::Real un[3], un_1[3], un_2[3];

        this->shell_map_eval_derivatives(x, X, X_1, X_2);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2);

        g1[0] = X_1[0]+x3*un_1[0]; g2[0] = X_2[0]+x3*un_2[0]; g3[0] = un[0];
        g1[1] = X_1[1]+x3*un_1[1]; g2[1] = X_2[1]+x3*un_2[1]; g3[1] = un[1];
        g1[2] = X_1[2]+x3*un_1[2]; g2[2] = X_2[2]+x3*un_2[2]; g3[2] = un[2];
    }
    
    /**
     * \brief Return the transformation matrix R.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_R(const amrex::Real * x, const amrex::Real x3,
                          amrex::Real * R) const
    {
        amrex::Real X[3], X_1[3], X_2[3];
        amrex::Real un[3], un_1[3], un_2[3];
        amrex::Real dXdx[3*3];

        this->shell_map_eval_derivatives(x, X, X_1, X_2);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2);

        dXdx[0+0*3] = X_1[0]+x3*un_1[0]; dXdx[0+1*3] = X_2[0]+x3*un_2[0]; dXdx[0+2*3] = un[0];
        dXdx[1+0*3] = X_1[1]+x3*un_1[1]; dXdx[1+1*3] = X_2[1]+x3*un_2[1]; dXdx[1+2*3] = un[1];
        dXdx[2+0*3] = X_1[2]+x3*un_1[2]; dXdx[2+1*3] = X_2[2]+x3*un_2[2]; dXdx[2+2*3] = un[2];

        amrex::dG::linalg::dge3it(dXdx, R);
    }
    
    /**
     * \brief Return up to first-order derivatives of the transformation matrix R.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_map_eval_R_derivatives(const amrex::Real * x, const amrex::Real x3,
                                      amrex::Real * R,
                                      amrex::Real * R_1, amrex::Real * R_2, amrex::Real * R_3) const
    {
        amrex::Real X[3], X_1[3], X_2[3], X_11[3], X_22[3], X_12[3];
        amrex::Real un[3], un_1[3], un_2[3], un_11[3], un_22[3], un_12[3];
        amrex::Real dXdx[3*3], dXdx_1[3*3], dXdx_2[3*3], dXdx_3[3*3];
        amrex::Real R11, R12, R13, R21, R22, R23, R31, R32, R33;
        amrex::Real A11, A12, A13, A21, A22, A23, A31, A32, A33;

        this->shell_map_eval_derivatives(x, X, X_1, X_2, X_11, X_22, X_12);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2, un_11, un_22, un_12);

        dXdx[0+0*3] = X_1[0]+x3*un_1[0]; dXdx[0+1*3] = X_2[0]+x3*un_2[0]; dXdx[0+2*3] = un[0];
        dXdx[1+0*3] = X_1[1]+x3*un_1[1]; dXdx[1+1*3] = X_2[1]+x3*un_2[1]; dXdx[1+2*3] = un[1];
        dXdx[2+0*3] = X_1[2]+x3*un_1[2]; dXdx[2+1*3] = X_2[2]+x3*un_2[2]; dXdx[2+2*3] = un[2];

        dXdx_1[0+0*3] = X_11[0]+x3*un_11[0]; dXdx_1[0+1*3] = X_12[0]+x3*un_12[0]; dXdx_1[0+2*3] = un_1[0];
        dXdx_1[1+0*3] = X_11[1]+x3*un_11[1]; dXdx_1[1+1*3] = X_12[1]+x3*un_12[1]; dXdx_1[1+2*3] = un_1[1];
        dXdx_1[2+0*3] = X_11[2]+x3*un_11[2]; dXdx_1[2+1*3] = X_12[2]+x3*un_12[2]; dXdx_1[2+2*3] = un_1[2];

        dXdx_2[0+0*3] = X_12[0]+x3*un_12[0]; dXdx_2[0+1*3] = X_22[0]+x3*un_22[0]; dXdx_2[0+2*3] = un_2[0];
        dXdx_2[1+0*3] = X_12[1]+x3*un_12[1]; dXdx_2[1+1*3] = X_22[1]+x3*un_22[1]; dXdx_2[1+2*3] = un_2[1];
        dXdx_2[2+0*3] = X_12[2]+x3*un_12[2]; dXdx_2[2+1*3] = X_22[2]+x3*un_22[2]; dXdx_2[2+2*3] = un_2[2];

        dXdx_3[0+0*3] = un_1[0]; dXdx_3[0+1*3] = un_2[0]; dXdx_3[0+2*3] = 0.0;
        dXdx_3[1+0*3] = un_1[1]; dXdx_3[1+1*3] = un_2[1]; dXdx_3[1+2*3] = 0.0;
        dXdx_3[2+0*3] = un_1[2]; dXdx_3[2+1*3] = un_2[2]; dXdx_3[2+2*3] = 0.0;

        amrex::dG::linalg::dge3it(dXdx, R);

        R11 = R[0+0*3]; R12 = R[0+1*3]; R13 = R[0+2*3];
        R21 = R[1+0*3]; R22 = R[1+1*3]; R23 = R[1+2*3];
        R31 = R[2+0*3]; R32 = R[2+1*3]; R33 = R[2+2*3];

        // R_1
        A11 = dXdx_1[0+0*3]; A12 = dXdx_1[1+0*3]; A13 = dXdx_1[2+0*3];
        A21 = dXdx_1[0+1*3]; A22 = dXdx_1[1+1*3]; A23 = dXdx_1[2+1*3];
        A31 = dXdx_1[0+2*3]; A32 = dXdx_1[1+2*3]; A33 = dXdx_1[2+2*3];

        R_1[0+0*3] = -(R11*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R21 + (A13*R11 + A23*R12 + A33*R13)*R31);
        R_1[0+1*3] = -(R12*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R22 + (A13*R11 + A23*R12 + A33*R13)*R32);
        R_1[0+2*3] = -(A11*R11*R13 + A21*R12*R13 + A12*R11*R23 + A22*R12*R23 + A13*R11*R33 + A23*R12*R33 + R13*(A31*R13 + A32*R23 + A33*R33));
        R_1[1+0*3] = -(R11*(A11*R21 + A21*R22 + A31*R23) + R21*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R31);
        R_1[1+1*3] = -(A11*R12*R21 + A21*R12*R22 + A31*R12*R23 + R22*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R32);
        R_1[1+2*3] = -(A11*R13*R21 + A21*R13*R22 + R23*(A31*R13 + A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R33);
        R_1[2+0*3] = -(A11*R11*R31 + A12*R21*R31 + A21*R11*R32 + A22*R21*R32 + A31*R11*R33 + A32*R21*R33 + R31*(A13*R31 + A23*R32 + A33*R33));
        R_1[2+1*3] = -(A11*R12*R31 + A12*R22*R31 + R32*(A21*R12 + A22*R22 + A13*R31 + A23*R32) + (A31*R12 + A32*R22 + A33*R32)*R33);
        R_1[2+2*3] = -(A11*R13*R31 + A12*R23*R31 + A21*R13*R32 + A22*R23*R32 + (A31*R13 + A32*R23 + A13*R31 + A23*R32)*R33 + A33*R33*R33);

        // R_2
        A11 = dXdx_2[0+0*3]; A12 = dXdx_2[1+0*3]; A13 = dXdx_2[2+0*3];
        A21 = dXdx_2[0+1*3]; A22 = dXdx_2[1+1*3]; A23 = dXdx_2[2+1*3];
        A31 = dXdx_2[0+2*3]; A32 = dXdx_2[1+2*3]; A33 = dXdx_2[2+2*3];

        R_2[0+0*3] = -(R11*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R21 + (A13*R11 + A23*R12 + A33*R13)*R31);
        R_2[0+1*3] = -(R12*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R22 + (A13*R11 + A23*R12 + A33*R13)*R32);
        R_2[0+2*3] = -(A11*R11*R13 + A21*R12*R13 + A12*R11*R23 + A22*R12*R23 + A13*R11*R33 + A23*R12*R33 + R13*(A31*R13 + A32*R23 + A33*R33));
        R_2[1+0*3] = -(R11*(A11*R21 + A21*R22 + A31*R23) + R21*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R31);
        R_2[1+1*3] = -(A11*R12*R21 + A21*R12*R22 + A31*R12*R23 + R22*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R32);
        R_2[1+2*3] = -(A11*R13*R21 + A21*R13*R22 + R23*(A31*R13 + A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R33);
        R_2[2+0*3] = -(A11*R11*R31 + A12*R21*R31 + A21*R11*R32 + A22*R21*R32 + A31*R11*R33 + A32*R21*R33 + R31*(A13*R31 + A23*R32 + A33*R33));
        R_2[2+1*3] = -(A11*R12*R31 + A12*R22*R31 + R32*(A21*R12 + A22*R22 + A13*R31 + A23*R32) + (A31*R12 + A32*R22 + A33*R32)*R33);
        R_2[2+2*3] = -(A11*R13*R31 + A12*R23*R31 + A21*R13*R32 + A22*R23*R32 + (A31*R13 + A32*R23 + A13*R31 + A23*R32)*R33 + A33*R33*R33);

        // R_3
        A11 = dXdx_3[0+0*3]; A12 = dXdx_3[1+0*3]; A13 = dXdx_3[2+0*3];
        A21 = dXdx_3[0+1*3]; A22 = dXdx_3[1+1*3]; A23 = dXdx_3[2+1*3];
        A31 = dXdx_3[0+2*3]; A32 = dXdx_3[1+2*3]; A33 = dXdx_3[2+2*3];

        R_3[0+0*3] = -(R11*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R21 + (A13*R11 + A23*R12 + A33*R13)*R31);
        R_3[0+1*3] = -(R12*(A11*R11 + A21*R12 + A31*R13) + (A12*R11 + A22*R12 + A32*R13)*R22 + (A13*R11 + A23*R12 + A33*R13)*R32);
        R_3[0+2*3] = -(A11*R11*R13 + A21*R12*R13 + A12*R11*R23 + A22*R12*R23 + A13*R11*R33 + A23*R12*R33 + R13*(A31*R13 + A32*R23 + A33*R33));
        R_3[1+0*3] = -(R11*(A11*R21 + A21*R22 + A31*R23) + R21*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R31);
        R_3[1+1*3] = -(A11*R12*R21 + A21*R12*R22 + A31*R12*R23 + R22*(A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R32);
        R_3[1+2*3] = -(A11*R13*R21 + A21*R13*R22 + R23*(A31*R13 + A12*R21 + A22*R22 + A32*R23) + (A13*R21 + A23*R22 + A33*R23)*R33);
        R_3[2+0*3] = -(A11*R11*R31 + A12*R21*R31 + A21*R11*R32 + A22*R21*R32 + A31*R11*R33 + A32*R21*R33 + R31*(A13*R31 + A23*R32 + A33*R33));
        R_3[2+1*3] = -(A11*R12*R31 + A12*R22*R31 + R32*(A21*R12 + A22*R22 + A13*R31 + A23*R32) + (A31*R12 + A32*R22 + A33*R32)*R33);
        R_3[2+2*3] = -(A11*R13*R31 + A12*R23*R31 + A21*R13*R32 + A22*R23*R32 + (A31*R13 + A32*R23 + A13*R31 + A23*R32)*R33 + A33*R33*R33);
    }

    /**
     * \brief Return the Jacobian for volume integral evaluation.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real shell_map_eval_volume_Jacobian(const amrex::Real * x, const amrex::Real x3) const
    {
        amrex::Real X[3], X_1[3], X_2[3];
        amrex::Real un[3], un_1[3], un_2[3];
        amrex::Real dXdx[3*3], G[3*3];
        amrex::Real J;

        this->shell_map_eval_derivatives(x, X, X_1, X_2);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2);

        dXdx[0+0*3] = X_1[0]+x3*un_1[0]; dXdx[0+1*3] = X_2[0]+x3*un_2[0]; dXdx[0+2*3] = un[0];
        dXdx[1+0*3] = X_1[1]+x3*un_1[1]; dXdx[1+1*3] = X_2[1]+x3*un_2[1]; dXdx[1+2*3] = un[1];
        dXdx[2+0*3] = X_1[2]+x3*un_1[2]; dXdx[2+1*3] = X_2[2]+x3*un_2[2]; dXdx[2+2*3] = un[2];

        amrex::dG::linalg::dgemm('T', 'N', 3, 3, 3, 1.0, dXdx, 3, dXdx, 3, 0.0, G, 3);

        J = std::sqrt(amrex::dG::linalg::det3(G));

        return J;
    }

    /**
     * \brief Return the Jacobian for surface integral evaluation.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real shell_map_eval_surface_Jacobian(const amrex::Real * x, const amrex::Real x3, const int dir) const
    {
        amrex::Real X[3], X_1[3], X_2[3];
        amrex::Real un[3], un_1[3], un_2[3];
        amrex::Real s1, s2, s3;
        amrex::Real dXdx[3*3], G[3*3], dxdX[3*3];
        amrex::Real gu11, gu12, gu13, gu22, gu23, gu33;
        amrex::Real J;

        this->shell_map_eval_derivatives(x, X, X_1, X_2);
        this->shell_map_eval_unit_normal_derivatives(x, un, un_1, un_2);

        dXdx[0+0*3] = X_1[0]+x3*un_1[0]; dXdx[0+1*3] = X_2[0]+x3*un_2[0]; dXdx[0+2*3] = un[0];
        dXdx[1+0*3] = X_1[1]+x3*un_1[1]; dXdx[1+1*3] = X_2[1]+x3*un_2[1]; dXdx[1+2*3] = un[1];
        dXdx[2+0*3] = X_1[2]+x3*un_1[2]; dXdx[2+1*3] = X_2[2]+x3*un_2[2]; dXdx[2+2*3] = un[2];

        amrex::dG::linalg::dgemm('T', 'N', 3, 3, 3, 1.0, dXdx, 3, dXdx, 3, 0.0, G, 3);
        amrex::dG::linalg::dge3it(dXdx, dxdX);

        gu11 = amrex::dG::linalg::dot3(&dxdX[0], &dxdX[0]);
        gu12 = amrex::dG::linalg::dot3(&dxdX[0], &dxdX[3]);
        gu13 = amrex::dG::linalg::dot3(&dxdX[0], &dxdX[6]);
        gu22 = amrex::dG::linalg::dot3(&dxdX[3], &dxdX[3]);
        gu23 = amrex::dG::linalg::dot3(&dxdX[3], &dxdX[6]);
        gu33 = amrex::dG::linalg::dot3(&dxdX[6], &dxdX[6]);

        // We use J as a temporary variable
        J = 1.0/std::sqrt(amrex::dG::linalg::dot3(&dXdx[3*dir], &dXdx[3*dir]));
        s1 = J*dXdx[3*dir];
        s2 = J*dXdx[3*dir+1];
        s3 = J*dXdx[3*dir+2];

        J = std::sqrt(amrex::dG::linalg::det3(G));
        J *= std::sqrt(s1*(gu11*s1+gu12*s2+gu13*s3)+s2*(gu12*s1+gu22*s2+gu23*s3)+s3*(gu13*s1+gu23*s2+gu33*s3));

        /* DEBUG
        amrex::Print() << "x: "; amrex::dG::io::print_reals(AMREX_SPACEDIM, x); amrex::Print() << std::endl;
        amrex::Print() << "dXdx: " << std::endl;
        amrex::dG::io::print_real_array_2d(3, 3, dXdx);
        amrex::Print() << "dxdX: " << std::endl;
        amrex::dG::io::print_real_array_2d(3, 3, dxdX);
        */

        return J;
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
        return 1;
    }
    /**
     * \brief Return the number of unknown fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_unknown_fields() const
    {
        const int NU = this->theory[0]+this->theory[1]+this->theory[2]+3;
        return NU;
    }
    /**
     * \brief Return the unknown fields index bounds for the dom-th domain.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void domain_unknown_fields_index_bounds(const int /*dom*/, int & u_lo, int & u_hi) const
    {
        u_lo = 0;
        u_hi = this->theory[0]+this->theory[1]+this->theory[2]+3;
    }
    /**
     * \brief Return the parent domain of the u-th unknown field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int unknown_field_parent_domain(const int /*u*/) const
    {
        return 0;
    }
    /**
     * \brief Return the domain that is neighbor with domain dom.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_nbr_dom(const int /*dom*/) const
    {
        return -1;
    }

    /**
     * \brief Fill U with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_exact_solution(const amrex::Real t, const amrex::Real * x, amrex::Real * U) const
    {
        const int m1 = 1;
        const amrex::Real L1 = 1.0;
        const int m2 = 1;
        const amrex::Real L2 = 1.0;
        const amrex::Real c1 = std::cos((m1*M_PI/L1)*x[0]);
        const amrex::Real s1 = std::sin((m1*M_PI/L1)*x[0]);
        const amrex::Real c2 = std::cos((m2*M_PI/L2)*x[1]);
        const amrex::Real s2 = std::sin((m2*M_PI/L2)*x[1]);
        const amrex::Real fxi[3] = {c1*s2,
                                    s1*c2,
                                    s1*s2};

        for (int r = 0; r < SHELL_NU; ++r)
        {
            U[r] = this->Us[r];
            for (int k = 0; k < SHELL_NU; ++k)
            {
                U[r] += this->vA[r+k*SHELL_NU]*this->cvA[k]*std::cos(this->wA[k]*t);
            }

            U[r] *= fxi[r%3];
        }
    }

    /**
     * \brief Fill U, U_1, U_2, with exact solution at (t, x).
     * 
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U: pointer to memory that will be filled with the value of the conserved variables
     *                at (t, x).
     * \param[out] U_1, U_2: pointer to memory that will be filled with the value of the derivatives of
     *                       the conserved variables at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_exact_solution(const amrex::Real t, const amrex::Real * x, amrex::Real * U, amrex::Real * U_1, amrex::Real * U_2) const
    {
        const int m1 = 1;
        const amrex::Real L1 = 1.0;
        const int m2 = 1;
        const amrex::Real L2 = 1.0;
        const amrex::Real c1 = std::cos((m1*M_PI/L1)*x[0]);
        const amrex::Real s1 = std::sin((m1*M_PI/L1)*x[0]);
        const amrex::Real c2 = std::cos((m2*M_PI/L2)*x[1]);
        const amrex::Real s2 = std::sin((m2*M_PI/L2)*x[1]);
        const amrex::Real fxi[3] = {c1*s2,
                                    s1*c2,
                                    s1*s2};
        const amrex::Real fxi_1[3] = {-(m1*M_PI/L1)*s1*s2,
                                       (m1*M_PI/L1)*c1*c2,
                                       (m1*M_PI/L1)*c1*s2};
        const amrex::Real fxi_2[3] = { (m2*M_PI/L2)*c1*c2,
                                      -(m2*M_PI/L2)*s1*s2,
                                       (m2*M_PI/L2)*s1*c2};

        for (int r = 0; r < SHELL_NU; ++r)
        {
            U[r] = this->Us[r];
            U_1[r] = this->Us[r];
            U_2[r] = this->Us[r];
            for (int k = 0; k < SHELL_NU; ++k)
            {
                U[r] += this->vA[r+k*SHELL_NU]*this->cvA[k]*std::cos(this->wA[k]*t);
                U_1[r] += this->vA[r+k*SHELL_NU]*this->cvA[k]*std::cos(this->wA[k]*t);
                U_2[r] += this->vA[r+k*SHELL_NU]*this->cvA[k]*std::cos(this->wA[k]*t);
            }

            U[r] *= fxi[r%3];
            U_1[r] *= fxi_1[r%3];
            U_2[r] *= fxi_2[r%3];
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
    void eval_ICs(const amrex::Real * /*x*/, amrex::Real * U) const
    {
        for (int u = 0; u < SHELL_NU; ++u)
        {
            U[u] = 0.0;
        }
    }

    /**
     * \brief Fill U_t with initial conditions of the time-derivative of U at x.
     * 
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] U_t: pointer to memory that will be filled with the time-derivative of the conserved
     *                  variables at x.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_ICs_der(const amrex::Real * /*x*/, amrex::Real * U_t) const
    {
        for (int u = 0; u < SHELL_NU; ++u)
        {
            U_t[u] = 0.0;
        }
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    /**
     * \brief Eval the error at (t, x) to be used for the L_inf norm error.
     *
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the value of the conserved variables at (t, x).
     * \param[out] err: error at (t, x).
     * \param[out] err_norm: error normalization at (t, x).
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_error_L_inf(const int /*dom*/,
                          const amrex::Real t, const amrex::Real * x,
                          const amrex::Real * U,
                          amrex::Real & err, amrex::Real & err_norm) const
    {
        amrex::Real U_exact[SHELL_NU];

        this->eval_exact_solution(t, x, U_exact);

        err = std::abs(U[0]-U_exact[0]);
        err_norm = std::abs(U_exact[0]);
        for (int u = 1; u < SHELL_NU; ++u)
        {
            err = amrex::max(err, std::abs(U[u]-U_exact[u]));
            err_norm = amrex::max(err_norm, std::abs(U_exact[u]));
        }
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
     * \param[in] U: pointer to memory containing the value of the conserved variables at (t, x).
     *
     * \return: time increment based on solution at (t, x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eval_dt(const int /*dom*/,
                        const amrex::Real * dx,
                        const amrex::Real /*t*/, const amrex::Real * /*x*/,
                        const amrex::Real * /*U*/) const
    {
        const amrex::Real ws = this->max_wave_speed;
        const amrex::Real tau = this->thickness;
        const amrex::Real h = amrex::min(dx[0], dx[1]);
        
        const amrex::Real dt = h/(ws*std::sqrt(this->mu0)*tau);

        return dt;
    }
    // ================================================================


    // GOVERNING EQUATIONS ============================================
    /**
     * \brief Eval the thickness functions for the considered shell theory.
     *
     * \param[in] x3: thickness location.
     * \param[out] Z: pointer to memory that will contain the thickness functions.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_thickness_functions(const amrex::Real x3, amrex::Real * Z, amrex::Real * dZ) const
    {
#if (SHELL_THEORY == SHELL_THEORY_2D)
        Z[0] = 1.0;
        dZ[0] = 0.0;
#elif (SHELL_THEORY == SHELL_THEORY_FSDT)
        Z[0] = 1.0;
        Z[1] = x3;
        dZ[0] = 0.0;
        dZ[1] = 1.0;
#else
amrex::Print() << "shell_theory_eval_thickness_functions" << std::endl;
exit(-1);
#endif
    }

    /**
     * \brief Eval the RF operator for the considered shell theory.
     *
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] x3: thickness location.
     * \param[out] RF: pointers to memory that will contain the RF operator.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_RF_operator(const amrex::Real * x, const amrex::Real x3, amrex::Real * RF) const
    {
        amrex::Real Z[SHELL_NU], dZ[SHELL_NU];
        amrex::Real R[3*3];
        amrex::Real R11, R12, R13, R21, R22, R23, R31, R32, R33;

        this->shell_theory_eval_thickness_functions(x3, Z, dZ);
        this->shell_map_eval_R(x, x3, R);

        R11 = R[0+0*3]; R12 = R[0+1*3]; R13 = R[0+2*3];
        R21 = R[1+0*3]; R22 = R[1+1*3]; R23 = R[1+2*3];
        R31 = R[2+0*3]; R32 = R[2+1*3]; R33 = R[2+2*3];

#if (SHELL_THEORY == SHELL_THEORY_2D)
        const amrex::Real Z0 = Z[0];

        RF[0+0*3] = R11*Z0; RF[0+1*3] = R12*Z0;
        RF[1+0*3] = R21*Z0; RF[1+1*3] = R22*Z0;
        RF[2+0*3] = R31*Z0; RF[2+1*3] = R32*Z0;

#elif (SHELL_THEORY == SHELL_THEORY_FSDT)
        const amrex::Real Z0 = Z[0];
        const amrex::Real Z1 = Z[1];

        RF[0+0*3] = R11*Z0; RF[0+1*3] = R12*Z0; RF[0+2*3] = R13*Z0; RF[0+3*3] = R11*Z1; RF[0+4*3] = R12*Z1;
        RF[1+0*3] = R21*Z0; RF[1+1*3] = R22*Z0; RF[1+2*3] = R23*Z0; RF[1+3*3] = R21*Z1; RF[1+4*3] = R22*Z1;
        RF[2+0*3] = R31*Z0; RF[2+1*3] = R32*Z0; RF[2+2*3] = R33*Z0; RF[2+3*3] = R31*Z1; RF[2+4*3] = R32*Z1;
        
#else
amrex::Print() << "shell_theory_eval_RF_operator" << std::endl;
exit(-1);
#endif
        /* DEBUG
        amrex::Print() << "R: " << std::endl;
        amrex::dG::io::print_real_array_2d(3, 3, R);
        amrex::Print() << "RF: " << std::endl;
        amrex::dG::io::print_real_array_2d(3, SHELL_NU, RF);
        exit(-1);
        */
    }
    
    /**
     * \brief Eval the Ji operators for the considered shell theory.
     *
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] x3: thickness location.
     * \param[out] J1, J2, J3: pointers to memory that will contain the Ji operators.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_Ji_operators(const amrex::Real * x, const amrex::Real x3, amrex::Real * J1, amrex::Real * J2, amrex::Real * J3) const
    {
        amrex::Real Z[SHELL_NU], dZ[SHELL_NU];
        amrex::Real R[3*3], R_1[3*3], R_2[3*3], R_3[3*3];
        amrex::Real R11, R12, R13, R21, R22, R23, R31, R32, R33;
        amrex::Real R11_1, R12_1, R13_1, R21_1, R22_1, R23_1, R31_1, R32_1, R33_1;
        amrex::Real R11_2, R12_2, R13_2, R21_2, R22_2, R23_2, R31_2, R32_2, R33_2;
        amrex::Real R11_3, R12_3, R13_3, R21_3, R22_3, R23_3, R31_3, R32_3, R33_3;

        this->shell_theory_eval_thickness_functions(x3, Z, dZ);
        this->shell_map_eval_R_derivatives(x, x3, R, R_1, R_2, R_3);

        R11 = R[0+0*3]; R12 = R[0+1*3]; R13 = R[0+2*3];
        R21 = R[1+0*3]; R22 = R[1+1*3]; R23 = R[1+2*3];
        R31 = R[2+0*3]; R32 = R[2+1*3]; R33 = R[2+2*3];

        R11_1 = R_1[0+0*3]; R12_1 = R_1[0+1*3]; R13_1 = R_1[0+2*3];
        R21_1 = R_1[1+0*3]; R22_1 = R_1[1+1*3]; R23_1 = R_1[1+2*3];
        R31_1 = R_1[2+0*3]; R32_1 = R_1[2+1*3]; R33_1 = R_1[2+2*3];

        R11_2 = R_2[0+0*3]; R12_2 = R_2[0+1*3]; R13_2 = R_2[0+2*3];
        R21_2 = R_2[1+0*3]; R22_2 = R_2[1+1*3]; R23_2 = R_2[1+2*3];
        R31_2 = R_2[2+0*3]; R32_2 = R_2[2+1*3]; R33_2 = R_2[2+2*3];

        R11_3 = R_3[0+0*3]; R12_3 = R_3[0+1*3]; R13_3 = R_3[0+2*3];
        R21_3 = R_3[1+0*3]; R22_3 = R_3[1+1*3]; R23_3 = R_3[1+2*3];
        R31_3 = R_3[2+0*3]; R32_3 = R_3[2+1*3]; R33_3 = R_3[2+2*3];

#if (SHELL_THEORY == SHELL_THEORY_2D)
        const amrex::Real Z0 = Z[0];
        const amrex::Real dZ0 = dZ[0];

        // J1
        J1[0+0*6] =     R11*R11*Z0; J1[0+1*6] =             R11*R12*Z0;
        J1[1+0*6] =     R21*R21*Z0; J1[1+1*6] =             R21*R22*Z0;
        J1[2+0*6] =     R31*R31*Z0; J1[2+1*6] =             R31*R32*Z0;
        J1[3+0*6] = 2.0*R21*R31*Z0; J1[3+1*6] = (R22*R31 + R21*R32)*Z0;
        J1[4+0*6] = 2.0*R11*R31*Z0; J1[4+1*6] = (R12*R31 + R11*R32)*Z0;
        J1[5+0*6] = 2.0*R11*R21*Z0; J1[5+1*6] = (R12*R21 + R11*R22)*Z0;

        // J2
        J2[0+0*6] =           R11*R12*Z0; J2[0+1*6] =     R12*R12*Z0;
        J2[1+0*6] =           R21*R22*Z0; J2[1+1*6] =     R22*R22*Z0;
        J2[2+0*6] =           R31*R32*Z0; J2[2+1*6] =     R32*R32*Z0;
        J2[3+0*6] = (R22*R31+R21*R32)*Z0; J2[3+1*6] = 2.0*R22*R32*Z0;
        J2[4+0*6] = (R12*R31+R11*R32)*Z0; J2[4+1*6] = 2.0*R12*R32*Z0;
        J2[5+0*6] = (R12*R21+R11*R22)*Z0; J2[5+1*6] = 2.0*R12*R22*Z0;

        // J3
        J3[0+0*6] = 0.0;     J3[0+1*6] =     0.0;
        J3[1+0*6] = 0.0;     J3[1+1*6] =     0.0;
        J3[2+0*6] = dZ0*R31; J3[2+1*6] = dZ0*R32;
        J3[3+0*6] = dZ0*R21; J3[3+1*6] = dZ0*R22;
        J3[4+0*6] = dZ0*R11; J3[4+1*6] = dZ0*R12;
        J3[5+0*6] = 0.0;     J3[5+1*6] =     0.0;

        J3[0+0*6] += (R11*R11_1+R11_2*R12+R11_3*R13)*Z0;                               J3[0+1*6] += (R11*R12_1+R12*R12_2+R12_3*R13)*Z0;
        J3[1+0*6] += (R21*R21_1+R21_2*R22+R21_3*R23)*Z0;                               J3[1+1*6] += (R21*R22_1+R22*R22_2+R22_3*R23)*Z0; 
        J3[2+0*6] += (R31*R31_1+R31_2*R32+R31_3*R33)*Z0;                               J3[2+1*6] += (R31*R32_1+R32*R32_2+R32_3*R33)*Z0;
        J3[3+0*6] += (R21_1*R31+R21*R31_1+R22*R31_2+R23*R31_3+R21_2*R32+R21_3*R33)*Z0; J3[3+1*6] += (R22_1*R31+R22_2*R32+R21*R32_1+R22*R32_2+R23*R32_3+R22_3*R33)*Z0;
        J3[4+0*6] += (R11_1*R31+R11*R31_1+R12*R31_2+R13*R31_3+R11_2*R32+R11_3*R33)*Z0; J3[4+1*6] += (R12_1*R31+R12_2*R32+R11*R32_1+R12*R32_2+R13*R32_3+R12_3*R33)*Z0;
        J3[5+0*6] += (R11_1*R21+R11*R21_1+R12*R21_2+R13*R21_3+R11_2*R22+R11_3*R23)*Z0; J3[5+1*6] += (R12_1*R21+R12_2*R22+R11*R22_1+R12*R22_2+R13*R22_3+R12_3*R23)*Z0;

#elif (SHELL_THEORY == SHELL_THEORY_FSDT)
        const amrex::Real Z0 = Z[0];
        const amrex::Real Z1 = Z[1];
        const amrex::Real dZ0 = dZ[0];
        const amrex::Real dZ1 = dZ[1];

        // J1
        J1[0+0*6] =     R11*R11*Z0;        J1[0+1*6] =           R11*R12*Z0;  J1[0+2*6] =           R11*R13*Z0; J1[0+3*6] =     R11*R11*Z1;       J1[0+4*6] =           R11*R12*Z1;
        J1[1+0*6] =     R21*R21*Z0;        J1[1+1*6] =           R21*R22*Z0;  J1[1+2*6] =           R21*R23*Z0; J1[1+3*6] =     R21*R21*Z1;       J1[1+4*6] =           R21*R22*Z1;
        J1[2+0*6] =     R31*R31*Z0;        J1[2+1*6] =           R31*R32*Z0;  J1[2+2*6] =           R31*R33*Z0; J1[2+3*6] =     R31*R31*Z1;       J1[2+4*6] =           R31*R32*Z1;
        J1[3+0*6] = 2.0*R21*R31*Z0;        J1[3+1*6] = (R22*R31+R21*R32)*Z0;  J1[3+2*6] = (R23*R31+R21*R33)*Z0; J1[3+3*6] = 2.0*R21*R31*Z1;       J1[3+4*6] = (R22*R31+R21*R32)*Z1;
        J1[4+0*6] = 2.0*R11*R31*Z0;        J1[4+1*6] = (R12*R31+R11*R32)*Z0;  J1[4+2*6] = (R13*R31+R11*R33)*Z0; J1[4+3*6] = 2.0*R11*R31*Z1;       J1[4+4*6] = (R12*R31+R11*R32)*Z1;
        J1[5+0*6] = 2.0*R11*R21*Z0;        J1[5+1*6] = (R12*R21+R11*R22)*Z0;  J1[5+2*6] = (R13*R21+R11*R23)*Z0; J1[5+3*6] = 2.0*R11*R21*Z1;       J1[5+4*6] = (R12*R21+R11*R22)*Z1;

        // J2
        J2[0+0*6] =           R11*R12*Z0;  J2[0+1*6] =     R12*R12*Z0;        J2[0+2*6] =           R12*R13*Z0; J2[0+3*6] =           R11*R12*Z1; J2[0+4*6] =     R12*R12*Z1;
        J2[1+0*6] =           R21*R22*Z0;  J2[1+1*6] =     R22*R22*Z0;        J2[1+2*6] =           R22*R23*Z0; J2[1+3*6] =           R21*R22*Z1; J2[1+4*6] =     R22*R22*Z1;
        J2[2+0*6] =           R31*R32*Z0;  J2[2+1*6] =     R32*R32*Z0;        J2[2+2*6] =           R32*R33*Z0; J2[2+3*6] =           R31*R32*Z1; J2[2+4*6] =     R32*R32*Z1;
        J2[3+0*6] = (R22*R31+R21*R32)*Z0;  J2[3+1*6] = 2.0*R22*R32*Z0;        J2[3+2*6] = (R23*R32+R22*R33)*Z0; J2[3+3*6] = (R22*R31+R21*R32)*Z1; J2[3+4*6] = 2.0*R22*R32*Z1;
        J2[4+0*6] = (R12*R31+R11*R32)*Z0;  J2[4+1*6] = 2.0*R12*R32*Z0;        J2[4+2*6] = (R13*R32+R12*R33)*Z0; J2[4+3*6] = (R12*R31+R11*R32)*Z1; J2[4+4*6] = 2.0*R12*R32*Z1;
        J2[5+0*6] = (R12*R21+R11*R22)*Z0;  J2[5+1*6] = 2.0*R12*R22*Z0;        J2[5+2*6] = (R13*R22+R12*R23)*Z0; J2[5+3*6] = (R12*R21+R11*R22)*Z1; J2[5+4*6] = 2.0*R12*R22*Z1;

        // J3
        J3[0+0*6] =           dZ0*R11*R13; J3[0+1*6] =           dZ0*R12*R13; J3[0+2*6] =     dZ0*R13*R13;      J3[0+3*6] =           dZ1*R11*R13; J3[0+4*6] =           dZ1*R12*R13;
        J3[1+0*6] =           dZ0*R21*R23; J3[1+1*6] =           dZ0*R22*R23; J3[1+2*6] =     dZ0*R23*R23;      J3[1+3*6] =           dZ1*R21*R23; J3[1+4*6] =           dZ1*R22*R23;
        J3[2+0*6] =           dZ0*R31*R33; J3[2+1*6] =           dZ0*R32*R33; J3[2+2*6] =     dZ0*R33*R33;      J3[2+3*6] =           dZ1*R31*R33; J3[2+4*6] =           dZ1*R32*R33;
        J3[3+0*6] = dZ0*(R23*R31+R21*R33); J3[3+1*6] = dZ0*(R23*R32+R22*R33); J3[3+2*6] = 2.0*dZ0*R23*R33;      J3[3+3*6] = dZ1*(R23*R31+R21*R33); J3[3+4*6] = dZ1*(R23*R32+R22*R33);
        J3[4+0*6] = dZ0*(R13*R31+R11*R33); J3[4+1*6] = dZ0*(R13*R32+R12*R33); J3[4+2*6] = 2.0*dZ0*R13*R33;      J3[4+3*6] = dZ1*(R13*R31+R11*R33); J3[4+4*6] = dZ1*(R13*R32+R12*R33);
        J3[5+0*6] = dZ0*(R13*R21+R11*R23); J3[5+1*6] = dZ0*(R13*R22+R12*R23); J3[5+2*6] = 2.0*dZ0*R13*R23;      J3[5+3*6] = dZ1*(R13*R21+R11*R23); J3[5+4*6] = dZ1*(R13*R22+R12*R23);
        
        J3[0+0*6] +=             R11*R11_1*Z0; J3[0+1*6] +=             R11*R12_1*Z0; J3[0+2*6] +=             R11*R13_1*Z0; J3[0+3*6] +=             R11*R11_1*Z1; J3[0+4*6] +=             R11*R12_1*Z1;
        J3[1+0*6] +=             R21*R21_1*Z0; J3[1+1*6] +=             R21*R22_1*Z0; J3[1+2*6] +=             R21*R23_1*Z0; J3[1+3*6] +=             R21*R21_1*Z1; J3[1+4*6] +=             R21*R22_1*Z1;
        J3[2+0*6] +=             R31*R31_1*Z0; J3[2+1*6] +=             R31*R32_1*Z0; J3[2+2*6] +=             R31*R33_1*Z0; J3[2+3*6] +=             R31*R31_1*Z1; J3[2+4*6] +=             R31*R32_1*Z1;
        J3[3+0*6] += (R21_1*R31+R21*R31_1)*Z0; J3[3+1*6] += (R22_1*R31+R21*R32_1)*Z0; J3[3+2*6] += (R23_1*R31+R21*R33_1)*Z0; J3[3+3*6] += (R21_1*R31+R21*R31_1)*Z1; J3[3+4*6] += (R22_1*R31+R21*R32_1)*Z1;
        J3[4+0*6] += (R11_1*R31+R11*R31_1)*Z0; J3[4+1*6] += (R12_1*R31+R11*R32_1)*Z0; J3[4+2*6] += (R13_1*R31+R11*R33_1)*Z0; J3[4+3*6] += (R11_1*R31+R11*R31_1)*Z1; J3[4+4*6] += (R12_1*R31+R11*R32_1)*Z1;
        J3[5+0*6] += (R11_1*R21+R11*R21_1)*Z0; J3[5+1*6] += (R12_1*R21+R11*R22_1)*Z0; J3[5+2*6] += (R13_1*R21+R11*R23_1)*Z0; J3[5+3*6] += (R11_1*R21+R11*R21_1)*Z1; J3[5+4*6] += (R12_1*R21+R11*R22_1)*Z1;

        J3[0+0*6] +=             R11_2*R12*Z0; J3[0+1*6] +=             R12*R12_2*Z0; J3[0+2*6] +=            R12*R13_2*Z0; J3[0+3*6] +=             R11_2*R12*Z1; J3[0+4*6] +=              R12*R12_2*Z1;
        J3[1+0*6] +=             R21_2*R22*Z0; J3[1+1*6] +=             R22*R22_2*Z0; J3[1+2*6] +=            R22*R23_2*Z0; J3[1+3*6] +=             R21_2*R22*Z1; J3[1+4*6] +=              R22*R22_2*Z1;
        J3[2+0*6] +=             R31_2*R32*Z0; J3[2+1*6] +=             R32*R32_2*Z0; J3[2+2*6] +=            R32*R33_2*Z0; J3[2+3*6] +=             R31_2*R32*Z1; J3[2+4*6] +=              R32*R32_2*Z1;
        J3[3+0*6] += (R22*R31_2+R21_2*R32)*Z0; J3[3+1*6] += (R22_2*R32+R22*R32_2)*Z0; J3[3+2*6] += (R23_2*R32+R22*R33_2)*Z0; J3[3+3*6] += (R22*R31_2+R21_2*R32)*Z1; J3[3+4*6] += (R22_2*R32+R22*R32_2)*Z1;
        J3[4+0*6] += (R12*R31_2+R11_2*R32)*Z0; J3[4+1*6] += (R12_2*R32+R12*R32_2)*Z0; J3[4+2*6] += (R13_2*R32+R12*R33_2)*Z0; J3[4+3*6] += (R12*R31_2+R11_2*R32)*Z1; J3[4+4*6] += (R12_2*R32+R12*R32_2)*Z1;
        J3[5+0*6] += (R12*R21_2+R11_2*R22)*Z0; J3[5+1*6] += (R12_2*R22+R12*R22_2)*Z0; J3[5+2*6] += (R13_2*R22+R12*R23_2)*Z0; J3[5+3*6] += (R12*R21_2+R11_2*R22)*Z1; J3[5+4*6] += (R12_2*R22+R12*R22_2)*Z1;
        
        J3[0+0*6] +=             R11_3*R13*Z0; J3[0+1*6] +=             R12_3*R13*Z0; J3[0+2*6] +=             R13*R13_3*Z0; J3[0+3*6] +=             R11_3*R13*Z1; J3[0+4*6] +=             R12_3*R13*Z1;
        J3[1+0*6] +=             R21_3*R23*Z0; J3[1+1*6] +=             R22_3*R23*Z0; J3[1+2*6] +=             R23*R23_3*Z0; J3[1+3*6] +=             R21_3*R23*Z1; J3[1+4*6] +=             R22_3*R23*Z1;
        J3[2+0*6] +=             R31_3*R33*Z0; J3[2+1*6] +=             R32_3*R33*Z0; J3[2+2*6] +=             R33*R33_3*Z0; J3[2+3*6] +=             R31_3*R33*Z1; J3[2+4*6] +=             R32_3*R33*Z1;
        J3[3+0*6] += (R23*R31_3+R21_3*R33)*Z0; J3[3+1*6] += (R23*R32_3+R22_3*R33)*Z0; J3[3+2*6] += (R23_3*R33+R23*R33_3)*Z0; J3[3+3*6] += (R23*R31_3+R21_3*R33)*Z1; J3[3+4*6] += (R23*R32_3+R22_3*R33)*Z1;
        J3[4+0*6] += (R13*R31_3+R11_3*R33)*Z0; J3[4+1*6] += (R13*R32_3+R12_3*R33)*Z0; J3[4+2*6] += (R13_3*R33+R13*R33_3)*Z0; J3[4+3*6] += (R13*R31_3+R11_3*R33)*Z1; J3[4+4*6] += (R13*R32_3+R12_3*R33)*Z1;
        J3[5+0*6] += (R13*R21_3+R11_3*R23)*Z0; J3[5+1*6] += (R13*R22_3+R12_3*R23)*Z0; J3[5+2*6] += (R13_3*R23+R13*R23_3)*Z0; J3[5+3*6] += (R13*R21_3+R11_3*R23)*Z1; J3[5+4*6] += (R13*R22_3+R12_3*R23)*Z1;
#else
amrex::Print() << "shell_theory_eval_Ji_operators" << std::endl;
exit(-1);
#endif
        /* DEBUG
        amrex::Print() << "R: " << std::endl;
        amrex::dG::io::print_real_array_2d(3, 3, R);
        amrex::Print() << "J1: " << std::endl;
        amrex::dG::io::print_real_array_2d(6, SHELL_NU, J1);
        amrex::Print() << "J2: " << std::endl;
        amrex::dG::io::print_real_array_2d(6, SHELL_NU, J2);
        amrex::Print() << "J3: " << std::endl;
        amrex::dG::io::print_real_array_2d(6, SHELL_NU, J3);
        exit(-1);
        */
    }

    /**
     * \brief Eval the stiffness matrix at a shell location.
     *
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] x3: thickness location.
     * \param[in] local_cSE: pointer to memory containing the stiffness matrix in local reference system.
     * \param[in] th: orientation of the layer.
     * \param[out] cSE: pointer to memory that will contain the stiffness matrix in the global reference system.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_cSE(const amrex::Real * x, const amrex::Real x3, const amrex::Real * local_cSE, const amrex::Real th, amrex::Real * cSE) const
    {
        const amrex::Real cth = std::cos(th);
        const amrex::Real sth = std::sin(th);

        // Note: we are treating the material as orthotropic
        const amrex::Real c11 = local_cSE[C11];
        const amrex::Real c22 = local_cSE[C22];
        const amrex::Real c33 = local_cSE[C33];
        const amrex::Real c23 = local_cSE[C32];
        const amrex::Real c13 = local_cSE[C31];
        const amrex::Real c12 = local_cSE[C21];
        const amrex::Real c44 = local_cSE[C44];
        const amrex::Real c55 = local_cSE[C55];
        const amrex::Real c66 = local_cSE[C66];

        amrex::Real g1[3], g2[3], g3[3];
        amrex::Real r1[3], r2[3], r3[3];
        amrex::Real Rm11, Rm12, Rm13, Rm21, Rm22, Rm23, Rm31, Rm32, Rm33;
        amrex::Real Rs11, Rs12, Rs13, Rs14, Rs15, Rs16;
        amrex::Real Rs21, Rs22, Rs23, Rs24, Rs25, Rs26;
        amrex::Real Rs31, Rs32, Rs33, Rs34, Rs35, Rs36;
        amrex::Real Rs41, Rs42, Rs43, Rs44, Rs45, Rs46;
        amrex::Real Rs51, Rs52, Rs53, Rs54, Rs55, Rs56;
        amrex::Real Rs61, Rs62, Rs63, Rs64, Rs65, Rs66;
        amrex::Real tmp, tmp3[3];

        this->shell_map_eval_covariant_basis(x, x3, g1, g2, g3);

        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(g1, g1));
        r1[0] = g1[0]*tmp;
        r1[1] = g1[1]*tmp;
        r1[2] = g1[2]*tmp;

        amrex::dG::linalg::cross3(g1, g2, tmp3);
        tmp = 1.0/std::sqrt(amrex::dG::linalg::dot3(tmp3, tmp3));
        r3[0] = tmp3[0]*tmp;
        r3[1] = tmp3[1]*tmp;
        r3[2] = tmp3[2]*tmp;

        amrex::dG::linalg::cross3(r3, r1, r2);

        Rm11 = r1[0]*cth+r2[0]*sth; Rm12 = -r1[0]*sth+r2[0]*cth; Rm13 = r3[0];
        Rm21 = r1[1]*cth+r2[1]*sth; Rm22 = -r1[1]*sth+r2[1]*cth; Rm23 = r3[1];
        Rm31 = r1[2]*cth+r2[2]*sth; Rm32 = -r1[2]*sth+r2[2]*cth; Rm33 = r3[2];

        Rs11 = Rm11*Rm11; Rs12 = Rm12*Rm12; Rs13 = Rm13*Rm13; Rs14 =       2.0*Rm12*Rm13; Rs15 =       2.0*Rm11*Rm13; Rs16 =       2.0*Rm11*Rm12;
        Rs21 = Rm21*Rm21; Rs22 = Rm22*Rm22; Rs23 = Rm23*Rm23; Rs24 =       2.0*Rm22*Rm23; Rs25 =       2.0*Rm21*Rm23; Rs26 =       2.0*Rm21*Rm22;
        Rs31 = Rm31*Rm31; Rs32 = Rm32*Rm32; Rs33 = Rm33*Rm33; Rs34 =       2.0*Rm32*Rm33; Rs35 =       2.0*Rm31*Rm33; Rs36 =       2.0*Rm31*Rm32;
        Rs41 = Rm21*Rm31; Rs42 = Rm22*Rm32; Rs43 = Rm23*Rm33; Rs44 = Rm23*Rm32+Rm22*Rm33; Rs45 = Rm23*Rm31+Rm21*Rm33; Rs46 = Rm22*Rm31+Rm21*Rm32;
        Rs51 = Rm11*Rm31; Rs52 = Rm12*Rm32; Rs53 = Rm13*Rm33; Rs54 = Rm13*Rm32+Rm12*Rm33; Rs55 = Rm13*Rm31+Rm11*Rm33; Rs56 = Rm12*Rm31+Rm11*Rm32;
        Rs61 = Rm11*Rm21; Rs62 = Rm12*Rm22; Rs63 = Rm13*Rm23; Rs64 = Rm13*Rm22+Rm12*Rm23; Rs65 = Rm13*Rm21+Rm11*Rm23; Rs66 = Rm12*Rm21+Rm11*Rm22;

        cSE[0+0*6] = c11*Rs11*Rs11 + 2.0*c12*Rs11*Rs12 + c22*Rs12*Rs12 + 2.0*c13*Rs11*Rs13 + 2.0*c23*Rs12*Rs13 + c33*Rs13*Rs13 + c44*Rs14*Rs14 + c55*Rs15*Rs15 + c66*Rs16*Rs16;
        cSE[1+0*6] = c11*Rs11*Rs21 + c12*Rs12*Rs21 + c13*Rs13*Rs21 + c12*Rs11*Rs22 + c22*Rs12*Rs22 + c23*Rs13*Rs22 + c13*Rs11*Rs23 + c23*Rs12*Rs23 + c33*Rs13*Rs23 + c44*Rs14*Rs24 + c55*Rs15*Rs25 + c66*Rs16*Rs26;
        cSE[2+0*6] = c11*Rs11*Rs31 + c12*Rs12*Rs31 + c13*Rs13*Rs31 + c12*Rs11*Rs32 + c22*Rs12*Rs32 + c23*Rs13*Rs32 + c13*Rs11*Rs33 + c23*Rs12*Rs33 + c33*Rs13*Rs33 + c44*Rs14*Rs34 + c55*Rs15*Rs35 + c66*Rs16*Rs36;
        cSE[3+0*6] = c11*Rs11*Rs41 + c12*Rs12*Rs41 + c13*Rs13*Rs41 + c12*Rs11*Rs42 + c22*Rs12*Rs42 + c23*Rs13*Rs42 + c13*Rs11*Rs43 + c23*Rs12*Rs43 + c33*Rs13*Rs43 + c44*Rs14*Rs44 + c55*Rs15*Rs45 + c66*Rs16*Rs46;
        cSE[4+0*6] = c11*Rs11*Rs51 + c12*Rs12*Rs51 + c13*Rs13*Rs51 + c12*Rs11*Rs52 + c22*Rs12*Rs52 + c23*Rs13*Rs52 + c13*Rs11*Rs53 + c23*Rs12*Rs53 + c33*Rs13*Rs53 + c44*Rs14*Rs54 + c55*Rs15*Rs55 + c66*Rs16*Rs56;
        cSE[5+0*6] = c11*Rs11*Rs61 + c12*Rs12*Rs61 + c13*Rs13*Rs61 + c12*Rs11*Rs62 + c22*Rs12*Rs62 + c23*Rs13*Rs62 + c13*Rs11*Rs63 + c23*Rs12*Rs63 + c33*Rs13*Rs63 + c44*Rs14*Rs64 + c55*Rs15*Rs65 + c66*Rs16*Rs66;
         
        cSE[0+1*6] = cSE[1+0*6];
        cSE[1+1*6] = c11*Rs21*Rs21 + 2.0*c12*Rs21*Rs22 + c22*Rs22*Rs22 + 2.0*c13*Rs21*Rs23 + 2.0*c23*Rs22*Rs23 + c33*Rs23*Rs23 + c44*Rs24*Rs24 + c55*Rs25*Rs25 + c66*Rs26*Rs26;
        cSE[2+1*6] = c11*Rs21*Rs31 + c12*Rs22*Rs31 + c13*Rs23*Rs31 + c12*Rs21*Rs32 + c22*Rs22*Rs32 + c23*Rs23*Rs32 + c13*Rs21*Rs33 + c23*Rs22*Rs33 + c33*Rs23*Rs33 + c44*Rs24*Rs34 + c55*Rs25*Rs35 + c66*Rs26*Rs36;
        cSE[3+1*6] = c11*Rs21*Rs41 + c12*Rs22*Rs41 + c13*Rs23*Rs41 + c12*Rs21*Rs42 + c22*Rs22*Rs42 + c23*Rs23*Rs42 + c13*Rs21*Rs43 + c23*Rs22*Rs43 + c33*Rs23*Rs43 + c44*Rs24*Rs44 + c55*Rs25*Rs45 + c66*Rs26*Rs46; 
        cSE[4+1*6] = c11*Rs21*Rs51 + c12*Rs22*Rs51 + c13*Rs23*Rs51 + c12*Rs21*Rs52 + c22*Rs22*Rs52 + c23*Rs23*Rs52 + c13*Rs21*Rs53 + c23*Rs22*Rs53 + c33*Rs23*Rs53 + c44*Rs24*Rs54 + c55*Rs25*Rs55 + c66*Rs26*Rs56;
        cSE[5+1*6] = c11*Rs21*Rs61 + c12*Rs22*Rs61 + c13*Rs23*Rs61 + c12*Rs21*Rs62 + c22*Rs22*Rs62 + c23*Rs23*Rs62 + c13*Rs21*Rs63 + c23*Rs22*Rs63 + c33*Rs23*Rs63 + c44*Rs24*Rs64 + c55*Rs25*Rs65 + c66*Rs26*Rs66;
         
        cSE[0+2*6] = cSE[2+0*6];
        cSE[1+2*6] = cSE[2+1*6];
        cSE[2+2*6] = c11*Rs31*Rs31 + 2.0*c12*Rs31*Rs32 + c22*Rs32*Rs32 + 2.0*c13*Rs31*Rs33 + 2.0*c23*Rs32*Rs33 + c33*Rs33*Rs33 + c44*Rs34*Rs34 + c55*Rs35*Rs35 + c66*Rs36*Rs36;
        cSE[3+2*6] = c11*Rs31*Rs41 + c12*Rs32*Rs41 + c13*Rs33*Rs41 + c12*Rs31*Rs42 + c22*Rs32*Rs42 + c23*Rs33*Rs42 + c13*Rs31*Rs43 + c23*Rs32*Rs43 + c33*Rs33*Rs43 + c44*Rs34*Rs44 + c55*Rs35*Rs45 + c66*Rs36*Rs46;
        cSE[4+2*6] = c11*Rs31*Rs51 + c12*Rs32*Rs51 + c13*Rs33*Rs51 + c12*Rs31*Rs52 + c22*Rs32*Rs52 + c23*Rs33*Rs52 + c13*Rs31*Rs53 + c23*Rs32*Rs53 + c33*Rs33*Rs53 + c44*Rs34*Rs54 + c55*Rs35*Rs55 + c66*Rs36*Rs56;
        cSE[5+2*6] = c11*Rs31*Rs61 + c12*Rs32*Rs61 + c13*Rs33*Rs61 + c12*Rs31*Rs62 + c22*Rs32*Rs62 + c23*Rs33*Rs62 + c13*Rs31*Rs63 + c23*Rs32*Rs63 + c33*Rs33*Rs63 + c44*Rs34*Rs64 + c55*Rs35*Rs65 + c66*Rs36*Rs66;
        
        cSE[0+3*6] = cSE[3+0*6];
        cSE[1+3*6] = cSE[3+1*6];
        cSE[2+3*6] = cSE[3+2*6];
        cSE[3+3*6] = c11*Rs41*Rs41 + 2.0*c12*Rs41*Rs42 + c22*Rs42*Rs42 + 2.0*c13*Rs41*Rs43 + 2.0*c23*Rs42*Rs43 + c33*Rs43*Rs43 + c44*Rs44*Rs44 + c55*Rs45*Rs45 + c66*Rs46*Rs46;
        cSE[4+3*6] = c11*Rs41*Rs51 + c12*Rs42*Rs51 + c13*Rs43*Rs51 + c12*Rs41*Rs52 + c22*Rs42*Rs52 + c23*Rs43*Rs52 + c13*Rs41*Rs53 + c23*Rs42*Rs53 + c33*Rs43*Rs53 + c44*Rs44*Rs54 + c55*Rs45*Rs55 + c66*Rs46*Rs56;
        cSE[5+3*6] = c11*Rs41*Rs61 + c12*Rs42*Rs61 + c13*Rs43*Rs61 + c12*Rs41*Rs62 + c22*Rs42*Rs62 + c23*Rs43*Rs62 + c13*Rs41*Rs63 + c23*Rs42*Rs63 + c33*Rs43*Rs63 + c44*Rs44*Rs64 + c55*Rs45*Rs65 + c66*Rs46*Rs66; 
        
        cSE[0+4*6] = cSE[4+0*6];
        cSE[1+4*6] = cSE[4+1*6];
        cSE[2+4*6] = cSE[4+2*6];
        cSE[3+4*6] = cSE[4+3*6];
        cSE[4+4*6] = c11*Rs51*Rs51 + 2.0*c12*Rs51*Rs52 + c22*Rs52*Rs52 + 2.0*c13*Rs51*Rs53 + 2.0*c23*Rs52*Rs53 + c33*Rs53*Rs53 + c44*Rs54*Rs54 + c55*Rs55*Rs55 + c66*Rs56*Rs56;
        cSE[5+4*6] = c11*Rs51*Rs61 + c12*Rs52*Rs61 + c13*Rs53*Rs61 + c12*Rs51*Rs62 + c22*Rs52*Rs62 + c23*Rs53*Rs62 + c13*Rs51*Rs63 + c23*Rs52*Rs63 + c33*Rs53*Rs63 + c44*Rs54*Rs64 + c55*Rs55*Rs65 + c66*Rs56*Rs66;
        
        cSE[0+5*6] = cSE[5+0*6];
        cSE[1+5*6] = cSE[5+1*6];
        cSE[2+5*6] = cSE[5+2*6];
        cSE[3+5*6] = cSE[5+3*6];
        cSE[4+5*6] = cSE[5+4*6];
        cSE[5+5*6] = c11*Rs61*Rs61 + 2.0*c12*Rs61*Rs62 + c22*Rs62*Rs62 + 2.0*c13*Rs61*Rs63 + 2.0*c23*Rs62*Rs63 + c33*Rs63*Rs63 + c44*Rs64*Rs64 + c55*Rs65*Rs65 + c66*Rs66*Rs66;

        /* DEBUG
        amrex::Print() << "cSE: " << std::endl;
        amrex::dG::io::print_real_array_2d(6, 6, cSE);
        exit(-1);
        */
    }

    /**
     * \brief Eval the generalized mass matrix.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[out] M: pointer to memory that will contain the mass matrix.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_mass_matrix(const amrex::Real * x, amrex::Real * M_) const
    {
        // VARIABLES
        amrex::Real RF[3*SHELL_NU];

        // INITIALIZATION
        std::fill(M_, M_+SHELL_NU*SHELL_NU, 0.0);

        // INTEGRATION OVER THE LAYERS
        for (int l = 0; l < this->n_layers; ++l)
        {
            const amrex::Real x3b = this->layers_x3b[l];
            const amrex::Real tau = this->layers_thickness[l];
            const Material & mat = this->materials[this->layers_material[l]];

            for (int q = 0; q < (QTH+1); ++q)
            {
                const amrex::Real x3 = x3b+this->xGth[q]*tau;
                const amrex::Real g = this->shell_map_eval_volume_Jacobian(x, x3);
                const amrex::Real w = g*this->wGth[q]*tau;

                this->shell_theory_eval_RF_operator(x, x3, RF);

                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 3, mat.density*w, RF, 3, RF, 3, 1.0, M_, SHELL_NU);
            }
        }

        /* DEBUG */
        amrex::Print() << "M: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, M_);
        //exit(-1);
        /**/
    }

    /**
     * \brief Eval the generalized stiffness matrices.
     *
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] Qab, Ra, S: pointer to memory containing the generalized stiffness matrices.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void shell_theory_eval_stiffness_matrices(const amrex::Real * x,
                                              amrex::Real * Q11_, amrex::Real * Q22_, amrex::Real * Q12_,
                                              amrex::Real * R1_, amrex::Real * R2_,
                                              amrex::Real * S_) const
    {
        amrex::Real J1[6*SHELL_NU], J2[6*SHELL_NU], J3[6*SHELL_NU];
        amrex::Real cSE[6*6];
        amrex::Real tmp, tmp6NU[6*SHELL_NU];

        // INITIALIZATION
        std::fill(Q11_, Q11_+SHELL_NU*SHELL_NU, 0.0);
        std::fill(Q22_, Q22_+SHELL_NU*SHELL_NU, 0.0);
        std::fill(Q12_, Q12_+SHELL_NU*SHELL_NU, 0.0);
        std::fill(R1_, R1_+SHELL_NU*SHELL_NU, 0.0);
        std::fill(R2_, R2_+SHELL_NU*SHELL_NU, 0.0);
        std::fill(S_, S_+SHELL_NU*SHELL_NU, 0.0);

        // INTEGRATION OVER THE LAYERS
        tmp = 0.0;
        for (int l = 0; l < this->n_layers; ++l)
        {
            const amrex::Real x3b = this->layers_x3b[l];
            const amrex::Real tau = this->layers_thickness[l];
            const amrex::Real th = this->layers_orientation[l];
            const Material & mat = this->materials[this->layers_material[l]];

            for (int q = 0; q < (QTH+1); ++q)
            {
                const amrex::Real x3 = x3b+this->xGth[q]*tau;
                const amrex::Real g = this->shell_map_eval_volume_Jacobian(x, x3);
                const amrex::Real w = g*this->wGth[q]*tau;

                this->shell_theory_eval_Ji_operators(x, x3, J1, J2, J3);
                this->shell_theory_eval_cSE(x, x3, mat.cSE, th, cSE);

                amrex::dG::linalg::dgemm('N', 'N', 6, SHELL_NU, 6, 1.0, cSE, 6, J1, 6, 0.0, tmp6NU, 6);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J1, 6, tmp6NU, 6, 1.0, Q11_, SHELL_NU);
                amrex::dG::linalg::dgemm('N', 'N', 6, SHELL_NU, 6, 1.0, cSE, 6, J2, 6, 0.0, tmp6NU, 6);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J2, 6, tmp6NU, 6, 1.0, Q22_, SHELL_NU);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J1, 6, tmp6NU, 6, 1.0, Q12_, SHELL_NU);

                amrex::dG::linalg::dgemm('N', 'N', 6, SHELL_NU, 6, 1.0, cSE, 6, J3, 6, 0.0, tmp6NU, 6);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J1, 6, tmp6NU, 6, 1.0, R1_, SHELL_NU);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J2, 6, tmp6NU, 6, 1.0, R2_, SHELL_NU);
                amrex::dG::linalg::dgemm('T', 'N', SHELL_NU, SHELL_NU, 6, w, J3, 6, tmp6NU, 6, 1.0, S_, SHELL_NU);

                /*
                amrex::Print() << "J3: " << std::endl;
                amrex::dG::io::print_real_array_2d(6, SHELL_NU, J3);
                amrex::Print() << "R2_: " << std::endl;
                amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2_);
                amrex::Print() << "S_: " << std::endl;
                amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, S_);
                exit(-1);
                */

                tmp += w;
            }
        }

        /**/
        amrex::Print() << "this->thickness: " << this->thickness << std::endl;
        amrex::Print() << "tmp: " << tmp << std::endl;

        amrex::Print() << "Q11_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11_);
        amrex::Print() << "Q22_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22_);
        amrex::Print() << "Q12_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12_);
        amrex::Print() << "R1_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1_);
        amrex::Print() << "R2_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2_);
        amrex::Print() << "S_: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, S_);
        //exit(-1);
        /**/
    }

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
                const amrex::Real /*t*/, const amrex::Real * /*x*/,
                amrex::Real * M_) const
    {
        
        std::copy(this->M, this->M+SHELL_NU*SHELL_NU, M_);
    }
    
    /**
     * \brief Eval the diffusive/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *            variables.
     * \param[out] D1, D2, D3: pointer to memory that will contain the diffusive terms.
     * \param[out] R: pointer to memory that will contain the reactive term.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_DR(const int /*dom*/,
                const amrex::Real /*t*/, const amrex::Real * x,
                const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * U_2, const amrex::Real * U_3),
                AMREX_D_DECL(amrex::Real * D1, amrex::Real * D2, amrex::Real * D3), amrex::Real * R) const
    {
        // VARIABLES
        amrex::Real RF[3*SHELL_NU];

        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->Q11, SHELL_NU, U_1, 1, 0.0, D1, 1);
        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->Q12, SHELL_NU, U_2, 1, 1.0, D1, 1);
        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->R1, SHELL_NU, U, 1, 1.0, D1, 1);

        amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 1.0, this->Q12, SHELL_NU, U_1, 1, 0.0, D2, 1);
        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->Q22, SHELL_NU, U_2, 1, 1.0, D2, 1);
        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->R2, SHELL_NU, U, 1, 1.0, D2, 1);

        amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 1.0, this->S, SHELL_NU, U, 1, 0.0, R, 1);
        amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 1.0, this->R1, SHELL_NU, U_1, 1, 1.0, R, 1);
        amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 1.0, this->R2, SHELL_NU, U_2, 1, 1.0, R, 1);

        // CONTRIBUTION OF THE EXTERNAL FORCES: TOP SURFACE
        {
            const int m1 = 1;
            const amrex::Real L1 = 1.0;
            const int m2 = 1;
            const amrex::Real L2 = 1.0;
            const amrex::Real q1 = 0.0, q2 = 0.0, q3 = 1.0;
            const amrex::Real tt[3] = {q1*std::cos((m1*M_PI/L1)*x[0])*std::sin((m2*M_PI/L2)*x[1]),
                                       q2*std::sin((m1*M_PI/L1)*x[0])*std::cos((m2*M_PI/L2)*x[1]),
                                       q3*std::sin((m1*M_PI/L1)*x[0])*std::sin((m2*M_PI/L2)*x[1])};

            const int l = this->n_layers-1;
            const amrex::Real x3b = this->layers_x3b[l];
            const amrex::Real tau = this->layers_thickness[l];
            const amrex::Real x3 = x3b+tau;
            const amrex::Real g = this->shell_map_eval_surface_Jacobian(x, x3, 2);

            this->shell_theory_eval_RF_operator(x, x3, RF);

            amrex::dG::linalg::dgemv('T', 3, SHELL_NU, -g, RF, 3, tt, 1, 1.0, R, 1);
        }

        // CONTRIBUTION OF THE EXTERNAL FORCES: BOTTOM SURFACE
        //...

        // CONTRIBUTION OF THE EXTERNAL FORCES: BODY FORCES
        //...

        /* DEBUG
        //if ((std::abs(x[0]-0.25) < 1.0e-5) && (std::abs(x[1]-0.5) < 1.0e-5) && (t > 0.0))
        {
            amrex::Print() << "x: "; amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, x);

            amrex::Print() << "Q11: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11);
            amrex::Print() << "Q22: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22);
            amrex::Print() << "Q12: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12);
            amrex::Print() << "R1: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1);
            amrex::Print() << "R2: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2);
            amrex::Print() << "S: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, S);

            amrex::Print() << "RF: " << std::endl;
            amrex::dG::io::print_real_array_2d(3, SHELL_NU, RF);
            
            amrex::Print() << "U: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
            amrex::Print() << "U_1: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
            amrex::Print() << "U_2: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);
            
            amrex::Print() << "D1: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, D1);
            amrex::Print() << "D2: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, D2);
            amrex::Print() << "R: " << std::endl;
            amrex::dG::io::print_real_array_2d(SHELL_NU, 1, R);
            exit(-1);
        }
        */
    }

    /**
     * \brief Eval the gradients of diffusive/reactive terms for the considered IBVP.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *            variables.
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
                    const amrex::Real /*t*/, const amrex::Real * /*x*/,
                    const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * /*U_1*/, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                    amrex::Real * dD1dU_1, amrex::Real * dD1dU_2, amrex::Real * dD2dU_1, amrex::Real * dD2dU_2, amrex::Real * dD1dU, amrex::Real * dD2dU,
                    amrex::Real * dRdU_1, amrex::Real * dRdU_2, amrex::Real * dRdU) const
    {
        for (int v = 0; v < SHELL_NU; ++v)
        for (int u = 0; u < SHELL_NU; ++u)
        {
            dD1dU_1[u+v*SHELL_NU] = this->Q11[u+v*SHELL_NU];
            dD1dU_2[u+v*SHELL_NU] = this->Q12[u+v*SHELL_NU];
            dD2dU_1[u+v*SHELL_NU] = this->Q12[v+u*SHELL_NU];
            dD2dU_2[u+v*SHELL_NU] = this->Q22[u+v*SHELL_NU];

            dD1dU[u+v*SHELL_NU] = this->R1[u+v*SHELL_NU];
            dD2dU[u+v*SHELL_NU] = this->R2[u+v*SHELL_NU];

            dRdU_1[u+v*SHELL_NU] = this->R1[v+u*SHELL_NU];
            dRdU_2[u+v*SHELL_NU] = this->R2[v+u*SHELL_NU];

            dRdU[u+v*SHELL_NU] = this->S[u+v*SHELL_NU];
        }
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
     * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
     *                 unit normal.
     * \param[in] m_U_1, m_U_2, m_U_3: pointer to memory containing the derivatives of the conserved
     *                                 variables on the negative side of the unit normal.
     * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
     *                 unit normal.
     * \param[in] p_U_1, p_U_2, p_U_3: pointer to memory containing the derivatives of the conserved
     *                                 variables on the positive side of the unit normal.
     * \param[out] m_ND1, m_ND2, m_ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] m_NFn: pointer to memory that will contain the numerical flux.
     * \param[out] p_ND1, p_ND2, p_ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] p_NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_intraphase(const int /*dom*/,
                            const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                            const amrex::Real * /*m_U*/, AMREX_D_DECL(const amrex::Real * /*m_U_1*/, const amrex::Real * /*m_U_2*/, const amrex::Real * /*m_U_3*/),
                            const amrex::Real * /*p_U*/, AMREX_D_DECL(const amrex::Real * /*p_U_1*/, const amrex::Real * /*p_U_2*/, const amrex::Real * /*p_U_3*/),
                            AMREX_D_DECL(amrex::Real * /*m_ND1*/, amrex::Real * /*m_ND2*/, amrex::Real * /*m_ND3*/), amrex::Real * /*m_NFn*/,
                            AMREX_D_DECL(amrex::Real * /*p_ND1*/, amrex::Real * /*p_ND2*/, amrex::Real * /*p_ND3*/), amrex::Real * /*p_NFn*/) const
    {
amrex::Print() << "eval_NF_intraphase" << std::endl;
exit(-1);
    }

    /**
     * \brief Eval the numerical flux at the intraphase boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables on the negative side of the
     *               unit normal.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *                           variables on the negative side of the unit normal.
     * \param[in] Up: pointer to memory containing the conserved variables on the positive side of the
     *                unit normal.
     * \param[in] Up_1, Up_2, Up_3: pointer to memory containing the derivatives of the conserved
     *                              variables on the positive side of the unit normal.
     * \param[out] ND1, ND2, ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_intraphase(const int /*dom*/,
                            const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                            const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * U_2, const amrex::Real * U_3),
                            const amrex::Real * Up, AMREX_D_DECL(const amrex::Real * Up_1, const amrex::Real * Up_2, const amrex::Real * Up_3),
                            AMREX_D_DECL(amrex::Real * ND1, amrex::Real * ND2, amrex::Real * ND3), amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muI;

        // VARIABLES
        amrex::Real dU[SHELL_NU];
        amrex::Real T[SHELL_NU];

        // GENERALIZED DISPLACEMENT JUMP
        for (int u = 0; u < SHELL_NU; ++u)
        {
            dU[u] = U[u]-Up[u];
        }

        if (std::abs(un[0]) > 0.5)
        {
            // GENERALIZED TRACTIONS AVERAGE
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q11, SHELL_NU, U_1, 1, 0.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q12, SHELL_NU, U_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->R1, SHELL_NU, U, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q11, SHELL_NU, Up_1, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q12, SHELL_NU, Up_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->R1, SHELL_NU, Up, 1, 1.0, T, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                T[u] *= 0.5;
            }

            // -
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 0.5*un[0], this->R1, SHELL_NU, dU, 1, 0.0, NFn, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                NFn[u] += -mu*dU[u]+T[u];
            }

            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 0.5*un[0], this->Q11, SHELL_NU, dU, 1, 0.0, ND1, 1);
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 0.5*un[0], this->Q12, SHELL_NU, dU, 1, 0.0, ND2, 1);
        }
        else
        {
            // GENERALIZED TRACTIONS AVERAGE
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[1], this->Q12, SHELL_NU, U_1, 1, 0.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->Q22, SHELL_NU, U_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->R2, SHELL_NU, U, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[1], this->Q12, SHELL_NU, Up_1, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->Q22, SHELL_NU, Up_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->R2, SHELL_NU, Up, 1, 1.0, T, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                T[u] *= 0.5;
            }

            // -
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, 0.5*un[1], this->R2, SHELL_NU, dU, 1, 0.0, NFn, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                NFn[u] += -mu*dU[u]+T[u];
            }

            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 0.5*un[1], this->Q12, SHELL_NU, dU, 1, 0.0, ND1, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, 0.5*un[1], this->Q22, SHELL_NU, dU, 1, 0.0, ND2, 1);
        }

        /* DEBUG
        amrex::Print() << "Q11: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11);
        amrex::Print() << "Q22: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22);
        amrex::Print() << "Q12: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12);
        amrex::Print() << "R1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1);
        amrex::Print() << "R2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2);
        amrex::Print() << "S: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, S);

        amrex::Print() << "U: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
        amrex::Print() << "U_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
        amrex::Print() << "U_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);

        amrex::Print() << "Up: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up);
        amrex::Print() << "Up_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up_1);
        amrex::Print() << "Up_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up_2);
        
        amrex::Print() << "T: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, T);
        exit(-1);
        */
    }
    
    /**
     * \brief Eval the gradient of the numerical flux at the intraphase boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables on the negative side of the
     *               unit normal.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *                           variables on the negative side of the unit normal.
     * \param[in] Up: pointer to memory containing the conserved variables on the positive side of the
     *                unit normal.
     * \param[in] Up_1, Up_2, Up_3: pointer to memory containing the derivatives of the conserved
     *                              variables on the positive side of the unit normal.
     * \param[out] dNDidU: pointer to memory that will contain the derivative of the numerical flux
     *                     NDi with respect to m_U.
     * \param[out] dNDidUp: pointer to memory that will contain the derivative of the numerical flux
     *                      NDi with respect to p_U.
     * \param[out] dNFndU_j: pointer to memory that will contain the derivative of the numerical flux
     *                       NFn with respect to U_j.
     * \param[out] dNFndU: pointer to memory that will contain the derivative of the numerical flux
     *                     NFn with respect to U.
     * \param[out] dNFndUp_j: pointer to memory that will contain the derivative of the numerical flux
     *                        NFn with respect to Up_j.
     * \param[out] dNFndUp: pointer to memory that will contain the derivative of the numerical flux
     *                      NFn with respect to Up.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_dNFdU_intraphase(const int /*dom*/,
                               const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                               const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * /*U_1*/, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                               const amrex::Real * /*Up*/, AMREX_D_DECL(const amrex::Real * /*Up_1*/, const amrex::Real * /*Up_2*/, const amrex::Real * /*Up_3*/),
                               AMREX_D_DECL(amrex::Real * dND1dU, amrex::Real * dND2dU, amrex::Real * dND3dU),
                               AMREX_D_DECL(amrex::Real * dND1dUp, amrex::Real * dND2dUp, amrex::Real * dND3dUp),
                               AMREX_D_DECL(amrex::Real * dNFndU_1, amrex::Real * dNFndU_2, amrex::Real * dNFndU_3),
                               amrex::Real * dNFndU,
                               AMREX_D_DECL(amrex::Real * dNFndUp_1, amrex::Real * dNFndUp_2, amrex::Real * dNFndUp_3),
                               amrex::Real * dNFndUp) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muI;

        if (std::abs(un[0]) > 0.5)
        {
            for (int v = 0; v < SHELL_NU; ++v)
            for (int u = 0; u < SHELL_NU; ++u)
            {
                // U
                dND1dU[u+v*SHELL_NU] = 0.5*this->Q11[u+v*SHELL_NU]*un[0];
                dND2dU[u+v*SHELL_NU] = 0.5*this->Q12[v+u*SHELL_NU]*un[0];

                dNFndU_1[u+v*SHELL_NU] = 0.5*this->Q11[u+v*SHELL_NU]*un[0];
                dNFndU_2[u+v*SHELL_NU] = 0.5*this->Q12[u+v*SHELL_NU]*un[0];

                dNFndU[u+v*SHELL_NU]  = 0.5*this->R1[u+v*SHELL_NU]*un[0];
                dNFndU[u+v*SHELL_NU] += 0.5*this->R1[v+u*SHELL_NU]*un[0];

                // U'
                dND1dUp[u+v*SHELL_NU] = -0.5*this->Q11[u+v*SHELL_NU]*un[0];
                dND2dUp[u+v*SHELL_NU] = -0.5*this->Q12[v+u*SHELL_NU]*un[0];

                dNFndUp_1[u+v*SHELL_NU] = 0.5*this->Q11[u+v*SHELL_NU]*un[0];
                dNFndUp_2[u+v*SHELL_NU] = 0.5*this->Q12[u+v*SHELL_NU]*un[0];

                dNFndUp[u+v*SHELL_NU]  = 0.5*this->R1[u+v*SHELL_NU]*un[0];
                dNFndUp[u+v*SHELL_NU] -= 0.5*this->R1[v+u*SHELL_NU]*un[0];
            }

            for (int u = 0; u < SHELL_NU; ++u)
            {
                // U
                dNFndU[u+u*SHELL_NU] -= mu;

                // U'
                dNFndUp[u+u*SHELL_NU] += mu;
            }
        }
        else
        {
            for (int v = 0; v < SHELL_NU; ++v)
            for (int u = 0; u < SHELL_NU; ++u)
            {
                // U
                dND1dU[u+v*SHELL_NU] = 0.5*this->Q12[u+v*SHELL_NU]*un[1];
                dND2dU[u+v*SHELL_NU] = 0.5*this->Q22[u+v*SHELL_NU]*un[1];

                dNFndU_1[u+v*SHELL_NU] = 0.5*this->Q12[v+u*SHELL_NU]*un[1];
                dNFndU_2[u+v*SHELL_NU] = 0.5*this->Q22[u+v*SHELL_NU]*un[1];

                dNFndU[u+v*SHELL_NU]  = 0.5*this->R2[u+v*SHELL_NU]*un[1];
                dNFndU[u+v*SHELL_NU] += 0.5*this->R2[v+u*SHELL_NU]*un[1];

                // U'
                dND1dUp[u+v*SHELL_NU] = -0.5*this->Q12[u+v*SHELL_NU]*un[1];
                dND2dUp[u+v*SHELL_NU] = -0.5*this->Q22[u+v*SHELL_NU]*un[1];

                dNFndUp_1[u+v*SHELL_NU] = 0.5*this->Q12[v+u*SHELL_NU]*un[1];
                dNFndUp_2[u+v*SHELL_NU] = 0.5*this->Q22[u+v*SHELL_NU]*un[1];

                dNFndUp[u+v*SHELL_NU]  = 0.5*this->R2[u+v*SHELL_NU]*un[1];
                dNFndUp[u+v*SHELL_NU] -= 0.5*this->R2[v+u*SHELL_NU]*un[1];
            }

            for (int u = 0; u < SHELL_NU; ++u)
            {
                // U
                dNFndU[u+u*SHELL_NU] -= mu;

                // U'
                dNFndUp[u+u*SHELL_NU] += mu;
            }
        }

        /* DEBUG
        amrex::Print() << "Q11: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11);
        amrex::Print() << "Q22: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22);
        amrex::Print() << "Q12: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12);
        amrex::Print() << "R1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1);
        amrex::Print() << "R2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2);
        amrex::Print() << "S: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, S);

        amrex::Print() << "U: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
        amrex::Print() << "U_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
        amrex::Print() << "U_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);

        amrex::Print() << "Up: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up);
        amrex::Print() << "Up_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up_1);
        amrex::Print() << "Up_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, Up_2);
        exit(-1);
        */
    }

    /**
     * \brief Eval the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *            variables.
     * \param[out] ND1, ND2, ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_BCs(const int /*dom*/,
                     const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * U_2, const amrex::Real * U_3),
                     AMREX_D_DECL(amrex::Real * ND1, amrex::Real * ND2, amrex::Real * ND3),
                     amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muD;

        // VARIABLES
        amrex::Real dU[SHELL_NU];
        amrex::Real T[SHELL_NU];

        if (std::abs(un[0]) > 0.5)
        {
            // SIMPLY SUPPORTED BCS
            amrex::Real cU[3] = {0.0, 1.0, 1.0};

            // GENERALIZED DISPLACEMENT JUMP
            for (int u = 0; u < SHELL_NU; ++u)
            {
                dU[u] = cU[u%3]*U[u];
            }

            // GENERALIZED TRACTIONS (INTERNAL, NON-PRESCRIBED)
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q11, SHELL_NU, U_1, 1, 0.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q12, SHELL_NU, U_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->R1, SHELL_NU, U, 1, 1.0, T, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                T[u] *= cU[u%3];
            }

            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[0], this->R1, SHELL_NU, dU, 1, 0.0, NFn, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                NFn[u] += -mu*dU[u]+T[u];
            }

            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[0], this->Q11, SHELL_NU, dU, 1, 0.0, ND1, 1);
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[0], this->Q12, SHELL_NU, dU, 1, 0.0, ND2, 1);
        }
        else
        {
            // SIMPLY SUPPORTED BCS
            amrex::Real cU[3] = {1.0, 0.0, 1.0};

            // GENERALIZED DISPLACEMENT JUMP
            for (int u = 0; u < SHELL_NU; ++u)
            {
                dU[u] = cU[u%3]*U[u];
            }

            // GENERALIZED TRACTIONS (INTERNAL, NON-PRESCRIBED)
            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[1], this->Q12, SHELL_NU, U_1, 1, 0.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->Q22, SHELL_NU, U_2, 1, 1.0, T, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->R2, SHELL_NU, U, 1, 1.0, T, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                T[u] *= cU[u%3];
            }

            amrex::dG::linalg::dgemv('T', SHELL_NU, SHELL_NU, un[1], this->R2, SHELL_NU, dU, 1, 0.0, NFn, 1);
            for (int u = 0; u < SHELL_NU; ++u)
            {
                NFn[u] += -mu*dU[u]+T[u];
            }

            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->Q12, SHELL_NU, dU, 1, 0.0, ND1, 1);
            amrex::dG::linalg::dgemv('N', SHELL_NU, SHELL_NU, un[1], this->Q22, SHELL_NU, dU, 1, 0.0, ND2, 1);
        }

        /* DEBUG
        amrex::Print() << "un: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
        
        amrex::Print() << "Q11: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11);
        amrex::Print() << "Q22: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22);
        amrex::Print() << "Q12: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12);
        amrex::Print() << "R1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1);
        amrex::Print() << "R2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2);
        
        amrex::Print() << "U: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
        amrex::Print() << "U_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
        amrex::Print() << "U_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);
        
        amrex::Print() << "dU: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, dU);
        amrex::Print() << "T: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, T);

        amrex::Print() << "NFn: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, NFn);
        amrex::Print() << "ND1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, ND1);
        amrex::Print() << "ND2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, ND2);
        exit(-1);
        */
    }

    /**
     * \brief Eval the gradient of the numerical flux at a grid's boundary.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *            variables.
     * \param[out] dNDidU: pointer to memory that will contain the derivative of the numerical flux
     *                     NDi with respect to U.
     * \param[out] dNFndU_j: pointer to memory that will contain the derivative of the numerical flux
     *                       NFn with respect to U_j.
     * \param[out] dNFndU: pointer to memory that will contain the derivative of the numerical flux
     *                     NFn with respect to U.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_dNFdU_BCs(const int /*dom*/,
                        const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * un,
                        const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * /*U_1*/, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                        AMREX_D_DECL(amrex::Real * dND1dU, amrex::Real * dND2dU, amrex::Real * dND3dU), 
                        AMREX_D_DECL(amrex::Real * dNFndU_1, amrex::Real * dNFndU_2, amrex::Real * dNFndU_3),
                        amrex::Real * dNFndU) const
    {
        // PARAMETERS
        const amrex::Real mu = this->muD;

        if (std::abs(un[0]) > 0.5)
        {
            // SIMPLY SUPPORTED BCS
            amrex::Real cU[3] = {0.0, 1.0, 1.0};

            for (int v = 0; v < SHELL_NU; ++v)
            for (int u = 0; u < SHELL_NU; ++u)
            {
                dND1dU[u+v*SHELL_NU] = this->Q11[u+v*SHELL_NU]*cU[v%3]*un[0];
                dND2dU[u+v*SHELL_NU] = this->Q12[v+u*SHELL_NU]*cU[v%3]*un[0];

                dNFndU_1[u+v*SHELL_NU] = cU[u%3]*this->Q11[u+v*SHELL_NU]*un[0];
                dNFndU_2[u+v*SHELL_NU] = cU[u%3]*this->Q12[u+v*SHELL_NU]*un[0];

                dNFndU[u+v*SHELL_NU]  = cU[u%3]*this->R1[u+v*SHELL_NU]*un[0];
                dNFndU[u+v*SHELL_NU] += this->R1[v+u*SHELL_NU]*cU[v%3]*un[0];
            }

            for (int u = 0; u < SHELL_NU; ++u)
            {
                dNFndU[u+u*SHELL_NU] -= cU[u%3]*mu;
            }
        }
        else
        {
            // SIMPLY SUPPORTED BCS
            amrex::Real cU[3] = {1.0, 0.0, 1.0};

            for (int v = 0; v < SHELL_NU; ++v)
            for (int u = 0; u < SHELL_NU; ++u)
            {
                dND1dU[u+v*SHELL_NU] = this->Q12[u+v*SHELL_NU]*cU[v%3]*un[1];
                dND2dU[u+v*SHELL_NU] = this->Q22[u+v*SHELL_NU]*cU[v%3]*un[1];

                dNFndU_1[u+v*SHELL_NU] = cU[u%3]*this->Q12[v+u*SHELL_NU]*un[1];
                dNFndU_2[u+v*SHELL_NU] = cU[u%3]*this->Q22[u+v*SHELL_NU]*un[1];

                dNFndU[u+v*SHELL_NU]  = cU[u%3]*this->R2[u+v*SHELL_NU]*un[1];
                dNFndU[u+v*SHELL_NU] += this->R2[v+u*SHELL_NU]*cU[v%3]*un[1];
            }

            for (int u = 0; u < SHELL_NU; ++u)
            {
                dNFndU[u+u*SHELL_NU] -= cU[u%3]*mu;
            }
        }

        /* DEBUG
        amrex::Print() << "un: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, AMREX_SPACEDIM, un);
        
        amrex::Print() << "Q11: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q11);
        amrex::Print() << "Q22: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q22);
        amrex::Print() << "Q12: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, Q12);
        amrex::Print() << "R1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R1);
        amrex::Print() << "R2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, R2);
        
        amrex::Print() << "U: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
        amrex::Print() << "U_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
        amrex::Print() << "U_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);

        amrex::Print() << "dNFndU: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, SHELL_NU, dNFndU);
        exit(-1);
        */
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] U: pointer to memory containing the conserved variables.
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *                           variables.
     * \param[out] ND1, ND2, ND3: pointer to memory that will contain the numerical fluxes.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_EB(const int /*dom*/,
                    const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                    const amrex::Real * /*U*/, AMREX_D_DECL(const amrex::Real * /*U_1*/, const amrex::Real * /*U_2*/, const amrex::Real * /*U_3*/),
                    AMREX_D_DECL(amrex::Real * /*ND1*/, amrex::Real * /*ND2*/, amrex::Real * ND3),
                    amrex::Real * /*NFn*/) const
    {
amrex::Print() << "eval_NFn_EB - 1" << std::endl;
exit(-1);
    }

    /**
     * \brief Eval the numerical flux at the embedded boundaries.
     *
     * \param[in] dom: index of the domain.
     * \param[in] nbr_dom: index of the domain that is neighbor with dom.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] un: pointer to memory containing the outward unit normal of the boundary.
     * \param[in] m_U: pointer to memory containing the conserved variables on the negative side of the
     *                 unit normal.
     * \param[in] p_U: pointer to memory containing the conserved variables on the positive side of the
     *                 unit normal.
     * \param[out] NFn: pointer to memory that will contain the numerical flux.
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_NF_EB(const int /*dom*/, const int /*nbr_dom*/,
                    const amrex::Real /*t*/, const amrex::Real * /*x*/, const amrex::Real * /*un*/,
                    const amrex::Real * /*m_U*/, const amrex::Real * /*p_U*/, amrex::Real * /*NFn*/) const
    {
amrex::Print() << "eval_NFn_EB - 2" << std::endl;
exit(-1);
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
amrex::Print() << "eval_FV_slopes" << std::endl;
exit(-1);
    }
    // ================================================================


    // OUTPUT FIELDS ==================================================
//#define OUTPUT_PRIMARY_VARIABLES

    /**
     * \brief Return the number of output fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_output_fields() const
    {
        return SHELL_NU;
    }
    /**
     * \brief Return the number of output fields.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int get_number_of_shell_output_fields() const
    {
        return 12;
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int output_field_parent_domain(const int /*u*/) const
    {
        return 0;
    }
    /**
     * \brief Return the parent domain of the u-th output field.
     *
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int shell_output_field_parent_domain(const int /*u*/) const
    {
        return 0;
    }

    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
        output_fields_names = {"U1", "U2", "U3", "U4", "U5"};
    }

    /**
     * \brief Set the names of the fields to be output.
     * 
     * \param[out] fields_name: vector of strings containing the names of the output fields.
    */
    void get_shell_output_fields_names(amrex::Vector<std::string> & output_fields_names) const
    {
        output_fields_names = {"X1", "X2", "X3",
                               "ux", "uy", "uz",
                               "sxx", "syy", "szz", "syz", "sxz", "sxy"};
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const amrex::Real /*t*/, const amrex::Real * /*x*/,
                            const amrex::Real * /*U*/,
                            amrex::Real * /*F*/) const
    {
amrex::Print() << "eval_output_fields" << std::endl;
exit(-1);
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *                           variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_output_fields(const int /*dom*/, const amrex::Real /*t*/, const amrex::Real * x,
                            const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * U_2, const amrex::Real * U_3),
                            amrex::Real * F) const
    {
        for (int u = 0; u < SHELL_NU; ++u)
        {
            F[u] = U[u];
        }
    }

    /**
     * \brief Fill F with fields to be output.
     * 
     * \param[in] dom: domain index.
     * \param[in] t: time.
     * \param[in] x: pointer to memory containing the space location.
     * \param[in] U: pointer to memory that stores the conserved variables at (t,x).
     * \param[in] U_1, U_2, U_3: pointer to memory containing the derivatives of the conserved
     *                           variables at (t,x).
     * \param[out] F: pointer to memory that will be filled with values of the output fields at (t,x).
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void eval_shell_output_fields(const int /*dom*/, const amrex::Real /*t*/, const amrex::Real * x, const amrex::Real x3,
                                  const amrex::Real * U, AMREX_D_DECL(const amrex::Real * U_1, const amrex::Real * U_2, const amrex::Real * U_3),
                                  amrex::Real * F) const
    {
        // Find the corresponding layer
        int l;
        bool found;

        l = 0;
        found = false;
        while ((l < this->n_layers) && (!found))
        {
            const amrex::Real x3b = this->layers_x3b[l];
            const amrex::Real x3t = x3b+this->layers_thickness[l];
            const amrex::Real tol = 1.0e-5*this->layers_thickness[l];

            if ((x3b-tol < x3) && (x3 < x3t+tol))
            {
                found = true;
            }
            else
            {
                l += 1;
            }
        }
        
        const amrex::Real th = this->layers_orientation[l];
        const Material & mat = this->materials[this->layers_material[l]];

        amrex::Real RF[3*SHELL_NU];
        amrex::Real J1[6*SHELL_NU], J2[6*SHELL_NU], J3[6*SHELL_NU];
        amrex::Real cSE[6*6];
        amrex::Real X[3], u[3], E[6], S_[6];

        this->shell_map_volume(x, x3, X);
        this->shell_theory_eval_RF_operator(x, x3, RF);
        this->shell_theory_eval_Ji_operators(x, x3, J1, J2, J3);
        this->shell_theory_eval_cSE(x, x3, mat.cSE, th, cSE);

        amrex::dG::linalg::dgemv('N', 3, SHELL_NU, 1.0, RF, 3, U, 1, 0.0, u, 1);
        
        amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J3, 6, U, 1, 0.0, E, 1);
        amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J1, 6, U_1, 1, 1.0, E, 1);
        amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J2, 6, U_2, 1, 1.0, E, 1);

        amrex::dG::linalg::dgemv('N', 6, 6, 1.0, cSE, 6, E, 1, 0.0, S_, 1);

        F[0] = X[0];
        F[1] = X[1];
        F[2] = X[2];
        F[0+3] = u[0];
        F[1+3] = u[1];
        F[2+3] = u[2];
        F[0+6] = S_[0];
        F[1+6] = S_[1];
        F[2+6] = S_[2];
        F[3+6] = S_[3];
        F[4+6] = S_[4];
        F[5+6] = S_[5];

        /*
        if ((std::abs(x[0]-0.5) < 1.0e-5) && (std::abs(x[1]-0.5) < 1.0e-5) && (std::abs(x3-0.5*0.01) < 1.0e-7))
        {
            amrex::Print() << "l: " << l << std::endl;
            amrex::Print() << "th: " << th << std::endl;
            amrex::Print() << "x3: " << x3 << std::endl;
            amrex::Print() << "S_: "; amrex::dG::io::print_real_array_2d(1, NS, S_);
            exit(-1);
        }
        */

        /* DEBUG
        amrex::Print() << "U: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U);
        amrex::Print() << "U_1: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_1);
        amrex::Print() << "U_2: " << std::endl;
        amrex::dG::io::print_real_array_2d(SHELL_NU, 1, U_2);
        
        amrex::Print() << "u: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, 3, u);
        amrex::Print() << "E: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, NS, E);
        amrex::Print() << "S_: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, NS, S_);
        amrex::Print() << "F: " << std::endl;
        amrex::dG::io::print_real_array_2d(1, 9, F);
        exit(-1);
        */
    }
    // ================================================================
};
// ####################################################################



// SINGLE PHASE DOMAIN: AMR ###########################################
struct AMR
:
public amrex::dG::amr::SinglePatch
{
    // DATA MEMBERS ===================================================
    amrex::Real t;
    amrex::Real mu;

    // LEVEL SET
    LevelSet<AMREX_SPACEDIM> level_set;
    
    // IBVP
    IBVP ibvp;

    // MASS MATRIX MULTIFAB
    amrex::Vector<amrex::MultiFab> MCh;

    // SOLUTION MULTIFAB
    amrex::Vector<amrex::MultiFab> X;

    // SOLUTION MULTIFABS: EXPLICIT CENTRAL DIFFERENCE SCHEME
    amrex::Vector<amrex::MultiFab> X_prev;

    // SOLUTION MULTIFABS: IMPLICIT NEWMARK SCHEME
    amrex::Vector<amrex::MultiFab> Xd;

    // LINEAR DIRECT SOLVER
#ifdef AMREX_USE_PARDISO
    amrex::dG::LinearSolverPardiso pardiso;
#endif
#ifdef AMREX_USE_HYPRE
    amrex::dG::LinearSolverHypre hypre;
#endif

    // NORM
    std::string norm;
    // ================================================================


    // CONSTRUCTOR ====================================================
    AMR()
    :
    amrex::dG::amr::SinglePatch::SinglePatch(),
    ibvp(this->geom[0].CellSize())
    {
        const int n_levels = this->max_level+1;

        this->mu = amrex::max(ibvp.muD, ibvp.muI);

        // MASS MATRIX MULTIFAB
        this->MCh.resize(n_levels);

        // SOLUTION MULTIFAB
        this->X.resize(n_levels);

        // SOLUTION MULTIFABS: EXPLICIT CENTRAL DIFFERENCE SCHEME
        this->X_prev.resize(n_levels);

        // SOLUTION MULTIFABS: IMPLICIT NEWMARK SCHEME
        this->Xd.resize(n_levels);

        // NORM
        {
            amrex::ParmParse pp;
            pp.get("norm", this->norm);
        }
    }
    // ================================================================


    // INITIALIZATION =================================================
    /**
     * \brief Initialize AMR data.
     * 
    */
    void init()
    {
        // INIT PARENT DATA STRUCTURE ---------------------------------
        amrex::dG::amr::SinglePatch::init();
        // ------------------------------------------------------------

        // INIT SOLUTION FROM SCRATCH ---------------------------------
        if (this->inputs.restart < 0)
        {
            this->t = 0.0;
            this->InitFromScratch(this->t);
            
            amrex::dG::Solution & solution = *this->solutions[0];

            if (solution.time_integration_is_explicit_central_difference())
            {
                // After this->InitFromScratch(this->t), the vectors of
                // MultiFab X_prev, X contains the projection of the
                // initial conditions and the projection of the time
                // derivative of the initial conditions, respectively.
                // We now compute the correct X to be used in the
                // central-difference scheme.
                amrex::Real dt;
                dt = amrex::dG::hyperbolic_2nd_order_pde::eval_dt(*this, this->t, this->X_prev, this->ibvp);
                dt *= 1.0/std::sqrt(this->mu);

                amrex::Vector<amrex::MultiFab> d2Xdt2(this->finest_level+1);
                for (int lev = 0; lev <= this->finest_level; ++lev)
                {
                    d2Xdt2[lev].define(this->X_prev[lev].boxarray, this->X_prev[lev].distributionMap, this->X_prev[lev].n_comp, this->X_prev[lev].n_grow);
                    d2Xdt2[lev] = 0.0;
                }
                amrex::dG::hyperbolic_2nd_order_pde::eval_d2Xdt2_explicit_central_difference(*this, this->t, this->MCh, this->X_prev, d2Xdt2, this->ibvp);
                
                for (int lev = 0; lev <= this->finest_level; ++lev)
                {
                    const amrex::dG::Mesh & mesh = *this->meshes[lev];

                    // X := X_prev+dt*X+0.5*dt*dt*d2Xdt2(t, X_prev)
                    this->X[lev].mult(dt, mesh.dom_data_n_grow);
                    amrex::MultiFab::Saxpy(this->X[lev],       1.0, this->X_prev[lev], 0, 0, this->X[lev].n_comp, this->X[lev].n_grow);
                    amrex::MultiFab::Saxpy(this->X[lev], 0.5*dt*dt,       d2Xdt2[lev], 0, 0, this->X[lev].n_comp, this->X[lev].n_grow);
                }
            }
            else if (solution.time_integration_is_implicit_Newmark())
            {
                // Do nothing
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::init\n";
                msg += "| Unexpected time integration method: "+solution.params.time_integration+".\n";
                amrex::Abort(msg);
            }
        }
        // ------------------------------------------------------------
        else
        // INIT SOLUTION FROM CHECKPOINT ------------------------------
        {
amrex::Print() << "INIT SOLUTION FROM CHECKPOINT" << std::endl;
exit(-1);
        }
        // ------------------------------------------------------------

        // UPDATE MASKS -----------------------------------------------
        this->update_masks();
        this->update_ordering(this->ibvp);
        // ------------------------------------------------------------
    }
    // ================================================================


    // MAKE NEW LEVEL FROM SCRATCH ====================================
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/, const amrex::BoxArray & ba, const amrex::DistributionMapping & dm) override
    {
        // PARAMETERS -------------------------------------------------
        const int mask_n_grow = 1;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::dG::Mesh & mesh = *this->meshes[lev];
        amrex::dG::Solution & solution = *this->solutions[lev];
        // ------------------------------------------------------------

        // MESH -------------------------------------------------------
        if (mesh.uses_level_set())
        {
            mesh.make_from_scratch_by_level_set(this->t, this->geom[lev], ba, dm, this->ibvp, this->level_set);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::MakeNewLevelFromScratch\n";
            msg += "| Unexpected method used to define the embedded geometry.\n";
            msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
            amrex::Abort(msg);
        }
        this->SetBoxArray(lev, mesh.ba);
        this->SetDistributionMap(lev, mesh.dm);
        // ------------------------------------------------------------

        // SOLUTION ---------------------------------------------------
        solution.make_from_scratch(this->t, this->geom[lev], mesh, this->ibvp);
        solution.eval_space_mass_matrix(this->geom[lev], mesh, this->ibvp);

        solution.init_pde_mass_matrix_multifab(mesh, this->MCh[lev], this->ibvp);
        solution.eval_pde_mass_matrix(this->t, this->geom[lev], mesh, this->MCh[lev], this->ibvp);

        if (solution.time_integration_is_explicit_central_difference())
        {
            // SOLUTION MULTIFABS
            solution.init_space_solution_multifabs(mesh, this->X_prev[lev], this->ibvp);
            solution.init_space_solution_multifabs(mesh, this->X[lev], this->ibvp);
            solution.project_initial_conditions(this->geom[lev], mesh, this->X_prev[lev], this->ibvp);
            solution.project_initial_conditions_derivative(this->geom[lev], mesh, this->X[lev], this->ibvp);
        }
        else if (solution.time_integration_is_implicit_Newmark())
        {
            // SOLUTION MULTIFABS
            solution.init_space_solution_multifabs(mesh, this->X[lev], this->ibvp);
            solution.init_space_solution_multifabs(mesh, this->Xd[lev], this->ibvp);
            solution.project_initial_conditions(this->geom[lev], mesh, this->X[lev], this->ibvp);
            solution.project_initial_conditions_derivative(this->geom[lev], mesh, this->Xd[lev], this->ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::MakeNewLevelFromScratch\n";
            msg += "| Unexpected time integration method: "+solution.params.time_integration+".\n";
            amrex::Abort(msg);
        }
        // ------------------------------------------------------------

        // MASKS ------------------------------------------------------
        this->masks[lev].define(mesh.ba, mesh.dm, 1, mask_n_grow);
        this->masks[lev] = 0;
        // ------------------------------------------------------------
    }
    // ================================================================


    // ERROR EVALUATION ===============================================
    void eval_error(const amrex::Real t_, amrex::Real & err, amrex::Real & err_norm) const
    {
        if (this->norm.compare("L_inf") == 0)
        {
            this->eval_error_L_inf(t_, this->X, err, err_norm, this->ibvp);
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::eval_error\n";
            msg += "| Unexpected error norm.\n";
            msg += "| requested norm           : "+this->norm+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // ADVANCE IN TIME ================================================
    amrex::Real eval_dt(const amrex::Real t_) const
    {
        amrex::Real dt = amrex::dG::hyperbolic_2nd_order_pde::eval_dt(*this, t_, this->X, this->ibvp);
        dt *= 1.0/std::sqrt(this->mu);

        return dt;
    }

    void take_time_step(const amrex::Real t_, const amrex::Real dt)
    {
        amrex::dG::Solution & solution = *this->solutions[0];
        
        if (solution.time_integration_is_explicit_central_difference())
        {
            amrex::dG::hyperbolic_2nd_order_pde::take_time_step_using_explicit_central_difference(*this,
                                                                                                  t_, dt,
                                                                                                  this->MCh,
                                                                                                  this->X_prev,
                                                                                                  this->X,
                                                                                                  this->ibvp);
        }
        else if (solution.time_integration_is_implicit_Newmark())
        {
#ifdef AMREX_USE_PARDISO
            amrex::dG::hyperbolic_2nd_order_pde::take_time_step_using_implicit_Newmark(*this,
                                                                                       t_, dt,
                                                                                       this->MCh,
                                                                                       this->X,
                                                                                       this->Xd,
                                                                                       this->pardiso,
                                                                                       this->ibvp);
#endif
#ifdef AMREX_USE_HYPRE
            amrex::dG::hyperbolic_2nd_order_pde::take_time_step_using_implicit_Newmark(*this,
                                                                                       t_, dt,
                                                                                       this->MCh,
                                                                                       this->X,
                                                                                       this->Xd,
                                                                                       this->hypre,
                                                                                       this->ibvp);
#endif
        }
        else
        {
            std::string msg;
            msg  = "\n";
            msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::take_time_step\n";
            msg += "| Unexpected time integration method: "+solution.params.time_integration+".\n";
            amrex::Abort(msg);
        }
    }
    // ================================================================


    // LINEAR SOLVER ==================================================
    void init_linear_solver()
    {
#ifdef AMREX_USE_PARDISO
        // INPUT FILE
        this->pardiso.read_input_file();

        // PARDISO BASE CONTROL PARAMETERS
        this->pardiso.init();
        
        // INIT DATA STRUCTURES
        amrex::dG::amr::SinglePatch::init_linear_solver(this->pardiso, this->ibvp);
#endif
#ifdef AMREX_USE_HYPRE
        // INPUT FILE
        this->hypre.read_input_file();

        // HYPRE BASE CONTROL PARAMETERS
        this->hypre.init();
        
        // INIT DATA STRUCTURES
        amrex::dG::amr::SinglePatch::init_linear_solver(this->hypre, this->ibvp);
#endif
    }

    void terminate_linear_solver()
    {
#ifdef AMREX_USE_PARDISO
        this->pardiso.terminate();
#endif
#ifdef AMREX_USE_PARDISO
        this->hypre.terminate();
#endif
    }
    // ================================================================


    // INPUT/OUTPUT ===================================================
    void write_sampled_solution(const int n, const amrex::Real t_)
    {
        // PARAMETERS -------------------------------------------------
        // CURRENT PROCESSOR
        const int rank = amrex::ParallelDescriptor::MyProc();

        // OUTPUT FILEPATH
        const std::string filepath = amrex::dG::io::make_path({this->inputs.output_folderpath, "sampled_solution.txt"});

        // DOMAIN
        const int dom = 0;

        // RECEIVERS INFO
        const int n_sampling_points = 2;
        const amrex::Vector<amrex::Real> sampling_points_coordinates = {0.0, 0.5, 0.5, 0.5};
        const int n_sampled_fields_per_point = 8;
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        amrex::Vector<int> sampling_points_rank(n_sampling_points);
        amrex::Vector<amrex::Real> sampled_fields(n_sampling_points*n_sampled_fields_per_point);

        // WRITING TOOLS
        std::ofstream fp;
        amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
        // ------------------------------------------------------------

        // SAMPLE THE FIELDS ------------------------------------------
        std::fill(sampling_points_rank.begin(), sampling_points_rank.end(), -1);

        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            // GRID
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = this->Geom(lev).CellSizeArray();
            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = this->Geom(lev).ProbLoArray();

            // ORDER
            const int sp = this->solutions[lev]->params.space_p;

            for (amrex::MFIter mfi(this->X[lev]); mfi.isValid(); ++mfi)
            {
                const amrex::Box & bx = mfi.validbox();
                const amrex::Dim3 lo = lbound(bx);
                const amrex::Dim3 hi = ubound(bx);

                amrex::Array4<short const> const & elm_type_fab = this->solutions[lev]->elm_type.array(mfi);
                amrex::Array4<amrex::Real const> const & X_fab = this->X[lev].array(mfi);

                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    // ELEMENT TYPE
                    const short etype = elm_type_fab(i,j,k,amrex::dG::ELM_TYPE(dom));

                    // LOCAL PARAMETERS
                    const amrex::Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                              prob_lo[1]+j*dx[1],
                                                                              prob_lo[2]+k*dx[2])};
                    const amrex::Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(cell_lo[0]+dx[0],
                                                                              cell_lo[1]+dx[1],
                                                                              cell_lo[2]+dx[2])};
                    
                    // LOCAL VARIABLES
                    bool cond;
                    amrex::Real U[SHELL_NU], U_1[SHELL_NU], U_2[SHELL_NU];
                    amrex::Real U_exact[SHELL_NU], U_1_exact[SHELL_NU], U_2_exact[SHELL_NU];
                    amrex::Real u[3], E[6], S[6];

                    // UNKNOWN FIELDS
                    amrex::dG::space_elm_bfx bfu(&prob_lo[0], &dx[0], i, j, k, etype, sp, X_fab);

                    for (int s = 0; s < n_sampling_points; ++s)
                    {
                        const amrex::Real * xs = &sampling_points_coordinates[AMREX_SPACEDIM*s];

                        AMREX_D_TERM
                        (
                            cond =         (cell_lo[0] <= xs[0]) && (xs[0] <= cell_hi[0]);,
                            cond = cond && (cell_lo[1] <= xs[1]) && (xs[1] <= cell_hi[1]);,
                            cond = cond && (cell_lo[2] <= xs[2]) && (xs[2] <= cell_hi[2]);
                        )

                        if (cond)
                        {
                            sampling_points_rank[s] = rank;
                            
                            bfu.eval_bf_table(xs);
                            bfu.eval_unsafe(0, SHELL_NU, U);
                            bfu.eval_derivatives_unsafe(0, SHELL_NU, U_1, U_2);

                            this->ibvp.eval_exact_solution(t_, xs, U_exact, U_1_exact, U_2_exact);

                            // EVAL FIELDS: DG
                            {
                                const int l = this->ibvp.n_layers-1;
                                const amrex::Real x3b = this->ibvp.layers_x3b[l];
                                const amrex::Real tau = this->ibvp.layers_thickness[l];
                                const amrex::Real th = this->ibvp.layers_orientation[l];
                                const Material & mat = this->ibvp.materials[this->ibvp.layers_material[l]];
                                const amrex::Real x3 = x3b+tau;

                                amrex::Real RF[3*SHELL_NU];
                                amrex::Real J1[6*SHELL_NU], J2[6*SHELL_NU], J3[6*SHELL_NU];
                                amrex::Real cSE[6*6];

                                this->ibvp.shell_theory_eval_RF_operator(xs, x3, RF);
                                this->ibvp.shell_theory_eval_Ji_operators(xs, x3, J1, J2, J3);
                                this->ibvp.shell_theory_eval_cSE(xs, x3, mat.cSE, th, cSE);

                                amrex::dG::linalg::dgemv('N', 3, SHELL_NU, 1.0, RF, 3, U, 1, 0.0, u, 1);
                                
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J3, 6, U, 1, 0.0, E, 1);
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J1, 6, U_1, 1, 1.0, E, 1);
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J2, 6, U_2, 1, 1.0, E, 1);

                                amrex::dG::linalg::dgemv('N', 6, 6, 1.0, cSE, 6, E, 1, 0.0, S, 1);
                            }
                            
                            sampled_fields[n_sampled_fields_per_point*s+0] = u[0];
                            sampled_fields[n_sampled_fields_per_point*s+1] = u[2];
                            sampled_fields[n_sampled_fields_per_point*s+2] = S[0];
                            sampled_fields[n_sampled_fields_per_point*s+3] = S[4];

                            // EVAL FIELDS: EXACT SOLUTION
                            {
                                const int l = this->ibvp.n_layers-1;
                                const amrex::Real x3b = this->ibvp.layers_x3b[l];
                                const amrex::Real tau = this->ibvp.layers_thickness[l];
                                const amrex::Real th = this->ibvp.layers_orientation[l];
                                const Material & mat = this->ibvp.materials[this->ibvp.layers_material[l]];
                                const amrex::Real x3 = x3b+tau;

                                amrex::Real RF[3*SHELL_NU];
                                amrex::Real J1[6*SHELL_NU], J2[6*SHELL_NU], J3[6*SHELL_NU];
                                amrex::Real cSE[6*6];

                                this->ibvp.shell_theory_eval_RF_operator(xs, x3, RF);
                                this->ibvp.shell_theory_eval_Ji_operators(xs, x3, J1, J2, J3);
                                this->ibvp.shell_theory_eval_cSE(xs, x3, mat.cSE, th, cSE);

                                amrex::dG::linalg::dgemv('N', 3, SHELL_NU, 1.0, RF, 3, U_exact, 1, 0.0, u, 1);
                                
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J3, 6, U_exact, 1, 0.0, E, 1);
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J1, 6, U_1_exact, 1, 1.0, E, 1);
                                amrex::dG::linalg::dgemv('N', 6, SHELL_NU, 1.0, J2, 6, U_2_exact, 1, 1.0, E, 1);

                                amrex::dG::linalg::dgemv('N', 6, 6, 1.0, cSE, 6, E, 1, 0.0, S, 1);
                            }
                            
                            sampled_fields[n_sampled_fields_per_point*s+4] = u[0];
                            sampled_fields[n_sampled_fields_per_point*s+5] = u[2];
                            sampled_fields[n_sampled_fields_per_point*s+6] = S[0];
                            sampled_fields[n_sampled_fields_per_point*s+7] = S[4];
                        }
                    }
                }
            }
        }
        // ------------------------------------------------------------

        // SEND INFO TO IO PROCESSOR ----------------------------------
        amrex::ParallelDescriptor::ReduceIntMax(sampling_points_rank.data(), n_sampling_points);

        /*
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            for (int s = 0; s < n_sampling_points; ++s)
            {
                amrex::Print() << " sampling_points_rank[" << s << "]: " << sampling_points_rank[s] << std::endl;
            }
        }
        */
        
        for (int s = 0; s < n_sampling_points; ++s)
        {
            if ((sampling_points_rank[s] != rank) && (amrex::ParallelDescriptor::IOProcessor()))
            {
                amrex::ParallelDescriptor::Recv(&sampled_fields[n_sampled_fields_per_point*s], n_sampled_fields_per_point, sampling_points_rank[s], 0);
            }
            else if ((sampling_points_rank[s] == rank) && (!amrex::ParallelDescriptor::IOProcessor()))
            {
                amrex::ParallelDescriptor::Send(&sampled_fields[n_sampled_fields_per_point*s], n_sampled_fields_per_point, amrex::ParallelDescriptor::IOProcessorNumber(), 0);
            }
        }
        // ------------------------------------------------------------

        if (amrex::ParallelDescriptor::IOProcessor())
        {
            // OPEN FILE FOR WRITING ----------------------------------
            fp.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
            if (n == 0)
            {
                fp.open(filepath.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

                if (!fp.good())
                {
                    amrex::FileOpenFailed(filepath);
                }

                fp << "SAMPLED SOLUTION\n";
                for (int s = 0; s < n_sampling_points; ++s)
                {
                    const amrex::Real * xs = &sampling_points_coordinates[AMREX_SPACEDIM*s];
                    fp << "R" << std::to_string(s) << ": " << xs[0] << " " << xs[1] << std::endl;
                }

                fp << std::setw(16) << "t ";
                for (int s = 0; s < n_sampling_points; ++s)
                {
                    fp << std::setw(13) << "u1(t,R" << std::to_string(s) << ") " << std::setw(13) << "u3(t,R" << std::to_string(s) << ") " << std::setw(13) << "s11(t,R" << std::to_string(s) << ") " << std::setw(13) << "s13(t,R" << std::to_string(s) << ") "
                       << std::setw(13) << " u1_e(t,R" << std::to_string(s) << ") " << std::setw(13) << "u3_e(t,R" << std::to_string(s) << ") " << std::setw(13) << "s11_e(t,R" << std::to_string(s) << ") " << std::setw(13) << "s13_e(t,R" << std::to_string(s) << ")";
                }
                fp << std::endl;
            }
            else
            {
                fp.open(filepath.c_str(), std::ofstream::app | std::ofstream::binary);

                if (!fp.good())
                {
                    amrex::FileOpenFailed(filepath);
                }
            }

            fp << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << t_;
            for (int f = 0; f < n_sampling_points*n_sampled_fields_per_point; ++f)
            {
                fp << " " << std::scientific << std::setprecision(8) << std::setw(15) << std::showpos << sampled_fields[f];
            }
            fp << std::endl;
            // --------------------------------------------------------

            // CLOSE FILE
            fp.close();
            // ----------
        }
    }

    void export_mesh(const int n, const std::string & filename_root)
    {
        this->export_mesh_to_VTK(n, filename_root, this->ibvp);
    }

    void export_quadrature_points(const int n, const std::string & filename_root)
    {
        this->export_quadrature_points_to_VTK(n, filename_root, this->ibvp);
    }

    void export_solution(const int n, const std::string & filename_root, const amrex::Real t_)
    {
        for (int lev = 0; lev <= this->finest_level; ++lev)
        {
            const amrex::dG::Mesh & mesh = *this->meshes[lev];
            const amrex::dG::Solution & solution = *this->solutions[lev];
            const amrex::iMultiFab & mask = this->masks[lev];

            const std::string level_folderpath = this->inputs.get_level_folderpath(lev);
            const std::string level_step_folderpath = this->inputs.get_level_step_folderpath(lev, n);
            const std::string step_string = this->inputs.get_step_string(n);

            if (mesh.uses_level_set())
            {
                amrex::dG::solution_io::export_as_shells_using_level_set_to_VTK(level_folderpath, level_step_folderpath, step_string, filename_root,
                                                                                t_, this->geom[lev], solution, this->X[lev], mask, this->ibvp, this->level_set,
                                                                                true);
            }
            else
            {
                std::string msg;
                msg  = "\n";
                msg +=  "ERROR: IBVP_MultilayeredPlate.H - AMR::export_solution\n";
                msg += "| Unexpected method used to define the embedded geometry.\n";
                msg += "| Unexpected method: "+mesh.params.embedded_geometry_defined_by+".\n";
                amrex::Abort(msg);
            }
        }
    }
    // ================================================================
};
// ####################################################################

}