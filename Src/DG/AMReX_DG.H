// AMReX_DG.H

#ifndef BL_DG_H_
#define BL_DG_H_

#include <AMReX_TagBox.H>

#include <AMReX_DG_Base.H>
#include <AMReX_DG_InputReader.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>
#include <AMReX_DG_Solution.H>
#include <AMReX_DG_Solution_AMR.H>
#include <AMReX_DG_SlopeLimiter.H>
#include <AMReX_DG_RungeKutta.H>

namespace amrex
{
namespace DG
{
// ####################################################################
// DG BASE CLASS ######################################################
// ####################################################################
template<int N_PHI, int N_DOM, int N_U>
struct DG
{
    // DATA MEMBERS ###################################################
    // CLASS-WIDE PARAMETERS
    static const int dom_data_ngr = 1;
    
    // PDES INFO
    std::string PDEs_info;

    // IMPLICIT MESH
    ImplicitMesh<N_PHI, N_DOM> mesh;

    // MATRIX FACTORY
    MatrixFactory<N_PHI, N_DOM> matfactory;

    // SLOPE LIMITER
    bool limiter_is_active;

    // SOLUTION MULTIFABS
    MultiFab X_old, X_new;
    charMultiFab tags;

    // TIME MARCHING
    int n_time_steps;
    Real CFL_base;
    std::string time_marching_info;
    int tp, tNp;
    RungeKutta<N_PHI, N_DOM, N_U> RK;

    // POST-PROCESSING
    std::string output_folder;
    Vector<int> fields_domain;
    Vector<std::string> fields_name;
    
    Vector<Array<Real, AMREX_SPACEDIM>> point_fields_location;
    Vector<Array<int, 3>> point_fields_ijk;
    Vector<int> point_fields_domain;
    Vector<Vector<std::string>> point_fields_name;
    Vector<int> point_fields_rank;
    bool print_point_fields;
    // ################################################################

    // CONSTRUCTOR ####################################################
    DG()
    :
    PDEs_info{""},
    limiter_is_active{false},
    n_time_steps{-1},
    CFL_base{0.0},
    time_marching_info{""},
    tp{-1},
    tNp{-1},
    output_folder{""},
    print_point_fields{false}
    {}

    DG(const InputReader & inputs, const int d = 0)
    {
        this->Define(inputs, d);
    }
    // ################################################################

    // DESTRUCTOR #####################################################
    ~DG()
    {}
    // ################################################################

    // INITIALIZATION #################################################
    void Define(const InputReader & inputs, const int d)
    {
        // PDES INFO ==================================================
        this->PDEs_info = inputs.PDEs[d].description;
        // ============================================================

        // IMPLICIT MESH ==============================================
        this->mesh.Define(inputs, d);
        // ============================================================

        // MATRIX FACTORY =============================================
        this->matfactory.Define(this->mesh.cc_ba, this->mesh.dm, inputs, d);
        // ============================================================

        // SLOPE LIMITER ==============================================
        this->limiter_is_active = inputs.dG[d].limiter.active;
        // ============================================================

        // SOLUTION MULTIFABS =========================================
        {
            int n_comp = (this->matfactory.cube.Np)*N_U;

            this->X_old.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_old = 0.0;

            this->X_new.define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->dom_data_ngr);
            this->X_new = 0.0;
        }

        this->tags.define(this->mesh.cc_ba, this->mesh.dm, N_DOM, this->dom_data_ngr);
        this->tags = SOUND_ELEMENT;
        // ============================================================

        // TIME MARCHING ==============================================
        this->n_time_steps = inputs.time.n_steps;

        this->CFL_base = inputs.dG[d].CFL;

        this->time_marching_info = inputs.dG[d].time_marching;

        this->tp = inputs.dG[d].time_p;
        this->tNp = inputs.dG[d].time_p+1;

        if (this->isTimeMarchingRungeKutta())
        {
            const int n_comp = (this->matfactory.cube.Np)*N_U;

            this->RK.Define(this->mesh.cc_ba, this->mesh.dm, n_comp, this->tNp);
        }
        else
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG.H - DG::Define" << std::endl;
            Print() << "| Unexpected time-stepping scheme: " << this->time_marching_info << std::endl;
            Print() << std::endl;
            exit(-1);
        }
        // ============================================================

        // POST-PROCESSING ============================================
        // ============================================================
    }
    // ################################################################

    // MISC ###########################################################
    bool isPDEsSystemHyperbolic() const
    {
        return (this->PDEs_info.compare("Hyperbolic") == 0);
    }

    bool isTimeMarchingRungeKutta() const
    {
        return (this->time_marching_info.compare("RungeKutta") == 0);
    }
    // ################################################################

    // PRINT SOME REPORTS #############################################
    void PrintMemoryReport() const
    {
        ParallelDescriptor::Barrier();
        Print() << "IMPLICIT MESH MEMORY REPORT:" << std::endl;

        for (int rank = 0; rank < ParallelDescriptor::NProcs(); ++rank)
        {
            if (ParallelDescriptor::MyProc() == rank)
            {
                printf("| Hello, I am processor %d and I have counted:\n", ParallelDescriptor::MyProc());
                printf("| %d cut elements\n", this->mesh.n_cut_elms);
                printf("| %d entire extended elements\n", this->mesh.n_entire_extended_elms);
                printf("| ...and I have allocated:\n");
                printf("| %lu doubles (%f MB) for implicit domain quadrature;\n", this->mesh.mem_for_eDomQuad, this->mesh.mem_for_eDomQuad*sizeof(Real)/(1024.0*1024.0));
                printf("| %lu doubles (%f MB) for implicit internal boundary quadrature;\n", this->mesh.mem_for_eIntBouQuad, this->mesh.mem_for_eIntBouQuad*sizeof(Real)/(1024.0*1024.0));
                printf("| %lu doubles (%f MB) for query points.\n", this->mesh.mem_for_eQuery, this->mesh.mem_for_eQuery*sizeof(Real)/(1024.0*1024.0));
                printf("| %lu doubles (%f MB) for implicit boundary quadrature.\n", this->mesh.mem_for_eBouQuad, this->mesh.mem_for_eBouQuad*sizeof(Real)/(1024.0*1024.0));
                printf("| %lu doubles (%f MB) for the mass matrices.\n", this->matfactory.mem_for_space_MMCh, this->matfactory.mem_for_space_MMCh*sizeof(Real)/(1024.0*1024.0));
            }

            ParallelDescriptor::Barrier();
        }

        ParallelDescriptor::Barrier();
        Print() << std::endl;
    }
    // ################################################################

    // INITIAL CONDITIONS #############################################
    template<typename IBVP_CLASS>
    void SetICs(const IBVP_CLASS & IBVP, const bool im_flag = true)
    {
        // PROJECT THE LEVEL SET FUNCTIONS OVER THE GRID ==============
        ProjectLevelSets<N_PHI, N_DOM>(this->mesh, IBVP);
        // ============================================================

        // MAKE THE IMPLICIT MESH =====================================
        MakeImplicitMesh<N_PHI, N_DOM>(this->mesh, IBVP, im_flag);
        // ============================================================

        // EVAL THE LINEAR OPERATORS FOR THE IMPLICIT MESH ============
        EvalMassMatrices<N_PHI, N_DOM>(this->mesh, this->matfactory, IBVP);

        if (this->limiter_is_active)
        {
            EvalAverageOperator<N_PHI, N_DOM>(this->mesh, this->matfactory, IBVP);
        }
        // ============================================================

        // PROJECT THE INITIAL CONDITIONS OVER THE IMPLICIT MESH ======
        ProjectInitialConditions<N_PHI, N_DOM, N_U>(this->mesh,
                                                    this->matfactory,
                                                    this->X_new,
                                                    IBVP);

        if (this->limiter_is_active)
        {
            const Real t = 0.0;
            FillInvalidCells<N_PHI, N_DOM, N_U>(t, this->mesh, this->matfactory, this->X_new, IBVP);

            if (this->matfactory.cube.Np != 1)
            {
                ApplySlopeLimiter_DG<N_PHI, N_DOM, N_U>(t, this->mesh, this->matfactory, this->X_new, IBVP);
            }
        }
        // ============================================================
    }
    // ################################################################

    // COMPUTE THE TIME INCREMENT #####################################
    template<typename IBVP_CLASS>
    Real Compute_dt(const Real t, const IBVP_CLASS & IBVP)
    {
        // VARIABLES ==================================================
        Real dt;
        // ============================================================

        // ============================================================
        if (this->isPDEsSystemHyperbolic() && this->isTimeMarchingRungeKutta())
        {
            dt = Compute_dt_ExplicitHyperbolic<N_PHI, N_DOM, N_U>(t,
                                                                  this->mesh,
                                                                  this->matfactory,
                                                                  this->X_new,
                                                                  this->tags,
                                                                  IBVP);

            // CFL CONDITION
            dt *= this->CFL_base/(2.0*this->matfactory.cube.p+1.0);
        }
        else
        {
            Print() << "ERROR: AMReX_DG.H - DG::Compute_dt" << std::endl;
            Print() << "| Unexpected type of PDEs               : " << this->PDEs_info << std::endl;
            Print() << "| and/or unexpected time-stepping scheme: " << this->time_marching_info << std::endl;
            exit(-1);
        }
        // ============================================================

        return dt;
    }
    // ################################################################

    // TAKE TIME STEP #################################################
    template<typename IBVP_CLASS>
    void TakeTimeStep(const Real & dt, const Real & t,
                      const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        const int n_comp = this->matfactory.cube.Np*N_U;
        // ============================================================

        // COPY NEW TO OLD ============================================
        MultiFab::Copy(this->X_old, this->X_new, 0, 0, n_comp, this->dom_data_ngr);
        // ============================================================

        // CALL SELECTED TIME-STEPPING SCHEME =========================
        if ((this->isPDEsSystemHyperbolic()) && (this->isTimeMarchingRungeKutta()))
        {
            if (this->limiter_is_active)
            {
                this->RK.TakeTimeStepWithSlopeLimiter(this->tNp, dt, t,
                                                      this->mesh,
                                                      this->matfactory,
                                                      this->X_old,
                                                      this->X_new,
                                                      this->PDEs_info,
                                                      IBVP);
            }
            else
            {
                this->RK.TakeTimeStep(this->tNp, dt, t,
                                      this->mesh,
                                      this->matfactory,
                                      this->X_old,
                                      this->X_new,
                                      this->PDEs_info,
                                      IBVP);
            }
        }
        else
        {
            Print() << "ERROR: AMReX_DG.H - DG::TakeTimeStep" << std::endl;
            Print() << "| Unexpected time-stepping scheme: " << this->time_marching_info << std::endl;
            exit(-1);
        }
        // ============================================================

        // CHECK SOLUTION =============================================
        if (this->X_new.contains_nan())
        {
            Print() << "ERROR: AMReX_DG.H - DG::TakeTimeStep" << std::endl;
            Print() << "| Solution contains nans." << std::endl;
            exit(-1);
        }
        // ============================================================
    }
    // ################################################################

    // SET OUTPUT INFORMATION #########################################
    template<typename IBVP_CLASS>
    void SetOutput(const std::string & dst_folder,
                   const IBVP_CLASS & IBVP)
    {
        // PARAMETERS =================================================
        // GRID
        const GpuArray<Real, AMREX_SPACEDIM> dx = this->mesh.dx;
        const GpuArray<Real, AMREX_SPACEDIM> prob_lo = this->mesh.prob_lo;
        const Box & domain = this->mesh.geom.Domain();
        // ============================================================

        // STORE THE DESTINATION FOLDER ===============================
        this->output_folder = dst_folder;
        // ============================================================

        // SET THE OUTPUT FIELDS ======================================
        IBVP.F_SOL_DESCRIPTION(this->fields_domain, this->fields_name);

        // CONSISTENCY CHECK
        if (this->fields_domain.size() != this->fields_name.size())
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG.H - DG::SetOutput" << std::endl;
            Print() << "| The size of this->fields_domain is not equal to the size of this->fields_name:" << std::endl;
            Print() << "| this->fields_domain.size() = " << this->fields_domain.size() << std::endl;
            Print() << "| this->fields_name.size() = " << this->fields_name.size() << std::endl;
            Print() << std::endl;
            exit(-1);
        }
        // ============================================================

        // SET THE OUTPUT POINT FIELDS ================================
        IBVP.F_POINT_SOL_DESCRIPTION(this->point_fields_location,
                                     this->point_fields_domain,
                                     this->point_fields_name);

        this->print_point_fields = (this->point_fields_location.size() > 0);

        // CONSISTENCY CHECK
        if (this->point_fields_location.size() != this->point_fields_domain.size())
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG.H - DG::SetOutput" << std::endl;
            Print() << "| The size of this->point_fields_location is not equal to the size of this->point_fields_domain:" << std::endl;
            Print() << "| this->point_fields_location.size() = " << this->point_fields_location.size() << std::endl;
            Print() << "| this->point_fields_domain.size() = " << this->point_fields_domain.size() << std::endl;
            Print() << std::endl;
            exit(-1);
        }

        if (this->point_fields_location.size() != this->point_fields_name.size())
        {
            Print() << std::endl;
            Print() << "ERROR: AMReX_DG.H - DG::SetOutput" << std::endl;
            Print() << "| The size of this->point_fields_location is not equal to the size of this->point_fields_name:" << std::endl;
            Print() << "| this->point_fields_location.size() = " << this->point_fields_location.size() << std::endl;
            Print() << "| this->point_fields_name.size() = " << this->point_fields_name.size() << std::endl;
            Print() << std::endl;
            exit(-1);
        }

        // FIND THE CELLS THAT CONTAIN THE POINT LOCATIONS
        const int n_point_locations = this->point_fields_location.size();
        this->point_fields_ijk.resize(n_point_locations);

        for (int p = 0; p < n_point_locations; ++p)
        {
            this->point_fields_ijk[p] = {0, 0, 0};

            AMREX_D_TERM
            (
                this->point_fields_ijk[p][0] = (int) std::floor((this->point_fields_location[p][0]-prob_lo[0])/dx[0]);,
                this->point_fields_ijk[p][1] = (int) std::floor((this->point_fields_location[p][1]-prob_lo[1])/dx[1]);,
                this->point_fields_ijk[p][2] = (int) std::floor((this->point_fields_location[p][2]-prob_lo[2])/dx[2]);
            )

            AMREX_D_TERM
            (
                this->point_fields_ijk[p][0] = std::max(domain.smallEnd(0), std::min(domain.bigEnd(0), this->point_fields_ijk[p][0]));,
                this->point_fields_ijk[p][1] = std::max(domain.smallEnd(1), std::min(domain.bigEnd(1), this->point_fields_ijk[p][1]));,
                this->point_fields_ijk[p][2] = std::max(domain.smallEnd(2), std::min(domain.bigEnd(2), this->point_fields_ijk[p][2]));
            )
        }

        // STORE WHICH RANK WILL TAKE CARE OF EACH POINT LOCATION
        this->point_fields_rank.resize(n_point_locations);

        for (int p = 0; p < n_point_locations; ++p)
        {
            this->point_fields_rank[p] = -1;
            
            for (MFIter mfi(this->X_new); mfi.isValid(); ++mfi)
            {
                const Box & bx = mfi.validbox();
                const Dim3 lo = lbound(bx);
                const Dim3 hi = ubound(bx);

                for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    if ((i == this->point_fields_ijk[p][0]) &&
                        (j == this->point_fields_ijk[p][1]) &&
                        (k == this->point_fields_ijk[p][2]))
                    {
                        this->point_fields_rank[p] = ParallelDescriptor::MyProc();
                    }
                }
            }
        }
        // ============================================================
    }
    // ################################################################

    // PRINT POINT SOLUTION AS A FUNCTION OF TIME #####################
    template<typename IBVP_CLASS>
    void PrintPointSolution(const std::string & filename,
                            const int n,
                            const Real time,
                            const IBVP_CLASS & IBVP) const
    {
        // PARAMETERS =================================================
        const int n_point_locations = this->point_fields_location.size();
        // ============================================================

        // QUICK RETURN ===============================================
        if (!this->print_point_fields) return;
        // ============================================================
        
        // CREATE THE FILES IF n == 0 =================================
        if (n == 0)
        {
            for (int p = 0; p < n_point_locations; ++p)
            {
                if (this->point_fields_rank[p] == ParallelDescriptor::MyProc())
                {
                    const std::string filepath = MakePath({this->output_folder, filename+"_loc_"+std::to_string(p)+".txt"});
                    
                    std::ofstream fp;
                    fp.open(filepath.c_str(), std::ofstream::out);
                    if (!fp.good())
                    {
                        FileOpenFailed(filepath);
                    }

                    fp << "Point solution at location: " << AMREX_D_TERM(this->point_fields_location[p][0] << " " <<,
                                                                         this->point_fields_location[p][1] << " " <<,
                                                                         this->point_fields_location[p][2] <<) "\n";

                    const int n_fields = this->point_fields_name[p].size();
                    fp << "t ";
                    for (int f = 0; f < (n_fields-1); ++f)
                    {
                        fp << this->point_fields_name[p][f]+" ";
                    }
                    fp << this->point_fields_name[p][n_fields-1]+"\n";

                    fp.close();
                }
            }
        }
        // ============================================================

        // CYCLE OVER THE QUERY POINTS ================================
Print() << "HERE WE ARE - dG::PrintPointSolution" << std::endl;
exit(-1);
        // ============================================================
    }
    // ################################################################

    // EXPORT SOLUTION ################################################
    template<typename IBVP_CLASS>
    void ExportMesh(const std::string & filename_root,
                    const int n,
                    const Real time,
                    const IBVP_CLASS & IBVP)
    {
        ExportMesh_VTK<N_PHI, N_DOM>(this->output_folder, filename_root,
                                     n, this->n_time_steps, time,
                                     this->mesh,
                                     this->tags,
                                     IBVP);
    }

    template<typename IBVP_CLASS>
    void ExportSolution(const std::string & filename_root,
                        const int n,
                        const Real time,
                        const IBVP_CLASS & IBVP)
    {
        ExportSolution_VTK<N_PHI, N_DOM, N_U>(this->output_folder, filename_root,
                                              n, this->n_time_steps, time,
                                              this->mesh,
                                              this->matfactory,
                                              this->X_new,
                                              this->tags,
                                              this->fields_domain,
                                              this->fields_name,
                                              IBVP);
    }
    // ################################################################
};
// ####################################################################
// ####################################################################

} // namespace DG
} // namespace amrex

#endif