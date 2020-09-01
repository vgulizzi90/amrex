/* ELASTIC WAVES IN SOLIDS: VELOCITY-STRESS FORMULATION */
/* HP CONVERGENCE ANALYSIS */

#include <IBVP_Levelsets.H>
#include <IBVP_Base.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF LEVEL SET FUNCTIONS/DOMAINS/UNKNOWN FIELDS ---------------
#define N_PHI 1
#define N_DOM 1
#define N_U (N_DOM*N_VS)
// --------------------------------------------------------------------

// EXACT SOLUTION
#define EIGEN_STATE 0
#define NAVIER_SOLUTION 1
#define EXACT_SOLUTION 1

// SOME PARAMETERS ----------------------------------------------------
#define R 0.25
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
class ELASTIC_WAVES
:
public ELASTIC_WAVES_BASE_VS<N_DOM>
{
public:
    // DATA MEMBERS ===================================================
    amrex::Real ICs_un[AMREX_SPACEDIM];
    amrex::Real ICs_w;
    amrex::Real ICs_v[N_VS];

    amrex::Real Navier_mui[AMREX_SPACEDIM];
    amrex::Real Navier_w;
    amrex::Real Navier_vRe[N_VS], Navier_vIm[N_VS];
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const amrex::Vector<std::string> & material_type,
                  const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    ELASTIC_WAVES_BASE_VS(material_type, material_properties)
    {
        // EIGEN STATE INITIAL CONDITIONS -----------------------------
        // DIRECTION OF WAVE PROPAGATION
        AMREX_D_TERM
        (
            this->ICs_un[0] = 1.0;,
            this->ICs_un[1] = 0.0;,
            this->ICs_un[2] = 0.0;
        )
        {
            amrex::Real tmp = 0.0;
            AMREX_D_TERM
            (
                tmp += this->ICs_un[0]*this->ICs_un[0];,
                tmp += this->ICs_un[1]*this->ICs_un[1];,
                tmp += this->ICs_un[2]*this->ICs_un[2];
            )
            tmp = 1.0/std::sqrt(tmp);
            AMREX_D_TERM
            (
                this->ICs_un[0] *= tmp;,
                this->ICs_un[1] *= tmp;,
                this->ICs_un[2] *= tmp;
            )
        }

        {
            const amrex::Real inv_rho = 1.0/this->rho[0];
            const amrex::Real * cij = this->cSE[0];
            const amrex::Real AMREX_D_DECL(n1 = this->ICs_un[0], n2 = this->ICs_un[1], n3 = this->ICs_un[2]);
            
            amrex::Real InT[N_V*N_S], cIn[N_S*N_V];
            amrex::Real An[N_VS*N_VS], An_w[N_VS], An_v[N_VS*N_VS];

            std::fill(InT, InT+N_V*N_S, 0.0);
            std::fill(cIn, cIn+N_S*N_V, 0.0);
            std::fill(An, An+N_VS*N_VS, 0.0);
            std::fill(An_w, An_w+N_VS, 0.0);
            std::fill(An_v, An_v+N_VS*N_VS, 0.0);
        
#if (AMREX_SPACEDIM == 2)
            // InT
            InT[0+0*N_V] =  n1; InT[0+1*N_V] = 0.0; InT[0+2*N_V] = n2;
            InT[1+0*N_V] = 0.0; InT[1+1*N_V] =  n2; InT[1+2*N_V] = n1;

            // cSE*In
            cIn[0+0*N_E] = cij[C11]*n1+cij[C61]*n2; cIn[0+1*N_E] = cij[C61]*n1+cij[C21]*n2;
            cIn[1+0*N_E] = cij[C21]*n1+cij[C62]*n2; cIn[1+1*N_E] = cij[C62]*n1+cij[C22]*n2;
            cIn[2+0*N_E] = cij[C61]*n1+cij[C66]*n2; cIn[2+1*N_E] = cij[C66]*n1+cij[C62]*n2;
#endif
#if (AMREX_SPACEDIM == 3)
            // InT
            InT[0+0*N_V] =  n1; InT[0+1*N_V] = 0.0; InT[0+2*N_V] = 0.0; InT[0+3*N_V] = 0.0; InT[0+4*N_V] =  n3; InT[0+5*N_V] =  n2;
            InT[1+0*N_V] = 0.0; InT[1+1*N_V] =  n2; InT[1+2*N_V] = 0.0; InT[1+3*N_V] =  n3; InT[1+4*N_V] = 0.0; InT[1+5*N_V] =  n1;
            InT[2+0*N_V] = 0.0; InT[2+1*N_V] = 0.0; InT[2+2*N_V] =  n3; InT[2+3*N_V] =  n2; InT[2+4*N_V] =  n1; InT[2+5*N_V] = 0.0;

            // cSE*In
            cIn[0+0*N_E] = cij[C11]*n1+cij[C61]*n2+cij[C51]*n3; cIn[0+1*N_E] = cij[C61]*n1+cij[C21]*n2+cij[C41]*n3; cIn[0+2*N_E] = cij[C51]*n1+cij[C41]*n2+cij[C31]*n3;
            cIn[1+0*N_E] = cij[C21]*n1+cij[C62]*n2+cij[C52]*n3; cIn[1+1*N_E] = cij[C62]*n1+cij[C22]*n2+cij[C42]*n3; cIn[1+2*N_E] = cij[C52]*n1+cij[C42]*n2+cij[C32]*n3;
            cIn[2+0*N_E] = cij[C31]*n1+cij[C63]*n2+cij[C53]*n3; cIn[2+1*N_E] = cij[C63]*n1+cij[C32]*n2+cij[C43]*n3; cIn[2+2*N_E] = cij[C53]*n1+cij[C43]*n2+cij[C33]*n3;
            cIn[3+0*N_E] = cij[C41]*n1+cij[C64]*n2+cij[C54]*n3; cIn[3+1*N_E] = cij[C64]*n1+cij[C42]*n2+cij[C44]*n3; cIn[3+2*N_E] = cij[C54]*n1+cij[C44]*n2+cij[C43]*n3;
            cIn[4+0*N_E] = cij[C51]*n1+cij[C65]*n2+cij[C55]*n3; cIn[4+1*N_E] = cij[C65]*n1+cij[C52]*n2+cij[C54]*n3; cIn[4+2*N_E] = cij[C55]*n1+cij[C54]*n2+cij[C53]*n3;
            cIn[5+0*N_E] = cij[C61]*n1+cij[C66]*n2+cij[C65]*n3; cIn[5+1*N_E] = cij[C66]*n1+cij[C62]*n2+cij[C64]*n3; cIn[5+2*N_E] = cij[C65]*n1+cij[C64]*n2+cij[C63]*n3;
#endif

            for (int c = N_V; c < N_VS; ++c)
            for (int r = 0; r < N_V; ++r)
            {
                An[r+c*N_VS] = -inv_rho*InT[r+(c-N_V)*N_V];
            }
            for (int c = 0; c < N_V; ++c)
            for (int r = N_V; r < N_VS; ++r)
            {
                An[r+c*N_VS] = -cIn[r-N_V+c*N_E];
            }

            // EIGENVALUES AND EIGENVECTORS
            amrex::Real wIm[N_VS];
            
            amrex::DG_utils::eig(false, true, N_VS, An, An_w, wIm, nullptr, An_v);

            // SELECT ONE WAVE
            const int iv = std::distance(An_w, std::max_element(An_w, An_w+N_VS));

            this->ICs_w = An_w[iv];
            std::copy(&An_v[iv*N_VS], &An_v[(iv+1)*N_VS], this->ICs_v);
        }
        // ------------------------------------------------------------

        // NAVIER SOLUTION INITIAL CONDITIONS -------------------------
        // Hypotheses:
        // - The geometry must be an L1xL2(xL3) rectangle;
        // - The material can be at most orthotropic and must have its
        //   principal axes aligned with the geometry reference system.
        // ------------------------------------------------------------
        {
            // L1, L2, L3 must match the dimensions of the geometry
            const amrex::Real AMREX_D_DECL(L1 = 1.0, L2 = 1.0, L3 = 1.0);
            // Number of waves
            const int AMREX_D_DECL(m1 = 1, m2 = 1, m3 = 1);
            const amrex::Real AMREX_D_DECL(mu1 = M_PI*m1/L1, mu2 = M_PI*m2/L2, mu3 = M_PI*m3/L3);

            const amrex::Real inv_rho = 1.0/this->rho[0];
            const amrex::Real * cij = this->cSE[0];

            amrex::Real A[N_VS*N_VS];

            std::fill(A, A+N_VS*N_VS, 0.0);
#if (AMREX_SPACEDIM == 2)
            A[0+2*N_VS] = -inv_rho*mu1; A[0+4*N_VS] = inv_rho*mu2;
            A[1+3*N_VS] = -inv_rho*mu2; A[1+4*N_VS] = inv_rho*mu1;
            A[2+0*N_VS] =  cij[C11]*mu1; A[2+1*N_VS] =  cij[C21]*mu2;
            A[3+0*N_VS] =  cij[C21]*mu1; A[3+1*N_VS] =  cij[C22]*mu2;
            A[4+0*N_VS] = -cij[C66]*mu2; A[4+1*N_VS] = -cij[C66]*mu1;
#endif
#if (AMREX_SPACEDIM == 3)
            
#endif
            // EIGENVALUES AND EIGENVECTORS
            amrex::Real A_wRe[N_VS], A_wIm[N_VS], A_v[N_VS*N_VS];

            amrex::DG_utils::eig(false, true, N_VS, A, A_wRe, A_wIm, nullptr, A_v);

            // SELECT ONE EIGENVALUE/EIGENVECTOR
            const int iv = std::distance(A_wIm, std::max_element(A_wIm, A_wIm+N_VS));

            AMREX_D_TERM
            (
                this->Navier_mui[0] = mu1;,
                this->Navier_mui[1] = mu2;,
                this->Navier_mui[2] = mu3;
            )
            this->Navier_w = A_wIm[iv];
            for (int u = 0; u < N_VS; ++u)
            {
                this->Navier_vRe[u] = A_v[iv*N_VS+u];
                this->Navier_vIm[u] = A_v[(iv+1)*N_VS+u];
            }
        }
        // ------------------------------------------------------------
    }
    // ================================================================

    // LEVEL SET FUNCTION =============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
#if (N_DOM == 1)
        const amrex::Real O[AMREX_SPACEDIM] = {AMREX_D_DECL(0.5, 0.5, 0.5)};
        return -F_PHI_SPHERE(t, x, R, O);
        //return -1.0;
#endif
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
#if (N_DOM == 1)
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1
        };
#endif

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }

    amrex::Real F_EXACT_VOLUME(const int & dom) const
    {
#if (N_DOM == 1)
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            1.0-AMREX_D_PICK(2.0*R, M_PI*R*R, (4.0/3.0)*M_PI*R*R*R)
        };
#endif
        return EXACT_VOLUME[dom];
    }

    amrex::Real F_EXACT_SURFACE(const int & dom) const
    {
#if (N_DOM == 1)
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            AMREX_D_PICK(1.0, 2.0*M_PI*R, 4.0*M_PI*R*R)
        };
#endif
        return EXACT_SURFACE[dom];
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
#if (N_DOM == 1)
#if (AMREX_SPACEDIM == 2)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0,  0, 0, 0
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0, 0,  0, 0, 0, 0, 0, 0
        };
#endif
#endif
        return DOM[u];
    }
    // ================================================================

    // RELATIONSHIP BETWEEN NEIGHBORING DOMAINS =======================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {

#if (N_DOM == 1)
        const int NBRDOM[N_DOM] =
        {
            // dom = 0
            -1
        };
#endif
#if (N_DOM == 2)
        const int NBRDOM[N_DOM] =
        {
            // dom = 0
            1,

            // dom = 1
            0
        };
#endif
        return NBRDOM[dom];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int u, const amrex::Real * x) const
    {
        // PARAMETERS
        const int uu = (u < N_VS) ? u : (u-N_VS);

        // EXACT SOLUTION
        amrex::Real Ue[N_VS];
        
        this->F_EXACT(0.0, x, Ue);

        return Ue[uu];
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * Ue) const
    {
#if (EXACT_SOLUTION == EIGEN_STATE)
        const amrex::Real * un = this->ICs_un;
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real w = this->ICs_w;
        const amrex::Real f0 = std::sin(2.0*M_PI*(x_dot_un-w*t));

        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] = this->ICs_v[u]*f0;
        }
#endif

#if (EXACT_SOLUTION == NAVIER_SOLUTION)
        const amrex::Real * vRe = this->Navier_vRe, * vIm = this->Navier_vIm;
        const amrex::Real w = this->Navier_w;
        const amrex::Real AMREX_D_DECL(mu1 = this->Navier_mui[0], mu2 = this->Navier_mui[1], mu3 = this->Navier_mui[2]);
        const amrex::Real cx1 = std::cos(mu1*x[0]), sx1 = std::sin(mu1*x[0]);
        const amrex::Real cx2 = std::cos(mu2*x[1]), sx2 = std::sin(mu2*x[1]);
#if (AMREX_SPACEDIM == 3)
        const amrex::Real cx3 = std::cos(mu3*x[2]), sx3 = std::sin(mu3*x[2]);
#endif
        const amrex::Real ct = std::cos(w*t), st = std::sin(w*t);

#if (AMREX_SPACEDIM == 2)
        Ue[V1]  = (vRe[V1]*ct+vIm[V1]*st)*cx1*sx2;
        Ue[V2]  = (vRe[V2]*ct+vIm[V2]*st)*sx1*cx2;
        Ue[S11] = (vRe[S11]*ct+vIm[S11]*st)*sx1*sx2;
        Ue[S22] = (vRe[S22]*ct+vIm[S22]*st)*sx1*sx2;
        Ue[S12] = (vRe[S12]*ct+vIm[S12]*st)*cx1*cx2;
#endif
#if (AMREX_SPACEDIM == 3)
#endif
#endif

    }

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const int dom,
                 const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
        // PARAMETERS
        const amrex::Real * Ud = &U[dom*N_VS];

        // EXACT SOLUTION
        amrex::Real Ue[N_VS];
        
        this->F_EXACT(t, x, Ue);

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY
        amrex::Real V[N_VS];
        AMREX_D_TERM
        (
            V[V1] = this->rho[dom]*Ue[V1];,
            V[V2] = this->rho[dom]*Ue[V2];,
            V[V3] = this->rho[dom]*Ue[V3];
        )
        this->_eval_E_(dom, &Ue[N_V], &V[N_V]);

        // NORMALIZATION QUANTITY
        norm_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            norm_x += Ue[u]*V[u];
        }

        // DG SOLUTION
        for (int u = 0; u < N_VS; ++u)
        {
            Ue[u] -= Ud[u];
        }

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY OF
        // THE DG SOLUTION
        AMREX_D_TERM
        (
            V[0] = this->rho[dom]*Ue[0];,
            V[1] = this->rho[dom]*Ue[1];,
            V[2] = this->rho[dom]*Ue[2];
        )
        this->_eval_E_(dom, &Ue[N_V], &V[N_V]);

        // ERROR
        err_x = 0.0;
        for (int u = 0; u < N_VS; ++u)
        {
            err_x += Ue[u]*V[u];
        }
    }
    // ================================================================

    // NUMERICAL FLUXES: INTRAPHASE ===================================
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->max_wave_speed;
        const amrex::Real * Ud = &U[dom*N_VS], * nbr_Ud = &nbr_U[dom*N_VS];

        // VARIABLES
        amrex::Real avg_U[N_VS];
        amrex::Real * NFnd = &NFn[dom*N_VS];

        AMREX_D_TERM
        (
            avg_U[0] = 0.5*(Ud[0]+nbr_Ud[0]);
            avg_U[1] = 0.5*(Ud[1]+nbr_Ud[1]);,
            avg_U[2] = 0.5*(Ud[2]+nbr_Ud[2]);
            avg_U[3] = 0.5*(Ud[3]+nbr_Ud[3]);
            avg_U[4] = 0.5*(Ud[4]+nbr_Ud[4]);,
            avg_U[5] = 0.5*(Ud[5]+nbr_Ud[5]);
            avg_U[6] = 0.5*(Ud[6]+nbr_Ud[6]);
            avg_U[7] = 0.5*(Ud[7]+nbr_Ud[7]);
            avg_U[8] = 0.5*(Ud[8]+nbr_Ud[8]);
        )

        if (std::abs(un[0]) > 0.5)
        {
            this->_eval_F1_(dom, avg_U, NFnd);

            AMREX_D_TERM
            (
                NFnd[0] *= un[0];
                NFnd[1] *= un[0];,
                NFnd[2] *= un[0];
                NFnd[3] *= un[0];
                NFnd[4] *= un[0];,
                NFnd[5] *= un[0];
                NFnd[6] *= un[0];
                NFnd[7] *= un[0];
                NFnd[8] *= un[0];
            )
        }
#if (AMREX_SPACEDIM > 1)
        else if (std::abs(un[1]) > 0.5)
        {
            this->_eval_F2_(dom, avg_U, NFnd);

            AMREX_D_TERM
            (
                NFnd[0] *= un[1];
                NFnd[1] *= un[1];,
                NFnd[2] *= un[1];
                NFnd[3] *= un[1];
                NFnd[4] *= un[1];,
                NFnd[5] *= un[1];
                NFnd[6] *= un[1];
                NFnd[7] *= un[1];
                NFnd[8] *= un[1];
            )
        }
#endif
#if (AMREX_SPACEDIM > 2)
        else if (std::abs(un[2]) > 0.5)
        {
            this->_eval_F3_(dom, avg_U, NFnd);

            AMREX_D_TERM
            (
                NFnd[0] *= un[2];
                NFnd[1] *= un[2];,
                NFnd[2] *= un[2];
                NFnd[3] *= un[2];
                NFnd[4] *= un[2];,
                NFnd[5] *= un[2];
                NFnd[6] *= un[2];
                NFnd[7] *= un[2];
                NFnd[8] *= un[2];
            )
        }
#endif

        AMREX_D_TERM
        (
            NFnd[0] += 0.5*mu*(Ud[0]-nbr_Ud[0]);
            NFnd[1] += 0.5*mu*(Ud[1]-nbr_Ud[1]);,
            NFnd[2] += 0.5*mu*(Ud[2]-nbr_Ud[2]);
            NFnd[3] += 0.5*mu*(Ud[3]-nbr_Ud[3]);
            NFnd[4] += 0.5*mu*(Ud[4]-nbr_Ud[4]);,
            NFnd[5] += 0.5*mu*(Ud[5]-nbr_Ud[5]);
            NFnd[6] += 0.5*mu*(Ud[6]-nbr_Ud[6]);
            NFnd[7] += 0.5*mu*(Ud[7]-nbr_Ud[7]);
            NFnd[8] += 0.5*mu*(Ud[8]-nbr_Ud[8]);
        )
    }
    // ================================================================

    // NUMERICAL FLUXES: GRID BOUNDARIES ==============================
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const int & dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->max_wave_speed;
        const amrex::Real * Ud = &U[dom*N_VS];

        // VARIABLES
        amrex::Real Ue[N_VS];
        amrex::Real bcs_U[N_VS], avg_U[N_VS];
        amrex::Real * NFnd = &NFn[dom*N_VS];

        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

        if (std::abs(un[0]) > 0.5)
        {
            bcs_U[0] = Ue[0];
            bcs_U[1] = Ue[1];
            bcs_U[2] = Ud[2];
            bcs_U[3] = Ud[3];
            bcs_U[4] = Ud[4];

            this->_eval_F1_(dom, bcs_U, NFnd);

            NFnd[0] *= un[0];
            NFnd[1] *= un[0];
            NFnd[2] *= un[0];
            NFnd[3] *= un[0];
            NFnd[4] *= un[0];

            NFnd[0] += mu*(Ud[0]-bcs_U[0]);
            NFnd[1] += mu*(Ud[1]-bcs_U[1]);
        }
        else if (std::abs(un[1]) > 0.5)
        {
            bcs_U[0] = Ue[0];
            bcs_U[1] = Ue[1];
            bcs_U[2] = Ud[2];
            bcs_U[3] = Ud[3];
            bcs_U[4] = Ud[4];

            this->_eval_F2_(dom, bcs_U, NFnd);

            NFnd[0] *= un[1];
            NFnd[1] *= un[1];
            NFnd[2] *= un[1];
            NFnd[3] *= un[1];
            NFnd[4] *= un[1];

            NFnd[0] += mu*(Ud[0]-bcs_U[0]);
            NFnd[1] += mu*(Ud[1]-bcs_U[1]);
        }
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL BOUNDARIES ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
        // PARAMETERS
        const amrex::Real mu = this->max_wave_speed;
        const amrex::Real * Ud = &U[dom*N_VS];

        // VARIABLES
        amrex::Real Ue[N_VS];
        amrex::Real avg_U[N_VS];
        amrex::Real * NFnd = &NFn[dom*N_VS];

        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

        AMREX_D_TERM
        (
            avg_U[0] = 0.5*(Ud[0]+Ue[0]);
            avg_U[1] = 0.5*(Ud[1]+Ue[1]);,
            avg_U[2] = 0.5*(Ud[2]+Ue[2]);
            avg_U[3] = 0.5*(Ud[3]+Ue[3]);
            avg_U[4] = 0.5*(Ud[4]+Ue[4]);,
            avg_U[5] = 0.5*(Ud[5]+Ue[5]);
            avg_U[6] = 0.5*(Ud[6]+Ue[6]);
            avg_U[7] = 0.5*(Ud[7]+Ue[7]);
            avg_U[8] = 0.5*(Ud[8]+Ue[8]);
        )

        this->_eval_AnU_(dom, un, avg_U, NFnd);

        AMREX_D_TERM
        (
            avg_U[0] += 0.5*mu*(Ud[0]-Ue[0]);
            avg_U[1] += 0.5*mu*(Ud[1]-Ue[1]);,
            avg_U[2] += 0.5*mu*(Ud[2]-Ue[2]);
            avg_U[3] += 0.5*mu*(Ud[3]-Ue[3]);
            avg_U[4] += 0.5*mu*(Ud[4]-Ue[4]);,
            avg_U[5] += 0.5*mu*(Ud[5]-Ue[5]);
            avg_U[6] += 0.5*mu*(Ud[6]-Ue[6]);
            avg_U[7] += 0.5*mu*(Ud[7]-Ue[7]);
            avg_U[8] += 0.5*mu*(Ud[8]-Ue[8]);
        )
    }
    // ================================================================

    // NUMERICAL FLUXES: INTERNAL INTERFACE ==========================
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int dom, const int nbr_dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_PHI_ICS");
    }
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
#if (N_DOM == 1)
        fields_domain = {0, 0,  0, 0, 0,
                         0, 0,  0, 0, 0};
        fields_name = {"Vx", "Vy",  "Sxx", "Syy", "Sxy",
                       "err_Vx", "err_Vy",  "err_Sxx", "err_Syy", "err_Sxy"};
#endif
#if (N_DOM == 2)
        fields_domain = {0, 0,  0, 0, 0,
                         0, 0,  0, 0, 0,
                         1, 1,  1, 1, 1,
                         1, 1,  1, 1, 1};
        fields_name = {"Vx_a", "Vy_a",  "Sxx_a", "Syy_a", "Sxy_a",
                       "err_Vx_a", "err_Vy_a",  "err_Sxx_a", "err_Syy_a", "err_Sxy_a",
                       "Vx_b", "Vy_b",  "Sxx_b", "Syy_b", "Sxy_b",
                       "err_Vx_b", "err_Vy_b",  "err_Sxx_b", "err_Syy_b", "err_Sxy_b"};
#endif
#endif

#if (AMREX_SPACEDIM == 3)
#if (N_DOM == 1)
        fields_domain = {0, 0, 0,  0, 0, 0, 0, 0, 0,
                         0, 0, 0,  0, 0, 0, 0, 0, 0};
        fields_name = {"Vx", "Vy", "Vz",  "Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy",
                       "err_Vx", "err_Vy", "err_Vz",  "err_Sxx", "err_Syy", "err_Szz", "err_Syz", "err_Sxz", "err_Sxy"};
#endif
#if (N_DOM == 2)
        fields_domain = {0, 0, 0,  0, 0, 0, 0, 0, 0,
                         0, 0, 0,  0, 0, 0, 0, 0, 0,
                         1, 1, 1,  1, 1, 1, 1, 1, 1,
                         1, 1, 1,  1, 1, 1, 1, 1, 1};
        fields_name = {"Vx_a", "Vy_a", "Vz_a",  "Sxx_a", "Syy_a", "Szz_a", "Syz_a", "Sxz_a", "Sxy_a",
                       "err_Vx_a", "err_Vy_a", "err_Vz_a",  "err_Sxx_a", "err_Syy_a", "err_Szz_a", "err_Syz_a", "err_Sxz_a", "err_Sxy_a",
                       "Vx_b", "Vy_b", "Vz_b",  "Sxx_b", "Syy_b", "Szz_b", "Syz_b", "Sxz_b", "Sxy_b",
                       "err_Vx_b", "err_Vy_b", "err_Vz_b",  "err_Sxx_b", "err_Syy_b", "err_Szz_b", "err_Syz_b", "err_Sxz_b", "err_Sxy_b"};
#endif
#endif
    }

    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // VARIABLES
        amrex::Real Ue[N_VS];
        
        // EXACT SOLUTION
        this->F_EXACT(t, x, Ue);

#if (AMREX_SPACEDIM == 2)
        // DOMAIN 0
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];

        F[0+N_VS] = U[0]-Ue[0];
        F[1+N_VS] = U[1]-Ue[1];
        F[2+N_VS] = U[2]-Ue[2];
        F[3+N_VS] = U[3]-Ue[3];
        F[4+N_VS] = U[4]-Ue[4];
#endif

#if (AMREX_SPACEDIM == 3)
        // DOMAIN 0
        F[0] = U[0];
        F[1] = U[1];
        F[2] = U[2];
        F[3] = U[3];
        F[4] = U[4];
        F[5] = U[5];
        F[6] = U[6];
        F[7] = U[7];
        F[8] = U[8];

        F[0+N_VS] = U[0]-Ue[0];
        F[1+N_VS] = U[1]-Ue[1];
        F[2+N_VS] = U[2]-Ue[2];
        F[3+N_VS] = U[3]-Ue[3];
        F[4+N_VS] = U[4]-Ue[4];
        F[5+N_VS] = U[5]-Ue[5];
        F[6+N_VS] = U[6]-Ue[6];
        F[7+N_VS] = U[7]-Ue[7];
        F[8+N_VS] = U[8]-Ue[8];
#endif
    }
    // ================================================================
};
// ####################################################################