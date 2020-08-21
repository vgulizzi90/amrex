/* ELASTIC WAVE EQUATION */

#include <IBVP_WavesInSolids_Base_VE.H>

// IBVP PARAMETERS ####################################################
// NUMBER OF LEVEL SET FUNCTIONS/DOMAINS/UNKNOWN FIELDS ---------------
#define N_PHI 1
#define N_DOM 2
#define N_U (2*N_VE)
// --------------------------------------------------------------------
// ####################################################################



// IBVP CLASS #########################################################
class ELASTIC_WAVES
:
public ELASTIC_WAVES_BASE<N_DOM>
{
public:
    // DATA MEMBERS ===================================================
    amrex::Real ICs_un[AMREX_SPACEDIM];
    amrex::Real ICs_w;
    amrex::Real ICs_v[N_VE];
    // ================================================================

public:
    // CONSTRUCTOR ====================================================
    ELASTIC_WAVES(const amrex::Vector<std::string> & material_type,
                  const amrex::Vector<amrex::Vector<amrex::Real>> material_properties)
    :
    ELASTIC_WAVES_BASE(material_type, material_properties)
    {
        // PARAMETERS -------------------------------------------------
        // ------------------------------------------------------------

        // VARIABLES --------------------------------------------------
        // ------------------------------------------------------------

        // DIRECTION OF WAVE PROPAGATION ------------------------------
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
        // ------------------------------------------------------------

        // EIGEN STATE INITIAL CONDITIONS -----------------------------
        amrex::Real inv_rho = 1.0/this->rho[0];
        amrex::Real * cij = this->cSE[0];
        amrex::Real AMREX_D_DECL(n1 = this->ICs_un[0], n2 = this->ICs_un[1], n3 = this->ICs_un[2]);
        
        amrex::Real InTc[N_V*N_E], In[N_E*N_V];
        amrex::Real An[N_VE*N_VE], An_w[N_VE], An_v[N_VE*N_VE];

        std::fill(InTc, InTc+N_V*N_E, 0.0);
        std::fill(In, In+N_E*N_V, 0.0);
        std::fill(An, An+N_VE*N_VE, 0.0);
        std::fill(An_w, An+N_VE, 0.0);
        std::fill(An_v, An_v+N_VE*N_VE, 0.0);

#if (AMREX_SPACEDIM == 2)
        // InT*cSE
        InTc[0+0*N_V] = cij[C11]*n1+cij[C61]*n2;
        InTc[1+0*N_V] = cij[C61]*n1+cij[C21]*n2;

        InTc[0+1*N_V] = cij[C21]*n1+cij[C62]*n2;
        InTc[1+1*N_V] = cij[C62]*n1+cij[C22]*n2;

        InTc[0+2*N_V] = cij[C61]*n1+cij[C66]*n2;
        InTc[1+2*N_V] = cij[C66]*n1+cij[C62]*n2;

        // In
        In[0+0*N_E] = n1;
        In[1+1*N_E] = n2;
        In[2+0*N_E] = n2; In[2+1*N_E] = n1;
#endif
#if (AMREX_SPACEDIM == 3)
        // InT*cSE
        InTc[0+0*N_V] = cij[C11]*n1+cij[C61]*n2+cij[C51]*n3;
        InTc[1+0*N_V] = cij[C61]*n1+cij[C21]*n2+cij[C41]*n3;
        InTc[2+0*N_V] = cij[C51]*n1+cij[C41]*n2+cij[C31]*n3;

        InTc[0+1*N_V] = cij[C21]*n1+cij[C62]*n2+cij[C52]*n3;
        InTc[1+1*N_V] = cij[C62]*n1+cij[C22]*n2+cij[C42]*n3;
        InTc[2+1*N_V] = cij[C52]*n1+cij[C42]*n2+cij[C32]*n3;

        InTc[0+2*N_V] = cij[C31]*n1+cij[C63]*n2+cij[C53]*n3;
        InTc[1+2*N_V] = cij[C63]*n1+cij[C32]*n2+cij[C43]*n3;
        InTc[2+2*N_V] = cij[C53]*n1+cij[C43]*n2+cij[C33]*n3;

        InTc[0+3*N_V] = cij[C41]*n1+cij[C64]*n2+cij[C54]*n3;
        InTc[1+3*N_V] = cij[C64]*n1+cij[C42]*n2+cij[C44]*n3;
        InTc[2+3*N_V] = cij[C54]*n1+cij[C44]*n2+cij[C43]*n3;

        InTc[0+4*N_V] = cij[C51]*n1+cij[C65]*n2+cij[C55]*n3;
        InTc[1+4*N_V] = cij[C65]*n1+cij[C52]*n2+cij[C54]*n3;
        InTc[2+4*N_V] = cij[C55]*n1+cij[C54]*n2+cij[C53]*n3;

        InTc[0+5*N_V] = cij[C61]*n1+cij[C66]*n2+cij[C65]*n3;
        InTc[1+5*N_V] = cij[C66]*n1+cij[C62]*n2+cij[C64]*n3;
        InTc[2+5*N_V] = cij[C65]*n1+cij[C64]*n2+cij[C63]*n3;

        // In
        In[0+0*N_E] = n1;
        In[1+1*N_E] = n2;
        In[2+2*N_E] = n3;
        In[3+1*N_E] = n3; In[3+2*N_E] = n2;
        In[4+0*N_E] = n3; In[4+2*N_E] = n1;
        In[5+0*N_E] = n2; In[5+1*N_E] = n1;
#endif

        for (int c = N_V; c < N_VE; ++c)
        for (int r = 0; r < N_V; ++r)
        {
            An[r+c*N_VE] = -inv_rho*InTc[r+(c-N_V)*N_V];
        }
        for (int c = 0; c < N_V; ++c)
        for (int r = N_V; r < N_VE; ++r)
        {
            An[r+c*N_VE] = -In[r-N_V+c*N_E];
        }

        // EIGENVALUES AND EIGENVECTORS
        amrex::Real wIm[N_VE];
        
        amrex::DG_utils::eig(false, true, N_VE, An, An_w, wIm, nullptr, An_v);
        // ------------------------------------------------------------

        // SELECT ONE WAVE --------------------------------------------
        const int iv = std::distance(An_w, std::max_element(An_w, An_w+N_VE));

        this->ICs_w = An_w[iv];
        std::copy(&An_v[iv*N_VE], &An_v[(iv+1)*N_VE], this->ICs_v);
        // ------------------------------------------------------------

amrex::Print() << "iv: " << iv << std::endl;
amrex::Print() << "N_VE: " << N_VE << std::endl;
amrex::Print() << "this->ICs_w: " << this->ICs_w << std::endl;
for (int r = 0; r < N_VE; ++r)
{
amrex::Print() << "this->ICs_v[" << r << "]: " << this->ICs_v[r] << std::endl;
}
    }
    // ================================================================

    // LEVEL SET FUNCTION =============================================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_PHI(const int & ph,
                      const amrex::Real & t, const amrex::Real * x) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real x1 = x[0];
        const amrex::Real x2 = x[1];
        const amrex::Real phi = 0.125-std::cos(2.0*M_PI*x1)*std::cos(2.0*M_PI*x2);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real x1 = x[0];
        const amrex::Real x2 = x[1];
        const amrex::Real x3 = x[2];
        const amrex::Real phi = 0.125+std::cos(2.0*M_PI*x1)*std::cos(2.0*M_PI*x2)*std::cos(2.0*M_PI*x3);
#endif
        return -phi;
    }
    // ================================================================

    // RELATIONSHIP AMONG LEVEL SET FUNCTIONS AND DOMAINS =============
    AMREX_GPU_HOST_DEVICE
    void F_DOM2PHI(const int & dom, int * phi_info) const
    {
        const int PHI_INFO[2*N_DOM] =
        {
            // dom = 0
            0, -1,

            // dom = 1
            0, +1
        };

        phi_info[0] = PHI_INFO[2*dom];
        phi_info[1] = PHI_INFO[2*dom+1];
    }

    amrex::Real F_EXACT_VOLUME(const int & dom) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real val = 0.3867888308923567;
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            1.0-val,

            // dom = 1
            val
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real val = 0.3867888308923567;
        const amrex::Real EXACT_VOLUME[N_DOM] =
        {
            // dom = 0
            1.0-val,

            // dom = 1
            val
        };
#endif

        return EXACT_VOLUME[dom];
    }
    amrex::Real F_EXACT_SURFACE(const int & dom) const
    {
#if (AMREX_SPACEDIM == 2)
        const amrex::Real val = 3.201999313783625;
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            val,

            // dom = 1
            val
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real EXACT_SURFACE[N_DOM] =
        {
            // dom = 0
            2.0,

            // dom = 1
            2.0
        };
#endif

        return EXACT_SURFACE[dom];
    }
    // ================================================================

    // RELATIONSHIP AMONG UNKNOWN FIELDS AND DOMAINS ==================
    AMREX_GPU_HOST_DEVICE
    int F_U2DOM(const int & u) const
    {
#if (AMREX_SPACEDIM == 2)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0,  0, 0, 0,

            // dom = 1
            1, 1,  1, 1, 1
        };
#endif
#if (AMREX_SPACEDIM == 3)
        const int DOM[N_U] =
        {
            // dom = 0
            0, 0, 0,  0, 0, 0, 0, 0, 0,

            // dom = 1
            1, 1, 1,  1, 1, 1, 1, 1, 1
        };
#endif
        return DOM[u];
    }
    // ================================================================

    // RELATIONSHIP BETWEEN NEIGHBORING DOMAINS =======================
    AMREX_GPU_HOST_DEVICE
    int F_DOM2NBRDOM(const int& dom) const
    {
        const int NBRDOM[N_DOM] =
        {
            // dom = 0
            1,

            // dom = 1
            0
        };
        return NBRDOM[dom];
    }
    // ================================================================

    // INITIAL CONDITIONS: UNKNOWN FIELDS =============================
    AMREX_GPU_HOST_DEVICE
    amrex::Real F_U0(const int & u, const amrex::Real * x) const
    {
        const amrex::Real * un = this->ICs_un;
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real f0 = std::sin(2.0*M_PI*x_dot_un);
        const int uu = (u < N_VE) ? u : (u-N_VE);

        amrex::Real res;

        res = this->ICs_v[uu]*f0;

        return res;
    }
    // ================================================================

    // EXACT SOLUTION/ERROR EVALUATION ================================
    AMREX_GPU_HOST_DEVICE
    void F_EXACT(const amrex::Real & t, const amrex::Real * x,
                 amrex::Real * Ue) const
    {
        const amrex::Real * un = this->ICs_un;
        const amrex::Real x_dot_un = AMREX_D_PICK(x[0]*un[0], x[0]*un[0]+x[1]*un[1], x[0]*un[0]+x[1]*un[1]+x[2]*un[2]);
        const amrex::Real w = this->ICs_w;
        const amrex::Real f0 = std::sin(2.0*M_PI*(x_dot_un-w*t));

        for (int u = 0; u < N_VE; ++u)
        {
            Ue[u] = this->ICs_v[u]*f0;
        }
    }

    AMREX_GPU_HOST_DEVICE
    void F_ERROR(const amrex::Real & t, const amrex::Real * x,
                 const amrex::Real * U,
                 amrex::Real & err_x,
                 amrex::Real & norm_x) const
    {
        // PARAMETERS
        const int dom = 0;

        // EXACT SOLUTION
        amrex::Real Ue[N_VE];
        
        this->F_EXACT(t, x, Ue);

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY
        amrex::Real V[N_VE];
        AMREX_D_TERM
        (
            V[V1] = this->rho[dom]*Ue[V1];,
            V[V2] = this->rho[dom]*Ue[V2];,
            V[V3] = this->rho[dom]*Ue[V3];
        )
        this->_eval_S_(dom, &Ue[N_V], &V[N_V]);

        // NORMALIZATION QUANTITY
        norm_x = 0.0;
        for (int u = 0; u < N_VE; ++u)
        {
            norm_x += Ue[u]*V[u];
        }

        // DETERMINE WHICH DOMAIN WE ARE IN
        const amrex::Real phi = this->F_PHI(0, t, x);

        // DG SOLUTION
        const int pos = (phi < 0.0) ? 0 : N_VE;

        for (int u = 0; u < N_VE; ++u)
        {
            Ue[u] -= U[u+pos];
        }

        // AUXILIARY VECTOR USED FOR CONSTRUCTING THE TOTAL ENERGY OF
        // THE DG SOLUTION
        AMREX_D_TERM
        (
            V[0] = this->rho[dom]*Ue[0];,
            V[1] = this->rho[dom]*Ue[1];,
            V[2] = this->rho[dom]*Ue[2];
        )
        this->_eval_S_(dom, &Ue[N_V], &V[N_V]);

        // ERROR
        err_x = 0.0;
        for (int u = 0; u < N_VE; ++u)
        {
            err_x += Ue[u]*V[u];
        }
    }
    // ================================================================

    // DG-RELATED AUXILIARY FUNCTIONS: NUMERICAL FLUXES ===============
    // INTRAPHASE -----------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_ICS(const int dom,
                  const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U, const amrex::Real * nbr_U,
                  amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        const amrex::Real * Ud = &U[dom*N_VE], * nbr_Ud = &nbr_U[dom*N_VE];
        amrex::Real * NFnd = &NFn[dom*N_VE];

        amrex::Real avg_U[N_VE];

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
    // ----------------------------------------------------------------

    // GRID BOUNDARIES ------------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_BCS(const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                  const amrex::Real * U,
                  amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_BCS (We should not end up here)");
    }
    // ----------------------------------------------------------------

    // INTERNAL BOUNDARIES --------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_BCS(const int & dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U,
                      amrex::Real * NFn) const
    {
amrex::Abort("Hello! F_NF_PHI_BCS (We should not end up here)");
    }
    // ----------------------------------------------------------------

    // INTERNAL INTERFACE ---------------------------------------------
    AMREX_GPU_HOST_DEVICE
    void F_NF_PHI_ICS(const int dom, const int nbr_dom,
                      const amrex::Real & t, const amrex::Real * x, const amrex::Real * un,
                      const amrex::Real * U, const amrex::Real * nbr_U,
                      amrex::Real * NFn) const
    {
        const amrex::Real mu = this->max_wave_speed;

        const amrex::Real * Ud = &U[dom*N_VE], * nbr_Ud = &nbr_U[nbr_dom*N_VE];
        amrex::Real * NFnd = &NFn[dom*N_VE];

        amrex::Real avg_U[N_VE];

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

        this->_eval_AnU_(dom, un, avg_U, NFnd);

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
    // ----------------------------------------------------------------
    // ================================================================

    // ADDITIONAL FUNCTIONS FOR POST-PROCESSING =======================
    void F_POINT_SOL_DESCRIPTION(amrex::Vector<amrex::Array<amrex::Real, AMREX_SPACEDIM>> & point_fields_location,
                                 amrex::Vector<int> & point_fields_domain,
                                 amrex::Vector<amrex::Vector<std::string>> & point_fields_name) const
    {
        point_fields_location.clear();
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_POINT_SOL(const int p,
                     const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
                     const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
                     amrex::Real * F) const
    {
amrex::Abort("Hello! F_POINT_SOL (We should not end up here)");
    }

    void F_SOL_DESCRIPTION(amrex::Vector<int> & fields_domain, amrex::Vector<std::string> & fields_name) const
    {
#if (AMREX_SPACEDIM == 2)
        fields_domain = {0, 0,  0, 0, 0,
                         1, 1,  1, 1, 1,
                         0, 0,  0, 0, 0,
                         1, 1,  1, 1, 1};
        fields_name = {"Vx_a", "Vy_a",  "Sxx_a", "Syy_a", "Sxy_a",
                       "Vx_b", "Vy_b",  "Sxx_b", "Syy_b", "Sxy_b",
                       "err_Vx_a", "err_Vy_a",  "err_Sxx_a", "err_Syy_a", "err_Sxy_a",
                       "err_Vx_b", "err_Vy_b",  "err_Sxx_b", "err_Syy_b", "err_Sxy_b"};
#endif
#if (AMREX_SPACEDIM == 3)
        fields_domain = {0, 0, 0,  0, 0, 0, 0, 0, 0,
                         1, 1, 1,  1, 1, 1, 1, 1, 1,
                         0, 0, 0,  0, 0, 0, 0, 0, 0,
                         1, 1, 1,  1, 1, 1, 1, 1, 1};
        fields_name = {"Vx_a", "Vy_a", "Vz_a",  "Sxx_a", "Syy_a", "Szz_a", "Syz_a", "Sxz_a", "Sxy_a",
                       "Vx_b", "Vy_b", "Vz_b",  "Sxx_b", "Syy_b", "Szz_b", "Syz_b", "Sxz_b", "Sxy_b",
                       "err_Vx_a", "err_Vy_a", "err_Vz_a",  "err_Sxx_a", "err_Syy_a", "err_Szz_a", "err_Syz_a", "err_Sxz_a", "err_Sxy_a",
                       "err_Vx_b", "err_Vy_b", "err_Vz_b",  "err_Sxx_b", "err_Syy_b", "err_Szz_b", "err_Syz_b", "err_Sxz_b", "err_Sxy_b"};
#endif
    }
    
    AMREX_GPU_HOST_DEVICE
    void F_SOL(const amrex::Real & t, const amrex::Real * x,
               const amrex::Real * PHI, AMREX_D_DECL(const amrex::Real * dPHIdx1, const amrex::Real * dPHIdx2, const amrex::Real * dPHIdx3),
               const amrex::Real * U, AMREX_D_DECL(const amrex::Real * dUdx1, const amrex::Real * dUdx2, const amrex::Real * dUdx3),
               amrex::Real * F) const
    {
        // STRESS COMPONENTS
        amrex::Real S[N_E], Se[N_E];

        // EXACT SOLUTION
        amrex::Real Ue[N_VE];
        
        this->F_EXACT(t, x, Ue);
        this->_eval_S_(0, &Ue[N_V], Se);

#if (AMREX_SPACEDIM == 2)
        // DOMAIN 0
        this->_eval_S_(0, &U[N_V], S);
        
        F[0] = U[0];
        F[1] = U[1];
        F[2] = S[0];
        F[3] = S[1];
        F[4] = S[2];

        F[0+N_VE+N_VE] = U[0]-Ue[0];
        F[1+N_VE+N_VE] = U[1]-Ue[1];
        F[2+N_VE+N_VE] = S[0]-Se[0];
        F[3+N_VE+N_VE] = S[1]-Se[1];
        F[4+N_VE+N_VE] = S[2]-Se[2];

        // DOMAIN 1
        this->_eval_S_(1, &U[N_V+N_VE], S);

        F[0+N_VE] = U[0+N_VE];
        F[1+N_VE] = U[1+N_VE];
        F[2+N_VE] = S[0];
        F[3+N_VE] = S[1];
        F[4+N_VE] = S[2];

        F[0+N_VE+N_VE+N_VE] = U[0+N_VE]-Ue[0];
        F[1+N_VE+N_VE+N_VE] = U[1+N_VE]-Ue[1];
        F[2+N_VE+N_VE+N_VE] = S[0]-Se[0];
        F[3+N_VE+N_VE+N_VE] = S[1]-Se[1];
        F[4+N_VE+N_VE+N_VE] = S[2]-Se[2];
#endif
#if (AMREX_SPACEDIM == 3)
#endif
    }
    // ================================================================

};