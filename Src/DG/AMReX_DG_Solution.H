// AMReX_DG_Solution.H

#ifndef BL_DG_SOLUTION_H_
#define BL_DG_SOLUTION_H_

#include <AMReX_DG_Base.H>
#include <AMReX_DG_ImplicitMesh.H>
#include <AMReX_DG_MatrixFactory.H>

namespace amrex
{
namespace DG
{

// PROJECT THE LEVEL SET FUNCTIONS OVER A GRID ########################
template<int N_PHI, int N_DOM, typename IBVP_CLASS>
void ProjectLevelSets(ImplicitMesh<N_PHI, N_DOM> & mesh, const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectLevelSets(ImplicitMesh<N_PHI, N_DOM> &, const IBVP_CLASS &)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;

    // STANDARD ELEMENT
    const int sNp = mesh.cube.Np;
    const int dom_Nq = mesh.cube.dom_Nq;
    AMREX_D_TERM
    (
        Real const * dom_XI1q_ptr = mesh.cube.dom_XI1q.data();,
        Real const * dom_XI2q_ptr = mesh.cube.dom_XI2q.data();,
        Real const * dom_XI3q_ptr = mesh.cube.dom_XI3q.data();
    )
    Real const * dom_Wq_ptr = mesh.cube.dom_Wq.data();
    Real const * dom_BF_ptr = mesh.cube.dom_BF.data();
    Real const * iMM_ptr = mesh.cube.iMM.data();
    // ================================================================

    // INITIALIZATION =================================================
    mesh.PHI = 0.0;
    // ================================================================

    // EVAL THE PROJECTION ============================================
    for (MFIter mfi(mesh.PHI); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        Array4<Real> const & PHI_fab = mesh.PHI.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            
            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM];
            Real integrand;

            // PROJECTION
            for (int q = 0; q < dom_Nq; ++q)
            {
                // Quadrature point
                AMREX_D_TERM
                (
                    x[0] = J[0]*dom_XI1q_ptr[q]+cell_m[0];,
                    x[1] = J[1]*dom_XI2q_ptr[q]+cell_m[1];,
                    x[2] = J[2]*dom_XI3q_ptr[q]+cell_m[2];
                )

                // Eval the integral
                for (int ru = 0; ru < N_PHI; ++ru)
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = 0.0;
                    for (int cs = 0; cs < sNp; ++cs)
                    {
                        integrand += iMM_ptr[rs+cs*sNp]*dom_BF_ptr[q+cs*dom_Nq];
                    }
                    integrand *= IBVP.F_PHI(ru, 0.0, x);

                    // Sum integral contribution
                    PHI_fab(i,j,k,rs+ru*sNp) += integrand*dom_Wq_ptr[q];
                }
            }
        });
        Gpu::synchronize();
    }
    mesh.PHI.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// COMPUTE THE IMPLICIT MESH ##########################################
template<int N_PHI, int N_DOM, typename IBVP_CLASS>
void MakeImplicitMesh(ImplicitMesh<N_PHI, N_DOM> & mesh, const IBVP_CLASS & IBVP, const bool im_flag = true)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("MakeImplicitMesh(ImplicitMesh<N_PHI, N_DOM> &, const IBVP_CLASS &, const bool im_flag)");
    // ----------------------------------------------------------------

#if (AMREX_SPACEDIM == 1)
    Print() << std::endl;
    Print() << "ERROR: AMReX_DG_Solution.H - MakeImplicitMesh" << std::endl;
    Print() << "| The method EvalImplicitMesh must be suitably modified to handle AMREX_SPACEDIM = 1." << std::endl;
    Print() << std::endl;
    exit(-1);
#endif

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;
    const GpuArray<Real, AMREX_SPACEDIM> bou_aJ = mesh.bou_aJ;
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.is_periodic;
    const Real atol = mesh.atol;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    const UnitCube<AMREX_SPACEDIM> & cube = mesh.cube;

    // VOLUME FRACTION THRESHOLD
    const Real vft = mesh.volume_fraction_threshold;

    // IMPLICIT MESH
    const int p = cube.p;
    const int qo = mesh.space_q_im+1;
    // ================================================================

    // VARIABLES ======================================================
    long eDomQuad_mem_pos, eBouQuad_mem_pos, eIntBouQuad_mem_pos, eQuery_mem_pos;
    // ================================================================

    // INITIALIZATION =================================================
    eDomQuad_mem_pos = 0L;
    eBouQuad_mem_pos = 0L;
    eIntBouQuad_mem_pos = 0L;
    eQuery_mem_pos = 0L;

    mesh.n_cut_elms = 0;
    mesh.n_cut_elm_bou = 0;
    mesh.n_entire_extended_elms = 0;
    // ================================================================

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM INFO ===========================
    // DOMAIN: QUADRATURE POINTS
    {
        // QUADRATURE INFO
        const int dom_Nq = cube.dom_Nq;

        // Resize memory
        mesh.eDomQuad_host_mem.resize((AMREX_SPACEDIM+1)*dom_Nq);

        // Store data
        for (int qi = 0; qi < dom_Nq; ++qi)
        {
            AMREX_D_TERM
            (
                mesh.eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+0] = cube.dom_XI1q[qi];,
                mesh.eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+1] = cube.dom_XI2q[qi];,
                mesh.eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+2] = cube.dom_XI3q[qi];
            )
            mesh.eDomQuad_host_mem[(AMREX_SPACEDIM+1)*qi+AMREX_SPACEDIM] = cube.dom_Wq[qi];
        }

        // Update memory (nodes coordinates + weights)
        eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
    }

    // BOUNDARIES
    {
        // QUADRATURE INFO
        const int bou_Nq = cube.bou_Nq;

        // Resize memory
        mesh.eBouQuad_host_mem.resize((AMREX_SPACEDIM+1)*bou_Nq*DG_CELL_N_SPACE_BOUNDARIES);

        // Store data
        for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
        for (int qi = 0; qi < bou_Nq; ++qi)
        {
            AMREX_D_TERM
            (
                mesh.eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+0] = cube.bou_XI1q[bou_Nq*b+qi];,
                mesh.eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+1] = cube.bou_XI2q[bou_Nq*b+qi];,
                mesh.eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+2] = cube.bou_XI3q[bou_Nq*b+qi];
            )
            mesh.eBouQuad_host_mem[(AMREX_SPACEDIM+1)*bou_Nq*b+(AMREX_SPACEDIM+1)*qi+AMREX_SPACEDIM] = cube.bou_Wq[qi];
        }

        // Update memory (nodes coordinates + weights)
        eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq*DG_CELL_N_SPACE_BOUNDARIES;
    }

    // DOMAIN: QUERY POINTS
    {
        // GRID INFO
        const int dom_Ng = cube.dom_Ng;

        // Resize memory
        mesh.eQuery_host_mem.resize(AMREX_SPACEDIM*dom_Ng);

        // Store data
        for (int qi = 0; qi < dom_Ng; ++qi)
        {
            AMREX_D_TERM
            (
                mesh.eQuery_host_mem[AMREX_SPACEDIM*qi+0] = cube.dom_XI1g[qi];,
                mesh.eQuery_host_mem[AMREX_SPACEDIM*qi+1] = cube.dom_XI2g[qi];,
                mesh.eQuery_host_mem[AMREX_SPACEDIM*qi+2] = cube.dom_XI3g[qi];
            )
        }

        // Update memory (nodes coordinates)
        eQuery_mem_pos += AMREX_SPACEDIM*dom_Ng;
    }
    // ================================================================

    // STORE THE IMPLICIT MESH INFO: DOMAIN (CELL-CENTERED) DATA ======
    // NOTE: We are looping over the fabbox because the implicit mesh
    //       information regarding the quadrature and query points will
    //       not be shared among the processors.
    // ================================================================
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        
        Array4<short> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<int> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<int> const & eQuery_Npt_fab = mesh.eQuery_Npt.array(mfi);
        Array4<long> const & eQuery_pos_fab = mesh.eQuery_pos.array(mfi);
        Array4<int> const & eIntBouQuad_Nq_fab = mesh.eIntBouQuad_Nq.array(mfi);
        Array4<long> const & eIntBouQuad_pos_fab = mesh.eIntBouQuad_pos.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};
            const Real entire_cell_volume = AMREX_D_PICK(dx[0], dx[0]*dx[1], dx[0]*dx[1]*dx[2]);

            // LOCAL VARIABLES
            bool cell_is_empty, cell_is_entire, cell_is_cut;
            Real cell_volume, volume_fraction;
            int phi_info[2];
            algoim_DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

            // Get info about the level set function associated to the
            // domain
            IBVP.F_DOM2PHI(dom, phi_info);
            phi.set_fab_ptrs(i, j, k, phi_info[0]);
            phi.set_sign(phi_info[1]);

            // DOMAIN: QUADRATURE POINTS
            {
                // Algoim data structure
                iQuadRule quad;
                if (im_flag)
                {
                    quad = iQuadGen(phi, cell_lo, cell_hi, -1, -1, qo);

                    // Compute the volume fraction
                    cell_volume = quad([](const auto& x) {return 1.0;});
                    volume_fraction = cell_volume/entire_cell_volume;
                }
                else
                {
                    volume_fraction = 1.0;
                }

                // EMPTY ELEMENTS -------------------------------------
                if (volume_fraction < atol)
                {
                    cell_is_empty = true;
                    cell_is_entire = false;
                    cell_is_cut = false;

                    // ELEMENT TYPE
                    eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_EMPTY;
                    
                    // ELEMENT INFO
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = 0.0;
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                    // QUADRATURE INFO
                    eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = 0;
                    eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = -1L;
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (volume_fraction > (1.0-atol))
                {
                    cell_is_empty = false;
                    cell_is_entire = true;
                    cell_is_cut = false;

                    // ELEMENT INFO
                    eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_ENTIRE;
                    
                    // ELEMENT INFO
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = 1.0;
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                    // QUADRATURE INFO
                    eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = cube.dom_Nq;
                    eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = 0L;
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else
                {
                    cell_is_empty = false;
                    cell_is_entire = false;
                    cell_is_cut = true;

                    mesh.n_cut_elms += 1;
                        
                    // ELEMENT TYPE
                    eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_PARTIAL;
                    
                    // ELEMENT INFO
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom) = volume_fraction;
                    eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1) = cell_volume;

                    // QUADRATURE INFO
                    const int dom_Nq = quad.nodes.size();
                    
                    // Store memory offset
                    eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = dom_Nq;
                    eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom) = eDomQuad_mem_pos;

                    // Expand memory
                    mesh.eDomQuad_host_mem.resize(eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*dom_Nq);

                    // Store data: remember that we store the quad
                    // abscissae and weights as if they were computed
                    // on a [-1,1]^AMREX_SPACEDIM cell
                    for (int qi = 0; qi < dom_Nq; ++qi)
                    {
                        const long pos = eDomQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                        AMREX_D_TERM
                        (
                            mesh.eDomQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]-cell_m[0])/J[0];,
                            mesh.eDomQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]-cell_m[1])/J[1];,
                            mesh.eDomQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]-cell_m[2])/J[2];
                        )
                        mesh.eDomQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/aJ;
                    }

                    // Update memory (nodes coordinates + weights)
                    eDomQuad_mem_pos += (AMREX_SPACEDIM+1)*dom_Nq;
                }
                // ----------------------------------------------------
            }

            // DOMAIN: CENTROID
            {
                // EMPTY ELEMENTS -------------------------------------
                if (cell_is_empty)
                {
                    // ELEMENT INFO
                    AMREX_D_TERM
                    (
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1) = 0.0;,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2) = 0.0;,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3) = 0.0;
                    )
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (cell_is_entire)
                {
                    // STORE CENTROID
                    AMREX_D_TERM
                    (
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1) += cell_m[0];,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2) += cell_m[1];,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3) += cell_m[2];
                    )
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else
                {
                    // PARAMETERS
                    const Real V = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1);
                    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    const long pos0 = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);

                    // VARIABLES
                    Real x[AMREX_SPACEDIM], w, S[AMREX_SPACEDIM];

                    // EVAL THE STATIC MOMENTS
                    AMREX_D_TERM
                    (
                        S[0] = 0.0;,
                        S[1] = 0.0;,
                        S[2] = 0.0;
                    )
                    for (int qi = 0; qi < dom_Nq; ++qi)
                    {
                        const long pos = pos0+(AMREX_SPACEDIM+1)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = J[0]*mesh.eDomQuad_host_mem[pos+0]+cell_m[0];,
                            x[1] = J[1]*mesh.eDomQuad_host_mem[pos+1]+cell_m[1];,
                            x[2] = J[2]*mesh.eDomQuad_host_mem[pos+2]+cell_m[2];
                        )
                        w = aJ*mesh.eDomQuad_host_mem[pos+AMREX_SPACEDIM];
                        AMREX_D_TERM
                        (
                            S[0] += x[0]*w;,
                            S[1] += x[1]*w;,
                            S[2] += x[2]*w;
                        )
                    }

                    // EVAL AND STORE CENTROID
                    AMREX_D_TERM
                    (
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1) += S[0]/V;,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2) += S[1]/V;,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3) += S[2]/V;
                    )
                }
                // ----------------------------------------------------
            }

            // DOMAIN: QUERY POINTS
            {
                // EMPTY ELEMENTS -------------------------------------
                if (cell_is_empty)
                {
                    // GRID INFO
                    eQuery_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = 0;
                    eQuery_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = -1L;
                }
                // ----------------------------------------------------
                // ENTIRE ELEMENTS ------------------------------------
                else if (cell_is_entire)
                {
                    // GRID INFO
                    eQuery_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = cube.dom_Ng;
                    eQuery_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = 0L;
                }
                // ----------------------------------------------------
                // PARTIAL ELEMENTS -----------------------------------
                else
                {
                    iTriRule tri = iTriGen(phi, cell_lo, cell_hi, -1, -1, qo);

                    // GRID INFO
                    const int dom_Npt = tri.nodes.size();

                    // Store memory offset
                    eQuery_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = dom_Npt;
                    eQuery_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom) = eQuery_mem_pos;

                    // Expand memory
                    mesh.eQuery_host_mem.resize(eQuery_mem_pos+AMREX_SPACEDIM*dom_Npt);

                    // Store data: remember that we store the query
                    // points as if they were computed on a
                    // [-1,1]^AMREX_SPACEDIM cell
                    for (int qi = 0; qi < dom_Npt; ++qi)
                    {
                        const long pos = eQuery_mem_pos+AMREX_SPACEDIM*qi;

                        AMREX_D_TERM
                        (
                            mesh.eQuery_host_mem[pos+0] = (tri.nodes[qi].x[0]-cell_m[0])/J[0];,
                            mesh.eQuery_host_mem[pos+1] = (tri.nodes[qi].x[1]-cell_m[1])/J[1];,
                            mesh.eQuery_host_mem[pos+2] = (tri.nodes[qi].x[2]-cell_m[2])/J[2];
                        )
                    }

                    // Update memory (nodes coordinates)
                    eQuery_mem_pos += AMREX_SPACEDIM*dom_Npt;
                }
                // ----------------------------------------------------
            }

            // INTERNAL BOUNDARY (IF ANY)
            if (cell_is_cut)
            {
                iQuadRule quad = iQuadGen(phi, cell_lo, cell_hi, AMREX_SPACEDIM, -1, qo);

                // QUADRATURE INFO
                const int bou_Nq = quad.nodes.size();
                Real un[AMREX_SPACEDIM], tmp;

                // Store memory offset
                eIntBouQuad_Nq_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                eIntBouQuad_pos_fab(i,j,k,DG_ELM_INT_BOU_QUAD_N_COMP_PER_DOM*dom) = eIntBouQuad_mem_pos;

                // Expand memory
                mesh.eIntBouQuad_host_mem.resize(eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq);

                // Store data
                for (int qi = 0; qi < bou_Nq; ++qi)
                {
                    const long pos = eIntBouQuad_mem_pos+(AMREX_SPACEDIM+1+AMREX_SPACEDIM)*qi;

                    AMREX_D_TERM
                    (
                        mesh.eIntBouQuad_host_mem[pos+0] = quad.nodes[qi].x[0];,
                        mesh.eIntBouQuad_host_mem[pos+1] = quad.nodes[qi].x[1];,
                        mesh.eIntBouQuad_host_mem[pos+2] = quad.nodes[qi].x[2];
                    )
                    mesh.eIntBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w;

                    phi.eval_grad(&quad.nodes[qi].x[0], un);
                    tmp = 1.0/std::sqrt(AMREX_D_PICK(un[0]*un[0], un[0]*un[0]+un[1]*un[1], un[0]*un[0]+un[1]*un[1]+un[2]*un[2]));
                    AMREX_D_TERM
                    (
                        un[0] *= tmp;,
                        un[1] *= tmp;,
                        un[2] *= tmp;
                    )

                    AMREX_D_TERM
                    (
                        mesh.eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+0] = un[0];,
                        mesh.eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+1] = un[1];,
                        mesh.eIntBouQuad_host_mem[pos+AMREX_SPACEDIM+1+2] = un[2];
                    )
                }

                // Update memory (nodes coordinates + weights + unit
                // normal components)
                eIntBouQuad_mem_pos += (AMREX_SPACEDIM+1+AMREX_SPACEDIM)*bou_Nq;
            }
        }
    }
    mesh.eInfo.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // STORE MEMORY INFO ==============================================
    mesh.mem_for_eDomQuad = eDomQuad_mem_pos;
    mesh.mem_for_eIntBouQuad = eIntBouQuad_mem_pos;
    mesh.mem_for_eQuery = eQuery_mem_pos;
    // ================================================================

    // STORE THE IMPLICIT MESH INFO: BOUNDARY (FACE-CENTERED) DATA ====
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        const int b = 2*dir;
        const int flag[AMREX_SPACEDIM] = {AMREX_D_DECL((int) (dir == 0), (int) (dir == 1), (int) (dir == 2))};
        const Real delta[AMREX_SPACEDIM] = {AMREX_D_DECL(dx[0]*flag[0], dx[1]*flag[1], dx[2]*flag[2])};

        for (MFIter mfi(mesh.eBouQuad_Nq[dir]); mfi.isValid(); ++mfi)
        {
            const Box & bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);

            Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

            Array4<short> const & eBouType_fab = mesh.eBouType[dir].array(mfi);
            Array4<int> const & eBouQuad_Nq_fab = mesh.eBouQuad_Nq[dir].array(mfi);
            Array4<long> const & eBouQuad_pos_fab = mesh.eBouQuad_pos[dir].array(mfi);

            for (int dom = 0; dom < N_DOM; ++dom)
            for (int k = lo.z; k <= hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i)
            {
                // ----------------------------------------------------
                // NOTE: Please, consider that we are using the indices
                //       defined from the face-centered box to access
                //       the data stored in the cell-centered PHI_fab.
                // ----------------------------------------------------

                // LOCAL PARAMETERS
                const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                                  prob_lo[1]+(j+0.5)*dx[1],
                                                                  prob_lo[2]+(k+0.5)*dx[2])};
                const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                                   prob_lo[1]+j*dx[1],
                                                                   prob_lo[2]+k*dx[2])};
                const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                                   prob_lo[1]+(j+1)*dx[1],
                                                                   prob_lo[2]+(k+1)*dx[2])};
                const Real entire_boundary_surface = AMREX_D_PICK(1.0, dx[1], dx[1]*dx[2]);

                // LOCAL VARIABLES
                Real boundary_surface, surface_fraction;
                int phi_info[2];
                algoim_DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

                // Get info about the level set function associated to
                // the domain
                IBVP.F_DOM2PHI(dom, phi_info);
                phi.set_fab_ptrs(i, j, k, phi_info[0]);
                phi.set_sign(phi_info[1]);

                // BOUNDARY QUADRATURE
                {
                    // Algoim data structure
                    iQuadRule quad;
                    if (im_flag)
                    {
                        quad = iQuadGen(phi, cell_lo, cell_hi, b/2, b%2, qo);

                        // Compute the surface fraction
                        boundary_surface = quad([](const auto& x) {return 1.0;});
                        surface_fraction = boundary_surface/entire_boundary_surface;
                    }
                    else
                    {
                        surface_fraction = 1.0;
                    }

                    // EMPTY BOUNDARY ---------------------------------
                    if (surface_fraction < atol)
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_EMPTY;

                        // QUADRATURE INFO
                        // BOUNDARY^-
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = 0;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = -1L;

                        // BOUNDARY^+
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = 0;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = -1L;
                    }
                    // ------------------------------------------------
                    // ENTIRE BOUNDARY --------------------------------
                    else if (surface_fraction > (1.0-atol))
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_ENTIRE;

                        // QUADRATURE INFO
                        const int bou_Nq = cube.bou_Nq;
                        
                        // BOUNDARY^-
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = (AMREX_SPACEDIM+1)*bou_Nq*(b+1);

                        // BOUNDARY^+
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = (AMREX_SPACEDIM+1)*bou_Nq*b;
                    }
                    // ------------------------------------------------
                    // PARTIAL BOUNDARY -------------------------------
                    else
                    {
                        // ELEMENT BOUNDARY TYPE
                        eBouType_fab(i,j,k,DG_ELM_BOU_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_BOU_TYPE_PARTIAL;

                        // QUADRATURE INFO
                        const int bou_Nq = quad.nodes.size();

                        // BOUNDARY^-
                        // Store memory offset
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom) = eBouQuad_mem_pos;

                        // Expand memory
                        mesh.eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                        // Store data: remember that we store the quad
                        // abscissae and weights as if they were
                        // computed on a [-1,1]^AMREX_SPACEDIM cell
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            const long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                            AMREX_D_TERM
                            (
                                mesh.eBouQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]+delta[0]-cell_m[0])/J[0];,
                                mesh.eBouQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]+delta[1]-cell_m[1])/J[1];,
                                mesh.eBouQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]+delta[2]-cell_m[2])/J[2];
                            )
                            mesh.eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/bou_aJ[dir];
                        }

                        // Update memory (nodes coordinates + weights)
                        eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;

                        // BOUNDARY^+
                        // Store memory offset
                        eBouQuad_Nq_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = bou_Nq;
                        eBouQuad_pos_fab(i,j,k,DG_ELM_BOU_QUAD_N_COMP_PER_DOM*dom+1) = eBouQuad_mem_pos;

                        // Expand memory
                        mesh.eBouQuad_host_mem.resize(eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*bou_Nq);

                        // Store data: remember that we store the quad
                        // abscissae and weights as if they were
                        // computed on a [-1,1]^AMREX_SPACEDIM cell
                        for (int qi = 0; qi < bou_Nq; ++qi)
                        {
                            const long pos = eBouQuad_mem_pos+(AMREX_SPACEDIM+1)*qi;

                            AMREX_D_TERM
                            (
                                mesh.eBouQuad_host_mem[pos+0] = (quad.nodes[qi].x[0]-cell_m[0])/J[0];,
                                mesh.eBouQuad_host_mem[pos+1] = (quad.nodes[qi].x[1]-cell_m[1])/J[1];,
                                mesh.eBouQuad_host_mem[pos+2] = (quad.nodes[qi].x[2]-cell_m[2])/J[2];
                            )
                            mesh.eBouQuad_host_mem[pos+AMREX_SPACEDIM] = quad.nodes[qi].w/bou_aJ[dir];
                        }

                        // Update memory (nodes coordinates + weights)
                        eBouQuad_mem_pos += (AMREX_SPACEDIM+1)*bou_Nq;
                    }
                    // ------------------------------------------------
                }
            }
        }
    }
    // ================================================================

    // STORE MEMORY INFO ==============================================
    mesh.mem_for_eBouQuad = eBouQuad_mem_pos;
    // ================================================================

    // SEARCH FOR SMALL ELEMENTS AND SETUP THE MERGING INFO ===========
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short> const & eType_fab = mesh.eType.array(mfi);
        
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const bool elm_is_partial = (eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) == DG_ELM_TYPE_PARTIAL);
            const Real vf = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom);

            // LOCAL VARIABLES
            int nbr_i, nbr_j, nbr_k, nbr_b;
            Real nbr_vf, best_nbr_vf;
            bool elm_has_been_merged, nbr_is_valid;

            // CONDITIONS FOR SMALL ELEMENTS
            if (elm_is_partial && (vf < vft))
            {
                elm_has_been_merged = false;
                best_nbr_vf = vft;

                // Search among the neighbors. We currently limit the
                // search to the neighboring cells that share a 
                // boundary with the current cell.
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    // Neighboring cell id and boundary
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    nbr_is_valid = true;
                    nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom);

                    if ((nbr_vf > best_nbr_vf) && nbr_is_valid)
                    {
                        eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_SMALL+b*10;
                        elm_has_been_merged = true;
                        best_nbr_vf = nbr_vf;
                    }
                }

                if (elm_has_been_merged)
                {
                    // I would like to mark an element as extended
                    // without falling into any raise condition. The
                    // conflict can be raised because an element can be
                    // extended to multiple small elements.
                }
                else
                {
                    printf("\n");
                    printf("ERROR: AMReX_DG_Solution.H - MakeImplicitMesh\n");
                    printf("| Could not find a neighboring cell suitable for merging:\n");
                    printf("| dom: %d\n", dom);
                    printf("| i,j,k: %d,%d,%d\n", i, j, k);
                    printf("| Volume fraction: %f\n", vf);
                    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                    {
                        // Neighboring cell id and boundary
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        nbr_vf = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom);
                        printf("| Volume fraction (nbr = %d): %f\n", b, nbr_vf);
                    }
                    printf("\n");
                    Abort("");
                }
            }
            // ELSE THE ELEMENT IS LARGE
            else if (elm_is_partial && (vf >= vft))
            {
                eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = DG_ELM_TYPE_LARGE;
            }
        });
        Gpu::synchronize();
    }
    mesh.eType.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // UPDATE THE CENTROIDS OF THE EXTENDED ELEMENTS ==================
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real> const & eInfo_fab = mesh.eInfo.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;

            // Cycle over the boundaries, check whether the valid
            // element is also extended and mark it
            if (elm_is_valid)
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            {
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                if (nbr_merged_b == nbr_b)
                {
                    const Real small_V = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1);
                    const Real V = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1);
                    
                    Real small_S[AMREX_SPACEDIM], S[AMREX_SPACEDIM];

                    AMREX_D_TERM
                    (
                        small_S[0] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1)*small_V;,
                        small_S[1] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2)*small_V;,
                        small_S[2] = eInfo_fab(nbr_i,nbr_j,nbr_k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3)*small_V;
                    )
                    AMREX_D_TERM
                    (
                        S[0] = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1)*V;,
                        S[1] = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2)*V;,
                        S[2] = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3)*V;
                    )
                    AMREX_D_TERM
                    (
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+1) = (S[0]+small_S[0])/(V+small_V);,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+2) = (S[1]+small_S[1])/(V+small_V);,
                        eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom+1+3) = (S[2]+small_S[2])/(V+small_V);
                    )
                }
            }
        });
        Gpu::synchronize();
    }
    mesh.eInfo.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // MARK THE EXTENDED ELEMENTS =====================================
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short> const & eType_fab = mesh.eType.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;

            // Cycle over the boundaries, check whether the valid
            // element is also extended and mark it
            if (elm_is_valid)
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            {
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                if (nbr_merged_b == nbr_b)
                {
                    eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom) = etype+10;
                }
            }
        });
        Gpu::synchronize();
    }
    mesh.eType.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // COUNT THE NUMBER OF ENTIRE EXTENDED ELEMENTS ===================
    mesh.n_entire_extended_elms = 0;

    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

            if (elm_is_entire && elm_is_extended) mesh.n_entire_extended_elms += 1;
        }
    }
    // ================================================================

    // COLOR THE IMPLICITLY-DEFINED MESH ELEMENTS =====================
    mesh.eColor = DG_ELM_COLOR_UNDEFINED;

    /*
    ParallelDescriptor::Barrier();
    {
        const int n_ranks = ParallelDescriptor::NProcs();

        for (int rank = 0; rank < n_ranks; ++rank)
        {
            if (ParallelDescriptor::MyProc() == rank)
            {
                for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
                {
                    const Box & bx = mfi.validbox();
                    const Dim3 lo = lbound(bx);
                    const Dim3 hi = ubound(bx);

                    Array4<short const> const & eType_fab = mesh.eType.array(mfi);
                    Array4<int> const & eColor_fab = mesh.eColor.array(mfi);

                    for (int dom = 0; dom < N_DOM; ++dom)
                    for (int k = lo.z; k <= hi.z; ++k)
                    for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        // LOCAL PARAMETERS
                        const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);
                        const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
                        const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                        const bool elm_is_valid = elm_is_entire || elm_is_large;

                        // LIST OF NEIGHBORS' COLORS
                        Vector<int> nbr_colors;

                        // LOCAL VARIABLES
                        int nbr_i, nbr_j, nbr_k, nbr_b, nbr_BF_i, nbr_BF_j, nbr_BF_k;

                        if (elm_is_valid)
                        {
                            // LOOP OVER THE NEIGHBORS
                            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                            {
                                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                                // NEIGHBOR VARIABLES
                                const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                                const int nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                                const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                                const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                                const bool cells_are_merged = (nbr_merged_b == nbr_b);

                                if (nbr_is_empty) continue;

                                // LOOP OVER THE NEIGHBORS OF THE
                                // SMALL CELL
                                if (cells_are_merged)
                                {
                                    for (int bb = 0; bb < DG_CELL_N_SPACE_BOUNDARIES; ++bb)
                                    {
                                        int nbr_nbr_i, nbr_nbr_j, nbr_nbr_k, nbr_nbr_b, nbr_nbr_BF_i, nbr_nbr_BF_j, nbr_nbr_BF_k;
                                        NBR_CELL(nbr_i, nbr_j, nbr_k, bb, nbr_nbr_i, nbr_nbr_j, nbr_nbr_k, nbr_nbr_b);

                                        if (nbr_nbr_b == b) continue;
                                        if ((nbr_nbr_i < (lo.x-1)) || (nbr_nbr_i > (hi.x+1)) ||
                                            (nbr_nbr_j < (lo.y-1)) || (nbr_nbr_j > (hi.y+1)) ||
                                            (nbr_nbr_k < (lo.z-1)) || (nbr_nbr_k > (hi.z+1))) continue;

                                        const short nbr_nbr_etype = eType_fab(nbr_nbr_i,nbr_nbr_j,nbr_nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                                        
                                        // GET THE COLOR OF THE NEIGHBOR
                                        BF_CELL(nbr_nbr_i, nbr_nbr_j, nbr_nbr_k, nbr_nbr_etype, nbr_nbr_BF_i, nbr_nbr_BF_j, nbr_nbr_BF_k);
                                        const int nbr_nbr_color = eColor_fab(nbr_nbr_BF_i,nbr_nbr_BF_j,nbr_nbr_BF_k,DG_ELM_COLOR_N_COMP_PER_DOM*dom);

                                        if ((nbr_nbr_color != DG_ELM_COLOR_UNDEFINED) &&
                                            (std::count(nbr_colors.begin(), nbr_colors.end(), nbr_nbr_color) == 0))
                                        {
                                            nbr_colors.push_back(nbr_nbr_color);
                                        }
                                    }

                                    // IF THERE IS AN INTERFACE,
                                    // CONSIDER THE COLOR OF THE
                                    // ELEMENT ON THE OTHER SIDE OF THE
                                    // INTERFACE
                                    if (nbr_dom != -1)
                                    {
                                        const short nbr_nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);

                                        int nbr_nbr_BF_i, nbr_nbr_BF_j, nbr_nbr_BF_k;
                                        
                                        // GET THE COLOR OF THE NEIGHBOR
                                        BF_CELL(nbr_i, nbr_j, nbr_k, nbr_nbr_etype, nbr_nbr_BF_i, nbr_nbr_BF_j, nbr_nbr_BF_k);
                                        const int nbr_nbr_color = eColor_fab(nbr_nbr_BF_i,nbr_nbr_BF_j,nbr_nbr_BF_k,DG_ELM_COLOR_N_COMP_PER_DOM*nbr_dom);

                                        if ((nbr_nbr_color != DG_ELM_COLOR_UNDEFINED) &&
                                            (std::count(nbr_colors.begin(), nbr_colors.end(), nbr_nbr_color) == 0))
                                        {
                                            nbr_colors.push_back(nbr_nbr_color);
                                        }
                                    }
                                }
                                else
                                {
                                    // GET THE COLOR OF THE NEIGHBOR
                                    BF_CELL(nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                                    const int nbr_color = eColor_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG_ELM_COLOR_N_COMP_PER_DOM*dom);

                                    if ((nbr_color != DG_ELM_COLOR_UNDEFINED) &&
                                        (std::count(nbr_colors.begin(), nbr_colors.end(), nbr_color) == 0))
                                    {
                                        nbr_colors.push_back(nbr_color);
                                    }
                                }
                            }

                            // IF THERE IS AN INTERFACE, CONSIDER THE
                            // COLOR OF THE ELEMENT ON THE OTHER SIDE
                            // OF THE INTERFACE
                            if (elm_is_large && (nbr_dom != -1))
                            {
                                const short nbr_etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*nbr_dom);

                                // GET THE COLOR OF THE NEIGHBOR
                                BF_CELL(i, j, k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k);
                                const int nbr_color = eColor_fab(nbr_BF_i,nbr_BF_j,nbr_BF_k,DG_ELM_COLOR_N_COMP_PER_DOM*nbr_dom);

                                if ((nbr_color != DG_ELM_COLOR_UNDEFINED) &&
                                    (std::count(nbr_colors.begin(), nbr_colors.end(), nbr_color) == 0))
                                {
                                    nbr_colors.push_back(nbr_color);
                                }
                            }

                            // SET THE ELEMENT COLOR
                            if (nbr_colors.size() > 0)
                            {
                                const int n_colors = nbr_colors.size();
                                bool found;
                                int c;

                                found = false;
                                c = 0;
                                while (!found && (c < n_colors))
                                {
                                    if (std::count(nbr_colors.begin(), nbr_colors.end(), c) == 0)
                                    {
                                        found = true;
                                    }
                                    else
                                    {
                                        c += 1;
                                    }
                                }

                                eColor_fab(i,j,k,dom) = c;
                            }
                            else
                            {
                                eColor_fab(i,j,k,dom) = 0;
                            }
                        }
                    }

                    mesh.eColor.FillBoundary(mesh.geom.periodicity());
                }
            }

            ParallelDescriptor::Barrier();
        }
    }
    ParallelDescriptor::Barrier();

    mesh.eColor.FillBoundary(mesh.geom.periodicity());
    */
    // ================================================================
    // PROBLEM
    // I am sure there is a better way to color the mesh.
    // ================================================================

    // COPY MEMORY TO THE DEVICE ======================================
#ifdef AMREX_USE_CUDA
    mesh.eDomQuad_device_mem.resize(mesh.eDomQuad_host_mem.size());
    mesh.eBouQuad_device_mem.resize(mesh.eBouQuad_host_mem.size());
    mesh.eIntBouQuad_device_mem.resize(mesh.eIntBouQuad_host_mem.size());
    mesh.eQuery_device_mem.resize(mesh.eQuery_host_mem.size());

    Gpu::copy(Gpu::hostToDevice, mesh.eDomQuad_host_mem.begin(), mesh.eDomQuad_host_mem.end(), mesh.eDomQuad_device_mem.begin());
    Gpu::copy(Gpu::hostToDevice, mesh.eBouQuad_host_mem.begin(), mesh.eBouQuad_host_mem.end(), mesh.eBouQuad_device_mem.begin());
    Gpu::copy(Gpu::hostToDevice, mesh.eIntBouQuad_host_mem.begin(), mesh.eIntBouQuad_host_mem.end(), mesh.eIntBouQuad_device_mem.begin());
    Gpu::copy(Gpu::hostToDevice, mesh.eQuery_host_mem.begin(), eQuery_host_mem.end(), mesh.eQuery_device_mem.begin());
#endif
    // ================================================================

}
// ####################################################################

// EVAL THE CHOLESKY DECOMPOSITION OF THE MASS MATRICES FOR THE #######
// IMPLICIT MESH ######################################################
template<int N_PHI, int N_DOM, typename IBVP_CLASS>
void EvalMassMatrices(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                      MatrixFactory<N_PHI, N_DOM> & matfactory,
                      const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalMassMatrices(const ImplicitMesh<N_PHI, N_DOM> &, MatrixFactory<N_PHI, N_DOM> &, const IBVP_CLASS &)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;
    const Real iaJ = mesh.iaJ;

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    const UnitCube<AMREX_SPACEDIM> & cube = matfactory.cube;

    // IMPLICIT MESH
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();

    // DG
    const int sp = cube.p;
    const int sNp = cube.Np;
    // ================================================================

    // VARIABLES ======================================================
    long space_MMCh_mem_pos;
    // ================================================================

    // INITIALIZATION =================================================
    space_MMCh_mem_pos = 0L;
    // ================================================================

    // CHECK THAT THE STANDARD ELEMENTS OF THE GEOMETRY AND OF THE ====
    // MATRIX FACTORY ARE CONSISTENT ==================================
    if (mesh.cube.dom_Nq != matfactory.cube.dom_Nq)
    {
        Print() << std::endl;
        Print() << "ERROR: AMReX_DG_Solution.H - EvalMassMatrices" << std::endl;
        Print() << "| The [-1,1]^" << AMREX_SPACEDIM << " cube of the input ImplicitMesh and that of the MatrixFactory are not consistent." << std::endl;
        Print() << "| ImplicitMesh.cube.dom_Nq: " << mesh.cube.dom_Nq << std::endl;
        Print() << "| MatrixFactory.cube.dom_Nq   : " << matfactory.cube.dom_Nq << std::endl;
        Print() << std::endl;
        exit(-1);
    }
    // ================================================================

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM INFO ===========================
    // CHOLESKY DECOMPOSITION OF THE MASS MATRIX
    {
        // Resize memory
        matfactory.space_MMCh_host_mem.resize(sNp*sNp);

        // Store data
        std::copy(matfactory.cube.MMCh.begin(), matfactory.cube.MMCh.end(), matfactory.space_MMCh_host_mem.data());

        // Update memory
        space_MMCh_mem_pos += sNp*sNp;
    }
    // ================================================================

    // STORE THE MASS MATRIX INFO AS DOMAIN (CELL-CENTERED) DATA ======
    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.fabbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

        Array4<long> const & space_MMCh_pos_fab = matfactory.space_MMCh_pos.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
            const bool elm_is_partial = (elm_is_large || elm_is_small);
            const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], w;
            Real MM[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

            // Get the support of the basis functions. It is recalled
            // that if the element is small, the support of the basis
            // functions is given by the element that is merging into
            // the small current element.
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

            // Data structures used to compute the basis functions
            DG_sol_space_BF bf(BF_lo, BF_hi, sp);

            // EMPTY ELEMENTS -----------------------------------------
            if (elm_is_empty)
            {
                space_MMCh_pos_fab(i,j,k,dom) = -1L;
            }
            // --------------------------------------------------------
            // ENTIRE ELEMENTS ----------------------------------------
            else if (elm_is_entire)
            {
                // MASS MATRIX
                // At the moment we store the mass matrix of the
                // [-1,1]^AMREX_SPACEDIM cube which will be
                // subsequently combined with the mass matrix of the
                // small element the current element extends to.
                if (elm_is_entire_extended)
                {
                    // Store memory offset
                    space_MMCh_pos_fab(i,j,k,dom) = space_MMCh_mem_pos;

                    // Expand memory
                    matfactory.space_MMCh_host_mem.resize(space_MMCh_mem_pos+sNp*sNp);

                    // Store data
                    std::copy(matfactory.cube.MM.begin(), matfactory.cube.MM.end(), &matfactory.space_MMCh_host_mem[space_MMCh_mem_pos]);

                    // Update memory
                    space_MMCh_mem_pos += sNp*sNp;
                }
                else
                {
                    space_MMCh_pos_fab(i,j,k,dom) = 0L;
                }
            }
            // --------------------------------------------------------
            // PARTIAL ELEMENTS ---------------------------------------
            else if (elm_is_partial)
            {
                // MASS MATRIX
                // At the moment we compute and store the mass matrix
                // of the element.
                {
                    // QUADRATURE INFO
                    const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    const long pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                    const Real * xptr = &eDomQuad_mem_ptr[pos];
                    long qi_pos;

                    // Store memory offset
                    space_MMCh_pos_fab(i,j,k,dom) = space_MMCh_mem_pos;

                    // Expand memory
                    matfactory.space_MMCh_host_mem.resize(space_MMCh_mem_pos+sNp*sNp);

                    // Store data
                    std::fill(MM, MM+sNp*sNp, 0.0);

                    for (int qi = 0; qi < dom_Nq; ++qi)
                    {
                        qi_pos = (AMREX_SPACEDIM+1)*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                        )
                        w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

                        bf._eval_BF_only_table_(x);

                        for (int cs = 0; cs < sNp; ++cs)
                        for (int rs = 0; rs < sNp; ++rs)
                        {
                            MM[rs+cs*sNp] += bf.BF[rs]*bf.BF[cs]*w;
                        }
                    }

                    // NOTE: consistently with the storing of the quad
                    //       points, we are storing the mass matrix as
                    //       it were computed on a
                    //       a [-1,1]^AMREX_SPACEDIM domain.
                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        matfactory.space_MMCh_host_mem[space_MMCh_mem_pos+rs] = MM[rs]*iaJ;
                    }

                    // Update memory
                    space_MMCh_mem_pos += sNp*sNp;
                }
            }
            // --------------------------------------------------------
        }
    }
    // ================================================================

    // STORE MEMORY INFO ==============================================
    matfactory.mem_for_space_MMCh = space_MMCh_mem_pos;
    // ================================================================

    // COPY MEMORY TO THE DEVICE ======================================
#ifdef AMREX_USE_CUDA
    matfactory.space_MMCh_device_mem.resize(matfactory.space_MMCh_host_mem.size());

    Gpu::copy(Gpu::hostToDevice, matfactory.space_MMCh_host_mem.begin(), matfactory.space_MMCh_host_mem.end(), matfactory.space_MMCh_device_mem.begin());
#endif
    // ================================================================

    // ADD THE CONTRIBUTION FOR THE EXTENDED ELEMENTS =================
#ifdef AMREX_USE_CUDA
    Real * space_MMCh_mem_ptr = matfactory.space_MMCh_device_mem.data();
#else
    Real * space_MMCh_mem_ptr = matfactory.space_MMCh_host_mem.data();
#endif

    for (MFIter mfi(mesh.eType); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<long const> const & space_MMCh_pos_fab = matfactory.space_MMCh_pos.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_entire_extended = elm_is_entire && (etype/10 == 1);
            const bool elm_is_special = elm_is_large || elm_is_entire_extended;
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

            const long pos = space_MMCh_pos_fab(i,j,k,dom);

            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;
            long nbr_pos;
            Real MM[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];
            Real work[DG_SOL_MAX_SPACE_Np*DG_SOL_MAX_SPACE_Np];

            // RETRIEVE THE DG MATRICES FOR VALID SPECIAL ELEMENTS
            if (elm_is_special)
            {
                for (int rs = 0; rs < (sNp*sNp); ++rs)
                {
                    // Mass matrix
                    MM[rs] = space_MMCh_mem_ptr[pos+rs];
                }
            }

            // Cycle over the boundaries, check whether the element
            // is extended and add the contribution
            if (elm_is_extended)
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            {
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                if (nbr_merged_b == nbr_b)
                {
                    nbr_pos = space_MMCh_pos_fab(nbr_i,nbr_j,nbr_k,dom);

                    for (int rs = 0; rs < (sNp*sNp); ++rs)
                    {
                        // Mass matrix
                        MM[rs] += space_MMCh_mem_ptr[nbr_pos+rs];
                    }
                }
            }

            // CHOLESKY DECOMPOSITION OF MASS MATRIX FOR VALID
            // SPECIAL ELEMENTS
            if (elm_is_special)
            {
                DG_utils::Cholesky(sNp, MM, &space_MMCh_mem_ptr[pos], work);
            }
        });
        Gpu::synchronize();
    }
    // ================================================================
}
// ####################################################################

// EVAL THE AVERAGE OPERATOR FOR THE IMPLICIT MESH ####################
template<int N_PHI, int N_DOM, typename IBVP_CLASS>
void EvalAverageOperator(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                         MatrixFactory<N_PHI, N_DOM> & matfactory,
                         const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("EvalAverageOperator(const ImplicitMesh<N_PHI, N_DOM> &, MatrixFactory<N_PHI, N_DOM> &, const IBVP_CLASS &)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID

    // UNIT CUBE [-1,1]^AMREX_SPACEDIM
    const UnitCube<AMREX_SPACEDIM> & cube = matfactory.cube;

    // IMPLICIT MESH
    //const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();

    // DG
    const int sp = cube.p;
    //const int sNp = cube.Np;
    // ================================================================

    // QUICK RETURN IF p = 0 ==========================================
    if (sp == 0) return;
    // ================================================================

Print() << "EvalAverageOperator" << std::endl;
exit(-1);
}
// ####################################################################

// ADD CONTRIBUTION OF SMALL ELEMENTS TO EXTENDED ELEMENTS ############
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void AddSmallElementsContribution(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                  const MatrixFactory<N_PHI, N_DOM> & matfactory,
                                  MultiFab & X,
                                  const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("AddSmallElementsContribution(ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // DG
    const int sNp = matfactory.cube.Np;
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);
            
            // LOCAL VARIABLES
            short nbr_etype;
            int nbr_i, nbr_j, nbr_k, nbr_b, nbr_merged_b;
            bool nbr_is_small;

            // EXTENDED ELEMENTS
            if (elm_is_extended)
            for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
            {
                NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;

                if (nbr_merged_b == nbr_b)
                {
                    for (int rs = 0; rs < sNp; ++rs)
                    {
                        X_fab(i,j,k,rs+ru*sNp) += X_fab(nbr_i,nbr_j,nbr_k,rs+ru*sNp);
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// MULTIPLY BY THE INVERSE OF THE MASS MATRICES OF THE IMPLICIT MESH ##
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void MultiplyByInverseMassMatrix(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                 const MatrixFactory<N_PHI, N_DOM> & matfactory,
                                 MultiFab & X,
                                 const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("MultiplyByInverseMassMatrix(ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const Real iaJ = mesh.iaJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * space_MMCh_mem_ptr = matfactory.space_MMCh_device_mem.data();
#else
    const Real * space_MMCh_mem_ptr = matfactory.space_MMCh_host_mem.data();
#endif

    // DG
    const int sNp = matfactory.cube.Np;
    // ================================================================

    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<long const> const & space_MMCh_pos_fab = matfactory.space_MMCh_pos.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;

            // MASS MATRIX
            const long pos = space_MMCh_pos_fab(i,j,k,dom);
            const Real * space_MMCh_ptr = &space_MMCh_mem_ptr[pos];
            
            // LOCAL VARIABLES
            Real Xh[DG_SOL_MAX_SPACE_Np];

            // IF VALID, MULTIPLY BY THE INVERSE OF THE MASS MATRIX
            if (elm_is_valid)
            {
                // Copy from memory. Remember that the mass matrix is
                // that of an element defined over the
                // [-1,1]^AMREX_SPACEDIM domain.
                for (int rs = 0; rs < sNp; ++rs)
                {
                    Xh[rs] = X_fab(i,j,k,rs+ru*sNp)*iaJ;
                }

                // Actual matrix operation (using mass matrix' Cholesky
                // decomposition)
                DG_utils::Cholesky_solve_overwrite(sNp, 1, space_MMCh_ptr, Xh);

                // Copy back to memory
                for (int rs = 0; rs < sNp; ++rs)
                {
                    X_fab(i,j,k,rs+ru*sNp) = Xh[rs];
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// PROJECT THE INITIAL CONDITIONS OVER AN IMPLICITLY-DEFINED MESH #####
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ProjectInitialConditions(const ImplicitMesh<N_PHI, N_DOM> & mesh,
                              const MatrixFactory<N_PHI, N_DOM> & matfactory,
                              MultiFab & X,
                              const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ProjectInitialConditions(ImplicitMesh<N_PHI, N_DOM> &, const MatrixFactory<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    const int sNp = matfactory.cube.Np;
    // ================================================================

    // INITIALIZATION =================================================
    X = 0.0;
    // ================================================================

    // EVAL THE PROJECTION ============================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);

        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_U,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int ru) noexcept
        {
            // LOCAL PARAMETERS
            const int dom = IBVP.F_U2DOM(ru);
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            
            // QUADRATURE INFO
            const int dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
            const long eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
            const Real * xptr = &eDomQuad_mem_ptr[eDomQuad_pos];

            // LOCAL VARIABLES
            long qi_pos;
            Real x[AMREX_SPACEDIM], w;
            Real integrand;
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

            // Get the support of the basis functions
            BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_lo, BF_hi);

            // Data structures used to compute the basis functions
            DG_sol_space_BF bf(BF_lo, BF_hi, sp);

            // PROJECTION
            for (int qi = 0; qi < dom_Nq; ++qi)
            {
                qi_pos = (AMREX_SPACEDIM+1)*qi;
                AMREX_D_TERM
                (
                    x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                    x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                    x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                )
                w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;

                bf._eval_BF_only_table_(x);
                
                for (int rs = 0; rs < sNp; ++rs)
                {
                    integrand = bf.BF[rs]*IBVP.F_U0(ru, x);
                    X_fab(i,j,k,rs+ru*sNp) += integrand*w;
                }
            }
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // EXTENDED ELEMENTS AND MULTIPLICATION BY INVERSE MASS MATRIX ====
    AddSmallElementsContribution<N_PHI, N_DOM, N_U>(mesh, matfactory, X, IBVP);
    MultiplyByInverseMassMatrix<N_PHI, N_DOM, N_U>(mesh, matfactory, X, IBVP);
    // ================================================================

}
// ####################################################################



// FILL BOUNDARY AND SMALL OR EMPTY CELLS USING BCS OR ICS INFO #######
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void FillInvalidCells(const Real t,
                      const ImplicitMesh<N_PHI, N_DOM> & mesh,
                      const MatrixFactory<N_PHI, N_DOM> & matfactory,
                      MultiFab & X,
                      const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("FillInvalidCells(const Real, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    const int ngr = 1;

    // GRID
    const Box domain = mesh.geom.Domain();
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<int, AMREX_SPACEDIM> is_periodic = mesh.is_periodic;

    // LEVEL SET
    const int phi_sp = mesh.cube.p;

    // DG
    const int sNp = matfactory.cube.Np;

    // AUXILIARY MULTIFABS
    MultiFab Xa;
    MultiFab * pXa;
    // ================================================================

    // EVAL THE AVERAGE IF p > 0 ======================================
    if (sNp != 1)
    {
Print() << "(We need to compute the average)" << std::endl;
exit(-1);
    }
    else
    {
        pXa = &X;
    }
    // ================================================================
    
    // ================================================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();

        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        
        Array4<Real> const & Xa_fab = pXa->array(mfi);
        Array4<Real> const & X_fab = X.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_empty = (etype == DG_ELM_TYPE_EMPTY);
            const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);

            // LOCAL VARIABLES
            Real tmp;
            Real cell_m[AMREX_SPACEDIM], un[AMREX_SPACEDIM];
            Real Uh[N_U], U[N_U], nbr_Uh[N_U];

            // CURRENT CELL DATA
            for (int ru = 0; ru < N_U; ++ru)
            {
                Uh[ru] = Xa_fab(i,j,k,ru);
            }

            // UPDATE GRID BOUNDARY VALUES ============================
            if ((i == domain.smallEnd(0)) && (is_periodic[0] == 0))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i-1+0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k  +0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] = -1.0;,
                    un[1] =  0.0;,
                    un[2] =  0.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i-1,j,k,ru*sNp) = U[ru];
                }
            }
            if ((i == domain.bigEnd(0)) && (is_periodic[0] == 0))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i+1+0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k  +0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] = +1.0;,
                    un[1] =  0.0;,
                    un[2] =  0.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i+1,j,k,ru*sNp) = U[ru];
                }
            }

#if (AMREX_SPACEDIM > 1)
            if (j == domain.smallEnd(1))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j-1+0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k  +0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] =  0.0;,
                    un[1] = -1.0;,
                    un[2] =  0.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i,j-1,k,ru*sNp) = U[ru];
                }
            }
            if (j == domain.bigEnd(1))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j+1+0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k  +0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] =  0.0;,
                    un[1] = +1.0;,
                    un[2] =  0.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i,j+1,k,ru*sNp) = U[ru];
                }
            }
#endif
#if (AMREX_SPACEDIM > 2)
            if (k == domain.smallEnd(2))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k-1+0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] =  0.0;,
                    un[1] =  0.0;,
                    un[2] = -1.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i,j,k-1,ru*sNp) = U[ru];
                }
            }
            if (k == domain.bigEnd(2))
            {
                AMREX_D_TERM
                (
                    cell_m[0] = prob_lo[0]+(i  +0.5)*dx[0];,
                    cell_m[1] = prob_lo[1]+(j  +0.5)*dx[1];,
                    cell_m[2] = prob_lo[2]+(k+1+0.5)*dx[2];
                )
                AMREX_D_TERM
                (
                    un[0] =  0.0;,
                    un[1] =  0.0;,
                    un[2] = +1.0;;
                )

                IBVP.F_BCS(dom, t, cell_m, un, Uh, U);
                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i,j,k+1,ru*sNp) = U[ru];
                }
            }
#endif
            // ========================================================

            // LEVEL SET INFO =========================================
            DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, phi_sp, i, j, k, -1);

            // WHICH LEVEL SET
            int phi_info[2];
            IBVP.F_DOM2PHI(dom, phi_info);
            phi.set_fab_ptrs(i, j, k, phi_info[0]);
            phi.set_sign(phi_info[1]);
            
            // LEVEL SET GRADIENT DIRECTION
            phi.eval_grad(cell_m, un);
            tmp = 1.0/std::sqrt(AMREX_D_PICK(un[0]*un[0], un[0]*un[0]+un[1]*un[1], un[0]*un[0]+un[1]*un[1]+un[2]*un[2]));
            AMREX_D_TERM
            (
                un[0] *= -tmp;,
                un[1] *= -tmp;,
                un[2] *= -tmp;
            )

            // AUXILIARY TABLES
#if (AMREX_SPACEDIM == 2)
            const int nbr_i_table[8] = {1,1,0,-1,-1,-1,0,1};
            const int nbr_j_table[8] = {0,1,1,1,0,-1,-1,-1};
            const int nbr_k_table[1] = {0};
#endif
#if (AMREX_SPACEDIM == 3)
            const int nbr_i_table[8] = {1,1,0,-1,-1,-1,0,1};
            const int nbr_j_table[8] = {0,1,1,1,0,-1,-1,-1};
            const int nbr_k_table[3] = {1,0,-1};
#endif
            // ========================================================

            // UPDATE EMPTY CELLS VALUES ==============================
            if (elm_is_empty)
            {
                const amrex::Real dph = 2.0*M_PI/8.0;
                const amrex::Real ph = std::atan2(un[1], un[0]);
                const int i_ph = (int) std::round((ph < 0.0) ? ((2.0*M_PI+ph-0.5*dph)/dph) : ((ph-0.5*dph)/dph));
#if (AMREX_SPACEDIM == 2)
                const int i_th = 0;
#endif
#if (AMREX_SPACEDIM == 3)
                const amrex::Real dth = M_PI/3.0;
                const amrex::Real th = std::acos(un[2]);
                const int i_th = (int) fmin(std::round(th/dth), 2);
#endif
                const int nbr_i = i+nbr_i_table[i_ph];
                const int nbr_j = j+nbr_j_table[i_ph];
                const int nbr_k = k+nbr_k_table[i_th];

                const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                const bool nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);

                if (nbr_is_empty || nbr_is_small)
                {
                    // do nothing
                }
                else
                {
                    const int nbr_dom = IBVP.F_DOM2NBRDOM(dom);

                    // WE ARE CLOSE TO A BOUNDARY ---------------------
                    if (nbr_dom == -1)
                    {
                        // NEIGHBORING CELL DATA
                        for (int ru = 0; ru < N_U; ++ru)
                        {
                            nbr_Uh[ru] = Xa_fab(nbr_i,nbr_j,nbr_k,ru);
                        }

                        IBVP.F_BCS(dom, t, cell_m, un, nbr_Uh, U);

                        for (int ru = 0; ru < N_U; ++ru)
                        {
                            X_fab(i,j,k,ru*sNp) = U[ru];
                        }
                    }
                    // ------------------------------------------------
                    // WE ARE CLOSE TO AN INTERFACE -------------------
                    else
                    {
Abort("HERE WE ARE - FillInvalidCells");
exit(-1);
                    }
                    // ------------------------------------------------
                }
            }
            // ========================================================

            // UPDATE SMALL CELLS VALUES ==============================
            if (elm_is_small)
            {
                int BF_i, BF_j, BF_k;

                BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                for (int ru = 0; ru < N_U; ++ru)
                {
                    X_fab(i,j,k,ru*sNp) = X_fab(BF_i,BF_j,BF_k,ru*sNp);
                }
            }
            // ========================================================
        });
        Gpu::synchronize();
    }
    X.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// COMPUTE THE TIME INCREMENT #########################################
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
Real Compute_dt_ExplicitHyperbolic(const Real t,
                                   const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                   const MatrixFactory<N_PHI, N_DOM> & matfactory,
                                   const MultiFab & X,
                                   const charMultiFab & tags,
                                   const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("Compute_dt_ExplicitHyperbolic(const Real, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eQuery_mem_ptr = mesh.eQuery_device_mem.data();
#else
    const Real * eQuery_mem_ptr = mesh.eQuery_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    // ================================================================

    // VARIABLES ======================================================
    MultiFab dt(mesh.cc_ba, mesh.dm, 1, 0);
    dt = 1.0e12;
    Real dt_min;
    // ================================================================

    // EVAL THE MINIMUM TIME STEP =====================================
    for (MFIter mfi(dt); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
        
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<Real const> const & eInfo_fab = mesh.eInfo.array(mfi);
        Array4<char const> const & tags_fab = tags.array(mfi);
        Array4<int const> const & eQuery_Npt_fab = mesh.eQuery_Npt.array(mfi);
        Array4<long const> const & eQuery_pos_fab = mesh.eQuery_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);
        Array4<Real> const & dt_fab = dt.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};

            // LOCAL VARIABLES
            short etype;
            bool elm_is_not_empty, elm_is_large;
            Real vf;
            bool elm_is_not_troubled;
            int BF_i, BF_j, BF_k;
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
            Real x[AMREX_SPACEDIM];
            Real Uh[N_U];
            Real elm_dt, x_dt;

            // QUERY INFO
            int dom_Npt;
            long eQuery_pos, qi_pos;
            const Real * xptr;

            // Data structures used to compute the basis functions
            // and the unknown fields
            DG_sol_space_BFX sol(X_fab, sp);

            // CYCLE OVER THE DOMAINS
            for (int dom = 0; dom < N_DOM; ++dom)
            {
                // Element info
                etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);
                elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
                vf = eInfo_fab(i,j,k,DG_ELM_INFO_N_COMP_PER_DOM*dom);
                elm_is_not_troubled = (tags_fab(i,j,k,dom) != TROUBLED_ELEMENT);

                if (elm_is_not_empty && elm_is_not_troubled)
                {
                    // Get the support of the basis functions
                    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);
                    sol.set_support(BF_lo, BF_hi);

                    // Query info
                    dom_Npt = eQuery_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                    eQuery_pos = eQuery_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                    xptr = &eQuery_mem_ptr[eQuery_pos];

                    // CYCLE OVER THE QUERY POINTS
                    elm_dt = 1.0e12;
                    for (int qi = 0; qi < dom_Npt; ++qi)
                    {
                        qi_pos = AMREX_SPACEDIM*qi;
                        AMREX_D_TERM
                        (
                            x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                            x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                            x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                        )

                        sol._eval_BF_only_table_(x);
                        for (int ru = 0; ru < N_U; ++ru)
                        {
                            sol.set_fab_ptrs(BF_i, BF_j, BF_k, ru);

                            Uh[ru] = sol._eval_unsafe_();
                        }

                        x_dt = IBVP.F_DT(&dx[0], t, x, Uh);
                        elm_dt = fmin(elm_dt, x_dt);
                    }
                    elm_dt = (elm_is_large) ? (vf*elm_dt) : elm_dt;
                    dt_fab(i,j,k,0) = fmin(dt_fab(i,j,k,0), elm_dt);
                }
            }
        });
        Gpu::synchronize();
    }
    // ===============================================================

    // GET THE MINUMUM 
    dt_min = dt.min(0);
    // ================

    return dt_min;
}
// ####################################################################

// IDENTIFY ELEMENTS TO BE LIMITED ####################################
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void IdentifyTroubledElements(const Real t,
                              const ImplicitMesh<N_PHI, N_DOM> & mesh,
                              const MatrixFactory<N_PHI, N_DOM> & matfactory,
                              MultiFab & X,
                              charMultiFab & tags,
                              const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("IdentifyTroubledElements(const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;
    const Real aJ = mesh.aJ;

    // IMPLICIT MESH
#ifdef AMREX_USE_CUDA
    const Real * eQuery_mem_ptr = mesh.eQuery_device_mem.data();
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_device_mem.data();
#else
    const Real * eQuery_mem_ptr = mesh.eQuery_host_mem.data();
    const Real * eDomQuad_mem_ptr = mesh.eDomQuad_host_mem.data();
#endif

    // DG
    const int sp = matfactory.cube.p;
    // ================================================================

    // INITIALIZATION =================================================
    tags = SOUND_ELEMENT;
    // ================================================================

    // IDENTIFY THE TROUBLED ELEMENTS =================================
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
            
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eQuery_Npt_fab = mesh.eQuery_Npt.array(mfi);
        Array4<long const> const & eQuery_pos_fab = mesh.eQuery_pos.array(mfi);
        Array4<int const> const & eDomQuad_Nq_fab = mesh.eDomQuad_Nq.array(mfi);
        Array4<long const> const & eDomQuad_pos_fab = mesh.eDomQuad_pos.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<char> const & tags_fab = tags.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_not_empty = (etype != DG_ELM_TYPE_EMPTY);

            // LOCAL VARIABLES
            int BF_i, BF_j, BF_k;
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];
            Real x[AMREX_SPACEDIM], w;
            Real Uh[N_U], Uh_truncated[N_U], aux_Uh[N_U];

            // QUERY INFO
            int dom_Npt;
            long eQuery_pos, qi_pos;
            const Real * xptr;
            bool troubled_solution;
            Real Uh_max[N_U], Uh_min[N_U], nbr_Uh_max[N_U], nbr_Uh_min[N_U];

            // QUADRATURE INFO
            int dom_Nq;
            long eDomQuad_pos;

            if (elm_is_not_empty)
            {
                // Get the support of the basis functions
                BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                // Data structures used to compute the dG solution
                DG_sol_space_BFX sol(BF_lo, BF_hi, X_fab, sp, BF_i, BF_j, BF_k, -1);

                // Query info
                dom_Npt = eQuery_Npt_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                eQuery_pos = eQuery_pos_fab(i,j,k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                xptr = &eQuery_mem_ptr[eQuery_pos];

                // PHYSICAL CONDITIONS (GIVEN BY THE IBVP) ------------
                // ALSO INIT MIN AND MAX
                for (int ru = 0; ru < N_U; ++ru)
                {
                    Uh_max[ru] = -1.0e12;
                    Uh_min[ru] = +1.0e12;
                    nbr_Uh_max[ru] = -1.0e12;
                    nbr_Uh_min[ru] = +1.0e12;
                }
                
                // CYCLE OVER THE QUERY POINTS
                for (int qi = 0; qi < dom_Npt; ++qi)
                {
                    qi_pos = AMREX_SPACEDIM*qi;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                        x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                        x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                    )

                    sol._eval_BF_only_table_(x);
                    for (int ru = 0; ru < N_U; ++ru)
                    {
                        sol.set_fab_ptrs(BF_i, BF_j, BF_k, ru);

                        Uh[ru] = sol._eval_unsafe_();

                        // STORE MIN AND MAX
                        Uh_max[ru] = std::max(Uh_max[ru], Uh[ru]);
                        Uh_min[ru] = std::min(Uh_min[ru], Uh[ru]);
                    }

                    troubled_solution = IBVP.F_TROUBLED(dom, t, x, Uh);

                    if (troubled_solution)
                    {
                        tags_fab(i,j,k,dom) = TROUBLED_ELEMENT;
                    }
                }
                // ----------------------------------------------------

                // DISCRETE MAXIMUM PRINCIPLE -------------------------
                // ----------------------------------------------------
                /*
                if (tags_fab(i,j,k,dom) == SOUND_ELEMENT)
                {
                    const bool elm_is_small = (etype%10 == DG_ELM_TYPE_SMALL);
                    const int merged_b = (elm_is_small) ? (etype/10) : -1;

                    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                    {
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);
                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
//const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
//const bool cells_are_merged = ((nbr_merged_b == nbr_b) || (merged_b == b));
                        const bool nbr_is_empty = (nbr_etype == DG_ELM_TYPE_EMPTY);
                        
                        if (nbr_is_small || nbr_is_empty) continue;

                        // Get the support of the basis functions
                        int nbr_BF_i, nbr_BF_j, nbr_BF_k;
                        Real nbr_BF_lo[AMREX_SPACEDIM], nbr_BF_hi[AMREX_SPACEDIM];
                        BF_CELL(&prob_lo[0], &dx[0], nbr_i, nbr_j, nbr_k, nbr_etype, nbr_BF_i, nbr_BF_j, nbr_BF_k, nbr_BF_lo, nbr_BF_hi);
                        
                        DG_sol_space_BFX nbr_sol(nbr_BF_lo, nbr_BF_hi, X_fab, sp, nbr_BF_i, nbr_BF_j, nbr_BF_k, -1);

                        // Query info
                        dom_Npt = eQuery_Npt_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        eQuery_pos = eQuery_pos_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TRI_N_COMP_PER_DOM*dom);
                        xptr = &eQuery_mem_ptr[eQuery_pos];
                    }
                }
                */
                // ----------------------------------------------------

                // WE ALSO IMPLEMENT PERSSON AND PERAIRE SHOCK SENSOR -
                // REF: Sub-Cell Shock Capturing for dG Methods
                //      Persson and Peraire
                //      2006
                // ----------------------------------------------------
                DG_sol_space_BFX_truncated sol_truncated(BF_lo, BF_hi, X_fab, sp, sp-1, BF_i, BF_j, BF_k, -1);

                // Quadrature info
                dom_Nq = eDomQuad_Nq_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                eDomQuad_pos = eDomQuad_pos_fab(i,j,k,DG_ELM_DOM_QUAD_N_COMP_PER_DOM*dom);
                xptr = &eDomQuad_mem_ptr[eDomQuad_pos];

                Real err, norm, integrand_err, integrand_norm, r_err;

                // EVAL THE INTEGRAL
                err = 0.0;
                norm = 0.0;

                for (int qi = 0; qi < dom_Nq; ++qi)
                {
                    // QUADRATURE POINT AND WEIGHT
                    qi_pos = (AMREX_SPACEDIM+1)*qi;
                    AMREX_D_TERM
                    (
                        x[0] = xptr[qi_pos+0]; x[0] = J[0]*x[0]+cell_m[0];,
                        x[1] = xptr[qi_pos+1]; x[1] = J[1]*x[1]+cell_m[1];,
                        x[2] = xptr[qi_pos+2]; x[2] = J[2]*x[2]+cell_m[2];
                    )
                    w = xptr[qi_pos+AMREX_SPACEDIM]; w = w*aJ;
                    
                    // SOLUTION
                    sol._eval_BF_table_(x);
                    sol_truncated._eval_BF_table_(x);
                    for (int cu = 0; cu < N_U; ++cu)
                    {
                        sol.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
                        sol_truncated.set_fab_ptrs(BF_i, BF_j, BF_k, cu);
                        
                        Uh[cu] = sol._eval_unsafe_();
                        Uh_truncated[cu] = sol_truncated._eval_unsafe_();
                    }
                    // INTEGRAND CONTRIBUTION
                    integrand_err = 0.0;
                    integrand_norm = 0.0;
                    for (int cu = 0; cu < N_U; ++cu)
                    {
                        integrand_err += (Uh[cu]-Uh_truncated[cu])*(Uh[cu]-Uh_truncated[cu]);
                        integrand_norm += Uh[cu]*Uh[cu];
                    }

                    err += integrand_err*w;
                    norm += integrand_norm*w;
                }

                r_err = err/norm;

                if (r_err > 1.0e-12)
                {
                    const Real s = std::log10(r_err);
                    const Real s0 = -4.0*std::log10(sp);
                    const Real ds = -2.0;
                    
                    if (s > (s0+ds))
                    {
                        tags_fab(i,j,k,dom) = TROUBLED_ELEMENT;
                    }
                }
                // ----------------------------------------------------
            }
        });
        Gpu::synchronize();
    }
    tags.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // MAKE SURE THAT, IF AN EXTENDED CELL IS MARKED AS TROUBLED, SO ==
    // WILL THE SMALL CELLS IT EXTENDS TO. AND VICEVERSA ==============
    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box & bx = mfi.validbox();
            
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);

        Array4<char> const & tags_fab = tags.array(mfi);

        ParallelFor(bx, N_DOM,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int dom) noexcept
        {
            // LOCAL PARAMETERS
            const short etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
            const bool elm_is_entire = (etype%10 == DG_ELM_TYPE_ENTIRE);
            const bool elm_is_large = (etype%10 == DG_ELM_TYPE_LARGE);
            const bool elm_is_valid = elm_is_entire || elm_is_large;
            const bool elm_is_extended = elm_is_valid && (etype/10 == 1);

            if (elm_is_extended)
            {
                // SEARCH FOR ANY TROUBLED ELEMENT AMONG THE SMALL ONES
                // THE CURRENT ELEMENT EXTENDS TO
                for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                {
                    int nbr_i, nbr_j, nbr_k, nbr_b;
                    NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                    const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                    const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                    const bool cells_are_merged = (nbr_merged_b == nbr_b);
                    const bool nbr_is_troubled = (tags_fab(nbr_i,nbr_j,nbr_k,dom) == TROUBLED_ELEMENT);

                    if (cells_are_merged && nbr_is_troubled)
                    {
                        tags_fab(i,j,k,dom) = TROUBLED_ELEMENT;
                    }
                }

                // IF THE CURRENT ELEMENT IS TROUBLED, MARK ALL THE
                // SMALL ELEMENTS IT EXTENDS TO
                const bool elm_is_troubled = (tags_fab(i,j,k,dom) == TROUBLED_ELEMENT);

                if (elm_is_troubled)
                {
                    for (int b = 0; b < DG_CELL_N_SPACE_BOUNDARIES; ++b)
                    {
                        int nbr_i, nbr_j, nbr_k, nbr_b;
                        NBR_CELL(i, j, k, b, nbr_i, nbr_j, nbr_k, nbr_b);

                        const short nbr_etype = eType_fab(nbr_i,nbr_j,nbr_k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        const bool nbr_is_small = (nbr_etype%10 == DG_ELM_TYPE_SMALL);
                        const int nbr_merged_b = (nbr_is_small) ? (nbr_etype/10) : -1;
                        const bool cells_are_merged = (nbr_merged_b == nbr_b);

                        if (cells_are_merged)
                        {
                            tags_fab(nbr_i,nbr_j,nbr_k,dom) = TROUBLED_ELEMENT;
                        }
                    }
                }
            }
        });
        Gpu::synchronize();
    }
    tags.FillBoundary(mesh.geom.periodicity());
    // ================================================================
}
// ####################################################################

// EXPORT IMPLICIT MESH TO VTK FORMAT #################################
template<int N_PHI, int N_DOM, typename IBVP_CLASS>
void ExportMesh_VTK(const std::string & output_folder,
                    const std::string & filename_root,
                    const int time_step,
                    const int n_time_steps,
                    const Real t,
                    const ImplicitMesh<N_PHI, N_DOM> & mesh,
                    const charMultiFab & tags,
                    const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ExportMesh_VTK(const std::string &, const std::string &, ....)");
    // ----------------------------------------------------------------
    
    // PARAMETERS =====================================================
    // CURRENT PROCESSOR
    const int rank = ParallelDescriptor::MyProc();

    // DATE AND TIME
    //const time_t now = time(0);
    //const char * date_time = ctime(&now);

    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;

    // STANDARD ELEMENT
    const int cell_dom_Ng = mesh.cube.dom_Ng;
    const int cell_e = mesh.cube.g;
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
    const int cell_conn_len = AMREX_D_PICK(2, 4, 8)*cell_dom_Ne;
    const Gpu::ManagedVector<int> cell_base_conn = Get_VTK_BaseCell_Connectivity(cell_e);
    const int * cell_base_conn_ptr = cell_base_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = mesh.cube.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = mesh.cube.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = mesh.cube.dom_XI3g.data();
    )

    // STANDARD ELEMENT BOUNDARY
    const int cell_bou_Ng = mesh.cube.bou_Ng;
    const int cell_bou_Ne = AMREX_D_PICK(1, cell_e, cell_e*cell_e);
    const int bou_cell_conn_len = AMREX_D_PICK(1, 2*cell_bou_Ne, 4*cell_bou_Ne);
    const Gpu::ManagedVector<int> bou_cell_base_conn = Get_VTK_BaseBouCell_Connectivity(cell_e);
    const int * bou_cell_base_conn_ptr = bou_cell_base_conn.data();
    
    AMREX_D_TERM
    (
        ,
        Real const * bou_XI1g_ptr = mesh.cube.bou_XI1g.data();,
        Real const * bou_XI2g_ptr = mesh.cube.bou_XI2g.data();
    )

    // DG
    const int p = mesh.cube.p;

    // CELL-DATA
    // Processor rank
    // Box id
    // Element type (per each domain)
    // Element color (per each domain)
    // Element id (per each domain)
    const int n_cell_fields = 1+1+N_DOM+N_DOM+N_DOM;

    // BOUNDARY CELL-DATA
    // Element type (per each domain)
    const int n_bou_cell_fields = N_DOM;

    // NODAL-DATA
    // Level set functions
    const int n_nodal_fields = N_PHI;

    // BOUNDARY NODAL-DATA
    // Level set functions
    const int n_bou_nodal_fields = N_PHI;

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(3, 9, 12);
    const int VTK_bou_cell_id = AMREX_D_PICK(1, 3, 9);
    // ================================================================

    // VARIABLES ======================================================
    long n_base_cells, n_nodes, n_cells;
    long nodes_pos, conn_pos, cell_fields_pos, nodal_fields_pos;
    longMultiFab output_data_info(mesh.cc_ba, mesh.dm, 4, mesh.dom_data_ngr);
    longMultiFab elm_id(mesh.cc_ba, mesh.dm, N_DOM, mesh.dom_data_ngr);

    Gpu::ManagedVector<long> conn;
    Gpu::ManagedVector<Real> nodes, cell_fields, nodal_fields;

    long n_base_bou_cells, n_bou_nodes, n_bou_cells;
    long bou_nodes_pos, bou_conn_pos, bou_cell_fields_pos, bou_nodal_fields_pos;
    amrex::Vector<longMultiFab> output_bou_data_info(AMREX_SPACEDIM);
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        output_bou_data_info[dim].define(mesh.fc_ba[dim], mesh.dm, 4, mesh.bou_data_ngr);
    }

    Gpu::ManagedVector<long> bou_conn;
    Gpu::ManagedVector<Real> bou_nodes, bou_cell_fields, bou_nodal_fields;
    // ================================================================

    // COUNT THE AMOUNT OF MEMORY REQUIRED BY THE OUTPUT DATA =========
    n_base_cells = 0L;
    
    nodes_pos = 0L;
    conn_pos = 0L;
    cell_fields_pos = 0L;
    nodal_fields_pos = 0L;

    for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);
        const Dim3 len = length(bx);

        Array4<long> const & output_data_info_fab = output_data_info.array(mfi);

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // UPDATE INFO
            output_data_info_fab(i,j,k,0) = nodes_pos;
            output_data_info_fab(i,j,k,1) = conn_pos;
            output_data_info_fab(i,j,k,2) = cell_fields_pos;
            output_data_info_fab(i,j,k,3) = nodal_fields_pos;

            nodes_pos += cell_dom_Ng*AMREX_SPACEDIM;
            conn_pos += cell_conn_len;
            cell_fields_pos += cell_dom_Ne*n_cell_fields;
            nodal_fields_pos += cell_dom_Ng*n_nodal_fields;
        }

        n_base_cells += len.x*len.y*len.z;
    }
    
    n_nodes = nodes_pos/AMREX_SPACEDIM;
    nodes.resize(nodes_pos);

    n_cells = n_base_cells*cell_dom_Ne;
    conn.resize(conn_pos);

    cell_fields.resize(cell_fields_pos);

    nodal_fields.resize(nodal_fields_pos);

    // SET POINTERS TO MEMORY
    Real * nodes_ptr = nodes.data();
    long * conn_ptr = conn.data();
    Real * cell_fields_ptr = cell_fields.data();
    Real * nodal_fields_ptr = nodal_fields.data();
    // ================================================================

    // COUNT THE AMOUNT OF MEMORY REQUIRED BY THE OUTPUT BOU DATA =====
    n_base_bou_cells = 0L;
    
    bou_nodes_pos = 0L;
    bou_conn_pos = 0L;
    bou_cell_fields_pos = 0L;
    bou_nodal_fields_pos = 0L;

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        for (MFIter mfi(output_bou_data_info[dim]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const Dim3 lo = lbound(bx);
            const Dim3 hi = ubound(bx);
            const Dim3 len = length(bx);

            Array4<long> const & output_bou_data_info_fab = output_bou_data_info[dim].array(mfi);

            for (int fk = lo.z; fk <= hi.z; ++fk)
            for (int fj = lo.y; fj <= hi.y; ++fj)
            for (int fi = lo.x; fi <= hi.x; ++fi)
            {
                // UPDATE INFO
                output_bou_data_info_fab(fi,fj,fk,0) = bou_nodes_pos;
                output_bou_data_info_fab(fi,fj,fk,1) = bou_conn_pos;
                output_bou_data_info_fab(fi,fj,fk,2) = bou_cell_fields_pos;
                output_bou_data_info_fab(fi,fj,fk,3) = bou_nodal_fields_pos;

                bou_nodes_pos += cell_bou_Ng*AMREX_SPACEDIM;
                bou_conn_pos += bou_cell_conn_len;
                bou_cell_fields_pos += cell_bou_Ne*n_bou_cell_fields;
                bou_nodal_fields_pos += cell_bou_Ng*n_bou_nodal_fields;
            }

            n_base_bou_cells += len.x*len.y*len.z;
        }
    }

    n_bou_nodes = bou_nodes_pos/AMREX_SPACEDIM;
    bou_nodes.resize(bou_nodes_pos);

    n_bou_cells = n_base_bou_cells*cell_bou_Ne;
    bou_conn.resize(bou_conn_pos);

    bou_cell_fields.resize(bou_cell_fields_pos);

    bou_nodal_fields.resize(bou_nodal_fields_pos);

    // SET POINTERS TO MEMORY
    Real * bou_nodes_ptr = bou_nodes.data();
    long * bou_conn_ptr = bou_conn.data();
    Real * bou_cell_fields_ptr = bou_cell_fields.data();
    Real * bou_nodal_fields_ptr = bou_nodal_fields.data();
    // ================================================================

    // POPULATE THE ELEMENT ID (RANDOM COLORING) ======================
    for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);

        Array4<long> const & elm_id_fab = elm_id.array(mfi);

        for (int dom = 0; dom < N_DOM; ++dom)
        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            elm_id_fab(i,j,k,dom) = Random_int(N_DOM*n_cells);
        }
    }
    elm_id.FillBoundary(mesh.geom.periodicity());
    // ================================================================

    // POPULATE OUTPUT DATA ===========================================
    long bx_id;
    bx_id = 0L;

    for (MFIter mfi(output_data_info); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        Array4<long const> const & output_data_info_fab = output_data_info.array(mfi);
        Array4<long const> const & elm_id_fab = elm_id.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<int const> const & eColor_fab = mesh.eColor.array(mfi);
        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};
            const long nodes_offset = output_data_info_fab(i,j,k,0);
            const long conn_offset = output_data_info_fab(i,j,k,1);
            const long cell_fields_offset = output_data_info_fab(i,j,k,2);
            const long nodal_fields_offset = output_data_info_fab(i,j,k,3);

            // LOCAL VARIABLES
            int etype;
            int BF_i, BF_j, BF_k;
            Real x[AMREX_SPACEDIM];

            // Data structures used to compute the level set
            // functions
            DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, p, i, j, k, -1);

            // UPDATE INFO: NODES AND NODAL FIELDS
            for (int n = 0; n < cell_dom_Ng; ++n)
            {
                // NODES
                AMREX_D_TERM
                (
                    x[0] = J[0]*dom_XI1g_ptr[n]+cell_m[0];,
                    x[1] = J[1]*dom_XI2g_ptr[n]+cell_m[1];,
                    x[2] = J[2]*dom_XI3g_ptr[n]+cell_m[2];
                )
                AMREX_D_TERM
                (
                    nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+0] = x[0];,
                    nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+1] = x[1];,
                    nodes_ptr[nodes_offset+AMREX_SPACEDIM*n+2] = x[2];
                )

                phi._eval_BF_table_(x);
                for (int rph = 0; rph < N_PHI; ++rph)
                {
                    phi.set_fab_ptrs(i, j, k, rph);

                    nodal_fields_ptr[nodal_fields_offset+n_nodal_fields*n+rph] = phi._eval_unsafe_();
                }
            }

            // UPDATE INFO: CONNECTIVITY
            for (int c = 0; c < cell_conn_len; ++c)
            {
                conn_ptr[conn_offset+c] = cell_base_conn_ptr[c]+nodes_offset/AMREX_SPACEDIM;
            }

            // UPDATE INFO: CELL FIELDS
            for (int c = 0; c < cell_dom_Ne; ++c)
            {
                cell_fields_ptr[cell_fields_offset+c*n_cell_fields+0] = rank*1.0;
                cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1] = bx_id;

                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    BF_CELL(i, j, k, etype, BF_i, BF_j, BF_k);

                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1+1+dom] = (etype%10)*1.0;
                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1+1+N_DOM+dom] = eColor_fab(BF_i,BF_j,BF_k,dom)*1.0;
                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+1+1+N_DOM+N_DOM+dom] = elm_id_fab(BF_i,BF_j,BF_k,dom);
                }
            }
        });
        Gpu::synchronize();

        bx_id += 1;   
    }
    // ================================================================

    // POPULATE OUTPUT BOU DATA =======================================
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim)
    {
        // A few auxiliary constants to avoid a bunch of ifs
#if (AMREX_SPACEDIM == 3)
        const int ci = ((1-dim)*(2-dim))/2;
        const int cj = (-dim*dim+dim+4)/2;
#endif
        const Real d0 = 0.5*(dim-1.0)*(dim-2.0);
        const Real d1 = dim*(2.0-dim);
        const Real nd0 = 1.0-d0;
        const Real nd1 = 1.0-d1;
#if (AMREX_SPACEDIM == 3)
        const Real d2 = 0.5*dim*(dim-1);
        const Real nd2 = 1.0-d2;
#endif

        for (MFIter mfi(output_bou_data_info[dim]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();

            Array4<long> const & output_bou_data_info_fab = output_bou_data_info[dim].array(mfi);
            Array4<short const> const & eType_fab = mesh.eType.array(mfi);
            Array4<char const> const & tags_fab = tags.array(mfi);
            Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int fi, int fj, int fk) noexcept
            {
                // LOCAL PARAMETERS
                const Real face_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(fi+0.5*nd0)*dx[0],
                                                                  prob_lo[1]+(fj+0.5*nd1)*dx[1],
                                                                  prob_lo[2]+(fk+0.5*nd2)*dx[2])};
                const long bou_nodes_offset = output_bou_data_info_fab(fi,fj,fk,0);
                const long bou_conn_offset = output_bou_data_info_fab(fi,fj,fk,1);
                const long bou_cell_fields_offset = output_bou_data_info_fab(fi,fj,fk,2);
                const long bou_nodal_fields_offset = output_bou_data_info_fab(fi,fj,fk,3);

                // LOCAL VARIABLES
                int mi, mj, mk, pi, pj, pk;
                Real m_cell_lo[AMREX_SPACEDIM], m_cell_hi[AMREX_SPACEDIM];
                Real p_cell_lo[AMREX_SPACEDIM], p_cell_hi[AMREX_SPACEDIM];
                int m_etype, p_etype;
                bool m_elm_is_small, p_elm_is_small, cells_are_merged;
                int m_merged_b, p_merged_b;
                char m_tag, p_tag;
                Real x[AMREX_SPACEDIM];

                // PAIRED CELLS SHARING THE FACE fi,fj,fk
                FACE_2_NBRS(fi, fj, fk, dim, mi, mj, mk, pi, pj, pk);

                AMREX_D_TERM
                (
                    m_cell_lo[0] = prob_lo[0]+mi*dx[0];,
                    m_cell_lo[1] = prob_lo[1]+mj*dx[1];,
                    m_cell_lo[2] = prob_lo[2]+mk*dx[2];
                )
                AMREX_D_TERM
                (
                    m_cell_hi[0] = prob_lo[0]+(mi+1)*dx[0];,
                    m_cell_hi[1] = prob_lo[1]+(mj+1)*dx[1];,
                    m_cell_hi[2] = prob_lo[2]+(mk+1)*dx[2];
                )
                AMREX_D_TERM
                (
                    p_cell_lo[0] = prob_lo[0]+pi*dx[0];,
                    p_cell_lo[1] = prob_lo[1]+pj*dx[1];,
                    p_cell_lo[2] = prob_lo[2]+pk*dx[2];
                )
                AMREX_D_TERM
                (
                    p_cell_hi[0] = prob_lo[0]+(pi+1)*dx[0];,
                    p_cell_hi[1] = prob_lo[1]+(pj+1)*dx[1];,
                    p_cell_hi[2] = prob_lo[2]+(pk+1)*dx[2];
                )

                // Data structures used to compute the level set
                // functions
                DG_phi_space_BFX m_phi(m_cell_lo, m_cell_hi, PHI_fab, p, mi, mj, mk, -1);
                DG_phi_space_BFX p_phi(p_cell_lo, p_cell_hi, PHI_fab, p, pi, pj, pk, -1);

                // UPDATE INFO: NODES AND NODAL FIELDS
                for (int n = 0; n < cell_bou_Ng; ++n)
                {                        
                    // NODES
#if (AMREX_SPACEDIM == 1)
                    x[0] = face_m[0];
#endif
#if (AMREX_SPACEDIM == 2)
                    x[dim] = face_m[dim];
                    x[1-dim] = J[1-dim]*bou_XI1g_ptr[n]+face_m[1-dim];
#endif
#if (AMREX_SPACEDIM == 3)
                    x[dim] = face_m[dim];
                    x[ci] = J[ci]*bou_XI1g_ptr[n]+face_m[ci];
                    x[cj] = J[cj]*bou_XI2g_ptr[n]+face_m[cj];
#endif
                    AMREX_D_TERM
                    (
                        bou_nodes_ptr[bou_nodes_offset+AMREX_SPACEDIM*n+0] = x[0];,
                        bou_nodes_ptr[bou_nodes_offset+AMREX_SPACEDIM*n+1] = x[1];,
                        bou_nodes_ptr[bou_nodes_offset+AMREX_SPACEDIM*n+2] = x[2];
                    )

                    m_phi._eval_BF_table_(x);
                    p_phi._eval_BF_table_(x);
                    for (int rph = 0; rph < N_PHI; ++rph)
                    {
                        m_phi.set_fab_ptrs(mi, mj, mk, rph);
                        p_phi.set_fab_ptrs(pi, pj, pk, rph);

                        bou_nodal_fields_ptr[bou_nodal_fields_offset+n_bou_nodal_fields*n+rph] = 0.5*(m_phi._eval_unsafe_()+p_phi._eval_unsafe_());
                    }
                }

                // UPDATE INFO: CONNECTIVITY
                for (int c = 0; c < bou_cell_conn_len; ++c)
                {
                    bou_conn_ptr[bou_conn_offset+c] = bou_cell_base_conn_ptr[c]+bou_nodes_offset/AMREX_SPACEDIM;
                }

                // UPDATE INFO: CELL FIELDS
                for (int c = 0; c < cell_bou_Ne; ++c)
                {
                    for (int dom = 0; dom < N_DOM; ++dom)
                    {
                        m_etype = eType_fab(mi,mj,mk,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                        p_etype = eType_fab(pi,pj,pk,DG_ELM_TYPE_N_COMP_PER_DOM*dom);

                        m_elm_is_small = (m_etype%10 == DG_ELM_TYPE_SMALL);
                        p_elm_is_small = (p_etype%10 == DG_ELM_TYPE_SMALL);

                        m_merged_b = (m_elm_is_small) ? (m_etype/10) : -1;
                        p_merged_b = (p_elm_is_small) ? (p_etype/10) : -1;

                        cells_are_merged = ((m_merged_b == 2*dim+1) || (p_merged_b == 2*dim));

                        m_tag = tags_fab(mi,mj,mk,dom);
                        p_tag = tags_fab(pi,pj,pk,dom);
            
                        if (cells_are_merged || ((m_tag == TROUBLED_ELEMENT) && (p_tag == TROUBLED_ELEMENT)))
                        {
                            bou_cell_fields_ptr[bou_cell_fields_offset+c*n_bou_cell_fields+dom] = 1.0;
                        }
                        else
                        {
                            bou_cell_fields_ptr[bou_cell_fields_offset+c*n_bou_cell_fields+dom] = 0.0;
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
    // ================================================================

    // ALIGN PROCESSORS ==========
    ParallelDescriptor::Barrier();
    // ===========================

    // OUTPUT FILES: XML FORMAT =======================================
    {
        // VARIABLES ----------------------------------------------
        Vector<VTK::Float_t> VTK_nodes;
        Vector<VTK::Cell_conn_t> VTK_cell_conn;
        Vector<VTK::Cell_offs_t> VTK_cell_offset;
        Vector<VTK::Cell_type_t> VTK_cell_type;
        Vector<Vector<VTK::Int_t>> VTK_cell_field(n_cell_fields);
        Vector<std::string> VTK_cell_field_name(n_cell_fields);
        Vector<Vector<VTK::Float_t>> VTK_nodal_field(n_nodal_fields);
        Vector<std::string> VTK_nodal_field_name(n_nodal_fields);
        // --------------------------------------------------------

        // RESIZE MEMORY ------------------------------------------
        VTK_nodes.resize(3*n_nodes);
        VTK_cell_conn.resize(AMREX_D_PICK(2, 4, 8)*n_cells);
        VTK_cell_offset.resize(n_cells+1);
        VTK_cell_type.resize(n_cells);
        for (int f = 0; f < n_cell_fields; ++f)
        {
            VTK_cell_field[f].resize(n_cells);
        }
        for (int f = 0; f < n_nodal_fields; ++f)
        {
            VTK_nodal_field[f].resize(n_nodes);
        }
        // --------------------------------------------------------

        // FILL IN DATA -------------------------------------------
        for (int n = 0; n < n_nodes; ++n)
        {
#if (AMREX_SPACEDIM == 2)
            VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
            VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
            VTK_nodes[3*n+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
            VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
            VTK_nodes[3*n+2] = nodes[AMREX_SPACEDIM*n+2];
#endif
        }

        for (int k = 0; k < conn.size(); ++k)
        {
            VTK_cell_conn[k] = conn[k];
        }
        
        VTK_cell_offset[0];
        for (int k = 1; k < (n_cells+1); ++k)
        {
            VTK_cell_offset[k] = VTK_cell_offset[k-1]+AMREX_D_PICK(2, 4, 8);
        }

        for (int k = 0; k < n_cells; ++k)
        {
            VTK_cell_type[k] = VTK_cell_id;
        }

        VTK_cell_field_name[0] = "RANK";
        VTK_cell_field_name[1] = "BOX";
        for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[2+dom] = "ELM_TYPE_DOM_"+std::to_string(dom);
        for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[2+N_DOM+dom] = "ELM_COLOR_DOM_"+std::to_string(dom);
        for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[2+N_DOM+N_DOM+dom] = "ELM_ID_DOM_"+std::to_string(dom);
        for (int f = 0; f < n_cell_fields; ++f)
        {
            for (int n = 0; n < n_cells; ++n)
            {
                VTK_cell_field[f][n] = (VTK::Int_t) cell_fields[n_cell_fields*n+f];
            }
        }

        for (int rph = 0; rph < N_PHI; ++rph) VTK_nodal_field_name[rph] = "PHI_"+std::to_string(rph);

        for (int f = 0; f < n_nodal_fields; ++f)
        {
            for (int n = 0; n < n_nodes; ++n)
            {
                VTK_nodal_field[f][n] = nodal_fields[n_nodal_fields*n+f];
            }
        }
        // --------------------------------------------------------

        // DUMP HEADER FILE ---------------------------------------
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            VTK::PrintHeaderFile_VTU(output_folder, filename_root, time_step, n_time_steps, VTK_cell_field_name, VTK_nodal_field_name);
        }
        // --------------------------------------------------------

        // DUMP DATA FILE -----------------------------------------
        const std::string filepath = GetOutputFilepath(output_folder, filename_root, time_step, n_time_steps)+".vtu";

        VTK::PrintUnstructuredGridData_VTU(filepath,
                                           n_nodes,
                                           n_cells,
                                           VTK_nodes,
                                           VTK_cell_conn,
                                           VTK_cell_offset,
                                           VTK_cell_type,
                                           VTK_cell_field,
                                           VTK_cell_field_name,
                                           VTK_nodal_field,
                                           VTK_nodal_field_name,
                                           "binary");
        // --------------------------------------------------------
    }

    {
        // VARIABLES ----------------------------------------------
        Vector<VTK::Float_t> VTK_nodes;
        Vector<VTK::Cell_conn_t> VTK_cell_conn;
        Vector<VTK::Cell_offs_t> VTK_cell_offset;
        Vector<VTK::Cell_type_t> VTK_cell_type;
        Vector<Vector<VTK::Int_t>> VTK_cell_field(n_bou_cell_fields);
        Vector<std::string> VTK_cell_field_name(n_bou_cell_fields);
        Vector<Vector<VTK::Float_t>> VTK_nodal_field(n_bou_nodal_fields);
        Vector<std::string> VTK_nodal_field_name(n_bou_nodal_fields);
        // --------------------------------------------------------

        // RESIZE MEMORY ------------------------------------------
        VTK_nodes.resize(3*n_bou_nodes);
        VTK_cell_conn.resize(AMREX_D_PICK(1, 2, 4)*n_bou_cells);
        VTK_cell_offset.resize(n_bou_cells+1);
        VTK_cell_type.resize(n_bou_cells);
        for (int f = 0; f < n_bou_cell_fields; ++f)
        {
            VTK_cell_field[f].resize(n_bou_cells);
        }
        for (int f = 0; f < n_bou_nodal_fields; ++f)
        {
            VTK_nodal_field[f].resize(n_bou_nodes);
        }
        // --------------------------------------------------------

        // FILL IN DATA -------------------------------------------
        for (int n = 0; n < n_bou_nodes; ++n)
        {
#if (AMREX_SPACEDIM == 2)
            VTK_nodes[3*n+0] = bou_nodes[AMREX_SPACEDIM*n+0];
            VTK_nodes[3*n+1] = bou_nodes[AMREX_SPACEDIM*n+1];
            VTK_nodes[3*n+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
            VTK_nodes[3*n+0] = bou_nodes[AMREX_SPACEDIM*n+0];
            VTK_nodes[3*n+1] = bou_nodes[AMREX_SPACEDIM*n+1];
            VTK_nodes[3*n+2] = bou_nodes[AMREX_SPACEDIM*n+2];
#endif
        }

        for (int k = 0; k < bou_conn.size(); ++k)
        {
            VTK_cell_conn[k] = bou_conn[k];
        }
        
        VTK_cell_offset[0];
        for (int k = 1; k < (n_bou_cells+1); ++k)
        {
            VTK_cell_offset[k] = VTK_cell_offset[k-1]+AMREX_D_PICK(1, 2, 4);
        }

        for (int k = 0; k < n_bou_cells; ++k)
        {
            VTK_cell_type[k] = VTK_bou_cell_id;
        }

        for (int dom = 0; dom < N_DOM; ++dom) VTK_cell_field_name[dom] = "ELM_BOU_TYPE_DOM_"+std::to_string(dom);
        for (int f = 0; f < n_bou_cell_fields; ++f)
        {
            for (int n = 0; n < n_bou_cells; ++n)
            {
                VTK_cell_field[f][n] = (VTK::Int_t) bou_cell_fields[n_bou_cell_fields*n+f];
            }
        }
        
        for (int rph = 0; rph < N_PHI; ++rph) VTK_nodal_field_name[rph] = "PHI_"+std::to_string(rph);

        for (int f = 0; f < n_bou_nodal_fields; ++f)
        {
            for (int n = 0; n < n_bou_nodes; ++n)
            {
                VTK_nodal_field[f][n] = bou_nodal_fields[n_bou_nodal_fields*n+f];
            }
        }
        // --------------------------------------------------------

        // DUMP HEADER FILE ---------------------------------------
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            VTK::PrintHeaderFile_VTU(output_folder, filename_root+"Bou", time_step, n_time_steps, VTK_cell_field_name, VTK_nodal_field_name);
        }
        // --------------------------------------------------------

        // DUMP DATA FILE -----------------------------------------
        const std::string filepath = GetOutputFilepath(output_folder, filename_root+"Bou", time_step, n_time_steps)+".vtu";

        VTK::PrintUnstructuredGridData_VTU(filepath,
                                           n_bou_nodes,
                                           n_bou_cells,
                                           VTK_nodes,
                                           VTK_cell_conn,
                                           VTK_cell_offset,
                                           VTK_cell_type,
                                           VTK_cell_field,
                                           VTK_cell_field_name,
                                           VTK_nodal_field,
                                           VTK_nodal_field_name,
                                           "binary");
        // --------------------------------------------------------
    }
    // ================================================================

    // ALIGN PROCESSORS ==========
    ParallelDescriptor::Barrier();
    // ===========================
    
}
// ####################################################################

// EXPORT SOLUTION TO VTK FORMAT ######################################
template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ExportSolution_VTK_GetData(const Real t,
                                const ImplicitMesh<N_PHI, N_DOM> & mesh,
                                const MatrixFactory<N_PHI, N_DOM> & matfactory,
                                const MultiFab & X,
                                const charMultiFab & tags,
                                const Vector<int> & fields_domain,
                                const Vector<std::string> & fields_name,
                                const IBVP_CLASS & IBVP,
                                long & n_nodes,
                                long & n_cells,
                                int & n_nodal_fields,
                                Vector<VTK::Float_t> & VTK_nodes,
                                Vector<VTK::Cell_conn_t> & VTK_cell_conn,
                                Vector<VTK::Cell_offs_t> & VTK_cell_offset,
                                Vector<VTK::Cell_type_t> & VTK_cell_type,
                                Vector<Vector<VTK::Float_t>> & VTK_nodal_field,
                                Vector<std::string> & VTK_nodal_field_name,
                                Vector<Vector<VTK::Int_t>> & VTK_cell_field,
                                Vector<std::string> & VTK_cell_field_name)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ExportSolution_VTK_GetData(const Real, const ImplicitMesh<N_PHI, N_DOM> &, ....)");
    // ----------------------------------------------------------------

    // PARAMETERS =====================================================
    // GRID
    const GpuArray<Real, AMREX_SPACEDIM> dx = mesh.dx;
    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = mesh.prob_lo;
    const GpuArray<Real, AMREX_SPACEDIM> J = mesh.J;

    // STANDARD ELEMENT
    const int cell_dom_Ng = std::max(mesh.cube.dom_Ng, matfactory.cube.dom_Ng);
    const int cell_e = std::max(mesh.cube.g, matfactory.cube.g);
    const int cell_dom_Ne = AMREX_D_PICK(cell_e, cell_e*cell_e, cell_e*cell_e*cell_e);
    const int cell_conn_len = AMREX_D_PICK(2, 4, 8)*cell_dom_Ne;
    const Gpu::ManagedVector<int> cell_base_conn = Get_VTK_BaseCell_Connectivity(cell_e);
    const int * cell_base_conn_ptr = cell_base_conn.data();

    AMREX_D_TERM
    (
        Real const * dom_XI1g_ptr = (mesh.cube.g > matfactory.cube.g) ? mesh.cube.dom_XI1g.data() : matfactory.cube.dom_XI1g.data();,
        Real const * dom_XI2g_ptr = (mesh.cube.g > matfactory.cube.g) ? mesh.cube.dom_XI2g.data() : matfactory.cube.dom_XI2g.data();,
        Real const * dom_XI3g_ptr = (mesh.cube.g > matfactory.cube.g) ? mesh.cube.dom_XI3g.data() : matfactory.cube.dom_XI3g.data();
    )

    // DG
    const int phi_sp = mesh.cube.p;
    const int sol_sp = matfactory.cube.p;

    // CELL-DATA
    // Element tag (per each domain)
    const int n_cell_fields = N_DOM;

    // DOMAINS AND FIELDS
    const int n_requested_fields = fields_domain.size();
    const int n_total_fields = n_requested_fields+N_PHI;
    Gpu::ManagedVector<int> Gpu_field_domains;
    Gpu_field_domains.resize(n_requested_fields);
    for (int f = 0; f < n_requested_fields; ++f) Gpu_field_domains[f] = fields_domain[f];
    const int * field_domains_ptr = Gpu_field_domains.data();

    // VTK
    const int VTK_cell_id = AMREX_D_PICK(3, 9, 12);
    // ================================================================

    // VARIABLES ======================================================
    long n_dG_cells;
    Gpu::ManagedVector<Real> nodes;
    Gpu::ManagedVector<long> conn;

    long nodes_pos, conn_pos, conn_offset;
    longMultiFab grid_info(mesh.cc_ba, mesh.dm, 3, mesh.dom_data_ngr);

    // FIELDS
    Gpu::ManagedVector<Real> fields, cell_fields;

    long fields_pos, cell_fields_pos;
    longMultiFab fields_info(mesh.cc_ba, mesh.dm, 2, mesh.dom_data_ngr);
    // ================================================================

    // SOME CHECKS ====================================================
    if (n_total_fields > DG_MAX_N_FIELDS)
    {
        std::cout << std::endl;
        std::cout << "ERROR: AMReX_DG_Solution.H - ExportSolution_VTK_GetData" << std::endl;
        std::cout << "| The number of requested fields exceeds the number of allowed fields:" << std::endl;
        std::cout << "| DG_MAX_N_FIELDS = " << DG_MAX_N_FIELDS << std::endl;
        std::cout << "| n_total_fields = " << n_total_fields << std::endl;
        std::cout << std::endl;
        exit(-1);
    }
    // ================================================================

    // ALIGN PROCESSORS ==========
    ParallelDescriptor::Barrier();
    // ===========================

    // COLLECT INFORMATION ============================================
    n_dG_cells = 0L;
    nodes_pos = 0L;
    conn_pos = 0L;
    conn_offset = 0L;

    fields_pos = 0L;
    cell_fields_pos = 0L;

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Dim3 lo = lbound(bx);
        const Dim3 hi = ubound(bx);
        const Dim3 len = length(bx);

        Array4<long> const & grid_info_fab = grid_info.array(mfi);
        Array4<long> const & fields_info_fab = fields_info.array(mfi);

        n_dG_cells += len.x*len.y*len.z;

        for (int k = lo.z; k <= hi.z; ++k)
        for (int j = lo.y; j <= hi.y; ++j)
        for (int i = lo.x; i <= hi.x; ++i)
        {
            // UPDATE INFO
            grid_info_fab(i,j,k,0) = nodes_pos;
            grid_info_fab(i,j,k,1) = conn_pos;
            grid_info_fab(i,j,k,2) = conn_offset;

            nodes_pos += cell_dom_Ng*AMREX_SPACEDIM;
            conn_pos += cell_conn_len;
            conn_offset += cell_dom_Ng;

            fields_info_fab(i,j,k,0) = fields_pos;
            fields_info_fab(i,j,k,1) = cell_fields_pos;

            fields_pos += cell_dom_Ng*n_total_fields;
            cell_fields_pos += cell_dom_Ne*n_cell_fields;
        }
    }
    // ================================================================

    // COMPUTE OUTPUT DATA ============================================
    nodes.resize(n_dG_cells*cell_dom_Ng*AMREX_SPACEDIM);
    conn.resize(n_dG_cells*cell_conn_len);
    fields.resize(n_dG_cells*cell_dom_Ng*n_total_fields);
    cell_fields.resize(n_dG_cells*cell_dom_Ne*n_total_fields);

    Real * nodes_ptr = nodes.data();
    long * conn_ptr = conn.data();
    Real * fields_ptr = fields.data();
    Real * cell_fields_ptr = cell_fields.data();

    for (MFIter mfi(X); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        Array4<Real const> const & PHI_fab = mesh.PHI.array(mfi);
        Array4<short const> const & eType_fab = mesh.eType.array(mfi);
        Array4<char const> const & tags_fab = tags.array(mfi);
        Array4<Real const> const & X_fab = X.array(mfi);

        Array4<long const> const & grid_info_fab = grid_info.array(mfi);
        Array4<long const> const & fields_info_fab = fields_info.array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // LOCAL PARAMETERS
            const Real cell_lo[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+i*dx[0],
                                                               prob_lo[1]+j*dx[1],
                                                               prob_lo[2]+k*dx[2])};
            const Real cell_hi[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+1)*dx[0],
                                                               prob_lo[1]+(j+1)*dx[1],
                                                               prob_lo[2]+(k+1)*dx[2])};
            const Real cell_m[AMREX_SPACEDIM] = {AMREX_D_DECL(prob_lo[0]+(i+0.5)*dx[0],
                                                              prob_lo[1]+(j+0.5)*dx[1],
                                                              prob_lo[2]+(k+0.5)*dx[2])};
            const long cell_nodes_pos = grid_info_fab(i,j,k,0);
            const long cell_conn_pos = grid_info_fab(i,j,k,1);
            const long cell_conn_offset = grid_info_fab(i,j,k,2);
            const long nodal_fields_offset = fields_info_fab(i,j,k,0);
            const long cell_fields_offset = fields_info_fab(i,j,k,1);

            // LOCAL VARIABLES
            Real x[AMREX_SPACEDIM], grad[AMREX_SPACEDIM];
            Real PHIh[N_PHI], Uh[N_U];
            Real AMREX_D_DECL(dPHIhdx1[N_PHI], dPHIhdx2[N_PHI], dPHIhdx3[N_PHI]);
            Real AMREX_D_DECL(dUhdx1[N_U], dUhdx2[N_U], dUhdx3[N_U]);
            Real Fh[DG_MAX_N_FIELDS];
            short etype;
            int BF_i, BF_j, BF_k;
            Real BF_lo[AMREX_SPACEDIM], BF_hi[AMREX_SPACEDIM];

            // Data structures used to compute the level set
            // functions and the dG solution
            DG_phi_space_BFX phi(cell_lo, cell_hi, PHI_fab, phi_sp, i, j, k, -1);
            DG_sol_space_BFX sol(X_fab, sol_sp);

            // UPDATE INFO: NODES AND FIELDS
            for (int n = 0; n < cell_dom_Ng; ++n)
            {
                // NODES
                AMREX_D_TERM
                (
                    x[0] = J[0]*dom_XI1g_ptr[n]+cell_m[0];,
                    x[1] = J[1]*dom_XI2g_ptr[n]+cell_m[1];,
                    x[2] = J[2]*dom_XI3g_ptr[n]+cell_m[2];
                )
                AMREX_D_TERM
                (
                    nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+0] = x[0];,
                    nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+1] = x[1];,
                    nodes_ptr[cell_nodes_pos+AMREX_SPACEDIM*n+2] = x[2];
                )

                // FIELDS (LEVEL SET FUNCTIONS)
                phi._eval_BF_table_(x);
                for (int rph = 0; rph < N_PHI; ++rph)
                {
                    phi.set_fab_ptrs(i, j, k, rph);

                    PHIh[rph] = phi._eval_unsafe_();

                    phi._eval_grad_unsafe_(grad);
                    AMREX_D_TERM
                    (
                        dPHIhdx1[rph] = grad[0];,
                        dPHIhdx2[rph] = grad[1];,
                        dPHIhdx3[rph] = grad[2];
                    );

                    fields_ptr[nodal_fields_offset+n_total_fields*n+rph] = PHIh[rph];
                }

                // FIELDS
                for (int rf = 0; rf < n_requested_fields; ++rf)
                {
                    const int dom = field_domains_ptr[rf];
                    etype = eType_fab(i,j,k,DG_ELM_TYPE_N_COMP_PER_DOM*dom);
                    BF_CELL(&prob_lo[0], &dx[0], i, j, k, etype, BF_i, BF_j, BF_k, BF_lo, BF_hi);

                    sol.set_support(BF_lo, BF_hi);
                    sol._eval_BF_table_(x);

                    for (int ru = 0; ru < N_U; ++ru)
                    {
                        sol.set_fab_ptrs(BF_i, BF_j, BF_k, ru);

                        Uh[ru] = sol._eval_unsafe_();
                
                        sol._eval_grad_unsafe_(grad);
                        AMREX_D_TERM
                        (
                            dUhdx1[ru] = grad[0];,
                            dUhdx2[ru] = grad[1];,
                            dUhdx3[ru] = grad[2];
                        );
                    }
                    
                    IBVP.F_SOL(t, x,
                               PHIh, AMREX_D_DECL(dPHIhdx1, dPHIhdx2, dPHIhdx3),
                               Uh, AMREX_D_DECL(dUhdx1, dUhdx2, dUhdx3),
                               Fh);

                    fields_ptr[nodal_fields_offset+n_total_fields*n+N_PHI+rf] = Fh[rf];
                }
            }

            // UPDATE INFO: CONNECTIVITY
            for (int c = 0; c < cell_conn_len; ++c)
            {
                conn_ptr[cell_conn_pos+c] = cell_base_conn_ptr[c]+cell_conn_offset;
            }

            // UPDATE INFO: CELL FIELDS
            for (int c = 0; c < cell_dom_Ne; ++c)
            {
                for (int dom = 0; dom < N_DOM; ++dom)
                {
                    cell_fields_ptr[cell_fields_offset+c*n_cell_fields+dom] = 1.0*((int)tags_fab(i,j,k,dom));
                }
            }
        });
        Gpu::synchronize();
    }
    // ================================================================

    // GATHER OUTPUT DATA =============================================
    n_nodes = n_dG_cells*cell_dom_Ng;
    n_cells = conn.size()/AMREX_D_PICK(2, 4, 8);
    n_nodal_fields = n_total_fields;
    VTK_nodal_field.resize(n_nodal_fields);
    VTK_nodal_field_name.resize(n_nodal_fields);
    VTK_cell_field.resize(n_cell_fields);
    VTK_cell_field_name.resize(n_cell_fields);

    // RESIZE MEMORY --------------------------------------------------
    VTK_nodes.resize(3*n_nodes);
    VTK_cell_conn.resize(conn.size());
    VTK_cell_offset.resize(n_cells+1);
    VTK_cell_type.resize(n_cells);
    for (int f = 0; f < n_nodal_fields; ++f)
    {
        VTK_nodal_field[f].resize(n_nodes);
    }
    for (int f = 0; f < n_cell_fields; ++f)
    {
        VTK_cell_field[f].resize(n_cells);
    }
    // ----------------------------------------------------------------

    // FILL IN DATA ---------------------------------------------------
    for (int n = 0; n < n_nodes; ++n)
    {
#if (AMREX_SPACEDIM == 2)
        VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
        VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
        VTK_nodes[3*n+2] = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
        VTK_nodes[3*n+0] = nodes[AMREX_SPACEDIM*n+0];
        VTK_nodes[3*n+1] = nodes[AMREX_SPACEDIM*n+1];
        VTK_nodes[3*n+2] = nodes[AMREX_SPACEDIM*n+2];
#endif
    }

    for (int k = 0; k < conn.size(); ++k)
    {
        VTK_cell_conn[k] = conn[k];
    }
    
    VTK_cell_offset[0];
    for (int k = 1; k < (n_cells+1); ++k)
    {
        VTK_cell_offset[k] = VTK_cell_offset[k-1]+AMREX_D_PICK(2, 4, 8);
    }

    for (int k = 0; k < n_cells; ++k)
    {
        VTK_cell_type[k] = VTK_cell_id;
    }

    for (int rph = 0; rph < N_PHI; ++rph)
    {
        VTK_nodal_field_name[rph] = "PHI_"+std::to_string(rph);
    }
    for (int rf = 0; rf < n_requested_fields; ++rf)
    {
        VTK_nodal_field_name[N_PHI+rf] = fields_name[rf];
    }

    for (int f = 0; f < n_nodal_fields; ++f)
    {
        for (int n = 0; n < n_nodes; ++n)
        {
            VTK_nodal_field[f][n] = fields[n_nodal_fields*n+f];
        }
    }

    for (int dom = 0; dom < N_DOM; ++dom)
    {
        VTK_cell_field_name[dom] = "ELM_TAG_DOM_"+std::to_string(dom);
    }

    for (int f = 0; f < n_cell_fields; ++f)
    {
        for (int n = 0; n < n_cells; ++n)
        {
            VTK_cell_field[f][n] = (VTK::Int_t) cell_fields[n_cell_fields*n+f];
        }
    }
    // ----------------------------------------------------------------
    // ================================================================
}

template<int N_PHI, int N_DOM, int N_U, typename IBVP_CLASS>
void ExportSolution_VTK(const std::string & output_folder,
                        const std::string & filename_root,
                        const int time_step,
                        const int n_time_steps,
                        const Real t,
                        const ImplicitMesh<N_PHI, N_DOM> & mesh,
                        const MatrixFactory<N_PHI, N_DOM> & matfactory,
                        const MultiFab & X,
                        const charMultiFab & tags,
                        const Vector<int> & fields_domain,
                        const Vector<std::string> & fields_name,
                        const IBVP_CLASS & IBVP)
{
    // PROFILING ------------------------------------------------------
    BL_PROFILE("ExportSolution_VTK(const std::string &, const std::string &, ....)");
    // ----------------------------------------------------------------

    // VARIABLES ======================================================
    // VTK DATA
    long n_nodes, n_cells;
    int n_nodal_fields;

    Vector<VTK::Float_t> VTK_nodes;
    Vector<VTK::Cell_conn_t> VTK_cell_conn;
    Vector<VTK::Cell_offs_t> VTK_cell_offset;
    Vector<VTK::Cell_type_t> VTK_cell_type;
    Vector<Vector<VTK::Float_t>> VTK_nodal_field;
    Vector<std::string> VTK_nodal_field_name;
    Vector<Vector<VTK::Int_t>> VTK_cell_field;
    Vector<std::string> VTK_cell_field_name;

    std::string filepath;
    // ================================================================

    // DUMP DATA ======================================================
    // GET DATA TO BE DUMPED ------------------------------------------
    ExportSolution_VTK_GetData<N_PHI, N_DOM, N_U>(t, mesh, matfactory, X, tags,
                                                  fields_domain, fields_name,
                                                  IBVP,
                                                  n_nodes, n_cells, n_nodal_fields,
                                                  VTK_nodes,
                                                  VTK_cell_conn, VTK_cell_offset, VTK_cell_type,
                                                  VTK_nodal_field, VTK_nodal_field_name,
                                                  VTK_cell_field, VTK_cell_field_name);
    // ----------------------------------------------------------------

    // DUMP HEADER FILE -----------------------------------------------
    if (ParallelDescriptor::IOProcessor())
    {
        VTK::PrintHeaderFile_VTU(output_folder, filename_root, time_step, n_time_steps, VTK_cell_field_name, VTK_nodal_field_name);
    }
    // ----------------------------------------------------------------

    // DUMP DATA FILE -------------------------------------------------
    filepath = GetOutputFilepath(output_folder, filename_root, time_step, n_time_steps)+".vtu";

    VTK::PrintUnstructuredGridData_VTU(filepath,
                                        n_nodes,
                                        n_cells,
                                        VTK_nodes,
                                        VTK_cell_conn,
                                        VTK_cell_offset,
                                        VTK_cell_type,
                                        VTK_cell_field,
                                        VTK_cell_field_name,
                                        VTK_nodal_field,
                                        VTK_nodal_field_name,
                                        "binary");
    // ----------------------------------------------------------------
    // ================================================================

    // ALIGN PROCESSORS ==========
    ParallelDescriptor::Barrier();
    // ===========================
}
// ####################################################################

} // namespace DG
} // namespace amrex

#endif

/*

if (((i == 9 && k == 7) || (i == 7 && k == 9)) && (j == 9 || j == 7) && (ru == (N_U-1)))
{
PrintElementInfo(&prob_lo[0], &dx[0], i, j, k, dom, eType_fab);
Print() << " | dom_Nq: " << dom_Nq << std::endl;
}

if (((i == 9 && k == 7) || (i == 7 && k == 9)) && (j == 9 || j == 7) && (ru == (N_U-1)))
{
Real Uh[N_U];
for (int cu = 0; cu < N_U; ++cu)
{
    Uh[cu] = X_fab(i,j,k,cu);
}
Print() << "Uh: "; DG_utils::PrintRealArray2D(1, N_U, Uh);
}

*/