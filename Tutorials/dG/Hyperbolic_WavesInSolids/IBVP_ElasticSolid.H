//
// Author : Vincenzo Gulizzi (LBL)
// Contact: vgulizzi@lbl.gov
//
/** \file IBVP_ElasticSolid.H
 * \brief Contains base constants and data structures for IBVP of elastic solids.
*/

// IBVP PARAMETERS ####################################################
// NUMBER OF UNKNOWN SOLUTION FIELDS
#define NV AMREX_SPACEDIM
#define NS AMREX_D_PICK(1, 3, 6)
#define ELASTIC_SOLID_NU (NV+NS)

// SOLUTION FIELDS ORDER
#if (AMREX_SPACEDIM == 1)
#define V1 0
#define E11 1
#define S11 1
#endif
#if (AMREX_SPACEDIM == 2)
#define V1 0
#define V2 1
#define E11 2
#define E22 3
#define E12 4
#define S11 2
#define S22 3
#define S12 4
#endif
#if (AMREX_SPACEDIM == 3)
#define V1 0
#define V2 1
#define V3 2
#define E11 3
#define E22 4
#define E33 5
#define E23 6
#define E13 7
#define E12 8
#define S11 3
#define S22 4
#define S33 5
#define S23 6
#define S13 7
#define S12 8
#endif

// ORDERING OF THE ELASTIC COEFFICIENTS
#if (AMREX_SPACEDIM == 2)
#define C11 0
#define C21 1
#define C61 2

#define C22 3
#define C62 4

#define C66 5
#endif
#if (AMREX_SPACEDIM == 3)
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
#endif
// ####################################################################



namespace elastic_solid
{
    
}