// f2c.h

#include <cmath>
#include <algorithm>

#ifdef BL_AMREX_H
#else
#define AMREX_GPU_HOST_DEVICE
#define AMREX_FORCE_INLINE inline
#endif

#ifndef BL_F2C_H_
#define BL_F2C_H_

typedef int integer;
typedef char *address;
typedef double doublereal;
typedef float real;
typedef bool logical;

#define TRUE_ (1)
#define FALSE_ (0)

typedef int ftnlen;

namespace amrex
{
namespace linalg
{

// DOUBLE OPERATIONS ##################################################
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
integer i_nint(real *x)
{
return (integer)(*x >= 0 ? std::floor(*x + .5) : -std::floor(.5 - *x));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double pow_di(doublereal *ap, integer *bp)
{
double pow, x;
integer n;
unsigned long u;

pow = 1;
x = *ap;
n = *bp;

if(n != 0)
	{
	if(n < 0)
		{
		n = -n;
		x = 1/x;
		}
	for(u = n; ; )
		{
		if(u & 01)
			pow *= x;
		if(u >>= 1)
			x *= x;
		else
			break;
		}
	}
return(pow);
}

#define log10e 0.43429448190325182765

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double d_lg10(doublereal *x)
{
return( log10e * std::log(*x) );
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double d_sign(doublereal *a, doublereal *b)
{
double x;
x = (*a >= 0 ? *a : - *a);
return( *b >= 0 ? x : -x);
}
// ####################################################################

// STRING OPERATIONS ##################################################
#define NO_OVERWRITE

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int s_len(const char *s) {
    int i;
    for (i = 0; s[i] != '\0'; i++) ;
    return i;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void s_copy(char *a, const char *b, ftnlen la, ftnlen lb)
{
	char *aend;
	const char *bend;

	aend = a + la;

	if(la <= lb)
#ifndef NO_OVERWRITE
		if (a <= b || a >= b + la)
#endif
			while(a < aend)
				*a++ = *b++;
#ifndef NO_OVERWRITE
		else
			for(b += la; a < aend; )
				*--aend = *--b;
#endif

	else {
		bend = b + lb;
#ifndef NO_OVERWRITE
		if (a <= b || a >= bend)
#endif
			while(b < bend)
				*a++ = *b++;
#ifndef NO_OVERWRITE
		else {
			a += lb;
			while(b < bend)
				*--a = *--bend;
			a += lb;
			}
#endif
		while(a < aend)
			*a++ = ' ';
		}
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
integer s_cmp(const char *a0, const char *b0, ftnlen la, ftnlen lb)
{
/*
register unsigned char *a, *aend, *b, *bend;
a = (unsigned char *)a0;
b = (unsigned char *)b0;
*/
const char *a, *aend, *b, *bend;
a = (const char *)a0;
b = (const char *)b0;

aend = a + la;
bend = b + lb;

if(la <= lb)
	{
	while(a < aend)
		if(*a != *b)
			return( *a - *b );
		else
			{ ++a; ++b; }

	while(b < bend)
		if(*b != ' ')
			return( ' ' - *b );
		else	++b;
	}

else
	{
	while(b < bend)
		if(*a == *b)
			{ ++a; ++b; }
		else
			return( *a - *b );
	while(a < aend)
		if(*a != ' ')
			return(*a - ' ');
		else	++a;
	}
return(0);
}
// ####################################################################

}
}

#endif