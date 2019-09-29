#pragma once

#include <cuda_runtime.h>

#include <internal/attribute.hpp>
#include <internal/format.hpp>

namespace cufx
{
struct Extent
{
	CUFX_DEFINE_ATTRIBUTE( std::size_t, width );
	CUFX_DEFINE_ATTRIBUTE( std::size_t, height );
	CUFX_DEFINE_ATTRIBUTE( std::size_t, depth );

public:
	std::size_t size() const { return width * height * depth; }
	cudaExtent get() const { return make_cudaExtent( width, height, depth ); }
};

}  // namespace cufx

#define CUFX_DEFINE_VECTOR1234_FMT( T ) \
	CUFX_DEFINE_VECTOR1_FMT( T##1, x )       \
	CUFX_DEFINE_VECTOR2_FMT( T##2, x, y )    \
	CUFX_DEFINE_VECTOR3_FMT( T##3, x, y, z ) \
	CUFX_DEFINE_VECTOR4_FMT( T##4, x, y, z, w )

CUFX_DEFINE_VECTOR1234_FMT( char )
CUFX_DEFINE_VECTOR1234_FMT( uchar )
CUFX_DEFINE_VECTOR1234_FMT( short )
CUFX_DEFINE_VECTOR1234_FMT( ushort )
CUFX_DEFINE_VECTOR1234_FMT( int )
CUFX_DEFINE_VECTOR1234_FMT( uint )
CUFX_DEFINE_VECTOR1234_FMT( long )
CUFX_DEFINE_VECTOR1234_FMT( ulong )
CUFX_DEFINE_VECTOR1234_FMT( longlong )
CUFX_DEFINE_VECTOR1234_FMT( ulonglong )
CUFX_DEFINE_VECTOR1234_FMT( float )
CUFX_DEFINE_VECTOR1234_FMT( double )
CUFX_DEFINE_VECTOR3_FMT( dim3, x, y, z )
CUFX_DEFINE_VECTOR3_FMT( cufx::Extent, width, height, depth )
CUFX_DEFINE_VECTOR3_FMT( cudaExtent, width, height, depth )
