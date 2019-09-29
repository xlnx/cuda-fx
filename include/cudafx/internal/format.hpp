#pragma once

#include <iostream>

#define CUFX_DEFINE_VECTOR1_FMT( T, x )                             \
	inline std::ostream &operator<<( std::ostream &os, T const &_ ) \
	{                                                               \
		return os << "(" << _.x << ")";                             \
	}

#define CUFX_DEFINE_VECTOR2_FMT( T, x, y )                          \
	inline std::ostream &operator<<( std::ostream &os, T const &_ ) \
	{                                                               \
		return os << "(" << _.x << ", " << _.y << ")";              \
	}

#define CUFX_DEFINE_VECTOR3_FMT( T, x, y, z )                       \
	inline std::ostream &operator<<( std::ostream &os, T const &_ ) \
	{                                                               \
		return os << "(" << _.x << ", " << _.y << ", "              \
				  << _.z << ")";                                    \
	}

#define CUFX_DEFINE_VECTOR4_FMT( T, x, y, z, w )                    \
	inline std::ostream &operator<<( std::ostream &os, T const &_ ) \
	{                                                               \
		return os << "(" << _.x << ", " << _.y << ", "              \
				  << _.z << ", " << _.w << ")";                     \
	}
