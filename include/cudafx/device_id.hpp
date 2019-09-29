#pragma once

#include <cuda_runtime.h>

#include <VMUtils/concepts.hpp>

namespace cufx
{
struct DeviceId
{
private:
	struct Lock : vm::NoCopy, vm::NoHeap
	{
		Lock( int _ ) :
		  _( _ ) {}
		Lock( Lock &&_ ) :
		  _( _._ )
		{
			_._ = -1;
		}
		Lock &operator=( Lock && ) = delete;
		~Lock()
		{
			if ( _ >= 0 ) { cudaSetDevice( _ ); }
		}

	private:
		int _;
	};

	using Props = cudaDeviceProp;

public:
	int id() const { return _; }
	bool is_host() const { return _ < 0; }
	bool is_device() const { return _ >= 0; }
	Lock lock() const
	{
		int d = -1;
		if ( _ >= 0 ) {
			cudaGetDevice( &d );
			cudaSetDevice( _ );
		}
		return Lock( d );
	}
	Props props() const
	{
		Props val;
		cudaGetDeviceProperties( &val, _ );
		return val;
	}

	explicit DeviceId( int _ = 0 ) :
	  _( _ ) {}

private:
	int _;
};

}  // namespace cufx
