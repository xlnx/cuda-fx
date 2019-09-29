#pragma once

#include "stream.hpp"
#include "device_id.hpp"
#include "misc.hpp"

#include "internal/attribute.hpp"

namespace cufx
{
struct GlobalMemory;

namespace _
{
template <typename T, std::size_t N>
struct MemoryViewND
{
	__host__ __device__ T *data() const { return reinterpret_cast<T *>( _.ptr ); }
	__host__ __device__ explicit operator bool() const { return _.ptr; }
	cudaPitchedPtr get() const { return _; }
	DeviceId device_id() const { return device; }

protected:
	cudaPitchedPtr _ = { 0 };
	DeviceId device = DeviceId{ -1 };
	friend struct cufx::GlobalMemory;
};
}  // namespace _

template <typename T, std::size_t N>
struct MemoryViewND;

struct MemoryView2DInfo
{
	CUFX_DEFINE_ATTRIBUTE( std::size_t, stride ) = 0;
	CUFX_DEFINE_ATTRIBUTE( std::size_t, width ) = 0;
	CUFX_DEFINE_ATTRIBUTE( std::size_t, height ) = 0;
};

template <typename T>
struct MemoryViewND<T, 1> : _::MemoryViewND<T, 1>
{
	__host__ __device__ T &at( std::size_t x ) const
	{
		return reinterpret_cast<T *>( this->_.ptr )[ x ];
	}
	__host__ __device__ std::size_t size() { return this->_.xsize; }

public:
	MemoryViewND() = default;
	MemoryViewND( void *ptr, std::size_t len )
	{
		this->_ = make_cudaPitchedPtr( ptr, 0, len, 0 );
	}
};

template <typename T>
struct MemoryViewND<T, 2> : _::MemoryViewND<T, 2>
{
	__host__ __device__ T &at( std::size_t x, std::size_t y ) const
	{
		auto ptr = reinterpret_cast<char *>( this->_.ptr );
		auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
		return line[ x ];
	}
	__host__ __device__ std::size_t width() const { return this->_.xsize; }
	__host__ __device__ std::size_t height() const { return this->_.ysize; }

public:
	MemoryViewND() = default;
	MemoryViewND( void *ptr, MemoryView2DInfo const &info )
	{
		this->_ = make_cudaPitchedPtr( ptr, info.stride,
									   info.width, info.height );
	}
};

template <typename T>
struct MemoryViewND<T, 3> : _::MemoryViewND<T, 3>
{
	// __host__ __device__ T &at( std::size_t x, std::size_t y ) const
	// {
	// 	auto ptr = reinterpret_cast<char *>( this->_.ptr );
	// 	auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
	// 	return line[ x ];
	// }
	__host__ __device__ cudaExtent extent() const { return dim.get(); }

public:
	MemoryViewND() = default;
	MemoryViewND( void *ptr, MemoryView2DInfo const &info, cufx::Extent dim ) :
	  dim( dim )
	{
		this->_ = make_cudaPitchedPtr( ptr, info.stride,
									   info.width, info.height );
	}

private:
	cufx::Extent dim;
};

struct GlobalMemory
{
private:
	struct Inner : vm::NoCopy, vm::NoMove
	{
		~Inner() { cudaFree( _ ); }

		char *_;
		std::size_t size;
		DeviceId device;
	};

public:
	GlobalMemory( std::size_t size, DeviceId const &device = DeviceId{} )
	{
		auto lock = device.lock();
		cudaMalloc( &_->_, _->size = size );
		_->device = device;
	}

	std::size_t size() const { return _->size; }

public:
	template <typename T>
	MemoryViewND<T, 2> view_2d( MemoryView2DInfo const &info, std::size_t offset = 0 ) const
	{
		auto mem = MemoryViewND<T, 2>( _->_ + offset, info );
		static_cast<_::MemoryViewND<T, 2> &>( mem ).device = _->device;
		return mem;
	}

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

template <typename T>
using MemoryView1D = MemoryViewND<T, 1>;
template <typename T>
using MemoryView2D = MemoryViewND<T, 2>;
template <typename T>
using MemoryView3D = MemoryViewND<T, 3>;

}  // namespace cufx
