#pragma once

#include "stream.hpp"
#include "device_id.hpp"
#include <utils/attribute.hpp>

namespace cufx
{
struct KernelLaunchInfo
{
	CUFX_DEFINE_ATTRIBUTE( DeviceId, device ) = DeviceId{ 0 };
	CUFX_DEFINE_ATTRIBUTE( dim3, grid_dim );
	CUFX_DEFINE_ATTRIBUTE( dim3, block_dim );
	CUFX_DEFINE_ATTRIBUTE( std::size_t, shm_per_block );
};

template <typename F>
struct Kernel;

template <typename Ret, typename... Args>
struct Kernel<Ret( Args... )>
{
private:
	using Launcher = void( KernelLaunchInfo const &, Args... args, cudaStream_t );

public:
	Kernel( Launcher *_ ) :
	  _( _ ) {}

	Task operator()( KernelLaunchInfo const &info, Args... args )
	{
		return Task( [=]( cudaStream_t stream ) { _( info, args..., stream ); } );
	}

private:
	Launcher *_;
};

template <typename F>
struct Functionlify;

template <typename Ret, typename... Args>
struct Functionlify<Ret( Args... )>
{
	using type = Ret( Args... );
};

template <typename Ret, typename... Args>
struct Functionlify<Ret ( * )( Args... )> : Functionlify<Ret( Args... )>
{
};

template <typename Ret, typename... Args>
struct Functionlify<Ret ( *const )( Args... )> : Functionlify<Ret( Args... )>
{
};

/* clang-format off */
#define VOL_DEFINE_CUDA_KERNEL(name, impl)                                     \
  namespace {                                                                  \
  template <typename F>                                                        \
  struct __Kernel_Impl_##name;                                                 \
  template <typename Ret, typename... Args>                                    \
  struct __Kernel_Impl_##name<Ret(Args...)> {                                  \
    static void launch(vol::cufx::KernelLaunchInfo const &info, Args... args,  \
                       cudaStream_t stream) {                                  \
      auto lock = info.device.lock();                                          \
      impl<<<info.grid_dim, info.block_dim, info.shm_per_block, stream>>>      \
          (args...);                                                           \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  ::vol::cufx::Kernel<                                                         \
      typename ::vol::cufx::Functionlify<decltype(impl)>::type>                \
  name(__Kernel_Impl_##name<                                                   \
      typename ::vol::cufx::Functionlify<decltype(impl)>::type>::launch)
/* clang-format on */

// #define VOL_CUDA_DEVICE __device__
// #define VOL_CUDA_DEVICE __device__

}  // namespace cufx
