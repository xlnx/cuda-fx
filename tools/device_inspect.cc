#include <cudafx/device.hpp>
#include <VMUtils/fmt.hpp>

using namespace vm;
namespace fx = cufx;
using namespace std;

int main(int argc, char **argv) {
  auto devices = fx::Device::scan();
  println("device count: {}", devices.size());

  for (int i = 0; i != devices.size(); ++i) {
    auto &device = devices[i];
    auto props = device.props();
    println("\ndevice #{}", i);
#define PRINT_PROPERTY(prop) println("{} = {}", #prop, props.prop)
    PRINT_PROPERTY(name);
    PRINT_PROPERTY(ECCEnabled);
    PRINT_PROPERTY(asyncEngineCount);
    PRINT_PROPERTY(canMapHostMemory);
    PRINT_PROPERTY(canUseHostPointerForRegisteredMem);
    PRINT_PROPERTY(clockRate);
    PRINT_PROPERTY(computeMode);
    PRINT_PROPERTY(computePreemptionSupported);
    PRINT_PROPERTY(concurrentKernels);
    PRINT_PROPERTY(concurrentManagedAccess);
    PRINT_PROPERTY(cooperativeLaunch);
    PRINT_PROPERTY(cooperativeMultiDeviceLaunch);
    PRINT_PROPERTY(deviceOverlap);
    PRINT_PROPERTY(directManagedMemAccessFromHost);
    PRINT_PROPERTY(globalL1CacheSupported);
    PRINT_PROPERTY(hostNativeAtomicSupported);
    PRINT_PROPERTY(integrated);
    PRINT_PROPERTY(isMultiGpuBoard);
    PRINT_PROPERTY(kernelExecTimeoutEnabled);
    PRINT_PROPERTY(l2CacheSize);
    PRINT_PROPERTY(localL1CacheSupported);
    PRINT_PROPERTY(luid);
    PRINT_PROPERTY(luidDeviceNodeMask);
    PRINT_PROPERTY(major);
    PRINT_PROPERTY(managedMemory);
    PRINT_PROPERTY(maxGridSize);
    PRINT_PROPERTY(maxSurface1D);
    PRINT_PROPERTY(maxSurface1DLayered);
    PRINT_PROPERTY(maxSurface2D);
    PRINT_PROPERTY(maxSurface2DLayered);
    PRINT_PROPERTY(maxSurface3D);
    PRINT_PROPERTY(maxSurfaceCubemap);
    PRINT_PROPERTY(maxSurfaceCubemapLayered);
    PRINT_PROPERTY(maxTexture1D);
    PRINT_PROPERTY(maxTexture1DLayered);
    PRINT_PROPERTY(maxTexture1DLinear);
    PRINT_PROPERTY(maxTexture1DMipmap);
    PRINT_PROPERTY(maxTexture2D);
    PRINT_PROPERTY(maxTexture2DGather);
    PRINT_PROPERTY(maxTexture2DLayered);
    PRINT_PROPERTY(maxTexture2DLinear);
    PRINT_PROPERTY(maxTexture2DMipmap);
    PRINT_PROPERTY(maxTexture3D);
    PRINT_PROPERTY(maxTexture3DAlt);
    PRINT_PROPERTY(maxTextureCubemap);
    PRINT_PROPERTY(maxTextureCubemapLayered);
    PRINT_PROPERTY(maxThreadsDim);
    PRINT_PROPERTY(maxThreadsPerBlock);
    PRINT_PROPERTY(maxThreadsPerMultiProcessor);
    PRINT_PROPERTY(memPitch);
    PRINT_PROPERTY(memoryBusWidth);
    PRINT_PROPERTY(memoryClockRate);
    PRINT_PROPERTY(minor);
    PRINT_PROPERTY(multiGpuBoardGroupID);
    PRINT_PROPERTY(multiProcessorCount);
    PRINT_PROPERTY(pageableMemoryAccess);
    PRINT_PROPERTY(pageableMemoryAccessUsesHostPageTables);
    PRINT_PROPERTY(pciBusID);
    PRINT_PROPERTY(pciDeviceID);
    PRINT_PROPERTY(pciDomainID);
    PRINT_PROPERTY(regsPerBlock);
    PRINT_PROPERTY(regsPerMultiprocessor);
    PRINT_PROPERTY(sharedMemPerBlock);
    PRINT_PROPERTY(sharedMemPerBlockOptin);
    PRINT_PROPERTY(sharedMemPerMultiprocessor);
    PRINT_PROPERTY(singleToDoublePrecisionPerfRatio);
    PRINT_PROPERTY(streamPrioritiesSupported);
    PRINT_PROPERTY(surfaceAlignment);
    PRINT_PROPERTY(tccDriver);
    PRINT_PROPERTY(textureAlignment);
    PRINT_PROPERTY(texturePitchAlignment);
    PRINT_PROPERTY(totalConstMem);
    PRINT_PROPERTY(totalGlobalMem);
    PRINT_PROPERTY(unifiedAddressing);
    // PRINT_PROPERTY(uuid);
    PRINT_PROPERTY(warpSize);
  }
}
