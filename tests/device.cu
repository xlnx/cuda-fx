#include <gtest/gtest.h>
#include <cudafx/device.hpp>

TEST(test_device, test_device)
{
    auto devices = cufx::Device::scan();
    std::cout << devices.size() << std::endl;
}
