#ifndef HeterogenousCore_SYCLUtilities_deviceCount_h
#define HeterogenousCore_SYCLUtilities_deviceCount_h

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
//#include "SYCLCore/cudaCheck.h"

namespace cms {
  namespace sycltools {
    inline int deviceCount() {
      (sycl::device::get_devices(sycl::info::device_type::all)).size();
    }
  }  // namespace sycltools
}  // namespace cms

#endif
