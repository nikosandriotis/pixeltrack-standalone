#ifndef HeterogenousCore_SYCLUtilities_currentDevice_h
#define HeterogenousCore_SYCLUtilities_currentDevice_h

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
//#include "SYCLCore/cudaCheck.h"

namespace cms {
  namespace sycltools {
    inline int currentDevice() {
      int dev;
      dev = dpct::dev_mgr::instance().current_device_id();
      return dev;
    }
  }  // namespace sycltools
}  // namespace cms

#endif
