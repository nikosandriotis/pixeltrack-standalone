#ifndef HeterogenousCore_SYCLUtilities_deviceCount_h
#define HeterogenousCore_SYCLUtilities_deviceCount_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
//#include "SYCLCore/cudaCheck.h"

namespace cms {
  namespace sycltools {
    inline int deviceCount() {
      int ndevices;
      /*
      DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      (ndevices = dpct::dev_mgr::instance().device_count(), 0);
      return ndevices;
    }
  }  // namespace sycltools
}  // namespace cms

#endif
