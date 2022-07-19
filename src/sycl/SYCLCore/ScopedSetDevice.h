#ifndef HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h
#define HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
//#include "SYCLCore/cudaCheck.h"

namespace cms {
  namespace sycltools {
    class ScopedSetDevice {
    public:
      // Store the original device, without setting a new one
      ScopedSetDevice() {
        // Store the original device
        originalDevice_ = dpct::dev_mgr::instance().current_device_id();
      }

      // Store the original device, and set a new current device
      explicit ScopedSetDevice(int device) : ScopedSetDevice() {
        // Change the current device
        set(device);
      }

      // Restore the original device
      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        dpct::dev_mgr::instance().select_device(originalDevice_);
      }

      // Set a new current device, without changing the original device
      // that will be restored when this object is destroyed
      void set(int device) {
        // Change the current device
        /*
        DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
        */
        cudaCheck((dpct::dev_mgr::instance().select_device(device), 0));
      }

    private:
      int originalDevice_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif
