//#ifndef HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h
//#define HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h
//
//
//#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
////#include "SYCLCore/cudaCheck.h"
//
//namespace cms {
//  namespace sycltools {
//    class ScopedSetDevice {
//    public:
//      // Store the original device, without setting a new one
//      ScopedSetDevice() {
//        // Store the original device
//        stream.get_device();
//      }
//
//      // Store the original device, and set a new current device
//      explicit ScopedSetDevice(sycl::device device) : ScopedSetDevice() {
//        // Change the current device
//        set(device);
//      }
//
//      // Restore the original device
//      ~ScopedSetDevice() {
//        // Intentionally don't check the return value to avoid
//        // exceptions to be thrown. If this call fails, the process is
//        // doomed anyway.
//         cudaSetDevice(originalDevice_);
//      }
//
//      // Set a new current device, without changing the original device
//      // that will be restored when this object is destroyed
//      void set(sycl::device device) {
//        // Change the current device
//        /*
//        DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
//        */
//        vector<sycl::device> device_list = sycl::device::get_devices(sycl::info::device_type::all);
//	      int dev_idx = distance(device_list.begin(), find(device_list.begin(), device_list.end(), dev));
//
//        cudaCheck(cudaSetDevice(device));
//      }
//
//    private:
//      sycl::device originalDevice_;
//    };
//  }  // namespace sycltools
//}  // namespace cms
//
//#endif
//