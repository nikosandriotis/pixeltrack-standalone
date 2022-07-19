#ifndef HeterogeneousCore_SYCLUtilities_copyAsync_h
#define HeterogeneousCore_SYCLUtilities_copyAsync_h

//#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_noncached_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

#include <type_traits>

namespace cms {
  namespace sycltools {

    // Single element

    template <typename T>
    inline void copyAsync(device::unique_ptr<T>& dst, const host::unique_ptr<T>& src, sycl::queue stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      stream.memcpy(dst.get(), src.get(), sizeof(T));
    }

    template <typename T>
    inline void copyAsync(device::unique_ptr<T>& dst, const host::noncached::unique_ptr<T>& src, sycl::queue stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      stream.memcpy(dst.get(), src.get(), sizeof(T));
    }

    template <typename T>
    inline void copyAsync(host::unique_ptr<T>& dst, const device::unique_ptr<T>& src, sycl::queue stream) {
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      stream.memcpy(dst.get(), src.get(), sizeof(T));
    }

    // Multiple elements

    template <typename T>
    inline void copyAsync(device::unique_ptr<T[]>& dst,
                          const host::unique_ptr<T[]>& src,
                          size_t nelements,
                          sycl::queue stream) {
      stream.memcpy(dst.get(), src.get(), nelements * sizeof(T));
    }

    template <typename T>
    inline void copyAsync(device::unique_ptr<T[]>& dst,
                          const host::noncached::unique_ptr<T[]>& src,
                          size_t nelements,
                          sycl::queue stream) {
      stream.memcpy(dst.get(), src.get(), nelements * sizeof(T));
    }

    template <typename T>
    inline void copyAsync(host::unique_ptr<T[]>& dst,
                          const device::unique_ptr<T[]>& src,
                          size_t nelements,
                          sycl::queue stream) {
      stream.memcpy(dst.get(), src.get(), nelements * sizeof(T));
    }
  }  // namespace sycltools
}  // namespace cms

#endif
