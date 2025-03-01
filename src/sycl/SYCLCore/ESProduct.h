#ifndef HeterogeneousCore_SYCLCore_ESProduct_h
#define HeterogeneousCore_SYCLCore_ESProduct_h

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>

#include "SYCLCore/EventCache.h"
#include "SYCLCore/ScopedSetDevice.h"
//#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/deviceCount.h"
#include "SYCLCore/currentDevice.h"
#include "SYCLCore/eventWorkHasCompleted.h"

namespace cms {
  namespace sycltools {
    template <typename T>
    class ESProduct {
    public:
      ESProduct() : gpuDataPerDevice_(deviceCount()) {
        if (not gpuDataPerDevice_.empty()) {
                    for (size_t i = 0; i < gpuDataPerDevice_.size(); ++i) {
            gpuDataPerDevice_[i].m_event = getEventCache().get(sycl::device::get_devices(sycl::info::device_type::all)[i]);
          }
        }
      }
//sycl::device::get_devices(sycl::info::device_type::all)[i]
      ~ESProduct() = default;

      // transferAsync should be a function of (T&, cudaStream_t)
      // which enqueues asynchronous transfers (possibly kernels as well)
      // to the SYCL stream
      template <typename F>
      const T& dataForCurrentDeviceAsync(sycl::queue stream, F transferAsync) const {
        //int device = currentDevice();
        //auto& data = gpuDataPerDevice_[device];

        //taken from AUR
        sycl::device device = stream.get_device();
        std::vector<sycl::device> device_list = sycl::device::get_devices(sycl::info::device_type::all);
	      int dev_idx = distance(device_list.begin(), find(device_list.begin(), device_list.end(), device));
        auto& data = gpuDataPerDevice_[dev_idx];

        // If the GPU data has already been filled, we can return it immediately
        if (not data.m_filled.load()) {
          // It wasn't, so need to fill it
          std::scoped_lock<std::mutex> lk{data.m_mutex};

          if (data.m_filled.load()) {
            // Other thread marked it filled while we were locking the mutex, so we're free to return it
            return data.m_data;
          }

          if ((data.m_fillingStream) != nullptr) {
            // Someone else is filling

            // Check first if the recorded event has occurred
            if (eventWorkHasCompleted(data.m_event.get())) {
              // It was, so data is accessible from all SYCL streams on
              // the device. Set the 'filled' for all subsequent calls and
              // return the value
              auto should_be_false = data.m_filled.exchange(true);
              assert(not should_be_false);
              (data.m_fillingStream) = nullptr;
            } else if (*(data.m_fillingStream) != stream) {
              // Filling is still going on. For other SYCL stream, add
              // wait on the SYCL stream and return the value. Subsequent
              // work queued on the stream will wait for the event to
              // occur (i.e. transfer to finish).
              //cudaStreamWaitEvent(cudaStream, data.m_event.get(), 0);
              data.m_event.get()->wait();


            }
            // else: filling is still going on. But for the same SYCL
            // stream (which would be a bit strange but fine), we can just
            // return as all subsequent work should be enqueued to the
            // same SYCL stream (or stream to be explicitly synchronized
            // by the caller)
          } else {
            // Now we can be sure that the data is not yet on the GPU, and
            // this thread is the first to try that.
            transferAsync(data.m_data, stream);
            assert(data.m_fillingStream == nullptr);
            data.m_fillingStream = &stream;
            // Record in the cudaStream an event to mark the readiness of the
            // EventSetup data on the GPU, so other streams can check for it
            //cudaEventRecord(data.m_event.get()->record;
            // Now the filling has been enqueued to the cudaStream, so we
            // can return the GPU data immediately, since all subsequent
            // work must be either enqueued to the cudaStream, or the cudaStream
            // must be synchronized by the caller
          }
        }

        return data.m_data;
      }

    private:
      struct Item {
        mutable std::mutex m_mutex;
        mutable SharedEventPtr m_event;  // guarded by m_mutex
        // non-null if some thread is already filling (cudaStream_t is just a pointer)
        mutable sycl::queue* m_fillingStream = nullptr;  // guarded by m_mutex
        mutable std::atomic<bool> m_filled = false;      // easy check if data has been filled already or not
        mutable T m_data;                                // guarded by m_mutex
      };

      std::vector<Item> gpuDataPerDevice_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_SYCLCore_ESProduct_h
