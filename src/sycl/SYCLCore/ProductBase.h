#ifndef SYCLDataFormats_Common_ProductBase_h
#define SYCLDataFormats_Common_ProductBase_h

#include <atomic>
#include <memory>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    namespace impl {
      class ScopedContextBase;
    }

    /**
     * Base class for all instantiations of CUDA<T> to hold the
     * non-T-dependent members.
     */
    class ProductBase {
    public:
      ProductBase() = default;  // Needed only for ROOT dictionary generation
      ~ProductBase();

      ProductBase(const ProductBase&) = delete;
      ProductBase& operator=(const ProductBase&) = delete;
      ProductBase(ProductBase&& other)
          : stream_{std::move(other.stream_)},
            event_{std::move(other.event_)},
            mayReuseStream_{other.mayReuseStream_.load()} {}
      ProductBase& operator=(ProductBase&& other) {
        stream_ = std::move(other.stream_);
        event_ = std::move(other.event_);
        mayReuseStream_ = other.mayReuseStream_.load();
        return *this;
      }

      bool isValid() const { return stream_.has_value(); }
      bool isAvailable() const;

      sycl::device device() const { return stream_->get_device(); }

      sycl::queue stream() const { return *stream_; }

      sycl::event event() const { return *event_; }

    protected:
      explicit ProductBase(sycl::queue stream, sycl::event event) : stream_{stream}, event_{event} {}

    private:
      friend class impl::ScopedContextBase;
      friend class ScopedContextProduce;

      bool mayReuseStream() const {
        bool expected = true;
        bool changed = mayReuseStream_.compare_exchange_strong(expected, false);
        // If the current thread is the one flipping the flag, it may
        // reuse the stream.
        return changed;
      }

      std::optional<sycl::queue> stream_;  //!
      std::optional<sycl::event> event_;   //!

      // This flag tells whether the CUDA stream may be reused by a
      // consumer or not. The goal is to have a "chain" of modules to
      // queue their work to the same stream.
      mutable std::atomic<bool> mayReuseStream_ = true;  //!
    };
  }  // namespace sycltools
}  // namespace cms

#endif
