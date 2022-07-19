#include "SYCLDataFormats/SiPixelDigiErrorsSYCL.h"

#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/copyAsync.h"
#include "SYCLCore/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsSYCL::SiPixelDigiErrorsSYCL(size_t maxFedWords, PixelFormatterErrors errors, sycl::queue stream)
    : formatterErrors_h(std::move(errors)) {
  error_d = cms::sycltools::make_device_unique<cms::sycltools::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cms::sycltools::make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

  cms::sycltools::memsetAsync(data_d, 0x00, maxFedWords, stream);

  error_h = cms::sycltools::make_host_unique<cms::sycltools::SimpleVector<PixelErrorCompact>>(stream);
  cms::sycltools::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->empty());
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cms::sycltools::copyAsync(error_d, error_h, stream);
}

void SiPixelDigiErrorsSYCL::copyErrorToHostAsync(sycl::queue stream) {
  cms::sycltools::copyAsync(error_h, error_d, stream);
}

SiPixelDigiErrorsSYCL::HostDataError SiPixelDigiErrorsSYCL::dataErrorToHostAsync(sycl::queue stream) const {
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cms::sycltools::make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if (not error_h->empty()) {
    cms::sycltools::copyAsync(data, data_d, error_h->size(), stream);
  }
  auto err = *error_h;
  err.set_data(data.get());
  return HostDataError(err, std::move(data));
}
