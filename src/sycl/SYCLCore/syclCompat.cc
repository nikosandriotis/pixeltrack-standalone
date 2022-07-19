#include "SYCLCore/syclCompat.h"

namespace cms {
  namespace syclcompat {
    sycl::range<3> blockIdx(1, 1, 1);
    sycl::range<3> gridDim(1, 1, 1);
  }  // namespace syclcompat
}  // namespace cms

namespace {
  struct InitGrid {
    InitGrid() { cms::syclcompat::resetGrid(); }
  };

  const InitGrid initGrid;

}  // namespace
