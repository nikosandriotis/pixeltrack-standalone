#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "SYCLCore/sycl_assert.h"

#include "gpuClusteringConstants.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

  void calibDigis(bool isRun2,
                  uint16_t* id,
                  uint16_t const* __restrict__ x,
                  uint16_t const* __restrict__ y,
                  uint16_t* adc,
                  SiPixelGainForHLTonGPU const* __restrict__ ped,
                  int numElements,
                  uint32_t* __restrict__ moduleStart,        // just to zero first
                  uint32_t* __restrict__ nClustersInModule,  // just to zero them
                  uint32_t* __restrict__ clusModuleStart,    // just to zero first
                  sycl::nd_item<3> item,
                  const sycl::stream out
  ) {
    int first = item.get_local_range().get(2) * item.get_group(2) + item.get_local_id(2);

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < (int)(gpuClustering::MaxNumModules);
         i += item.get_group_range(2) * item.get_local_range().get(2)) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += item.get_group_range(2) * item.get_local_range().get(2)) {
      if (InvId == id[i])
        continue;

      float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
      float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[i] = InvId;
        adc[i] = 0;
        out << "bad pixel at " << i << " in " << id[i] << "\n";
      } else {
        float vcal = adc[i] * gain - pedestal * gain;
        adc[i] = sycl::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h