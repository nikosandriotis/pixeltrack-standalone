#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "SYCLCore/ESProduct.h"

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const &gain, std::vector<char> gainData);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU *getGPUProductAsync(sycl::queue syclQueue) const;
  const SiPixelGainForHLTonGPU *getCPUProduct() const { return gainForHLTonHost_; }

private:
  SiPixelGainForHLTonGPU *gainForHLTonHost_ = nullptr;
  std::vector<char> gainData_;
  struct GPUData {
    ~GPUData();
    SiPixelGainForHLTonGPU *gainForHLTonGPU = nullptr;
    SiPixelGainForHLTonGPU_DecodingStructure *gainDataOnGPU = nullptr;
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
