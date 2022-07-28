#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
//#include "CUDACore/cudaCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  //cudaCheck(cudaMallocHost(&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
   gainForHLTonHost_ = new SiPixelGainForHLTonGPU();
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {  
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  //cudaCheck(cudaFree(gainForHLTonGPU));
  //cudaCheck(cudaFree(gainDataOnGPU));
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue syclStream) const {
    const auto& data = gpuData_.dataForCurrentDeviceAsync(syclStream, [this](GPUData& data, sycl::queue stream) {


      data.gainForHLTonGPU = (SiPixelGainForHLTonGPU *)sycl::malloc_device(sizeof(SiPixelGainForHLTonGPU), stream);
      data.gainDataOnGPU = (SiPixelGainForHLTonGPU_DecodingStructure *)sycl::malloc_device(this->gainData_.size(), stream);

      //cudaMalloc((void**)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU));
      //cudaMalloc((void**)&data.gainDataOnGPU, this->gainData_.size());
      // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
      stream.memcpy(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size());
      //cudaMemcpyAsync(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size(), cudaMemcpyDefault, stream);
      
      stream.memcpy(data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU));
      //cudaMemcpyAsync(
      //    data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream);
      
      stream.memcpy(&(data.gainForHLTonGPU->v_pedestals), &(data.gainDataOnGPU), sizeof(SiPixelGainForHLTonGPU_DecodingStructure*));
      //cudaMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals),
      //                          &(data.gainDataOnGPU),
      //                          sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
      //                          cudaMemcpyDefault,
      //                          stream);
  });
  return data.gainForHLTonGPU;
}
