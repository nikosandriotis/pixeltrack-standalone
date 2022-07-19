#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
//#include "CUDACore/cudaCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  //Here we need the stream....
  /*
  ?????????
  void *mem = sycl::malloc_host(sizeof(T), stream);
  */
  //void *gainForHLTonHost_ = sycl::malloc_host(sizeof(SiPixelGainForHLTonGPU), dpct::get_default_queue());
  //cudaCheck(cudaMallocHost(&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  //*gainForHLTonHost_ = gain;
}

/*
DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
*/
SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  sycl::free(gainForHLTonHost_, dpct::get_default_queue());
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  /*
  DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  sycl::free(gainForHLTonGPU, q_ct1);
  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  sycl::free(gainDataOnGPU, q_ct1);
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue* cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, sycl::queue* stream) {

    void *gainForHLTonHost_ = sycl::malloc_host(sizeof(SiPixelGainForHLTonGPU), stream);
    cudaMalloc((void**)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
    cudaMalloc((void**)&data.gainDataOnGPU, this->gainData_.size()));
    // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
    
        cudaMemcpyAsync(data.gainDataOnGPU, this->gainData_.data(), this->gainData_.size(), cudaMemcpyDefault, stream));

    cudaCheck(cudaMemcpyAsync(
        data.gainForHLTonGPU, this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(&(data.gainForHLTonGPU->v_pedestals),
                              &(data.gainDataOnGPU),
                              sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.gainForHLTonGPU;
}
