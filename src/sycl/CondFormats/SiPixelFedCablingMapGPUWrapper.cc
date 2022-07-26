// C++ includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// SYCL includes
#include <CL/sycl.hpp>

// CMSSW includes
//#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  //cudaCheck(cudaMallocHost(&cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));
  std::unique_ptr<SiPixelFedCablingMapGPU> cablingMapHost = std::make_unique<SiPixelFedCablingMapGPU>();
  
  std::memcpy(cablingMapHost.get(), &cablingMap, sizeof(SiPixelFedCablingMapGPU));
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { 
  //cudaCheck(cudaFreeHost(cablingMapHost)); 
  }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(sycl::queue syclStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(syclStream, [this](GPUData& data, sycl::queue stream) {
    // allocate
    //cudaCheck(cudaMalloc(&data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));
    data.cablingMapDevice = (SiPixelFedCablingMapGPU *)sycl::malloc_device(sizeof(SiPixelFedCablingMapGPU), stream);
    // transfer
    stream.memcpy(data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU));
    //cudaCheck(cudaMemcpyAsync(
    //    data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault, stream));
 // });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(sycl::queue syclStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(syclStream, [this](ModulesToUnpack& data, sycl::queue stream) {
         data.modToUnpDefault = (unsigned char *)sycl::malloc_device(pixelgpudetails::MAX_SIZE_BYTE_BOOL, stream);
         stream.memcpy(data.modToUnpDefault, this->modToUnpDefault.data(), this->modToUnpDefault.size() * sizeof(unsigned char)));
        //cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        //cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
        //                          this->modToUnpDefault.data(),
        //                          this->modToUnpDefault.size() * sizeof(unsigned char),
        //                          cudaMemcpyDefault,
        //                          stream));
      });
  return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() { 
  //cudaCheck(cudaFree(cablingMapDevice)); 
  }

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() { 
  //cudaCheck(cudaFree(modToUnpDefault)); 
  }
