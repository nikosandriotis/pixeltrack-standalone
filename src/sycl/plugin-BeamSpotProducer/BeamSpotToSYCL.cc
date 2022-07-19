#include <fstream>

#include <CL/sycl.hpp>

#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"
#include "SYCLCore/copyAsync.h"
//#include "SYCLCore/host_noncached_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLDataFormats/BeamSpotSYCL.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"


class BeamSpotToSYCL : public edm::EDProducer {
public:
  explicit BeamSpotToSYCL(edm::ProductRegistry& reg);
  ~BeamSpotToSYCL() override = default;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  const edm::EDPutTokenT<cms::sycltools::Product<BeamSpotSYCL>> bsPutToken_;

  //In CUDA
  //cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> bsHost;
  //In ALPAKA
  //cms::alpakatools::host_buffer<BeamSpotPOD> bsHost_;
  //In SYCL???????????
};

BeamSpotToSYCL::BeamSpotToSYCL(edm::ProductRegistry& reg)
    : bsPutToken_{reg.produces<cms::sycltools::Product<BeamSpotSYCL>>()}{}

   void BeamSpotToSYCL::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    cms::sycltools::ScopedContextProduce ctx{iEvent.streamID()};
    BeamSpotSYCL bsDevice(ctx.stream());
    cms::sycltools::host::unique_ptr<BeamSpotPOD> bsHost;
    bsHost = cms::sycltools::make_host_unique<BeamSpotPOD>(ctx.stream());
    *bsHost = iSetup.get<BeamSpotPOD>();
    cms::sycltools::copyAsync(bsDevice.ptr(), bsHost, ctx.stream());

    ctx.emplace(iEvent, bsPutToken_, std::move(bsDevice));
}

DEFINE_FWK_MODULE(BeamSpotToSYCL);

