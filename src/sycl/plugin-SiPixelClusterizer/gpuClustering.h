#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>

#include "Geometry/phase1PixelTopology.h"
#include "SYCLCore/HistoContainer.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/AtomicPairCounter.h"

#include "gpuClusteringConstants.h"

namespace gpuClustering {

#ifdef GPU_DEBUG
   uint32_t gMaxHit = 0;
#endif

  void countModules(uint16_t const* __restrict__ id,
                               uint32_t* __restrict__ moduleStart,
                               int32_t* __restrict__ clusterId,
                               int numElements,
                               sycl::nd_item<3> item_ct1) {
    int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    for (int i = first; i < numElements; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
      clusterId[i] = i;
      if (InvId == id[i])
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == InvId)
        --j;
      if (j < 0 or id[j] != id[i]) {
        // boundary...
        /*
        DPCT1039:0: The generated code assumes that "moduleStart" points to the global memory address space. If it points to a local memory address space, replace "dpct::atomic_fetch_compare_inc" with "dpct::atomic_fetch_compare_inc<uint32_t, sycl::access::address_space::local_space>".
        */
        auto loc = cms::sycltools::AtomicInc(moduleStart, MaxNumModules); //FIXME_ write real atomicInc
        moduleStart[loc + 1] = i;
      }
    }
  }

      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
      using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
  
      //  __launch_bounds__(256,4)
      void
      findClus(uint16_t const* __restrict__ id,           // module id of each pixel
               uint16_t const* __restrict__ x,            // local coordinates of each pixel
               uint16_t const* __restrict__ y,            //
               uint32_t const* __restrict__ moduleStart,  // index of the first pixel of each module
               uint32_t* __restrict__ nClustersInModule,  // output: number of clusters found in each module
               uint32_t* __restrict__ moduleId,           // output: module id of each module
               int32_t* __restrict__ clusterId,           // output: cluster id of each pixel
               int numElements,
               sycl::nd_item<3> item_ct1,
               uint32_t *gMaxHit,
               int *msize,
               Hist  *hist,
               Hist::Counter* ws,
               uint32_t *totGood,
               uint32_t *n40,
               uint32_t *n60,
               int *n0,
               unsigned int *foundClusters) {
    if (item_ct1.get_group(2) >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + item_ct1.get_group(2)];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (item_ct1.get_local_id(2) == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx.x);
#endif

    auto first = firstPixel + item_ct1.get_local_id(2);

    // find the index of the first pixel not belonging to this module (or invalid)
       *msize = numElements;
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // skip threads not associated to an existing pixel
    for (int i = first; i < numElements; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (id[i] != thisModuleId) {  // find the first pixel in a different module
        sycl::atomic<int, sycl::access::address_space::local_space>(sycl::local_ptr<int>(msize)).fetch_min(i);
        break;
      }
    }

    //init hist  (ymax=416 < 512 : 9bits)
    constexpr uint32_t maxPixInModule = 4000;
    for (auto j = item_ct1.get_local_id(2); j < Hist::totbins(); j += item_ct1.get_local_range().get(2)) {
      hist->off[j] = 0;
    }
    /*
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    assert((*msize == numElements) or ((*msize < numElements) and (id[(*msize)] != thisModuleId)));

    // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
    if (0 == item_ct1.get_local_id(2)) {
      if (*msize - firstPixel > maxPixInModule) {
        /*
        DPCT1015:12: Output needs adjustment.
        */
        *msize = maxPixInModule + firstPixel;
      }
    }

    /*
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();
    assert(*msize - firstPixel <= maxPixInModule);

#ifdef GPU_DEBUG
    __shared__ uint32_t totGood;
    *totGood = 0;
     item_ct1.barrier();
#endif

    // fill histo
    for (int i = first; i < *msize; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->count(y[i]);
#ifdef GPU_DEBUG
      atomicAdd(&totGood, 1);
#endif
    }
    /*
    DPCT1065:4: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < 32)
      ws[item_ct1.get_local_id(2)] = 0;  // used by prefix scan...
    /*
    DPCT1065:5: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();
    hist->finalize(item_ct1, ws);
    /*
    DPCT1065:6: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();
#ifdef GPU_DEBUG
    assert(hist->size() == totGood);
    if (thisModuleId % 100 == 1)
      if (item_ct1.get_local_id(2) == 0)
        printf("histo size %d\n", hist->size());
#endif
    for (int i = first; i < *msize; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      hist->fill(y[i], i - firstPixel);
    }

#ifdef DPCT_COMPATIBILITY_TEMP
    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations
    constexpr int maxiter = 16;
#else
    auto maxiter = hist->size();
#endif
    // allocate space for duplicate pixels: a pixel can appear more than once with different charge in the same event
    constexpr int maxNeighbours = 10;
    assert((hist->size() / item_ct1.get_local_range().get(2)) <= maxiter);
    // nearest neighbour
    int16_t nn[16][10]; //FIXME_ variable length arrays are not supported for the current target
    		 	 // uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[10];  // number of nn
    for (uint32_t k = 0; k < maxiter; ++k)
      nnn[k] = 0;

    /*
    DPCT1065:7: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();  // for hit filling!

#ifdef GPU_DEBUG
    // look for anomalous high occupancy
    *n40 = *n60 = 0;
     item_ct1.barrier();
    for (auto j = item_ct1.get_local_id(2); j < Hist::nbins(); j += item_ct1.get_local_range().get(2)) {
      if (hist->size(j) > 60)
        atomicAdd(&n60, 1);
      if (hist->size(j) > 40)
        atomicAdd(&n40, 1);
    }
     item_ct1.barrier();
    if (0 == item_ct1.get_local_id(2)) {
      if (n60 > 0)
        printf("columns with more than 60 px %d in %d\n", n60, thisModuleId);
      else if (n40 > 0)
        printf("columns with more than 40 px %d in %d\n", n40, thisModuleId);
    }
     item_ct1.barrier();
#endif

    // fill NN
    for (auto j = item_ct1.get_local_id(2), k = (unsigned long)0U; j < hist->size(); j += item_ct1.get_local_range().get(2), ++k) {
      assert(k < maxiter);
      auto p = hist->begin() + j;
      auto i = *p + firstPixel;
      assert(id[i] != InvId);
      assert(id[i] == thisModuleId);  // same module
      int be = Hist::bin(y[i] + 1);
      auto e = hist->end(be);
      ++p;
      assert(0 == nnn[k]);
      for (; p < e; ++p) {
        auto m = (*p) + firstPixel;
        assert(m != i);
        assert(int(y[m]) - int(y[i]) >= 0);
        assert(int(y[m]) - int(y[i]) <= 1);
        if (std::abs(int(x[m]) - int(x[i])) > 1)
          continue;
        auto l = nnn[k]++;
        assert(l < maxNeighbours);
        nn[k][l] = *p;
      }
    }

    // for each pixel, look at all the pixels until the end of the module;
    // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
    // after the loop, all the pixel in each cluster should have the id equeal to the lowest
    // pixel in the cluster ( clus[i] == i ).
    bool more = true;
    int nloops = 0;
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    while ((item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), more))) {
      if (1 == nloops % 2) {
        for (auto j = item_ct1.get_local_id(2), k = (unsigned long)0U; j < hist->size(); j += item_ct1.get_local_range().get(2), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          auto m = clusterId[i];
          while (m != clusterId[m])
            m = clusterId[m];
          clusterId[i] = m;
        }
      } else {
        more = false;
        for (auto j = item_ct1.get_local_id(2), k = (unsigned long)0U; j < hist->size(); j += item_ct1.get_local_range().get(2), ++k) {
          auto p = hist->begin() + j;
          auto i = *p + firstPixel;
          for (int kk = 0; kk < nnn[k]; ++kk) {
            auto l = nn[k][kk];
            auto m = l + firstPixel;
            assert(m != i);
            auto old = atomicMin(&clusterId[m], clusterId[i]);
            if (old != clusterId[i]) {
              // end the loop only if no changes were applied
              more = true;
            }
            atomicMin(&clusterId[i], old);
          }  // nnloop
        }    // pixel loop
      }
      ++nloops;
    }  // end while

#ifdef GPU_DEBUG
    {
      if (item_ct1.get_local_id(2) == 0)
        *n0 = nloops;
       item_ct1.barrier();
      auto ok = n0 == nloops;
      assert((item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), ok)));
      if (thisModuleId % 100 == 1)
        if (item_ct1.get_local_id(2) == 0)
          printf("# loops %d\n", nloops);
    }
#endif

    *foundClusters = 0;
    /*
    DPCT1065:8: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // find the number of different clusters, identified by a pixels with clus[i] == i;
    // mark these pixels with a negative id.
    for (int i = first; i < *msize; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] == i) {
        //auto old = dpct::atomic_fetch_compare_inc<unsigned int, sycl::access::address_space::local_space>(foundClusters,0xffffffff);
        auto old = cms::sycltools::AtomicInc(foundClusters, 0xffffffff); //FIXME_
        clusterId[i] = -(old + 1);
      }
    }
    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // propagate the negative id to all the pixels in the cluster.
    for (int i = first; i < *msize; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId)  // skip invalid pixels
        continue;
      if (clusterId[i] >= 0) {
        // mark each pixel in a cluster with the same id as the first one
        clusterId[i] = clusterId[clusterId[i]];
      }
    }
    /*
    DPCT1065:10: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // adjust the cluster id to be a positive value starting from 0
    for (int i = first; i < *msize; i += item_ct1.get_local_range().get(2)) {
      if (id[i] == InvId) {  // skip invalid pixels
        clusterId[i] = -9999;
        continue;
      }
      clusterId[i] = -clusterId[i] - 1;
    }
    /*
    DPCT1065:11: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_local_id(2) == 0) {
      nClustersInModule[thisModuleId] = *foundClusters;
      moduleId[item_ct1.get_group(2)] = thisModuleId;
#ifdef GPU_DEBUG
      if (foundClusters > gMaxHit) {
        gMaxHit = foundClusters;
        if (foundClusters > 8)
          printf("max hit %d in %d\n", foundClusters, thisModuleId);
      }
#endif
#ifdef GPU_DEBUG
      if (thisModuleId % 100 == 1)
        printf("%d clusters in module %d\n", foundClusters, thisModuleId);
#endif
    }
  }

}  // namespace gpuClustering

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
