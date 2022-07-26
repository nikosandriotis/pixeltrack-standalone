/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToClusterGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/

// C++ includes
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// SYCL includes

// CMSSW includes
#include "SYCLDataFormats/gpuClusteringConstants.h"
#include "SYCLCore/cudaCheck.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "SYCLCore/copyAsync.h"
#include "Geometry/phase1PixelTopology.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"
#include "gpuCalibPixel.h"
#include "gpuClusterChargeCut.h"
#include "gpuClustering.h"

namespace pixelgpudetails {

  // number of words for all the FEDs
  constexpr uint32_t MAX_FED_WORDS = pixelgpudetails::MAX_FED * pixelgpudetails::MAX_WORD;

  SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender() {
    /*
    DPCT1048:0: The original value cudaHostAllocWriteCombined is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
    */
    word_ = cms::sycltools::make_host_noncached_unique<unsigned int[]>(MAX_FED_WORDS, 0);
    /*
    DPCT1048:1: The original value cudaHostAllocWriteCombined is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
    */
    fedId_ = cms::sycltools::make_host_noncached_unique<unsigned char[]>(MAX_FED_WORDS, 0);
  }

  void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId,
                                                                        unsigned int wordCounterGPU,
                                                                        const uint32_t *src,
                                                                        unsigned int length) {
    std::memcpy(word_.get() + wordCounterGPU, src, sizeof(uint32_t) * length);
    std::memset(fedId_.get() + wordCounterGPU / 2, fedId - 1200, length / 2);
  }

  ////////////////////

  uint32_t getLink(uint32_t ww) {
    return ((ww >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask);
  }

  uint32_t getRoc(uint32_t ww) { return ((ww >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask); }

  uint32_t getADC(uint32_t ww) { return ((ww >> pixelgpudetails::ADC_shift) & pixelgpudetails::ADC_mask); }

  bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

  pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU *cablingMap,
                                                uint8_t fed,
                                                uint32_t link,
                                                uint32_t roc) {
    uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
    pixelgpudetails::DetIdGPU detId = {
        cablingMap->RawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index]};
    return detId;
  }

  //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
  //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
  // Convert local pixel to pixelgpudetails::global pixel
  pixelgpudetails::Pixel frameConversion(
      bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, pixelgpudetails::Pixel local) {
    int slopeRow = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {
      if (side == -1 && layer != 1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }       // if roc
      } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = rocIdInDetUnit * pixelgpudetails::numColsInRoc;
        } else {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (16 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        }
      }

    } else {             // fpix
      if (side == -1) {  // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }
      } else {  // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }

      }  // side
    }

    uint32_t gRow = rowOffset + slopeRow * local.row;
    uint32_t gCol = colOffset + slopeCol * local.col;
    //printf("Inside frameConversion row: %u, column: %u\n", gRow, gCol);
    pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }

  uint8_t conversionError(uint8_t fedId, uint8_t status, const sycl::stream &stream_ct1, bool debug = false) {
    uint8_t errorType = 0;

    // debug = true;

    switch (status) {
      case (1): {
        if (debug)
          /*
          DPCT1015:23: Output needs adjustment.
          */
          stream_ct1 << "Error in Fed: %i, invalid channel Id (errorType = 35\n)";
        errorType = 35;
        break;
      }
      case (2): {
        if (debug)
          /*
          DPCT1015:24: Output needs adjustment.
          */
          stream_ct1 << "Error in Fed: %i, invalid ROC Id (errorType = 36)\n";
        errorType = 36;
        break;
      }
      case (3): {
        if (debug)
          /*
          DPCT1015:25: Output needs adjustment.
          */
          stream_ct1 << "Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n";
        errorType = 37;
        break;
      }
      case (4): {
        if (debug)
          /*
          DPCT1015:26: Output needs adjustment.
          */
          stream_ct1 << "Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n";
        errorType = 38;
        break;
      }
      default:
        if (debug)
          /*
          DPCT1015:27: Output needs adjustment.
          */
          stream_ct1 << "Cabling check returned unexpected result, status = %i\n";
    };

    return errorType;
  }

  bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
    uint32_t numRowsInRoc = 80;
    uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

  bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

  uint8_t checkROC(
      uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *cablingMap,
      const sycl::stream &stream_ct1, bool debug = false) {
    uint8_t errorType = (errorWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ERROR_mask;
    if (errorType < 25)
      return 0;
    bool errorFound = false;

    switch (errorType) {
      case (25): {
        errorFound = true;
        uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + 1;
        if (index > 1 && index <= cablingMap->size) {
          if (!(link == cablingMap->link[index] && 1 == cablingMap->roc[index]))
            errorFound = false;
        }
        if (debug and errorFound)
          stream_ct1 << "Invalid ROC = 25 found (errorType = 25)\n";
        break;
      }
      case (26): {
        if (debug)
          stream_ct1 << "Gap word found (errorType = 26)\n";
        errorFound = true;
        break;
      }
      case (27): {
        if (debug)
          stream_ct1 << "Dummy word found (errorType = 27)\n";
        errorFound = true;
        break;
      }
      case (28): {
        if (debug)
          stream_ct1 << "Error fifo nearly full (errorType = 28)\n";
        errorFound = true;
        break;
      }
      case (29): {
        if (debug)
          stream_ct1 << "Timeout on a channel (errorType = 29)\n";
        if ((errorWord >> pixelgpudetails::OMIT_ERR_shift) & pixelgpudetails::OMIT_ERR_mask) {
          if (debug)
            stream_ct1 << "...first errorType=29 error, this gets masked out\n";
        }
        errorFound = true;
        break;
      }
      case (30): {
        if (debug)
          stream_ct1 << "TBM error trailer (errorType = 30)\n";
        int StateMatch_bits = 4;
        int StateMatch_shift = 8;
        uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
        int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
        if (StateMatch != 1 && StateMatch != 8) {
          if (debug)
            stream_ct1 << "FED error 30 with unexpected State Bits (errorType = 30)\n";
        }
        if (StateMatch == 1)
          errorType = 40;  // 1=Overflow -> 40, 8=number of ROCs -> 30
        errorFound = true;
        break;
      }
      case (31): {
        if (debug)
          stream_ct1 << "Event number error (errorType = 31)\n";
        errorFound = true;
        break;
      }
      default:
        errorFound = false;
    };

    return errorFound ? errorType : 0;
  }

  uint32_t getErrRawID(uint8_t fedId,
                                  uint32_t errWord,
                                  uint32_t errorType,
                                  const SiPixelFedCablingMapGPU *cablingMap,
                                  bool debug = false) {
    uint32_t rID = 0xffffffff;

    switch (errorType) {
      case 25:
      case 30:
      case 31:
      case 36:
      case 40: {
        //set dummy values for cabling just to get detId from link
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 29: {
        int chanNmbr = 0;
        const int DB0_shift = 0;
        const int DB1_shift = DB0_shift + 1;
        const int DB2_shift = DB1_shift + 1;
        const int DB3_shift = DB2_shift + 1;
        const int DB4_shift = DB3_shift + 1;
        const uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

        int CH1 = (errWord >> DB0_shift) & DataBit_mask;
        int CH2 = (errWord >> DB1_shift) & DataBit_mask;
        int CH3 = (errWord >> DB2_shift) & DataBit_mask;
        int CH4 = (errWord >> DB3_shift) & DataBit_mask;
        int CH5 = (errWord >> DB4_shift) & DataBit_mask;
        int BLOCK_bits = 3;
        int BLOCK_shift = 8;
        uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
        int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
        int localCH = 1 * CH1 + 2 * CH2 + 3 * CH3 + 4 * CH4 + 5 * CH5;
        if (BLOCK % 2 == 0)
          chanNmbr = (BLOCK / 2) * 9 + localCH;
        else
          chanNmbr = ((BLOCK - 1) / 2) * 9 + 4 + localCH;
        if ((chanNmbr < 1) || (chanNmbr > 36))
          break;  // signifies unexpected result

        // set dummy values for cabling just to get detId from link if in Barrel
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = chanNmbr;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 37:
      case 38: {
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = (errWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      default:
        break;
    };

    return rID;
  }

  // Kernel to perform Raw to Digi conversion
  void RawToDigi_kernel(const SiPixelFedCablingMapGPU *cablingMap,
                                   const unsigned char *modToUnp,
                                   const uint32_t wordCounter,
                                   const uint32_t *word,
                                   const uint8_t *fedIds,
                                   uint16_t *xx,
                                   uint16_t *yy,
                                   uint16_t *adc,
                                   uint32_t *pdigi,
                                   uint32_t *rawIdArr,
                                   uint16_t *moduleId,
                                   cms::sycltools::SimpleVector<PixelErrorCompact> *err,
                                   bool useQualityInfo,
                                   bool includeErrors,
                                   bool debug,
                                   sycl::nd_item<3> item_ct1) {
    //if (threadIdx.x==0) printf("Event: %u blockIdx.x: %u start: %u end: %u\n", eventno, blockIdx.x, begin, end);

    int32_t first = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    for (int32_t iloop = first, nend = wordCounter; iloop < nend;
         iloop += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
      auto gIndex = iloop;
      xx[gIndex] = 0;
      yy[gIndex] = 0;
      adc[gIndex] = 0;
      bool skipROC = false;

      uint8_t fedId = fedIds[gIndex / 2];  // +1200;

      // initialize (too many coninue below)
      pdigi[gIndex] = 0;
      rawIdArr[gIndex] = 0;
      moduleId[gIndex] = 9999;

      uint32_t ww = word[gIndex];  // Array containing 32 bit raw data
      if (ww == 0) {
        // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
        continue;
      }

      uint32_t link = getLink(ww);  // Extract link
      uint32_t roc = getRoc(ww);    // Extract Roc in link
      pixelgpudetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

      uint8_t errorType = checkROC(ww, fedId, link, cablingMap, stream_ct1, debug);
      skipROC = (roc < pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
      if (includeErrors and skipROC) {
        uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug);
        err->push_back(PixelErrorCompact{rID, ww, errorType, fedId});
        continue;
      }

      uint32_t rawId = detId.RawId;
      uint32_t rocIdInDetUnit = detId.rocInDet;
      bool barrel = isBarrel(rawId);

      uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
      if (useQualityInfo) {
        skipROC = cablingMap->badRocs[index];
        if (skipROC)
          continue;
      }
      skipROC = modToUnp[index];
      if (skipROC)
        continue;

      uint32_t layer = 0;                   //, ladder =0;
      int side = 0, panel = 0, module = 0;  //disk = 0, blade = 0

      if (barrel) {
        layer = (rawId >> pixelgpudetails::layerStartBit) & pixelgpudetails::layerMask;
        module = (rawId >> pixelgpudetails::moduleStartBit) & pixelgpudetails::moduleMask;
        side = (module < 5) ? -1 : 1;
      } else {
        // endcap ids
        layer = 0;
        panel = (rawId >> pixelgpudetails::panelStartBit) & pixelgpudetails::panelMask;
        //disk  = (rawId >> diskStartBit_) & diskMask_;
        side = (panel == 1) ? -1 : 1;
        //blade = (rawId >> bladeStartBit_) & bladeMask_;
      }

      // ***special case of layer to 1 be handled here
      pixelgpudetails::Pixel localPix;
      if (layer == 1) {
        uint32_t col = (ww >> pixelgpudetails::COL_shift) & pixelgpudetails::COL_mask;
        uint32_t row = (ww >> pixelgpudetails::ROW_shift) & pixelgpudetails::ROW_mask;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors) {
          if (not rocRowColIsValid(row, col)) {
            uint8_t error = conversionError(fedId, 3, stream_ct1, debug);  //use the device function and fill the arrays
            err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
            if (debug)
              /*
              DPCT1015:28: Output needs adjustment.
              */
              stream_ct1 << "BPIX1  Error status: %i\n";
            continue;
          }
        }
      } else {
        // ***conversion rules for dcol and pxid
        uint32_t dcol = (ww >> pixelgpudetails::DCOL_shift) & pixelgpudetails::DCOL_mask;
        uint32_t pxid = (ww >> pixelgpudetails::PXID_shift) & pixelgpudetails::PXID_mask;
        uint32_t row = pixelgpudetails::numRowsInRoc - pxid / 2;
        uint32_t col = dcol * 2 + pxid % 2;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors and not dcolIsValid(dcol, pxid)) {
          uint8_t error = conversionError(fedId, 3, stream_ct1, debug);
          err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
          if (debug)
            /*
            DPCT1015:29: Output needs adjustment.
            */
            stream_ct1 << "Error status: %i %d %d %d %d\n";
          continue;
        }
      }

      pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      xx[gIndex] = globalPix.row;  // origin shifting by 1 0-159
      yy[gIndex] = globalPix.col;  // origin shifting by 1 0-415
      adc[gIndex] = getADC(ww);
      pdigi[gIndex] = pixelgpudetails::pack(globalPix.row, globalPix.col, adc[gIndex]);
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
    }  // end of loop (gIndex < end)

  }  // end of Raw to Digi kernel

  void fillHitsModuleStart(uint32_t const *__restrict__ cluStart, uint32_t *__restrict__ moduleStart,
                           sycl::nd_item<3> item_ct1, uint32_t *ws) {
    assert(gpuClustering::MaxNumModules < 2048);  // easy to extend at least till 32*1024
    assert(1 == item_ct1.get_group_range(2));
    assert(0 == item_ct1.get_group(2));

    int first = item_ct1.get_local_id(2);

    // limit to MaxHitsInModule;
    for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += item_ct1.get_local_range().get(2)) {
      moduleStart[i + 1] = std::min(gpuClustering::maxHitsInModule(), cluStart[i]);
    }

    cms::sycltools::blockPrefixScan(moduleStart + 1, moduleStart + 1, 1024, ws);
    cms::sycltools::blockPrefixScan(moduleStart + 1025, moduleStart + 1025, gpuClustering::MaxNumModules - 1024, ws);

    for (int i = first + 1025, iend = gpuClustering::MaxNumModules + 1; i < iend;
         i += item_ct1.get_local_range().get(2)) {
      moduleStart[i] += moduleStart[1024];
    }
    /*
    DPCT1065:30: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
    */
    item_ct1.barrier();

#ifdef GPU_DEBUG
    assert(0 == moduleStart[0]);
    auto c0 = std::min(gpuClustering::maxHitsInModule(), cluStart[0]);
    assert(c0 == moduleStart[1]);
    assert(moduleStart[1024] >= moduleStart[1023]);
    assert(moduleStart[1025] >= moduleStart[1024]);
    assert(moduleStart[gpuClustering::MaxNumModules] >= moduleStart[1025]);

    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
      if (0 != i)
        assert(moduleStart[i] >= moduleStart[i - i]);
      // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
      // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
      if (i == 96 || i == 1184 || i == 1744 || i == gpuClustering::MaxNumModules)
        printf("moduleStart %d %d\n", i, moduleStart[i]);
    }
#endif

    // avoid overflow
    constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += item_ct1.get_local_range().get(2)) {
      if (moduleStart[i] > MAX_HITS)
        moduleStart[i] = MAX_HITS;
    }
  }

  // Interface to outside
  void SiPixelRawToClusterGPUKernel::makeClustersAsync(bool isRun2,
                                                       const SiPixelFedCablingMapGPU *cablingMap,
                                                       const unsigned char *modToUnp,
                                                       const SiPixelGainForHLTonGPU *gains,
                                                       const WordFedAppender &wordFed,
                                                       PixelFormatterErrors &&errors,
                                                       const uint32_t wordCounter,
                                                       const uint32_t fedCounter,
                                                       bool useQualityInfo,
                                                       bool includeErrors,
                                                       bool debug,
                                                       sycl::queue *stream) {
    nDigis = wordCounter;

#ifdef GPU_DEBUG
    std::cout << "decoding " << wordCounter << " digis. Max is " << pixelgpudetails::MAX_FED_WORDS << std::endl;
#endif

    digis_d = SiPixelDigisSYCL(pixelgpudetails::MAX_FED_WORDS, stream);
    if (includeErrors) {
      digiErrors_d = SiPixelDigiErrorsSYCL(pixelgpudetails::MAX_FED_WORDS, std::move(errors), stream);
    }
    clusters_d = SiPixelClustersSYCL(gpuClustering::MaxNumModules, stream);

    nModules_Clusters_h = cms::cuda::make_host_unique<uint32_t[]>(2, stream);

    if (wordCounter)  // protect in case of empty event....
    {
      const int threadsPerBlock = 512;
      const int blocks = (wordCounter + threadsPerBlock - 1) / threadsPerBlock;  // fill it all

      assert(0 == wordCounter % 2);
      // wordCounter is the total no of words in each event to be trasfered on device
      auto word_d = cms::sycltools::make_device_unique<uint32_t[]>(wordCounter, stream);
      auto fedId_d = cms::sycltools::make_device_unique<uint8_t[]>(wordCounter, stream);

      cms::sycltools::copyAsync(word_d, wordFed.word(), wordCounter, stream);
      cms::sycltools::copyAsync(fedId_d, wordFed.fedId(), wordCounter/2, stream);

      stream.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
                          [=](sycl::nd_item<3> item) {
                                RawToDigi_kernel(cablingMap,
                                                 modToUnp,
                                                 wordCounter,
                                                 word_d.get(),
                                                 fedId_d.get(),
                                                 digis_d.xx(),
                                                 digis_d.yy(),
                                                 digis_d.adc(),
                                                 digis_d.pdigi(),
                                                 digis_d.rawIdArr(),
                                                 digis_d.moduleInd(),
                                                 digiErrors_d.error(),  // returns nullptr if default-constructed
                                                 useQualityInfo,
                                                 includeErrors,
                                                 debug,
                                                 item);
                          });
      /*
      DPCT1010:31: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);
#ifdef GPU_DEBUG
      //cudaDeviceSynchronize();
      //cudaCheck(cudaGetLastError());
#endif

      if (includeErrors) {
        digiErrors_d.copyErrorToHostAsync(stream);
      }
    }
    // End of Raw2Digi and passing data for clustering

    {
      // clusterizer ...
      using namespace gpuClustering;
      int threadsPerBlock = 256;
      int blocks =
          (std::max(int(wordCounter), int(gpuClustering::MaxNumModules)) + threadsPerBlock - 1) / threadsPerBlock;
      
      stream.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
                          [=](sycl::nd_item<3> item) {
                                gpuCalibPixel::calibDigis(isRun2,
                                                          digis_d.moduleInd(),
                                                          digis_d.c_xx(),
                                                          digis_d.c_yy(),
                                                          digis_d.adc(),
                                                          gains,
                                                          wordCounter,
                                                          clusters_d.moduleStart(),
                                                          clusters_d.clusInModule(),
                                                          clusters_d.clusModuleStart(),
                                                          item);
                          });
      /*
      DPCT1010:32: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);
#ifdef GPU_DEBUG
      //cudaDeviceSynchronize();
      //cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
      std::cout << "SYCL countModules kernel launch with " << blocks << " blocks of " << threadsPerBlock
                << " threads\n";
#endif
       stream.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
                          [=](sycl::nd_item<3> item) {
                                countModules(digis_d.c_moduleInd(), clusters_d.moduleStart(), digis_d.clus(), wordCounter, item);
                          });
      /*
      DPCT1010:33: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);

      // read the number of modules into a data member, used by getProduct())
      cms::sycltools::copyAsync(&(nModules_Clusters_h[0]), clusters_d.moduleStart(), stream);
     
      threadsPerBlock = 256;
      blocks = MaxNumModules;
#ifdef GPU_DEBUG
      std::cout << "SYCL findClus kernel launch with " << blocks << " blocks of " << threadsPerBlock << " threads\n";
#endif
             
      constexpr uint32_t maxPixInModule = 4000;
      constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;
      using Hist = cms::sycltools::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;
       stream.submit([&](sycl::handler &cgh) {
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_gMaxHit_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_mSize_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<Hist, 1, sycl::access_mode::read_write, sycl::access::target::local>
              hist_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh); //FIXME_ why 32?
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_ws_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_totGood_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_n40_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_n60_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_n0_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              foundClusters_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh); 

          cgh.parallel_for(
              sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
              [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] { // explicitly specify sub-group size (32 is the maximum)
                                findClus(digis_d.c_moduleInd(),
                                         digis_d.c_xx(),
                                         digis_d.c_yy(),
                                         clusters_d.c_moduleStart(),
                                         clusters_d.clusInModule(),
                                         clusters_d.moduleId(),
                                         digis_d.clus(),
                                         wordCounter,
                                         item,
                                         (uint32_t *)local_gMaxHit_acc.get_pointer(),
                                         (uint8_t *)local_mSize_acc.get_pointer(),
                                         (Hist *)hist_acc.get_pointer(),
                                         (uint32_t *)local_ws_acc.get_pointer(),
                                         (uint32_t *)local_totGood_acc.get_pointer(),
                                         (uint32_t *)local_n40_acc.get_pointer(),
                                         (uint32_t *)local_n60_acc.get_pointer(),
                                         (uint32_t *)local_n0_acc.get_pointer(),
                                         (uint8_t *)foundClusters_acc.get_pointer(),
                                         );
              });
                  });

      /*
      DPCT1010:34: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      cudaCheck(0);
#ifdef GPU_DEBUG
      cudaDeviceSynchronize();
      cudaCheck(cudaGetLastError());
#endif
  
      stream.submit([&](sycl::handler &cgh) {
          sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              charge_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              ok_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
          sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              newclusId_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh); //FIXME_ why 32?
          sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_ws_acc(sycl::range<1>(sizeof(int32_t) * blocks), cgh);
      
      // apply charge cut
          cgh.parallel_for(
              sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)),
              [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] { // explicitly specify sub-group size (32 is the maximum)
                                clusterChargeCut(digis_d.moduleInd(),
                                                 digis_d.c_adc(),
                                                 clusters_d.c_moduleStart(),
                                                 clusters_d.clusInModule(),
                                                 clusters_d.c_moduleId(),
                                                 digis_d.clus(),
                                                 wordCounter,
                                                 item,
                                                 (uint32_t *)charge_acc.get_pointer(),
                                                 (uint8_t *)ok_acc.get_pointer(),
                                                 (uint16_t *)newclusId_acc.get_pointer(),
                                                 (uint16_t *)local_ws_acc.get_pointer()
                                                 );
              });
                  });
    
      /*
      DPCT1010:35: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);

      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWork

      stream.submit([&](sycl::handler &cgh) {
          sycl::accessor<int32_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_ws_acc(sycl::range<1>(sizeof(int32_t)), cgh);
      
      // apply charge cut
          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)),
              [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] { // explicitly specify sub-group size (32 is the maximum)
                                clusterChargeCut(clusters_d.c_clusInModule(), 
                                                 clusters_d.clusModuleStart(),
                                                 item,
                                                 (uint16_t *)local_ws_acc.get_pointer()
                                                 );
              });
                  });

      // MUST be ONE block

      // last element holds the number of all clusters
      cms::sycltools::copyAsync(&(nModules_Clusters_h[1]), 
                                  clusters_d.moduleStart() + gpuClustering::MaxNumModules, 
                                  stream);

#ifdef GPU_DEBUG
      cudaDeviceSynchronize();
      cudaCheck(cudaGetLastError());
#endif

    }  // end clusterizer scope
  }
}  // namespace pixelgpudetails