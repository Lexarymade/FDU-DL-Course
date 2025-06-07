# Install script for directory: /root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/faiss/libfaiss.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/AutoTune.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/Clustering.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IVFlib.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/Index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/Index2Layer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexAdditiveQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinary.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinaryFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinaryFromFloat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinaryHNSW.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinaryHash.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexBinaryIVF.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexFlatCodes.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexHNSW.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIDMap.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVF.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFAdditiveQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFIndependentQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFFlat.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFPQ.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFAdditiveQuantizerFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFPQFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFPQR.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFRaBitQ.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexIVFSpectralHash.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexLSH.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexNeuralNetCodec.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexLattice.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexNNDescent.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexNSG.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexPQ.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexAdditiveQuantizerFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexPQFastScan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexPreTransform.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexRefine.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexReplicas.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexRaBitQ.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexRowwiseMinMax.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexScalarQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexShards.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/IndexShardsIVF.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/MatrixStats.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/MetaIndexes.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/MetricType.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/VectorTransform.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/clone_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/index_factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/index_io.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/AdditiveQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/AuxIndexStructures.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/CodePacker.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/IDSelector.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/DistanceComputer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/FaissAssert.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/FaissException.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/HNSW.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/LocalSearchQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ProductAdditiveQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/LookupTableScaler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/maybe_owned_vector.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/NNDescent.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/NSG.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/PolysemousTraining.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ProductQuantizer-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ProductQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/Quantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/RaBitQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ResidualQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ResultHandler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ScalarQuantizer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ThreadedIndex-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/ThreadedIndex.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/index_read_utils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/io.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/io_macros.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/kmeans1d.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/lattice_Zn.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/platform_macros.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/pq4_fast_scan.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/residual_quantizer_encode_steps.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/simd_result_handlers.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/zerocopy_io.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/code_distance/code_distance.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/code_distance/code_distance-generic.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/code_distance/code_distance-avx2.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/code_distance/code_distance-avx512.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl/code_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/impl/code_distance/code_distance-sve.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/BlockInvertedLists.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/DirectMap.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/InvertedLists.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/InvertedListsIOHook.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/OnDiskInvertedLists.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/AlignedTable.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/bf16.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/Heap.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/NeuralNet.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/WorkerThread.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/distances.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/extra_distances-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/extra_distances.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/fp16-fp16c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/fp16-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/fp16-arm.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/fp16.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/ordered_key_value.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/partitioning.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/prefetch.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/quantize_lut.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/random.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/sorting.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib_avx2.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib_avx512.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib_emulated.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib_neon.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/simdlib_ppc64.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/utils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/distances_fused/avx512.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/distances_fused/distances_fused.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/distances_fused/simdlib_based.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/approx_topk/approx_topk.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/approx_topk/avx2-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/approx_topk/generic.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/approx_topk/mode.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/approx_topk_hamming" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/approx_topk_hamming/approx_topk_hamming.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/transpose" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/transpose/transpose-avx2-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/transpose" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/transpose/transpose-avx512-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/common.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/generic-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/hamdis-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/neon-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/avx2-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/hamming_distance" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/utils/hamming_distance/avx512-inl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-src/faiss/invlists/OnDiskInvertedLists.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES
    "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/cmake/faiss-config.cmake"
    "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/cmake/faiss-config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake"
         "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/faiss/CMakeFiles/Export/share/faiss/faiss-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/faiss/CMakeFiles/Export/share/faiss/faiss-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "/root/autodl-tmp/gaussian-splatting/colmap/build/_deps/faiss-build/faiss/CMakeFiles/Export/share/faiss/faiss-targets-release.cmake")
  endif()
endif()

