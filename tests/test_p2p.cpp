// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <numeric>
#include <optional>
#include <vector>

#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/p2p.h"

using namespace nvMolKit;

namespace {

template <typename T> AsyncDeviceVector<T> deviceVectorFromHost(const std::vector<T>& host, cudaStream_t stream) {
  AsyncDeviceVector<T> dev(host.size(), stream);
  if (!host.empty()) {
    dev.copyFromHost(host);
  }
  return dev;
}

template <typename T> std::vector<T> hostVectorFromDevice(const AsyncDeviceVector<T>& dev, cudaStream_t stream) {
  std::vector<T> host(dev.size());
  if (!host.empty()) {
    dev.copyToHost(host);
    cudaCheckError(cudaStreamSynchronize(stream));
  }
  return host;
}

}  // namespace

TEST(P2P, CopyDeviceToDeviceSameGpu) {
  ScopedStream              stream;
  const std::vector<double> src    = {1.0, 2.0, 3.0, 4.0, 5.0};
  AsyncDeviceVector<double> srcDev = deviceVectorFromHost(src, stream.stream());
  AsyncDeviceVector<double> dstDev(src.size(), stream.stream());

  copyDeviceToDeviceAsync(dstDev.data(),
                          srcDev.data(),
                          src.size() * sizeof(double),
                          /*srcGpu=*/0,
                          stream.stream(),
                          /*dstGpu=*/0,
                          stream.stream());

  const std::vector<double> result = hostVectorFromDevice(dstDev, stream.stream());
  EXPECT_EQ(result, src);
}

TEST(P2P, CopyDeviceToDeviceCrossGpu) {
  int nDevices = 0;
  cudaCheckError(cudaGetDeviceCount(&nDevices));
  if (nDevices < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  int canAccess = 0;
  cudaCheckError(cudaDeviceCanAccessPeer(&canAccess, 0, 1));
  if (canAccess == 0) {
    GTEST_SKIP() << "GPUs 0 and 1 cannot peer-access each other";
  }

  enablePeerAccess(0, 1);

  ScopedStream              srcStream;
  AsyncDeviceVector<double> srcDev;
  std::vector<double>       src(128);
  std::iota(src.begin(), src.end(), 0.0);
  {
    const WithDevice withSrc(0);
    srcDev = deviceVectorFromHost(src, srcStream.stream());
  }

  std::optional<ScopedStream> dstStream;
  AsyncDeviceVector<double>   dstDev;
  {
    const WithDevice withDst(1);
    dstStream.emplace();
    dstDev = AsyncDeviceVector<double>(src.size(), dstStream->stream());
  }

  copyDeviceToDeviceAsync(dstDev.data(),
                          srcDev.data(),
                          src.size() * sizeof(double),
                          /*srcGpu=*/0,
                          srcStream.stream(),
                          /*dstGpu=*/1,
                          dstStream->stream());

  std::vector<double> result(src.size());
  {
    const WithDevice withDst(1);
    cudaCheckError(cudaMemcpyAsync(result.data(),
                                   dstDev.data(),
                                   src.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   dstStream->stream()));
    cudaCheckError(cudaStreamSynchronize(dstStream->stream()));
  }
  EXPECT_EQ(result, src);
}

TEST(P2P, CopyZeroBytesIsNoop) {
  ScopedStream              stream;
  AsyncDeviceVector<double> srcDev(4, stream.stream());
  AsyncDeviceVector<double> dstDev(4, stream.stream());

  copyDeviceToDeviceAsync(dstDev.data(),
                          srcDev.data(),
                          /*byteCount=*/0,
                          /*srcGpu=*/0,
                          stream.stream(),
                          /*dstGpu=*/0,
                          stream.stream());
  cudaCheckError(cudaStreamSynchronize(stream.stream()));
}

TEST(P2P, EnablePeerSelfIsNoop) {
  EXPECT_NO_THROW(enablePeerAccess(0, 0));
}

TEST(P2P, EnablePeerIdempotent) {
  int nDevices = 0;
  cudaCheckError(cudaGetDeviceCount(&nDevices));
  if (nDevices < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }
  int canAccess = 0;
  cudaCheckError(cudaDeviceCanAccessPeer(&canAccess, 0, 1));
  if (canAccess == 0) {
    GTEST_SKIP() << "GPUs 0 and 1 cannot peer-access each other";
  }
  EXPECT_NO_THROW(enablePeerAccess(0, 1));
  EXPECT_NO_THROW(enablePeerAccess(0, 1));
}
