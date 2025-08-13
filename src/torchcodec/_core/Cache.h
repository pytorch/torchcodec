// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <memory>
#include <mutex>

namespace facebook::torchcodec {

// This header defines simple cache class primitives to store reusable objects
// across TorchCodec stream instances. Intended usage is to store hardware
// contexts creation of which is expensive. The cache mechanism is as follows:
// 1. 'PerGpuCache' provides a dynamic cache with the specified maximum capacity
//    for the given number of GPUs.
// 2. When stream object (e.g. SingleStreamDecoder) is destoyed cachable object
//    must be released to the cache. Cache will accept the object if it is not
//    full.
// 3. When stream object (e.g. SingleStreamDecoder) is created cachable object
//    must be first queried from the cache. If the cache is empty then new
//    object must be created.

template <typename T>
class Cache {
 public:
  Cache(int capacity) : capacity_(capacity) {}

  // Adds an object to the cache if the cache has capacity. Returns true
  // if object was added and false otherwise.
  bool addIfCacheHasCapacity(T&& obj);

  // Returns an object from the cache. Cache does not hold a reference
  // to the object after this call.
  T get();

 private:
  int capacity_;
  std::mutex mutex_;
  std::vector<T> cache_;
};

template <typename T>
bool Cache<T>::addIfCacheHasCapacity(T&& obj) {
  std::scoped_lock lock(mutex_);
  if (capacity_ >= 0 && cache_.size() >= static_cast<size_t>(capacity_)) {
    return false;
  }
  cache_.push_back(std::move(obj));
  return true;
}

template <typename T>
T Cache<T>::get() {
  std::scoped_lock lock(mutex_);
  if (!cache_.size())
    return {};

  T obj = std::move(cache_.back());
  cache_.pop_back();
  return obj;
}

template <typename T>
class PerGpuCache {
 public:
  // Initializes 'maxGpus' number of caches. Each cache can hold no
  // more than 'capacity' items. If 'capacity' <0 cache size is unlimited.
  PerGpuCache(int maxGpus, int capacity) {
    TORCH_CHECK(maxGpus > 0, "maxGpus for PerGpuCache must be >0");
    for (int i = 0; i < maxGpus; ++i) {
      cache_.emplace_back(std::make_unique<Cache<T>>(capacity));
    }
  }

  // Adds an object to the specified device cache if the cache has
  // capacity. Returns true if object was added and false otherwise.
  bool addIfCacheHasCapacity(const torch::Device& device, T&& obj);

  // Returns an object from the cache of the specified device. Cache
  // does not hold a reference to the object after this call.
  T get(const torch::Device& device);

 private:
  // 'Cache' class implementation contains mutex which makes it non-movable
  // and non-copyable, so we need to wrap it in std::unique_ptr.
  std::vector<std::unique_ptr<Cache<T>>> cache_;
};

torch::DeviceIndex getNonNegativeDeviceIndex(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = device.index();
  // For single GPU machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0. That's used in per-gpu cache
  // implementation and during initialization of CUDA and FFmpeg contexts
  // which require non negative indices.
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  return deviceIndex;
}

template <typename T>
bool PerGpuCache<T>::addIfCacheHasCapacity(
    const torch::Device& device,
    T&& obj) {
  torch::DeviceIndex deviceIndex = getNonNegativeDeviceIndex(device);
  if (static_cast<size_t>(deviceIndex) >= cache_.size()) {
    return false;
  }
  return cache_[deviceIndex]->addIfCacheHasCapacity(std::move(obj));
}

template <typename T>
T PerGpuCache<T>::get(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = getNonNegativeDeviceIndex(device);
  if (static_cast<size_t>(deviceIndex) >= cache_.size()) {
    return {};
  }
  return cache_[deviceIndex]->get();
}

} // namespace facebook::torchcodec
