/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_LIB_IO_ZSTD_ZSTD_COMPRESSION_OPTIONS_H_
#define TENSORFLOW_CORE_LIB_IO_ZSTD_ZSTD_COMPRESSION_OPTIONS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

class ZstdCompressionOptions {
 public:
  ZstdCompressionOptions();

  static ZstdCompressionOptions DEFAULT();

  // From the zstd manual: https://facebook.github.io/zstd/zstd_manual.html#Chapter5
  // Corresponds to ZSTD_c_compressionLevel
  // Set compression parameters according to pre-defined cLevel table.
  // Note that exact compression parameters are dynamically determined,
  // depending on both compression level and srcSize (when known).
  // Default level is ZSTD_CLEVEL_DEFAULT==3.
  // Special: value 0 means default, which is controlled by ZSTD_CLEVEL_DEFAULT.
  // Note 1: it's possible to pass a negative compression level.
  // Note 2: setting a level does not automatically set all other compression parameters
  // to default. Setting this will however eventually dynamically impact the compression
  // parameters which have not been manually set. The manually set
  // ones will 'stick'
  int8 compression_level;

  // From the zstd manual: https://facebook.github.io/zstd/zstd_manual.html#Chapter5
  // Corresponds to ZSTD_c_nbWorkers
  // Select how many threads will be spawned to compress in parallel.
  // When nbWorkers >= 1, triggers asynchronous mode when invoking ZSTD_compressStream*() :
  // ZSTD_compressStream*() consumes input and flush output if possible, but immediately gives back control to caller,
  // while compression is performed in parallel, within worker thread(s).
  // (note : a strong exception to this rule is when first invocation of ZSTD_compressStream2() sets ZSTD_e_end :
  //  in which case, ZSTD_compressStream2() delegates to ZSTD_compress2(), which is always a blocking call).
  // More workers improve speed, but also increase memory usage.
  // Default value is `0`, aka "single-threaded mode" : no worker is spawned,
  // compression is performed inside Caller's thread, and all invocations are blocking
  int8 threads;

  // From the zstd manual: https://facebook.github.io/zstd/zstd_manual.html#Chapter5
  // Corresponds to ZSTD_c_strategy
  // ZSTD_fast=1,
  // ZSTD_dfast=2,
  // ZSTD_greedy=3,
  // ZSTD_lazy=4,
  // ZSTD_lazy2=5,
  // ZSTD_btlazy2=6,
  // ZSTD_btopt=7,
  // ZSTD_btultra=8,
  // ZSTD_btultra2=9
  // 
  // note: new strategies _might_ be added in the future.
  // Only the order (from fast to strong) is guaranteed.
  // Special: value 0 means "use default strategy".
  int8 compression_strategy;
};

inline ZstdCompressionOptions ZstdCompressionOptions::DEFAULT() {
  return ZstdCompressionOptions();
}

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_ZSTD_ZSTD_COMPRESSION_OPTIONS_H_
