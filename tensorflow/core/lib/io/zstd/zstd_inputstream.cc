/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/zstd/zstd_inputstream.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace io {

ZstdInputStream::ZstdInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZstdCompressionOptions& zstd_options,
                                 bool owns_input_stream)
    : owns_input_stream_(owns_input_stream),
      input_stream_(input_stream),
      bytes_read_(0),
      zstd_options_(zstd_options) {}

ZstdInputStream::ZstdInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZstdCompressionOptions& zstd_options)
    : ZstdInputStream(input_stream, input_buffer_bytes, output_buffer_bytes,
                      zstd_options, false) {}

ZstdInputStream::~ZstdInputStream() {}

Status ZstdInputStream::ReadNBytes(int64 bytes_to_read, tstring* result) {
  return errors::Unimplemented("Not implemented");
}

#if defined(TF_CORD_SUPPORT)
Status ZstdInputStream::ReadNBytes(int64 bytes_to_read, absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return Status::OK();
}
#endif

Status ZstdInputStream::Inflate() {
  return errors::Unimplemented("Not implemented");
}

size_t ZstdInputStream::ReadBytesFromCache(size_t bytes_to_read, char* result) {
  return 0;
}

int64 ZstdInputStream::Tell() const { return bytes_read_; }

Status ZstdInputStream::Reset() {
  return errors::Unimplemented("Not implemented");
}

}  // namespace io
}  // namespace tensorflow
