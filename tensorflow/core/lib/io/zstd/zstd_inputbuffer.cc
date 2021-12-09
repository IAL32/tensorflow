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

#include "tensorflow/core/lib/io/zstd/zstd_inputbuffer.h"

namespace tensorflow {
namespace io {
ZstdInputBuffer::ZstdInputBuffer(
    RandomAccessFile* file,
    size_t input_buffer_bytes,  // size of input_buffer_
    size_t output_buffer_bytes  // size of output_buffer_
    )
    : file_(file), bytes_read_(0) {}

Status ZstdInputBuffer::ReadNBytes(int64 bytes_to_read, tstring* result) {
  return errors::Unimplemented("Not implemented");
}

int64 ZstdInputBuffer::Tell() const { return bytes_read_; }

Status ZstdInputBuffer::Reset() { return Status::OK(); }

size_t ZstdInputBuffer::ReadBytesFromCache(size_t bytes_to_read,
                                           char* result_ptr) {
  return 0;
}

Status ZstdInputBuffer::Inflate() {
  return errors::Unimplemented("Not implemented");
}

Status ZstdInputBuffer::ReadCompressedBlockLength(uint32* length) {
  return errors::Unimplemented("Not implemented");
}

Status ZstdInputBuffer::ReadFromFile() {
  return errors::Unimplemented("Not implemented");
}

}  // namespace io
}  // namespace tensorflow
