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

#include "tensorflow/core/lib/io/zstd/zstd_outputbuffer.h"

namespace tensorflow {
namespace io {

ZstdOutputBuffer::ZstdOutputBuffer(WritableFile* file, int32 input_buffer_bytes,
                                   int32 output_buffer_bytes,
                                   const ZstdCompressionOptions& zstd_options)
    : file_(file), zstd_options_(zstd_options) {}

ZstdOutputBuffer::~ZstdOutputBuffer() {
  size_t bytes_to_write = 0;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
}

Status ZstdOutputBuffer::Append(StringPiece data) { return Write(data); }

#if defined(TF_CORD_SUPPORT)
Status ZstdOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return Status::OK();
}
#endif

Status ZstdOutputBuffer::Close() {
  // Given that we do not own `file`, we don't close it.
  return Flush();
}

Status ZstdOutputBuffer::Name(StringPiece* result) const {
  return file_->Name(result);
}

Status ZstdOutputBuffer::Sync() {
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status ZstdOutputBuffer::Tell(int64* position) { return file_->Tell(position); }

Status ZstdOutputBuffer::Write(StringPiece data) {
  //
  // The deflated output is accumulated in output_buffer_ and gets written to
  // file as and when needed.

  size_t bytes_to_write = data.size();

  // If there is sufficient free space in input_buffer_ to fit data we
  // add it there and return.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  // If there isn't enough available space in the input_buffer_ we empty it
  // by uncompressing its contents. If data now fits in input_buffer_
  // we add it there else we directly deflate it.
  TF_RETURN_IF_ERROR(DeflateBuffered());

  // input_buffer_ should be empty at this point.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(Deflate());
  return Status::OK();
}

Status ZstdOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(DeflateBuffered());
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return Status::OK();
}

int32 ZstdOutputBuffer::AvailableInputSpace() const {
  return 0;  // TODO
}

void ZstdOutputBuffer::AddToInputBuffer(StringPiece data) {
  size_t bytes_to_write = data.size();
  DCHECK_LE(bytes_to_write, AvailableInputSpace());
}

Status ZstdOutputBuffer::AddToOutputBuffer(const char* data, size_t length) {
  return Status::OK();
}

Status ZstdOutputBuffer::DeflateBuffered() {
  TF_RETURN_IF_ERROR(Deflate());
  return Status::OK();
}

Status ZstdOutputBuffer::FlushOutputBufferToFile() { return Status::OK(); }

Status ZstdOutputBuffer::Deflate() { return Status::OK(); }

}  // namespace io
}  // namespace tensorflow
