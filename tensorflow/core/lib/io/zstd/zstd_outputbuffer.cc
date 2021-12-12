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
    : file_(file),
      input_buffer_(new char[input_buffer_bytes]),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_(new char[output_buffer_bytes]),
      output_buffer_capacity_(output_buffer_bytes),
      zstd_options_(zstd_options) {
  InitZstdBuffer();
}

ZstdOutputBuffer::~ZstdOutputBuffer() {
  size_t bytes_to_write = 0;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
}

void ZstdOutputBuffer::InitZstdBuffer() {
  context_ = ZSTD_createCCtx();
  if (context_ == nullptr) {
    LOG(FATAL) << "Creation of context failed.";
  }
  ZSTD_CCtx_setParameter(context_, ZSTD_c_compressionLevel,
                         zstd_options_.compression_level);
  // ZSTD_CCtx_setParameter(context_, ZSTD_c_checksumFlag, 1);
  ZSTD_CCtx_setParameter(context_, ZSTD_c_nbWorkers, zstd_options_.nb_workers);

  next_in_ = input_buffer_.get();
  next_out_ = output_buffer_.get();
  avail_in_ = 0;
  avail_out_ = output_buffer_capacity_;
}

Status ZstdOutputBuffer::Append(StringPiece data) {
  // The deflated output is accumulated in output_buffer_ and gets written to
  // file as and when needed.
  size_t bytes_to_write = data.size();

  // If there is sufficient free space in input_buffer_ to fit data we
  // add it there and return.

  std::cout << "Append(): bytes_to_write: " << bytes_to_write << std::endl;
  std::cout << "Append(): AvailableInputSpace: " << AvailableInputSpace()
            << std::endl;
  if (bytes_to_write <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  // If there isn't enough available space in the input_buffer_ we empty it
  // by uncompressing its contents. If data now fits in input_buffer_
  // we add it there else we directly deflate it.
  TF_RETURN_IF_ERROR(DeflateBuffered(false));

  // At this point input stream should be empty.
  if (bytes_to_write <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  std::cout << "Append(): Data is too large to fit input buffer" << std::endl;
  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  next_in_ = const_cast<char*>(data.data());
  avail_in_ = bytes_to_write;

  TF_RETURN_IF_ERROR(Deflate(true));

  DCHECK_EQ(avail_in_, 0);  // All input used up.

  next_in_ = input_buffer_.get();

  return Status::OK();
}

void ZstdOutputBuffer::AddToInputBuffer(StringPiece data) {
  size_t bytes_to_write = data.size();
  DCHECK_LE(bytes_to_write, AvailableInputSpace());

  // Input stream ->
  // [....................input_buffer_capacity_...............]
  // [<...read_bytes...><...avail_in...>......empty space......]
  //  ^                 ^
  //  |                 |
  //  input_buffer_   next_in
  //
  // Data in the input stream is sharded as shown above. next_in_ could
  // be pointing to some byte in the buffer with avail_in number of bytes
  // available to be read.
  //
  // In order to avoid shifting the avail_in bytes at next_in to the head of
  // the buffer we try to fit `data` in the empty space at the tail of the
  // input stream.
  // TODO(srbs): This could be avoided if we had a circular buffer.
  // If it doesn't fit we free the space at the head of the stream and then
  // append `data` at the end of existing data.

  const size_t read_bytes = next_in_ - input_buffer_.get();
  const size_t unread_bytes = avail_in_;
  const size_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  std::cout << "AddToInputBuffer(): free_tail_bytes: " << free_tail_bytes
            << std::endl;
  if (bytes_to_write > free_tail_bytes) {
    memmove(input_buffer_.get(), next_in_, avail_in_);
    next_in_ = input_buffer_.get();
  }
  memcpy(next_in_ + avail_in_, data.data(), bytes_to_write);
  avail_in_ += bytes_to_write;
}

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
  TF_RETURN_IF_ERROR(DeflateBuffered(true));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  ZSTD_freeCCtx(context_);
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

Status ZstdOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(DeflateBuffered(false));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return file_->Flush();
}

int32 ZstdOutputBuffer::AvailableInputSpace() const {
  return input_buffer_capacity_ - avail_in_;
}

Status ZstdOutputBuffer::DeflateBuffered(bool last_chunk) {
  std::cout << "DeflateBuffered(): avail_out_: " << avail_out_
            << ", avail_in_: " << avail_in_ << std::endl;
  do {
    if (avail_out_ == 0) {
      // No available output space.
      // Write output buffer to file.
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
    TF_RETURN_IF_ERROR(Deflate(last_chunk));
  } while (avail_out_ == 0);

  DCHECK_EQ(avail_in_, 0);  // We have used all the input

  // Restore z_stream input pointers.
  next_in_ = input_buffer_.get();
  return Status::OK();
}

Status ZstdOutputBuffer::FlushOutputBufferToFile() {
  size_t bytes_to_write = output_buffer_capacity_ - avail_out_;
  if (bytes_to_write > 0) {
    Status s = file_->Append(StringPiece(
        reinterpret_cast<char*>(output_buffer_.get()), bytes_to_write));
    if (s.ok()) {
      next_out_ = output_buffer_.get();
      avail_out_ = output_buffer_capacity_;
      std::cout << "FlushOutputBufferToFile(): wrote " << bytes_to_write
                << std::endl;
    }
    return s;
  }
  return Status::OK();
}

Status ZstdOutputBuffer::Deflate(bool last_chunk) {
  std::cout << "Deflate(): avail_in_: " << avail_in_ << std::endl;
  if (avail_in_ == 0) {
    return Status::OK();
  }

  const ZSTD_EndDirective mode = last_chunk ? ZSTD_e_end : ZSTD_e_continue;

  tstring data;
  data.append(next_in_, avail_in_);
  std::cout << "Deflate(): last_chunk: " << last_chunk
            << ", avail_out_: " << avail_out_ << std::endl
            << "\tdata: '" << data << "'" << std::endl;

  ZSTD_inBuffer input = {next_in_, avail_in_, 0};
  bool finished;
  do {
    ZSTD_outBuffer output = {next_out_, avail_out_, 0};

    remaining_ = ZSTD_compressStream2(context_, &output, &input, mode);
    // TODO: remaining should be == 0 every time, as we check the size
    // beforehand
    if (ZSTD_isError(remaining_)) {
      return errors::Internal(ZSTD_getErrorName(remaining_));
    }

    avail_out_ = output_buffer_capacity_ - output.pos;

    std::cout << "Deflate(): input.pos: " << input.pos
              << ", output.pos: " << output.pos << " remaining_: " << remaining_
              << ", avail_out_: " << avail_out_ << std::endl;

    tstring data_out;
    data_out.append(next_out_, output.pos);
    std::cout << "Deflate(): data_out: '" << data_out << "'" << std::endl;

    finished = last_chunk ? (remaining_ == 0) : (input.pos == input.size);
    TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  } while (!finished);

  avail_in_ = 0;
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
