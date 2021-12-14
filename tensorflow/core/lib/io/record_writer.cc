/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/record_writer.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {
namespace {
bool IsZlibCompressed(const RecordWriterOptions& options) {
  return options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION;
}

bool IsSnappyCompressed(const RecordWriterOptions& options) {
  return options.compression_type == RecordWriterOptions::SNAPPY_COMPRESSION;
}

bool IsZstdCompressed(const RecordWriterOptions& options) {
  return options.compression_type == RecordWriterOptions::ZSTD_COMPRESSION;
}

}  // namespace

RecordWriterOptions RecordWriterOptions::CreateRecordWriterOptions(
    const string& compression_type) {
  RecordWriterOptions options;
#if defined(IS_SLIM_BUILD)
  if (compression_type != compression::kNone) {
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
  }
#else
  if (compression_type == compression::kZlib) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
  } else if (compression_type == compression::kSnappy) {
    options.compression_type = io::RecordWriterOptions::SNAPPY_COMPRESSION;
  } else if (compression_type == compression::kZstd) {
    options.compression_type = io::RecordWriterOptions::ZSTD_COMPRESSION;
    options.zstd_options = io::ZstdCompressionOptions::DEFAULT();
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
#endif
  return options;
}

RecordWriter::RecordWriter(WritableFile* dest,
                           const RecordWriterOptions& options)
    : dest_(dest), options_(options) {
#if defined(IS_SLIM_BUILD)
  if (options.compression_type != RecordWriterOptions::NONE) {
    LOG(FATAL) << "Compression is unsupported on mobile platforms.";
  }
#else
  if (IsZlibCompressed(options)) {
    ZlibOutputBuffer* zlib_output_buffer = new ZlibOutputBuffer(
        dest, options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options);
    Status s = zlib_output_buffer->Init();
    if (!s.ok()) {
      LOG(FATAL) << "Failed to initialize Zlib inputbuffer. Error: "
                 << s.ToString();
    }
    dest_ = zlib_output_buffer;
  } else if (IsSnappyCompressed(options)) {
    dest_ =
        new SnappyOutputBuffer(dest, options.snappy_options.input_buffer_size,
                               options.snappy_options.output_buffer_size);
  } else if (IsZstdCompressed(options)) {
    ZstdOutputBuffer* zstd_output_buffer = new ZstdOutputBuffer(
        dest, options.zstd_options.input_buffer_size,
        options.zstd_options.output_buffer_size, options.zstd_options);

    dest_ = zstd_output_buffer;
  } else if (options.compression_type == RecordWriterOptions::NONE) {
    // Nothing to do
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
#endif
}

RecordWriter::~RecordWriter() {
  if (dest_ != nullptr) {
    Status s = Close();
    if (!s.ok()) {
      LOG(ERROR) << "Could not finish writing file: " << s;
    }
  }
}

Status RecordWriter::WriteRecord(StringPiece data) {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data.data(), data.size());
  PopulateFooter(footer, data.data(), data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  TF_RETURN_IF_ERROR(dest_->Append(data));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}

#if defined(TF_CORD_SUPPORT)
Status RecordWriter::WriteRecord(const absl::Cord& data) {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data);
  PopulateFooter(footer, data);
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  TF_RETURN_IF_ERROR(dest_->Append(data));
  return dest_->Append(StringPiece(footer, sizeof(footer)));
}
#endif

Status RecordWriter::Close() {
  if (dest_ == nullptr) return Status::OK();
  if (IsZlibCompressed(options_) || IsSnappyCompressed(options_) || IsZstdCompressed(options_)) {
    Status s = dest_->Close();
    delete dest_;
    dest_ = nullptr;
    return s;
  }
  return Status::OK();
}

Status RecordWriter::Flush() {
  if (dest_ == nullptr) {
    return Status(::tensorflow::error::FAILED_PRECONDITION,
                  "Writer not initialized or previously closed");
  }
  return dest_->Flush();
}

}  // namespace io
}  // namespace tensorflow
