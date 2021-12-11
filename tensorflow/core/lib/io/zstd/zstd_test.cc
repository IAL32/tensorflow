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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zstd/zstd_compression_options.h"
#include "tensorflow/core/lib/io/zstd/zstd_inputstream.h"
#include "tensorflow/core/lib/io/zstd/zstd_outputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace io {

static std::vector<int> InputBufferSizes() {
  return {10, 100, 200, 500, 1000, 10000};
}

static std::vector<int> OutputBufferSizes() { return {100, 200, 500, 1000}; }

static std::vector<int> NumCopies() { return {1, 50, 500}; }

static string GetRecord() {
  static const string lorem_ipsum =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
      " Fusce vehicula tincidunt libero sit amet ultrices. Vestibulum non "
      "felis augue. Duis vitae augue id lectus lacinia congue et ut purus. "
      "Donec auctor, nisl at dapibus volutpat, diam ante lacinia dolor, vel"
      "dignissim lacus nisi sed purus. Duis fringilla nunc ac lacus sagittis"
      " efficitur. Praesent tincidunt egestas eros, eu vehicula urna ultrices"
      " et. Aliquam erat volutpat. Maecenas vehicula risus consequat risus"
      " dictum, luctus tincidunt nibh imperdiet. Aenean bibendum ac erat"
      " cursus scelerisque. Cras lacinia in enim dapibus iaculis. Nunc porta"
      " felis lectus, ac tincidunt massa pharetra quis. Fusce feugiat dolor"
      " vel ligula rutrum egestas. Donec vulputate quam eros, et commodo"
      " purus lobortis sed.";
  return lorem_ipsum;
}

static string GenTestString(int copies = 1) {
  string result = "";
  for (int i = 0; i < copies; i++) {
    result += GetRecord();
  }
  return result;
}

typedef io::ZstdCompressionOptions CompressionOptions;

void TestWrite(uint8 input_buffer_size, uint8 output_buffer_size,
               bool with_flush = false) {
  Env* env = Env::Default();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString();
  std::unique_ptr<WritableFile> file_writer;
  string actual_result;
  string expected_result;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZstdOutputBuffer out(file_writer.get(), input_buffer_size, output_buffer_size,
                       output_options);

  TF_ASSERT_OK(out.Append(StringPiece(data)));

  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  // std::unique_ptr<RandomAccessFile> file_reader;
  // TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  // std::unique_ptr<RandomAccessInputStream> input_stream(
  //     new RandomAccessInputStream(file_reader.get()));
  // ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
  //                     input_options);
  // TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
  // EXPECT_EQ(result, data);
}

TEST(ZstdBuffers, SingleWrite) {
  TestWrite(100, 100, false);
}

}  // namespace io
}  // namespace tensorflow
