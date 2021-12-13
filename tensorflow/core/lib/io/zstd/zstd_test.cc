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

#include <zstd.h>

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

void TestAllCombinations(CompressionOptions input_options,
                         CompressionOptions output_options) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    // Write to compressed file
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        std::cout << "num_copies: " << file_size
                  << ", input_buf_size: " << input_buf_size
                  << ", output_buf_size: " << output_buf_size << std::endl
                  << std::endl;
        std::unique_ptr<WritableFile> file_writer;
        TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
        tstring result;

        ZstdOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                             output_options);

        TF_ASSERT_OK(out.Append(StringPiece(data)));
        TF_ASSERT_OK(out.Close());
        TF_ASSERT_OK(file_writer->Flush());
        TF_ASSERT_OK(file_writer->Close());

        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);
        TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
        EXPECT_EQ(result, data);
      }
    }
  }
}

TEST(ZlibBuffers, DefaultOptions) {
  TestAllCombinations(CompressionOptions::DEFAULT(),
                      CompressionOptions::DEFAULT());
}

void TestMultipleWrites(uint8 input_buf_size, uint8 output_buf_size,
                        int num_writes, bool with_flush = false) {
  Env* env = Env::Default();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString();
  std::unique_ptr<WritableFile> file_writer;
  string actual_result;
  string expected_result;

  std::cout << "Tmp file: " << fname << std::endl;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZstdOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);

  for (int i = 0; i < num_writes; i++) {
    TF_ASSERT_OK(out.Append(StringPiece(data)));
    if (with_flush) {
      TF_ASSERT_OK(out.Flush());
    }
    strings::StrAppend(&expected_result, data);
  }
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);

  for (int i = 0; i < num_writes; i++) {
    tstring decompressed_output;
    TF_ASSERT_OK(in.ReadNBytes(data.size(), &decompressed_output));
    std::cout << "TEST(): new read" << std::endl << std::endl;
    strings::StrAppend(&actual_result, decompressed_output);
  }

  EXPECT_EQ(actual_result, expected_result);
}

TEST(ZstdBuffers, MultipleWritesWithoutFlush) {
  TestMultipleWrites(200, 200, 10);
}

void TestWrite(uint8 input_buf_size, uint8 output_buf_size,
               bool with_flush = false) {
  Env* env = Env::Default();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString(500);
  std::unique_ptr<WritableFile> file_writer;
  tstring result;

  std::cout << "datasize: " << data.size()
            << ", data (last 10): " << data.substr(data.size() - 10)
            << std::endl
            << std::endl;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZstdOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);

  std::cout << "Tmp file: " << fname << std::endl;

  TF_ASSERT_OK(out.Append(StringPiece(data)));

  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));

  ZstdInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);
  TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
  EXPECT_EQ(result, data);
}

TEST(ZstdBuffers, SingleWrite) { TestWrite(100, 100, false); }

size_t ZstdReferenceCompress(WritableFile* file_writer, string* data,
                             size_t buffOutSize) {
  /* Create the context. */
  void* buffOut = malloc(buffOutSize);
  ZSTD_CCtx* const cctx = ZSTD_createCCtx();
  const size_t cLevel = ZSTD_CLEVEL_DEFAULT;
  const size_t nbThreads = 0;

  std::cout << "Original data size: " << data->size() << std::endl;

  /* Set any parameters you want.
   * Here we set the compression level, and enable the checksum.
   */
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, cLevel);
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1);
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, nbThreads);

  ZSTD_inBuffer input = {reinterpret_cast<const void*>(data->c_str()),
                         data->size(), 0};
  const int lastChunk = true;
  const ZSTD_EndDirective mode = lastChunk ? ZSTD_e_end : ZSTD_e_continue;

  int finished;
  size_t written_bytes = 0;

  tstring to_compress_data;
  to_compress_data.append(data->c_str() + input.pos, input.size - input.pos);
  std::cout << "Compressing data: '" << to_compress_data << "'" << std::endl;

  do {
    /* Compress into the output buffer and write all of the output to
     * the file so we can reuse the buffer next iteration.
     */
    ZSTD_outBuffer output = {buffOut, buffOutSize, 0};
    size_t const remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
    tstring compressed_data;
    written_bytes += output.pos;
    compressed_data.append(reinterpret_cast<char*>(buffOut), output.pos);
    std::cout << "Compressed data: '" << compressed_data << "'" << std::endl;
    file_writer->Append(compressed_data);
    /* If we're on the last chunk we're finished when zstd returns 0,
     * which means its consumed all the input AND finished the frame.
     * Otherwise, we're finished when we've consumed all the input.
     */

    std::cout << "remaining: " << remaining << std::endl;
    finished = lastChunk ? (remaining == 0) : (input.pos == input.size);
    std::cout << "input.pos: " << input.pos << std::endl;
  } while (!finished);

  ZSTD_freeCCtx(cctx);
  free(buffOut);

  return written_bytes;
}

tstring ZstdReferenceDecompress(char* compressed_data, size_t size,
                                size_t buffInSize, size_t buffOutSize) {
  void* const buffIn = reinterpret_cast<void*>(compressed_data);
  void* nextIn = buffIn;
  void* const buffOut = malloc(buffOutSize);
  size_t read = std::min(buffInSize, size);
  size_t totalRead = 0;
  tstring result;

  ZSTD_DCtx* const dctx = ZSTD_createDCtx();

  size_t lastRet = 0;

  std::cout << "size: " << size << std::endl;
  while (totalRead < size) {
    std::cout << "totalRead: " << totalRead << std::endl;
    ZSTD_inBuffer input = {nextIn, read, 0};

    /* Given a valid frame, zstd won't consume the last byte of the frame
     * until it has flushed all of the decompressed data of the frame.
     * Therefore, instead of checking if the return code is 0, we can
     * decompress just check if input.pos < input.size.
     */
    while (input.pos < input.size) {
      ZSTD_outBuffer output = {buffOut, buffOutSize, 0};
      /* The return code is zero if the frame is complete, but there may
       * be multiple frames concatenated together. Zstd will automatically
       * reset the context when a frame is complete. Still, calling
       * ZSTD_DCtx_reset() can be useful to reset the context to a clean
       * state, for instance if the last decompression call returned an
       * error.
       */
      size_t const ret = ZSTD_decompressStream(dctx, &output, &input);
      if (ZSTD_isError(ret)) {
        std::cout << ZSTD_getErrorName(ret) << std::endl;
        break;
      }
      tstring tmp;
      tmp.append(reinterpret_cast<char*>(buffOut), output.pos);
      result.append(reinterpret_cast<char*>(buffOut), output.pos);
      std::cout << "Uncompressed: " << tmp << std::endl;
      lastRet = ret;

      std::cout << "input.pos: " << input.pos << ", input.size: " << input.size
                << std::endl;
    }

    nextIn += read;
    totalRead += read;
    read = std::min(buffInSize, size - totalRead);
  }

  if (lastRet != 0) {
    /* The last return value from ZSTD_decompressStream did not end on a
     * frame, but we reached the end of the file! We assume this is an
     * error, and the input was truncated.
     */
    fprintf(stderr, "EOF before end of stream: %zu\n", lastRet);
    exit(1);
  }

  ZSTD_freeDCtx(dctx);
  free(buffOut);

  return result;
}

void ReferenceWriteRead() {
  Env* env = Env::Default();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString(1);
  std::unique_ptr<WritableFile> file_writer;
  tstring compressed_data;

  std::cout << "Tmp file: " << fname << std::endl;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  const size_t input_buf_size = 100;
  const size_t output_buf_size = 100;

  const size_t written_bytes =
      ZstdReferenceCompress(file_writer.get(), &data, output_buf_size);

  // TF_ASSERT_OK(out.Append(StringPiece(data)));

  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  // size_t written_bytes = 421;  // for GenTestString(1);
  // size_t written_bytes = 447;  // for GenTestString(500);

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));

  TF_ASSERT_OK(input_stream->ReadNBytes(written_bytes, &compressed_data));
  std::cout << "compressed_data: " << compressed_data << std::endl;

  tstring result = ZstdReferenceDecompress(
      compressed_data.data(), written_bytes, input_buf_size, output_buf_size);

  EXPECT_EQ(result, data);
}

TEST(ZstdReference, ReferenceWriteRead) { ReferenceWriteRead(); }

}  // namespace io
}  // namespace tensorflow
