#include "common.h"

#include "whisper.h"

#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <tensor_filter_cpp.hh>


class nnstreamer_whisper_filter : public tensor_filter_cpp
{
  private:
  struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
  };

  whisper_params params;
  whisper_full_params wparams;
  struct whisper_context *ctx;

  nnstreamer_whisper_filter (const char *modelName) : tensor_filter_cpp (modelName)
  {
    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    ctx = whisper_init_from_file(params.model.c_str());

    wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = !use_vad;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.audio_ctx        = params.audio_ctx;
    wparams.speed_up         = params.speed_up;

    // disable temperature fallback
    wparams.temperature_inc  = -1.0f;
  }

  ~nnstreamer_whisper_filter ()
  {
    whisper_print_timings(ctx);
    whisper_free (ctx);
  }

  public:
  int getInputDim (GstTensorsInfo *info)
  {
    info->num_tensors = 1;
    info->info[0].type = _NNS_FLOAT32;
    info->info[0].dimension[0] = 48000;
    info->info[0].dimension[1] = 1;
    info->info[0].dimension[2] = 1;
    for (int i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
      info->info[0].dimension[i] = 1;

    return 0;
  }

  int getOutputDim (GstTensorsInfo *info)
  {
    info->num_tensors = 1;
    info->info[0].type = _NNS_FLOAT32;
    info->info[0].dimension[0] = 48000;
    info->info[0].dimension[1] = 1;
    info->info[0].dimension[2] = 1;
    info->info[0].dimension[3] = 1;
    for (int i = 4; i < NNS_TENSOR_RANK_LIMIT; i++)
      info->info[0].dimension[i] = 1;

    return 0;
  }

  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out)
  {
    return -EINVAL;
  }

  bool isAllocatedBeforeInvoke ()
  {
    return true;
  }

  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    whisper_full (ctx, wparams, (float *) in->data, in->size / sizeof (float));
    // print result
    {
      printf("\33[2K\r");
      printf("%s", std::string(100, ' ').c_str());
      printf("\33[2K\r");

      const int n_segments = whisper_full_n_segments (ctx);
      for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text (ctx, i);
        printf("%s", text);
        fflush(stdout);
        printf("\n");
      }
    }

    return 0;
  }

  static nnstreamer_whisper_filter &get_instance ()
  {
    static nnstreamer_whisper_filter instance ("nnstreamer_whisper_filter");
    return instance;
  }
};

void init_shared_lib (void) __attribute__ ((constructor));

void
init_shared_lib (void)
{
  nnstreamer_whisper_filter &mccf = nnstreamer_whisper_filter::get_instance ();
  mccf._register ();
}
