# llama.cpp for Qwen3.8 27B Q8 on two RTX 3090s

This fork is tuned for one deployment: `Qwen3.8-27B-Q8_0.gguf` on two 24 GB RTX 3090 cards. The fast path uses a 1:1 tensor split, CUDA P2P, NCCL, Q8_0 paged KV, FlashAttention, MTP, and up to eight resident sequences. The normal llama.cpp codebase is still here, but this profile is deliberately specific to this model and hardware.

## Recommended deployment

Run one Linux `llama-server` process with both GPUs visible. Enable Resizable BAR, verify CUDA P2P read and write access in both directions, and install NCCL before configuring the build. Cooling matters on long prompts. Both cards need enough airflow to hold their normal clocks instead of sitting at the thermal limit.

Use the supplied SM86 release preset:

```bash
cmake --preset qwen38-3090-release
cmake --build --preset qwen38-3090-release -j
```

Start the server with tensor split and P2P enabled:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=nccl \
./build-qwen38-3090/bin/llama-server \
  --model /models/Qwen3.8-27B-Q8_0.gguf \
  --alias Qwen3.8-27B-Q8_0 \
  --profile qwen38-27b-q8-2x3090 \
  --split-mode tensor \
  --ubatch-size 1024 \
  --reasoning off \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics
```

The profile fixes the context limit at 262,144 tokens, allows eight resident sequences, keeps K and V cache in Q8_0, enables two-token MTP drafts, and rejects settings that fall outside the tested configuration. Use the pinned model revision and SHA-256 listed in [the deployment notes](docs/qwen38-2x3090.md). The included Docker Compose file is convenient for packaging, but the native command above is the reference deployment and makes driver, NCCL, P2P, and clock checks easier.

## Measured performance

These numbers were measured on August 18, 2026 with the server left running in the configuration above. The model was already loaded and the CUDA graph cache was warm. No server restart is part of the reported timings.

| Workload | Request shape | Result | MTP acceptance |
| --- | --- | ---: | ---: |
| Single generation | 1,011 uncached prompt tokens, 256 output tokens | 64.04 output tok/s | 60.43% (139/230) |
| Multi-turn coding, low reasoning | 3 turns | 80.95 output tok/s | 91.55% (271/296) |
| 8 concurrent agents | 8 x 4,083 uncached prompt tokens, 8 x 256 output tokens | 124.87 aggregate output tok/s | 59.43% (1,106/1,861) |
| 8K prefill | 8,181 uncached prompt tokens, 32 output tokens | 1,443.05 prompt tok/s | 60.71% (17/28) |
| 32K prefill | 32,757 uncached prompt tokens, 256 output tokens | 1,246.15 prompt tok/s | 60.17% (139/231) |

Generation speed starts after the first token. MTP acceptance is accepted draft tokens divided by generated draft tokens.

GPU 1 still reaches its thermal limit during repeated concurrent runs and falls to roughly 1.5 to 1.65 GHz near the end.

## 262K acceptance result

The earlier 262,120-token prompt plus one generated token acceptance run completed in 761.75 seconds and reported 344.17 prompt tokens per second. It was not repeated for the backend-sampling build above. This is a thermal-throttled result, not a clean peak measurement. GPU 1 stayed around 89 to 91 C and repeatedly dropped its core clock during the run. Treat this number as the result of the current cooling setup, not the throughput limit of two RTX 3090 cards holding their normal clocks.

## Upstream llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
