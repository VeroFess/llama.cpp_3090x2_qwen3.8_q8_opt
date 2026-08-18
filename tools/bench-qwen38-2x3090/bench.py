#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse


CORE_SCENARIOS = [
    "single_short",
    "eight_decode",
    "mixed_prefill_decode",
    "eight_varied_prefill",
    "shared_prefix_fork",
    "coding_agent_tool_loop",
]

ALL_SCENARIOS = [
    "single_short",
    "single_32k",
    "single_128k",
    "single_262k",
    "eight_decode",
    "mixed_prefill_decode",
    "eight_varied_prefill",
    "shared_prefix_fork",
    "coding_agent_tool_loop",
]

CHAT_TEMPLATE_RESERVE_TOKENS = 64


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def prompt_text(approx_tokens, request_id):
    return prompt_text_chars(max(1, approx_tokens) * 4, request_id)


def prompt_text_chars(target_chars, request_id):
    line = "int function_%d(int value) { return value + %d; }\n" % (request_id, request_id)
    return (line * (target_chars // len(line) + 1))[:target_chars]


def request_spec(request_id, prompt_tokens, content=None, messages=None, tools=None, output_tokens=None, calibrate=False):
    if messages is None:
        messages = [{"role": "user", "content": content if content is not None else prompt_text(prompt_tokens, request_id)}]
    return {
        "request_id": request_id,
        "prompt_tokens_requested": prompt_tokens,
        "messages": messages,
        "tools": tools,
        "output_tokens": output_tokens,
        "calibrate": calibrate,
        "calibration": None,
    }


def shared_prefix_scenario(prefix_tokens, output_tokens):
    prefix = prompt_text(prefix_tokens, 9000)
    warmup = request_spec(
        -1,
        prefix_tokens,
        content=prefix + "\nPrime the shared repository context.",
        output_tokens=1,
    )
    requests = [
        request_spec(
            request_id,
            prefix_tokens,
            content=prefix + "\nTask branch %d: identify one safe refactoring." % request_id,
            output_tokens=output_tokens,
        )
        for request_id in range(8)
    ]
    return {"warmups": [warmup], "requests": requests}


def coding_agent_scenario(context_tokens, output_tokens):
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a repository file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]
    requests = []
    for request_id in range(8):
        repo_context = prompt_text(context_tokens, 100 + request_id)
        tool_call_id = "call_%d" % request_id
        messages = [
            {"role": "system", "content": "You are a coding agent. Inspect evidence before editing."},
            {"role": "user", "content": repo_context + "\nInspect src/module_%d.cpp." % request_id},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "src/module_%d.cpp" % request_id}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "int module_%d(int value) { return value + %d; }\n" % (request_id, request_id),
            },
            {"role": "user", "content": "Explain the smallest safe change and its test."},
        ]
        requests.append(request_spec(
            request_id,
            context_tokens,
            messages=messages,
            tools=tools,
            output_tokens=output_tokens,
        ))
    return {"warmups": [], "requests": requests}


def scenario_catalog(quick, output_tokens):
    sizes = {
        "single_32k": 4096 if quick else 32768,
        "single_128k": 8192 if quick else 131072,
        "single_262k": 16384 if quick else 260000,
        "mixed_long": 16384 if quick else 131072,
        "shared_prefix": 4096 if quick else 32768,
        "coding_context": 1024 if quick else 8192,
    }

    def simple(prompts):
        return {
            "warmups": [],
            "requests": [
                request_spec(request_id, prompt_tokens, output_tokens=output_tokens, calibrate=True)
                for request_id, prompt_tokens in enumerate(prompts)
            ],
        }

    return {
        "single_short": simple([1024]),
        "single_32k": simple([sizes["single_32k"]]),
        "single_128k": simple([sizes["single_128k"]]),
        "single_262k": simple([sizes["single_262k"]]),
        "eight_decode": simple([4096] * 8),
        "mixed_prefill_decode": simple([sizes["mixed_long"]] + [4096] * 7),
        "eight_varied_prefill": simple(
            [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
            if not quick else [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
        ),
        "shared_prefix_fork": shared_prefix_scenario(sizes["shared_prefix"], output_tokens),
        "coding_agent_tool_loop": coding_agent_scenario(sizes["coding_context"], output_tokens),
    }


def request_body(model, spec, default_output_tokens):
    body = {
        "model": model,
        "messages": spec["messages"],
        "max_tokens": spec["output_tokens"] or default_output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if spec["tools"]:
        body["tools"] = spec["tools"]
        body["tool_choice"] = "auto"
    return body


def post_json(base_url, path, body, timeout=120):
    url = urllib.parse.urlparse(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=timeout)
    connection.request(
        "POST",
        (url.path.rstrip("/") or "") + path,
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    data = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise RuntimeError("request failed with HTTP %d: %s" % (response.status, data))
    return json.loads(data)


def tokenize_count(base_url, content):
    response = post_json(base_url, "/tokenize", {"content": content, "add_special": False})
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("tokenize response does not contain a token list")
    return len(tokens)


def calibrate_prompt(base_url, target_tokens, request_id):
    target_tokens = max(1, target_tokens)
    target_chars = target_tokens * 4
    attempts = []
    content = ""
    measured = 0
    for _ in range(4):
        content = prompt_text_chars(target_chars, request_id)
        measured = tokenize_count(base_url, content)
        attempts.append({"chars": target_chars, "tokens": measured})
        tolerance = max(4, target_tokens // 10000)
        if abs(measured - target_tokens) <= tolerance:
            break
        if measured == 0:
            raise RuntimeError("tokenizer returned zero tokens for non-empty content")
        target_chars = max(1, int(round(target_chars * target_tokens / measured)))
    return content, {
        "target_content_tokens": target_tokens,
        "measured_content_tokens": measured,
        "attempts": attempts,
    }


def calibrate_request(base_url, spec):
    if not spec["calibrate"]:
        return
    target_content_tokens = max(1, spec["prompt_tokens_requested"] - CHAT_TEMPLATE_RESERVE_TOKENS)
    try:
        content, calibration = calibrate_prompt(base_url, target_content_tokens, spec["request_id"])
        spec["messages"][0]["content"] = content
        spec["calibration"] = {"ok": True, **calibration}
    except Exception as exc:
        spec["calibration"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def stream_request(base_url, model, spec, default_output_tokens):
    url = urllib.parse.urlparse(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=7200)
    body = request_body(model, spec, default_output_tokens)
    encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_path = (url.path.rstrip("/") or "") + "/v1/chat/completions"
    request_headers = {"Content-Type": "application/json"}

    started_wall = time.time()
    started = time.perf_counter()
    connection.request("POST", request_path, encoded_body, request_headers)
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        ended = time.perf_counter()
        connection.close()
        return {
            "request_id": spec["request_id"],
            "success": False,
            "http_status": response.status,
            "error": error,
            "prompt_tokens_requested": spec["prompt_tokens_requested"],
            "calibration": spec["calibration"],
            "completion_tokens": 0,
            "ttft_seconds": ended - started,
            "elapsed_seconds": ended - started,
            "tpot_seconds": 0.0,
            "trace": {
                "started_unix": started_wall,
                "request_path": request_path,
                "request_headers": request_headers,
                "request_body": body,
                "request_sha256": hashlib.sha256(encoded_body).hexdigest(),
                "events": [],
            },
        }

    first_token = None
    chunks = 0
    prompt_tokens = 0
    completion_tokens = 0
    events = []
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        offset = time.perf_counter() - started
        if data == "[DONE]":
            events.append({"offset_seconds": offset, "data": "[DONE]"})
            break
        event = json.loads(data)
        events.append({"offset_seconds": offset, "data": event})
        usage = event.get("usage")
        if usage:
            prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens", 0)))
            completion_tokens = max(completion_tokens, int(usage.get("completion_tokens", 0)))
        choices = event.get("choices") or []
        delta = choices[0].get("delta") if choices else None
        if delta and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
            chunks += 1
            if first_token is None:
                first_token = time.perf_counter()

    ended = time.perf_counter()
    connection.close()
    if first_token is None:
        first_token = ended
    if completion_tokens == 0:
        completion_tokens = chunks
    decode_seconds = max(0.0, ended - first_token)
    return {
        "request_id": spec["request_id"],
        "success": True,
        "http_status": response.status,
        "prompt_tokens_requested": spec["prompt_tokens_requested"],
        "calibration": spec["calibration"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token - started,
        "elapsed_seconds": ended - started,
        "tpot_seconds": decode_seconds / max(1, completion_tokens - 1),
        "trace": {
            "started_unix": started_wall,
            "request_path": request_path,
            "request_headers": request_headers,
            "request_body": body,
            "request_sha256": hashlib.sha256(encoded_body).hexdigest(),
            "events": events,
        },
    }


def guarded_stream_request(base_url, model, spec, output_tokens):
    try:
        return stream_request(base_url, model, spec, output_tokens)
    except Exception as exc:
        return {
            "request_id": spec["request_id"],
            "success": False,
            "http_status": 0,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "prompt_tokens_requested": spec["prompt_tokens_requested"],
            "calibration": spec["calibration"],
            "completion_tokens": 0,
            "ttft_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "tpot_seconds": 0.0,
            "trace": {"request_body": request_body(model, spec, output_tokens), "events": []},
        }


def run_scenario(base_url, model, name, definition, output_tokens, prompt_calibration):
    if prompt_calibration:
        for spec in definition["warmups"] + definition["requests"]:
            calibrate_request(base_url, spec)
    warmups = [
        guarded_stream_request(base_url, model, spec, output_tokens)
        for spec in definition["warmups"]
    ]
    started = time.time()
    specs = definition["requests"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = [
            executor.submit(guarded_stream_request, base_url, model, spec, output_tokens)
            for spec in specs
        ]
        requests = [future.result() for future in futures]
    elapsed = time.time() - started
    successful = [request for request in requests if request["success"]]
    total_output = sum(request["completion_tokens"] for request in successful)
    return {
        "name": name,
        "success": len(successful) == len(requests) and all(item["success"] for item in warmups),
        "elapsed_seconds": elapsed,
        "aggregate_output_tokens_per_second": total_output / max(elapsed, 1e-9),
        "p50_ttft_seconds": statistics.median(request["ttft_seconds"] for request in successful) if successful else 0.0,
        "p95_ttft_seconds": percentile([request["ttft_seconds"] for request in successful], 0.95),
        "p50_tpot_seconds": statistics.median(request["tpot_seconds"] for request in successful) if successful else 0.0,
        "p95_tpot_seconds": percentile([request["tpot_seconds"] for request in successful], 0.95),
        "warmups": warmups,
        "requests": requests,
    }


def get_json(base_url, path):
    url = urllib.parse.urlparse(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=30)
    connection.request("GET", (url.path.rstrip("/") or "") + path)
    response = connection.getresponse()
    data = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise RuntimeError("request failed with HTTP %d: %s" % (response.status, data))
    return json.loads(data)


def safe_get_json(base_url, path):
    try:
        return {"ok": True, "value": get_json(base_url, path)}
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def get_text(base_url, path):
    url = urllib.parse.urlparse(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=30)
    connection.request("GET", (url.path.rstrip("/") or "") + path)
    response = connection.getresponse()
    data = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise RuntimeError("request failed with HTTP %d: %s" % (response.status, data))
    return data


def safe_get_text(base_url, path):
    try:
        return {"ok": True, "value": get_text(base_url, path)}
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def command_output(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=30)
        return {"ok": True, "argv": command, "stdout": result.stdout.strip()}
    except Exception as exc:
        return {"ok": False, "argv": command, "error": "%s: %s" % (type(exc).__name__, exc)}


class GpuMemoryPeakMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.peaks = []
        self.samples = 0
        self.error = ""

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.run([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], check=True, capture_output=True, text=True, timeout=10).stdout
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if not self.peaks:
                    self.peaks = values
                elif len(values) == len(self.peaks):
                    self.peaks = [max(old, new) for old, new in zip(self.peaks, values)]
                self.samples += 1
            except Exception as exc:
                self.error = "%s: %s" % (type(exc).__name__, exc)
            self.stop_event.wait(0.5)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        return {"peaks_mib": self.peaks, "samples": self.samples, "error": self.error}


def runtime_snapshot(base_url):
    return {
        "captured_unix": time.time(),
        "gpu_state": command_output([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "server": {
            "profile": safe_get_json(base_url, "/debug/profile"),
            "graphs": safe_get_json(base_url, "/debug/graphs"),
            "scheduler": safe_get_json(base_url, "/debug/scheduler"),
            "models": safe_get_json(base_url, "/v1/models"),
            "metrics": safe_get_text(base_url, "/metrics"),
        },
    }


def collect_manifest(args):
    script_dir = pathlib.Path(__file__).resolve().parent
    repository = script_dir.parent.parent
    git_head = command_output(["git", "rev-parse", "HEAD"], str(repository))
    host = {
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": git_head,
        "git_status": command_output(["git", "status", "--short"], str(repository)),
        "gpu_state": command_output([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "gpu_p2p_read": command_output(["nvidia-smi", "topo", "-p2p", "r"]),
        "gpu_p2p_write": command_output(["nvidia-smi", "topo", "-p2p", "w"]),
    }
    external = None
    if args.manifest_json:
        external = json.loads(pathlib.Path(args.manifest_json).read_text(encoding="utf-8"))
    return {
        "runner_argv": sys.argv,
        "source_commit": args.source_commit or (git_head.get("stdout", "") if git_head["ok"] else ""),
        "server_command": args.server_command,
        "deployment_axes": {
            "variant": args.variant,
            "speculation": args.speculation,
            "cuda_graph": args.cuda_graph,
            "pipeline": args.pipeline,
            "kv_mode": args.kv_mode,
            "split_boundary": args.split_boundary,
        },
        "model": args.model,
        "model_sha256": args.model_sha256,
        "base_url": args.base_url,
        "host": host,
        "server": {
            "profile": safe_get_json(args.base_url, "/debug/profile"),
            "graphs": safe_get_json(args.base_url, "/debug/graphs"),
            "scheduler": safe_get_json(args.base_url, "/debug/scheduler"),
            "models": safe_get_json(args.base_url, "/v1/models"),
        },
        "external": external,
        "notes": args.notes,
    }


def validate_manifest(args, manifest):
    checks = []

    def add(name, expected, actual, available=True):
        if not available:
            status = "incomplete"
        else:
            status = "pass" if actual == expected else "fail"
        checks.append({"name": name, "status": status, "expected": expected, "actual": actual})

    add("source_commit", True, bool(manifest["source_commit"]))
    add("model_sha256", True, bool(manifest["model_sha256"]))
    add("server_command", True, bool(manifest["server_command"]))
    add("gpu_topology", True, manifest["host"]["gpu_topology"]["ok"])
    add("gpu_p2p_read", True, manifest["host"]["gpu_p2p_read"]["ok"])
    add("gpu_p2p_write", True, manifest["host"]["gpu_p2p_write"]["ok"])

    profile_result = manifest["server"]["profile"]
    profile = profile_result.get("value", {}) if profile_result["ok"] else {}
    add("server_profile", True, profile_result["ok"])
    if args.kv_mode == "q8_0":
        add("kv_cache_type_k", "q8_0", profile.get("kv_cache_type_k"), profile_result["ok"])
        add("kv_cache_type_v", "q8_0", profile.get("kv_cache_type_v"), profile_result["ok"])
    if args.split_boundary != "auto":
        parts = args.split_boundary.split("/", 1)
        try:
            expected_boundary = {"gpu0": int(parts[0]), "gpu1": int(parts[1])} if len(parts) == 2 else None
        except ValueError:
            expected_boundary = None
        add("split_boundary", expected_boundary, profile.get("layer_boundary"), profile_result["ok"] and expected_boundary is not None)

    graph_result = manifest["after"]["server"]["graphs"]
    graph = graph_result.get("value", {}) if graph_result["ok"] else {}
    add("graph_debug", True, graph_result["ok"])
    if graph_result["ok"]:
        graph_activity = int(graph.get("hits", 0)) + int(graph.get("captures", 0))
        add("cuda_graph_activity", args.cuda_graph == "on", graph_activity > 0)
        pipeline = graph.get("pipeline_transfer", {})
        add("pipeline_depth", 2 if args.pipeline == "on" else 1, pipeline.get("depth"))
        if pipeline.get("mode") == "pinned_host_staged":
            add("direct_peer_transfers", 0, pipeline.get("direct_peer_transfers"))
            add("staged_byte_balance", pipeline.get("d2h_bytes"), pipeline.get("h2d_bytes"))

    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "incomplete" for check in checks):
        status = "incomplete"
    else:
        status = "pass"
    return {"status": status, "checks": checks}


def write_csv(path, scenarios, axes):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "variant", "speculation", "cuda_graph", "pipeline", "kv_mode", "split_boundary",
            "scenario", "success", "elapsed_seconds", "aggregate_output_tokens_per_second",
            "p50_ttft_seconds", "p95_ttft_seconds", "p50_tpot_seconds", "p95_tpot_seconds",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in scenarios:
            writer.writerow({
                **axes,
                "scenario": result["name"],
                "success": result["success"],
                "elapsed_seconds": result["elapsed_seconds"],
                "aggregate_output_tokens_per_second": result["aggregate_output_tokens_per_second"],
                "p50_ttft_seconds": result["p50_ttft_seconds"],
                "p95_ttft_seconds": result["p95_ttft_seconds"],
                "p50_tpot_seconds": result["p50_tpot_seconds"],
                "p95_tpot_seconds": result["p95_tpot_seconds"],
            })


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8 dual RTX 3090 serving workload")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3.8-27B-Q8_0.gguf")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--server-command", default="")
    parser.add_argument("--output-json", default="qwen38-benchmark.json")
    parser.add_argument("--output-csv", default="qwen38-benchmark.csv")
    parser.add_argument("--manifest-json")
    parser.add_argument("--notes", default="")
    parser.add_argument("--strict-manifest", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-prompt-calibration", dest="prompt_calibration", action="store_false")
    parser.set_defaults(prompt_calibration=True)
    parser.add_argument("--suite", choices=["core", "full"], default="core")
    parser.add_argument("--scenario", action="append", choices=ALL_SCENARIOS)
    parser.add_argument("--variant", default="production")
    parser.add_argument("--speculation", choices=["adaptive", "mtp", "dflash", "none"], default="adaptive")
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    parser.add_argument("--pipeline", choices=["on", "off"], default="on")
    parser.add_argument("--kv-mode", choices=["q8_0", "turbo", "tcq"], default="q8_0")
    parser.add_argument("--split-boundary", default="auto")
    args = parser.parse_args()

    output_tokens = 32 if args.quick else 256
    catalog = scenario_catalog(args.quick, output_tokens)
    selected = args.scenario or (CORE_SCENARIOS if args.suite == "core" else ALL_SCENARIOS)
    manifest = collect_manifest(args)
    peak_monitor = GpuMemoryPeakMonitor()
    peak_monitor.start()
    try:
        results = [
            run_scenario(args.base_url, args.model, name, catalog[name], output_tokens, args.prompt_calibration)
            for name in selected
        ]
    finally:
        manifest["gpu_memory_peak"] = peak_monitor.stop()
    manifest["after"] = runtime_snapshot(args.base_url)
    manifest["validation"] = validate_manifest(args, manifest)
    payload = {
        "schema_version": 2,
        "created_unix": int(time.time()),
        "quick": args.quick,
        "prompt_calibration": args.prompt_calibration,
        "manifest": manifest,
        "scenarios": results,
    }
    pathlib.Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output_csv, results, manifest["deployment_axes"])
    if not all(result["success"] for result in results) or (args.strict_manifest and manifest["validation"]["status"] != "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
