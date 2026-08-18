#!/usr/bin/env python3

import argparse
import http.client
import json
import pathlib
import time
import urllib.parse


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def split_sse_frames(buffer):
    frames = []
    while True:
        delimiters = [(buffer.find(b"\n\n"), 2), (buffer.find(b"\r\n\r\n"), 4)]
        delimiters = [(index, size) for index, size in delimiters if index >= 0]
        if not delimiters:
            return frames, buffer
        index, size = min(delimiters)
        frames.append(buffer[:index])
        buffer = buffer[index + size:]


def parse_sse_frame(frame):
    event = "message"
    data = []
    for raw_line in frame.replace(b"\r\n", b"\n").split(b"\n"):
        line = raw_line.decode("utf-8", errors="strict")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    payload = "\n".join(data)
    if not payload:
        return {"event": event, "data": None}
    if payload == "[DONE]":
        return {"event": event, "data": "[DONE]"}
    return {"event": event, "data": json.loads(payload)}


def request(base_url, path, body, headers=None, bytewise_sse=False):
    url = urllib.parse.urlparse(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=7200)
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request_path = (url.path.rstrip("/") or "") + path
    started = time.time()
    connection.request("POST", request_path, encoded, request_headers)
    response = connection.getresponse()
    response_headers = dict(response.getheaders())

    if bytewise_sse and response.status == 200:
        pending = b""
        events = []
        transport_chunks = 0
        while True:
            byte = response.read(1)
            if not byte:
                break
            transport_chunks += 1
            pending += byte
            frames, pending = split_sse_frames(pending)
            events.extend(parse_sse_frame(frame) for frame in frames if frame)
        require(not pending.strip(), "SSE stream ended with an incomplete frame")
        response_body = {"events": events, "transport_chunks": transport_chunks}
    else:
        raw = response.read().decode("utf-8", errors="replace")
        try:
            response_body = json.loads(raw)
        except json.JSONDecodeError:
            response_body = {"raw": raw}
    connection.close()
    return {
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "request": {
            "path": request_path,
            "headers": request_headers,
            "body": body,
        },
        "response": {
            "status": response.status,
            "headers": response_headers,
            "body": response_body,
        },
    }


def tools_fixture():
    return [{
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
    }, {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repository files",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }]


def openai_history(arguments_as_string, preserve_thinking):
    unicode_path = json.loads('"src/\\u5de5\\u5177.cpp"')
    escaped_path = "src/quoted\\\"name\"\\file.cpp"

    def arguments(value):
        return json.dumps(value, ensure_ascii=False) if arguments_as_string else value

    return [{"role": "user", "content": "Inspect the repository."}, {
        "role": "assistant",
        "content": "",
        "reasoning_content": "First reasoning step.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": arguments({"path": unicode_path})},
        }, {
            "id": "call_2",
            "type": "function",
            "function": {"name": "list_files", "arguments": arguments({"path": escaped_path})},
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "first result\nwith newline",
    }, {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": unicode_path,
    }, {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Second reasoning step after tool results.",
        "tool_calls": [{
            "id": "call_3",
            "type": "function",
            "function": {"name": "read_file", "arguments": arguments({"path": "src/main.cpp"})},
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_3",
        "content": "int main() { return 0; }",
    }, {
        "role": "user",
        "content": "Give the next action.",
    }], {
        "preserve_thinking": preserve_thinking,
        "parallel_tool_calls": True,
    }


def fixture_chat_history(base_url, model, arguments_as_string, preserve_thinking):
    messages, kwargs = openai_history(arguments_as_string, preserve_thinking)
    trace = request(base_url, "/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "tools": tools_fixture(),
        "parallel_tool_calls": kwargs["parallel_tool_calls"],
        "chat_template_kwargs": {"preserve_thinking": kwargs["preserve_thinking"]},
        "max_tokens": 2,
        "temperature": 0,
    })
    require(trace["response"]["status"] == 200, "Chat history request failed")
    body = trace["response"]["body"]
    require(body.get("object") == "chat.completion", "Unexpected Chat response object")
    require(bool(body.get("choices")), "Chat response has no choices")
    return [trace]


def fixture_malformed_arguments(base_url, model):
    messages, _ = openai_history(True, True)
    messages[1]["tool_calls"][0]["function"]["arguments"] = "{not json}"
    trace = request(base_url, "/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "tools": tools_fixture(),
        "max_tokens": 1,
        "temperature": 0,
    })
    require(trace["response"]["status"] == 400, "Malformed arguments were not rejected with HTTP 400")
    return [trace]


def fixture_sse_bytewise(base_url, model):
    trace = request(base_url, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 4,
        "temperature": 0,
        "stream": True,
    }, bytewise_sse=True)
    require(trace["response"]["status"] == 200, "SSE request failed")
    events = trace["response"]["body"]["events"]
    require(any(event["data"] == "[DONE]" for event in events), "SSE stream has no DONE event")
    require(any(isinstance(event["data"], dict) for event in events), "SSE stream has no JSON event")
    require(trace["response"]["body"]["transport_chunks"] > len(events), "SSE was not consumed bytewise")
    return [trace]


def fixture_responses_continuation(base_url, model):
    first = request(base_url, "/v1/responses", {
        "model": model,
        "input": "Remember marker alpha.",
        "max_output_tokens": 8,
        "temperature": 0,
    })
    require(first["response"]["status"] == 200, "First Responses request failed")
    response_id = first["response"]["body"].get("id", "")
    require(response_id.startswith("resp_"), "First Responses request has no response id")

    second = request(base_url, "/v1/responses", {
        "model": model,
        "input": "Continue with marker beta.",
        "previous_response_id": response_id,
        "instructions": "Answer briefly.",
        "max_output_tokens": 8,
        "temperature": 0,
    })
    require(second["response"]["status"] == 200, "Responses continuation failed")
    require(second["response"]["body"].get("id", "").startswith("resp_"), "Continuation has no response id")

    unknown = request(base_url, "/v1/responses", {
        "model": model,
        "input": "This must fail.",
        "previous_response_id": "resp_missing_fixture",
        "max_output_tokens": 1,
    })
    require(unknown["response"]["status"] == 400, "Unknown previous_response_id was not rejected")
    return [first, second, unknown]


def fixture_anthropic_claude_code(base_url, model):
    unicode_path = json.loads('"src/\\u5de5\\u5177.cpp"')
    trace = request(base_url, "/v1/messages", {
        "model": model,
        "system": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli; cch=a5145;You are Claude Code.",
        "messages": [{"role": "user", "content": "Inspect two files."}, {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "I should inspect both files."}, {
                "type": "tool_use",
                "id": "tool_1",
                "name": "read_file",
                "input": {"path": unicode_path},
            }, {
                "type": "tool_use",
                "id": "tool_2",
                "name": "read_file",
                "input": {"path": "src/quoted\\\"name\".cpp"},
            }],
        }, {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "first\nresult"}, {
                "type": "tool_result",
                "tool_use_id": "tool_2",
                "content": unicode_path,
            }],
        }, {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "I reviewed the tool results."}, {
                "type": "text",
                "text": "The files are valid.",
            }],
        }, {
            "role": "user",
            "content": "Continue.",
        }],
        "tools": [{
            "name": "read_file",
            "description": "Read a repository file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }],
        "max_tokens": 4,
        "temperature": 0,
    }, headers={"anthropic-version": "2023-06-01"})
    require(trace["response"]["status"] == 200, "Anthropic Messages request failed")
    body = trace["response"]["body"]
    require(body.get("type") == "message", "Unexpected Anthropic response type")
    require(body.get("role") == "assistant", "Unexpected Anthropic response role")
    require(isinstance(body.get("content"), list), "Anthropic response content is not an array")
    return [trace]


def run_fixture(name, requirements, callback):
    started = time.time()
    try:
        traces = callback()
        return {
            "name": name,
            "requirements": requirements,
            "success": True,
            "elapsed_seconds": time.time() - started,
            "traces": traces,
        }
    except Exception as exc:
        return {
            "name": name,
            "requirements": requirements,
            "success": False,
            "elapsed_seconds": time.time() - started,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traces": [],
        }


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8 protocol fixture runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3.8-27B-Q8_0.gguf")
    parser.add_argument("--output-json", default="qwen38-protocol-fixtures.json")
    args = parser.parse_args()

    fixture_defs = [
        ("openai_object_arguments", ["object arguments", "parallel calls", "multi-step history", "Unicode", "JSON escapes", "preserved thinking"],
            lambda: fixture_chat_history(args.base_url, args.model, False, True)),
        ("openai_string_arguments", ["JSON string arguments", "preserved thinking off"],
            lambda: fixture_chat_history(args.base_url, args.model, True, False)),
        ("openai_malformed_arguments", ["malformed arguments"],
            lambda: fixture_malformed_arguments(args.base_url, args.model)),
        ("openai_sse_bytewise", ["SSE arbitrary byte boundaries"],
            lambda: fixture_sse_bytewise(args.base_url, args.model)),
        ("responses_previous_response_id", ["Responses continuation", "instruction replacement", "unknown id"],
            lambda: fixture_responses_continuation(args.base_url, args.model)),
        ("anthropic_claude_code", ["Claude Code Messages", "thinking", "parallel tool use", "tool results"],
            lambda: fixture_anthropic_claude_code(args.base_url, args.model)),
    ]
    results = [run_fixture(name, requirements, callback) for name, requirements, callback in fixture_defs]
    payload = {
        "schema_version": 1,
        "created_unix": int(time.time()),
        "base_url": args.base_url,
        "model": args.model,
        "success": all(result["success"] for result in results),
        "fixtures": results,
    }
    pathlib.Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not payload["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
