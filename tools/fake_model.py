"""A stand-in model server, for checking the rig and the harness together without a GPU.

    python tools/fake_model.py 18080 tasks/example/slug-0001/solution/solution.py

It speaks just enough of the OpenAI streaming Chat Completions wire: the first request is
answered with one call to the harness's `write` tool, putting the given file's content into
`solution.py`; the request that follows the tool result is answered with a closing sentence.
Point a models.toml row at http://127.0.0.1:<port>/v1 and run a task whose answer that file is.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def chunks_for(messages: list[dict], solution: str) -> list[dict]:
    if messages and messages[-1].get("role") == "tool":
        delta = {"role": "assistant", "content": "Wrote solution.py."}
        finish = "stop"
    else:
        arguments = json.dumps({"path": "solution.py", "content": solution})
        delta = {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "write", "arguments": arguments}}]}
        finish = "tool_calls"
    return [{"choices": [{"index": 0, "delta": delta}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]


def main() -> None:
    port, solution = int(sys.argv[1]), Path(sys.argv[2]).read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks_for(body.get("messages", []), solution):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
