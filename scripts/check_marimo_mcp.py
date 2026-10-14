"""Check a live marimo code-mode MCP connection without running notebook cells.

Run with the optional collaboration environment. Open the notebook in a browser
first. This checks transport, session selection and scratchpad execution; it is
not a test of the notebook's scientific results.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit


class CheckError(Exception):
    """A diagnostic we can print without transport headers or token values."""


PROBE = """
import json as _json
import marimo._code_mode as _cm
_ctx = _cm.get_context()
print(_json.dumps({
    "answer": 6 * 7,
    "cells": [
        {"id": c.id, "status": str(c.status), "error_count": len(c.errors)}
        for c in _ctx.cells
    ],
}))
"""


def token_headers(args):
    headers = {}
    for path, header, prefix in (
        (args.auth_token_file, "Authorization", "Bearer "),
        (args.server_token_file, "Marimo-Server-Token", ""),
    ):
        if path is not None:
            value = path.read_text().strip()
            if not value or "\n" in value or "\r" in value:
                raise CheckError(f"{header} token file must contain one nonempty line")
            headers[header] = prefix + value
    return headers


def payload(result):
    if result.is_error:
        raise CheckError("MCP tool returned an error; inspect the marimo server log")
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(next(item.text for item in result.content if item.type == "text"))


def choose_session(sessions, args):
    matches = sessions
    if args.notebook is not None:
        target = args.notebook.resolve()
        matches = [s for s in matches if Path(s["path"]).resolve() == target]
    if args.session is not None:
        matches = [s for s in matches if s["session_id"] == args.session]
    if not matches:
        raise CheckError("No matching live notebook. Open it in the browser and use --list.")
    if len(matches) != 1:
        raise CheckError("Multiple live notebooks match. Use --list, then --notebook and/or --session.")
    return matches[0]


async def check(args):
    # Optional dependencies stay out of the base scientific environment.
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    base = urlsplit(args.url)
    if base.scheme not in {"http", "https"} or not base.netloc or base.query or base.fragment:
        raise CheckError("--url must be the base marimo HTTP(S) URL, without a query or fragment")
    if base.username is not None or base.password is not None:
        raise CheckError("Use --auth-token-file instead of credentials in the URL")
    endpoint = args.url.rstrip("/") + "/mcp/server"
    async with httpx2.AsyncClient(
        headers=token_headers(args), timeout=httpx2.Timeout(45, connect=5), trust_env=False
    ) as http:
        async with streamable_http_client(endpoint, http_client=http) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                tools = [tool.name for tool in (await client.list_tools()).tools]
                if not {"list_sessions", "execute_code"} <= set(tools):
                    raise CheckError("Expected code-mode MCP tools. Launch marimo with --mcp code-mode.")
                sessions = payload(await client.call_tool("list_sessions", {}))["sessions"]
                report = {"server": initialized.server_info.name, "tools": tools, "sessions": sessions}
                if args.list:
                    return report
                selected = choose_session(sessions, args)
                execution = payload(await client.call_tool("execute_code", {
                    "session_id": selected["session_id"], "code": PROBE,
                }))
                if not execution.get("success"):
                    raise CheckError("The MCP scratchpad probe failed; inspect the marimo server log")
                stdout = execution.get("stdout", [])
                probe = json.loads(stdout if isinstance(stdout, str) else "".join(stdout))
                if probe.get("answer") != 42 or not probe.get("cells"):
                    raise CheckError("Unexpected scratchpad result or empty notebook")
                report.update(passed=True, selected_session=selected["session_id"], notebook_state=probe)
                return report


def error_messages(error):
    """Unwrap transport task groups without printing token headers."""
    if isinstance(error, BaseExceptionGroup):
        return [message for child in error.exceptions for message in error_messages(child)]
    if isinstance(error, CheckError):
        return [str(error)]
    # HTTP libraries may include request headers in their exception messages.
    return [f"{type(error).__name__}; check the server log, URL, token files and collaboration environment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8772", help="base marimo server URL")
    parser.add_argument("--list", action="store_true", help="list tools and sessions without executing the probe")
    parser.add_argument("--notebook", type=Path, help="exact notebook path, relative to this working directory")
    parser.add_argument("--session", help="explicit session ID from --list")
    parser.add_argument("--auth-token-file", type=Path, help="optional Bearer access token file")
    parser.add_argument("--server-token-file", type=Path, help="optional Marimo-Server-Token header file")
    args = parser.parse_args()
    try:
        report = asyncio.run(check(args))
    except Exception as error:
        print("MCP check failed: " + "; ".join(error_messages(error)), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
