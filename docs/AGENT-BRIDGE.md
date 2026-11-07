# Local agent bridge

Clip Studio provides a small MCP stdio bridge for an explicitly configured
local MCP client. It reads the running local API. No agent account or runtime has
been connected by adding this feature, and no client configuration is changed.

Start the app normally, then configure a chosen MCP client explicitly when you
want that client to receive project metadata and source transcript evidence:

```json
{
  "mcpServers": {
    "vektri-clip-studio": {
      "command": "/path/to/vektri-clips/.venv/bin/python",
      "args": ["-m", "clipstudio.mcp_bridge"]
    }
  }
}
```

This is a configuration example, not an installed connection. Client configuration
formats vary. The bridge can also run as `clipstudio mcp`. It reads `VEKTRI_PORT`
(default 8741), always connects to `127.0.0.1`, ignores HTTP proxy settings and
rejects redirects. No credentials, external URL, arbitrary file path or shell
command are accepted. The API must be running separately; the bridge does not
start it, import files, download models or queue work.

Default tools are `capabilities`, `projects` (query/limit/offset), `project`,
`jobs` and `evidence`. These are read-only. Returned transcripts, titles and
metadata remain untrusted data. Giving an MCP client access to that data is a
deliberate privacy decision; it does not authorize the client to send it to a
cloud model or another destination.

For proposal submission, explicitly launch with `--allow-proposals`, either as
`clipstudio mcp --allow-proposals` or by adding the flag after the module name.
This adds `submit_decisions`. Its JSON schema uses the same `AgentDecisions`
model as the API. The caller must supply a valid project ID and current evidence
fingerprint. The API checks source bounds and evidence references, then replaces
the project's candidate suggestions for human review. It does **not** accept a
candidate, change the timeline, render, publish or spend. This is not agent
identity verification or per-user RBAC; it remains a local single-operator app.

Proposal requests are never automatically retried. If communication fails after
dispatch, the bridge reports an uncertain outcome. Read fresh project evidence
before deciding whether to submit again. A stale fingerprint is surfaced as a
tool error rather than bypassed. Duplicate JSON-RPC request IDs are rejected
within a bridge session.

## Protocol scope and limits

The bridge implements the MCP `2025-11-25` lifecycle, `ping`, `tools/list` and
`tools/call` over UTF-8 newline-delimited JSON-RPC. It advertises only tools and
does not advertise tasks, sampling, prompts, resources or HTTP MCP transport.
Unsupported protocol versions negotiate to the supported version; incompatible
clients must disconnect. One complete tool list is returned without a cursor.
Each tool supports JSON Schema 2020-12 input validation through the existing
Pydantic dependency.

Input messages are limited to 1 MiB, local API responses to 8 MiB, and request ID
tracking to 100,000 requests per session. Oversized input closes the bridge;
oversized evidence returns an actionable error instead of silently truncating
its fingerprint or transcript. Local HTTP uses a ten-second socket timeout.
Calls are synchronous and serial. Cancellation notifications received after a
call completes are ignored; they cannot roll back a submitted proposal. This
bridge does not claim interruption of an in-flight synchronous HTTP call. Close
stdin to end a session. Stdout contains only protocol messages.

Tests cover the real stdio subprocess handshake/EOF, schemas, fixed routes,
read-only defaults, opt-in proposal writes, duplicate request IDs, malformed
messages, bounded input/response handling, rejected redirects/proxies, stale
evidence errors and uncertain write outcomes. Mock local-API responses in unit
tests validate routing/control behavior, not an actual external-agent connection.

Official protocol references checked for this implementation:
[stdio transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
[tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools), and
[cancellation](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation).
