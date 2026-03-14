with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

content = content.replace('let addr = format!("0.0.0.0:{}", port);', 'let addr = format!("127.0.0.1:{}", port);')

search_str = """    if is_sse {
        let port = args.port.unwrap_or(8084);
        info!("""

replace_str = """    if is_sse {
        let port = args.port.unwrap_or(8084);
        if args.api_key.is_none() {
            tracing::warn!("MCP_API_KEY is not set! Server is running without authentication.");
        }
        info!("""

content = content.replace(search_str, replace_str)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
