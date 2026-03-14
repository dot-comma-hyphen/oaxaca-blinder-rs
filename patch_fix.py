with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

import re

search_str = """        "notifications/initialized" => {
            if args.api_key.is_none() {
            tracing::warn!("MCP_API_KEY is not set! Server is running without authentication.");
        }
        info!("Client confirmed initialization.");
            return None;
        }"""

replace_str = """        "notifications/initialized" => {
            info!("Client confirmed initialization.");
            return None;
        }"""

content = content.replace(search_str, replace_str)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
