sed -i 's/info!(/if args.api_key.is_none() {\n            tracing::warn!("MCP_API_KEY is not set! Server is running without authentication.");\n        }\n        info!(/' meridian-mcp/src/main.rs
