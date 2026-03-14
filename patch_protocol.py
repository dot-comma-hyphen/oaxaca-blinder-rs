with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

import re

# Fix method not found error map
search_str = """        "tools/call" => handle_tool_call(req.params).await,
        _ => Err(anyhow!("Method not found: {}", req.method)),
    };"""

replace_str = """        "tools/call" => handle_tool_call(req.params).await,
        "ping" => Ok(json!({})),
        _ => {
            return Some(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(json!({
                    "code": -32601,
                    "message": format!("Method not found: {}", req.method)
                })),
                id: req.id,
            });
        }
    };"""

content = content.replace(search_str, replace_str)


# Fix stdio parse errors
search_str = """        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to parse request: {}", e);
                continue;
            }
        };"""

replace_str = """        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                error!("Failed to parse request: {}", e);
                let response_json = serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(json!({
                        "code": -32700,
                        "message": "Parse error"
                    })),
                    id: None,
                })?;
                println!("{}", response_json);
                std::io::stdout().flush()?;
                continue;
            }
        };"""

content = content.replace(search_str, replace_str)


# Fix SSE GET session ID mapping
search_str = """async fn handle_sse_get(
    State(_state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if headers.get("mcp-session-id").is_some() {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    let host = headers
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let scheme = "http";
    let endpoint_url = format!("{}://{}/sse", scheme, host);

    let endpoint_event = Event::default().event("endpoint").data(format!(
        "{}?sessionId={}",
        endpoint_url,
        Uuid::new_v4()
    ));"""

replace_str = """async fn handle_sse_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if headers.get("mcp-session-id").is_some() {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    let host = headers
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost");
    let scheme = "http";
    let endpoint_url = format!("{}://{}/sse", scheme, host);

    let new_id = Uuid::new_v4().to_string();
    {
        let mut sessions = state.sessions.write().unwrap_or_else(|e| e.into_inner());
        sessions.insert(
            new_id.clone(),
            Session {
                id: new_id.clone(),
                created_at: Instant::now(),
            },
        );
    }

    let endpoint_event = Event::default().event("endpoint").data(format!(
        "{}?sessionId={}",
        endpoint_url,
        new_id
    ));"""

content = content.replace(search_str, replace_str)


with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
