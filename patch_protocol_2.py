with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

import re

search_str = """    match result {
        Ok(v) => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(v),
            error: None,
            id: req.id,
        }),
        Err(e) => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(json!({
                "code": -32603,
                "message": e.to_string()
            })),
            id: req.id,
        }),
    }"""

replace_str = """    match result {
        Ok(v) => Some(JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(v),
            error: None,
            id: req.id,
        }),
        Err(e) => {
            let code = if e.to_string().starts_with("Method not found:") {
                -32601
            } else {
                -32603
            };
            Some(JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(json!({
                    "code": code,
                    "message": e.to_string()
                })),
                id: req.id,
            })
        }
    }"""

content = content.replace(search_str, replace_str)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
