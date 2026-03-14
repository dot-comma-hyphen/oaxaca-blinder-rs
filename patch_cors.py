with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

content = content.replace('.allow_origin(tower_http::cors::Any)', '.allow_origin("http://127.0.0.1".parse::<axum::http::HeaderValue>().unwrap())')

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
