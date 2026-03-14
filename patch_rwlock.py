with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

content = content.replace('.sessions.write().unwrap()', '.sessions.write().unwrap_or_else(|e| e.into_inner())')
content = content.replace('.sessions.read().unwrap()', '.sessions.read().unwrap_or_else(|e| e.into_inner())')

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
