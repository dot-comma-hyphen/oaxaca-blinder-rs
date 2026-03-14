with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

import re

# replace decompose_inner
content = re.sub(
    r'let res = decompose_inner\(mcp_params\.into\(\)\)\.map_err\(\|e\| anyhow!\(e\)\)\?;',
    'let req = mcp_params.into(); let res = tokio::task::spawn_blocking(move || decompose_inner(req)).await.map_err(|e| anyhow!(e))?.map_err(|e| anyhow!(e))?;',
    content
)

# replace optimize_inner
content = re.sub(
    r'let res = optimize_inner\(req\)\.map_err\(\|e\| anyhow!\(e\)\)\?;',
    'let res = tokio::task::spawn_blocking(move || optimize_inner(req)).await.map_err(|e| anyhow!(e))?.map_err(|e| anyhow!(e))?;',
    content
)

# replace verify_inner
content = re.sub(
    r'let res = verify_inner\(p\.into\(\)\)\.map_err\(\|e\| anyhow!\(e\)\)\?;',
    'let req = p.into(); let res = tokio::task::spawn_blocking(move || verify_inner(req)).await.map_err(|e| anyhow!(e))?.map_err(|e| anyhow!(e))?;',
    content
)

# replace check_defensibility_inner
content = re.sub(
    r'let res = check_defensibility_inner\(p\.into\(\)\)\.map_err\(\|e\| anyhow!\(e\)\)\?;',
    'let req = p.into(); let res = tokio::task::spawn_blocking(move || check_defensibility_inner(req)).await.map_err(|e| anyhow!(e))?.map_err(|e| anyhow!(e))?;',
    content
)

# replace calculate_efficient_frontier_inner
content = re.sub(
    r'let res = calculate_efficient_frontier_inner\(req\)\.map_err\(\|e\| anyhow!\(e\)\)\?;',
    'let res = tokio::task::spawn_blocking(move || calculate_efficient_frontier_inner(req)).await.map_err(|e| anyhow!(e))?.map_err(|e| anyhow!(e))?;',
    content
)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
