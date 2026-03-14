with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

content = content.replace('.layer(cors)', '.layer(cors)\n        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))')

search_str = """    match name {
        "forensic_decomposition" => {
            let mcp_params: McpDecompositionParams = serde_json::from_value(arguments.clone())?;"""

replace_str = """    match name {
        "forensic_decomposition" => {
            let mut mcp_params: McpDecompositionParams = serde_json::from_value(arguments.clone())?;
            if let Some(reps) = mcp_params.bootstrap_reps {
                mcp_params.bootstrap_reps = Some(reps.min(10000));
            }"""

content = content.replace(search_str, replace_str)

search_str = """        "generate_efficient_frontier" => {
            let mcp_params: McpDecompositionParams = serde_json::from_value(arguments.clone())?;"""

replace_str = """        "generate_efficient_frontier" => {
            let mut mcp_params: McpDecompositionParams = serde_json::from_value(arguments.clone())?;
            if let Some(reps) = mcp_params.bootstrap_reps {
                mcp_params.bootstrap_reps = Some(reps.min(10000));
            }"""

content = content.replace(search_str, replace_str)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
