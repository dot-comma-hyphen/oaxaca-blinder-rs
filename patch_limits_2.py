with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

search_str_1 = """        "verify_adjustments" => {
            let p: McpVerificationParams = serde_json::from_value(arguments.clone())?;"""

replace_str_1 = """        "verify_adjustments" => {
            let mut p: McpVerificationParams = serde_json::from_value(arguments.clone())?;
            if let Some(reps) = p.decomposition_params.bootstrap_reps {
                p.decomposition_params.bootstrap_reps = Some(reps.min(10000));
            }"""
content = content.replace(search_str_1, replace_str_1)

search_str_2 = """        "check_defensibility" => {
            let p: McpVerificationParams = serde_json::from_value(arguments.clone())?;"""

replace_str_2 = """        "check_defensibility" => {
            let mut p: McpVerificationParams = serde_json::from_value(arguments.clone())?;
            if let Some(reps) = p.decomposition_params.bootstrap_reps {
                p.decomposition_params.bootstrap_reps = Some(reps.min(10000));
            }"""
content = content.replace(search_str_2, replace_str_2)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
