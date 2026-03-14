with open("meridian-mcp/src/main.rs", "r") as f:
    content = f.read()

import re

search_str = """struct AppState {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    api_key: Option<String>,
}"""

replace_str = """struct AppState {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    api_key: Option<String>,
    rate_limiter: Arc<governor::DefaultDirectRateLimiter>,
}"""

content = content.replace(search_str, replace_str)

search_str = """async fn run_sse_server(port: u16, api_key: Option<String>) -> Result<()> {
    let state = AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        api_key,
    };"""

replace_str = """async fn run_sse_server(port: u16, api_key: Option<String>) -> Result<()> {
    let quota = Quota::per_minute(NonZeroU32::new(60).unwrap());
    let rate_limiter = Arc::new(RateLimiter::direct(quota));

    let state = AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        api_key,
        rate_limiter,
    };"""

content = content.replace(search_str, replace_str)

search_str = """async fn handle_sse_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let is_initialize = req.method == "initialize";"""

replace_str = """async fn handle_sse_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    if state.rate_limiter.check().is_err() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded",
        )
            .into_response();
    }

    let is_initialize = req.method == "initialize";"""

content = content.replace(search_str, replace_str)

with open("meridian-mcp/src/main.rs", "w") as f:
    f.write(content)
