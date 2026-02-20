#[derive(Deserialize, Debug)]
pub struct ProposedAdjustment {
    pub index: usize,
    pub value: f64,
}

#[derive(Deserialize, Debug)]
pub struct VerificationRequest {
    #[serde(flatten)]
    pub decomposition_params: DecompositionRequest,
    pub adjustments: Vec<ProposedAdjustment>,
}
