import os

with open("oaxaca_blinder/src/akm.rs", "r") as f:
    akm_code = f.read()

# Fix collect() error
akm_code = akm_code.replace(
    ".map(|id| worker_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Worker ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?\n        .collect();",
    ".map(|id| worker_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Worker ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?;"
)
akm_code = akm_code.replace(
    ".map(|id| firm_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Firm ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?\n        .collect();",
    ".map(|id| firm_map.get(id).ok_or_else(|| AkmError::NotEnoughData(\"Firm ID not in map\".to_string())).map(|v| v.clone())).collect::<Result<Vec<_>, _>>()?;"
)

with open("oaxaca_blinder/src/akm.rs", "w") as f:
    f.write(akm_code)
