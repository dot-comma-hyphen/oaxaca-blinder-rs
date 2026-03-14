sed -i 's/PartnerStatus::Suspended => Ok(config.clone()),/PartnerStatus::Suspended => bail!("Partner access is suspended"),/' engine/src/access.rs
