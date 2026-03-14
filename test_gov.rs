use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;

fn main() {
    let quota = Quota::per_minute(NonZeroU32::new(60).unwrap());
    let limiter: governor::DefaultDirectRateLimiter = RateLimiter::direct(quota);
}
