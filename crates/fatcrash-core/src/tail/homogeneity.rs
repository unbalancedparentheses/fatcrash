//! Tail-index homogeneity test (Yang, 2026).
//!
//! If the tail index does not depend on a covariate X, normalized log
//! exceedances `Z = log(Y / w)` above a high threshold w are asymptotically
//! Exp(alpha), independent of X, even when tail frequency varies with X. The
//! test measures dependence between Z and X with the squared distance
//! covariance (Székely, Rizzo & Bakirov, 2007) and calibrates it by permuting
//! Z across the retained covariates. It does not estimate any conditional
//! tail index.
//!
//! Two selection schemes:
//!
//! - **Groups** (locally balanced, Section 3.4): each group, such as a
//!   country, regime, or pre/post period, keeps its own top `k` observations
//!   and is normalized by its own threshold. This keeps every group in the
//!   tail sample. Under heterogeneity, a common threshold would otherwise
//!   select mostly from the heaviest-tailed group, and the test would lose
//!   power.
//! - **Continuous covariate** (global, Section 3.3): one global top-k
//!   threshold, with X as a real covariate.
//!
//! Assumes independent observations. With volatility clustering, the
//! permutation null is only approximate, so decluster or subsample first.

use rand::prelude::*;

/// Result of a homogeneity test.
#[derive(Clone, Debug)]
pub struct HomogeneityResult {
    /// Observed squared distance covariance.
    pub statistic: f64,
    /// Permutation p-value with the plus-one correction.
    pub p_value: f64,
    /// Number of retained tail observations.
    pub n_tail: usize,
}

/// Double-centred n x n matrix built from pairwise distances `dist(i, j)`, row-major.
fn centred(n: usize, dist: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = dist(i, j);
        }
    }
    let row: Vec<f64> = (0..n)
        .map(|i| d[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
        .collect();
    let grand = row.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] += grand - row[i] - row[j];
        }
    }
    d
}

fn centred_distances(a: &[f64]) -> Vec<f64> {
    centred(a.len(), |i, j| (a[i] - a[j]).abs())
}

/// `(1/n^2) sum_jl A_jl B_{pi(j) pi(l)}`.
fn dcov2_perm(a: &[f64], b: &[f64], perm: &[usize]) -> f64 {
    let n = perm.len();
    let mut s = 0.0;
    for j in 0..n {
        let bj = perm[j] * n;
        let aj = j * n;
        for l in 0..n {
            s += a[aj + l] * b[bj + perm[l]];
        }
    }
    s / (n * n) as f64
}

/// Permutation test of independence between `z` and covariate `x` using
/// squared distance covariance.
pub fn dcov_permutation_test(z: &[f64], x: &[f64], n_perm: usize, seed: u64) -> HomogeneityResult {
    let n = z.len();
    if n < 4 || x.len() != n {
        return HomogeneityResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_tail: n,
        };
    }
    permutation_test(&centred_distances(z), &centred_distances(x), n_perm, seed)
}

/// Permutation p-value (plus-one correction) for dCov^2 of two centred matrices.
fn permutation_test(a: &[f64], b: &[f64], n_perm: usize, seed: u64) -> HomogeneityResult {
    let n = (a.len() as f64).sqrt() as usize;
    let mut perm: Vec<usize> = (0..n).collect();
    let stat = dcov2_perm(a, b, &perm);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut exceed = 0usize;
    for _ in 0..n_perm {
        perm.shuffle(&mut rng);
        if dcov2_perm(a, b, &perm) >= stat {
            exceed += 1;
        }
    }
    HomogeneityResult {
        statistic: stat,
        p_value: (1 + exceed) as f64 / (1 + n_perm) as f64,
        n_tail: n,
    }
}

fn magnitude(y: f64, use_abs: bool) -> f64 {
    if use_abs {
        y.abs()
    } else {
        y
    }
}

/// Locally balanced test across discrete groups.
///
/// Within each group, keep its `k_per_group` largest positive values and use
/// `log(Y / w_g)` as Z, where `w_g` is that group's (k+1)-th largest value.
/// The covariate is the group label, encoded so that every pair of distinct
/// groups is at distance 1.
pub fn tail_homogeneity_groups(
    values: &[f64],
    groups: &[usize],
    k_per_group: usize,
    n_perm: usize,
    seed: u64,
    use_abs: bool,
) -> Result<HomogeneityResult, String> {
    if values.len() != groups.len() {
        return Err("values and groups must have the same length".into());
    }
    let mut labels: Vec<usize> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return Err("need at least two groups".into());
    }
    let mut z = Vec::new();
    let mut g = Vec::new();
    for &label in &labels {
        let mut ys: Vec<f64> = values
            .iter()
            .zip(groups)
            .filter(|(_, &gr)| gr == label)
            .map(|(&y, _)| magnitude(y, use_abs))
            .filter(|y| y.is_finite() && *y > 0.0)
            .collect();
        ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if ys.len() <= k_per_group {
            return Err(format!(
                "group {label} has fewer than k_per_group + 1 positive values"
            ));
        }
        let w = ys[k_per_group].ln();
        for y in &ys[..k_per_group] {
            z.push(y.ln() - w);
            g.push(label);
        }
    }
    // Distance 1 between distinct groups. Using |x_i - x_j| on the raw labels
    // would impose an ordering on the groups.
    let b = centred(z.len(), |i, j| (g[i] != g[j]) as u8 as f64);
    Ok(permutation_test(&centred_distances(&z), &b, n_perm, seed))
}

/// Global test against a continuous covariate. Keeps the `k` largest
/// values, uses `log(Y / Y_(k+1))` as Z, and tests independence of Z and
/// the matching covariate values.
pub fn tail_homogeneity_covariate(
    values: &[f64],
    covariate: &[f64],
    k: usize,
    n_perm: usize,
    seed: u64,
    use_abs: bool,
) -> Result<HomogeneityResult, String> {
    if values.len() != covariate.len() {
        return Err("values and covariate must have the same length".into());
    }
    let mut idx: Vec<(f64, f64)> = values
        .iter()
        .zip(covariate)
        .map(|(&y, &x)| (magnitude(y, use_abs), x))
        .filter(|(y, x)| y.is_finite() && *y > 0.0 && x.is_finite())
        .collect();
    if idx.len() <= k {
        return Err("need more than k positive values".into());
    }
    idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let w = idx[k].0.ln();
    let z: Vec<f64> = idx[..k].iter().map(|(y, _)| y.ln() - w).collect();
    let x: Vec<f64> = idx[..k].iter().map(|(_, x)| *x).collect();
    Ok(dcov_permutation_test(&z, &x, n_perm, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pareto(rng: &mut StdRng, n: usize, alpha: f64, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|_| scale * rng.gen::<f64>().powf(-1.0 / alpha))
            .collect()
    }

    #[test]
    fn same_alpha_different_scale_not_rejected() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut values = pareto(&mut rng, 3000, 3.0, 1.0);
        values.extend(pareto(&mut rng, 3000, 3.0, 5.0));
        let groups: Vec<usize> = (0..6000).map(|i| (i >= 3000) as usize).collect();
        let r = tail_homogeneity_groups(&values, &groups, 150, 199, 1, false).unwrap();
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn different_alpha_rejected() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut values = pareto(&mut rng, 3000, 2.0, 1.0);
        values.extend(pareto(&mut rng, 3000, 5.0, 1.0));
        let groups: Vec<usize> = (0..6000).map(|i| (i >= 3000) as usize).collect();
        let r = tail_homogeneity_groups(&values, &groups, 150, 199, 1, false).unwrap();
        assert!(r.p_value < 0.01, "p={}", r.p_value);
    }

    #[test]
    fn covariate_driven_alpha_rejected() {
        let mut rng = StdRng::seed_from_u64(8);
        let x: Vec<f64> = (0..6000).map(|_| rng.gen::<f64>()).collect();
        let values: Vec<f64> = x
            .iter()
            .map(|&xi| rng.gen::<f64>().powf(-1.0 / (1.5 + 4.0 * xi)))
            .collect();
        let r = tail_homogeneity_covariate(&values, &x, 300, 199, 1, false).unwrap();
        assert!(r.p_value < 0.01, "p={}", r.p_value);
    }
}
