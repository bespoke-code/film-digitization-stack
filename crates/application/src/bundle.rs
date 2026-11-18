//! Global transform refinement over every accepted image pair.
//!
//! Registering each image against a single parent chains the pairwise
//! homographies, so their small inconsistencies accumulate along the chain.
//! Two images that overlap directly can then land tens of pixels apart even
//! though their own pair estimate was accurate. This module solves all image
//! transforms at once against every accepted overlap, which distributes the
//! error instead of stacking it.

use stitch_domain::{Homography, Point};

const MAX_ITERATIONS: usize = 60;
const POINTS_PER_LINK: usize = 220;
const INITIAL_DAMPING: f64 = 1e-3;

/// One accepted overlap: `homography` maps `query` coordinates into `target`
/// coordinates. `weight` reflects how much the pair should influence the fit.
#[derive(Clone, Copy, Debug)]
pub struct PairLink {
    pub query: usize,
    pub target: usize,
    pub homography: Homography,
    pub weight: f64,
}

struct Samples {
    query: usize,
    target: usize,
    points: Vec<Point>,
    expected: Vec<Point>,
    weight: f64,
}

/// Refines `initial` so every linked pair matches as closely as possible.
///
/// `sizes` holds each image's pixel dimensions. Image zero stays the reference.
/// The function returns the input unchanged when the links cannot be sampled.
pub fn refine_transforms(
    initial: &[Homography],
    sizes: &[(f64, f64)],
    links: &[PairLink],
) -> Vec<Homography> {
    let count = initial.len();
    if count < 2 || links.is_empty() || sizes.len() != count {
        return initial.to_vec();
    }
    let samples: Vec<Samples> = links
        .iter()
        .filter_map(|link| sample_link(link, sizes))
        .collect();
    if samples.is_empty() {
        return initial.to_vec();
    }

    let reference = initial[0];
    let mut transforms = initial.to_vec();
    let mut parameters = Vec::with_capacity((count - 1) * 8);
    for transform in &transforms[1..] {
        parameters.extend_from_slice(&from_homography(transform));
    }

    let mut damping = INITIAL_DAMPING;
    let mut base = residuals(&parameters, &reference, &samples, count);
    let mut cost = squared_sum(&base);
    for _ in 0..MAX_ITERATIONS {
        let jacobian = jacobian(&parameters, &reference, &samples, count, &base);
        let columns = parameters.len();
        let mut normal = vec![vec![0.0f64; columns]; columns];
        let mut gradient = vec![0.0f64; columns];
        for (row, value) in base.iter().enumerate() {
            let entries = &jacobian[row];
            for (a, entry_a) in entries.iter().enumerate() {
                if *entry_a == 0.0 {
                    continue;
                }
                gradient[a] += entry_a * value;
                for (b, entry_b) in entries.iter().enumerate().skip(a) {
                    normal[a][b] += entry_a * entry_b;
                }
            }
        }
        for a in 0..columns {
            for b in 0..a {
                normal[a][b] = normal[b][a];
            }
        }

        let diagonal: Vec<f64> = (0..columns).map(|index| normal[index][index]).collect();
        let mut accepted = false;
        for _ in 0..8 {
            let mut system = normal.clone();
            let mut rhs: Vec<f64> = gradient.iter().map(|value| -value).collect();
            for index in 0..columns {
                system[index][index] += damping * diagonal[index].max(1e-12);
            }
            let step = solve(system, &mut rhs);
            let mut candidate = parameters.clone();
            for (value, delta) in candidate.iter_mut().zip(step.iter()) {
                *value += delta;
            }
            let candidate_residuals = residuals(&candidate, &reference, &samples, count);
            let candidate_cost = squared_sum(&candidate_residuals);
            if candidate_cost < cost {
                parameters = candidate;
                base = candidate_residuals;
                cost = candidate_cost;
                damping = (damping / 3.0).max(1e-9);
                accepted = true;
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            break;
        }
    }

    for (index, transform) in transforms.iter_mut().enumerate().skip(1) {
        let start = (index - 1) * 8;
        *transform = to_homography(&parameters[start..start + 8]);
    }
    transforms
}

fn sample_link(link: &PairLink, sizes: &[(f64, f64)]) -> Option<Samples> {
    let count = sizes.len();
    if link.query >= count || link.target >= count || link.query == link.target {
        return None;
    }
    let (query_width, query_height) = sizes[link.query];
    let (target_width, target_height) = sizes[link.target];
    let step = (query_width.max(query_height) / 48.0).max(1.0);

    let mut points = Vec::new();
    let mut expected = Vec::new();
    let mut y = 0.0;
    while y < query_height {
        let mut x = 0.0;
        while x < query_width {
            let point = Point::new(x, y);
            if let Some(mapped) = link.homography.transform(point) {
                if mapped.x >= 1.0
                    && mapped.y >= 1.0
                    && mapped.x < target_width - 1.0
                    && mapped.y < target_height - 1.0
                {
                    points.push(point);
                    expected.push(mapped);
                }
            }
            x += step;
        }
        y += step;
    }
    if points.len() < 12 {
        return None;
    }
    let stride = (points.len() / POINTS_PER_LINK).max(1);
    let points: Vec<Point> = points.iter().step_by(stride).copied().collect();
    let expected: Vec<Point> = expected.iter().step_by(stride).copied().collect();
    Some(Samples {
        query: link.query,
        target: link.target,
        points,
        expected,
        weight: link.weight.max(1e-6).sqrt(),
    })
}

fn residuals(
    parameters: &[f64],
    reference: &Homography,
    samples: &[Samples],
    count: usize,
) -> Vec<f64> {
    let transforms = build_transforms(parameters, reference, count);
    let inverses: Vec<Option<Homography>> = transforms
        .iter()
        .map(|transform| transform.inverse())
        .collect();
    let mut residuals = Vec::with_capacity(samples.iter().map(|s| s.points.len() * 2).sum());
    for sample in samples {
        let query = &transforms[sample.query];
        let Some(target_inverse) = inverses[sample.target].as_ref() else {
            residuals.resize(residuals.len() + sample.points.len() * 2, 0.0);
            continue;
        };
        for (point, expected) in sample.points.iter().zip(sample.expected.iter()) {
            let predicted = query
                .transform(*point)
                .and_then(|mapped| target_inverse.transform(mapped));
            match predicted {
                Some(predicted) => {
                    residuals.push((predicted.x - expected.x) * sample.weight);
                    residuals.push((predicted.y - expected.y) * sample.weight);
                }
                None => {
                    residuals.push(0.0);
                    residuals.push(0.0);
                }
            }
        }
    }
    residuals
}

fn jacobian(
    parameters: &[f64],
    reference: &Homography,
    samples: &[Samples],
    count: usize,
    base: &[f64],
) -> Vec<Vec<f64>> {
    let columns = parameters.len();
    let mut matrix = vec![vec![0.0f64; columns]; base.len()];
    for column in 0..columns {
        let epsilon = 1e-7 * (1.0 + parameters[column].abs());
        let mut perturbed = parameters.to_vec();
        perturbed[column] += epsilon;
        let shifted = residuals(&perturbed, reference, samples, count);
        for (row, value) in shifted.iter().enumerate() {
            matrix[row][column] = (value - base[row]) / epsilon;
        }
    }
    matrix
}

fn squared_sum(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

fn build_transforms(parameters: &[f64], reference: &Homography, count: usize) -> Vec<Homography> {
    let mut transforms = Vec::with_capacity(count);
    transforms.push(*reference);
    for index in 1..count {
        let start = (index - 1) * 8;
        transforms.push(to_homography(&parameters[start..start + 8]));
    }
    transforms
}

fn from_homography(homography: &Homography) -> [f64; 8] {
    let m = &homography.m;
    let scale = if m[8].abs() < 1e-12 { 1.0 } else { m[8] };
    [
        m[0] / scale,
        m[1] / scale,
        m[2] / scale,
        m[3] / scale,
        m[4] / scale,
        m[5] / scale,
        m[6] / scale,
        m[7] / scale,
    ]
}

fn to_homography(parameters: &[f64]) -> Homography {
    Homography::from_rows([
        [parameters[0], parameters[1], parameters[2]],
        [parameters[3], parameters[4], parameters[5]],
        [parameters[6], parameters[7], 1.0],
    ])
}

fn solve(mut matrix: Vec<Vec<f64>>, rhs: &mut [f64]) -> Vec<f64> {
    let size = rhs.len();
    for column in 0..size {
        let mut pivot = column;
        for row in column + 1..size {
            if matrix[row][column].abs() > matrix[pivot][column].abs() {
                pivot = row;
            }
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        if divisor.abs() < 1e-15 {
            continue;
        }
        for entry in &mut matrix[column][column..] {
            *entry /= divisor;
        }
        rhs[column] /= divisor;
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0.0 {
                continue;
            }
            for index in column..size {
                matrix[row][index] -= factor * matrix[column][index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(count: usize) -> Vec<(f64, f64)> {
        vec![(400.0, 300.0); count]
    }

    #[test]
    fn refinement_removes_chain_drift() {
        // Three images on a line. The direct pair between the outer images is
        // exact, but the initial chain overshoots the middle transform.
        let first = Homography::identity();
        let second = Homography::translation(100.0, 0.0);
        let truth = Homography::translation(200.0, 0.0);
        let drifted = Homography::translation(230.0, 0.0);
        let links = [
            PairLink {
                query: 1,
                target: 0,
                homography: second,
                weight: 1.0,
            },
            PairLink {
                query: 2,
                target: 1,
                homography: Homography::translation(100.0, 0.0),
                weight: 1.0,
            },
            PairLink {
                query: 2,
                target: 0,
                homography: truth,
                weight: 1.0,
            },
        ];
        let refined = refine_transforms(&[first, second, drifted], &sizes(3), &links);
        let offset = refined[2].transform(Point::new(0.0, 0.0)).unwrap();
        assert!((offset.x - 200.0).abs() < 1.0, "x was {}", offset.x);
        assert!(offset.y.abs() < 1.0, "y was {}", offset.y);
    }

    #[test]
    fn refinement_keeps_a_consistent_chain() {
        let first = Homography::identity();
        let second = Homography::translation(100.0, 0.0);
        let third = Homography::translation(200.0, 0.0);
        let links = [
            PairLink {
                query: 1,
                target: 0,
                homography: second,
                weight: 1.0,
            },
            PairLink {
                query: 2,
                target: 1,
                homography: Homography::translation(100.0, 0.0),
                weight: 1.0,
            },
        ];
        let refined = refine_transforms(&[first, second, third], &sizes(3), &links);
        let offset = refined[2].transform(Point::new(0.0, 0.0)).unwrap();
        assert!((offset.x - 200.0).abs() < 1.0, "x was {}", offset.x);
    }

    #[test]
    fn single_image_is_unchanged() {
        let initial = [Homography::translation(5.0, 7.0)];
        let refined = refine_transforms(&initial, &sizes(1), &[]);
        assert_eq!(refined, initial);
    }
}
