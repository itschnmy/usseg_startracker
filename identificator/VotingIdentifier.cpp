#include "VotingIdentifier.h"
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <algorithm>

VotingIdentifier::VotingIdentifier(StarCatalog& scat, PairCatalog& pcat, double angleTol)
    : scat(scat), pcat(pcat), angleTol(angleTol) 
{}

// Helper structure for candidate sorting
struct CandidateScore {
    int cat_idx;
    int score;
    bool operator>(const CandidateScore& other) const {
        return score > other.score;
    }
};

// Recursive backtracking helper for global consistency
void globalConsistencySearch(
    int star_idx,
    int N_detected,
    const std::vector<std::vector<int>>& candidates,
    const std::vector<CatalogStar>& catalog_stars,
    const std::vector<std::vector<double>>& obs_pair_angles,
    std::vector<int>& current_assignment,
    std::vector<bool>& catalog_used,
    double current_cost,
    double final_verify_tolerance,
    bool strict,
    double& best_cost,
    std::vector<int>& best_assignment)
{
    if (star_idx == N_detected) {
        if (current_cost < best_cost) {
            best_cost = current_cost;
            best_assignment = current_assignment;
        }
        return;
    }

    for (int cat_idx : candidates[star_idx]) {
        if (catalog_used[cat_idx]) continue;

        double incremental_cost = 0.0;
        bool valid = true;

        for (int prev_idx = 0; prev_idx < star_idx; ++prev_idx) {
            double obs_angle = obs_pair_angles[prev_idx][star_idx];
            // Compute catalog angle
            double dot = catalog_stars[current_assignment[prev_idx]].u.dot(catalog_stars[cat_idx].u);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            double cat_angle = std::acos(dot);

            double err = std::abs(obs_angle - cat_angle);
            if (strict && err > final_verify_tolerance) {
                valid = false;
                break;
            }
            incremental_cost += err;
        }

        if (valid && (current_cost + incremental_cost < best_cost)) {
            current_assignment[star_idx] = cat_idx;
            catalog_used[cat_idx] = true;
            globalConsistencySearch(
                star_idx + 1, N_detected, candidates, catalog_stars, obs_pair_angles,
                current_assignment, catalog_used, current_cost + incremental_cost,
                final_verify_tolerance, strict, best_cost, best_assignment
            );
            catalog_used[cat_idx] = false;
        }
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> VotingIdentifier::identify(std::vector<DetectedStar> detected) {
    if (detected.size() < 2) {
        std::cerr << "Warning: identify requires at least 2 detected stars." << std::endl;
        return {};
    }

    std::vector<CatalogStar> catalog_stars = scat.getStars();
    if (catalog_stars.empty()) {
        std::cerr << "Error: Catalog is empty." << std::endl;
        return {};
    }

    int N_detected = detected.size();
    int N_catalog = catalog_stars.size();

    // 1. Calculate observed angles between all pairs of detected stars
    std::vector<std::vector<double>> obs_pair_angles(N_detected, std::vector<double>(N_detected, 0.0));
    for (int i = 0; i < N_detected; ++i) {
        for (int j = i + 1; j < N_detected; ++j) {
            double dot = detected[i].uBody.dot(detected[j].uBody);
            if (dot > 1.0) dot = 1.0;
            if (dot < -1.0) dot = -1.0;
            double angle = std::acos(dot);
            obs_pair_angles[i][j] = angle;
            obs_pair_angles[j][i] = angle;
        }
    }

    // 2. Score hypotheses (support matrix)
    std::vector<std::vector<int>> support(N_detected, std::vector<int>(N_catalog, 0));
    for (int i = 0; i < N_detected; ++i) {
        for (int j = i + 1; j < N_detected; ++j) {
            double obs_angle = obs_pair_angles[i][j];
            double min_query = obs_angle - angleTol;
            double max_query = obs_angle + angleTol;

            std::vector<KVectorPair> returned_pairs = pcat.queryPairs(min_query, max_query);

            for (const auto& pair : returned_pairs) {
                int a = pair.index1;
                int b = pair.index2;

                if (a >= 0 && a < N_catalog && b >= 0 && b < N_catalog) {
                    support[i][a]++;
                    support[j][b]++;
                    support[i][b]++;
                    support[j][a]++;
                }
            }
        }
    }

    // 3. Find top candidates for each detected star
    int top_k_per_star = 6;
    std::vector<std::vector<int>> candidates(N_detected);
    for (int i = 0; i < N_detected; ++i) {
        std::vector<CandidateScore> scored;
        for (int c = 0; c < N_catalog; ++c) {
            if (support[i][c] > 0) {
                scored.push_back({c, support[i][c]});
            }
        }
        
        // Sort descending by score
        std::sort(scored.begin(), scored.end(), [](const CandidateScore& x, const CandidateScore& y) {
            return x.score > y.score;
        });

        int limit = std::min(top_k_per_star, static_cast<int>(scored.size()));
        for (int k = 0; k < limit; ++k) {
            candidates[i].push_back(scored[k].cat_idx);
        }

        // Fallback: If no candidate had votes, just add the top global one or default to index 0
        if (candidates[i].empty()) {
            candidates[i].push_back(0);
        }
    }

    // 4. Global consistency search (backtracking)
    double final_verify_tolerance = 0.03; // rad
    std::vector<int> best_assignment;
    double best_cost = 1e9;

    std::vector<int> current_assignment(N_detected, -1);
    std::vector<bool> catalog_used(N_catalog, false);

    // Try strict search first
    globalConsistencySearch(
        0, N_detected, candidates, catalog_stars, obs_pair_angles,
        current_assignment, catalog_used, 0.0, final_verify_tolerance, true,
        best_cost, best_assignment
    );

    // Fallback: if strict search found nothing, do a non-strict search
    if (best_assignment.empty()) {
        best_cost = 1e9;
        std::fill(catalog_used.begin(), catalog_used.end(), false);
        globalConsistencySearch(
            0, N_detected, candidates, catalog_stars, obs_pair_angles,
            current_assignment, catalog_used, 0.0, final_verify_tolerance, false,
            best_cost, best_assignment
        );
    }

    // 5. Construct matched pairs output
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched_pairs;
    if (!best_assignment.empty()) {
        for (int i = 0; i < N_detected; ++i) {
            int cat_idx = best_assignment[i];
            matched_pairs.push_back(std::make_pair(detected[i].uBody, catalog_stars[cat_idx].u));
        }
    }

    return matched_pairs;
}