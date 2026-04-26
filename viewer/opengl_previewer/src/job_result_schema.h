#pragma once

#include <filesystem>
#include <string>

namespace faithc::viewer {

struct FaithCJobResult {
    bool success = false;
    std::string message;
    std::filesystem::path output_mesh;
    std::filesystem::path status_json;
    double total_seconds = 0.0;
    int input_faces = 0;
    int output_faces = 0;
    int active_voxels = 0;
    bool uv_projected = false;
    std::string uv_mode_used;
    std::string uv_seam_strategy_used;
    std::string uv_method;
    double uv_flip_ratio = -1.0;
    double uv_bad_tri_ratio = -1.0;
    double uv_stretch_p95 = -1.0;
    double uv_stretch_p99 = -1.0;
    double uv_correspondence_primary_ratio = -1.0;
    double uv_color_reproj_l1 = -1.0;
    double uv_color_reproj_l2 = -1.0;
    std::string uv_solver_backend_requested;
    std::string uv_solver_backend_used;
    std::string uv_solver_linear_backend_requested;
    std::string uv_solver_linear_backend_used;
    std::string uv_solver_stage;
    std::string uv_solver_backend_u;
    std::string uv_solver_backend_v;
    std::string uv_solver_fallback_reason;
    int uv_solver_iters_u = -1;
    int uv_solver_iters_v = -1;
    double uv_solver_residual_u = -1.0;
    double uv_solver_residual_v = -1.0;
    bool uv_solver_converged_u = false;
    bool uv_solver_converged_v = false;
    std::string uv_seam_strategy_requested;
    std::string uv_halfedge_island_error;
    std::string uv_island_validation_mode;
    bool uv_island_validation_ok = false;
    std::string uv_island_validation_error;
    std::string uv_semantic_transfer_sidecar_path;
    std::string uv_closure_validation_sidecar_path;
    bool uv_semantic_component_merge_enabled = false;
    int uv_semantic_component_merge_min_faces = -1;
    int uv_semantic_component_merge_merged_components = -1;
    int uv_semantic_component_merge_merged_faces = -1;
    int uv_semantic_pre_cleanup_fragmented_label_count = -1;
    int uv_semantic_pre_cleanup_severe_label_count = -1;
    int uv_semantic_final_fragmented_label_count = -1;
    int uv_semantic_final_severe_label_count = -1;
    int uv_high_island_count = -1;
    int uv_high_seam_edges = -1;
    int uv_high_boundary_edges = -1;
    int uv_high_nonmanifold_edges = -1;
    int uv_island_conflict_faces = -1;
    int uv_island_conflict_faces_excluded = -1;
    int uv_island_unknown_faces = -1;
    int uv_low_cut_edges = -1;
    int uv_low_split_vertices = -1;
    int uv_low_split_faces = -1;
    int uv_cross_seam_faces = -1;
    int uv_cross_seam_faces_excluded = -1;
    double uv_cross_seam_face_ratio = -1.0;
    double uv_seam_uv_span_threshold = -1.0;
    bool uv_island_guard_requested = false;
    bool uv_island_guard_enabled = false;
    bool uv_island_guard_allow_unknown = false;
    std::string uv_island_guard_mode_requested;
    std::string uv_island_guard_mode_used;
    std::string uv_island_guard_fallback_policy;
    std::string uv_island_guard_error;
    int uv_island_guard_constrained_points = -1;
    int uv_island_guard_reject_count = -1;
    double uv_island_guard_confidence_min = -1.0;
    double uv_island_guard_constrained_ratio = -1.0;
    double uv_island_guard_reject_ratio = -1.0;
    double uv_island_guard_fallback_success_ratio = -1.0;
    double uv_island_guard_invalid_after_guard_ratio = -1.0;
    int uv_m2_irls_iters_used = -1;
    double uv_m2_face_jacobian_cov_p95 = -1.0;
    double uv_m2_system_cond_proxy = -1.0;
    std::string uv_m2_laplacian_mode;
    std::string uv_m2_face_sample_counts_path;
    int uv_m2_face_sample_faces = -1;
    int uv_m2_face_sample_nonzero = -1;
    int uv_m2_face_sample_max = -1;
    std::string uv_m4_refine_status;
    std::string uv_m4_optimizer;
    std::string uv_m4_stop_reason;
    int uv_m4_nonlinear_iters = -1;
    int uv_m4_line_search_fail_count = -1;
    int uv_m4_patch_refine_rounds = -1;
    int uv_m4_barrier_violations = -1;
    int uv_m4_barrier_violation_count_tol = -1;
    int uv_m4_barrier_homotopy_warmup_iters = -1;
    int uv_m4_pre_repair_iters_used = -1;
    int uv_m4_pre_repair_initial_violations = -1;
    int uv_m4_pre_repair_final_violations = -1;
    bool uv_m4_barrier_homotopy_enabled = false;
    bool uv_m4_pre_repair_enabled = false;
    double uv_m4_energy_init = -1.0;
    double uv_m4_energy_final = -1.0;
    double uv_m4_det_min = -1.0;
    double uv_m4_det_p01 = -1.0;
    double uv_m4_barrier_violation_ratio = -1.0;
    double uv_m4_barrier_violation_ratio_tol = -1.0;
};

FaithCJobResult ParseJobResultFromStatusJson(const std::filesystem::path &status_json,
                                             const std::filesystem::path &default_output_mesh, int process_exit_code);

}  // namespace faithc::viewer
