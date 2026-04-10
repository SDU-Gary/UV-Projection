#include "job_result_schema.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace faithc::viewer {
namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::optional<double> TryParseDouble(const json &value) {
    if (value.is_number()) {
        return value.get<double>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (value.is_string()) {
        try {
            size_t pos = 0;
            const std::string raw = value.get<std::string>();
            const double parsed = std::stod(raw, &pos);
            while (pos < raw.size() && std::isspace(static_cast<unsigned char>(raw[pos])) != 0) {
                ++pos;
            }
            if (pos == raw.size()) {
                return parsed;
            }
        } catch (...) {
        }
    }
    return std::nullopt;
}

std::optional<int> TryParseInt(const json &value) {
    const std::optional<double> parsed = TryParseDouble(value);
    if (!parsed.has_value() || !std::isfinite(*parsed)) {
        return std::nullopt;
    }
    if (*parsed < static_cast<double>(std::numeric_limits<int>::min()) ||
        *parsed > static_cast<double>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }
    return static_cast<int>(std::llround(*parsed));
}

std::optional<bool> TryParseBool(const json &value) {
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number()) {
        return value.get<double>() != 0.0;
    }
    if (value.is_string()) {
        const std::string lowered = ToLower(value.get<std::string>());
        if (lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "y") {
            return true;
        }
        if (lowered == "false" || lowered == "0" || lowered == "no" || lowered == "n") {
            return false;
        }
    }
    return std::nullopt;
}

std::optional<std::string> TryParseString(const json &value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_float()) {
        return std::to_string(value.get<double>());
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? std::string("true") : std::string("false");
    }
    return std::nullopt;
}

template <typename T, typename Parser>
void AssignField(const json &payload, const char *key, T FaithCJobResult::*member, FaithCJobResult &out,
                 Parser &&parser) {
    const auto it = payload.find(key);
    if (it == payload.end() || it->is_null()) {
        return;
    }
    const auto parsed = parser(*it);
    if (parsed.has_value()) {
        out.*member = *parsed;
    }
}

void PopulateResultFromJson(const json &payload, FaithCJobResult &out) {
    using BoolField = std::pair<const char *, bool FaithCJobResult::*>;
    using IntField = std::pair<const char *, int FaithCJobResult::*>;
    using DoubleField = std::pair<const char *, double FaithCJobResult::*>;
    using StringField = std::pair<const char *, std::string FaithCJobResult::*>;

    static const std::array<BoolField, 9> kBoolFields = {
        BoolField{"success", &FaithCJobResult::success},
        BoolField{"uv_projected", &FaithCJobResult::uv_projected},
        BoolField{"uv_solver_converged_u", &FaithCJobResult::uv_solver_converged_u},
        BoolField{"uv_solver_converged_v", &FaithCJobResult::uv_solver_converged_v},
        BoolField{"uv_island_guard_requested", &FaithCJobResult::uv_island_guard_requested},
        BoolField{"uv_island_guard_enabled", &FaithCJobResult::uv_island_guard_enabled},
        BoolField{"uv_island_guard_allow_unknown", &FaithCJobResult::uv_island_guard_allow_unknown},
        BoolField{"uv_m4_barrier_homotopy_enabled", &FaithCJobResult::uv_m4_barrier_homotopy_enabled},
        BoolField{"uv_m4_pre_repair_enabled", &FaithCJobResult::uv_m4_pre_repair_enabled},
    };

    static const std::array<IntField, 32> kIntFields = {
        IntField{"num_input_faces", &FaithCJobResult::input_faces},
        IntField{"num_output_faces", &FaithCJobResult::output_faces},
        IntField{"active_voxels", &FaithCJobResult::active_voxels},
        IntField{"uv_solver_iters_u", &FaithCJobResult::uv_solver_iters_u},
        IntField{"uv_solver_iters_v", &FaithCJobResult::uv_solver_iters_v},
        IntField{"uv_high_island_count", &FaithCJobResult::uv_high_island_count},
        IntField{"uv_high_seam_edges", &FaithCJobResult::uv_high_seam_edges},
        IntField{"uv_high_boundary_edges", &FaithCJobResult::uv_high_boundary_edges},
        IntField{"uv_high_nonmanifold_edges", &FaithCJobResult::uv_high_nonmanifold_edges},
        IntField{"uv_island_conflict_faces", &FaithCJobResult::uv_island_conflict_faces},
        IntField{"uv_island_conflict_faces_excluded", &FaithCJobResult::uv_island_conflict_faces_excluded},
        IntField{"uv_island_unknown_faces", &FaithCJobResult::uv_island_unknown_faces},
        IntField{"uv_low_cut_edges", &FaithCJobResult::uv_low_cut_edges},
        IntField{"uv_low_split_vertices", &FaithCJobResult::uv_low_split_vertices},
        IntField{"uv_low_split_faces", &FaithCJobResult::uv_low_split_faces},
        IntField{"uv_cross_seam_faces", &FaithCJobResult::uv_cross_seam_faces},
        IntField{"uv_cross_seam_faces_excluded", &FaithCJobResult::uv_cross_seam_faces_excluded},
        IntField{"uv_island_guard_constrained_points", &FaithCJobResult::uv_island_guard_constrained_points},
        IntField{"uv_island_guard_reject_count", &FaithCJobResult::uv_island_guard_reject_count},
        IntField{"uv_m2_irls_iters_used", &FaithCJobResult::uv_m2_irls_iters_used},
        IntField{"uv_m2_face_sample_faces", &FaithCJobResult::uv_m2_face_sample_faces},
        IntField{"uv_m2_face_sample_nonzero", &FaithCJobResult::uv_m2_face_sample_nonzero},
        IntField{"uv_m2_face_sample_max", &FaithCJobResult::uv_m2_face_sample_max},
        IntField{"uv_m4_nonlinear_iters", &FaithCJobResult::uv_m4_nonlinear_iters},
        IntField{"uv_m4_line_search_fail_count", &FaithCJobResult::uv_m4_line_search_fail_count},
        IntField{"uv_m4_patch_refine_rounds", &FaithCJobResult::uv_m4_patch_refine_rounds},
        IntField{"uv_m4_barrier_violations", &FaithCJobResult::uv_m4_barrier_violations},
        IntField{"uv_m4_barrier_violation_count_tol", &FaithCJobResult::uv_m4_barrier_violation_count_tol},
        IntField{"uv_m4_barrier_homotopy_warmup_iters", &FaithCJobResult::uv_m4_barrier_homotopy_warmup_iters},
        IntField{"uv_m4_pre_repair_iters_used", &FaithCJobResult::uv_m4_pre_repair_iters_used},
        IntField{"uv_m4_pre_repair_initial_violations", &FaithCJobResult::uv_m4_pre_repair_initial_violations},
        IntField{"uv_m4_pre_repair_final_violations", &FaithCJobResult::uv_m4_pre_repair_final_violations},
    };

    static const std::array<IntField, 0> kLegacyIntFields = {
    };

    static const std::array<DoubleField, 23> kDoubleFields = {
        DoubleField{"total_seconds", &FaithCJobResult::total_seconds},
        DoubleField{"uv_flip_ratio", &FaithCJobResult::uv_flip_ratio},
        DoubleField{"uv_bad_tri_ratio", &FaithCJobResult::uv_bad_tri_ratio},
        DoubleField{"uv_stretch_p95", &FaithCJobResult::uv_stretch_p95},
        DoubleField{"uv_stretch_p99", &FaithCJobResult::uv_stretch_p99},
        DoubleField{"uv_correspondence_primary_ratio", &FaithCJobResult::uv_correspondence_primary_ratio},
        DoubleField{"uv_color_reproj_l1", &FaithCJobResult::uv_color_reproj_l1},
        DoubleField{"uv_color_reproj_l2", &FaithCJobResult::uv_color_reproj_l2},
        DoubleField{"uv_solver_residual_u", &FaithCJobResult::uv_solver_residual_u},
        DoubleField{"uv_solver_residual_v", &FaithCJobResult::uv_solver_residual_v},
        DoubleField{"uv_island_guard_confidence_min", &FaithCJobResult::uv_island_guard_confidence_min},
        DoubleField{"uv_island_guard_constrained_ratio", &FaithCJobResult::uv_island_guard_constrained_ratio},
        DoubleField{"uv_island_guard_reject_ratio", &FaithCJobResult::uv_island_guard_reject_ratio},
        DoubleField{"uv_island_guard_fallback_success_ratio",
                    &FaithCJobResult::uv_island_guard_fallback_success_ratio},
        DoubleField{"uv_island_guard_invalid_after_guard_ratio",
                    &FaithCJobResult::uv_island_guard_invalid_after_guard_ratio},
        DoubleField{"uv_m2_face_jacobian_cov_p95", &FaithCJobResult::uv_m2_face_jacobian_cov_p95},
        DoubleField{"uv_m2_system_cond_proxy", &FaithCJobResult::uv_m2_system_cond_proxy},
        DoubleField{"uv_m4_energy_init", &FaithCJobResult::uv_m4_energy_init},
        DoubleField{"uv_m4_energy_final", &FaithCJobResult::uv_m4_energy_final},
        DoubleField{"uv_m4_det_min", &FaithCJobResult::uv_m4_det_min},
        DoubleField{"uv_m4_det_p01", &FaithCJobResult::uv_m4_det_p01},
        DoubleField{"uv_m4_barrier_violation_ratio", &FaithCJobResult::uv_m4_barrier_violation_ratio},
        DoubleField{"uv_m4_barrier_violation_ratio_tol", &FaithCJobResult::uv_m4_barrier_violation_ratio_tol},
    };

    static const std::array<DoubleField, 2> kLegacyDoubleFields = {
        DoubleField{"uv_cross_seam_face_ratio", &FaithCJobResult::uv_cross_seam_face_ratio},
        DoubleField{"uv_seam_uv_span_threshold", &FaithCJobResult::uv_seam_uv_span_threshold},
    };

    static const std::array<StringField, 21> kStringFields = {
        StringField{"error", &FaithCJobResult::message},
        StringField{"uv_mode_used", &FaithCJobResult::uv_mode_used},
        StringField{"uv_method", &FaithCJobResult::uv_method},
        StringField{"uv_seam_strategy_requested", &FaithCJobResult::uv_seam_strategy_requested},
        StringField{"uv_seam_strategy_used", &FaithCJobResult::uv_seam_strategy_used},
        StringField{"uv_solver_backend_requested", &FaithCJobResult::uv_solver_backend_requested},
        StringField{"uv_solver_backend_used", &FaithCJobResult::uv_solver_backend_used},
        StringField{"uv_solver_linear_backend_requested", &FaithCJobResult::uv_solver_linear_backend_requested},
        StringField{"uv_solver_linear_backend_used", &FaithCJobResult::uv_solver_linear_backend_used},
        StringField{"uv_solver_stage", &FaithCJobResult::uv_solver_stage},
        StringField{"uv_solver_backend_u", &FaithCJobResult::uv_solver_backend_u},
        StringField{"uv_solver_backend_v", &FaithCJobResult::uv_solver_backend_v},
        StringField{"uv_solver_fallback_reason", &FaithCJobResult::uv_solver_fallback_reason},
        StringField{"uv_island_guard_mode_requested", &FaithCJobResult::uv_island_guard_mode_requested},
        StringField{"uv_island_guard_mode_used", &FaithCJobResult::uv_island_guard_mode_used},
        StringField{"uv_island_guard_fallback_policy", &FaithCJobResult::uv_island_guard_fallback_policy},
        StringField{"uv_m2_laplacian_mode", &FaithCJobResult::uv_m2_laplacian_mode},
        StringField{"uv_m2_face_sample_counts_path", &FaithCJobResult::uv_m2_face_sample_counts_path},
        StringField{"uv_m4_refine_status", &FaithCJobResult::uv_m4_refine_status},
        StringField{"uv_m4_optimizer", &FaithCJobResult::uv_m4_optimizer},
        StringField{"uv_m4_stop_reason", &FaithCJobResult::uv_m4_stop_reason},
    };

    for (const auto &[key, member] : kBoolFields) {
        AssignField(payload, key, member, out, TryParseBool);
    }
    for (const auto &[key, member] : kIntFields) {
        AssignField(payload, key, member, out, TryParseInt);
    }
    for (const auto &[key, member] : kLegacyIntFields) {
        AssignField(payload, key, member, out, TryParseInt);
    }
    for (const auto &[key, member] : kDoubleFields) {
        AssignField(payload, key, member, out, TryParseDouble);
    }
    for (const auto &[key, member] : kLegacyDoubleFields) {
        AssignField(payload, key, member, out, TryParseDouble);
    }
    for (const auto &[key, member] : kStringFields) {
        AssignField(payload, key, member, out, TryParseString);
    }

    AssignField(payload, "uv_halfedge_island_error", &FaithCJobResult::uv_halfedge_island_error, out, TryParseString);
    AssignField(payload, "uv_island_guard_error", &FaithCJobResult::uv_island_guard_error, out, TryParseString);

    const auto it_output = payload.find("output_mesh");
    if (it_output != payload.end() && !it_output->is_null()) {
        const std::optional<std::string> output_path = TryParseString(*it_output);
        if (output_path.has_value() && !output_path->empty()) {
            out.output_mesh = fs::path(*output_path);
        }
    }
}

}  // namespace

FaithCJobResult ParseJobResultFromStatusJson(const fs::path &status_json, const fs::path &default_output_mesh,
                                             int process_exit_code) {
    FaithCJobResult result;
    result.output_mesh = default_output_mesh;
    result.status_json = status_json;

    if (fs::exists(status_json)) {
        try {
            std::ifstream input(status_json);
            json payload;
            input >> payload;
            PopulateResultFromJson(payload, result);
            if (!result.success && result.message.empty()) {
                result.message = "FaithC bridge script failed";
            }
        } catch (const std::exception &e) {
            result.success = false;
            result.message = std::string("Failed to parse status JSON: ") + e.what();
        }
    } else {
        result.success = false;
        result.message = "FaithC bridge did not produce status JSON";
    }

    if (process_exit_code != 0 && result.message.empty()) {
        result.message = "FaithC process exited with non-zero status";
    }

    return result;
}

}  // namespace faithc::viewer
