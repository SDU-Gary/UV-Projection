#include "job_output_view.h"

#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

namespace faithc::viewer {
namespace {

struct Row {
    std::string label;
    std::string value;
    bool wrap = false;
};

struct Section {
    std::string title;
    std::vector<Row> rows;
};

std::string FormatFloat(double value, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string FormatSci(double value, int precision = 3) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

const char *YesNo(bool value) { return value ? "yes" : "no"; }

std::string UnknownIfEmpty(const std::string &value) { return value.empty() ? std::string("(unknown)") : value; }

void AddRow(std::vector<Row> &rows, const char *label, std::string value, bool wrap = false) {
    rows.push_back(Row{label, std::move(value), wrap});
}

void AddRowIf(std::vector<Row> &rows, const char *label, const std::optional<std::string> &value, bool wrap = false) {
    if (value.has_value()) {
        AddRow(rows, label, *value, wrap);
    }
}

std::optional<std::string> PairRowIfAny(const std::string &left, const std::string &right,
                                        const char *empty = "(unknown)") {
    if (left.empty() && right.empty()) {
        return std::nullopt;
    }
    const std::string lhs = left.empty() ? std::string(empty) : left;
    const std::string rhs = right.empty() ? std::string(empty) : right;
    return lhs + " / " + rhs;
}

std::optional<std::string> PairRowIfAnyInt(int left, int right) {
    if (left < 0 && right < 0) {
        return std::nullopt;
    }
    const int lhs = left < 0 ? -1 : left;
    const int rhs = right < 0 ? -1 : right;
    return std::to_string(lhs) + " / " + std::to_string(rhs);
}

std::optional<std::string> PairRowIfAnyFloat(double left, double right, int precision) {
    if (left < 0.0 && right < 0.0) {
        return std::nullopt;
    }
    const std::string lhs = left < 0.0 ? std::string("-") : FormatFloat(left, precision);
    const std::string rhs = right < 0.0 ? std::string("-") : FormatFloat(right, precision);
    return lhs + " / " + rhs;
}

std::optional<std::string> PairRowIfAnySci(double left, double right, int precision) {
    if (left < 0.0 && right < 0.0) {
        return std::nullopt;
    }
    const std::string lhs = left < 0.0 ? std::string("-") : FormatSci(left, precision);
    const std::string rhs = right < 0.0 ? std::string("-") : FormatSci(right, precision);
    return lhs + " / " + rhs;
}

void RenderSection(const Section &section) {
    if (section.rows.empty()) {
        return;
    }
    ImGui::SeparatorText(section.title.c_str());
    const std::string table_id = "##" + section.title;
    constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable(table_id.c_str(), 2, kFlags)) {
        ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthFixed, 220.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        for (const Row &row : section.rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(row.label.c_str());
            ImGui::TableSetColumnIndex(1);
            if (row.wrap) {
                ImGui::TextWrapped("%s", row.value.c_str());
            } else {
                ImGui::TextUnformatted(row.value.c_str());
            }
        }
        ImGui::EndTable();
    }
}

Section BuildMeshSection(const OutputPanelContext &ctx) {
    Section section{"Mesh Info", {}};
    const MeshData *mesh = ctx.mesh;
    AddRow(section.rows, "Source", ctx.source_mesh.empty() ? std::string("(none)") : ctx.source_mesh.string(), true);
    AddRow(section.rows, "Displayed", ctx.displayed_mesh.empty() ? std::string("(none)") : ctx.displayed_mesh.string(), true);

    if (mesh == nullptr) {
        AddRow(section.rows, "Vertices", "0");
        AddRow(section.rows, "Faces", "0");
        AddRow(section.rows, "Has UV", "no");
        AddRow(section.rows, "BaseColor Tex", "no");
        return section;
    }

    AddRow(section.rows, "Vertices", std::to_string(mesh->vertex_count()));
    AddRow(section.rows, "Faces", std::to_string(mesh->face_count()));
    AddRow(section.rows, "Has UV", YesNo(mesh->has_uv));
    AddRow(section.rows, "BaseColor Tex", YesNo(mesh->has_base_color_texture));
    if (ctx.used_source_texture_fallback) {
        AddRow(section.rows, "Texture Source", "source mesh fallback");
    } else if (mesh->has_base_color_texture) {
        AddRow(section.rows, "Texture Source", "embedded in current mesh");
    }
    if (mesh->has_base_color_texture) {
        AddRow(section.rows, "Texture Size",
               std::to_string(mesh->texture_width) + " x " + std::to_string(mesh->texture_height) + " x " +
                   std::to_string(mesh->texture_channels));
    }
    AddRow(section.rows, "Bounds Min",
           "[" + FormatFloat(mesh->min_bound[0], 3) + " " + FormatFloat(mesh->min_bound[1], 3) + " " +
               FormatFloat(mesh->min_bound[2], 3) + "]");
    AddRow(section.rows, "Bounds Max",
           "[" + FormatFloat(mesh->max_bound[0], 3) + " " + FormatFloat(mesh->max_bound[1], 3) + " " +
               FormatFloat(mesh->max_bound[2], 3) + "]");
    return section;
}

Section BuildRuntimeSection(const OutputPanelContext &ctx) {
    Section section{"Running", {}};
    if (ctx.job_running) {
        AddRow(section.rows, "Job Running", FormatFloat(ctx.job_elapsed_seconds, 3) + "s");
    }
    return section;
}

Section BuildGeneralResultSection(const FaithCJobResult &r) {
    Section section{"Last FaithC Result", {}};
    AddRow(section.rows, "Input Faces", std::to_string(r.input_faces));
    AddRow(section.rows, "Output Faces", std::to_string(r.output_faces));
    if (r.input_faces > 0) {
        const double ratio = static_cast<double>(r.output_faces) / static_cast<double>(r.input_faces);
        AddRow(section.rows, "Reduction Ratio", FormatFloat(ratio, 4));
    }
    AddRow(section.rows, "Active Voxels", std::to_string(r.active_voxels));
    AddRow(section.rows, "Runtime", FormatFloat(r.total_seconds, 3) + "s");
    AddRow(section.rows, "UV Projected", YesNo(r.uv_projected));
    AddRow(section.rows, "UV Mode Used", UnknownIfEmpty(r.uv_mode_used));
    AddRowIf(section.rows, "UV Seam Strategy Req/Used",
             PairRowIfAny(r.uv_seam_strategy_requested, r.uv_seam_strategy_used));
    if (!r.uv_island_validation_mode.empty()) {
        AddRow(section.rows, "UV Island Validation",
               r.uv_island_validation_mode + " / " + YesNo(r.uv_island_validation_ok));
    }
    AddRowIf(section.rows, "UV Solver Backend Req/Used",
             PairRowIfAny(r.uv_solver_backend_requested, r.uv_solver_backend_used));
    AddRowIf(section.rows, "UV Linear Solver Req/Used",
             PairRowIfAny(r.uv_solver_linear_backend_requested, r.uv_solver_linear_backend_used));
    AddRowIf(section.rows, "UV Solver Stage",
             r.uv_solver_stage.empty() ? std::optional<std::string>() : std::optional<std::string>(r.uv_solver_stage));
    AddRowIf(section.rows, "UV Solver Channel Backend U/V", PairRowIfAny(r.uv_solver_backend_u, r.uv_solver_backend_v));
    AddRowIf(section.rows, "UV Solver Iters U/V", PairRowIfAnyInt(r.uv_solver_iters_u, r.uv_solver_iters_v));
    AddRowIf(section.rows, "UV Solver Residual U/V", PairRowIfAnySci(r.uv_solver_residual_u, r.uv_solver_residual_v, 3));
    if (!r.uv_solver_backend_used.empty()) {
        AddRow(section.rows, "UV Solver Converged U/V",
               std::string(YesNo(r.uv_solver_converged_u)) + " / " + YesNo(r.uv_solver_converged_v));
    }
    if (!r.uv_method.empty()) {
        AddRow(section.rows, "UV Method", r.uv_method);
    }
    AddRowIf(section.rows, "UV Solver Fallback", r.uv_solver_fallback_reason.empty()
                                                     ? std::optional<std::string>()
                                                     : std::optional<std::string>(r.uv_solver_fallback_reason),
             true);
    if (r.uv_flip_ratio >= 0.0) {
        AddRow(section.rows, "UV Flip Ratio", FormatFloat(r.uv_flip_ratio, 6));
    }
    if (r.uv_bad_tri_ratio >= 0.0) {
        AddRow(section.rows, "UV Bad-Tri Ratio", FormatFloat(r.uv_bad_tri_ratio, 6));
    }
    AddRowIf(section.rows, "UV Stretch P95/P99", PairRowIfAnyFloat(r.uv_stretch_p95, r.uv_stretch_p99, 3));
    if (r.uv_correspondence_primary_ratio >= 0.0) {
        AddRow(section.rows, "UV Primary Corr Ratio", FormatFloat(r.uv_correspondence_primary_ratio, 4));
    }
    if (r.uv_color_reproj_l1 >= 0.0 && r.uv_color_reproj_l2 >= 0.0) {
        AddRow(section.rows, "UV Color Reproj L1/L2",
               FormatFloat(r.uv_color_reproj_l1, 5) + " / " + FormatFloat(r.uv_color_reproj_l2, 5));
    }
    if (r.uv_m2_irls_iters_used >= 0) {
        AddRow(section.rows, "M2 IRLS Iters", std::to_string(r.uv_m2_irls_iters_used));
    }
    if (r.uv_m2_face_jacobian_cov_p95 >= 0.0) {
        AddRow(section.rows, "M2 Jacobian Cov P95", FormatSci(r.uv_m2_face_jacobian_cov_p95, 3));
    }
    if (r.uv_m2_system_cond_proxy >= 0.0) {
        AddRow(section.rows, "M2 System Cond Proxy", FormatSci(r.uv_m2_system_cond_proxy, 3));
    }
    if (!r.uv_m2_laplacian_mode.empty()) {
        AddRow(section.rows, "M2 Laplacian", r.uv_m2_laplacian_mode);
    }
    if (!r.uv_m4_refine_status.empty()) {
        AddRow(section.rows, "M4 Refine Status", r.uv_m4_refine_status);
    }
    if (!r.uv_m4_optimizer.empty()) {
        AddRow(section.rows, "M4 Optimizer", r.uv_m4_optimizer);
    }
    if (!r.uv_m4_stop_reason.empty()) {
        AddRow(section.rows, "M4 Stop Reason", r.uv_m4_stop_reason);
    }
    if (r.uv_m4_nonlinear_iters >= 0) {
        AddRow(section.rows, "M4 Nonlinear Iters", std::to_string(r.uv_m4_nonlinear_iters));
    }
    if (r.uv_m4_energy_init >= 0.0 && r.uv_m4_energy_final >= 0.0) {
        AddRow(section.rows, "M4 Energy Init/Final",
               FormatSci(r.uv_m4_energy_init, 3) + " / " + FormatSci(r.uv_m4_energy_final, 3));
    }
    if (r.uv_m4_det_min >= 0.0 || r.uv_m4_det_p01 >= 0.0) {
        const std::string det_min = r.uv_m4_det_min >= 0.0 ? FormatSci(r.uv_m4_det_min, 3) : "-";
        const std::string det_p01 = r.uv_m4_det_p01 >= 0.0 ? FormatSci(r.uv_m4_det_p01, 3) : "-";
        AddRow(section.rows, "M4 det min/p01", det_min + " / " + det_p01);
    }
    if (r.uv_m4_barrier_violations >= 0 || r.uv_m4_line_search_fail_count >= 0 || r.uv_m4_patch_refine_rounds >= 0) {
        AddRow(section.rows, "M4 viol/linefail/patch",
               std::to_string(r.uv_m4_barrier_violations) + " / " + std::to_string(r.uv_m4_line_search_fail_count) +
                   " / " + std::to_string(r.uv_m4_patch_refine_rounds));
    }
    if (r.uv_m4_barrier_violation_ratio >= 0.0 || r.uv_m4_barrier_violation_ratio_tol >= 0.0 ||
        r.uv_m4_barrier_violation_count_tol >= 0) {
        const std::string vr = r.uv_m4_barrier_violation_ratio >= 0.0 ? FormatFloat(r.uv_m4_barrier_violation_ratio, 5) : "-";
        const std::string vr_tol =
            r.uv_m4_barrier_violation_ratio_tol >= 0.0 ? FormatFloat(r.uv_m4_barrier_violation_ratio_tol, 5) : "-";
        const std::string vc_tol =
            r.uv_m4_barrier_violation_count_tol >= 0 ? std::to_string(r.uv_m4_barrier_violation_count_tol) : "-";
        AddRow(section.rows, "M4 violation ratio / tol / cnt_tol", vr + " / " + vr_tol + " / " + vc_tol);
    }
    if (r.uv_m4_barrier_homotopy_warmup_iters >= 0 || r.uv_m4_pre_repair_iters_used >= 0) {
        AddRow(section.rows, "M4 homotopy/pre-repair",
               std::string(YesNo(r.uv_m4_barrier_homotopy_enabled)) + "@" +
                   std::to_string(r.uv_m4_barrier_homotopy_warmup_iters) + " / " + YesNo(r.uv_m4_pre_repair_enabled) +
                   "@" + std::to_string(r.uv_m4_pre_repair_iters_used));
    }
    if (r.uv_m4_pre_repair_initial_violations >= 0 || r.uv_m4_pre_repair_final_violations >= 0) {
        AddRow(section.rows, "M4 pre-repair viol init/final",
               std::to_string(r.uv_m4_pre_repair_initial_violations) + " / " +
                   std::to_string(r.uv_m4_pre_repair_final_violations));
    }
    if (r.uv_island_conflict_faces >= 0 || r.uv_island_unknown_faces >= 0 || r.uv_low_cut_edges >= 0 ||
        r.uv_low_split_vertices >= 0) {
        AddRow(section.rows, "Diag Summary (conflict/unknown/cut/splitv)",
               std::to_string(r.uv_island_conflict_faces) + " / " + std::to_string(r.uv_island_unknown_faces) +
                   " / " + std::to_string(r.uv_low_cut_edges) + " / " + std::to_string(r.uv_low_split_vertices));
    }
    AddRow(section.rows, "Output Mesh", r.output_mesh.string(), true);
    return section;
}

Section BuildSeamSection(const FaithCJobResult &r) {
    Section section{"Seam / Island Diagnostics", {}};
    if (r.uv_island_guard_requested || r.uv_island_guard_enabled || !r.uv_island_guard_mode_used.empty() ||
        !r.uv_island_guard_error.empty()) {
        AddRow(section.rows, "Island Guard Requested/Enabled",
               std::string(YesNo(r.uv_island_guard_requested)) + " / " + YesNo(r.uv_island_guard_enabled));
    }
    if (!r.uv_island_guard_mode_requested.empty() || !r.uv_island_guard_mode_used.empty()) {
        AddRow(section.rows, "Island Guard Mode Req/Used",
               UnknownIfEmpty(r.uv_island_guard_mode_requested) + " / " + UnknownIfEmpty(r.uv_island_guard_mode_used));
    }
    if (r.uv_island_guard_confidence_min >= 0.0) {
        AddRow(section.rows, "Island Guard Confidence Min", FormatFloat(r.uv_island_guard_confidence_min, 3));
    }
    if (r.uv_island_guard_constrained_points >= 0 || r.uv_island_guard_constrained_ratio >= 0.0) {
        AddRow(section.rows, "Island Guard Constrained Pts/Ratio",
               std::to_string(r.uv_island_guard_constrained_points) + " / " +
                   FormatFloat(r.uv_island_guard_constrained_ratio, 4));
    }
    if (r.uv_island_guard_reject_count >= 0 || r.uv_island_guard_reject_ratio >= 0.0) {
        AddRow(section.rows, "Island Guard Reject Cnt/Ratio",
               std::to_string(r.uv_island_guard_reject_count) + " / " + FormatFloat(r.uv_island_guard_reject_ratio, 4));
    }
    if (r.uv_island_guard_fallback_success_ratio >= 0.0) {
        AddRow(section.rows, "Island Guard Fallback Success Ratio",
               FormatFloat(r.uv_island_guard_fallback_success_ratio, 4));
    }
    if (r.uv_island_guard_invalid_after_guard_ratio >= 0.0) {
        AddRow(section.rows, "Island Guard Invalid-After Ratio",
               FormatFloat(r.uv_island_guard_invalid_after_guard_ratio, 4));
    }
    if (!r.uv_island_guard_fallback_policy.empty()) {
        AddRow(section.rows, "Island Guard Fallback Policy", r.uv_island_guard_fallback_policy);
    }
    if (!r.uv_island_guard_error.empty()) {
        AddRow(section.rows, "Island Guard Error", r.uv_island_guard_error, true);
    }
    if (r.uv_high_island_count >= 0) {
        AddRow(section.rows, "High UV Islands", std::to_string(r.uv_high_island_count));
    }
    if (r.uv_high_seam_edges >= 0 || r.uv_high_boundary_edges >= 0 || r.uv_high_nonmanifold_edges >= 0) {
        AddRow(section.rows, "High Seam/Boundary/Nonmanifold Edges",
               std::to_string(r.uv_high_seam_edges) + " / " + std::to_string(r.uv_high_boundary_edges) + " / " +
                   std::to_string(r.uv_high_nonmanifold_edges));
    }
    if (r.uv_island_conflict_faces >= 0 || r.uv_island_unknown_faces >= 0 || r.uv_island_conflict_faces_excluded >= 0) {
        AddRow(section.rows, "Low Face Conflict/Unknown/Excluded",
               std::to_string(r.uv_island_conflict_faces) + " / " + std::to_string(r.uv_island_unknown_faces) + " / " +
                   std::to_string(r.uv_island_conflict_faces_excluded));
    }
    if (r.uv_semantic_component_merge_min_faces >= 0 || r.uv_semantic_component_merge_merged_components >= 0 ||
        r.uv_semantic_component_merge_merged_faces >= 0) {
        AddRow(section.rows, "Component Merge Enabled/MinFaces",
               std::string(YesNo(r.uv_semantic_component_merge_enabled)) + " / " +
                   std::to_string(r.uv_semantic_component_merge_min_faces));
        AddRow(section.rows, "Component Merge MergedComp/Faces",
               std::to_string(r.uv_semantic_component_merge_merged_components) + " / " +
                   std::to_string(r.uv_semantic_component_merge_merged_faces));
    }
    if (r.uv_semantic_pre_cleanup_fragmented_label_count >= 0 || r.uv_semantic_pre_cleanup_severe_label_count >= 0 ||
        r.uv_semantic_final_fragmented_label_count >= 0 || r.uv_semantic_final_severe_label_count >= 0) {
        AddRow(section.rows, "Semantic Frag/Severe PreCleanup",
               std::to_string(r.uv_semantic_pre_cleanup_fragmented_label_count) + " / " +
                   std::to_string(r.uv_semantic_pre_cleanup_severe_label_count));
        AddRow(section.rows, "Semantic Frag/Severe Final",
               std::to_string(r.uv_semantic_final_fragmented_label_count) + " / " +
                   std::to_string(r.uv_semantic_final_severe_label_count));
    }
    if (r.uv_low_cut_edges >= 0 || r.uv_low_split_vertices >= 0 || r.uv_low_split_faces >= 0) {
        AddRow(section.rows, "Low Cut Edges/Split V/Split F",
               std::to_string(r.uv_low_cut_edges) + " / " + std::to_string(r.uv_low_split_vertices) + " / " +
                   std::to_string(r.uv_low_split_faces));
    }
    if (r.uv_cross_seam_faces >= 0 || r.uv_cross_seam_face_ratio >= 0.0 || r.uv_cross_seam_faces_excluded >= 0) {
        AddRow(section.rows, "Legacy Cross-Seam Faces/Ratio/Excluded",
               std::to_string(r.uv_cross_seam_faces) + " / " + FormatFloat(r.uv_cross_seam_face_ratio, 4) + " / " +
                   std::to_string(r.uv_cross_seam_faces_excluded));
    }
    if (r.uv_seam_uv_span_threshold >= 0.0) {
        AddRow(section.rows, "Legacy Seam Span Threshold", FormatFloat(r.uv_seam_uv_span_threshold, 4));
    }
    if (!r.uv_halfedge_island_error.empty()) {
        AddRow(section.rows, "Halfedge Seam Error", r.uv_halfedge_island_error, true);
    }
    if (!r.uv_island_validation_error.empty()) {
        AddRow(section.rows, "Island Validation Error", r.uv_island_validation_error, true);
    }
    return section;
}

bool HasSeamDiagnostics(const FaithCJobResult &r) {
    return r.uv_high_island_count >= 0 || r.uv_island_conflict_faces >= 0 || r.uv_island_unknown_faces >= 0 ||
           r.uv_semantic_component_merge_merged_components >= 0 || r.uv_semantic_final_severe_label_count >= 0 ||
           r.uv_low_cut_edges >= 0 || r.uv_cross_seam_faces >= 0 || !r.uv_halfedge_island_error.empty() ||
           !r.uv_island_validation_mode.empty() || !r.uv_island_validation_error.empty() ||
           r.uv_island_guard_requested || r.uv_island_guard_enabled || !r.uv_island_guard_mode_used.empty() ||
           !r.uv_island_guard_error.empty();
}

}  // namespace

std::string BuildFaithCJobSummary(const FaithCJobResult &result) {
    std::ostringstream oss;
    oss << "FaithC finished: faces " << result.input_faces << " -> " << result.output_faces << " in "
        << FormatFloat(result.total_seconds, 3) << "s";
    if (!result.uv_solver_backend_used.empty()) {
        oss << " | uv_solve=" << result.uv_solver_backend_used;
    }
    if (!result.uv_seam_strategy_used.empty()) {
        oss << " | seam=" << result.uv_seam_strategy_used;
    }
    if (!result.uv_island_validation_mode.empty()) {
        oss << " | island_val=" << result.uv_island_validation_mode << ":" << (result.uv_island_validation_ok ? "ok" : "fail");
    }
    if (result.uv_island_conflict_faces >= 0 || result.uv_island_unknown_faces >= 0) {
        oss << " | conflict/unknown=" << result.uv_island_conflict_faces << "/" << result.uv_island_unknown_faces;
    }
    if (result.uv_low_cut_edges >= 0 || result.uv_low_split_vertices >= 0) {
        oss << " | cut/splitv=" << result.uv_low_cut_edges << "/" << result.uv_low_split_vertices;
    }
    return oss.str();
}

void RenderFaithCOutputPanel(const OutputPanelContext &ctx) {
    RenderSection(BuildMeshSection(ctx));
    RenderSection(BuildRuntimeSection(ctx));

    if (ctx.last_job_result != nullptr) {
        const FaithCJobResult &result = *ctx.last_job_result;
        RenderSection(BuildGeneralResultSection(result));
        if (HasSeamDiagnostics(result)) {
            RenderSection(BuildSeamSection(result));
        }
    }

    ImGui::Separator();
    const char *status = ctx.status_text.empty() ? "ready" : ctx.status_text.c_str();
    ImGui::TextWrapped("Status: %s", status);
}

}  // namespace faithc::viewer
