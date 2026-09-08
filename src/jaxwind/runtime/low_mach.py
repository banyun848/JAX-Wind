"""Low-Mach profile reporting (execution lives in runtime.engine)."""
from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import numpy as np

def _write_mean_profile(
    directory: Path,
    workflow,
    grid,
    fields: tuple[np.ndarray, ...],
    frame_times: list[float],
    *,
    statistics_start_fraction: float = 0.0,
) -> dict[str, object]:
    """Write horizontally/time-averaged profiles and a neutral log-law check."""

    from jaxwind.config.abl_resolved import resolved
    from jaxwind.simulation.abl import build_models
    from jaxwind.io.reporting import write_log_law_svg
    from jaxwind import logarithmic_profile

    configuration = resolved(workflow.case)
    _boundaries, momentum, _scalar, _buoyancy, _surface = build_models(
        workflow.case,
        periodic_x=True,
        evolve_scalar=False,
    )
    wall = momentum.surface
    if wall is None:
        raise ValueError("mean-profile reporting requires a neutral wall model")
    z = np.asarray(grid.z_centers, dtype=np.float64)
    u_history, v_history, w_history = fields[6:9]
    u2_history, v2_history, w2_history = fields[9:12]
    wall_friction_history = fields[12]
    if not 0.0 <= statistics_start_fraction < 1.0:
        raise ValueError("statistics_start_fraction must lie in [0, 1)")
    statistics_start_index = int(
        statistics_start_fraction * len(frame_times)
    )
    statistics_slice = slice(statistics_start_index, None)
    statistics_times = frame_times[statistics_start_index:]
    u_samples = u_history[statistics_slice]
    v_samples = v_history[statistics_slice]
    w_samples = w_history[statistics_slice]
    u2_samples = u2_history[statistics_slice]
    v2_samples = v2_history[statistics_slice]
    w2_samples = w2_history[statistics_slice]
    wall_friction_samples = wall_friction_history[statistics_slice]
    mean_u = np.mean(u_samples, axis=0, dtype=np.float64)
    mean_v = np.mean(v_samples, axis=0, dtype=np.float64)
    mean_w = np.mean(w_samples, axis=0, dtype=np.float64)
    u_rms = np.sqrt(
        np.maximum(
            np.mean(u2_samples, axis=0, dtype=np.float64) - mean_u**2,
            0.0,
        )
    )
    v_rms = np.sqrt(
        np.maximum(
            np.mean(v2_samples, axis=0, dtype=np.float64) - mean_v**2,
            0.0,
        )
    )
    w_rms = np.sqrt(
        np.maximum(
            np.mean(w2_samples, axis=0, dtype=np.float64) - mean_w**2,
            0.0,
        )
    )
    pressure_acceleration = float(
        configuration["pressure_acceleration_m_s2"][0]
    )
    reference_friction = math.sqrt(max(pressure_acceleration * grid.lz, 0.0))
    if reference_friction <= 0.0:
        raise ValueError("log-law comparison requires positive pressure forcing")
    roughness = float(configuration["roughness_length_m"])
    von_karman = float(wall.von_karman)
    reference_u = np.asarray(
        logarithmic_profile(grid, reference_friction, wall),
        dtype=np.float64,
    )
    fit_upper = min(0.1 * grid.lz, 2.0)
    fit_mask = z <= fit_upper
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("log-law fit requires three cells below 0.1 H")
    # Use the exact finite-volume cell-average coordinate rather than the
    # centre-point logarithm used by a continuous wall-law fit.
    fit_coordinate = (
        reference_u[fit_mask] * von_karman / reference_friction
    )
    slope, intercept = np.polyfit(fit_coordinate, mean_u[fit_mask], 1)
    fitted = slope * fit_coordinate + intercept
    residual = mean_u[fit_mask] - fitted
    total = mean_u[fit_mask] - np.mean(mean_u[fit_mask])
    fitted_friction = von_karman * slope
    fitted_roughness = (
        roughness * math.exp(-intercept / slope)
        if slope > 0.0
        else math.nan
    )
    reference_residual = mean_u[fit_mask] - reference_u[fit_mask]

    profile_path = directory / "mean_velocity_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "z_m",
                "mean_u_m_s",
                "mean_v_m_s",
                "mean_w_m_s",
                "u_rms_m_s",
                "v_rms_m_s",
                "w_rms_m_s",
                "log_law_u_m_s",
            ),
        )
        writer.writeheader()
        for index, height in enumerate(z):
            writer.writerow(
                {
                    "z_m": height,
                    "mean_u_m_s": mean_u[index],
                    "mean_v_m_s": mean_v[index],
                    "mean_w_m_s": mean_w[index],
                    "u_rms_m_s": u_rms[index],
                    "v_rms_m_s": v_rms[index],
                    "w_rms_m_s": w_rms[index],
                    "log_law_u_m_s": reference_u[index],
                }
            )
    archive_path = directory / "mean_velocity_profile.npz"
    np.savez_compressed(
        archive_path,
        z_m=z,
        time_seconds=np.asarray(frame_times),
        u_profile_history_m_s=u_history,
        v_profile_history_m_s=v_history,
        w_profile_history_m_s=w_history,
        surface_friction_velocity_history_m_s=wall_friction_history,
        mean_u_m_s=mean_u,
        mean_v_m_s=mean_v,
        mean_w_m_s=mean_w,
        u_rms_m_s=u_rms,
        v_rms_m_s=v_rms,
        w_rms_m_s=w_rms,
        log_law_u_m_s=reference_u,
    )
    figure_path = directory / "mean_velocity_log_law.svg"
    write_log_law_svg(
        profile_path,
        figure_path,
        friction_velocity_m_s=reference_friction,
        roughness_length_m=roughness,
        von_karman=von_karman,
        model_label="low-Mach FV",
        statistics_label=f"{len(statistics_times)} post-spinup profiles",
    )
    summary = {
        "sample_count": len(statistics_times),
        "discarded_sample_count": statistics_start_index,
        "statistics_start_fraction": statistics_start_fraction,
        "sample_start_time_seconds": statistics_times[0],
        "sample_end_time_seconds": statistics_times[-1],
        "fit_height_range_m": [float(z[fit_mask][0]), fit_upper],
        "pressure_balance_friction_velocity_m_s": reference_friction,
        "mean_surface_friction_velocity_m_s": float(
            np.mean(wall_friction_samples, dtype=np.float64)
        ),
        "surface_to_pressure_friction_velocity_ratio": float(
            np.mean(wall_friction_samples, dtype=np.float64)
            / reference_friction
        ),
        "fitted_friction_velocity_m_s": fitted_friction,
        "fitted_roughness_length_m": fitted_roughness,
        "fitted_r_squared": float(
            1.0 - np.sum(residual**2) / np.maximum(np.sum(total**2), 1.0e-30)
        ),
        "fitted_rmse_m_s": float(np.sqrt(np.mean(residual**2))),
        "configured_log_law_rmse_m_s": float(
            np.sqrt(np.mean(reference_residual**2))
        ),
        "configured_log_law_maximum_error_m_s": float(
            np.max(np.abs(reference_residual))
        ),
        "profile_csv": str(profile_path),
        "profile_archive": str(archive_path),
        "figure": str(figure_path),
    }
    summary_path = directory / "mean_velocity_profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary
