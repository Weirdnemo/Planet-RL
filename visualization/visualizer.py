"""
visualizer.py  —  Publication-quality figures for Planet-RL.

Design rules (to prevent ALL overlap issues):
  - No twiny / twin axes in grid layouts. Ever.
  - Each variable gets its own subplot.
  - All spacing is explicit via gridspec, never tight_layout alone.
  - Fonts sized for 300 dpi print at journal column width.
  - Wong (2011) colorblind-safe palette throughout.
"""

from __future__ import annotations
import math
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from core.planet import Planet
from core.physics import SpacecraftState


# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
W_BLUE   = "#0072B2"
W_ORANGE = "#E69F00"
W_GREEN  = "#009E73"
W_RED    = "#D55E00"
W_PINK   = "#CC79A7"
W_SKYBLUE= "#56B4E9"
W_YELLOW = "#F0E442"
W_BLACK  = "#000000"

WONG_CYCLE = [W_BLUE, W_RED, W_GREEN, W_ORANGE, W_PINK, W_SKYBLUE, W_YELLOW, W_BLACK]

# Semantic assignments
C_DENSITY  = W_RED
C_PRESSURE = W_BLUE
C_TEMP     = W_GREEN
C_ALT      = W_BLUE
C_SPEED    = W_RED
C_FUEL     = W_GREEN
C_HEAT     = W_ORANGE
C_TARGET   = W_BLACK

# ── Typography ────────────────────────────────────────────────────────────────
F_TITLE  = 9
F_LABEL  = 8
F_TICK   = 7
F_LEGEND = 7
F_ANNOT  = 6.5

LW = 1.4   # main line weight
LW2 = 0.9  # secondary line weight

# ── Terrain / atmosphere colour maps ─────────────────────────────────────────
ATM_COLORS = {
    "NONE":        "#d9d9d9",
    "CO2_THICK":   "#fdae61",
    "CO2_THIN":    "#fee08b",
    "NITROGEN":    "#abd9e9",
    "EARTH_LIKE":  "#74add1",
    "HYDROGEN":    "#ffffbf",
    "METHANE":     "#c994c7",
    "CUSTOM":      "#cccccc",
}

TERRAIN_COLORS = {
    "FLAT":        "#b8cfa0",
    "CRATERED":    "#c4b49a",
    "MOUNTAINOUS": "#a89070",
    "OCEANIC":     "#7eb8d4",
    "VOLCANIC":    "#c08060",
    "RANDOM":      "#bbbbbb",
}


# ── Global rcParams ───────────────────────────────────────────────────────────
def apply_journal_style():
    matplotlib.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#444444",
        "axes.linewidth":      0.8,
        "axes.labelsize":      F_LABEL,
        "axes.titlesize":      F_TITLE,
        "axes.titleweight":    "bold",
        "axes.titlepad":       4,
        "axes.prop_cycle":     matplotlib.cycler(color=WONG_CYCLE),
        "axes.grid":           True,
        "grid.color":          "#e4e4e4",
        "grid.linewidth":      0.5,
        "grid.linestyle":      "--",
        "xtick.labelsize":     F_TICK,
        "ytick.labelsize":     F_TICK,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.major.size":    3,
        "ytick.major.size":    3,
        "legend.fontsize":     F_LEGEND,
        "legend.framealpha":   0.9,
        "legend.edgecolor":    "#cccccc",
        "legend.fancybox":     False,
        "lines.linewidth":     LW,
        "font.family":         "serif",
        "font.size":           F_LABEL,
        "mathtext.fontset":    "dejavuserif",
        "savefig.dpi":         300,
        "savefig.facecolor":   "white",
        "pdf.fonttype":        42,
    })

apply_journal_style()


def _clean_ax(ax):
    """Minimal clean style — no extra chrome."""
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
        sp.set_linewidth(0.8)
    ax.tick_params(labelsize=F_TICK, length=3, width=0.8)
    ax.grid(True, color="#e4e4e4", lw=0.5, ls="--")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def save_figure(fig, filename, output_dir=".", dpi_png=300,
                formats=("png", "pdf")):
    """Save figure as PNG (raster) and PDF (vector) for journal submission."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        kw = dict(bbox_inches="tight", facecolor="white")
        if fmt == "png":
            kw["dpi"] = dpi_png
        fig.savefig(path, format=fmt, **kw)
        print(f"  Saved: {path}")
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET CROSS-SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_cross_section(planet: Planet, ax=None, n_atm_rings=6):
    """Schematic diagram of a planet. Works correctly inside any grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        fig.patch.set_facecolor("white")

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.axis("off")

    comp = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    atm_c = ATM_COLORS.get(comp, "#cccccc")
    ter_c = TERRAIN_COLORS.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#bbbbbb")

    # Atmosphere rings
    if planet.atmosphere.enabled:
        max_alt = planet.atmosphere.scale_height * 5
        for i in range(n_atm_rings, 0, -1):
            frac = i / n_atm_rings
            r = (planet.radius + max_alt * frac) / planet.radius
            alpha = 0.07 + 0.18 * (1 - frac) ** 1.5
            ax.add_patch(plt.Circle((0, 0), r, color=atm_c, alpha=alpha, zorder=2))
        ax.add_patch(plt.Circle((0, 0), 1.03, color=atm_c, alpha=0.45,
                                 fill=False, lw=1.6, zorder=3))

    # Surface
    ax.add_patch(plt.Circle((0, 0), 1.0, fc=ter_c, ec="#555555", lw=0.7, zorder=5))
    # Lighting
    ax.add_patch(plt.Circle((-0.2, 0.25), 0.52, color="white", alpha=0.15, zorder=6))

    # Core
    core_c = W_RED if planet.mean_density > 5000 else "#999999"
    ax.add_patch(plt.Circle((0, 0), 0.27, fc=core_c, ec="#333333",
                             lw=0.5, alpha=0.8, zorder=7))

    # Magnetic field arcs
    if planet.magnetic_field.enabled:
        theta = np.linspace(0, 2 * math.pi, 200)
        for scale, alpha in [(1.65, 0.50), (2.20, 0.28), (2.80, 0.14)]:
            ax.plot(scale * np.cos(theta), scale * np.sin(theta) * 0.44,
                    color=W_BLUE, alpha=alpha, lw=0.9, zorder=4)

    # Moons
    if planet.moons.enabled:
        for i in range(min(planet.moons.count, 3)):
            angle = math.radians(i * 120 + 30)
            ax.add_patch(plt.Circle((2.3 * math.cos(angle),
                                     2.3 * math.sin(angle) * 0.5),
                                    0.10, fc="#aaaaaa", ec="#555555",
                                    lw=0.4, zorder=5))

    # Name + stats — placed well inside the bounding box
    ax.text(0, 2.55, planet.name,
            ha="center", va="center",
            fontsize=F_TITLE, fontweight="bold", color="#111111", zorder=10)
    ax.text(0, 2.25,
            f"R={planet.radius/1e3:,.0f} km   g={planet.surface_gravity:.2f} m/s²",
            ha="center", va="center",
            fontsize=F_ANNOT, color="#444444", zorder=10)

    # Feature tags
    tags = []
    if planet.atmosphere.enabled:
        tags.append(comp.lower().replace("_", " "))
    if planet.magnetic_field.enabled:
        tags.append("mag")
    if planet.oblateness.enabled:
        tags.append("J2")
    if planet.moons.enabled:
        n = planet.moons.count
        tags.append(f"{n}moon{'s' if n > 1 else ''}")
    if tags:
        ax.text(0, -2.45, " · ".join(tags),
                ha="center", va="center",
                fontsize=F_ANNOT - 0.5, color="#666666",
                style="italic", zorder=10)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.7, 2.85)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE PROFILE  —  3 separate subplots, no twiny
# ═══════════════════════════════════════════════════════════════════════════════

def plot_atmosphere_profile(planet: Planet, axes=None, max_altitude_km=150):
    """
    Three-panel atmosphere profile: density | pressure | temperature.
    Each variable has its own subplot with its own x-axis — no overlapping.

    Parameters
    ----------
    axes : array of 3 matplotlib Axes, or None (creates its own figure).
    
    Returns
    -------
    Array of 3 Axes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8), sharey=True)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(wspace=0.08)

    if not planet.atmosphere.enabled:
        # Span all three axes with a single message
        for ax in axes:
            ax.set_visible(False)
        # Use the middle axes bounding box via the figure
        mid = axes[1]
        mid.set_visible(True)
        _clean_ax(mid)
        mid.set_xticks([])
        mid.set_yticks([])
        mid.text(0.5, 0.5, "No atmosphere",
                 transform=mid.transAxes,
                 ha="center", va="center",
                 fontsize=F_LABEL, color="#aaaaaa", style="italic")
        mid.set_title(planet.name, fontsize=F_TITLE, fontweight="bold", pad=4)
        return axes

    alts    = np.linspace(0, max_altitude_km * 1e3, 500)
    alts_km = alts / 1e3
    rho  = np.array([planet.atmosphere.density_at_altitude(h)     for h in alts])
    pres = np.array([planet.atmosphere.pressure_at_altitude(h)/1e3 for h in alts])
    temp = np.array([planet.atmosphere.temperature_at_altitude(h)  for h in alts])

    H_km = planet.atmosphere.scale_height / 1e3

    panels = [
        (axes[0], rho,  r"Density  (kg m$^{-3}$)", C_DENSITY),
        (axes[1], pres, "Pressure  (kPa)",          C_PRESSURE),
        (axes[2], temp, "Temperature  (K)",          C_TEMP),
    ]

    for i, (ax, data, xlabel, color) in enumerate(panels):
        _clean_ax(ax)
        ax.fill_betweenx(alts_km, data, alpha=0.10, color=color)
        ax.plot(data, alts_km, color=color, lw=LW)

        ax.set_xlabel(xlabel, fontsize=F_LABEL, labelpad=3)
        if i == 0:
            ax.set_ylabel("Altitude  (km)", fontsize=F_LABEL, labelpad=3)
        else:
            ax.tick_params(labelleft=False)

        ax.set_ylim(0, max_altitude_km)

        # Scale height guide lines
        for mult in [1, 2, 3]:
            h = H_km * mult
            if h < max_altitude_km:
                ax.axhline(h, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
                if i == 2:
                    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 1 else data.max(),
                            h + max_altitude_km * 0.01,
                            f"{mult}H", fontsize=F_ANNOT - 0.5,
                            color="#999999", ha="right")

        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Title above middle panel
    axes[1].set_title(planet.name, fontsize=F_TITLE, fontweight="bold", pad=4)

    # Surface summary below left panel only
    axes[0].annotate(
        f"$P_0$={planet.atmosphere.surface_pressure:.3g} Pa   "
        f"$T_0$={planet.atmosphere.surface_temp:.0f} K   "
        f"$H$={planet.atmosphere.scale_height/1e3:.1f} km",
        xy=(0, -0.22), xycoords="axes fraction",
        fontsize=F_ANNOT, color="#555555", style="italic",
        annotation_clip=False,
    )

    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY 2D
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_2d(planet: Planet, trajectory, ax=None,
                       target_altitude=200_000, color_by="speed"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        fig.patch.set_facecolor("white")

    ax.set_facecolor("#f5f5f5")
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999")
        sp.set_linewidth(0.8)

    R = planet.radius
    comp = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    ter_c = TERRAIN_COLORS.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT", "#bbb")
    atm_c = ATM_COLORS.get(comp, "#cccccc")

    if planet.atmosphere.enabled:
        ax.add_patch(plt.Circle((0, 0), 1.08, color=atm_c, alpha=0.30, zorder=2))
    ax.add_patch(plt.Circle((0, 0), 1.0, fc=ter_c, ec="#555555", lw=0.7, zorder=3))

    r_tgt = (R + target_altitude) / R
    ax.add_patch(plt.Circle((0, 0), r_tgt, fill=False,
                             color=C_TARGET, ls="--", lw=LW2, zorder=4,
                             label=f"Target {target_altitude/1e3:.0f} km"))

    xs = np.array([s.x / R for s in trajectory])
    ys = np.array([s.y / R for s in trajectory])

    if color_by == "speed":
        vals  = np.array([s.speed for s in trajectory]) / 1e3
        cmap  = matplotlib.colormaps["plasma"]
        clabel = r"Speed  (km s$^{-1}$)"
    else:
        vals  = np.array([s.fuel_mass for s in trajectory])
        cmap  = matplotlib.colormaps["RdYlGn"]
        clabel = "Fuel  (kg)"

    norm   = matplotlib.colors.Normalize(vals.min(), vals.max())
    cvals  = norm(vals)
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(cvals[i]), lw=1.2, alpha=0.9, zorder=5)

    ax.plot(xs[0],  ys[0],  "o", color=W_GREEN,  ms=5, zorder=10, label="Start",
            markeredgecolor="#333", markeredgewidth=0.4)
    ax.plot(xs[-1], ys[-1], "s", color=W_RED,     ms=5, zorder=10, label="End",
            markeredgecolor="#333", markeredgewidth=0.4)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.03, aspect=22)
    cb.set_label(clabel, fontsize=F_LABEL)
    cb.ax.tick_params(labelsize=F_TICK)

    lim = max(abs(xs).max(), abs(ys).max()) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$x\;/\;R_p$", fontsize=F_LABEL)
    ax.set_ylabel(r"$y\;/\;R_p$", fontsize=F_LABEL)
    ax.tick_params(labelsize=F_TICK)
    ax.set_title(f"{planet.name} — trajectory (equatorial plane)",
                 fontsize=F_TITLE, fontweight="bold", pad=5)
    ax.legend(loc="upper left", fontsize=F_LEGEND,
              framealpha=0.92, edgecolor="#cccccc", fancybox=False)
    ax.grid(True, color="#dddddd", lw=0.5, ls="--")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mission_telemetry(trajectory, planet: Planet,
                           target_altitude=200_000, figsize=(7.0, 4.5)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.42, wspace=0.32,
                        left=0.10, right=0.97, top=0.90, bottom=0.12)

    times  = np.array([s.time                        for s in trajectory])
    alts   = np.array([s.radius - planet.radius       for s in trajectory]) / 1e3
    speeds = np.array([s.speed                        for s in trajectory]) / 1e3
    fuels  = np.array([s.fuel_mass                    for s in trajectory])
    heats  = np.array([s.heat_load                    for s in trajectory])
    t_min  = times / 60.0

    panels = [
        (axes[0,0], alts,   C_ALT,   "Altitude  (km)",
         "(a) Altitude", target_altitude/1e3,
         f"{target_altitude/1e3:.0f} km target"),
        (axes[0,1], speeds, C_SPEED, r"Speed  (km s$^{-1}$)",
         "(b) Speed",
         planet.circular_orbit_speed(target_altitude) / 1e3,
         r"$v_\mathrm{circ}$"),
        (axes[1,0], fuels,  C_FUEL,  "Propellant  (kg)",
         "(c) Propellant", None, None),
        (axes[1,1], heats,  C_HEAT,  r"Heat load  (J m$^{-2}$)",
         "(d) Aeroheating", None, None),
    ]

    for ax, y, color, ylabel, title, hline, hlabel in panels:
        _clean_ax(ax)
        ax.fill_between(t_min, y, alpha=0.12, color=color)
        ax.plot(t_min, y, color=color, lw=LW)
        if hline is not None:
            ax.axhline(hline, color=C_TARGET, ls="--", lw=LW2, label=hlabel)
            ax.legend(fontsize=F_LEGEND, framealpha=0.9,
                      edgecolor="#cccccc", fancybox=False)
        ax.set_xlabel("Time  (min)", fontsize=F_LABEL, labelpad=2)
        ax.set_ylabel(ylabel,        fontsize=F_LABEL, labelpad=2)
        ax.set_title(title,          fontsize=F_TITLE, fontweight="bold", pad=3)
        ax.tick_params(labelsize=F_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Orbital insertion telemetry — {planet.name}",
                 fontsize=F_TITLE + 1, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_comparison(planets, figsize=(7.0, 3.5)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.45, left=0.10, right=0.97,
                        top=0.85, bottom=0.22)

    names  = [p.name for p in planets]
    colors = [WONG_CYCLE[i % len(WONG_CYCLE)] for i in range(len(planets))]

    props = [
        (r"Radius  ($R_\oplus$)",
         [p.radius / 6.371e6 for p in planets]),
        (r"$g$  (m s$^{-2}$)",
         [p.surface_gravity for p in planets]),
        (r"$v_\mathrm{esc}$  (km s$^{-1}$)",
         [p.escape_velocity / 1e3 for p in planets]),
        (r"$\rho_0$  (kg m$^{-3}$)",
         [p.atmosphere.surface_density
          if p.atmosphere.enabled else 0 for p in planets]),
    ]

    for ax, (ylabel, vals) in zip(axes, props):
        _clean_ax(ax)
        bars = ax.bar(range(len(names)), vals, color=colors,
                      edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right",
                           fontsize=F_TICK, rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=F_LABEL, labelpad=2)
        ax.tick_params(axis="y", labelsize=F_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#e4e4e4", lw=0.5, ls="--")
        ax.grid(False, axis="x")
        # Value labels on top of bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    f"{val:.2g}",
                    ha="center", va="bottom",
                    fontsize=F_ANNOT, color="#333333")

    fig.suptitle("Planet physical properties",
                 fontsize=F_TITLE + 1, fontweight="bold", y=0.98)
    return fig