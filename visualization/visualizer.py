"""
visualizer.py  —  Planet-RL publication figures.

Rules that prevent ALL overlap/clipping:
  1. No twiny axes.
  2. No nested GridSpec. Every subplot is added with a flat index.
  3. Every figure sets left/right/top/bottom/hspace/wspace explicitly.
  4. Atmosphere profiles are ALWAYS 3 separate side-by-side subplots,
     never squeezed into a shared cell.
  5. Cross-section diagrams live in their own figure row.
  6. Text inside axes only — no annotations outside the axes box.
"""

from __future__ import annotations
import math, os
from typing import Optional, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from core.planet import Planet
from core.physics import SpacecraftState

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
W_BLUE    = "#0072B2"
W_ORANGE  = "#E69F00"
W_GREEN   = "#009E73"
W_RED     = "#D55E00"
W_PINK    = "#CC79A7"
W_SKY     = "#56B4E9"
W_YELLOW  = "#F0E442"
W_BLACK   = "#000000"
WONG      = [W_BLUE, W_RED, W_GREEN, W_ORANGE, W_PINK, W_SKY, W_YELLOW, W_BLACK]
WONG_CYCLE = WONG  # alias for backward compatibility

C_DENSITY  = W_RED
C_PRESSURE = W_BLUE
C_TEMP     = W_GREEN
C_ALT      = W_BLUE
C_SPEED    = W_RED
C_FUEL     = W_GREEN
C_HEAT     = W_ORANGE
C_TARGET   = W_BLACK

# ── Typography ────────────────────────────────────────────────────────────────
FT = 9    # title
FL = 8    # axis label
FK = 7    # tick label
FG = 7    # legend
FA = 6.5  # annotation
LW = 1.4  # main line weight
LW2= 0.9  # reference line weight

# ── Terrain / atmosphere colours ─────────────────────────────────────────────
ATM_COL = {
    "NONE":       "#d9d9d9",
    "CO2_THICK":  "#fdae61",
    "CO2_THIN":   "#fee08b",
    "NITROGEN":   "#abd9e9",
    "EARTH_LIKE": "#74add1",
    "HYDROGEN":   "#ffffbf",
    "METHANE":    "#c994c7",
    "CUSTOM":     "#cccccc",
}
TER_COL = {
    "FLAT":        "#b8cfa0",
    "CRATERED":    "#c4b49a",
    "MOUNTAINOUS": "#a89070",
    "OCEANIC":     "#7eb8d4",
    "VOLCANIC":    "#c08060",
    "RANDOM":      "#bbbbbb",
}

# ── Global style ──────────────────────────────────────────────────────────────
def apply_journal_style():
    matplotlib.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#444444",
        "axes.linewidth":    0.8,
        "axes.labelsize":    FL,
        "axes.titlesize":    FT,
        "axes.titleweight":  "bold",
        "axes.titlepad":     4,
        "axes.prop_cycle":   matplotlib.cycler(color=WONG),
        "axes.grid":         True,
        "grid.color":        "#e4e4e4",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "xtick.labelsize":   FK,
        "ytick.labelsize":   FK,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "legend.fontsize":   FG,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "legend.fancybox":   False,
        "lines.linewidth":   LW,
        "font.family":       "serif",
        "font.size":         FL,
        "mathtext.fontset":  "dejavuserif",
        "savefig.dpi":       300,
        "savefig.facecolor": "white",
        "pdf.fonttype":      42,
    })

apply_journal_style()


def _ax(ax):
    """Apply clean style to one axes."""
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_edgecolor("#444444")
        s.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FK, length=3, width=0.8)
    ax.grid(True, color="#e4e4e4", lw=0.5, ls="--")


def save_figure(fig, filename, output_dir=".", dpi_png=300, formats=("png","pdf")):
    """Save as raster PNG and vector PDF."""
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
# PLANET CROSS-SECTION  (single axes, no text outside bounds)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_cross_section(planet: Planet, ax=None, ref_radius: float = None):
    """
    Draw a schematic cross-section of a planet.

    Parameters
    ----------
    ref_radius : float, optional
        Reference radius (metres) used to scale this planet relative to
        others drawn in the same figure.  All planets in a group should
        receive the same ref_radius = max(p.radius for p in group).
        When None (default, standalone figure) the planet fills the axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.8))
        fig.patch.set_facecolor("white")

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.axis("off")

    comp  = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    atm_c = ATM_COL.get(comp, "#cccccc")
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT",
        "#bbbbbb")

    # Scale factor: planet radius relative to the reference (largest) planet.
    # ref_radius=None → standalone, fills the axes (scale=1.0).
    if ref_radius is None or ref_radius <= 0:
        scale = 1.0
    else:
        scale = planet.radius / ref_radius   # 0 < scale <= 1.0

    # Atmosphere halos (radii are in normalised units × scale)
    if planet.atmosphere.enabled:
        max_alt = planet.atmosphere.scale_height * 5
        atm_frac = max_alt / planet.radius   # fractional thickness
        for i in range(6, 0, -1):
            frac = i / 6
            r = scale * (1.0 + atm_frac * frac)
            ax.add_patch(plt.Circle((0, 0), r, color=atm_c,
                                    alpha=0.06 + 0.16*(1-frac)**1.5, zorder=2))
        ax.add_patch(plt.Circle((0, 0), scale * 1.03, color=atm_c,
                                 alpha=0.4, fill=False, lw=1.5, zorder=3))

    # Surface disc
    ax.add_patch(plt.Circle((0, 0), scale, fc=ter_c, ec="#555555", lw=0.6, zorder=5))
    ax.add_patch(plt.Circle((-0.2*scale, 0.25*scale), 0.50*scale,
                             color="white", alpha=0.14, zorder=6))

    # Core
    cc = W_RED if planet.mean_density > 5000 else "#999999"
    ax.add_patch(plt.Circle((0, 0), 0.26*scale, fc=cc, ec="#333333",
                             lw=0.5, alpha=0.8, zorder=7))

    # Magnetic arcs
    if planet.magnetic_field.enabled:
        t = np.linspace(0, 2*math.pi, 200)
        for sc2, al in [(1.6, 0.45), (2.1, 0.25), (2.7, 0.12)]:
            ax.plot(sc2*scale*np.cos(t), sc2*scale*np.sin(t)*0.44,
                    color=W_BLUE, alpha=al, lw=0.8, zorder=4)

    # Moons
    if planet.moons.enabled:
        for i in range(min(planet.moons.count, 3)):
            ang = math.radians(i*120 + 30)
            ax.add_patch(plt.Circle(
                (2.2*scale*math.cos(ang), 2.2*scale*math.sin(ang)*0.5),
                0.09*scale, fc="#aaaaaa", ec="#555555", lw=0.4, zorder=5))

    tags = []
    if planet.atmosphere.enabled:
        tags.append(comp.lower().replace("_"," "))
    if planet.magnetic_field.enabled:
        tags.append("mag")
    if planet.oblateness.enabled:
        tags.append("J2")
    if planet.moons.enabled:
        tags.append(f"{planet.moons.count}mn")

    # Title above axes (no collision with drawing)
    ax.set_title(planet.name, fontsize=FT, fontweight="bold",
                 color="#111111", pad=4)

    # Stats and tags anchored to fixed positions below the disc
    ax.text(0, -1.75,
            f"R={planet.radius/1e3:,.0f} km   g={planet.surface_gravity:.1f}",
            ha="center", va="center",
            fontsize=FA-0.5, color="#444444", zorder=10)
    if tags:
        ax.text(0, -2.20, " · ".join(tags),
                ha="center", va="center",
                fontsize=FA-1.0, color="#777777", style="italic", zorder=10)

    # Fixed world-space limits so all planets in a row share the same frame
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.55, 1.7)
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE PROFILE  —  always 3 separate axes, no twiny ever
# ═══════════════════════════════════════════════════════════════════════════════

def plot_atmosphere_profile(planet: Planet, axes=None, max_altitude_km=150):
    """
    Draw density / pressure / temperature each on their own axis.
    Pass axes = list/array of exactly 3 Axes, or None to create a new figure.
    Only the leftmost axis shows the y-label; the others suppress it.
    """
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.8),
                                 sharey=True, squeeze=True)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.10, right=0.97,
                            top=0.82, bottom=0.22, wspace=0.12)

    axes = list(axes)

    if not planet.atmosphere.enabled:
        for a in axes:
            a.set_visible(False)
        axes[1].set_visible(True)
        axes[1].set_facecolor("white")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].text(0.5, 0.5, "No atmosphere",
                     transform=axes[1].transAxes,
                     ha="center", va="center",
                     fontsize=FL, color="#aaaaaa", style="italic")
        axes[1].set_title(planet.name, fontsize=FT,
                          fontweight="bold", pad=4)
        return axes

    alts   = np.linspace(0, max_altitude_km * 1e3, 500)
    alts_km = alts / 1e3
    rho    = np.array([planet.atmosphere.density_at_altitude(h)      for h in alts])
    pres   = np.array([planet.atmosphere.pressure_at_altitude(h)/1e3 for h in alts])
    temp   = np.array([planet.atmosphere.temperature_at_altitude(h)  for h in alts])

    data   = [rho, pres, temp]
    colors = [C_DENSITY, C_PRESSURE, C_TEMP]
    xlabels= ["Density (kg/m³)", "Pressure (kPa)", "Temp (K)"]
    H_km   = planet.atmosphere.scale_height / 1e3

    for i, (ax, d, color, xlabel) in enumerate(zip(axes, data, colors, xlabels)):
        _ax(ax)
        ax.fill_betweenx(alts_km, d, alpha=0.10, color=color)
        ax.plot(d, alts_km, color=color, lw=LW)
        # Short x-labels, rotated ticks to prevent crowding
        ax.set_xlabel(xlabel, fontsize=FL-0.5, labelpad=1)
        ax.tick_params(axis='x', labelsize=FK-0.5, rotation=30)
        ax.set_ylim(0, max_altitude_km)

        # y-axis label and tick labels: ONLY on the first (leftmost) panel
        if i == 0:
            ax.set_ylabel("Altitude (km)", fontsize=FL, labelpad=2)
        else:
            # Hide y-axis completely on panels 1 and 2
            ax.set_ylabel("")
            ax.yaxis.set_visible(False)

        # Scale height dotted lines — no text labels to avoid clipping
        for mult in [1, 2, 3]:
            h = H_km * mult
            if 0 < h < max_altitude_km:
                ax.axhline(h, color="#cccccc", lw=0.5, ls=":", zorder=0)

        # Tight x limits
        ax.set_xlim(left=0)
        xmax = d.max()
        ax.set_xlim(0, xmax * 1.08)

    # Title on middle panel only
    axes[1].set_title(planet.name, fontsize=FT, fontweight="bold", pad=4)

    # Surface summary as a single clean line inside the left axes
    axes[0].text(0.97, 0.97,
                 f"P₀={planet.atmosphere.surface_pressure:.3g} Pa\n"
                 f"T₀={planet.atmosphere.surface_temp:.0f} K\n"
                 f"H={planet.atmosphere.scale_height/1e3:.1f} km",
                 transform=axes[0].transAxes,
                 ha="right", va="top",
                 fontsize=FA, color="#555555",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white",
                           ec="#cccccc", lw=0.5))

    return axes


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY 2D
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_2d(planet: Planet, trajectory, ax=None,
                       target_altitude=200_000, color_by="speed"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        fig.patch.set_facecolor("white")

    ax.set_facecolor("#f6f6f6")
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    R     = planet.radius
    comp  = planet.atmosphere.composition.name if planet.atmosphere.enabled else "NONE"
    ter_c = TER_COL.get(
        planet.terrain.terrain_type.name if planet.terrain.enabled else "FLAT", "#bbb")
    atm_c = ATM_COL.get(comp, "#cccccc")

    if planet.atmosphere.enabled:
        ax.add_patch(plt.Circle((0,0), 1.08, color=atm_c, alpha=0.28, zorder=2))
    ax.add_patch(plt.Circle((0,0), 1.0, fc=ter_c, ec="#555555", lw=0.6, zorder=3))

    r_tgt = (R + target_altitude) / R
    ax.add_patch(plt.Circle((0,0), r_tgt, fill=False,
                             color=C_TARGET, ls="--", lw=LW2, zorder=4,
                             label=f"Target {target_altitude/1e3:.0f} km"))

    xs = np.array([s.x / R for s in trajectory])
    ys = np.array([s.y / R for s in trajectory])

    if color_by == "speed":
        vals   = np.array([s.speed for s in trajectory]) / 1e3
        cmap   = matplotlib.colormaps["plasma"]
        clabel = "Speed (km/s)"
    else:
        vals   = np.array([s.fuel_mass for s in trajectory])
        cmap   = matplotlib.colormaps["RdYlGn"]
        clabel = "Fuel (kg)"

    norm  = matplotlib.colors.Normalize(vals.min(), vals.max())
    cvals = norm(vals)
    for i in range(len(xs)-1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(cvals[i]), lw=1.2, alpha=0.9, zorder=5)

    ax.plot(xs[0],  ys[0],  "o", color=W_GREEN, ms=5, zorder=10, label="Start",
            mec="#333", mew=0.4)
    ax.plot(xs[-1], ys[-1], "s", color=W_RED,   ms=5, zorder=10, label="End",
            mec="#333", mew=0.4)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.65, pad=0.03, aspect=20)
    cb.set_label(clabel, fontsize=FL)
    cb.ax.tick_params(labelsize=FK)

    lim = max(abs(xs).max(), abs(ys).max()) * 1.18
    ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / Rp", fontsize=FL)
    ax.set_ylabel("y / Rp", fontsize=FL)
    ax.tick_params(labelsize=FK)
    ax.set_title(f"{planet.name} — trajectory", fontsize=FT, fontweight="bold", pad=5)
    ax.legend(loc="upper left", fontsize=FG, framealpha=0.92,
              edgecolor="#cccccc", fancybox=False)
    ax.grid(True, color="#dddddd", lw=0.5, ls="--")
    return ax


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mission_telemetry(trajectory, planet: Planet,
                           target_altitude=200_000, figsize=(7.0, 4.5)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.46, wspace=0.32,
                        left=0.10, right=0.97,
                        top=0.90, bottom=0.12)

    times  = np.array([s.time                   for s in trajectory])
    alts   = np.array([s.radius - planet.radius  for s in trajectory]) / 1e3
    speeds = np.array([s.speed                   for s in trajectory]) / 1e3
    fuels  = np.array([s.fuel_mass               for s in trajectory])
    heats  = np.array([s.heat_load               for s in trajectory])
    t_min  = times / 60.0

    rows = [
        (axes[0,0], alts,   C_ALT,   "Altitude (km)",    "(a) Altitude",
         target_altitude/1e3, f"{target_altitude/1e3:.0f} km"),
        (axes[0,1], speeds, C_SPEED, "Speed (km/s)",     "(b) Speed",
         planet.circular_orbit_speed(target_altitude)/1e3, "v_circ"),
        (axes[1,0], fuels,  C_FUEL,  "Propellant (kg)",  "(c) Propellant", None, None),
        (axes[1,1], heats,  C_HEAT,  "Heat load (J/m²)", "(d) Aeroheating", None, None),
    ]

    for ax, y, color, ylabel, title, hline, hlabel in rows:
        _ax(ax)
        ax.fill_between(t_min, y, alpha=0.12, color=color)
        ax.plot(t_min, y, color=color, lw=LW)
        if hline is not None:
            ax.axhline(hline, color=C_TARGET, ls="--", lw=LW2, label=hlabel)
            ax.legend(fontsize=FG, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
        ax.set_xlabel("Time (min)", fontsize=FL, labelpad=2)
        ax.set_ylabel(ylabel,       fontsize=FL, labelpad=2)
        ax.set_title(title,         fontsize=FT, fontweight="bold", pad=3)
        ax.tick_params(labelsize=FK)

    fig.suptitle(f"Orbital insertion telemetry — {planet.name}",
                 fontsize=FT+1, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLANET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_planet_comparison(planets, figsize=(7.0, 3.2)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.50, left=0.08, right=0.97,
                        top=0.83, bottom=0.28)

    names  = [p.name for p in planets]
    colors = [WONG[i % len(WONG)] for i in range(len(planets))]

    props = [
        ("Radius (R⊕)",    [p.radius/6.371e6        for p in planets]),
        ("g (m/s²)",       [p.surface_gravity        for p in planets]),
        ("v_esc (km/s)",   [p.escape_velocity/1e3    for p in planets]),
        ("ρ₀ (kg/m³)",    [p.atmosphere.surface_density
                            if p.atmosphere.enabled else 0 for p in planets]),
    ]

    for ax, (ylabel, vals) in zip(axes, props):
        _ax(ax)
        bars = ax.bar(range(len(names)), vals, color=colors,
                      edgecolor="white", lw=0.5, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=38, ha="right",
                           fontsize=FK, rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=FL, labelpad=2)
        ax.tick_params(axis="y", labelsize=FK)
        ax.grid(True, axis="y", color="#e4e4e4", lw=0.5, ls="--")
        ax.grid(False, axis="x")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.03,
                    f"{val:.2g}", ha="center", va="bottom",
                    fontsize=FA, color="#333333")

    fig.suptitle("Planet physical properties",
                 fontsize=FT+1, fontweight="bold")
    return fig