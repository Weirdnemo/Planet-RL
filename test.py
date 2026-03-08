"""
test_planets.py  —  Generate and visualise planets using Planet-RL sandbox.

Run from Planet-RL/:
    python test_planets.py

Outputs saved to ./figures/:
    fig1_presets.png/pdf         5 real-world preset planets
    fig2_random.png/pdf          6 random planets (all features on)
    fig3_atm_zoo.png/pdf         one planet per atmosphere type
    fig4_airless.png/pdf         vacuum worlds, terrain comparison
    fig5_toggle.png/pdf          same seed, features added one by one
    fig6_batch.png/pdf           stats over 50-planet batch
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from core import PlanetGenerator, PRESETS, AtmosphereComposition, TerrainType
from visualization import (
    F_LEGEND,
    plot_planet_cross_section,
    plot_atmosphere_profile,
    plot_planet_comparison,
    save_figure,
    apply_journal_style,
    WONG_CYCLE,
    F_TITLE, F_LABEL, F_TICK, F_ANNOT,
    W_BLUE, W_RED, W_GREEN, W_ORANGE,
)
from visualization.visualizer import _clean_ax

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
apply_journal_style()

print("=" * 56)
print("Planet-RL  —  Generation & Visualisation Test")
print("=" * 56)


# ─────────────────────────────────────────────────────────────
# Helper: draw one planet column (cross-section above, atm below)
# ─────────────────────────────────────────────────────────────
def _planet_column(fig, outer_gs, col, ncols, planet, max_alt=150):
    ax_top = fig.add_subplot(outer_gs[0, col])
    plot_planet_cross_section(planet, ax=ax_top)

    # Three atmosphere sub-panels side by side inside the bottom cell
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs[1, col], wspace=0.10)
    atm_axes = [fig.add_subplot(inner[j]) for j in range(3)]
    plot_atmosphere_profile(planet, axes=atm_axes, max_altitude_km=max_alt)


# ═════════════════════════════════════════════════════════════
# FIG 1 — Preset planets
# ═════════════════════════════════════════════════════════════
print("\n[1/6] Preset planets ...")
presets = [PRESETS[k]() for k in PRESETS]   # Earth, Mars, Venus, Moon, Titan
N = len(presets)

fig1 = plt.figure(figsize=(7.08, 5.6))
fig1.patch.set_facecolor("white")
outer = gridspec.GridSpec(2, N, figure=fig1,
                          hspace=0.55, wspace=0.30,
                          top=0.93, bottom=0.10,
                          left=0.04, right=0.97)

for col, planet in enumerate(presets):
    _planet_column(fig1, outer, col, N, planet)

fig1.suptitle("Solar-system analogue presets",
              fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig1, "fig1_presets", output_dir=OUT)
plt.close(fig1)


# ═════════════════════════════════════════════════════════════
# FIG 2 — 6 random planets, all features on
# ═════════════════════════════════════════════════════════════
print("[2/6] Random planets ...")
gen = PlanetGenerator(seed=2024)
randoms = [gen.generate(name=f"Planet-{chr(65+i)}",
                        atmosphere_enabled=True, terrain_enabled=True,
                        magnetic_field_enabled=True, oblateness_enabled=True,
                        moons_enabled=True)
           for i in range(6)]

for p in randoms:
    print(f"  {p.name}  R={p.radius/1e6:.1f}Mm  "
          f"g={p.surface_gravity:.1f}m/s²  "
          f"atm={p.atmosphere.composition.name}")

fig2 = plt.figure(figsize=(7.08, 5.6))
fig2.patch.set_facecolor("white")
outer2 = gridspec.GridSpec(2, 6, figure=fig2,
                           hspace=0.55, wspace=0.30,
                           top=0.93, bottom=0.10,
                           left=0.03, right=0.98)

for col, planet in enumerate(randoms):
    _planet_column(fig2, outer2, col, 6, planet)

fig2.suptitle("Randomly generated planets — all features enabled",
              fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig2, "fig2_random", output_dir=OUT)
plt.close(fig2)


# ═════════════════════════════════════════════════════════════
# FIG 3 — Atmosphere zoo (one per composition)
# ═════════════════════════════════════════════════════════════
print("[3/6] Atmosphere zoo ...")

comp_info = [
    (AtmosphereComposition.CO2_THIN,   "CO₂ thin\n(Mars-like)"),
    (AtmosphereComposition.CO2_THICK,  "CO₂ thick\n(Venus-like)"),
    (AtmosphereComposition.EARTH_LIKE, "N₂/O₂\n(Earth-like)"),
    (AtmosphereComposition.NITROGEN,   "N₂\n(inert)"),
    (AtmosphereComposition.HYDROGEN,   "H₂\n(gas giant)"),
    (AtmosphereComposition.METHANE,    "CH₄\n(Titan-like)"),
]

gen_atm = PlanetGenerator(seed=7)
atm_planets = [
    gen_atm.generate(
        name=label.split("\n")[0],
        atmosphere_enabled=True,
        atmosphere_composition=comp,
        terrain_enabled=False,
        magnetic_field_enabled=False,
        oblateness_enabled=False,
        moons_enabled=False,
    )
    for comp, label in comp_info
]

# Layout: 2 rows × 3 cols, each cell = 3-panel atmosphere profile
fig3 = plt.figure(figsize=(7.08, 5.0))
fig3.patch.set_facecolor("white")
outer3 = gridspec.GridSpec(2, 3, figure=fig3,
                            hspace=0.72, wspace=0.35,
                            top=0.90, bottom=0.12,
                            left=0.08, right=0.98)

for idx, (planet, (_, label)) in enumerate(zip(atm_planets, comp_info)):
    row, col = divmod(idx, 3)
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer3[row, col], wspace=0.12)
    atm_axes = [fig3.add_subplot(inner[j]) for j in range(3)]
    # Only show y-axis label on leftmost panel of each row
    plot_atmosphere_profile(planet, axes=atm_axes, max_altitude_km=180)
    # Override title with composition label
    atm_axes[1].set_title(label, fontsize=F_LABEL,
                          fontweight="bold", pad=3)
    atm_axes[0].set_title("")
    atm_axes[2].set_title("")
    # Only keep y-tick labels on left edge of figure
    if col > 0:
        atm_axes[0].set_ylabel("")
        atm_axes[0].tick_params(labelleft=False)

fig3.suptitle("Atmosphere profiles by composition type",
              fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig3, "fig3_atm_zoo", output_dir=OUT)
plt.close(fig3)


# ═════════════════════════════════════════════════════════════
# FIG 4 — Airless worlds (terrain types)
# ═════════════════════════════════════════════════════════════
print("[4/6] Airless worlds ...")

terrain_info = [
    (TerrainType.CRATERED,    "Cratered"),
    (TerrainType.MOUNTAINOUS, "Mountainous"),
    (TerrainType.VOLCANIC,    "Volcanic"),
    (TerrainType.FLAT,        "Flat"),
]

gen_bare = PlanetGenerator(seed=99)
bare_planets = [
    gen_bare.generate(
        name=name,
        atmosphere_enabled=False,
        terrain_enabled=True,
        terrain_type=ttype,
        magnetic_field_enabled=False,
        oblateness_enabled=False,
        moons_enabled=False,
    )
    for ttype, name in terrain_info
]

fig4, axes4 = plt.subplots(1, 4, figsize=(7.08, 2.5))
fig4.patch.set_facecolor("white")
fig4.subplots_adjust(wspace=0.20, top=0.88, bottom=0.06,
                     left=0.02, right=0.98)

for ax, planet in zip(axes4, bare_planets):
    plot_planet_cross_section(planet, ax=ax)

fig4.suptitle("Airless worlds — terrain archetypes",
              fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig4, "fig4_airless", output_dir=OUT)
plt.close(fig4)


# ═════════════════════════════════════════════════════════════
# FIG 5 — Feature toggle (same seed, incremental complexity)
# ═════════════════════════════════════════════════════════════
print("[5/6] Feature toggle ...")

toggle_configs = [
    ("Atm only",
     dict(atmosphere_enabled=True,  terrain_enabled=False,
          magnetic_field_enabled=False, oblateness_enabled=False,
          moons_enabled=False)),
    ("+ Terrain",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=False, oblateness_enabled=False,
          moons_enabled=False)),
    ("+ Magnetosphere",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=True,  oblateness_enabled=False,
          moons_enabled=False)),
    ("+ J2 + Moons",
     dict(atmosphere_enabled=True,  terrain_enabled=True,
          magnetic_field_enabled=True,  oblateness_enabled=True,
          moons_enabled=True)),
]

fig5, axes5 = plt.subplots(1, 4, figsize=(7.08, 2.6))
fig5.patch.set_facecolor("white")
fig5.subplots_adjust(wspace=0.18, top=0.85, bottom=0.06,
                     left=0.02, right=0.98)

for ax, (label, kwargs) in zip(axes5, toggle_configs):
    g = PlanetGenerator(seed=42)
    p = g.generate(name=label,
                   radius_range=(0.85 * 6.371e6, 1.15 * 6.371e6),
                   density_range=(4800, 5800),
                   **kwargs)
    plot_planet_cross_section(p, ax=ax)
    # Replace auto-title with config label
    for txt in ax.texts:
        if txt.get_position()[1] > 2.2:
            txt.set_text(label)
            txt.set_fontsize(F_ANNOT + 0.5)

fig5.suptitle("Feature toggle — identical planet seed",
              fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig5, "fig5_toggle", output_dir=OUT)
plt.close(fig5)


# ═════════════════════════════════════════════════════════════
# FIG 6 — Batch statistics (50 planets)
# ═════════════════════════════════════════════════════════════
print("[6/6] Batch statistics ...")

gen_b = PlanetGenerator(seed=1337)
batch = gen_b.batch(50, atmosphere_enabled=True, terrain_enabled=True,
                    magnetic_field_enabled=True, oblateness_enabled=True,
                    moons_enabled=True)

radii    = np.array([p.radius / 6.371e6         for p in batch])
gravs    = np.array([p.surface_gravity           for p in batch])
escvs    = np.array([p.escape_velocity / 1e3     for p in batch])
densities= np.array([p.mean_density              for p in batch])
atm_d    = np.array([p.atmosphere.surface_density
                     if p.atmosphere.enabled else 0.0 for p in batch])
n_moons  = np.array([p.moons.count if p.moons.enabled else 0 for p in batch])
has_mag  = np.array([p.magnetic_field.enabled    for p in batch])

fig6, axes6 = plt.subplots(2, 3, figsize=(7.08, 4.5))
fig6.patch.set_facecolor("white")
fig6.subplots_adjust(hspace=0.52, wspace=0.38,
                     left=0.10, right=0.97,
                     top=0.91, bottom=0.11)

def _hist(ax, data, xlabel, title, color):
    _clean_ax(ax)
    ax.hist(data, bins=12, color=color, edgecolor="white", lw=0.4)
    mu = data.mean()
    ax.axvline(mu, color=W_RED, lw=1.2, ls="--",
               label=f"μ={mu:.2f}")
    ax.set_xlabel(xlabel, fontsize=F_LABEL, labelpad=2)
    ax.set_ylabel("Count",  fontsize=F_LABEL, labelpad=2)
    ax.set_title(title,     fontsize=F_TITLE, fontweight="bold", pad=3)
    ax.tick_params(labelsize=F_TICK)
    ax.legend(fontsize=F_LEGEND, framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

_hist(axes6[0,0], radii,    r"Radius  ($R_\oplus$)",         "(a) Radius",       W_BLUE)
_hist(axes6[0,1], gravs,    r"$g$  (m s$^{-2}$)",           "(b) Gravity",      W_ORANGE)
_hist(axes6[0,2], escvs,    r"$v_\mathrm{esc}$  (km/s)",    "(c) Escape vel.",  W_GREEN)
_hist(axes6[1,0], densities,r"Density  (kg m$^{-3}$)",      "(d) Mean density", W_RED)
_hist(axes6[1,1], atm_d,    r"$\rho_0$  (kg m$^{-3}$)",    "(e) Atm. density", W_BLUE)

# Moon count bar chart
ax_m = axes6[1,2]
_clean_ax(ax_m)
unique, counts = np.unique(n_moons, return_counts=True)
ax_m.bar(unique, counts,
         color=[WONG_CYCLE[i % len(WONG_CYCLE)] for i in range(len(unique))],
         edgecolor="white", lw=0.4, width=0.6)
ax_m.set_xlabel("Moon count",  fontsize=F_LABEL, labelpad=2)
ax_m.set_ylabel("Count",       fontsize=F_LABEL, labelpad=2)
ax_m.set_title("(f) Moon distribution",
               fontsize=F_TITLE, fontweight="bold", pad=3)
ax_m.tick_params(labelsize=F_TICK)
ax_m.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_m.spines["top"].set_visible(False)
ax_m.spines["right"].set_visible(False)

fig6.suptitle(
    f"Batch statistics — 50 random planets"
    f"  |  magnetosphere: {has_mag.mean()*100:.0f}% of batch",
    fontsize=F_TITLE + 1, fontweight="bold")
save_figure(fig6, "fig6_batch", output_dir=OUT)
plt.close(fig6)


# ─────────────────────────────────────────────────────────────
print()
print("=" * 56)
print(f"Done. All figures saved to ./{OUT}/")
print("=" * 56)
print(f"  Radius    {radii.min():.2f} – {radii.max():.2f} R⊕  "
      f"(mean {radii.mean():.2f})")
print(f"  Gravity   {gravs.min():.1f} – {gravs.max():.1f} m/s²  "
      f"(mean {gravs.mean():.1f})")
print(f"  Escape v  {escvs.min():.1f} – {escvs.max():.1f} km/s  "
      f"(mean {escvs.mean():.1f})")
print(f"  Moons     max {n_moons.max()}, mean {n_moons.mean():.1f}")
print(f"  Mag field {has_mag.sum()}/50 planets")