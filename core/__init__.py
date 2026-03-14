from core.planet import (
    Planet, AtmosphereConfig, TerrainConfig, MagneticFieldConfig,
    OblatenessConfig, MoonConfig,
    AtmosphereComposition, TerrainType, MagneticFieldStrength,
    G, R_EARTH, M_EARTH,
)
from core.generator import PlanetGenerator, PRESETS
from core.physics import (
    SpacecraftState, OrbitalIntegrator, ThrusterConfig, AeroConfig,
    state_to_orbital_elements,
)
from core.env import OrbitalInsertionEnv

from core.interior import (
    InteriorConfig, InteriorLayer, ConvectionState,
    interior_from_bulk_density, MATERIAL_DENSITY,
)
from core.star import (
    Star, SpectralType, STAR_PRESETS,
    star_sun, star_proxima_centauri, star_trappist1,
    star_tau_ceti, star_kepler452,
    star_alpha_centauri_a, star_eps_eridani,
    AU,
)

__all__ = [
    "Planet", "AtmosphereConfig", "TerrainConfig", "MagneticFieldConfig",
    "OblatenessConfig", "MoonConfig",
    "AtmosphereComposition", "TerrainType", "MagneticFieldStrength",
    "G", "R_EARTH", "M_EARTH",
    "PlanetGenerator", "PRESETS",
    "SpacecraftState", "OrbitalIntegrator", "ThrusterConfig", "AeroConfig",
    "state_to_orbital_elements",
    "OrbitalInsertionEnv",
    "InteriorConfig", "InteriorLayer", "ConvectionState",
    "interior_from_bulk_density", "MATERIAL_DENSITY",
    "Star", "SpectralType", "STAR_PRESETS",
    "star_sun", "star_proxima_centauri", "star_trappist1",
    "star_tau_ceti", "star_kepler452",
    "star_alpha_centauri_a", "star_eps_eridani",
    "AU",
]

# ── Step 2: Atmosphere science + Habitability ─────────────────────────────────
from core.atmosphere_science import (
    MultiLayerAtmosphere, AtmosphericLayer, JeansEscape, GreenhouseModel,
    analyse_atmosphere, STANDARD_COMPOSITIONS, MOLAR_MASS,
)
from core.habitability import (
    HabitabilityAssessment, assess_habitability,
    size_class, composition_class,
)
