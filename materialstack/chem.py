"""Formula parsing, cleaning, family classification and simple physics descriptors."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pymatgen.core import Composition, Element, Species, Structure

HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"S", "Se", "Te"}
PNICTOGENS = {"N", "P", "As", "Sb"}
ANIONS = {"O"} | HALOGENS | CHALCOGENS | PNICTOGENS

# Common polytype / phase prefixes and allotrope suffixes seen in literature material names.
_PHASE_PREFIX = re.compile(r"^(1T'?|1Td|2H|3R|1T|alpha|beta|gamma|delta|a|b|c|m|r|w|zb|h|cubic|tetragonal|"
                           r"orthorhombic|hexagonal|anatase|rutile|brookite|wurtzite|zincblende)[-_ ]", re.I)
_PHASE_SUFFIX = re.compile(r"[-_ ](c|m|np|nc|meso|mp|planar|anatase|rutile|brookite|wurtzite|zb|w|b|h|"
                           r"amorphous|a|film|bulk|monolayer|ML|1L|2L|few[- ]layer)$", re.I)
_DOPING = re.compile(r":\s*[A-Za-z0-9,]+$")

ORGANIC_MARKERS = {"spiro", "ptaa", "pedot", "pss", "pcbm", "p3ht", "pbdb", "pm6", "itic", "y6", "c60", "c70",
                   "pffbt", "tfb", "polytpd", "poly", "meh-ppv", "pbtt", "npb", "bcp", "tpbi", "alq3", "npd",
                   "cupc", "pcdtbt", "ppv", "pvk", "f8bt", "pfo", "pentacene", "rubrene", "dcv", "ome", "bphen"}


@dataclass
class CleanName:
    raw: str
    formula_clean: str | None
    phase_tag: str
    dopant_tag: str
    material_class: str  # inorganic_bulk | inorganic_2d | organic | doped_composite | unknown
    parseable: bool
    notes: list[str] = field(default_factory=list)


def clean_material_name(raw: str) -> CleanName:
    """Strip phase/polytype tags and dopant suffixes; decide material class; test pymatgen parseability."""
    s = raw.strip()
    notes: list[str] = []
    phase = ""
    dopant = ""

    m = _DOPING.search(s)
    if m and not s.lower().startswith("pedot"):
        dopant = m.group(0).lstrip(":").strip()
        s = s[: m.start()].strip()
        notes.append(f"dopant tag '{dopant}' stripped")

    m = _PHASE_PREFIX.match(s)
    if m:
        phase = m.group(1)
        s = s[m.end():]
    m = _PHASE_SUFFIX.search(s)
    if m and len(s) - len(m.group(0)) >= 2:
        phase = (phase + " " + m.group(1)).strip()
        s = s[: m.start()]

    lower = s.lower()
    is_organic_name = any(tok in lower for tok in ORGANIC_MARKERS)
    comp = try_composition(s)

    if comp is None and is_organic_name:
        cls = "organic"
    elif comp is None:
        cls = "unknown"
    elif dopant:
        cls = "doped_composite"
    elif _looks_organic(comp):
        cls = "organic"
    elif phase and re.match(r"^(1T'?|1Td|2H|3R|1T|monolayer|ML|1L|2L|few[- ]layer)$", phase, re.I):
        cls = "inorganic_2d"
    else:
        cls = "inorganic_bulk"

    return CleanName(raw=raw, formula_clean=comp.reduced_formula if comp else None, phase_tag=phase,
                     dopant_tag=dopant, material_class=cls, parseable=comp is not None, notes=notes)


def _looks_organic(comp: Composition) -> bool:
    els = {e.symbol for e in comp}
    return "C" in els and "H" in els and not (els & {"Pb", "Sn", "Ge", "Bi", "Ag", "Cu"} and els & HALOGENS)


def try_composition(formula: str) -> Composition | None:
    try:
        c = Composition(formula)
        if c.num_atoms <= 0 or any(not isinstance(e, Element) for e in c):
            return None
        return c
    except Exception:
        return None


def reduced_formula(formula: str) -> str | None:
    c = try_composition(formula)
    return c.reduced_formula if c else None


# --------------------------------------------------------------------------- families

def classify_family(comp: Composition) -> dict[str, Any]:
    els = {e.symbol for e in comp}
    anon = comp.anonymized_formula
    counts = {e.symbol: v for e, v in comp.reduced_composition.items()}
    amounts = sorted(counts.items(), key=lambda kv: kv[1])
    anion_syms = els & ANIONS
    family = "other"
    is_perov = False

    if _looks_organic(comp):
        family = "organic"
    elif els & {"Pb", "Sn", "Ge", "Bi"} and els & HALOGENS and {"C", "H", "N"} <= els:
        family, is_perov = "hybrid_perovskite", True
    elif anon == "ABC3" and amounts[-1][0] in ANIONS:
        family, is_perov = "perovskite_ABX3", True
    elif anon in {"ABC2D6", "AB2C6"} and amounts[-1][0] in ANIONS:
        family, is_perov = "double_perovskite", True
    elif not anion_syms:
        family = "intermetallic"
    elif "O" in els and not (els & HALOGENS):
        family = "oxide"
    elif els & HALOGENS:
        family = "halide"
    elif els & CHALCOGENS:
        family = "chalcogenide"
    elif els & PNICTOGENS:
        family = "pnictide"

    out = {"family": family, "is_perovskite_like": int(is_perov), "anonymized_formula": anon,
           "n_elements": len(els), "tolerance_factor": None, "octahedral_factor": None}
    if is_perov and family != "hybrid_perovskite":
        t, mu = perovskite_factors(comp)
        out["tolerance_factor"], out["octahedral_factor"] = t, mu
    return out


def perovskite_factors(comp: Composition) -> tuple[float | None, float | None]:
    """Goldschmidt tolerance factor and octahedral factor from Shannon radii (oxidation-state guess)."""
    import warnings
    try:
        warnings.simplefilter("ignore", UserWarning)
        guesses = comp.reduced_composition.oxi_state_guesses(max_sites=-1)
        if not guesses:
            return None, None
        oxi = guesses[0]
        counts = {e.symbol: v for e, v in comp.reduced_composition.items()}
        anion = max(counts, key=counts.get)
        cations = [s for s in counts if s != anion]
        r = {}
        for s in counts:
            sp = Species(s, oxi[s])
            r[s] = sp.ionic_radius or Element(s).average_ionic_radius
            if r[s] is None:
                return None, None
        rX = float(r[anion])
        if len(cations) == 2:
            rA, rB = sorted((float(r[c]) for c in cations), reverse=True)
        else:  # double perovskite: A2 B B' X6 -> average B-site
            a_site = max(cations, key=lambda c: counts[c])
            rA = float(r[a_site])
            rB = sum(float(r[c]) for c in cations if c != a_site) / max(1, len(cations) - 1)
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        return round(t, 4), round(rB / rX, 4)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- electronegativity

@lru_cache(maxsize=None)
def mulliken_chi_element(symbol: str) -> float | None:
    e = Element(symbol)
    ie = e.ionization_energy
    ea = e.electron_affinity
    if ie is None:
        return None
    if ea is None:
        ea = 0.0
    return (ie + ea) / 2.0


def mulliken_chi(comp: Composition) -> float | None:
    """Geometric-mean Mulliken electronegativity (Butler-Ginley) in eV (absolute scale)."""
    total = comp.num_atoms
    acc = 0.0
    for e, n in comp.items():
        chi = mulliken_chi_element(e.symbol)
        if chi is None or chi <= 0:
            return None
        acc += n * math.log(chi)
    return math.exp(acc / total)


def butler_ginley_edges(chi: float, gap: float) -> tuple[float, float]:
    """Return (CBM, VBM) vs vacuum from Mulliken electronegativity and gap."""
    cbm = -(chi - gap / 2.0)
    return cbm, cbm - gap


# --------------------------------------------------------------------------- structures

def jarvis_atoms_to_structure(atoms: dict) -> Structure | None:
    try:
        return Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"],
                         coords_are_cartesian=bool(atoms.get("cartesian", False)))
    except Exception:
        return None


def structure_summary(s: Structure, spg: int | None = None) -> dict[str, Any]:
    """Density / volume / site count. Space-group analysis is skipped here: it dominates ingest time
    (~0.5 s/structure) and most sources already ship a space-group number."""
    return {"n_sites": len(s), "density": float(s.density), "volume_per_atom": float(s.volume / len(s)),
            "space_group_number": spg, "crystal_system": None}


def structure_to_cif(s: Structure) -> str | None:
    try:
        return s.to(fmt="cif")
    except Exception:
        return None


def to_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v
