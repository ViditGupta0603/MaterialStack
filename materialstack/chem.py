"""Formulas and the two pieces of chemistry the tool needs: Mulliken electronegativity and Butler–Ginley."""
from __future__ import annotations

import math
import re
from functools import lru_cache

from pymatgen.core import Composition, Element

# Phase words people put around a formula ("anatase TiO2", "c-TiO2", "ZnO film"); they do not change the formula.
_PHASE_PREFIX = re.compile(r"^(alpha|beta|gamma|delta|cubic|tetragonal|orthorhombic|hexagonal|anatase|rutile|"
                           r"brookite|wurtzite|zincblende|c|m|mp)[-_ ]", re.I)
_PHASE_SUFFIX = re.compile(r"[-_ ](anatase|rutile|brookite|wurtzite|film|bulk|planar|meso|c|m|np)$", re.I)

# Radioactive elements without device use; materials containing them are not used (rule R2 in build_data.py).
RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
               "Cf", "Es", "Fm", "Md", "No", "Lr"}


def composition(text: str) -> Composition | None:
    """pymatgen Composition of a formula, or None if it is not a formula (e.g. 'Spiro-OMeTAD')."""
    try:
        c = Composition(text)
        if c.num_atoms <= 0 or any(not isinstance(e, Element) for e in c):
            return None
        return c
    except Exception:
        return None


def formula_key(text: str) -> str | None:
    """The one spelling used everywhere as a key: pymatgen's reduced formula ('O2Ti', 'TiO2 film' → 'TiO2')."""
    s = str(text).strip()
    s = _PHASE_PREFIX.sub("", s)
    s = _PHASE_SUFFIX.sub("", s)
    c = composition(s)
    return c.reduced_formula if c is not None else None


def elements(formula: str) -> list[str]:
    c = composition(formula)
    return sorted(e.symbol for e in c) if c is not None else []


@lru_cache(maxsize=None)
def _chi_element(symbol: str) -> float | None:
    e = Element(symbol)
    if e.ionization_energy is None:
        return None
    return (e.ionization_energy + (e.electron_affinity or 0.0)) / 2.0


def mulliken_chi(formula: str) -> float | None:
    """Geometric mean of the atoms' Mulliken electronegativities (eV), as used by Butler & Ginley (1978)."""
    c = composition(formula)
    if c is None:
        return None
    acc = 0.0
    for e, n in c.items():
        chi = _chi_element(e.symbol)
        if chi is None or chi <= 0:
            return None
        acc += n * math.log(chi)
    return math.exp(acc / c.num_atoms)


def butler_ginley_vbm(chi: float, gap: float) -> float:
    """VBM vs vacuum when nothing better is known: the gap sits symmetrically around −χ."""
    return -chi - gap / 2.0


def to_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v
