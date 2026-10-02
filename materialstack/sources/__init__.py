"""Source loaders. Each loader is a generator yielding ("record" | "interface" | "alias", dict) tuples.

Record dict keys (all optional except formula + method):
    formula, name, phase_tag, material_class, method, gap, vbm, cbm, edge_reference, is_direct, is_metal,
    structure (pymatgen Structure), spg, e_above_hull, external_id, reference, ip, ea, work_function, miller, extra
"""
from __future__ import annotations

from typing import Callable, Iterator

from materialstack.sources import borlido, jarvis_sources, literature, matminer_sources, materials_project

Loader = Callable[[dict], Iterator[tuple[str, dict]]]

LOADERS: dict[str, Loader] = {
    # matminer-hosted
    "expt_gap": matminer_sources.expt_gap,
    "expt_gap_kingsbury": matminer_sources.expt_gap_kingsbury,
    "castelli_perovskites": matminer_sources.castelli_perovskites,
    "double_perovskites_gap": matminer_sources.double_perovskites_gap,
    "borlido_expt": borlido.load,
    "mp_nostruct_20181018": matminer_sources.mp_nostruct_20181018,
    "mp_all_20181018": matminer_sources.mp_all_20181018,
    "matbench_mp_gap": matminer_sources.matbench_mp_gap,
    "wolverton_oxides": matminer_sources.wolverton_oxides,
    "dielectric_constant": matminer_sources.dielectric_constant,
    # JARVIS-hosted
    "jarvis_dft_3d": jarvis_sources.dft_3d,
    "jarvis_dft_2d": jarvis_sources.dft_2d,
    "snumat": jarvis_sources.snumat,
    "jarvis_halide_perovskites": jarvis_sources.halide_perovskites,
    "foundry_ml_exp_bandgaps": jarvis_sources.foundry_exp_gaps,
    "jarvis_surfacedb": jarvis_sources.surfacedb,
    "jarvis_interfacedb": jarvis_sources.interfacedb,
    "borlido_expt": borlido.load,
    # live / user-supplied
    "materials_project": materials_project.load,
    "literature_csv": literature.literature_csv,
    "aliases": literature.aliases,
}
