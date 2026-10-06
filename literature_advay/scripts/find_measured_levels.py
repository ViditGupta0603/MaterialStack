"""Find measured ionization energies / electron affinities in open-access papers (Europe PMC full text).

For a material and a search query, fetch the open-access full texts of matching papers and print
every sentence that names the material and reports an energy level (IE, EA, VBM, CBM, work function,
HOMO, LUMO) with a value in eV, with journal, year and DOI. Used to vet primary sources by hand;
nothing is written to the database.

Usage: .venv/bin/python literature_advay/scripts/find_measured_levels.py "NiO" "NiOx ionization energy UPS" [n_papers]
"""
import html
import json
import re
import sys
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
LEVEL = re.compile(r"\b(IE|EA|IP|ionization (?:energy|potential)|electron affinity|VBM|CBM|valence band "
                   r"(?:maximum|edge)|conduction band (?:minimum|edge)|work function|HOMO|LUMO)\b", re.I)
VALUE = re.compile(r"[-−]?\d\.\d{1,2}\s*(?:±\s*\d\.\d+\s*)?eV")


def search(query: str, n: int) -> list[dict]:
    url = f"{API}/search?" + urllib.parse.urlencode(
        {"query": f"({query}) AND OPEN_ACCESS:y AND HAS_FT:y", "format": "json", "pageSize": n,
         "resultType": "core", "sort": "CITED desc"})
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["resultList"]["result"]


def sentences(pmcid: str) -> list[str]:
    with urllib.request.urlopen(f"{API}/{pmcid}/fullTextXML", timeout=120) as r:
        x = r.read().decode("utf-8", "replace")
    x = re.sub(r"<sub>(.*?)</sub>", r"\1", x)               # keep chemical formulas intact (SnO2, C60)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x)))
    return re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)


def main() -> None:
    material, query = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    name = re.compile(re.escape(material), re.I)
    for paper in search(query, n):
        pmcid = paper.get("pmcid")
        if not pmcid:
            continue
        try:
            hits = [s for s in sentences(pmcid) if name.search(s) and LEVEL.search(s) and VALUE.search(s)]
        except Exception as e:
            print(f"  ({pmcid}: {e})")
            continue
        if not hits:
            continue
        print(f"\n## {paper.get('journalTitle', '')} {paper.get('pubYear')} · cited {paper.get('citedByCount')} · "
              f"doi:{paper.get('doi')} · {pmcid}\n   {paper.get('title', '')[:140]}")
        for s in hits[:6]:
            print("   -", s.strip()[:330])


if __name__ == "__main__":
    main()
