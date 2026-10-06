"""Drive the MaterialStack UI in a headless browser: enter a stack, predict, save screenshots, check width.

Needs the server running (`python -m materialstack serve --port 8001`) and Playwright
(`.venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium`).
Usage (repo root):  .venv/bin/python literature_advay/scripts/ui_screenshots.py ["TiO2\nMAPbI3"] [out_dir]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ui_screenshots")
OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8001/"
STACK = sys.argv[1] if len(sys.argv) > 1 else "ZnO\nCdS\nCuInSe2"


def run(page, tag):
    page.goto(URL)
    page.wait_for_load_state("networkidle")
    page.screenshot(path=OUT / f"{tag}_0_top.png")
    page.fill("textarea", STACK)
    page.click("button[type=submit]")
    page.wait_for_selector(".junction, .layer", timeout=60000)
    page.wait_for_timeout(1200)                 # let the entry animations finish
    page.locator(".results").screenshot(path=OUT / f"{tag}_1_results.png")
    page.click("#validation summary")              # open "All metrics"
    page.locator("#validation").screenshot(path=OUT / f"{tag}_2_validation.png")
    width = page.evaluate("document.documentElement.scrollWidth")
    print(tag, "page width", width, "(sideways scroll!)" if width > page.viewport_size["width"] else "ok")
    errors = [m for m in msgs if m.type == "error"]
    print(tag, "console errors:", [e.text for e in errors] or "none")


with sync_playwright() as p:
    b = p.chromium.launch()
    for tag, opts in [("desktop", {"viewport": {"width": 1400, "height": 900}}),
                      ("dark", {"viewport": {"width": 1400, "height": 900}, "color_scheme": "dark"}),
                      ("phone", {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 2})]:
        ctx = b.new_context(**opts)
        page = ctx.new_page()
        msgs = []
        page.on("console", lambda m: msgs.append(m))
        run(page, tag)
        ctx.close()
    b.close()
print("saved to", OUT)
