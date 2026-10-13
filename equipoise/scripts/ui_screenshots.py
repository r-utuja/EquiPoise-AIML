"""Screenshot every main page of the running app at desktop and phone width (Playwright).

Start the app first (`streamlit run app/streamlit_app.py`), then:
    python scripts/ui_screenshots.py [--url http://localhost:8501] [--out reports/figures/ui]
                                     [--example poorer|moderate|good]

Without --example the wizard uses the default form (typical training values), so committed
screenshots show no individual trial patient. Needs the dev-only extra:
`pip install -e ".[ui-dev]"` and `python -m playwright install chromium`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = {"desktop": 1280, "phone": 390}
EXAMPLES = {"poorer": "Poorer starting vision", "moderate": "Moderate starting vision",
            "good": "Good starting vision"}
LOAD_TIMEOUT_MS = 180_000


def settle(page: Page, ms: int = 800) -> None:
    """Wait until Streamlit has finished running the script."""
    page.wait_for_timeout(300)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached",
                           timeout=LOAD_TIMEOUT_MS)
    page.wait_for_timeout(ms)


def shot(page: Page, out: Path, name: str, width: int, max_height: int | None = None) -> Path:
    """Full-height screenshot (Streamlit scrolls inside its main container), optionally cut
    at `max_height` pixels."""
    h = page.evaluate("""() => {
        const m = document.querySelector('[data-testid="stMain"]');
        return Math.max(m ? m.scrollHeight : 0, document.body.scrollHeight);
    }""")
    h = min(int(h) + 40, max_height) if max_height else int(h) + 40
    page.set_viewport_size({"width": width, "height": h})
    page.wait_for_timeout(700)
    path = out / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": width, "height": 900})
    return path


def nav(page: Page, base: str, label: str, slug: str, phone: bool) -> None:
    """Go to a page inside the same session (keeps the patient); phone: open the sidebar."""
    if phone:
        btn = page.locator('[data-testid="stExpandSidebarButton"], '
                           '[data-testid="stSidebarCollapsedControl"] button')
        if btn.count():
            btn.first.click()
            page.wait_for_timeout(400)
    link = page.locator('[data-testid="stSidebarNav"] a', has_text=label)
    if link.count():
        link.first.click()
    else:
        page.goto(f"{base}/{slug}")
    settle(page)
    if phone:                       # the sidebar overlays the page on a phone: close it
        close = page.locator('[data-testid="stSidebarCollapseButton"] button')
        if close.count() and close.first.is_visible():
            close.first.click()
            page.wait_for_timeout(400)


def run(base: str, out: Path, example: str | None, dark: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for size, width in WIDTHS.items():
            phone = size == "phone"
            ctx = browser.new_context(viewport={"width": width, "height": 900},
                                      device_scale_factor=1,
                                      color_scheme="dark" if dark else "light")
            page = ctx.new_page()
            page.goto(base)
            page.get_by_text("How it works").wait_for(timeout=LOAD_TIMEOUT_MS)
            settle(page)
            paths.append(shot(page, out, f"home_{size}", width))

            nav(page, base, "Get a suggestion", "suggest", phone)
            page.get_by_text("Step 1 of 3").wait_for(timeout=LOAD_TIMEOUT_MS)
            settle(page)
            if example:
                page.get_by_role("button", name=EXAMPLES[example]).click()
                page.get_by_text("Loaded the example").wait_for()
                settle(page)
            paths.append(shot(page, out, f"suggest_1_details_{size}", width))
            page.get_by_role("button", name="Compare the medicines").click()
            page.get_by_text("Step 2 of 3").wait_for(timeout=LOAD_TIMEOUT_MS)
            settle(page, 1500)
            paths.append(shot(page, out, f"suggest_2_comparison_{size}", width))
            page.get_by_role("button", name="See why").click()
            page.get_by_text("What mattered most").wait_for(timeout=LOAD_TIMEOUT_MS)
            settle(page)
            paths.append(shot(page, out, f"suggest_3_why_{size}", width))

            nav(page, base, "Price check", "price_check", phone)
            paths.append(shot(page, out, f"price_check_{size}", width))

            nav(page, base, "Progress check", "progress", phone)
            page.get_by_role("button", name="Show expected progress").click()
            page.get_by_text("On this path").wait_for(timeout=LOAD_TIMEOUT_MS)
            settle(page, 1500)
            paths.append(shot(page, out, f"progress_{size}", width))

            nav(page, base, "About the trial", "about", phone)
            paths.append(shot(page, out, f"about_{size}", width))

            nav(page, base, "Behind the model", "behind", phone)
            # top part only: the held-out patient picker further down shows patient-level
            # data, which must not end up in committed screenshots
            paths.append(shot(page, out, f"behind_{size}", width, max_height=1500))
            ctx.close()
        browser.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures" / "ui"))
    ap.add_argument("--example", choices=sorted(EXAMPLES), default=None)
    ap.add_argument("--dark", action="store_true", help="emulate a dark colour scheme")
    args = ap.parse_args()
    for path in run(args.url.rstrip("/"), Path(args.out), args.example, args.dark):
        print(path)


if __name__ == "__main__":
    main()
