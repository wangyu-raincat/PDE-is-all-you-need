"""Optional UI acceptance test; requires Playwright and Chromium.

Start the app at localhost:8765, then run this file. Use CHROMIUM_PATH to point
to an existing browser, or install Playwright's browser normally.
"""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path(__file__).resolve().parents[1] / "artifacts"
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        launch = {"headless": True}
        if os.environ.get("CHROMIUM_PATH"):
            launch["executable_path"] = os.environ["CHROMIUM_PATH"]
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://127.0.0.1:8765", wait_until="networkidle")
        page.locator("input[name=scenario][value=ambiguity]").check()
        page.locator("#noise").select_option("0.01")
        page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('status-pill').textContent === '已识别候选方程'", timeout=30000)
        page.screenshot(path=str(output / "demo-desktop.png"), full_page=True)
        print(json.dumps({"status": page.locator("#status-pill").inner_text(),
                          "equation": page.locator("#equation").inner_text(),
                          "error": page.locator("#validation-error").inner_text(),
                          "probe": page.locator("#probe-scores").inner_text()}, ensure_ascii=False))
        page.locator("#auto-experiment").uncheck()
        page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('status-pill').textContent === '存在多种解释'", timeout=30000)
        assert page.locator("#candidate-panel").is_visible()
        assert page.locator("#goal-banner").is_visible()
        page.screenshot(path=str(output / "demo-ambiguity.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "demo-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        page.locator("#upload-tab").click()
        page.locator("#data-text").fill("{invalid")
        page.locator("#run-button").click()
        assert page.locator("#error-banner").is_visible()
        bundle = page.request.get("http://127.0.0.1:8765/api/example?kind=bundle").json()
        page.locator("#data-text").fill(json.dumps(bundle))
        page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('status-pill').textContent === '已识别候选方程'", timeout=30000)
        with page.expect_download() as info:
            page.locator("#download-button").click()
        info.value.save_as(str(output / "browser-export.json"))
        assert not errors, errors
        print("Browser checks passed: ambiguity, independent experiment, upload, errors, download, mobile layout; no JS errors.")
        browser.close()


if __name__ == "__main__":
    main()
