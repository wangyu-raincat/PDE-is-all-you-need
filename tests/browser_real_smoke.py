"""Optional local UI checks; reads saved results, never refits a holdout."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium"),
                                    headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto("http://127.0.0.1:8765/real", wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('#benchmark-body tr').length === 9")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        assert page.locator("#case-select").input_value() == "50ss"
        assert "48.8" in page.locator("#relative").inner_text()
        assert "4/9" in page.locator("#benchmark-summary").inner_text()
        page.screenshot(path=str(output / "measured-desktop.png"), full_page=True)
        page.select_option("#case-select", "50al")
        page.wait_for_function("document.querySelector('#source-title').textContent.includes('50al')")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        evidence = page.locator("#evidence-note").inner_text()
        assert "5 项全部保留" in evidence and "未找到简洁" in evidence
        with page.expect_download() as event:
            page.click("#download")
        event.value.save_as(str(output / "measured-browser-export.json"))
        exported = json.loads((output / "measured-browser-export.json").read_text())
        assert exported["validation"]["relative_l2"] == 0.012465374450470534
        assert any("/plot?" in url and "case=50al" in url for url in requests)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "measured-mobile.png"), full_page=True)
        # Current failed report must not show the previous case's successful image.
        failed = {"experiment": {"title": "Controlled UI failure fixture"},
                  "discovery": {"status": "insufficient", "equation": "没有可验证候选", "terms": []},
                  "protocol": {}, "validation": {"status": "failed", "failure_reason": "controlled test"},
                  "plot_available": False, "conclusion": "受控失败路径检查", "limitations": []}
        def intercept(route):
            if route.request.method == "POST":
                route.fulfill(status=422, content_type="application/json", body=json.dumps({"detail": "受控复现错误"}))
            else:
                route.fulfill(status=200, content_type="application/json", body=json.dumps(failed))
        page.route("**/api/real-experiment?case=75al", intercept)
        page.select_option("#case-select", "75al")
        page.wait_for_function("document.querySelector('#conclusion').textContent.includes('受控失败')")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#plot").get_attribute("src") is None
        assert "本次报告没有可用验证图" in page.locator("#plot-empty").inner_text()
        page.click("#reproduce")
        page.wait_for_function("!document.querySelector('#error').hidden")
        assert "受控复现错误" in page.locator("#error").inner_text()
        assert page.locator("#download").is_disabled()
        assert not errors, errors
        browser.close()
    (output / "measured-browser-checks.json").write_text(json.dumps({
        "nine_cases": True, "case_switching": True, "json_export": True,
        "mobile_no_document_overflow": True, "failed_report_hides_old_image": True,
        "post_error_visible": True, "page_errors": errors,
        "real_training_requests_executed": False}, indent=2))
    print("Real-data browser checks passed; no training rerun.")


if __name__ == "__main__":
    main()
