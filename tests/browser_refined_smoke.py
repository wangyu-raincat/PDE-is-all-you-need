"""Optional local UI checks for saved retrospective results; never trains."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path("artifacts/refined-ui")
    output.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("PDE_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
    errors, methods = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium"),
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: methods.append(request.method))
        page.goto(base + "/refined", wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('#benchmark-body tr').length === 9")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        assert page.locator("#case-select").input_value() == "50ss"
        assert "48.8" in page.locator("#old-error").inner_text()
        assert "回顾性" in page.locator("body").inner_text()
        assert "不是" in page.locator("body").inner_text()
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.select_option("#case-select", "50al")
        page.wait_for_function("document.querySelector('#selected-case').textContent.includes('50al')")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        with page.expect_download() as event:
            page.click("#download")
        event.value.save_as(str(output / "export.json"))
        exported = json.loads((output / "export.json").read_text())
        assert exported["case"] == "50al"
        assert exported["prior_seen_test"] is True
        assert exported["new_blind_validation"] is False
        assert exported["protocol"]["is_l1_optimization"] is False
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        failed = {
            "case": "75al", "experiment": {"title": "Controlled failure fixture"},
            "discovery": {"status": "insufficient", "equation": "没有可验证候选", "terms": []},
            "protocol": {}, "validation": {"status": "failed", "failure_reason": "controlled test"},
            "comparison": {}, "plot_available": False,
            "conclusion": "受控失败路径检查", "limitations": [],
        }
        page.route("**/api/refined-experiment?case=75al", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(failed),
        ))
        page.select_option("#case-select", "75al")
        page.wait_for_function("document.querySelector('#conclusion').textContent.includes('受控失败')")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#plot").get_attribute("src") is None
        page.route("**/api/refined-experiment?case=100al", lambda route: route.fulfill(
            status=404, content_type="application/json", body=json.dumps({"detail": "missing fixture"}),
        ))
        page.select_option("#case-select", "100al")
        page.wait_for_function("!document.querySelector('#empty').hidden")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#download").is_disabled()
        assert not errors, errors
        assert all(method == "GET" for method in methods), methods
        browser.close()
    (output / "checks.json").write_text(json.dumps({
        "nine_cases": True, "case_switching": True, "json_export": True,
        "retrospective_and_physics_prior_labels": True,
        "mobile_no_document_overflow": True, "failed_report_hides_old_plot": True,
        "missing_report_disables_download": True, "only_get_requests": True,
        "page_errors": errors,
    }, indent=2))
    print("Refined UI checks passed; no training or scoring executed.")


if __name__ == "__main__":
    main()
