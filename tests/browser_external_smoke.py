"""Read-only browser checks of saved independent-source measurement results."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path("artifacts/external-ui")
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
        page.goto(base + "/external", wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('#benchmark-body tr').length === 2")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        assert page.locator("#case-select").input_value() == "aluminum"
        assert page.locator("#plot-open").is_visible()
        assert page.locator("#plot-open").get_attribute("href").endswith("case=aluminum")
        assert "实测边界" in page.locator("body").inner_text()
        assert "歧义" in page.locator("#ambiguity-note").inner_text()
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.select_option("#case-select", "brass")
        page.wait_for_function("document.querySelector('#case-title').textContent.includes('黄铜') && !document.querySelector('#download').disabled")
        page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
        with page.expect_download() as event:
            page.click("#download")
        event.value.save_as(str(output / "export.json"))
        exported = json.loads((output / "export.json").read_text())
        assert exported["case"] == "brass"
        assert exported["validation"]["scored_sensor_count"] == 4
        assert exported["protocol"]["future_interior_is_prediction_input"] is False
        assert exported["protocol"]["holdout_selects_model"] is False
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        failed = {"case": "aluminum", "models": {
            "physical": {"validation": {"status": "failed", "failure_reason": "CONTROLLED FAILURE"}},
            "l1": {"validation": {"status": "failed"}},
        }, "validation": {"status": "failed"}, "plot_available": False}
        page.route("**/api/external-experiment?case=aluminum", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(failed),
        ))
        page.select_option("#case-select", "aluminum")
        page.wait_for_function("document.querySelector('#failure-detail').textContent.includes('CONTROLLED FAILURE')")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#plot").get_attribute("src") is None
        assert page.locator("#plot-open").is_hidden()
        assert page.locator("#plot-open").get_attribute("href") is None
        assert page.locator("#rmse").inner_text() == "—"
        page.route("**/api/external-experiment?case=brass", lambda route: route.fulfill(
            status=404, content_type="application/json", body=json.dumps({"detail": "missing fixture"}),
        ))
        page.select_option("#case-select", "brass")
        page.wait_for_function("!document.querySelector('#empty').hidden")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#download").is_disabled()
        assert not errors, errors
        assert all(method == "GET" for method in methods), methods
        browser.close()
    (output / "checks.json").write_text(json.dumps({
        "two_prespecified_conditions": True, "case_switching": True, "json_export": True,
        "ambiguity_and_conditional_prediction_labels": True,
        "mobile_no_document_overflow": True, "failed_report_hides_old_plot": True,
        "missing_report_disables_download": True, "only_get_requests": True,
        "page_errors": errors,
        "full_figure_link_matches_case_and_clears_on_failure": True,
    }, indent=2))
    print("External measurement UI checks passed; read-only, no fitting or scoring.")


if __name__ == "__main__":
    main()
