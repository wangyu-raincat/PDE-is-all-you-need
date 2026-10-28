"""Read-only acceptance of saved cross-equation discovery results."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    output = Path("artifacts/generic-ui")
    output.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("PDE_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
    errors, methods = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium"),
                                   headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: methods.append(request.method))
        page.goto(base + "/discovery", wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('#benchmark-body tr').length === 3")
        page.wait_for_function("!document.querySelector('#download').disabled")
        assert "不是真实测量" in page.locator("body").inner_text()
        assert page.locator("#exact-support").inner_text() != "—"
        assert page.locator("#coefficient-error").inner_text() != "—"
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.select_option("#case-select", "kdv")
        page.wait_for_function("document.querySelector('#report-id').textContent.includes('KdV') && !document.querySelector('#download').disabled")
        assert page.locator("#boundary-note").is_visible()
        assert "实现假设" in page.locator("#boundary-note").inner_text()
        with page.expect_download() as event:
            page.click("#download")
        event.value.save_as(str(output / "export.json"))
        report = json.loads((output / "export.json").read_text())
        assert report["case"] == "kdv"
        assert report["protocol"]["reference_equations_used_for_selection"] is False
        assert report["protocol"]["future_field_or_boundary_input"] is False
        if report.get("plot_available"):
            page.wait_for_function("document.querySelector('#plot').naturalWidth > 0 && !document.querySelector('#plot').hidden")
            assert page.locator("#plot-open").is_visible()
            assert page.locator("#plot-open").get_attribute("href").endswith("case=kdv")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        failed = {"case": "ks", "discovery": {"status": "insufficient"},
                  "validation": {"status": "failed", "failure_reason": "CONTROLLED FAILURE"},
                  "equation_recovery": {"status": "insufficient"}, "plot_available": False}
        page.route("**/api/generic-experiment?case=ks", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(failed)))
        page.select_option("#case-select", "ks")
        page.wait_for_function("document.querySelector('#failure-detail').textContent.includes('CONTROLLED FAILURE')")
        assert page.locator("#plot").is_hidden()
        assert page.locator("#plot").get_attribute("src") is None
        assert page.locator("#plot-open").is_hidden()
        assert page.locator("#plot-open").get_attribute("href") is None
        assert page.locator("#coefficient-error").inner_text() == "—"
        assert page.locator("#exact-support").inner_text() == "未评分"
        page.route("**/api/generic-experiment?case=burgers", lambda route: route.fulfill(
            status=404, content_type="application/json", body=json.dumps({"detail": "missing fixture"})))
        page.select_option("#case-select", "burgers")
        page.wait_for_function("!document.querySelector('#empty').hidden")
        assert page.locator("#download").is_disabled()
        assert page.locator("#plot").is_hidden()
        assert not errors, errors
        assert all(method == "GET" for method in methods), methods
        browser.close()
    (output / "checks.json").write_text(json.dumps({
        "three_prespecified_equations": True, "structure_and_coefficient_primary_metrics": True,
        "simulation_and_kdv_assumption_labels": True, "case_switching": True,
        "json_export": True, "full_figure_link": True, "mobile_no_document_overflow": True,
        "failure_and_404_clear_stale_results": True, "only_get": True, "page_errors": errors,
    }, indent=2))
    print("Generic discovery UI checks passed; only saved reports were read.")


if __name__ == "__main__":
    main()
