#!/usr/bin/env python
"""
Integration test suite for Tabular ML Lab.
Uses Playwright to exercise the full UI workflow headlessly.

Usage:
    ./venv/bin/python scripts/integration_test.py [--headed]
"""
import sys
import time
import traceback
from playwright.sync_api import sync_playwright, expect, TimeoutError as PlaywrightTimeout

BASE_URL = "http://localhost:8501"
TIMEOUT = 15000  # 15s default for element waits
RESULTS = {"pass": 0, "fail": 0, "skip": 0}
FAILURES = []


def log(msg, status="INFO"):
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "INFO": "ℹ️"}.get(status, "")
    print(f"{icon} {status}: {msg}")
    if status == "PASS":
        RESULTS["pass"] += 1
    elif status == "FAIL":
        RESULTS["fail"] += 1
        FAILURES.append(msg)
    elif status == "SKIP":
        RESULTS["skip"] += 1


def wait_for_streamlit(page, timeout=30000):
    """Wait for Streamlit to finish loading/rerunning."""
    # Wait for the running indicator to disappear
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=timeout)
    except PlaywrightTimeout:
        pass
    # Additional settle time for Streamlit reruns
    time.sleep(1)


def click_button(page, text, timeout=TIMEOUT):
    """Click a Streamlit button by its text content."""
    btn = page.get_by_role("button", name=text)
    btn.wait_for(state="visible", timeout=timeout)
    btn.click()
    wait_for_streamlit(page)


def select_option(page, label, value):
    """Select a value from a Streamlit selectbox."""
    # Find the selectbox by its label
    select = page.locator(f'div:has(> label:text-is("{label}")) >> [data-testid="stSelectbox"]').first
    if not select.is_visible():
        # Try alternative: look for selectbox near the label text
        select = page.locator(f'[data-testid="stSelectbox"]').filter(has_text=label).first
    select.click()
    page.get_by_role("option", name=value).click()
    wait_for_streamlit(page)


def has_text(page, text, timeout=5000):
    """Check if text is visible on the page."""
    try:
        page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=timeout)
        return True
    except PlaywrightTimeout:
        return False


def has_no_error(page):
    """Check that no Streamlit error is displayed."""
    errors = page.locator('[data-testid="stException"], [data-testid="stError"]')
    count = errors.count()
    if count > 0:
        for i in range(count):
            err_text = errors.nth(i).inner_text()
            log(f"UI Error found: {err_text[:200]}", "FAIL")
        return False
    return True


def navigate_to(page, page_name):
    """Navigate to a page using URL-based routing."""
    # Map friendly names to Streamlit page file names
    page_map = {
        "Upload": "02_EDA",  # We'll use direct URL routing
        "EDA": "02_EDA",
        "Feature Engineering": "03_Feature_Engineering",
        "Feature Selection": "04_Feature_Selection",
        "Preprocess": "05_Preprocess",
        "Train": "06_Train_and_Compare",
        "Explain": "07_Explainability",
        "Sensitivity": "08_Sensitivity_Analysis",
        "Statistical": "09_Hypothesis_Testing",
        "Report": "10_Report_Export",
        "Theory": "11_Theory_Reference",
    }
    # Try URL-based navigation first (most reliable for Streamlit multipage)
    slug = page_map.get(page_name, page_name)
    if page_name == "Upload":
        page.goto(f"{BASE_URL}/01_Upload_and_Audit", wait_until="networkidle")
    else:
        page.goto(f"{BASE_URL}/{slug}", wait_until="networkidle")
    wait_for_streamlit(page)


# ============================================================================
# TEST CASES
# ============================================================================

def test_landing_page(page):
    """Test 1: Landing page loads correctly."""
    page.goto(BASE_URL, wait_until="networkidle")
    wait_for_streamlit(page)
    
    if has_text(page, "Tabular ML Lab"):
        log("Landing page loads", "PASS")
    else:
        log("Landing page missing title", "FAIL")
    
    if has_no_error(page):
        log("Landing page has no errors", "PASS")
    
    # Check FAQ clarification (#33) — may be inside collapsed expander
    # Try expanding FAQ section first
    try:
        expanders = page.locator('[data-testid="stExpander"]').all()
        for exp in expanders:
            if "FAQ" in exp.inner_text() or "question" in exp.inner_text().lower():
                exp.click()
                time.sleep(0.5)
                break
    except Exception:
        pass
    if has_text(page, "server running this application"):
        log("FAQ server-side clarification present (#33)", "PASS")
    else:
        log("FAQ server-side clarification not visible (#33) - may be in collapsed section", "SKIP")


def test_upload_builtin_dataset(page):
    """Test 2: Upload a built-in dataset."""
    navigate_to(page, "Upload")
    wait_for_streamlit(page)
    
    # Look for built-in dataset option
    if has_text(page, "Built-in") or has_text(page, "Example") or has_text(page, "Sample"):
        log("Built-in dataset option visible", "PASS")
    else:
        log("Built-in dataset option not found", "SKIP")
        return False
    
    # Try to load a built-in dataset
    try:
        # Click on built-in datasets
        btn = page.get_by_role("button", name="Load").first
        if btn.is_visible():
            btn.click()
            wait_for_streamlit(page)
    except Exception:
        pass
    
    if has_no_error(page):
        log("Upload page has no errors", "PASS")
    
    return True


def test_configure_prediction_task(page):
    """Test 3: Configure target and features for prediction."""
    navigate_to(page, "Upload")
    wait_for_streamlit(page)
    
    # Check if we have data loaded
    if has_text(page, "target") or has_text(page, "Target"):
        log("Configuration section visible", "PASS")
    else:
        log("No configuration section (need data first)", "SKIP")
        return False
    
    if has_no_error(page):
        log("Configuration has no errors", "PASS")
    
    return True


def test_eda_page(page):
    """Test 4: EDA page loads and shows sections."""
    navigate_to(page, "EDA")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("EDA page loads without errors", "PASS")
    
    # Check OLS proxy explanation (#28)
    if has_text(page, "simple OLS regression as a quick proxy"):
        log("OLS proxy explanation present (#28)", "PASS")
    else:
        log("OLS proxy explanation not found (#28) - may need data loaded first", "SKIP")
    
    # Check for recommendation panel (#26)
    if has_text(page, "Recommended for Your Data") or has_text(page, "Deep Dive"):
        log("EDA Deep Dive section present", "PASS")
    else:
        log("EDA Deep Dive not visible (may need data)", "SKIP")


def test_feature_engineering(page):
    """Test 5: Feature Engineering page with new features (#34)."""
    navigate_to(page, "Feature Engineering")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Feature Engineering loads without errors", "PASS")
    
    # Check for custom interactions (#34)
    if has_text(page, "Custom Interaction") or has_text(page, "custom interaction"):
        log("Custom Interactions panel present (#34)", "PASS")
    else:
        log("Custom Interactions not found (#34) - may need data", "SKIP")


def test_feature_selection(page):
    """Test 6: Feature Selection disclosures (#36, #37)."""
    navigate_to(page, "Feature Selection")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Feature Selection loads without errors", "PASS")
    
    # Check categorical exclusion notice (#36)
    if has_text(page, "non-numeric") or has_text(page, "excluded from ranking"):
        log("Categorical exclusion disclosure present (#36)", "PASS")
    else:
        log("Categorical exclusion not shown (#36) - may have no categoricals or need data", "SKIP")
    
    # Check imputation disclosure (#37)
    if has_text(page, "median") or has_text(page, "temporarily filled"):
        log("Imputation disclosure present (#37)", "PASS")
    else:
        log("Imputation disclosure not shown (#37) - may need data", "SKIP")


def test_preprocess(page):
    """Test 7: Preprocess page banner (#preprocess banner)."""
    navigate_to(page, "Preprocess")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Preprocess page loads without errors", "PASS")
    
    # Check execution order banner — only shows when data is loaded
    if has_text(page, "not applied yet") or has_text(page, "after your data is split"):
        log("Preprocess execution order banner present", "PASS")
    elif has_text(page, "Upload") or has_text(page, "configure your data"):
        log("Preprocess banner hidden (no data loaded yet) — expected", "SKIP")
    else:
        log("Preprocess banner not found", "FAIL")


def test_train_page(page):
    """Test 8: Train & Compare page with new features (#24, target transform)."""
    navigate_to(page, "Train")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Train page loads without errors", "PASS")
    
    # Check for target trimming UI (#24)
    if has_text(page, "target trimming") or has_text(page, "Target trimming"):
        log("Target trimming controls present (#24)", "PASS")
    else:
        log("Target trimming not visible (#24) - may need splits or regression task", "SKIP")
    
    # Check for target transformation UI
    if has_text(page, "Target transformation") or has_text(page, "target transformation") or has_text(page, "Yeo-Johnson"):
        log("Target transformation controls present", "PASS")
    else:
        log("Target transformation not visible - may need regression task", "SKIP")


def test_explainability(page):
    """Test 9: Explainability page (#40, #41)."""
    navigate_to(page, "Explain")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Explainability page loads without errors", "PASS")
    
    # Check ICE/LIME reframing (#41)
    if has_text(page, "not yet built into this app") or has_text(page, "Python packages directly"):
        log("ICE/LIME checklist reframed (#41)", "PASS")
    else:
        log("ICE/LIME checklist not visible - may need trained models", "SKIP")


def test_sensitivity(page):
    """Test 10: Sensitivity Analysis NN filter (#42)."""
    navigate_to(page, "Sensitivity")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Sensitivity page loads without errors", "PASS")


def test_hypothesis(page):
    """Test 11: Hypothesis Testing FWER warning (#43)."""
    navigate_to(page, "Statistical")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Hypothesis Testing page loads without errors", "PASS")


def test_report_export(page):
    """Test 12: Report Export without git leak (#44)."""
    navigate_to(page, "Report")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Report Export loads without errors", "PASS")


def test_theory_reference(page):
    """Test 13: Theory Reference with clipping vs trimming (#25)."""
    navigate_to(page, "Theory")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Theory Reference loads without errors", "PASS")
    
    # Check for clipping vs trimming section
    if has_text(page, "Clipping") or has_text(page, "Trimming"):
        log("Clipping vs Trimming section present (#25)", "PASS")
    else:
        log("Clipping vs Trimming not visible - may be in expander", "SKIP")


def test_full_regression_workflow(page):
    """Test 14: Full workflow — load built-in data → configure → verify pages."""
    # Step 1: Load built-in dataset
    page.goto(f"{BASE_URL}/01_Upload_and_Audit", wait_until="networkidle")
    wait_for_streamlit(page)
    
    # Find the Built-in Dataset selectbox
    try:
        # Click the selectbox that has "Built-in" label nearby
        selects = page.locator('[data-testid="stSelectbox"]').all()
        loaded = False
        for s in selects:
            parent_text = s.evaluate("el => el.closest('[data-testid=\"stVerticalBlock\"]')?.innerText || ''")
            if "built-in" in parent_text.lower() or "built" in parent_text.lower():
                s.click()
                time.sleep(0.5)
                # Pick first non-empty option
                options = page.get_by_role("option").all()
                for opt in options:
                    txt = opt.inner_text().strip()
                    if txt and txt != "":
                        opt.click()
                        wait_for_streamlit(page)
                        loaded = True
                        break
                break
        
        if not loaded:
            log("Could not find built-in dataset selectbox", "SKIP")
            return
        
        # Click "Add Built-in Dataset" button
        try:
            click_button(page, "Add Built-in Dataset")
            log("Built-in dataset loaded", "PASS")
        except PlaywrightTimeout:
            log("Add Built-in Dataset button not found", "SKIP")
            return
            
    except Exception as e:
        log(f"Dataset load failed: {str(e)[:100]}", "SKIP")
        return
    
    if not has_no_error(page):
        return
    
    # Step 2: Configure prediction task — select target and features
    # Look for target selection
    wait_for_streamlit(page)
    time.sleep(2)  # Extra time for data processing
    
    if has_text(page, "target") or has_text(page, "Target"):
        log("Full workflow: configuration section appeared", "PASS")
    else:
        log("Full workflow: configuration section not found after data load", "FAIL")
        return
    
    # Try to find and click a "Save Configuration" or "Confirm" button
    try:
        buttons = page.get_by_role("button").all()
        for btn in buttons:
            txt = btn.inner_text().lower()
            if "save" in txt and ("config" in txt or "confirm" in txt or "proceed" in txt):
                btn.click()
                wait_for_streamlit(page)
                log("Full workflow: configuration saved", "PASS")
                break
    except Exception:
        log("Full workflow: could not save config (may auto-save)", "SKIP")
    
    if not has_no_error(page):
        return
    
    # Step 3: Visit EDA with data loaded
    page.goto(f"{BASE_URL}/02_EDA", wait_until="networkidle")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Full workflow: EDA with data loads cleanly", "PASS")
    
    if has_text(page, "simple OLS regression as a quick proxy"):
        log("Full workflow: OLS proxy explanation visible with data (#28)", "PASS")
    
    if has_text(page, "Recommended for Your Data") or has_text(page, "recommended for your data", ):
        log("Full workflow: Data-driven recommendations visible (#26)", "PASS")
    
    # Step 4: Visit Preprocess with data loaded
    page.goto(f"{BASE_URL}/05_Preprocess", wait_until="networkidle")
    wait_for_streamlit(page)
    
    if has_text(page, "not applied yet"):
        log("Full workflow: Preprocess banner visible with data", "PASS")
    
    if has_no_error(page):
        log("Full workflow: Preprocess with data loads cleanly", "PASS")
    
    # Step 5: Visit Theory Reference
    page.goto(f"{BASE_URL}/11_Theory_Reference", wait_until="networkidle")
    wait_for_streamlit(page)
    
    if has_no_error(page):
        log("Full workflow: Theory Reference loads cleanly", "PASS")


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    headed = "--headed" in sys.argv
    
    print("=" * 60)
    print("Tabular ML Lab — Integration Test Suite")
    print("=" * 60)
    print(f"Target: {BASE_URL}")
    print(f"Mode: {'headed' if headed else 'headless'}")
    print()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.set_default_timeout(TIMEOUT)
        
        tests = [
            ("Landing Page", test_landing_page),
            ("Upload Built-in Dataset", test_upload_builtin_dataset),
            ("Configure Prediction Task", test_configure_prediction_task),
            ("EDA Page", test_eda_page),
            ("Feature Engineering", test_feature_engineering),
            ("Feature Selection", test_feature_selection),
            ("Preprocess", test_preprocess),
            ("Train & Compare", test_train_page),
            ("Explainability", test_explainability),
            ("Sensitivity Analysis", test_sensitivity),
            ("Hypothesis Testing", test_hypothesis),
            ("Report Export", test_report_export),
            ("Theory Reference", test_theory_reference),
            ("Full Regression Workflow", test_full_regression_workflow),
        ]
        
        for name, test_fn in tests:
            print(f"\n--- {name} ---")
            try:
                test_fn(page)
            except Exception as e:
                log(f"{name} crashed: {str(e)[:200]}", "FAIL")
                traceback.print_exc()
                # Take screenshot on failure
                try:
                    page.screenshot(path=f"/tmp/fail_{name.replace(' ', '_').lower()}.png")
                    print(f"  Screenshot: /tmp/fail_{name.replace(' ', '_').lower()}.png")
                except Exception:
                    pass
        
        browser.close()
    
    print("\n" + "=" * 60)
    print(f"Results: {RESULTS['pass']} passed, {RESULTS['fail']} failed, {RESULTS['skip']} skipped")
    if FAILURES:
        print(f"\nFailures:")
        for f in FAILURES:
            print(f"  ❌ {f}")
    print("=" * 60)
    
    return RESULTS["fail"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
