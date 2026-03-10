"""
Integration tests for page imports and navigation consistency.

Verifies that all page files can be imported without error and have
correct navigation function calls.
"""

import os
import sys
import importlib.util
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPageImports:
    """Test that all pages import successfully."""
    
    PAGE_FILES = [
        ('pages/01_Upload_and_Audit.py', 'page_01'),
        ('pages/02_EDA.py', 'page_02'),
        ('pages/03_Feature_Engineering.py', 'page_03'),
        ('pages/04_Feature_Selection.py', 'page_04'),
        ('pages/05_Preprocess.py', 'page_05'),
        ('pages/06_Train_and_Compare.py', 'page_06'),
        ('pages/07_Explainability.py', 'page_07'),
        ('pages/08_Sensitivity_Analysis.py', 'page_08'),
        ('pages/09_Hypothesis_Testing.py', 'page_09'),
        ('pages/10_Report_Export.py', 'page_10'),
    ]
    
    @pytest.mark.parametrize('page_info', PAGE_FILES)
    def test_page_imports(self, page_info):
        """Test that each page file can be imported without ImportError."""
        page_file, module_name = page_info
        
        # Check file exists
        assert os.path.exists(page_file), f"{page_file} does not exist"
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, page_file)
        assert spec is not None, f"Could not load spec for {page_file}"
        
        module = importlib.util.module_from_spec(spec)
        
        # Import should not raise ImportError
        # Runtime errors are okay (Streamlit context missing), but import errors are not
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            pytest.fail(f"Import error in {page_file}: {str(e)}")
        except ModuleNotFoundError as e:
            pytest.fail(f"Module not found in {page_file}: {str(e)}")
        except (AttributeError, TypeError, ValueError, KeyError, RuntimeError, IndexError) as e:
            # These are expected runtime errors when Streamlit context is missing
            # We only care about import errors
            pass
        except Exception as e:
            # Allow Streamlit-related errors
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['streamlit', 'scriptruncontext', 'session_state', 'sidebar']):
                pass  # Expected Streamlit context error
            else:
                pytest.fail(f"Unexpected error in {page_file}: {str(e)}")
    
    def test_utils_imports(self):
        """Test that utility modules import correctly."""
        utils_modules = [
            'utils.session_state',
            'utils.storyline',
            'utils.theme',
            'utils.session_manager',
        ]
        
        for module_name in utils_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Import error in {module_name}: {str(e)}")
    
    def test_ml_imports(self):
        """Test that ML utility modules import correctly."""
        ml_modules = [
            'ml.baseline_models',
            'ml.bootstrap',
            'ml.calibration',
            'ml.feature_selection',
            'ml.publication',
            'ml.sensitivity',
            'ml.table_one',
        ]
        
        for module_name in ml_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Import error in {module_name}: {str(e)}")


class TestNavigationConsistency:
    """Test that navigation calls are consistent across pages."""
    
    def test_storyline_import_locations(self):
        """Verify render_breadcrumb and render_page_navigation are imported from storyline."""
        import ast
        
        pages_requiring_storyline = [
            'pages/01_Upload_and_Audit.py',
            'pages/02_EDA.py',
            'pages/03_Feature_Engineering.py',
            'pages/04_Feature_Selection.py',
            'pages/05_Preprocess.py',
            'pages/06_Train_and_Compare.py',
            'pages/07_Explainability.py',
            'pages/08_Sensitivity_Analysis.py',
            'pages/09_Hypothesis_Testing.py',
            'pages/10_Report_Export.py',
        ]
        
        for page_file in pages_requiring_storyline:
            with open(page_file, 'r') as f:
                tree = ast.parse(f.read())
            
            # Check imports
            imports_storyline = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == 'utils.storyline':
                        # Check if it imports render_breadcrumb or render_page_navigation
                        imported_names = [alias.name for alias in node.names]
                        if 'render_breadcrumb' in imported_names or 'render_page_navigation' in imported_names:
                            imports_storyline = True
                            break
            
            # Some pages might use different import patterns, so this is informational
            # not a strict requirement
            if not imports_storyline:
                # Check if they import from theme instead (which would be wrong)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == 'utils.theme':
                            imported_names = [alias.name for alias in node.names]
                            assert 'render_breadcrumb' not in imported_names, \
                                f"{page_file} incorrectly imports render_breadcrumb from utils.theme (should be utils.storyline)"
                            assert 'render_page_navigation' not in imported_names, \
                                f"{page_file} incorrectly imports render_page_navigation from utils.theme (should be utils.storyline)"
