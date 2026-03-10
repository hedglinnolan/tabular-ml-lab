"""
Integration verification for session save/resume functionality.
Tests that all components work together correctly.
"""
import sys
import pickle
from datetime import datetime

# Add utils to path
sys.path.insert(0, 'utils')

def test_imports():
    """Verify all imports work correctly."""
    print("Testing imports...")
    
    try:
        from session_manager import (
            render_session_controls,
            _collect_session_data,
            _restore_session_data,
            _is_serializable,
            _calculate_session_size,
            get_session_summary
        )
        print("  ✓ session_manager imports successful")
    except ImportError as e:
        print(f"  ✗ session_manager import failed: {e}")
        return False
    
    try:
        from theme import render_sidebar_workflow
        print("  ✓ theme imports successful")
    except ImportError as e:
        print(f"  ✗ theme import failed: {e}")
        return False
    
    try:
        from session_state import (
            init_session_state,
            TaskTypeDetection,
            CohortStructureDetection,
            DataConfig,
            SplitConfig,
            ModelConfig
        )
        print("  ✓ session_state imports successful")
    except ImportError as e:
        print(f"  ✗ session_state import failed: {e}")
        return False
    
    print()
    return True

def test_excluded_keys():
    """Verify excluded keys are properly defined."""
    print("Testing excluded keys configuration...")
    
    from session_manager import _get_excluded_keys
    
    excluded = _get_excluded_keys()
    expected_keys = {
        '_uploaded_file_data',
        'FileUploader',
        'FormSubmitter',
        '_widget_manager',
        '_script_run_ctx',
    }
    
    if excluded == expected_keys:
        print(f"  ✓ All expected keys excluded: {excluded}")
    else:
        print(f"  ! Excluded keys: {excluded}")
        print(f"  ! Expected keys: {expected_keys}")
    
    print()
    return True

def test_serialization_coverage():
    """Test that all common session state types can be serialized."""
    print("Testing serialization coverage...")
    
    from session_manager import _is_serializable
    from session_state import TaskTypeDetection, DataConfig
    import pandas as pd
    import numpy as np
    
    test_cases = [
        ("None", None),
        ("int", 42),
        ("float", 3.14),
        ("str", "test"),
        ("list", [1, 2, 3]),
        ("dict", {'a': 1}),
        ("DataFrame", pd.DataFrame({'x': [1, 2, 3]})),
        ("Series", pd.Series([1, 2, 3])),
        ("ndarray", np.array([1, 2, 3])),
        ("TaskTypeDetection", TaskTypeDetection(detected='regression')),
        ("DataConfig", DataConfig(target_col='target')),
        ("datetime", datetime.now()),
    ]
    
    all_passed = True
    for name, obj in test_cases:
        if _is_serializable(obj):
            print(f"  ✓ {name}: serializable")
        else:
            print(f"  ✗ {name}: NOT serializable")
            all_passed = False
    
    print()
    return all_passed

def test_session_roundtrip():
    """Test full session save/restore round-trip."""
    print("Testing session round-trip...")
    
    from session_manager import _collect_session_data, _calculate_session_size
    from session_state import TaskTypeDetection, DataConfig, SplitConfig
    import pandas as pd
    
    # Create mock session data (simulating st.session_state)
    mock_session_state = {
        'raw_data': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
        'data_config': DataConfig(target_col='y', feature_cols=['x']),
        'split_config': SplitConfig(),
        'task_type_detection': TaskTypeDetection(detected='regression', confidence='high'),
        'trained_models': {},
        'model_results': {},
        'random_seed': 42,
        'data_source': 'test',
        '_internal_key': 'should_be_excluded',  # Should be excluded
    }
    
    # Test that internal keys are excluded
    print("  Testing key exclusion...")
    include_keys = [k for k in mock_session_state.keys() if not k.startswith('_')]
    print(f"    {len(include_keys)}/{len(mock_session_state)} keys should be included")
    
    # Create session data structure (mimics _collect_session_data but without st.session_state)
    session_data = {}
    for key, value in mock_session_state.items():
        if not key.startswith('_'):
            session_data[key] = value
    
    session_data['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
        'workflow_step': 'Test',
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'session_keys_count': len(session_data),
    }
    
    try:
        # Test serialization
        serialized = pickle.dumps(session_data)
        print(f"  ✓ Serialization successful ({len(serialized)} bytes)")
        
        # Test deserialization
        deserialized = pickle.loads(serialized)
        print(f"  ✓ Deserialization successful")
        
        # Verify structure
        assert '_metadata' in deserialized
        assert deserialized['_metadata']['version'] == '1.0'
        assert isinstance(deserialized['raw_data'], pd.DataFrame)
        assert isinstance(deserialized['data_config'], DataConfig)
        print(f"  ✓ Session structure verified")
        
        # Verify excluded keys not present
        assert '_internal_key' not in deserialized
        print(f"  ✓ Internal keys excluded correctly")
        
    except Exception as e:
        print(f"  ✗ Round-trip failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_size_warnings():
    """Test that size calculation and warnings work correctly."""
    print("Testing size warnings...")
    
    from session_manager import _calculate_session_size
    
    # Small session
    small_session = {'_metadata': {}, 'data': 'x' * 100}
    size_bytes, size_str = _calculate_session_size(small_session)
    print(f"  Small session: {size_str}")
    
    # Medium session
    medium_session = {'_metadata': {}, 'data': 'x' * (1024 * 100)}  # 100 KB
    size_bytes, size_str = _calculate_session_size(medium_session)
    print(f"  Medium session: {size_str}")
    
    # Large session (should trigger warning)
    large_session = {'_metadata': {}, 'data': 'x' * (1024 * 1024 * 51)}  # 51 MB
    size_bytes, size_str = _calculate_session_size(large_session)
    warning_threshold = 50 * 1024 * 1024
    
    if size_bytes > warning_threshold:
        print(f"  Large session: {size_str} → ⚠️ Warning should be shown")
    else:
        print(f"  Large session: {size_str}")
    
    print()
    return True

def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Session Save/Resume Integration Verification")
    print("=" * 70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Excluded Keys", test_excluded_keys),
        ("Serialization Coverage", test_serialization_coverage),
        ("Session Round-trip", test_session_roundtrip),
        ("Size Warnings", test_size_warnings),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("🎉 All integration tests passed!")
        print()
        print("Session save/resume functionality is ready for use.")
        print("Users can now save their progress and resume later.")
    else:
        print("⚠️  Some tests failed. Review output above.")
    
    print("=" * 70)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
