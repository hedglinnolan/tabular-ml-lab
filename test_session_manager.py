"""
Quick test of session manager functionality.
"""
import pickle
import sys
from datetime import datetime

# Add utils to path
sys.path.insert(0, 'utils')

from session_manager import _get_excluded_keys, _is_serializable, _collect_session_data, _calculate_session_size

def test_serialization():
    """Test that basic Python objects serialize correctly."""
    print("Testing serialization...")
    
    test_objects = {
        'int': 42,
        'float': 3.14,
        'str': 'hello',
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2},
        'nested': {'list': [1, 2], 'dict': {'x': 10}},
        'datetime': datetime.now(),
    }
    
    for name, obj in test_objects.items():
        if _is_serializable(obj):
            print(f"  ✓ {name}: serializable")
        else:
            print(f"  ✗ {name}: NOT serializable")
    
    print()

def test_session_data_structure():
    """Test that session data structure is valid."""
    print("Testing session data structure...")
    
    # Create mock session data
    mock_session = {
        'raw_data': None,
        'data_config': {'target_col': 'target'},
        'trained_models': {},
        'preprocessing_pipeline': None,
    }
    
    # Add metadata
    mock_session['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
        'workflow_step': 'Test',
        'skipped_keys': [],
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'session_keys_count': len(mock_session) - 1,
    }
    
    # Test serialization
    try:
        serialized = pickle.dumps(mock_session)
        print(f"  ✓ Serialization successful")
        print(f"  Size: {len(serialized)} bytes")
        
        # Test deserialization
        deserialized = pickle.loads(serialized)
        print(f"  ✓ Deserialization successful")
        print(f"  Metadata: {deserialized['_metadata']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

def test_size_calculation():
    """Test size calculation for different session sizes."""
    print("Testing size calculation...")
    
    test_sizes = [
        100,  # 100 B
        1024,  # 1 KB
        1024 * 1024,  # 1 MB
        50 * 1024 * 1024,  # 50 MB
    ]
    
    for size_bytes in test_sizes:
        mock_data = {'_metadata': {}, 'data': 'x' * size_bytes}
        size_bytes_calc, size_str = _calculate_session_size(mock_data)
        print(f"  {size_bytes:,} bytes → {size_str}")
    
    print()

def test_excluded_keys():
    """Test that excluded keys are properly defined."""
    print("Testing excluded keys...")
    
    excluded = _get_excluded_keys()
    print(f"  Excluded keys: {excluded}")
    print(f"  Count: {len(excluded)}")
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Session Manager Test Suite")
    print("=" * 60)
    print()
    
    test_serialization()
    test_session_data_structure()
    test_size_calculation()
    test_excluded_keys()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
