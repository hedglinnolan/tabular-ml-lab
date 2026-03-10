"""
Test session manager with realistic session state objects.
"""
import pickle
import sys
import pandas as pd
import numpy as np

# Add utils to path
sys.path.insert(0, 'utils')

from session_state import TaskTypeDetection, CohortStructureDetection, DataConfig, SplitConfig, ModelConfig
from session_manager import _is_serializable, _calculate_session_size

def test_dataclass_serialization():
    """Test that session_state dataclasses serialize correctly."""
    print("Testing dataclass serialization...")
    
    # Create instances of each dataclass
    task_type = TaskTypeDetection(
        detected='regression',
        confidence='high',
        reasons=['Target is continuous'],
        override_enabled=False,
        override_value=None
    )
    
    cohort = CohortStructureDetection(
        detected='cross_sectional',
        confidence='high',
        reasons=['No repeated measures detected'],
        entity_id_candidates=['patient_id'],
        entity_id_detected='patient_id'
    )
    
    data_config = DataConfig(
        target_col='glucose',
        feature_cols=['age', 'bmi', 'bp'],
        datetime_col=None,
        task_type='regression'
    )
    
    split_config = SplitConfig(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        stratify=False
    )
    
    model_config = ModelConfig(
        nn_epochs=200,
        nn_batch_size=256,
        nn_lr=0.0015
    )
    
    test_objects = {
        'task_type_detection': task_type,
        'cohort_structure_detection': cohort,
        'data_config': data_config,
        'split_config': split_config,
        'model_config': model_config,
    }
    
    for name, obj in test_objects.items():
        if _is_serializable(obj):
            print(f"  ✓ {name}: serializable")
            # Test round-trip
            try:
                serialized = pickle.dumps(obj)
                deserialized = pickle.loads(serialized)
                print(f"    Round-trip successful, size: {len(serialized)} bytes")
            except Exception as e:
                print(f"    ✗ Round-trip failed: {e}")
        else:
            print(f"  ✗ {name}: NOT serializable")
    
    print()

def test_realistic_session():
    """Test a realistic session state with multiple components."""
    print("Testing realistic session state...")
    
    # Create mock session with realistic data
    mock_session = {
        # Basic data
        'raw_data': pd.DataFrame({
            'glucose': [100, 120, 90, 110],
            'age': [25, 30, 35, 40],
            'bmi': [22.5, 25.0, 27.5, 30.0]
        }),
        
        # Configs
        'data_config': DataConfig(
            target_col='glucose',
            feature_cols=['age', 'bmi'],
            task_type='regression'
        ),
        'split_config': SplitConfig(),
        'model_config': ModelConfig(),
        'task_type_detection': TaskTypeDetection(detected='regression', confidence='high'),
        'cohort_structure_detection': CohortStructureDetection(detected='cross_sectional'),
        
        # Training data
        'X_train': pd.DataFrame({'age': [25, 30], 'bmi': [22.5, 25.0]}),
        'y_train': pd.Series([100, 120]),
        'feature_names': ['age', 'bmi'],
        
        # Models (empty dicts for now)
        'trained_models': {},
        'model_results': {},
        'fitted_estimators': {},
        
        # Misc
        'random_seed': 42,
        'data_source': 'uploaded',
        'data_filename': 'test_data.csv',
    }
    
    # Add metadata
    from datetime import datetime
    mock_session['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
        'workflow_step': '05_Train_and_Compare',
        'skipped_keys': [],
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'session_keys_count': len(mock_session) - 1,
    }
    
    # Test serialization
    try:
        size_bytes, size_str = _calculate_session_size(mock_session)
        print(f"  ✓ Session serialization successful")
        print(f"  Size: {size_str} ({size_bytes:,} bytes)")
        print(f"  Keys: {len(mock_session) - 1} (+ metadata)")
        
        # Test deserialization
        serialized = pickle.dumps(mock_session)
        deserialized = pickle.loads(serialized)
        
        # Verify key data types
        assert isinstance(deserialized['raw_data'], pd.DataFrame)
        assert isinstance(deserialized['data_config'], DataConfig)
        assert isinstance(deserialized['X_train'], pd.DataFrame)
        assert isinstance(deserialized['y_train'], pd.Series)
        
        print(f"  ✓ Deserialization successful")
        print(f"  ✓ Data types preserved correctly")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Realistic Session Manager Tests")
    print("=" * 60)
    print()
    
    test_dataclass_serialization()
    test_realistic_session()
    
    print("=" * 60)
    print("All realistic tests completed!")
    print("=" * 60)
