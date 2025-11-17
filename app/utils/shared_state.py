"""
Shared State Management
Manages shared data across pages using Streamlit session state
"""

import streamlit as st
import pandas as pd
from pathlib import Path


def get_shared_dataset():
    """
    Get the currently active dataset from session state.
    This ensures all pages use the same dataset.
    
    Returns:
        pd.DataFrame or None: The active dataset
    """
    # Check if Data Explorer has uploaded data
    if 'data_explorer_state' in st.session_state:
        explorer_state = st.session_state['data_explorer_state']
        
        # If user uploaded data in Data Explorer
        if explorer_state.get('source') == 'uploaded' and explorer_state.get('uploaded_bytes'):
            try:
                import io
                df = pd.read_csv(io.BytesIO(explorer_state['uploaded_bytes']))
                return df
            except Exception:
                pass
    
    # Check if there's a global shared dataset
    if 'shared_dataset' in st.session_state:
        return st.session_state['shared_dataset']
    
    # Fallback to default dataset
    return load_default_dataset()


def set_shared_dataset(df, source='default'):
    """
    Set the shared dataset for all pages.
    
    Args:
        df (pd.DataFrame): Dataset to share
        source (str): Source of the dataset ('uploaded', 'default', etc.)
    """
    st.session_state['shared_dataset'] = df
    st.session_state['shared_dataset_source'] = source


def get_dataset_info():
    """
    Get information about the current dataset.
    
    Returns:
        dict: Dataset information
    """
    df = get_shared_dataset()
    
    if df is None:
        return {
            'loaded': False,
            'source': 'none',
            'rows': 0,
            'columns': 0
        }
    
    # Determine source
    source = 'default'
    if 'data_explorer_state' in st.session_state:
        explorer_state = st.session_state['data_explorer_state']
        if explorer_state.get('source') == 'uploaded':
            source = f"uploaded: {explorer_state.get('uploaded_name', 'custom')}"
    
    return {
        'loaded': True,
        'source': source,
        'rows': len(df),
        'columns': len(df.columns),
        'has_qa': 'has_quick_assessment' in df.columns,
        'qa_coverage': df['has_quick_assessment'].sum() / len(df) * 100 if 'has_quick_assessment' in df.columns else 0,
        'promotion_rate': df['has_promotion'].sum() / len(df) * 100 if 'has_promotion' in df.columns else 0
    }


def load_default_dataset():
    """
    Load the default dataset.
    
    Returns:
        pd.DataFrame or None: Default dataset
    """
    repo_root = Path(__file__).resolve().parents[2]
    
    # Priority order for default datasets
    dataset_paths = [
        repo_root / "data" / "final" / "sample_dataset_100_balanced.csv",
        repo_root / "data" / "final" / "integrated_full_dataset.csv",
        repo_root / "data" / "final" / "sample_dataset_100.csv",
        repo_root / "data" / "final" / "integrated_performance_behavioral.csv",
    ]
    
    for path in dataset_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    
    return None


def clear_shared_dataset():
    """Clear the shared dataset and reset to default."""
    if 'shared_dataset' in st.session_state:
        del st.session_state['shared_dataset']
    if 'shared_dataset_source' in st.session_state:
        del st.session_state['shared_dataset_source']
