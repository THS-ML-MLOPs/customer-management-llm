"""
================================================================================
    PATHS CONFIGURATION
    Centralized path management for Customer Management System
    Works on both Local Windows and Google Colab
================================================================================
"""

from pathlib import Path
import os

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def detect_environment():
    """Detect if running on Colab or local"""
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

def get_project_root():
    """Get project root path based on environment"""
    env = detect_environment()

    if env == 'colab':
        return Path("/content/drive/MyDrive/ML_Projects/Customer_Management")
    else:
        # Local: Navigate from ui_files/config to project root
        # ui_files/config/paths.py -> ui_files -> project_root
        current_file = Path(__file__)
        return current_file.parent.parent.parent

# Base path (auto-detected)
BASE_PATH = get_project_root()

# All project paths
PATHS = {
    # Root directories
    'base': BASE_PATH,
    'data': BASE_PATH / 'data',
    'models': BASE_PATH / 'models',
    'dashboards': BASE_PATH / 'dashboards',
    'llm_workspace': BASE_PATH / 'llm_workspace',
    'ui_files': BASE_PATH / 'ui_files',
    
    # Data subdirectories
    'data_processed': BASE_PATH / 'data' / 'processed',
    'data_features': BASE_PATH / 'data' / 'features',
    'transactions': BASE_PATH / 'data' / 'processed' / 'transactions_enriched.parquet',
    
    # Dashboard directories
    'churn_dashboard_dir': BASE_PATH / 'dashboards' / '01_churn_risk',
    'churn_dashboard': BASE_PATH / 'dashboards' / '01_churn_risk' / 'dashboard_complete_churn_risk.html',
    
    # Individual churn charts (all in same folder)
    'churn_chart_01': BASE_PATH / 'dashboards' / '01_churn_risk' / '01_churn_distribution_stacked_dark.html',
    'churn_chart_02': BASE_PATH / 'dashboards' / '01_churn_risk' / '02_churn_gauge_at_risk_dark.html',
    'churn_chart_03': BASE_PATH / 'dashboards' / '01_churn_risk' / '03_churn_sunburst_value_at_risk_dark.html',
    'churn_chart_04': BASE_PATH / 'dashboards' / '01_churn_risk' / '04_churn_score_histogram_dark.html',
    'churn_chart_05': BASE_PATH / 'dashboards' / '01_churn_risk' / '05_churn_heatmap_category_dark.html',
    'churn_chart_06': BASE_PATH / 'dashboards' / '01_churn_risk' / '06_churn_top_subcategories_dark.html',
    'churn_chart_07': BASE_PATH / 'dashboards' / '01_churn_risk' / '07_churn_price_scatter_dark.html',
    'churn_chart_08': BASE_PATH / 'dashboards' / '01_churn_risk' / '08_churn_discount_impact_dark.html',
    
    # LLM workspace
    'conversations': BASE_PATH / 'llm_workspace' / 'conversations',
    'charts': BASE_PATH / 'llm_workspace' / 'charts',
    'system_prompt': BASE_PATH / 'llm_workspace' / 'system_prompt.txt',
    'feedback_db': BASE_PATH / 'llm_workspace' / 'feedback.db',
    'logs': BASE_PATH / 'llm_workspace' / 'logs',
}

def verify_paths():
    """
    Verify that critical paths exist
    Returns dict with status of each critical path
    """
    critical_paths = [
        'base',
        'data',
        'dashboards',
        'churn_dashboard',
        'llm_workspace'
    ]
    
    status = {}
    for key in critical_paths:
        path = PATHS.get(key)
        if path:
            status[key] = path.exists()
        else:
            status[key] = False
    
    return status

if __name__ == "__main__":
    print("📁 PATH VERIFICATION")
    print("="*80)
    status = verify_paths()
    for key, exists in status.items():
        icon = "✅" if exists else "❌"
        print(f"{icon} {key}: {PATHS[key]}")
    print("="*80)
