"""
================================================================================
    DATA LOADER - HUGGING FACE DATASET
    Loads data from HF Dataset instead of local files
    Automatically downloads and caches data
================================================================================
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import logging

from huggingface_hub import hf_hub_download
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your HuggingFace dataset repository
DATASET_REPO_ID = os.environ.get(
    "HF_DATASET_REPO",
    "ThiagoHDS/customer-management-data"
)

# Local cache directory
CACHE_DIR = Path("./hf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA FILES MAPPING
# ============================================================================

DATA_FILES = {
    # Core tables
    'customers': 'processed/customers_clean.parquet',
    'products': 'processed/fashion_catalog_clean.parquet',
    'discounts': 'processed/discounts_clean.parquet',
    'transactions': 'processed/transaction_items.parquet',

    # Features
    'churn_risk': 'processed/features/customer_churn_risk.parquet',
    'customer_segments': 'processed/features/customer_segments.parquet',
    'customer_ltv': 'processed/features/customer_lifetime_value.parquet',

    # Optional embeddings
    'product_embeddings': 'product_embeddings.npy',
    'embedding_mapping': 'embedding_mapping.parquet',
}

# ============================================================================
# DATA LOADER CLASS
# ============================================================================

class HFDataLoader:
    """
    Loads data from Hugging Face Dataset
    Falls back to local files if HF not available
    """

    def __init__(
        self,
        repo_id: str = DATASET_REPO_ID,
        cache_dir: Path = CACHE_DIR,
        use_local_fallback: bool = True
    ):
        """
        Initialize data loader

        Args:
            repo_id: HuggingFace dataset repo ID
            cache_dir: Local cache directory
            use_local_fallback: If True, tries local files if HF fails
        """
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.use_local_fallback = use_local_fallback
        self._cached_paths: Dict[str, Path] = {}

        logger.info(f"Initialized HFDataLoader with repo: {repo_id}")

    def get_file_path(self, file_key: str) -> Optional[Path]:
        """
        Get path to data file (downloads from HF if needed)

        Args:
            file_key: Key from DATA_FILES dict

        Returns:
            Path to local file or None if not available
        """
        # Check cache first
        if file_key in self._cached_paths:
            if self._cached_paths[file_key].exists():
                return self._cached_paths[file_key]

        # Get filename from mapping
        if file_key not in DATA_FILES:
            logger.error(f"Unknown file key: {file_key}")
            return None

        filename = DATA_FILES[file_key]

        # Try to download from HF
        try:
            logger.info(f"Downloading {filename} from HF Dataset...")

            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type='dataset',
                cache_dir=str(self.cache_dir),
                token=os.environ.get("HF_TOKEN")
            )

            file_path = Path(file_path)
            self._cached_paths[file_key] = file_path

            logger.info(f"✅ Downloaded: {file_key} → {file_path}")
            return file_path

        except Exception as e:
            logger.warning(f"Failed to download {filename} from HF: {str(e)}")

            # Try local fallback
            if self.use_local_fallback:
                return self._try_local_fallback(file_key, filename)

            return None

    def _try_local_fallback(self, file_key: str, filename: str) -> Optional[Path]:
        """Try to find file locally"""
        # Check common local paths
        possible_paths = [
            Path(f"data/{filename}"),
            Path(f"../{filename}"),
            Path(filename),
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Using local file: {file_key} → {path}")
                self._cached_paths[file_key] = path
                return path

        logger.error(f"❌ File not found locally or on HF: {file_key}")
        return None

    def get_all_available_files(self) -> Dict[str, Path]:
        """
        Get paths to all available data files

        Returns:
            Dict mapping file_key to Path
        """
        available = {}

        for file_key in DATA_FILES.keys():
            path = self.get_file_path(file_key)
            if path:
                available[file_key] = path

        logger.info(f"Available files: {len(available)}/{len(DATA_FILES)}")
        return available

    def download_embeddings(self) -> Optional[tuple[Path, Path]]:
        """
        Download CLIP embeddings and mapping

        Returns:
            (embeddings_path, mapping_path) or None if not available
        """
        embeddings_path = self.get_file_path('product_embeddings')
        mapping_path = self.get_file_path('embedding_mapping')

        if embeddings_path and mapping_path:
            return (embeddings_path, mapping_path)

        return None

    def verify_dataset_exists(self) -> bool:
        """
        Verify that HF dataset repository exists and is accessible

        Returns:
            True if dataset is accessible
        """
        try:
            # Try to list files in the repo
            from huggingface_hub import list_repo_files

            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type='dataset',
                token=os.environ.get("HF_TOKEN")
            )

            logger.info(f"✅ Dataset accessible: {len(files)} files found")
            return len(files) > 0

        except Exception as e:
            logger.warning(f"Cannot access HF dataset: {str(e)}")
            return False

    def clear_cache(self):
        """Clear local cache"""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self._cached_paths = {}
            logger.info("Cache cleared")

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_data_loader = None

def get_data_loader() -> HFDataLoader:
    """
    Get or create data loader singleton

    Returns:
        HFDataLoader instance
    """
    global _data_loader

    if _data_loader is None:
        _data_loader = HFDataLoader()

    return _data_loader

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_parquet_path(table_name: str) -> Optional[Path]:
    """
    Get path to parquet file by table name

    Args:
        table_name: One of: customers, products, discounts, transactions,
                    churn_risk, customer_segments, customer_ltv

    Returns:
        Path to parquet file or None
    """
    loader = get_data_loader()
    return loader.get_file_path(table_name)

def load_parquet(table_name: str):
    """
    Load parquet file as pandas DataFrame

    Args:
        table_name: Table name (see get_parquet_path)

    Returns:
        pandas DataFrame or None
    """
    import pandas as pd

    path = get_parquet_path(table_name)
    if path:
        return pd.read_parquet(path)

    return None

def get_embeddings_paths() -> Optional[tuple[Path, Path]]:
    """
    Get paths to CLIP embeddings

    Returns:
        (embeddings_npy_path, mapping_parquet_path) or None
    """
    loader = get_data_loader()
    return loader.download_embeddings()

# ============================================================================
# DATASET STATUS
# ============================================================================

def check_dataset_status() -> Dict[str, bool]:
    """
    Check which data files are available

    Returns:
        Dict mapping file_key to availability (True/False)
    """
    loader = get_data_loader()
    status = {}

    for file_key in DATA_FILES.keys():
        path = loader.get_file_path(file_key)
        status[file_key] = path is not None

    return status

def print_dataset_status():
    """Print dataset availability status"""
    status = check_dataset_status()

    print("=" * 60)
    print("DATASET STATUS")
    print("=" * 60)

    for file_key, available in status.items():
        icon = "✅" if available else "❌"
        print(f"{icon} {file_key}: {DATA_FILES[file_key]}")

    available_count = sum(status.values())
    total_count = len(status)

    print("=" * 60)
    print(f"Available: {available_count}/{total_count}")
    print("=" * 60)

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test data loader
    print("Testing HFDataLoader...")
    print()

    # Check if dataset is accessible
    loader = get_data_loader()

    if loader.verify_dataset_exists():
        print("✅ Dataset repository is accessible")
    else:
        print("⚠️ Dataset repository not accessible - will use local files")

    print()

    # Print status
    print_dataset_status()

    print()

    # Test loading a file
    print("Testing parquet load...")
    df = load_parquet('customers')

    if df is not None:
        print(f"✅ Loaded customers: {len(df)} rows")
        print(f"Columns: {list(df.columns[:5])}...")
    else:
        print("❌ Failed to load customers")
