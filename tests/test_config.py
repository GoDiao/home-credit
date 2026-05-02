"""config.py 单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    OUTPUT_DIR, MODEL_DIR, REPORT_DIR, VIZ_DIR,
    TARGET, RANDOM_STATE, LGD,
    CUTOFF_RANGE, CORRELATION_THRESHOLD, REMOVE_CORRELATED,
    USE_IV_SELECTION, IV_THRESHOLD, IV_TOP_N,
    setup_logging
)


class TestPaths:
    def test_base_dir_exists(self):
        assert BASE_DIR.exists()

    def test_output_dirs_exist(self):
        for d in [OUTPUT_DIR, MODEL_DIR, REPORT_DIR, VIZ_DIR]:
            assert d.exists()

    def test_raw_data_dir(self):
        assert RAW_DATA_DIR == DATA_DIR / "raw"


class TestConfig:
    def test_target(self):
        assert TARGET == "TARGET"

    def test_random_state(self):
        assert RANDOM_STATE == 42

    def test_lgd_range(self):
        assert 0 < LGD < 1

    def test_cutoff_range(self):
        assert len(CUTOFF_RANGE) > 0
        assert all(0 < c <= 1 for c in CUTOFF_RANGE)

    def test_correlation_threshold(self):
        assert 0 < CORRELATION_THRESHOLD <= 1

    def test_iv_config(self):
        assert IV_THRESHOLD > 0
        assert IV_TOP_N > 0


class TestSetupLogging:
    def test_setup_runs(self):
        import logging
        setup_logging()
        # basicConfig only takes effect on first call; subsequent calls are no-ops
        # Just verify it doesn't raise
        logger = logging.getLogger("test")
        assert logger is not None
