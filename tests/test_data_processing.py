"""data_processing.py 单元测试"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import DataProcessor


class TestDataProcessor:
    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'SK_ID_CURR': range(n),
            'TARGET': np.random.binomial(1, 0.1, n),
            'DAYS_BIRTH': np.random.randint(-25000, -7000, n),
            'DAYS_EMPLOYED': np.random.choice(
                [365243] * 20 + list(np.random.randint(-15000, 0, 30)),
                n
            ),
            'AMT_INCOME_TOTAL': np.random.lognormal(11, 0.8, n),
            'AMT_CREDIT': np.random.lognormal(13, 0.7, n),
            'NAME_INCOME_TYPE': np.random.choice(
                ['Working', 'Commercial associate', 'Pensioner'], n
            ),
            'EXT_SOURCE_1': np.random.uniform(0, 1, n),
            'EXT_SOURCE_2': np.random.uniform(0, 1, n),
            'EXT_SOURCE_3': np.random.uniform(0, 1, n),
        })

    def test_business_rules_retired(self, processor, sample_df):
        df = processor.apply_business_rules(sample_df)
        assert 'IS_RETIRED' in df.columns
        assert df['IS_RETIRED'].sum() > 0
        retired_mask = df['IS_RETIRED'] == 1
        assert df.loc[retired_mask, 'DAYS_EMPLOYED'].isna().all()

    def test_handle_missing_values(self, processor, sample_df):
        sample_df.loc[0:5, 'EXT_SOURCE_1'] = np.nan
        df = processor.handle_missing_values(sample_df)
        assert df.isnull().sum().sum() == 0

    def test_process_pipeline_output(self, processor, sample_df):
        result = processor.process_pipeline(sample_df, is_train=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert 'IS_RETIRED' in result.columns
