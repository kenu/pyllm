import pandas as pd
import numpy as np

def get_sales_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sales_data = pd.DataFrame({
        '날짜': dates,
        '매출': np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000,
        '방문자': np.cumsum(np.random.randn(365) * 50 + 200) + 5000,
        '카테고리': np.random.choice(['전제품', '의류', '식품', '가구'], 365),
        '지역': np.random.choice(['서울', '부산', '대구', '대전'], 365)
    })
    return sales_data
