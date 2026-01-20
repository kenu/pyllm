# export_static.py
import plotly.express as px
import pandas as pd
import numpy as np
import os

# 데이터 생성
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'sales': np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000
})

# 인터랙티브 차트 생성
fig = px.line(df, x='date', y='sales', title='2023년 매출 추이')

# HTML 파일로 저장
if not os.path.exists('/Users/kenu/git/pyllm/08/output'):
    os.makedirs('/Users/kenu/git/pyllm/08/output')

fig.write_html("/Users/kenu/git/pyllm/08/output/sales_chart.html")

# PNG로 저장 (정적)
fig.write_image("/Users/kenu/git/pyllm/08/output/sales_chart.png", width=1200, height=600)

print("차트가 HTML과 PNG 형식으로 'output' 디렉토리에 저장되었습니다.")
