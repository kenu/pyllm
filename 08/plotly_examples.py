import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
sales_data = pd.DataFrame({
    '날짜': dates,
    '매출': np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000,
    '방문자': np.cumsum(np.random.randn(365) * 50 + 200) + 5000,
    '카테고리': np.random.choice(['전자제품', '의류', '식품', '가구'], 365),
    '지역': np.random.choice(['서울', '부산', '대구', '대전'], 365)
})

# 1. 기본 라인 차트
fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=sales_data['날짜'],
    y=sales_data['매출'],
    mode='lines',
    name='매출',
    line=dict(color='blue', width=2),
    hovertemplate='날짜: %{x}<br>매출: %{y:,.0f}원<extra></extra>'
))

fig_line.add_trace(go.Scatter(
    x=sales_data['날짜'],
    y=sales_data['방문자'] * 100,  # 스케일 조정
    mode='lines',
    name='방문자 (x100)',
    line=dict(color='red', width=2),
    yaxis='y2',
    hovertemplate='날짜: %{x}<br>방문자: %{y:,.0f}<extra></extra>'
))

# 레이아웃 설정
fig_line.update_layout(
    title='일별 매출과 방문자 추이',
    xaxis_title='날짜',
    yaxis_title='매출 (원)',
    yaxis2=dict(
        title='방문자 수 (x100)',
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    showlegend=True,
    template='plotly_white'
)

fig_line.show()
fig_line.write_html('/Users/kenu/git/pyllm/08/plotly_line_chart.html')


# 2. 인터랙티브 산점도
fig_scatter = px.scatter(
    sales_data.sample(100),  # 샘플링하여 표시
    x='매출',
    y='방문자',
    color='카테고리',
    size='매출',
    hover_data=['지역'],
    title='매출과 방문자 관계 (카테고리별)',
    template='plotly_white'
)

fig_scatter.update_traces(
    marker=dict(line=dict(width=1, color='DarkSlateGrey'))
)

fig_scatter.show()
fig_scatter.write_html('/Users/kenu/git/pyllm/08/plotly_scatter_chart.html')

# 3. 박스플롯
fig_box = px.box(
    sales_data,
    x='카테고리',
    y='매출',
    color='지역',
    title='카테고리별 매출 분포',
    template='plotly_white'
)
fig_box.show()
fig_box.write_html('/Users/kenu/git/pyllm/08/plotly_box_chart.html')


# 4. 히스토그램
fig_hist = px.histogram(
    sales_data,
    x='매출',
    color='카테고리',
    marginal='box',  # 주변에 박스플롯 추가
    title='매출 분포',
    template='plotly_white'
)
fig_hist.show()
fig_hist.write_html('/Users/kenu/git/pyllm/08/plotly_histogram_chart.html')


# 5. 3D 산점도
fig_3d = px.scatter_3d(
    sales_data.sample(200),
    x='매출',
    y='방문자',
    z=sales_data.index,
    color='카테고리',
    symbol='지역',
    title='3D 매출 데이터 시각화',
    template='plotly_white'
)
fig_3d.show()
fig_3d.write_html('/Users/kenu/git/pyllm/08/plotly_3d_scatter_chart.html')


# 6. 지리적 시각화 (지역별 데이터)
region_summary = sales_data.groupby('지역').agg({
    '매출': 'sum',
    '방문자': 'sum'
}).reset_index()

# 한국 지역 좌표 (예시)
region_coords = {
    '서울': [37.5665, 126.9780],
    '부산': [35.1796, 129.0756],
    '대구': [35.8722, 128.6014],
    '대전': [36.3504, 127.3845]
}

region_summary['위도'] = region_summary['지역'].map(lambda x: region_coords[x][0])
region_summary['경도'] = region_summary['지역'].map(lambda x: region_coords[x][1])

fig_map = px.scatter_mapbox(
    region_summary,
    lat='위도',
    lon='경도',
    size='매출',
    color='방문자',
    hover_name='지역',
    hover_data=['매출', '방문자'],
    size_max=30,
    zoom=6,
    mapbox_style='open-street-map',
    title='지역별 매출 현황'
)
fig_map.show()
fig_map.write_html('/Users/kenu/git/pyllm/08/plotly_map_chart.html')
