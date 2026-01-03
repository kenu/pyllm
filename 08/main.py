import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'x': np.random.randn(n_samples),
    'y': np.random.randn(n_samples) * 2 + np.random.randn(n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
    'group': np.random.choice(['Group1', 'Group2'], n_samples),
    'value': np.random.randint(1, 100, n_samples),
    'date': pd.date_range('2023-01-01', periods=n_samples, freq='D')
})

print("=== 인터랙티브 시각화 (Plotly) 예제 ===")
print(f"데이터 크기: {data.shape}")
print("\n데이터 샘플:")
print(data.head())

# 1. 기본 인터랙티브 차트
print("\n=== 1. 기본 인터랙티브 라인 차트 ===")

# 시계열 데이터 준비
time_data = data.groupby('date').agg({
    'value': 'sum',
    'x': 'mean'
}).reset_index()

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=time_data['date'],
    y=time_data['value'],
    mode='lines+markers',
    name='일별 총합',
    line=dict(color='blue', width=2),
    hovertemplate='날짜: %{x}<br>값: %{y:,.0f}<extra></extra>'
))

fig_line.add_trace(go.Scatter(
    x=time_data['date'],
    y=time_data['x'] * 100,  # 스케일 조정
    mode='lines',
    name='X 평균 (x100)',
    line=dict(color='red', width=2),
    yaxis='y2',
    hovertemplate='날짜: %{x}<br>X 평균: %{y:,.0f}<extra></extra>'
))

fig_line.update_layout(
    title='일별 데이터 추이 (인터랙티브)',
    xaxis_title='날짜',
    yaxis_title='총합',
    yaxis2=dict(
        title='X 평균 (x100)',
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    showlegend=True,
    template='plotly_white'
)

# HTML 파일로 저장
fig_line.write_html('/Users/kenu/git/pyllm/08/line_chart.html')
print("라인 차트가 HTML 파일로 저장되었습니다.")

# 2. 인터랙티브 산점도
print("\n=== 2. 인터랙티브 산점도 ===")

fig_scatter = px.scatter(
    data,
    x='x',
    y='y',
    color='category',
    size='value',
    hover_data=['group'],
    title='인터랙티브 산점도 (카테고리별)',
    template='plotly_white'
)

fig_scatter.update_traces(
    marker=dict(line=dict(width=1, color='DarkSlateGrey'))
)

fig_scatter.write_html('/Users/kenu/git/pyllm/08/scatter_chart.html')
print("산점도가 HTML 파일로 저장되었습니다.")

# 3. 다양한 Plotly Express 차트
print("\n=== 3. 다양한 차트 유형 ===")

# 박스플롯
fig_box = px.box(
    data,
    x='category',
    y='value',
    color='group',
    title='카테고리별 값 분포',
    template='plotly_white'
)
fig_box.write_html('/Users/kenu/git/pyllm/08/box_chart.html')

# 히스토그램
fig_hist = px.histogram(
    data,
    x='value',
    color='category',
    marginal='box',
    title='값 분포 (카테고리별)',
    template='plotly_white'
)
fig_hist.write_html('/Users/kenu/git/pyllm/08/histogram_chart.html')

# 3D 산점도
fig_3d = px.scatter_3d(
    data.sample(100),
    x='x',
    y='y',
    z='value',
    color='category',
    symbol='group',
    title='3D 산점도',
    template='plotly_white'
)
fig_3d.write_html('/Users/kenu/git/pyllm/08/3d_scatter_chart.html')

# 파이 차트
category_counts = data['category'].value_counts()
fig_pie = px.pie(
    values=category_counts.values,
    names=category_counts.index,
    title='카테고리 비중',
    template='plotly_white'
)
fig_pie.write_html('/Users/kenu/git/pyllm/08/pie_chart.html')

# 막대 차트
category_means = data.groupby('category')['value'].mean().reset_index()
fig_bar = px.bar(
    category_means,
    x='category',
    y='value',
    title='카테고리별 평균값',
    template='plotly_white'
)
fig_bar.write_html('/Users/kenu/git/pyllm/08/bar_chart.html')

print("다양한 차트들이 HTML 파일로 저장되었습니다.")

# 4. 고급 인터랙티브 기능
print("\n=== 4. 고급 인터랙티브 기능 ===")

# 서브플롯
from plotly.subplots import make_subplots

fig_subplots = make_subplots(
    rows=2, cols=2,
    subplot_titles=('산점도', '히스토그램', '박스플롯', '라인 차트'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 산점도
fig_subplots.add_trace(
    go.Scatter(x=data['x'], y=data['y'], mode='markers', name='산점도'),
    row=1, col=1
)

# 히스토그램
fig_subplots.add_trace(
    go.Histogram(x=data['value'], name='히스토그램'),
    row=1, col=2
)

# 박스플롯
for i, cat in enumerate(['A', 'B', 'C']):
    fig_subplots.add_trace(
        go.Box(y=data[data['category'] == cat]['value'], name=f'카테고리 {cat}'),
        row=2, col=1
    )

# 라인 차트
fig_subplots.add_trace(
    go.Scatter(x=time_data['date'], y=time_data['value'], mode='lines', name='시계열'),
    row=2, col=2
)

fig_subplots.update_layout(
    title='다중 차트 대시보드',
    showlegend=False,
    template='plotly_white'
)

fig_subplots.write_html('/Users/kenu/git/pyllm/08/subplots_chart.html')
print("서브플롯 차트가 HTML 파일로 저장되었습니다.")

# 5. 애니메이션 차트
print("\n=== 5. 애니메이션 차트 ===")

# 애니메이션 데이터 준비
animation_data = []
for i in range(50):
    chunk = data.iloc[i*4:(i+1)*4].copy()
    chunk['frame'] = i
    animation_data.append(chunk)

animation_df = pd.concat(animation_data)

fig_animation = px.scatter(
    animation_df,
    x='x',
    y='y',
    color='category',
    size='value',
    animation_frame='frame',
    title='애니메이션 산점도',
    template='plotly_white'
)

fig_animation.write_html('/Users/kenu/git/pyllm/08/animation_chart.html')
print("애니메이션 차트가 HTML 파일로 저장되었습니다.")

# 6. 통계 차트
print("\n=== 6. 통계 분석 차트 ===")

# 상관관계 히트맵
corr_matrix = data[['x', 'y', 'value']].corr()

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_matrix.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 10},
    hoverongaps=False
))

fig_heatmap.update_layout(
    title='상관관계 히트맵',
    template='plotly_white'
)

fig_heatmap.write_html('/Users/kenu/git/pyllm/08/heatmap_chart.html')
print("상관관계 히트맵이 HTML 파일로 저장되었습니다.")

# 7. 대시보드 스타일 차트
print("\n=== 7. 대시보드 스타일 차트 ===")

# 복합 대시보드
fig_dashboard = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        '일별 추이', '카테고리별 비중', '상자 그림',
        '분포', '산점도', '막대 그래프',
        '히스토그램', '상관관계', '3D 뷰'
    ),
    specs=[[{"secondary_y": True}, {"type": "pie"}, {"type": "box"}],
           [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "heatmap"}, {"type": "scatter3d"}]]
)

# 다양한 차트 추가
fig_dashboard.add_trace(
    go.Scatter(x=time_data['date'], y=time_data['value'], name='추이'),
    row=1, col=1
)

fig_dashboard.add_trace(
    go.Pie(values=category_counts.values, labels=category_counts.index, name='비중'),
    row=1, col=2
)

fig_dashboard.add_trace(
    go.Box(y=data['value'], name='분포'),
    row=1, col=3
)

fig_dashboard.add_trace(
    go.Histogram(x=data['value'], name='히스토그램'),
    row=2, col=1
)

fig_dashboard.add_trace(
    go.Scatter(x=data['x'], y=data['y'], mode='markers', name='산점도'),
    row=2, col=2
)

fig_dashboard.add_trace(
    go.Bar(x=category_means['category'], y=category_means['value'], name='막대'),
    row=2, col=3
)

fig_dashboard.add_trace(
    go.Histogram(x=data['x'], name='X 분포'),
    row=3, col=1
)

fig_dashboard.add_trace(
    go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, name='상관관계'),
    row=3, col=2
)

fig_dashboard.update_layout(
    title='종합 분석 대시보드',
    showlegend=False,
    height=1200,
    template='plotly_white'
)

fig_dashboard.write_html('/Users/kenu/git/pyllm/08/dashboard.html')
print("종합 대시보드가 HTML 파일로 저장되었습니다.")

print("\n=== 생성된 파일 목록 ===")
print("1. line_chart.html - 인터랙티브 라인 차트")
print("2. scatter_chart.html - 인터랙티브 산점도")
print("3. box_chart.html - 박스플롯")
print("4. histogram_chart.html - 히스토그램")
print("5. 3d_scatter_chart.html - 3D 산점도")
print("6. pie_chart.html - 파이 차트")
print("7. bar_chart.html - 막대 차트")
print("8. subplots_chart.html - 서브플롯")
print("9. animation_chart.html - 애니메이션 차트")
print("10. heatmap_chart.html - 상관관계 히트맵")
print("11. dashboard.html - 종합 대시보드")

print("\n인터랙티브 시각화 예제 완료!")
print("HTML 파일들을 웹 브라우저에서 열어 인터랙티브 기능을 확인해보세요.")
