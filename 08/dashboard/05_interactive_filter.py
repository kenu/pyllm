# interactive_filter.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# 샘플 데이터
df = pd.DataFrame({
    'x': np.random.randn(1000),
    'y': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'size': np.random.randint(10, 100, 1000)
})

app.layout = html.Div([
    html.H1("인터랙티브 데이터 필터링"),
    
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(df, x='x', y='y', color='category', size='size',
                         title='드래그하여 데이터 선택')
    ),
    
    html.Div([
        html.H3("선택된 데이터 통계"),
        html.Div(id='selected-stats')
    ])
])

@app.callback(
    Output('selected-stats', 'children'),
    Input('scatter-plot', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "데이터를 선택해주세요"
    
    # 선택된 데이터 추출
    selected_points = selectedData['points']
    selected_indices = [point['pointIndex'] for point in selected_points]
    
    if len(selected_indices) == 0:
        return "선택된 데이터가 없습니다"
    
    selected_df = df.iloc[selected_indices]
    
    # 통계 계산
    stats = [
        html.P(f"선택된 데이터 수: {len(selected_df)}"),
        html.P(f"X 평균: {selected_df['x'].mean():.2f}"),
        html.P(f"Y 평균: {selected_df['y'].mean():.2f}"),
        html.P(f"카테고리 분포: {selected_df['category'].value_counts().to_dict()}")
    ]
    
    return stats

if __name__ == '__main__':
    app.run_server(debug=True)
