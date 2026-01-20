# real_time_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 실시간 데이터 생성 함수
def generate_realtime_data():
    """실시간 데이터 생성"""
    now = datetime.now()
    return {
        'timestamp': now,
        'value': random.randint(100, 200),
        'category': random.choice(['A', 'B', 'C'])
    }

# 데이터 저장소
realtime_data = []

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("실시간 데이터 모니터링"),
    
    dcc.Graph(id='realtime-chart'),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1초마다 업데이트
        n_intervals=0
    )
])

@app.callback(
    Output('realtime-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_realtime_chart(n):
    # 새 데이터 추가
    new_data = generate_realtime_data()
    realtime_data.append(new_data)
    
    # 최근 50개 데이터만 유지
    if len(realtime_data) > 50:
        realtime_data.pop(0)
    
    # DataFrame으로 변환
    df = pd.DataFrame(realtime_data)
    
    # 카테고리별로 분리
    fig = go.Figure()
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        fig.add_trace(go.Scatter(
            x=cat_data['timestamp'],
            y=cat_data['value'],
            mode='lines+markers',
            name=f'카테고리 {category}',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='실시간 데이터 모니터링',
        xaxis_title='시간',
        yaxis_title='값',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
