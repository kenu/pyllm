# websocket_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import random

# 웹소켓 데이터 시뮬레이션
class WebSocketSimulator:
    def __init__(self):
        self.data = []
    
    def get_latest_data(self):
        """최신 데이터 가져오기 (시뮬레이션)"""
        new_point = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'cpu_usage': random.uniform(20, 80),
            'memory_usage': random.uniform(30, 70),
            'network_io': random.uniform(10, 100)
        }
        self.data.append(new_point)
        
        # 최근 100개 데이터만 유지
        if len(self.data) > 100:
            self.data.pop(0)
        
        return pd.DataFrame(self.data)

ws_simulator = WebSocketSimulator()

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    html.H1("시스템 모니터링 대시보드"),
    
    html.Div([
        html.Div([
            dcc.Graph(id='cpu-chart')
        ], className='six columns'),
        
        html.Div([
            dcc.Graph(id='memory-chart')
        ], className='six columns')
    ], className='row'),
    
    html.Div([
        dcc.Graph(id='network-chart')
    ], className='row'),
    
    dcc.Interval(
        id='interval-component',
        interval=2000,  # 2초마다 업데이트
        n_intervals=0
    )
])

@app.callback(
    [Output('cpu-chart', 'figure'),
     Output('memory-chart', 'figure'),
     Output('network-chart', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_system_charts(n):
    df = ws_simulator.get_latest_data()
    
    if len(df) == 0:
        return {}, {}, {}
    
    # CPU 사용량 차트
    cpu_fig = go.Figure()
    cpu_fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['cpu_usage'],
        mode='lines+markers',
        name='CPU 사용률',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    cpu_fig.update_layout(
        title='CPU 사용률 (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark'
    )
    
    # 메모리 사용량 차트
    memory_fig = go.Figure()
    memory_fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['memory_usage'],
        mode='lines+markers',
        name='메모리 사용률',
        line=dict(color='blue', width=2),
        fill='tonexty'
    ))
    memory_fig.update_layout(
        title='메모리 사용률 (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark'
    )
    
    # 네트워크 I/O 차트
    network_fig = go.Figure()
    network_fig.add_trace(go.Bar(
        x=df['time'],
        y=df['network_io'],
        name='네트워크 I/O',
        marker_color='green'
    ))
    network_fig.update_layout(
        title='네트워크 I/O (MB/s)',
        template='plotly_dark'
    )
    
    return cpu_fig, memory_fig, network_fig

if __name__ == '__main__':
    app.run_server(debug=True)
