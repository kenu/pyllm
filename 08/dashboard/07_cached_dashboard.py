# cached_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import time
from flask_caching import Cache

# 대용량 데이터 생성
large_df = pd.DataFrame({
    'x': np.random.randn(100000),
    'y': np.random.randn(100000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100000)
})

app = dash.Dash(__name__)

# 캐시 설정
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})


app.layout = html.Div([
    html.H1("캐시된 대시보드"),
    
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': cat, 'value': cat} for cat in large_df['category'].unique()],
        value='A'
    ),
    
    dcc.Graph(id='cached-chart')
])

# 캐시 데코레이터 사용
@app.callback(
    Output('cached-chart', 'figure'),
    Input('category-dropdown', 'value')
)
@cache.memoize(timeout=60)  # 60초 캐시
def update_chart(selected_category):
    # 시뮬레이션된 무거운 연산
    time.sleep(2)  # 2초 지연
    
    filtered_df = large_df[large_df['category'] == selected_category]
    
    fig = px.scatter(filtered_df.sample(1000), x='x', y='y',
                    title=f'카테고리 {selected_category} 데이터')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
