# dynamic_charts.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# 샘플 데이터
categories = ['전제품', '의류', '식품', '가구', '도서']
regions = ['서울', '부산', '대구', '대전', '광주']

# 데이터 생성 함수
def generate_data(chart_type, category=None, region=None):
    n_points = 100
    df = pd.DataFrame({
        'x': np.random.randn(n_points),
        'y': np.random.randn(n_points),
        'value': np.random.randint(1, 100, n_points),
        'category': np.random.choice(categories, n_points),
        'region': np.random.choice(regions, n_points)
    })
    
    if category:
        df = df[df['category'] == category]
    if region:
        df = df[df['region'] == region]
    
    return df

app.layout = html.Div([
    html.H1("동적 차트 생성기"),
    
    # 컨트롤 패널
    html.Div([
        html.Div([
            html.Label("차트 유형:"),
            dcc.Dropdown(
                id='chart-type',
                options=[
                    {'label': '산점도', 'value': 'scatter'},
                    {'label': '라인 차트', 'value': 'line'},
                    {'label': '막대 차트', 'value': 'bar'},
                    {'label': '히스토그램', 'value': 'histogram'},
                    {'label': '파이 차트', 'value': 'pie'},
                    {'label': '3D 산점도', 'value': '3d_scatter'}
                ],
                value='scatter'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label("카테고리 필터:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': cat, 'value': cat} for cat in categories],
                value=None,
                placeholder='전체'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label("지역 필터:"),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': reg, 'value': reg} for reg in regions],
                value=None,
                placeholder='전체'
            )
        ], className='three columns'),
        
        html.Div([
            html.Button('차트 생성', id='generate-btn', n_clicks=0,
                       style={'marginTop': '25px'})
        ], className='three columns')
    ], className='row'),
    
    # 차트 표시 영역
    html.Div([
        dcc.Graph(id='dynamic-chart')
    ], className='row', style={'marginTop': '30px'})
])

@app.callback(
    Output('dynamic-chart', 'figure'),
    [Input('generate-btn', 'n_clicks')],
    [State('chart-type', 'value'),
     State('category-filter', 'value'),
     State('region-filter', 'value')]
)
def generate_chart(n_clicks, chart_type, category, region):
    if n_clicks == 0:
        return go.Figure()  # 빈 차트
    
    # 데이터 생성
    df = generate_data(chart_type, category, region)
    
    # 차트 유형별 생성
    if chart_type == 'scatter':
        fig = px.scatter(df, x='x', y='y', color='category', size='value',
                        title='인터랙티브 산점도')
    
    elif chart_type == 'line':
        fig = px.line(df.sort_values('x'), x='x', y='y', color='category',
                     title='라인 차트')
    
    elif chart_type == 'bar':
        fig = px.bar(df.groupby('category')['value'].sum().reset_index(),
                    x='category', y='value', title='막대 차트')
    
    elif chart_type == 'histogram':
        fig = px.histogram(df, x='value', color='category',
                         title='히스토그램', nbins=20)
    
    elif chart_type == 'pie':
        cat_counts = df['category'].value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                    title='파이 차트')
    
    elif chart_type == '3d_scatter':
        fig = px.scatter_3d(df, x='x', y='y', z='value', color='category',
                           title='3D 산점도')
    
    else:
        fig = go.Figure()
    
    fig.update_template('plotly_white')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
