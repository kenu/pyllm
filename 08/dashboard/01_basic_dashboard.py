# dash_app.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import sys
import os

# Add the parent directory to the path to import data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import get_sales_data

# 데이터 준비
df = get_sales_data()

# Dash 앱 생성
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("실시간 판매 대시보드", style={'textAlign': 'center'}),
    
    # 컨트롤 패널
    html.Div([
        html.Div([
            html.Label("카테고리 선택:"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': cat, 'value': cat} for cat in df['카테고리'].unique()],
                value=df['카테고리'].unique()[0],
                multi=True
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("날짜 범위:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=df['날짜'].min(),
                end_date=df['날짜'].max(),
                display_format='YYYY-MM-DD'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '20px'})
    ], style={'padding': '20px'}),
    
    # 차트 영역
    html.Div([
        dcc.Graph(id='sales-trend'),
        dcc.Graph(id='category-distribution'),
        dcc.Graph(id='region-performance')
    ])
])

# 콜백 함수
@app.callback(
    [Output('sales-trend', 'figure'),
     Output('category-distribution', 'figure'),
     Output('region-performance', 'figure')],
    [Input('category-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_charts(selected_categories, start_date, end_date):
    # 데이터 필터링
    filtered_df = df.copy()
    
    if selected_categories:
        if isinstance(selected_categories, str):
            selected_categories = [selected_categories]
        filtered_df = filtered_df[filtered_df['카테고리'].isin(selected_categories)]
    
    filtered_df = filtered_df[(filtered_df['날짜'] >= start_date) & 
                               (filtered_df['날짜'] <= end_date)]
    
    # 1. 매출 추이 차트
    fig1 = px.line(filtered_df, x='날짜', y='매출', color='카테고리',
                   title='선택된 기간의 매출 추이')
    
    # 2. 카테고리별 분포
    fig2 = px.pie(filtered_df.groupby('카테고리')['매출'].sum().reset_index(),
                  values='매출', names='카테고리', title='카테고리별 매출 비중')
    
    # 3. 지역별 성과
    fig3 = px.bar(filtered_df.groupby('지역')['매출'].sum().reset_index(),
                  x='지역', y='매출', color='지역', title='지역별 매출')
    
    return fig1, fig2, fig3

if __name__ == '__main__':
    app.run_server(debug=True)
