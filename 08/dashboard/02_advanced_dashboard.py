# advanced_dashboard.py
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data import get_sales_data

# 데이터 준비
df = get_sales_data()
df = df.rename(columns={'날짜': 'date', '매출': 'revenue', '방문자': 'visitors', '카테고리': 'category', '지역': 'region'})


# 앱 생성
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# 레이아웃
app.layout = html.Div([
    # 헤더
    html.Div([
        html.H1("📊 종합 비즈니스 분석 대시보드", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.Hr()
    ]),
    
    # KPI 카드
    html.Div([
        html.Div([
            html.Div([
                html.H3("총매출", style={'color': '#7f8c8d'}),
                html.H2(id='total-revenue', style={'color': '#27ae60'})
            ], className='four columns', style={'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ], className='row'),
        
        html.Div([
            html.Div([
                html.H3("총거래수", style={'color': '#7f8c8d'}),
                html.H2(id='total-transactions', style={'color': '#3498db'})
            ], className='four columns', style={'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ], className='row'),
        
        html.Div([
            html.Div([
                html.H3("평균객단가", style={'color': '#7f8c8d'}),
                html.H2(id='avg-transaction', style={'color': '#e74c3c'})
            ], className='four columns', style={'textAlign': 'center', 'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
        ], className='row')
    ], className='row', style={'marginBottom': '30px'}),
    
    # 필터 섹션
    html.Div([
        html.Div([
            html.Label("카테고리 필터:"),
            dcc.Checklist(
                id='category-filter',
                options=[{'label': cat, 'value': cat} for cat in df['category'].unique()],
                value=df['category'].unique(),
                inline=True
            )
        ], className='six columns'),
        
        html.Div([
            html.Label("지역 필터:"),
            dcc.Checklist(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in df['region'].unique()],
                value=df['region'].unique(),
                inline=True
            )
        ], className='six columns')
    ], className='row', style={'marginBottom': '20px'}),
    
    # 차트 섹션
    html.Div([
        html.Div([
            dcc.Graph(id='main-chart')
        ], className='twelve columns')
    ], className='row'),
    
    html.Div([
        html.Div([
            dcc.Graph(id='secondary-chart-1')
        ], className='six columns'),
        
        html.Div([
            dcc.Graph(id='secondary-chart-2')
        ], className='six columns')
    ], className='row'),
    
    # 데이터 테이블
    html.Div([
        html.H3("상세 데이터"),
        dash_table.DataTable(
            id='data-table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'}
        )
    ], className='row', style={'marginTop': '30px'})
])

# 콜백 함수
@app.callback(
    [Output('total-revenue', 'children'),
     Output('total-transactions', 'children'),
     Output('avg-transaction', 'children'),
     Output('main-chart', 'figure'),
     Output('secondary-chart-1', 'figure'),
     Output('secondary-chart-2', 'figure'),
     Output('data-table', 'data')],
    [Input('category-filter', 'value'),
     Input('region-filter', 'value')]
)
def update_dashboard(selected_categories, selected_regions):
    # 데이터 필터링
    filtered_df = df.copy()
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]
    
    # KPI 계산
    total_revenue = filtered_df['revenue'].sum()
    total_transactions = len(filtered_df)
    avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
    
    # 포맷팅
    revenue_text = f"₩{total_revenue:,.0f}"
    transactions_text = f"{total_transactions:,}건"
    avg_text = f"₩{avg_transaction:,.0f}"
    
    # 메인 차트 (시계열)
    main_fig = px.line(filtered_df, x='date', y='revenue', color='category',
                       title='일별 매출 추이')
    
    # 보조 차트 1 (카테고리별)
    cat_fig = px.pie(filtered_df.groupby('category')['revenue'].sum().reset_index(),
                    values='revenue', names='category', title='카테고리별 매출 비중')
    
    # 보조 차트 2 (지역별)
    region_fig = px.bar(filtered_df.groupby('region')['revenue'].sum().reset_index(),
                        x='region', y='revenue', title='지역별 매출')
    
    # 데이터 테이블
    table_data = filtered_df.sort_values('revenue', ascending=False).to_dict('records')
    
    return revenue_text, transactions_text, avg_text, main_fig, cat_fig, region_fig, table_data

if __name__ == '__main__':
    app.run_server(debug=True)
