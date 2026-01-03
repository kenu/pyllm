# 인터랙티브 시각화 (Plotly)와 대시보드

## Plotly 기초
웹 기반의 인터랙티브 시각화 라이브러리인 Plotly를 사용하여 동적인 차트를 만듭니다.

### 1. Plotly 기본 차트
```python
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
```

### 2. Plotly Express 활용
```python
# Plotly Express로 다양한 차트 생성

# 1. 박스플롯
fig_box = px.box(
    sales_data,
    x='카테고리',
    y='매출',
    color='지역',
    title='카테고리별 매출 분포',
    template='plotly_white'
)
fig_box.show()

# 2. 히스토그램
fig_hist = px.histogram(
    sales_data,
    x='매출',
    color='카테고리',
    marginal='box',  # 주변에 박스플롯 추가
    title='매출 분포',
    template='plotly_white'
)
fig_hist.show()

# 3. 3D 산점도
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

# 4. 지리적 시각화 (지역별 데이터)
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
```

## 대시보드 구축
Dash를 사용하여 웹 기반 대시보드를 만듭니다.

### 1. Dash 기초 앱
```python
# dash_app.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# 데이터 준비
df = sales_data.copy()

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
```

### 2. 고급 대시보드 레이아웃
```python
# advanced_dashboard.py
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

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
                options=[{'label': cat, 'value': cat} for cat in df['카테고리'].unique()],
                value=df['카테고리'].unique(),
                inline=True
            )
        ], className='six columns'),
        
        html.Div([
            html.Label("지역 필터:"),
            dcc.Checklist(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in df['지역'].unique()],
                value=df['지역'].unique(),
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
        filtered_df = filtered_df[filtered_df['카테고리'].isin(selected_categories)]
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['지역'].isin(selected_regions)]
    
    # KPI 계산
    total_revenue = filtered_df['매출'].sum()
    total_transactions = len(filtered_df)
    avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
    
    # 포맷팅
    revenue_text = f"₩{total_revenue:,.0f}"
    transactions_text = f"{total_transactions:,}건"
    avg_text = f"₩{avg_transaction:,.0f}"
    
    # 메인 차트 (시계열)
    main_fig = px.line(filtered_df, x='날짜', y='매출', color='카테고리',
                       title='일별 매출 추이')
    
    # 보조 차트 1 (카테고리별)
    cat_fig = px.pie(filtered_df.groupby('카테고리')['매출'].sum().reset_index(),
                    values='매출', names='카테고리', title='카테고리별 매출 비중')
    
    # 보조 차트 2 (지역별)
    region_fig = px.bar(filtered_df.groupby('지역')['매출'].sum().reset_index(),
                        x='지역', y='매출', title='지역별 매출')
    
    # 데이터 테이블
    table_data = filtered_df.sort_values('매출', ascending=False).to_dict('records')
    
    return revenue_text, transactions_text, avg_text, main_fig, cat_fig, region_fig, table_data

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 실시간 데이터 시각화
실시간으로 업데이트되는 차트를 만듭니다.

### 1. 실시간 차트
```python
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
```

### 2. 웹소켓을 이용한 실시간 데이터
```python
# websocket_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

# 웹소켓 데이터 시뮬레이션
class WebSocketSimulator:
    def __init__(self):
        self.data = []
    
    def get_latest_data(self):
        """최신 데이터 가져오기 (시뮬레이션)"""
        import random
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

app = dash.Dash(__name__)

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
        name='CPU 사용率',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    cpu_fig.update_layout(
        title='CPU 사용率 (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark'
    )
    
    # 메모리 사용량 차트
    memory_fig = go.Figure()
    memory_fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['memory_usage'],
        mode='lines+markers',
        name='메모리 사용率',
        line=dict(color='blue', width=2),
        fill='tonexty'
    ))
    memory_fig.update_layout(
        title='메모리 사용率 (%)',
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
```

## 고급 인터랙티브 기능
사용자와 상호작용하는 고급 기능들을 구현합니다.

### 1. 드래그 앤 드롭 필터링
```python
# interactive_filter.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

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
```

### 2. 동적 차트 생성
```python
# dynamic_charts.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# 샘플 데이터
categories = ['전자제품', '의류', '식품', '가구', '도서']
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
```

## 배포 및 공유
만든 대시보드를 웹에 배포하고 공유하는 방법을 익힙니다.

### 1. Heroku 배포 준비
```python
# requirements.txt
dash==2.14.1
plotly==5.17.0
pandas==2.1.1
numpy==1.24.3
gunicorn==21.2.0
```

```python
# Procfile (Heroku 배포용)
web: gunicorn app:server
```

```python
# app.py (배포용 메인 파일)
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# 전역 데이터
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'sales': np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000,
    'visitors': np.cumsum(np.random.randn(365) * 50 + 200) + 5000
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("비즈니스 분석 대시보드"),
    
    dcc.Graph(id='main-chart'),
    
    html.Div([
        html.Label("날짜 범위 선택:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df['date'].min(),
            end_date=df['date'].max()
        )
    ])
])

@app.callback(
    Output('main-chart', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_chart(start_date, end_date):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    fig = px.line(filtered_df, x='date', y='sales', 
                  title='선택된 기간의 매출 추이')
    
    return fig

# 서버 객체 추가 (배포용)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
```

### 2. 정적 파일로 내보내기
```python
# export_static.py
import plotly.express as px
import pandas as pd
import numpy as np

# 데이터 생성
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'sales': np.cumsum(np.random.randn(365) * 1000 + 50000) + 1000000
})

# 인터랙티브 차트 생성
fig = px.line(df, x='date', y='sales', title='2023년 매출 추이')

# HTML 파일로 저장
fig.write_html("/Users/kenu/git/pyllm/08/sales_chart.html")

# PNG로 저장 (정적)
fig.write_image("/Users/kenu/git/pyllm/08/sales_chart.png", width=1200, height=600)

print("차트가 HTML과 PNG 형식으로 저장되었습니다.")
```

## 성능 최적화
대시보드의 성능을 최적화하는 방법을 익힙니다.

### 1. 데이터 캐싱
```python
# cached_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import time

# 대용량 데이터 생성
large_df = pd.DataFrame({
    'x': np.random.randn(100000),
    'y': np.random.randn(100000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100000)
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("캐싱된 대시보드"),
    
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': cat, 'value': cat} for cat in large_df['category'].unique()],
        value='A'
    ),
    
    dcc.Graph(id='cached-chart')
])

# 캐싱 데코레이터 사용
from dash.exceptions import PreventUpdate
from flask_caching import Cache

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

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
```

이 인터랙티브 시각화 기법들을 통해 사용자가 직접 데이터를 탐색하고 분석할 수 있는 동적인 대시보드를 만들 수 있습니다.
