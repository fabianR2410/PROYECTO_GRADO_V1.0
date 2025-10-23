"""
Dashboard COVID-19 Profesional - VERSIÓN SIMPLIFICADA v6.6 (Corregida)
======================================================================
Dashboard interactivo enfocado en las 14 columnas de datos provistas.
Se eliminó el selector de rango de fechas de la pestaña Vista General.
La lógica de filtrado de fechas y rankings se mueve a la API.
"""

import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Importar make_subplots
import plotly.express as px
import requests
import logging
from functools import lru_cache
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
import time
import sys
import pandas as pd

# Variables de entorno
try:
    from decouple import config as env_config
except ImportError:
    print("ERROR: python-decouple no instalado.")
    sys.exit(1)

# Configuración y Logging
logging.basicConfig(level=env_config("LOG_LEVEL", default="INFO"), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CovidDashboardSimple")

class Config:
    """Configuración del Dashboard desde variables de entorno."""
    API_BASE_URL: str = env_config("API_BASE_URL", default="http://127.0.0.1:8000")
    API_KEY: str = env_config("API_KEY", default="")
    UPDATE_INTERVAL_MS: int = env_config("UPDATE_INTERVAL_MS", default=300000, cast=int)
    REQUEST_TIMEOUT: int = 15
    COLORS: Dict[str, str] = {
        'bg_dark': '#f8f9fa',
        'bg_card': '#ffffff',
        'text_primary': '#1a1a1a',
        'text_secondary': '#666666',
        'accent_blue': '#2563eb',
        'accent_green': '#10b981',
        'accent_red': '#ef4444',
        'accent_yellow': '#f59e0b',
        'accent_purple': '#8b5cf6',
        'accent_teal': '#14b8a6',
        'border': '#e5e7eb',
        'grid': '#f0f0f0'
    }

    METRICS: Dict[str, str] = {
        # Casos y Muertes
        'total_cases': 'Casos Totales',
        'new_cases': 'Nuevos Casos (diarios)',
        'new_cases_smoothed': 'Nuevos Casos (media 7 días)',
        'total_deaths': 'Muertes Totales',
        'new_deaths': 'Nuevas Muertes (diarias)',
        'new_deaths_smoothed': 'Nuevas Muertes (media 7 días)',
        'mortality_rate': 'Tasa de Mortalidad (%)',

        # Métricas Normalizadas
        'total_cases_per_100k': 'Casos por 100k hab',
        'total_deaths_per_100k': 'Muertes por 100k hab',

        # Demografía
        'population': 'Población'
    }

    SECONDARY_AXIS_METRICS: List[str] = [
        'mortality_rate'
    ]


config = Config()

# ============================================================================
# CLIENTE API (MODIFICADO PARA ENVIAR FECHAS Y TOP N)
# ============================================================================
class CovidAPI:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
            logger.info("🔑 API Key configurada")
        else:
            logger.warning("⚠️ Sin API Key configurada")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        try:
            logger.debug(f"Request: {endpoint} - Params: {params}")
            response = requests.get(url, params=params, headers=self.headers, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en request a {endpoint}")
            return {"error": "Timeout"} # Devolver error
        except requests.exceptions.HTTPError as e:
            detail = f"HTTP Error {e.response.status_code}"
            try:
                error_json = e.response.json()
                detail += f": {error_json.get('detail', e.response.text)}"
            except ValueError:
                 detail += f": {e.response.text}"
            logger.error(f"Error en {endpoint}: {detail}")
            return {"error": detail}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en request a {endpoint}: {e}")
            return {"error": str(e)}

    @lru_cache(maxsize=1)
    def get_countries(self) -> Optional[Dict[str, Any]]: return self._make_request("/covid/countries")

    @lru_cache(maxsize=4)
    def get_global_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # --- API todavía no usa fechas, así que las ignoramos aquí ---
        return self._make_request("/covid/global")

    # ========================================================================
    # >>>>> INICIO DE MODIFICACIÓN (DASHBOARD) <<<<<
    # ========================================================================
    @lru_cache(maxsize=20)
    def get_map_data(self, metric: str, start_date: Optional[str] = None, end_date: Optional[str] = None, top: Optional[int] = None) -> Optional[Dict[str, Any]]:
         # --- API todavía no usa fechas, así que las ignoramos aquí ---
         params = {'metric': metric}
         if top:
             params['top'] = top
         return self._make_request("/covid/map-data", params=params)

    @lru_cache(maxsize=20)
    def get_country_timeseries_all(self, country: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 5000) -> Optional[Dict[str, Any]]:
        logger.info(f"API Call: get_country_timeseries_all (País: {country}, Rango: {start_date}-{end_date})")
        params = {'limit': limit}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        return self._make_request(f"/covid/country/{country}", params=params)
    # ========================================================================
    # >>>>> FIN DE MODIFICACIÓN (DASHBOARD) <<<<<
    # ========================================================================

    def compare_countries(self, countries: List[str], metric: str, normalize: bool = False) -> Optional[Dict[str, Any]]:
        return self._make_request("/covid/compare", params={'countries': countries, 'metric': metric, 'normalize': normalize})

    def get_statistical_summary(self, metric: str, include_outliers: bool = False) -> Optional[Dict[str, Any]]:
        return self._make_request("/covid/metrics/statistics", params={'metric': metric, 'grouping': 'global', 'include_outliers': include_outliers})
    def get_correlations(self, metrics: List[str], method: str = 'pearson') -> Optional[Dict[str, Any]]:
        return self._make_request("/covid/metrics/correlations", params={'metrics': metrics, 'grouping': 'global', 'method': method})

api = CovidAPI(config.API_BASE_URL, config.API_KEY)

# ============================================================================
# INICIALIZAR DASH Y LAYOUT (Sin cambios)
# ============================================================================
app = dash.Dash(__name__, title="Panel COVID-19 - TRABAJO GRADO", suppress_callback_exceptions=True)
app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}''' + \
                   '''<style>body{margin:0;font-family:sans-serif;} .metric-table{width:100%;border-collapse:collapse;font-size:11px;} ''' + \
                   '''.metric-table th,.metric-table td{padding:6px 8px;text-align:left;border-bottom:1px solid #f0f0f0;} .metric-table th{font-weight:600;color:#666;background:#f8f9fa;position:sticky;top:0;}''' + \
                   '''.stat-header{font-size:12px;font-weight:600;color:#666;margin-bottom:8px;} .stat-value{font-size:20px;font-weight:700;color:#1a1a1a;} .stat-subvalue{font-size:11px;color:#888;margin-top:4px;}''' + \
                   '''.warning-message{background:#ffc;border:1px solid #fc6;padding:15px;border-radius:8px;color:#963;margin:20px;}</style></head>''' + \
                   '''<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

def get_tab_style(): return {'padding':'12px 24px','fontWeight':'600','fontSize':'13px','border':'none','borderBottom':'3px solid transparent','backgroundColor':config.COLORS['bg_card'],'color':config.COLORS['text_secondary']}
def get_tab_selected_style(): return {'padding':'12px 24px','fontWeight':'700','fontSize':'13px','border':'none','borderBottom':f"3px solid {config.COLORS['accent_blue']}",'backgroundColor':config.COLORS['bg_card'],'color':config.COLORS['accent_blue']}
def get_card_style(padding='20px'): return {'background':config.COLORS['bg_card'],'borderRadius':'12px','padding':padding,'boxShadow':'0 1px 3px rgba(0,0,0,0.1)','border':f"1px solid {config.COLORS['border']}",'marginBottom':'20px'}

app.layout = html.Div([
    html.Div([
        html.Div([ html.H1("🌍 Panel COVID-19 - Análisis", style={'fontSize':'28px','fontWeight':'700','color':config.COLORS['text_primary'],'margin':'0'}),
                   html.P("Datos y comparativas de COVID-19 a nivel mundial y por país", style={'fontSize':'13px','color':config.COLORS['text_secondary'],'margin':'5px 0 0 0'}) ], style={'flex':'1'}),
        html.Div([ html.Span("Cargando...", id='api-status', style={'fontSize':'11px','padding':'6px 12px','background':config.COLORS['accent_yellow'],'color':'white','borderRadius':'20px','fontWeight':'600'}) ])
    ], style={'background':config.COLORS['bg_card'],'padding':'25px 35px','borderBottom':f"3px solid {config.COLORS['accent_blue']}",'display':'flex','alignItems':'center','justifyContent':'space-between','boxShadow':'0 2px 8px rgba(0,0,0,0.05)'}),

    html.Div([
        dcc.Tabs(id='main-tabs', value='tab-overview', children=[
            dcc.Tab(label='📊 Vista General', value='tab-overview', style=get_tab_style(), selected_style=get_tab_selected_style()),
            dcc.Tab(label='📈 Evolución por País', value='tab-timeseries', style=get_tab_style(), selected_style=get_tab_selected_style()),
            dcc.Tab(label='📈 Comparaciones (Países)', value='tab-comparisons', style=get_tab_style(), selected_style=get_tab_selected_style()),
            dcc.Tab(label='📉 Estadísticas (Global)', value='tab-statistics', style=get_tab_style(), selected_style=get_tab_selected_style()),
            dcc.Tab(label='🔗 Correlaciones (Global)', value='tab-correlations', style=get_tab_style(), selected_style=get_tab_selected_style()),
        ], style={'borderBottom': f"1px solid {config.COLORS['border']}"})
    ], style={'background':config.COLORS['bg_card'],'boxShadow':'0 2px 4px rgba(0,0,0,0.05)'}),

    html.Div(id='tabs-content', style={'padding':'20px 35px','background':config.COLORS['bg_dark'],'minHeight':'calc(100vh - 200px)'}),
    dcc.Store(id='store-countries'),
    dcc.Interval(id='interval-component', interval=config.UPDATE_INTERVAL_MS, n_intervals=0)
], style={'background':config.COLORS['bg_dark']})

# ============================================================================
# CALLBACKS: CARGA INICIAL Y ESTADO (Sin cambios)
# ============================================================================
@app.callback(
    Output('store-countries', 'data'),
    Input('interval-component', 'n_intervals')
)
def load_initial_data(n):
    if n > 0 and callback_context.triggered_id == 'interval-component':
         return no_update

    logger.info("Cargando lista de países...")
    data = api.get_countries()
    if data and isinstance(data, dict) and 'countries' in data:
        logger.info(f"Países cargados: {len(data['countries'])}")
        return data['countries']
    elif data and isinstance(data, dict) and 'error' in data:
         logger.error(f"Error cargando países: {data['error']}")
         return []
    else:
         logger.error(f"Respuesta inesperada o error desconocido cargando países: {data}")
         return []


@app.callback( Output('api-status', 'children'), Output('api-status', 'style'), Input('interval-component', 'n_intervals'))
def update_api_status(n):
    data = api.get_countries()
    base_style = {'fontSize':'11px','padding':'6px 12px','color':'white','borderRadius':'20px','fontWeight':'600'}
    if data and isinstance(data, dict) and 'countries' in data:
        style = {**base_style, 'background': config.COLORS['accent_green']}
        return "🟢 API Conectada", style
    else:
        style = {**base_style, 'background': config.COLORS['accent_red']}
        error_msg = data.get('error', 'Desconectada') if isinstance(data, dict) else 'Respuesta inválida'
        logger.warning(f"Estado API: {error_msg}")
        return f"🔴 API ({error_msg})", style


# ============================================================================
# CALLBACKS PARA CONTENIDO DE TABS (Sin cambios)
# ============================================================================
@app.callback(Output('tabs-content', 'children'), Input('main-tabs', 'value'))
def render_tab_content(active_tab):
    logger.debug(f"Renderizando pestaña: {active_tab}")
    if active_tab == 'tab-overview':
        return create_overview_layout()
    elif active_tab == 'tab-timeseries':
        return create_timeseries_layout()
    elif active_tab == 'tab-comparisons':
        return create_comparisons_layout()
    elif active_tab == 'tab-statistics':
        return create_statistics_layout()
    elif active_tab == 'tab-correlations':
        return create_correlations_layout()
    logger.warning(f"Intento de renderizar pestaña desconocida: {active_tab}")
    return html.Div(f"Contenido no disponible para {active_tab}")

# ============================================================================
# LAYOUTS
# ============================================================================
# ========================================================================
# >>>>> LAYOUT MODIFICADO (SIN RANGO DE FECHAS) <<<<<
# ========================================================================
def create_overview_layout():
     logger.debug("Creando layout: Vista General")
     return html.Div([
        html.H2("Vista General Global", style={'fontSize':'22px','fontWeight':'700','color':config.COLORS['text_primary'],'marginBottom':'20px'}),

        # --- Controles de Vista General ELIMINADOS ---

        dcc.Loading(id="loading-global-cards", children=html.Div(id='global-metrics-cards', style={'marginBottom':'30px'})),

        html.Div([
            html.Div([
                html.H3("🗺️ Distribución Global", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
                html.Div([
                    dcc.Dropdown(
                        id='map-metric-selector',
                        options=[{'label':v,'value':k} for k,v in config.METRICS.items() if k != 'population'],
                        value='total_cases',
                        clearable=False,
                        style={'marginBottom':'15px'},
                        persistence=True,
                        persistence_type='local'
                    ),
                    dcc.Loading(id='loading-map', type='default', children=html.Div(id='world-map', children=dcc.Graph(figure=go.Figure())))
                ])
            ], style={**get_card_style(), 'marginBottom':'20px'}),
            html.Div([
                html.H3("🏆 Top 10 Países", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
                dcc.Loading(id='loading-top-countries', type='default', children=html.Div(id='top-countries-chart', children=dcc.Graph(figure=go.Figure())))
            ], style=get_card_style())
        ])
    ])
# ========================================================================
# >>>>> FIN LAYOUT MODIFICADO <<<<<
# ========================================================================

def create_timeseries_layout():
    """Crea el layout para la pestaña de evolución por país."""
    logger.debug("Creando layout: Evolución por País")

    total_or_static_metrics = [
        'total_cases', 'total_deaths', 'total_cases_per_100k', 'total_deaths_per_100k',
        'population'
    ]
    line_chart_metrics = {k: v for k, v in config.METRICS.items() if k not in total_or_static_metrics}
    flat_line_chart_options = [{"label": v, "value": k} for k, v in line_chart_metrics.items()]

    return html.Div([
        html.H2("Evolución de Métricas por País", style={'fontSize':'22px','fontWeight':'700','color':config.COLORS['text_primary'],'marginBottom':'20px'}),

        html.Div([
            html.Div([
                html.Label("País Seleccionado", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Dropdown(
                    id='ts-country-selector',
                    options=[], value=None, clearable=False,
                    persistence=True, persistence_type='local'
                )
            ], style={'width':'30%', 'marginRight':'15px'}),
            html.Div([
                html.Label("Métricas a Graficar", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Dropdown(
                    id='ts-metrics-selector',
                    options=flat_line_chart_options,
                    value=['new_cases_smoothed', 'new_deaths_smoothed'],
                    multi=True, placeholder="Selecciona métricas...",
                    persistence=True, persistence_type='local'
                )
            ], style={'width':'45%', 'marginRight':'15px'}),
            html.Div([
                html.Label("Rango de Fechas", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.DatePickerRange(
                    id='ts-date-picker',
                    min_date_allowed=date(2020, 1, 1), max_date_allowed=date.today(),
                    start_date=date(2020, 3, 1), end_date=date.today(),
                    display_format='YYYY-MM-DD', style={'width': '100%'}
                )
            ], style={'width':'25%'})
        ], style={**get_card_style('15px'), 'display':'flex', 'alignItems':'flex-end', 'marginBottom':'10px'}),

        html.Div([
             html.Div([
                dcc.Checklist(
                    id='ts-log-scale', options=[{'label':' Usar escala logarítmica','value':'log'}],
                    value=[], style={'marginTop':'8px'}
                )
             ], style={'flex':'1'}),
             html.Div([
                html.Button("Descargar CSV", id="btn-download-ts", n_clicks=0, style={'padding':'8px 12px', 'cursor':'pointer'}),
                dcc.Download(id="download-timeseries-csv")
             ], style={'textAlign':'right'})
        ], style={'display':'flex', 'alignItems':'center', 'marginBottom':'20px', 'padding':'0 10px'}),

        html.Div([
            html.H3("📈 Gráfico de Series de Tiempo", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
            dcc.Loading(id='loading-timeseries-chart', type='default',
                children=html.Div(dcc.Graph(id='timeseries-line-chart', figure=go.Figure()))
            )
        ], style=get_card_style())
    ])

def create_comparisons_layout():
    logger.debug("Creando layout: Comparaciones")
    return html.Div([
        html.H2("Comparación entre Países", style={'fontSize':'22px','fontWeight':'700','color':config.COLORS['text_primary'],'marginBottom':'20px'}),

        html.Div([
            html.Div([
                html.Label("Métrica", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Dropdown(
                    id='comparison-metric-selector', options=[{'label':v,'value':k} for k,v in config.METRICS.items()],
                    value='total_cases', clearable=False,
                    persistence=True, persistence_type='local'
                )
            ], style={'width':'40%', 'marginRight':'15px'}),
            html.Div([
                html.Label("Selecciona Países", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Dropdown(
                    id='comparison-locations-selector', options=[], value=None, multi=True,
                    placeholder="Selecciona países...",
                    persistence=True, persistence_type='local'
                )
            ], style={'width':'60%'}),
        ], style={**get_card_style('15px'), 'display':'flex', 'alignItems':'flex-end'}),

        html.Div([
                html.Button("Descargar CSV", id="btn-download-comp", n_clicks=0, style={'padding':'8px 12px', 'cursor':'pointer', 'marginTop':'-10px'}),
                dcc.Download(id="download-comparison-csv")
        ], style={'textAlign':'right', 'paddingRight':'10px', 'marginBottom':'10px'}),

        html.Div([
            html.H3("📊 Comparación", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
            dcc.Loading(id='loading-comparison-chart', type='default', children=html.Div(id='comparison-chart', children=dcc.Graph(figure=go.Figure())))
        ], style=get_card_style())
    ])

def create_statistics_layout():
    logger.debug("Creando layout: Estadísticas")
    return html.Div([
        html.H2("Análisis Estadístico Global", style={'fontSize':'22px','fontWeight':'700','color':config.COLORS['text_primary'],'marginBottom':'20px'}),
        html.Div([
            html.Div([
                html.Label("Métrica", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Dropdown(
                    id='stats-metric-selector', options=[{'label':v,'value':k} for k,v in config.METRICS.items() if k != 'population'],
                    value='total_cases', clearable=False,
                    persistence=True, persistence_type='local'
                )
            ], style={'width':'60%', 'marginRight':'15px'}),
            html.Div([
                html.Label("Outliers", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                dcc.Checklist(
                    id='stats-include-outliers', options=[{'label':' Incluir outliers','value':'yes'}],
                    value=[], style={'marginTop':'8px'}
                )
            ], style={'width':'40%'})
        ], style={**get_card_style('15px'), 'display':'flex', 'alignItems':'flex-end', 'marginBottom':'20px'}),
        html.Div([
            html.H3("📊 Estadísticas Descriptivas Globales", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
            dcc.Loading(id='loading-statistics', type='default', children=html.Div(id='statistics-display'))
        ], style=get_card_style())
    ])

def create_correlations_layout():
     logger.debug("Creando layout: Correlaciones")
     return html.Div([
        html.H2("Análisis de Correlaciones Globales", style={'fontSize':'22px','fontWeight':'700','color':config.COLORS['text_primary'],'marginBottom':'20px'}),
        html.Div([ html.P("Analiza las relaciones globales entre métricas a nivel de país.", style={'fontSize':'13px','color':config.COLORS['text_secondary'],'marginBottom':'15px'}) ], className='info-message'),
        html.Div([
             html.Div([
                 html.Label("Métricas a Comparar", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                 dcc.Dropdown(
                     id='correlation-metrics-selector', options=[{'label':v,'value':k} for k,v in config.METRICS.items()],
                     value=['total_cases_per_100k', 'total_deaths_per_100k', 'mortality_rate'],
                     multi=True, placeholder="Selecciona >= 2 métricas...",
                     persistence=True, persistence_type='local'
                 )
             ], style={'width':'70%', 'marginRight':'15px'}),
             html.Div([
                 html.Label("Método", style={'fontSize':'12px','fontWeight':'600','marginBottom':'5px'}),
                 dcc.Dropdown(
                     id='correlation-method', options=[{'label':'Pearson','value':'pearson'},{'label':'Spearman','value':'spearman'}],
                     value='pearson', clearable=False,
                     persistence=True, persistence_type='local'
                 )
             ], style={'width':'30%'})
        ], style={**get_card_style('15px'), 'display':'flex', 'alignItems':'flex-end', 'marginBottom':'20px'}),
        html.Div([
            html.H3("🔗 Matriz de Correlación Global", style={'fontSize':'16px','fontWeight':'600','marginBottom':'15px','color':config.COLORS['text_primary']}),
            dcc.Loading(id='loading-correlation-matrix', type='default', children=html.Div(id='correlation-matrix-heatmap', children=dcc.Graph(figure=go.Figure())))
        ], style=get_card_style())
    ])

# ============================================================================
# CALLBACKS DE VISUALIZACIÓN
# ============================================================================

# --- CALLBACKS PARA POBLAR SELECTORES ---
@app.callback(
    Output('ts-country-selector', 'options'),
    Output('ts-country-selector', 'value'),
    Input('store-countries', 'data'),
    State('ts-country-selector', 'value')
)
def update_timeseries_country_selector(countries_data, persisted_value):
    if not countries_data:
        logger.warning("No hay datos de países en store para poblar selector ts-country.")
        return [], None

    options = [{'label': c, 'value': c} for c in countries_data]

    current_value = persisted_value if persisted_value and persisted_value in countries_data else None

    if not current_value:
        current_value = 'Ecuador' if 'Ecuador' in countries_data else (countries_data[0] if countries_data else None)

    logger.debug(f"Actualizando selector ts-country. Opciones: {len(options)}, Valor: {current_value}")
    return options, current_value

@app.callback(
    Output('comparison-locations-selector', 'options'),
    Output('comparison-locations-selector', 'value'),
    Input('store-countries', 'data'),
    State('comparison-locations-selector', 'value')
)
def update_comparison_country_selector(countries_data, persisted_value):
    if not countries_data:
        logger.warning("No hay datos de países en store para poblar selector comparison-locations.")
        return [], []

    options = [{'label': c, 'value': c} for c in countries_data]

    current_value = []
    if persisted_value and isinstance(persisted_value, list): # Asegurar que es lista
        current_value = [c for c in persisted_value if c in countries_data]

    if not current_value:
        default_value_list = ['Ecuador', 'Peru', 'Colombia']
        current_value = [c for c in default_value_list if c in countries_data]
        if not current_value and countries_data:
            current_value = [countries_data[0]]

    logger.debug(f"Actualizando selector comparison-locations. Opciones: {len(options)}, Valor: {current_value}")
    return options, current_value


# ========================================================================
# >>>>> INICIO DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================
@app.callback(
    Output('global-metrics-cards','children'),
    Input('interval-component','n_intervals') # Solo depende del intervalo
)
def update_global_metrics(n):
    # Solo se necesita una llamada a la API
    data = api.get_global_summary()

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        return html.Div(f"⚠️ No se pudieron cargar métricas globales: {error_msg}", className='warning-message')

    # Obtener todos los valores de la única respuesta
    tc=data.get('total_cases',0)
    td=data.get('total_deaths',0)
    ca=data.get('countries_affected',0)
    tp=data.get('total_population', 0) # Obtener población desde la API

    # La llamada separada a get_map_data('population') se elimina

    return html.Div([
        html.Div([html.Div([html.Span("😷",style={'fontSize':'24px'}),html.Div([html.Div("CASOS TOTALES",className='stat-header'),html.Div(fmt(tc),className='stat-value',style={'color':config.COLORS['accent_blue']}),html.Div(f"{ca} países/territorios",className='stat-subvalue')])],style={'display':'flex','alignItems':'center','gap':'15px'})], style={**get_card_style('20px'),'flex':'1','marginRight':'15px'}),
        html.Div([html.Div([html.Span("💀",style={'fontSize':'24px'}),html.Div([html.Div("MUERTES TOTALES",className='stat-header'),html.Div(fmt(td),className='stat-value',style={'color':config.COLORS['accent_red']}),html.Div(f"Tasa: {(td/tc*100):.2f}%" if tc and tc>0 else "N/A",className='stat-subvalue')])],style={'display':'flex','alignItems':'center','gap':'15px'})], style={**get_card_style('20px'),'flex':'1','marginRight':'15px'}),
        html.Div([html.Div([html.Span("👥",style={'fontSize':'24px'}),html.Div([html.Div("POBLACIÓN TOTAL",className='stat-header'),html.Div(fmt(tp),className='stat-value',style={'color':config.COLORS['accent_green']}),html.Div("Suma de países",className='stat-subvalue')])],style={'display':'flex','alignItems':'center','gap':'15px'})], style={**get_card_style('20px'),'flex':'1'})
    ], style={'display':'flex','marginBottom':'20px'})
# ========================================================================
# >>>>> FIN DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================

# ========================================================================
# >>>>> CALLBACK MODIFICADO (SIN INPUTS DE FECHA) <<<<<
# ========================================================================
@app.callback(
    Output('world-map','children'),
    Input('map-metric-selector','value') # Solo depende de la métrica
)
def update_world_map(metric):
    if not metric: return no_update

    # No se pide top=10 aquí, se quiere el mapa completo
    data = api.get_map_data(metric)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        return html.Div(f"⚠️ No hay datos de mapa: {error_msg}", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    if 'data' not in data:
         return html.Div(f"⚠️ Respuesta inesperada de API para mapa: {data}", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    map_data = data['data'];
    if not map_data: return html.Div("⚠️ Sin datos para esta métrica", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    df = pd.DataFrame(map_data);
    if df.empty or 'value' not in df.columns or df['value'].isnull().all():
         return html.Div("⚠️ Datos insuficientes para generar el mapa", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    fig = px.choropleth(df, locations='iso_code', color='value', hover_name='country', color_continuous_scale='Blues', labels={'value':config.METRICS.get(metric, metric)})
    fig.update_layout(geo=dict(showframe=False,showcoastlines=True,projection_type='equirectangular'), margin=dict(l=0,r=0,t=0,b=0), height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'])
    return dcc.Graph(figure=fig, config={'displayModeBar':False})
# ========================================================================
# >>>>> FIN CALLBACK MODIFICADO <<<<<
# ========================================================================

# ========================================================================
# >>>>> INICIO DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================
@app.callback(
    Output('top-countries-chart','children'),
    Input('map-metric-selector','value') # Solo depende de la métrica
)
def update_top_countries(metric):
    if not metric: return no_update

    # Pedir explícitamente el TOP 10 a la API
    data = api.get_map_data(metric, top=10)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        return html.Div(f"⚠️ No hay datos de ranking: {error_msg}", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    if 'data' not in data:
         return html.Div(f"⚠️ Respuesta inesperada de API para ranking: {data}", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    ranking_data = data['data']; # Esto ya es el Top 10, ordenado
    if not ranking_data: return html.Div("⚠️ Sin datos para métrica", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    # Ya no se necesita Pandas para ordenar o filtrar
    # Los datos vienen de la API como [{'country': 'USA', 'value': 100}, ...]
    
    if not ranking_data:
        return html.Div("⚠️ Sin datos suficientes para el Top 10", style={'textAlign':'center','padding':'40px','color':config.COLORS['text_secondary']})

    # Extraer listas directamente. La API los devuelve ordenados (el más alto primero)
    locations = [item['country'] for item in ranking_data]
    values = [item['value'] for item in ranking_data]

    fig = go.Figure([go.Bar(
        x=values, 
        y=locations, 
        orientation='h', 
        marker=dict(color=values,colorscale='Blues',showscale=False), 
        text=[fmt(v) for v in values], 
        textposition='outside', 
        hovertemplate='<b>%{y}</b><br>%{x:,.0f}<extra></extra>'
    )])
    
    # 'autorange': 'reversed' asegura que el primer item (el más alto) esté en la parte superior
    fig.update_layout(
        margin=dict(l=150,r=50,t=20,b=40), 
        height=400, 
        paper_bgcolor=config.COLORS['bg_card'], 
        plot_bgcolor=config.COLORS['bg_card'], 
        xaxis=dict(showgrid=True,gridcolor=config.COLORS['grid'],tickformat=','), 
        yaxis=dict(showgrid=False, autorange='reversed'), 
        font=dict(size=11)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar':False})
# ========================================================================
# >>>>> FIN DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================


@app.callback(
    Output('comparison-chart', 'children'),
    [Input('comparison-metric-selector', 'value'),
     Input('comparison-locations-selector', 'value')]
)
def update_comparison_chart(metric, locations):
    if not metric:
        logger.warning("Callback update_comparison_chart: Métrica no seleccionada.")
        return html.Div("⚠️ Selecciona una métrica", style={'textAlign':'center','padding':'40px'})
    if not locations:
        logger.warning("Callback update_comparison_chart: Países no seleccionados.")
        return html.Div("⚠️ Selecciona al menos 1 país", style={'textAlign':'center','padding':'40px'})

    logger.debug(f"Actualizando gráfico de comparación. Métrica: {metric}, Países: {locations}")

    data = api.compare_countries(countries=locations, metric=metric, normalize=False)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        logger.error(f"Error API en compare_countries: {error_msg}")
        return html.Div(f"⚠️ Error al obtener datos de comparación: {error_msg}", style={'textAlign': 'center', 'padding': '40px', 'color': config.COLORS['accent_red']})

    if 'data' not in data:
        logger.error(f"Respuesta inesperada de API /compare: {data}")
        return html.Div("⚠️ Respuesta inesperada de la API para comparación", style={'textAlign': 'center', 'padding': '40px'})

    comparison = data['data']
    if not comparison:
        logger.warning("Callback update_comparison_chart: API devolvió datos vacíos.")
        return html.Div("⚠️ Sin datos para comparar (posiblemente países no encontrados o métrica sin datos)", style={'textAlign':'center','padding':'40px'})

    fig = go.Figure()
    colors = [config.COLORS['accent_blue'], config.COLORS['accent_red'], config.COLORS['accent_green'], config.COLORS['accent_yellow'], config.COLORS['accent_purple'], config.COLORS['accent_teal']]

    valid_data_found = False
    for i, item in enumerate(comparison):
        location = item.get('location', 'N/A')
        value_key = 'value'
        value = item.get(value_key)
        if value is not None:
             valid_data_found = True
             fig.add_trace(go.Bar(
                 x=[location],
                 y=[value],
                 name=location,
                 marker_color=colors[i%len(colors)],
                 text=fmt(value),
                 textposition='outside',
                 hovertemplate=f'<b>{location}</b><br>{config.METRICS.get(metric, metric)}: %{{y:,.2f}}<extra></extra>'
            ))
        else:
            logger.warning(f"Valor nulo para {metric} en {location} durante comparación.")

    if not valid_data_found:
        logger.warning("Callback update_comparison_chart: No se encontraron datos válidos para graficar.")
        return html.Div("⚠️ No se encontraron datos válidos para la métrica y países seleccionados.", style={'textAlign':'center','padding':'40px'})

    fig.update_layout(
        title=dict(text=f"<b>Comparación: {config.METRICS.get(metric, metric)}</b>",x=0.5,xanchor='center',font=dict(size=14)),
        margin=dict(l=60,r=30,t=50,b=40), height=450,
        paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor=config.COLORS['grid'],tickformat=','),
        legend=dict(orientation='h',yanchor='bottom',y=-0.25,xanchor='center',x=0.5, traceorder='reversed'),
        font=dict(size=11),
        barmode='group'
    )
    return dcc.Graph(figure=fig, config={'displayModeBar':False})

@app.callback(
    Output('statistics-display', 'children'),
    [Input('stats-metric-selector', 'value'),
     Input('stats-include-outliers', 'value')]
)
def update_statistics(metric, include_outliers):
    if not metric: return no_update
    include_outliers_flag = len(include_outliers) > 0
    data = api.get_statistical_summary(metric=metric, include_outliers=include_outliers_flag)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        return html.Div(f"⚠️ No hay datos estadísticos: {error_msg}", style={'textAlign':'center','padding':'40px'})

    if 'statistics' not in data:
         return html.Div(f"⚠️ Respuesta inesperada de API para estadísticas: {data}", style={'textAlign':'center','padding':'40px'})

    statistics = data['statistics']
    if not statistics or 'global' not in statistics: return html.Div("⚠️ Sin datos globales", style={'textAlign':'center','padding':'40px'})

    stats = statistics['global']
    card = html.Div([
        html.Div([
             html.Div([html.Span("Media: ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('mean')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Mediana: ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('median')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Desv. Std: ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('std')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Mín: ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('min')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Máx: ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('max')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Q1 (25%): ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('q25')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("Q3 (75%): ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(fmt(stats.get('q75')),style={'fontWeight':'600','fontSize':'12px'})],style={'marginBottom':'5px'}),
             html.Div([html.Span("N (Países): ",style={'color':config.COLORS['text_secondary'],'fontSize':'11px'}),html.Span(str(stats.get('count',0)),style={'fontWeight':'600','fontSize':'12px'})])
        ])
    ], style={**get_card_style('15px'), 'marginRight':'15px', 'marginBottom':'15px'})
    return card

@app.callback(
    Output('correlation-matrix-heatmap', 'children'),
    [Input('correlation-metrics-selector', 'value'),
     Input('correlation-method', 'value')]
)
def update_correlations(metrics, method):
    if not metrics or len(metrics) < 2: return html.Div("⚠️ Selecciona >= 2 métricas", style={'textAlign':'center','padding':'40px'})

    logger.debug(f"Actualizando matriz de correlación. Métricas: {metrics}, Método: {method}")
    data = api.get_correlations(metrics=metrics, method=method)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        logger.error(f"Error API en get_correlations: {error_msg}")
        return html.Div(f"⚠️ No hay datos de correlación: {error_msg}", style={'textAlign':'center','padding':'40px'})

    if 'correlation_matrix' not in data:
        logger.error(f"Respuesta inesperada de API para correlaciones: {data}")
        return html.Div(f"⚠️ Respuesta inesperada de API para correlaciones: {data}", style={'textAlign':'center','padding':'40px'})

    corr_matrix = data['correlation_matrix']
    if not corr_matrix:
        logger.warning("Callback update_correlations: Matriz de correlación vacía.")
        return html.Div("⚠️ No se pudo calcular la matriz de correlación (datos insuficientes?).", style={'textAlign':'center','padding':'40px'})

    metric_names = [config.METRICS.get(m, m) for m in metrics]
    z_values = []

    for m1 in metrics:
        row = []
        inner_dict = corr_matrix.get(m1, {})
        for m2 in metrics:
            value = inner_dict.get(m2, None)
            row.append(value)
        z_values.append(row)

    fig_heatmap = go.Figure(data=go.Heatmap(z=z_values, x=metric_names, y=metric_names, colorscale='RdBu', zmid=0,
    text=[[f"{val:.2f}" if val is not None else "N/A" for val in row] for row in z_values],
    texttemplate='%{text}', textfont={"size":12}, colorbar=dict(title="Corr.")))
    fig_heatmap.update_layout(title=dict(text=f"<b>Matriz de Correlación Global ({method.title()})</b>",x=0.5,xanchor='center',font=dict(size=14)), margin=dict(l=100,r=50,t=50,b=100), height=400, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], xaxis=dict(tickangle=-45), font=dict(size=11))
    return dcc.Graph(figure=fig_heatmap, config={'displayModeBar':False})


# ========================================================================
# >>>>> INICIO DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================
@app.callback(
    Output('timeseries-line-chart', 'figure'),
    [Input('ts-country-selector', 'value'),
     Input('ts-metrics-selector', 'value'),
     Input('ts-date-picker', 'start_date'),
     Input('ts-date-picker', 'end_date'),
     Input('ts-log-scale', 'value')]
)
def update_timeseries_chart(country, metrics, start_date, end_date, log_scale):
    """Actualiza el gráfico de líneas de evolución basado en la selección."""

    triggered_id = callback_context.triggered_id
    logger.debug(f"Update TS chart triggered by: {triggered_id}")

    # Si el país es None, mostrar mensaje inicial
    if country is None:
        logger.warning("Callback update_timeseries_chart: País es None.")
        fig_empty = go.Figure()
        fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text="Cargando país...", showarrow=False)])
        return fig_empty

    # Si no hay métricas seleccionadas
    if not metrics:
        logger.warning(f"Callback update_timeseries_chart (País: {country}): Métricas no seleccionadas.")
        fig_empty = go.Figure()
        fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text="Selecciona al menos una métrica", showarrow=False)])
        return fig_empty

    logger.debug(f"Actualizando gráfico TS. País: {country}, Métricas: {metrics}, Fechas: {start_date}-{end_date}, Log: {log_scale}")

    # --- Cargar datos ---
    # La API ahora recibe las fechas y filtra los datos
    data = api.get_country_timeseries_all(country, start_date=start_date, end_date=end_date)

    if not data or isinstance(data, dict) and data.get("error"):
        error_msg = data.get("error", "Error desconocido") if isinstance(data, dict) else "Error desconocido"
        logger.error(f"Error API en get_country_timeseries_all para {country}: {error_msg}")
        fig_empty = go.Figure()
        fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text=f"No se pudieron cargar datos para {country}: {error_msg}", showarrow=False)])
        return fig_empty

    if 'data' not in data:
         logger.error(f"Respuesta inesperada de API para TS de {country}: {data}")
         fig_empty = go.Figure()
         fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text=f"Respuesta inesperada de API para {country}", showarrow=False)])
         return fig_empty

    # --- Procesar DataFrame ---
    try:
        df = pd.DataFrame(data['data'])
        if df.empty: raise ValueError("DataFrame vacío después de cargar (o sin datos en el rango)")

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values(by='date')

        # --- EL FILTRADO DE FECHAS SE ELIMINA DE AQUÍ ---
        # La API ya ha filtrado los datos por start_date y end_date
        # if df.empty: raise ValueError("DataFrame vacío después de filtrar fecha") # <-- ELIMINADO

    except Exception as e:
        logger.error(f"Error procesando DataFrame para {country}: {e}")
        fig_empty = go.Figure()
        fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text=f"Error procesando datos o sin datos en el rango para {country}", showarrow=False)])
        return fig_empty

    # --- Crear Figura ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [config.COLORS['accent_blue'], config.COLORS['accent_red'], config.COLORS['accent_green'],
              config.COLORS['accent_yellow'], config.COLORS['accent_purple'], config.COLORS['accent_teal']]
    has_primary_axis = False
    has_secondary_axis = False
    valid_traces_added = False

    # --- Añadir Trazas ---
    for i, metric in enumerate(metrics):
        if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]) or df[metric].isnull().all():
            logger.warning(f"Métrica '{metric}' no encontrada, no numérica o sin datos válidos para {country}.")
            continue

        color = colors[i % len(colors)]
        metric_name = config.METRICS.get(metric, metric)
        is_secondary = metric in config.SECONDARY_AXIS_METRICS
        if is_secondary: has_secondary_axis = True
        else: has_primary_axis = True
        valid_traces_added = True

        # Gráfico Combinado: Casos
        if metric == 'new_cases_smoothed' and 'new_cases' in df.columns and pd.api.types.is_numeric_dtype(df['new_cases']) and not df['new_cases'].isnull().all():
            fig.add_trace(go.Bar(x=df['date'], y=df['new_cases'], name=config.METRICS.get('new_cases'), marker_color=color, opacity=0.3, hovertemplate=f"<b>{config.METRICS.get('new_cases')}</b><br>%{{x|%Y-%m-%d}}<br>Valor: %{{y:,.0f}}<extra></extra>"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df['date'], y=df[metric], name=metric_name, line=dict(color=color, width=2.5), hovertemplate=f"<b>{metric_name}</b><br>%{{x|%Y-%m-%d}}<br>Valor: %{{y:,.1f}}<extra></extra>"), secondary_y=False)
        # Gráfico Combinado: Muertes
        elif metric == 'new_deaths_smoothed' and 'new_deaths' in df.columns and pd.api.types.is_numeric_dtype(df['new_deaths']) and not df['new_deaths'].isnull().all():
            fig.add_trace(go.Bar(x=df['date'], y=df['new_deaths'], name=config.METRICS.get('new_deaths'), marker_color=color, opacity=0.3, hovertemplate=f"<b>{config.METRICS.get('new_deaths')}</b><br>%{{x|%Y-%m-%d}}<br>Valor: %{{y:,.0f}}<extra></extra>"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df['date'], y=df[metric], name=metric_name, line=dict(color=color, width=2.5), hovertemplate=f"<b>{metric_name}</b><br>%{{x|%Y-%m-%d}}<br>Valor: %{{y:,.1f}}<extra></extra>"), secondary_y=False)
        # Ignorar 'raw' si 'smoothed' está seleccionado
        elif not ((metric == 'new_cases' and 'new_cases_smoothed' in metrics) or \
                  (metric == 'new_deaths' and 'new_deaths_smoothed' in metrics)):
            fig.add_trace(go.Scatter(x=df['date'], y=df[metric], name=metric_name, line=dict(color=color, width=2.5), hovertemplate=f"<b>{metric_name}</b><br>%{{x|%Y-%m-%d}}<br>Valor: %{{y:,.2f}}<extra></extra>"), secondary_y=is_secondary)

    if not valid_traces_added:
        logger.warning(f"Callback update_timeseries_chart (País: {country}): No se añadieron trazas válidas.")
        fig_empty = go.Figure()
        fig_empty.update_layout(height=450, paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'], annotations=[dict(text=f"No hay datos válidos para las métricas seleccionadas en {country}", showarrow=False)])
        return fig_empty

    # --- Estilizar Figura ---
    yaxis_type = 'log' if log_scale else 'linear'
    fig.update_layout(
        title=dict(text=f"<b>Evolución en {country}</b>", x=0.5, xanchor='center', font=dict(size=14)),
        margin=dict(l=60, r=40, t=50, b=40), height=450,
        paper_bgcolor=config.COLORS['bg_card'], plot_bgcolor=config.COLORS['bg_card'],
        xaxis=dict(showgrid=True, gridcolor=config.COLORS['grid']),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
        font=dict(size=11), hovermode='x unified', barmode='overlay'
    )
    fig.update_yaxes(type=yaxis_type, secondary_y=False, title_standoff = 5)
    fig.update_yaxes(type=yaxis_type, secondary_y=True, title_standoff = 10)
    fig.update_yaxes(title_text="Conteo / Personas", showgrid=True, gridcolor=config.COLORS['grid'], secondary_y=False, visible=has_primary_axis)
    fig.update_yaxes(title_text="Tasa / Índice (%)", showgrid=False, secondary_y=True, visible=has_secondary_axis)

    return fig
# ========================================================================
# >>>>> FIN DE MODIFICACIÓN (DASHBOARD) <<<<<
# ========================================================================


# --- CALLBACKS DE DESCARGA (Modificados para usar la API con filtros) ---
@app.callback(
    Output("download-timeseries-csv", "data"),
    Input("btn-download-ts", "n_clicks"),
    [State('ts-country-selector', 'value'),
     State('ts-metrics-selector', 'value'),
     State('ts-date-picker', 'start_date'),
     State('ts-date-picker', 'end_date')],
    prevent_initial_call=True,
)
def download_timeseries_data(n_clicks, country, metrics, start_date, end_date):
    if not country or not metrics:
        return no_update

    # Llamar a la API CON los filtros de fecha
    data = api.get_country_timeseries_all(country, start_date=start_date, end_date=end_date)
    
    if not data or 'data' not in data:
        return no_update

    df = pd.DataFrame(data['data'])
    
    # Ya no es necesario filtrar por fecha aquí, la API lo hizo
    # df['date'] = pd.to_datetime(df['date'])
    # if start_date: df = df[df['date'] >= pd.to_datetime(start_date)]
    # if end_date: df = df[df['date'] <= pd.to_datetime(end_date)]

    download_metrics = list(metrics)
    if 'new_cases_smoothed' in metrics and 'new_cases' not in download_metrics:
        download_metrics.append('new_cases')
    if 'new_deaths_smoothed' in metrics and 'new_deaths' not in download_metrics:
        download_metrics.append('new_deaths')

    cols_to_keep = ['date', 'iso_code', 'location'] + download_metrics
    df_filtered = df[[c for c in cols_to_keep if c in df.columns]]

    filename = f"timeseries_{country}_{start_date}_to_{end_date}.csv"
    return dcc.send_data_frame(df_filtered.to_csv, filename, index=False)

@app.callback(
    Output("download-comparison-csv", "data"),
    Input("btn-download-comp", "n_clicks"),
    [State('comparison-metric-selector', 'value'),
     State('comparison-locations-selector', 'value')],
    prevent_initial_call=True,
)
def download_comparison_data(n_clicks, metric, locations):
    if not metric or not locations:
        return no_update

    data = api.compare_countries(countries=locations, metric=metric, normalize=False)

    if not data or 'data' not in data:
        return no_update

    df = pd.DataFrame(data['data'])
    filename = f"comparison_{metric}_raw.csv"
    return dcc.send_data_frame(df.to_csv, filename, index=False)

# --- Utilidad fmt (Sin cambios) ---
def fmt(num: Optional[Union[float, int, str]]) -> str:
    if num is None: return 'N/A';
    try: num = float(num)
    except (ValueError,TypeError,AttributeError): return 'N/A'
    if abs(num)>=1e9: return f"{num/1e9:.1f}B";
    if abs(num)>=1e6: return f"{num/1e6:.1f}M";
    if abs(num)>=1e4: return f"{num/1e3:.0f}K";
    if abs(num)>=1e3: return f"{num/1e3:.1f}K";
    try:
      if not pd.isna(num) and num != float('inf') and num != float('-inf'):
          if float(num).is_integer():
              return f"{int(num):,}"
          else:
             if abs(num) < 0.01 and num != 0: return f"{num:.4f}"
             if abs(num) < 1: return f"{num:.3f}"
             if abs(num) < 10: return f"{num:.2f}"
             return f"{num:,.1f}"
      else:
            return 'N/A'
    except Exception as e:
        logger.error(f"Error en función fmt con número {num}: {e}")
        return str(num)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70); print(f"🚀 Iniciando Panel COVID-19 (Países) v6.6 (Lógica API Corregida)"); print(f"API URL: {config.API_BASE_URL}"); print("="*70)
    print(f"\n🌍 Dashboard: http://127.0.0.1:8050"); print(f"💡 API debe estar en: {config.API_BASE_URL}")
    print("\n✨ Funcionalidades: Vista General, Evolución (Doble Eje/Barras), Comparación, Estadísticas, Correlaciones.")
    print("\n⏳ CTRL+C para detener"); print("="*70 + "\n")
    try: app.run(debug=True, host='127.0.0.1', port=8050)
    except KeyboardInterrupt: print("\n✅ Dashboard detenido.")
    except Exception as e:
        logger.exception("❌ Error fatal durante la ejecución")
        print(f"❌ Error: {e}")