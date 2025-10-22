"""
Dashboard COVID-19 Profesional - PRODUCTION VERSION
===================================================
Dashboard interactivo con Dash y Plotly optimizado para producción.
"""

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
import plotly.express as px
import requests
import logging
from functools import lru_cache
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
import time
import sys

# Variables de entorno
try:
    from decouple import config as env_config
except ImportError:
    print("ERROR: python-decouple no instalado. Ejecuta: pip install python-decouple")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN Y LOGGING
# ============================================================================

logging.basicConfig(
    level=env_config("LOG_LEVEL", default="INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CovidDashboard")


class Config:
    """Configuración del Dashboard desde variables de entorno."""

    API_BASE_URL: str = env_config("API_BASE_URL", default="http://127.0.0.1:8000")
    API_KEY: str = env_config("API_KEY", default="")

    DEFAULT_COUNTRY: str = env_config("DEFAULT_COUNTRY", default="Ecuador")
    UPDATE_INTERVAL_MS: int = env_config("UPDATE_INTERVAL_MS", default=300000, cast=int)  # 5 minutos
    MAX_COMPARE_COUNTRIES: int = 5

    # Timeout para requests
    REQUEST_TIMEOUT: int = 15

    # Color Palette (Light Theme)
    COLORS: Dict[str, str] = {
        'bg_dark': '#f8f9fa',
        'bg_card': '#ffffff',
        'bg_card_hover': '#f0f0f0',
        'text_primary': '#1a1a1a',
        'text_secondary': '#666666',
        'accent_blue': '#2563eb',
        'accent_green': '#10b981',
        'accent_red': '#ef4444',
        'accent_yellow': '#f59e0b',
        'accent_purple': '#8b5cf6',
        'border': '#e5e7eb',
        'grid': '#f0f0f0'
    }


config = Config()


# ============================================================================
# CLIENTE API CON AUTENTICACIÓN
# ============================================================================

class CovidAPI:
    """Cliente HTTP para la API COVID-19 con autenticación."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
            logger.info("🔑 API Key configurada")
        else:
            logger.warning("⚠️ Sin API Key configurada (modo desarrollo)")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Realiza request HTTP con manejo de errores."""
        url = f"{self.base_url}{endpoint}"
        try:
            logger.debug(f"Request: {endpoint} - Params: {params}")
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en request a {endpoint}")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("❌ Error de autenticación: API Key inválida o faltante")
            elif e.response.status_code == 429:
                logger.warning("⚠️ Rate limit excedido")
            else:
                logger.error(f"HTTP Error {e.response.status_code} en {endpoint}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en request a {endpoint}: {e}")
            return None

    @lru_cache(maxsize=1)
    def get_countries(self) -> Union[List, Dict[str, Any]]:
        """Obtiene lista de países (cacheado)."""
        data = self._make_request("/covid/countries")
        # CORRECCIÓN: Devuelve dict en éxito, o lista vacía en fallo
        return data if data else []

    @lru_cache(maxsize=2)
    def get_continent_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de continentes (cacheado)."""
        data = self._make_request("/covid/metrics/continents")
        return data if data else {}

    @lru_cache(maxsize=10)
    def get_map_data(self, metric: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos para mapa."""
        return self._make_request("/covid/map-data", params={'metric': metric})

    @lru_cache(maxsize=200)
    def get_summary(self, country: str) -> Optional[Dict[str, Any]]:
        """Obtiene resumen de país (cacheado)."""
        return self._make_request("/covid/summary", params={'country': country})

    def get_timeseries(self, country: str, metric: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Obtiene serie temporal."""
        params = {'country': country, 'metric': metric}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        return self._make_request("/covid/timeseries", params=params)

    def compare_countries(self, countries: List[str], metric: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Compara métrica entre países."""
        params = {'countries': countries, 'metric': metric}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        return self._make_request("/covid/compare", params=params)


# Global API client
api = CovidAPI(config.API_BASE_URL, config.API_KEY)


# ============================================================================
# INICIALIZAR DASH
# ============================================================================

app = dash.Dash(
    __name__,
    title="Panel COVID-19",
    suppress_callback_exceptions=True,
    external_stylesheets=[]
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            @keyframes pulse { 0%%, 100%% { opacity: 1; } 50%% { opacity: 0.5; } }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #f1f1f1; }
            ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
            .metric-table { width: 100%%; border-collapse: collapse; font-size: 11px; }
            .metric-table th, .metric-table td { padding: 4px 6px; text-align: left; border-bottom: 1px solid #f0f0f0; }
            .metric-table th { font-weight: 600; color: #666; }
            .metric-table td:nth-child(2), .metric-table td:nth-child(3) { text-align: right; font-weight: 500; }
            .metric-table tr:last-child td { border-bottom: none; }
            .metric-table-container { max-height: 200px; overflow-y: auto; }
            .error-message { background: #fee; border: 1px solid #fcc; padding: 15px; border-radius: 8px; color: #c33; margin: 20px; }
            .warning-message { background: #ffc; border: 1px solid #fc6; padding: 15px; border-radius: 8px; color: #963; margin: 20px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# ============================================================================
# ESTILOS
# ============================================================================

STAT_CARD_STYLE: Dict[str, Any] = {
    'backgroundColor': config.COLORS['bg_card'],
    'padding': '25px',
    'borderRadius': '12px',
    'textAlign': 'center',
    'border': f"1px solid {config.COLORS['border']}",
    'transition': 'all 0.3s ease',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.08)'
}

CARD_STYLE: Dict[str, Any] = {
    'backgroundColor': config.COLORS['bg_card'],
    'padding': '20px',
    'borderRadius': '12px',
    'border': f"1px solid {config.COLORS['border']}",
    'marginBottom': '20px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.08)'
}


# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div(id="main-container", children=[
    # Header
    html.Header(style={
        'backgroundColor': config.COLORS['bg_card'],
        'padding': '20px 40px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'borderBottom': f"2px solid {config.COLORS['border']}",
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)'
    }, children=[
        html.Div([
            html.H1("PANEL COVID-19", style={
                'margin': '0',
                'fontSize': '28px',
                'fontWeight': '700',
                'color': config.COLORS['text_primary'],
                'letterSpacing': '2px'
            }),
            html.P("Análisis Global de la Pandemia", style={
                'margin': '5px 0 0 0',
                'fontSize': '14px',
                'color': config.COLORS['text_secondary'],
                'fontWeight': '300'
            })
        ]),
        html.Div([
            html.Div(id='live-indicator', style={
                'display': 'inline-block',
                'width': '10px',
                'height': '10px',
                'borderRadius': '50%',
                'backgroundColor': config.COLORS['accent_green'],
                'marginRight': '8px',
                'animation': 'pulse 2s infinite'
            }),
            html.Span("EN LINEA", style={
                'fontSize': '12px',
                'color': config.COLORS['accent_green'],
                'fontWeight': '600',
                'letterSpacing': '1px'
            }),
            html.Span(id='update-time', style={
                'marginLeft': '15px',
                'fontSize': '12px',
                'color': config.COLORS['text_secondary']
            })
        ], style={'display': 'flex', 'alignItems': 'center'})
    ]),

    # Main Content
    html.Div(style={
        'display': 'flex',
        'padding': '20px',
        'gap': '0',
        'minHeight': 'calc(100vh - 80px)'
    }, children=[
        # Sidebar
        html.Aside(style={'width': '320px', 'flexShrink': '0', 'paddingRight': '20px'}, children=[
            # Country Selector
            html.Div(style=CARD_STYLE, children=[
                html.Label("SELECCIONAR PAÍS", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id='country-select',
                    placeholder="🌍 Selecciona un país...",
                    value=config.DEFAULT_COUNTRY,
                    style={
                        'backgroundColor': config.COLORS['bg_card'],
                        'color': config.COLORS['text_primary']
                    }
                )
            ]),

            # Main Stats
            html.Div(id='main-stats', children=[
                html.Div("Cargando estadísticas...", style={
                    'color': config.COLORS['text_secondary'],
                    'textAlign': 'center',
                    'padding': '40px'
                })
            ]),

            # Continent Metrics
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICAS GLOBALES - CONTINENTES", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Loading(
                    id="loading-continent-metrics",
                    type="circle",
                    color=config.COLORS['accent_blue'],
                    children=html.Div(id='continent-metrics-box')
                )
            ]),

            # Trend Metric Selector
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICA DE TENDENCIA / COMPARACIÓN", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id='trend-metric-select',
                    options=[
                        {'label': '📈 Nuevos Casos (Promedio 7 días)', 'value': 'new_cases_smoothed'},
                        {'label': '💀 Nuevas Muertes (Promedio 7 días)', 'value': 'new_deaths_smoothed'},
                        {'label': '🦠 Casos Totales', 'value': 'total_cases'},
                        {'label': '☠️ Muertes Totales', 'value': 'total_deaths'},
                    ],
                    value='new_cases_smoothed',
                    clearable=False,
                    style={
                        'backgroundColor': config.COLORS['bg_card'],
                        'color': config.COLORS['text_primary']
                    }
                )
            ]),

            # Map Metric Selector
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICA DEL MAPA", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id='map-metric',
                    options=[
                        {'label': '🦠 Casos Totales', 'value': 'total_cases'},
                        {'label': '💀 Muertes Totales', 'value': 'total_deaths'},
                        {'label': '💉 Vacunados (Completos)', 'value': 'people_fully_vaccinated'},
                        {'label': '📊 Tasa de Vacunación (%)', 'value': 'vaccination_rate'},
                    ],
                    value='total_cases',
                    clearable=False,
                    style={
                        'backgroundColor': config.COLORS['bg_card'],
                        'color': config.COLORS['text_primary']
                    }
                )
            ]),

            # Date Range
            html.Div(style=CARD_STYLE, children=[
                html.Label("RANGO DE FECHAS", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2020-03-01',
                    end_date='2023-11-30', # Considera actualizar esta fecha o hacerla dinámica
                    min_date_allowed='2019-12-01',
                    max_date_allowed=date.today().isoformat(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'},
                    start_date_placeholder_text='Fecha inicio',
                    end_date_placeholder_text='Fecha fin'
                )
            ]),

            # Compare Countries
            html.Div(style=CARD_STYLE, children=[
                html.Label(f"COMPARAR PAÍSES (Máx {config.MAX_COMPARE_COUNTRIES})", style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id='compare-countries',
                    multi=True,
                    placeholder=f"🔄 Selecciona hasta {config.MAX_COMPARE_COUNTRIES} países...",
                    style={
                        'backgroundColor': config.COLORS['bg_card'],
                        'color': config.COLORS['text_primary']
                    }
                )
            ])
        ]),

        # Main Content Area
        html.Main(style={'flex': '1'}, children=[
            # Map
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Section(style={**CARD_STYLE, 'height': '500px'}, children=[
                    html.Label("MAPA MUNDIAL INTERACTIVO (Haz clic en un país)", style={
                        'fontSize': '11px',
                        'fontWeight': '600',
                        'color': config.COLORS['text_secondary'],
                        'letterSpacing': '1px',
                        'marginBottom': '10px',
                        'display': 'block'
                    }),
                    dcc.Loading(
                        id="loading-map",
                        type="circle",
                        color=config.COLORS['accent_blue'],
                        children=html.Div(id='map-container', style={'height': 'calc(100% - 30px)'})
                    )
                ])
            ]),

            # Charts Row
            html.Div(style={'display': 'flex'}, children=[
                # Trend Chart
                html.Section(style={
                    **CARD_STYLE,
                    'height': '350px',
                    'marginRight': '20px',
                    'flex': '1'
                }, children=[
                    dcc.Loading(
                        id="loading-trend",
                        type="circle",
                        color=config.COLORS['accent_blue'],
                        children=html.Div(id='trend-chart', style={'height': '100%'})
                    )
                ]),

                # Comparison Chart
                html.Section(style={
                    **CARD_STYLE,
                    'height': '350px',
                    'flex': '1'
                }, children=[
                    dcc.Loading(
                        id="loading-comparison",
                        type="circle",
                        color=config.COLORS['accent_blue'],
                        children=html.Div(id='comparison-chart', style={'height': '100%'})
                    )
                ])
            ])
        ])
    ]),

    # Interval for updates
    dcc.Interval(id='interval', interval=config.UPDATE_INTERVAL_MS, n_intervals=0)

], style={
    'backgroundColor': config.COLORS['bg_dark'],
    'color': config.COLORS['text_primary'],
    'minHeight': '100vh',
    'margin': '0',
    'padding': '0'
})


# ============================================================================
# CALLBACKS
# ============================================================================

# --- BLOQUE MODIFICADO PARA BÚSQUEDA POR ISO ---
@app.callback(
    [Output('country-select', 'options'),
     Output('compare-countries', 'options'),
     Output('update-time', 'children')],
    [Input('interval', 'n_intervals')]
)
def load_countries(n: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    """Carga lista de países (permitiendo búsqueda por nombre o ISO)."""
    logger.info(f"Callback: load_countries (interval={n})")
    start_time = time.time()

    # response_data será un DICT en caso de éxito, o una LISTA VACÍA [] en caso de fallo
    response_data = api.get_countries()
    timestamp = datetime.now().strftime('%H:%M:%S')

    countries_list = []
    options = []

    # 1. Comprobar si la respuesta es un diccionario (éxito de la API)
    if isinstance(response_data, dict):
        countries_list = response_data.get('countries', [])
    # Si no es un dict, response_data es [] (fallo de API), y countries_list permanecerá vacía

    # 2. Comprobar si la lista de países (extraída o vacía) tiene contenido
    if countries_list:
        # La 'label' incluye el nombre y el iso_code para la búsqueda.
        # El 'value' sigue siendo solo el nombre para compatibilidad.
        options = sorted(
            [
                {
                    'label': f"{c.get('name', 'Unknown')} ({c.get('iso_code', 'N/A')})",
                    'value': c.get('name')
                }
                for c in countries_list if c.get('name') and c.get('iso_code')
            ],
            key=lambda x: x['label']
        )
    else:
        logger.warning("No se pudieron cargar países (API falló o la lista está vacía)")
        options = [{'label': f"{config.DEFAULT_COUNTRY} (Default)", 'value': config.DEFAULT_COUNTRY}]


    elapsed = time.time() - start_time
    logger.info(f"load_countries completado en {elapsed:.2f}s, {len(options)} opciones generadas")

    return options, options, f"Actualizado: {timestamp}"
# --- FIN DE BLOQUE MODIFICADO ---


@app.callback(
    Output('main-stats', 'children'),
    [Input('country-select', 'value')]
)
def update_stats(country: Optional[str]) -> html.Div:
    """Actualiza estadísticas del país."""
    logger.info(f"Callback: update_stats (country={country})")

    if not country:
        country = config.DEFAULT_COUNTRY

    summary = api.get_summary(country)

    if not summary:
        return html.Div([
            html.Div("⚠️", style={'fontSize': '36px', 'marginBottom': '10px'}),
            html.Div(f"No hay datos para {country}", style={
                'color': config.COLORS['text_primary'],
                'fontSize': '13px'
            })
        ], style={'textAlign': 'center', 'paddingTop': '50px', **CARD_STYLE})

    return html.Div([
        # Cases Card
        html.Div(style={**STAT_CARD_STYLE, 'marginBottom': '15px'}, children=[
            html.Div("CASOS CONFIRMADOS", style={
                'fontSize': '11px',
                'fontWeight': '600',
                'color': config.COLORS['text_secondary'],
                'letterSpacing': '1px',
                'marginBottom': '10px'
            }),
            html.Div(fmt(summary.get('total_cases')), style={
                'fontSize': '36px',
                'fontWeight': '700',
                'color': config.COLORS['accent_blue'],
                'marginBottom': '5px',
                'lineHeight': '1'
            }),
            html.Div(f"Población: {fmt(summary.get('population'))}", style={
                'fontSize': '12px',
                'color': config.COLORS['text_secondary']
            })
        ]),

        # Deaths & Vaccinated Row
        html.Div(style={'display': 'flex', 'marginBottom': '15px'}, children=[
            html.Div(style={**STAT_CARD_STYLE, 'marginRight': '10px', 'flex': '1'}, children=[
                html.Div("MUERTES", style={
                    'fontSize': '10px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '8px'
                }),
                html.Div(fmt(summary.get('total_deaths')), style={
                    'fontSize': '24px',
                    'fontWeight': '700',
                    'color': config.COLORS['accent_red'],
                    'marginBottom': '3px'
                }),
                html.Div(f"Mortalidad: {summary.get('mortality_rate', 0):.2f}%", style={
                    'fontSize': '11px',
                    'color': config.COLORS['text_secondary']
                })
            ]),

            html.Div(style={**STAT_CARD_STYLE, 'flex': '1'}, children=[
                html.Div("VACUNADOS (COMPLETO)", style={
                    'fontSize': '10px',
                    'fontWeight': '600',
                    'color': config.COLORS['text_secondary'],
                    'letterSpacing': '1px',
                    'marginBottom': '8px'
                }),
                html.Div(f"{summary.get('vaccination_rate', 0):.1f}%", style={
                    'fontSize': '24px',
                    'fontWeight': '700',
                    'color': config.COLORS['accent_green'],
                    'marginBottom': '3px'
                }),
                html.Div(f"({fmt(summary.get('people_fully_vaccinated'))})", style={
                    'fontSize': '11px',
                    'color': config.COLORS['text_secondary']
                })
            ])
        ]),

        # Additional Info
        html.Div(style={
            'textAlign': 'center',
            'padding': '15px',
            'backgroundColor': '#f9fafb',
            'borderRadius': '8px',
            'border': f"1px solid {config.COLORS['border']}"
        }, children=[
            html.Div(f"📍 Continente: {summary.get('continent', 'N/A')}", style={
                'fontSize': '12px',
                'color': config.COLORS['text_secondary'],
                'marginBottom': '5px'
            }),
            html.Div(f"🗓️ Última Actualización: {summary.get('latest_date', 'N/A')}", style={
                'fontSize': '12px',
                'color': config.COLORS['text_secondary'],
                'marginBottom': '5px'
            }),
            html.Div(f"🌍 ISO Code: {summary.get('iso_code', 'N/A')}", style={
                'fontSize': '12px',
                'color': config.COLORS['text_secondary']
            })
        ])
    ])


@app.callback(
    Output('continent-metrics-box', 'children'),
    Input('interval', 'n_intervals')
)
def update_continent_metrics(n: int) -> html.Div:
    """Actualiza métricas de continentes."""
    logger.info(f"Callback: update_continent_metrics (interval={n})")

    data = api.get_continent_metrics()

    if not data:
        return html.Div("No hay datos de continentes disponibles.", style={
            'fontSize': '11px',
            'color': config.COLORS['text_secondary'],
            'padding': '20px 0'
        })

    try:
        sorted_items = sorted(
            data.items(),
            key=lambda item: item[1].get('total_cases', 0) if isinstance(item[1], dict) else 0,
            reverse=True
        )
        sorted_data = dict(sorted_items)
    except Exception as e:
        logger.error(f"Error ordenando datos de continentes: {e}")
        sorted_data = data

    return create_aggregate_table(sorted_data)


@app.callback(
    Output('map-container', 'children'),
    [Input('map-metric', 'value')]
)
def update_map(metric: Optional[str]) -> Union[dcc.Graph, html.Div]:
    """Actualiza mapa mundial."""
    logger.info(f"Callback: update_map (metric={metric})")

    if not metric:
        metric = 'total_cases'

    try:
        map_data = api.get_map_data(metric)

        if not map_data or not map_data.get('data'):
            return html.Div("⚠️ No hay datos disponibles para el mapa.", style={
                'textAlign': 'center',
                'padding': '100px 40px',
                'color': config.COLORS['text_secondary'],
                'fontSize': '14px'
            })

        locations = []
        values = []
        country_names = []

        for item in map_data['data']:
            iso = item.get('iso_code')
            val = item.get('value')
            country_name = item.get('country', 'N/A')

            if iso and val is not None:
                locations.append(iso)
                values.append(val)
                country_names.append(country_name)

        if not locations:
            return html.Div("⚠️ No hay países con datos válidos.", style={
                'textAlign': 'center',
                'padding': '100px 40px',
                'color': config.COLORS['text_secondary'],
                'fontSize': '14px'
            })

        fig = go.Figure(data=go.Choropleth(
            locations=locations,
            z=values,
            locationmode='ISO-3',
            colorscale=[
                [0, '#E6F7FF'], [0.1, '#B3E0FF'], [0.3, '#80C9FF'],
                [0.5, '#FFDDAA'], [0.7, '#FFBB77'], [1, '#FF9944']
            ],
            marker_line_color=config.COLORS['border'],
            marker_line_width=0.5,
            colorbar=dict(
                title="",
                thickness=15,
                len=0.6,
                x=0.95,
                y=0.7,
                tickfont=dict(color=config.COLORS['text_primary'], size=10),
                bgcolor='rgba(255,255,255,0.7)',
                tickformat=','
            ),
            customdata=country_names,
            hovertemplate=(
                '<b>%{customdata}</b><br>'
                f'{get_metric_name(metric)}: %{{z:,.0f}}'
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{get_metric_name(metric).upper()} POR PAÍS</b>",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color=config.COLORS['text_primary'])
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor=config.COLORS['bg_card'],
                landcolor='#f9fafb',
                oceancolor="#1d87cd", # Un azul más suave para el océano
                subunitcolor=config.COLORS['border']
            ),
            paper_bgcolor=config.COLORS['bg_card'],
            margin=dict(t=50, r=10, b=10, l=10),
            height=450, # Ajustado para que quepa bien
            font=dict(color=config.COLORS['text_primary'])
        )

        return dcc.Graph(
            id='world-map-graph',
            figure=fig,
            config={'displayModeBar': False},
            style={'height': 'calc(100% - 10px)'} # Ajuste para padding
        )

    except Exception as e:
        logger.exception(f"Error generando mapa: {e}")
        return html.Div([
            html.Div("❌ Error al Cargar Mapa", style={
                'fontSize': '16px',
                'fontWeight': '600',
                'color': config.COLORS['accent_red'],
                'marginBottom': '10px'
            }),
            html.Div(f"Detalle: {str(e)}", style={
                'fontSize': '11px',
                'color': config.COLORS['text_secondary']
            })
        ], style={'textAlign': 'center', 'padding': '100px 40px'})


@app.callback(
    Output('country-select', 'value'),
    Input('world-map-graph', 'clickData'),
    State('country-select', 'value'),
    prevent_initial_call=True
)
def update_country_from_map(clickData: Optional[Dict[str, Any]],
                            current_country: Optional[str]) -> Union[str, Any]:
    """Actualiza país seleccionado al hacer clic en el mapa."""
    if not clickData:
        return no_update

    try:
        country_name = clickData['points'][0]['customdata']
        logger.info(f"Map click: {country_name}")

        if country_name and country_name != current_country:
            return country_name
        return no_update

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error procesando clickData: {e}")
        return no_update


@app.callback(
    Output('date-range', 'end_date'),
    Input('date-range', 'start_date'),
    Input('trend-metric-select', 'value'),
    State('date-range', 'end_date'),
    prevent_initial_call=True
)
def auto_update_end_date(start_date_str: Optional[str],
                         selected_metric: Optional[str],
                         current_end_date_str: Optional[str]) -> Union[str, Any]:
    """Auto-ajusta fecha final si la fecha de inicio cambia."""
    # (Esta función puede necesitar ajustes si quieres que reaccione a la métrica también)
    if not start_date_str or not current_end_date_str:
         return no_update

    try:
         start_date_obj = date.fromisoformat(start_date_str)
         current_end_date_obj = date.fromisoformat(current_end_date_str)

         # Si la fecha final es anterior a la inicial, ajustarla
         if current_end_date_obj < start_date_obj:
             new_end_date_obj = start_date_obj + timedelta(days=1) # O cualquier lógica deseada
             max_allowed = date.today()
             if new_end_date_obj > max_allowed:
                 new_end_date_obj = max_allowed
             new_end_date_str = new_end_date_obj.isoformat()
             logger.info(f"Auto-updating end date to {new_end_date_str} because it was before start date")
             return new_end_date_str

         return no_update

    except (ValueError, TypeError) as e:
         logger.warning(f"Error auto-updating date: {e}")
         return no_update


@app.callback(
    Output('trend-chart', 'children'),
    [Input('country-select', 'value'),
     Input('trend-metric-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_trend(country: Optional[str], selected_metric: Optional[str],
                 start_date: Optional[str], end_date: Optional[str]) -> Union[dcc.Graph, html.Div]:
    """Actualiza gráfico de tendencia."""
    logger.info(f"Callback: update_trend (country={country}, metric={selected_metric})")

    if not country:
        country = config.DEFAULT_COUNTRY
    if not selected_metric:
        selected_metric = 'new_cases_smoothed'

    data = api.get_timeseries(country, selected_metric, start_date, end_date)

    if not data or not data.get('data'):
        return html.Div(f"📈 No hay datos de '{get_metric_name(selected_metric)}' para {country}.", style={
            'textAlign': 'center',
            'paddingTop': '120px',
            'color': config.COLORS['text_secondary'],
            'fontSize': '14px'
        })

    try:
        dates = [d['date'] for d in data['data']]
        values = [d.get('value', 0) or 0 for d in data['data']] # Manejo de None o 0
    except (KeyError, TypeError) as e:
        logger.error(f"Error procesando datos timeseries: {e}")
        return html.Div("❌ Error procesando datos.", style={
            'textAlign': 'center',
            'paddingTop': '120px',
            'color': config.COLORS['accent_red'],
            'fontSize': '14px'
        })

    fig = go.Figure()

    line_color = config.COLORS['accent_blue']
    if 'death' in selected_metric.lower():
        line_color = config.COLORS['accent_red']
    elif 'vaccin' in selected_metric.lower():
        line_color = config.COLORS['accent_green']

    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name=get_metric_name(selected_metric),
        line=dict(color=line_color, width=2.5),
        fill='tozeroy',
        fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)",
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    metric_title = get_metric_name(selected_metric)
    subtitle = f"{country}" + (" - Promedio Móvil 7 Días" if "smoothed" in selected_metric else "")

    fig.update_layout(
        title=dict(
            text=f"<b>TENDENCIA: {metric_title.upper()}</b><br><span style='font-size:11px;color:{config.COLORS['text_secondary']}'>{subtitle}</span>",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, color=config.COLORS['text_primary'])
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=config.COLORS['grid'],
            showline=False,
            tickfont=dict(size=10, color=config.COLORS['text_secondary'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=config.COLORS['grid'],
            showline=False,
            tickfont=dict(size=10, color=config.COLORS['text_secondary']),
            tickformat=','
        ),
        plot_bgcolor=config.COLORS['bg_card'],
        paper_bgcolor=config.COLORS['bg_card'],
        margin=dict(t=65, r=20, b=40, l=60),
        height=310, # Ajustado para que quepa bien
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(color=config.COLORS['text_primary'])
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%'})


@app.callback(
    Output('comparison-chart', 'children'),
    [Input('country-select', 'value'),
     Input('compare-countries', 'value'),
     Input('trend-metric-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_comparison(main_country: Optional[str], compare_countries: Optional[List[str]],
                      selected_metric: Optional[str], start_date: Optional[str],
                      end_date: Optional[str]) -> Union[dcc.Graph, html.Div]:
    """Actualiza gráfico de comparación."""
    logger.info(f"Callback: update_comparison (main={main_country}, compare={compare_countries})")

    if not main_country:
        main_country = config.DEFAULT_COUNTRY
    if not compare_countries:
        compare_countries = []
    if not selected_metric:
        selected_metric = 'new_cases_smoothed'

    all_countries = list(dict.fromkeys([main_country] + compare_countries))[:config.MAX_COMPARE_COUNTRIES + 1]

    if len(all_countries) < 2:
        return html.Div("📊 Selecciona al menos un país adicional para comparar.", style={
            'textAlign': 'center',
            'paddingTop': '120px',
            'color': config.COLORS['text_secondary'],
            'fontSize': '14px'
        })

    data = api.compare_countries(all_countries, selected_metric, start_date, end_date)

    if not data or not data.get('data'):
        return html.Div(f"🔄 No hay datos de comparación para '{get_metric_name(selected_metric)}'.", style={
            'textAlign': 'center',
            'paddingTop': '120px',
            'color': config.COLORS['text_secondary'],
            'fontSize': '14px'
        })

    fig = go.Figure()
    colors = [
        config.COLORS['accent_blue'],
        config.COLORS['accent_green'],
        config.COLORS['accent_yellow'],
        config.COLORS['accent_red'],
        config.COLORS['accent_purple'],
        '#17a2b8', # Teal
        '#fd7e14'  # Orange
    ]

    try:
        dates = [d['date'] for d in data['data']]
        for i, country in enumerate(data.get('countries', [])):
            values = [d.get('values', {}).get(country, 0) or 0 for d in data['data']] # Manejo de None o 0
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name=country,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>{get_metric_name(selected_metric)}: %{{y:,.0f}}<extra></extra>"
            ))
    except (KeyError, TypeError, IndexError) as e:
        logger.error(f"Error procesando datos comparación: {e}")
        return html.Div("❌ Error procesando datos de comparación.", style={
            'textAlign': 'center',
            'paddingTop': '120px',
            'color': config.COLORS['accent_red'],
            'fontSize': '14px'
        })

    metric_title = get_metric_name(selected_metric)
    subtitle = metric_title + (" - Promedio Móvil 7 Días" if "smoothed" in selected_metric else "")

    fig.update_layout(
        title=dict(
            text=f"<b>COMPARACIÓN: {metric_title.upper()}</b><br><span style='font-size:11px;color:{config.COLORS['text_secondary']}'>{subtitle}</span>",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, color=config.COLORS['text_primary'])
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=config.COLORS['grid'],
            showline=False,
            tickfont=dict(size=10, color=config.COLORS['text_secondary'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=config.COLORS['grid'],
            showline=False,
            tickfont=dict(size=10, color=config.COLORS['text_secondary']),
            tickformat=','
        ),
        plot_bgcolor=config.COLORS['bg_card'],
        paper_bgcolor=config.COLORS['bg_card'],
        margin=dict(t=65, r=20, b=40, l=60),
        height=310, # Ajustado para que quepa bien
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.4, # Ajustado para dar espacio
            xanchor='center',
            x=0.5,
            font=dict(size=10, color=config.COLORS['text_secondary']),
            bgcolor='rgba(255,255,255,0.8)' # Fondo semi-transparente
        ),
        font=dict(color=config.COLORS['text_primary'])
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%'})


# ============================================================================
# UTILIDADES
# ============================================================================

def fmt(num: Optional[Union[float, int, str]]) -> str:
    """Formatea números para visualización (K, M, B)."""
    if num is None:
        return '0'
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        if abs(num) >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif abs(num) >= 10_000: # Mostrar sin decimales para miles altos
            return f"{num/1_000:.0f}K"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.1f}K"
        elif abs(num) < 10 and not num.is_integer(): # Mostrar decimales para números pequeños
             return f"{num:.2f}"
        return f"{int(num):,}" # Formato con comas para miles
    except (ValueError, TypeError):
        return 'N/A'


def get_metric_name(metric: str) -> str:
    """Retorna nombre legible de métrica."""
    names = {
        'total_cases': 'Casos Totales',
        'new_cases': 'Nuevos Casos',
        'new_cases_smoothed': 'Nuevos Casos (Prom. 7 días)',
        'total_deaths': 'Muertes Totales',
        'new_deaths': 'Nuevas Muertes',
        'new_deaths_smoothed': 'Nuevas Muertes (Prom. 7 días)',
        'people_fully_vaccinated': 'Personas Completamente Vacunadas',
        'vaccination_rate': 'Tasa de Vacunación Completa (%)',
        'mortality_rate': 'Tasa de Mortalidad (%)'
        # Añade más si es necesario
    }
    return names.get(metric, metric.replace('_', ' ').title())


def create_aggregate_table(data: Dict[str, Any]) -> html.Div:
    """Crea tabla HTML para métricas agregadas (continentes/regiones)."""
    if not data:
        return html.Div("No hay datos disponibles.", style={
            'fontSize': '11px',
            'color': config.COLORS['text_secondary']
        })

    header = [html.Thead(html.Tr([
        html.Th("Ubicación"),
        html.Th("Casos", style={'textAlign': 'right'}),
        html.Th("Muertes", style={'textAlign': 'right'}),
    ]))]

    rows = []
    for location, metrics in data.items():
        if not isinstance(metrics, dict): # Seguridad extra
            continue
        rows.append(html.Tr([
            html.Td(location),
            html.Td(fmt(metrics.get('total_cases')), style={'color': config.COLORS['accent_blue']}),
            html.Td(fmt(metrics.get('total_deaths')), style={'color': config.COLORS['accent_red']}),
        ]))

    body = [html.Tbody(rows)]

    return html.Div(
        html.Table(header + body, className="metric-table"),
        className="metric-table-container" # Clase para aplicar scroll si es necesario
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(f"🚀 Iniciando Panel COVID-19 v4.0")
    print(f"API URL: {config.API_BASE_URL}")
    print(f"API Key: {'Configurada ✅' if config.API_KEY else 'No configurada ⚠️'}")
    print("=" * 70)
    print(f"\n🌍 Dashboard disponible en: http://127.0.0.1:8050")
    print(f"💡 Asegúrate de que la API esté corriendo en: {config.API_BASE_URL}")
    print("\n⏳ Presiona CTRL+C para detener el servidor")
    print("=" * 70 + "\n")

    try:
        # debug=True es útil para desarrollo, considera quitarlo para producción real
        app.run(debug=True, host='127.0.0.1', port=8050)
    except KeyboardInterrupt:
        print("\n✅ Servidor del dashboard detenido por el usuario.")
    except Exception as e:
        logger.exception("❌ Error fatal al iniciar el dashboard")
        print(f"❌ Error al iniciar el dashboard: {e}")