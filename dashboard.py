"""
Dashboard COVID-19 Profesional - Estilo Similar a la Imagen v1.5 (Búsqueda ISO/Continente)
======================================================================================
Dashboard interactivo utilizando Dash y Plotly para visualizar datos de COVID-19
servidos por la API complementaria (covid_api.py).

Características:
- Mapa coroplético interactivo.
- Clic en el mapa selecciona el país.
- El rango de fechas se ajusta a 7 días si se elige una métrica "smoothed".
- ¡NUEVO! El selector de país ahora muestra [ISO] y (Continente) y permite la búsqueda por ellos.
- Estadísticas clave por país.
- Gráfico de tendencia configurable por métrica y fecha.
- Gráfico de comparación entre países.
- Recuadro de métricas globales por continente.
- Actualización automática periódica.
- Logging añadido a callbacks.
"""

import dash
from dash import dcc, html, Input, Output, State, no_update # <-- 1. IMPORTADO no_update
import plotly.graph_objects as go
import plotly.express as px
import requests
import logging
from functools import lru_cache
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
import time # <--- IMPORT ADDED HERE

# ============================================================================
# CONFIGURACIÓN Y LOGGING
# ============================================================================
# Basic logging setup for the dashboard
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger("CovidDashboard")

class Config:
    """Holds configuration settings for the Dashboard."""
    API_BASE_URL: str = "http://127.0.0.1:8000"
    DEFAULT_COUNTRY: str = "Ecuador" # Default country to show on load
    UPDATE_INTERVAL_MS: int = 300000 # 5 minutes
    MAX_COMPARE_COUNTRIES: int = 5 # Limit for comparison dropdown

    # Color Palette (Light Theme)
    COLORS: Dict[str, str] = {
        'bg_dark': '#f8f9fa',        # Main background
        'bg_card': '#ffffff',        # Card background
        'bg_card_hover': '#f0f0f0',   # Potential hover effect
        'text_primary': '#1a1a1a',    # Main text
        'text_secondary': '#666666',   # Lighter text
        'accent_blue': '#2563eb',    # Primary accent (cases, trends)
        'accent_green': '#10b981',    # Success/Positive (vaccinations, online status)
        'accent_red': '#ef4444',      # Danger/Negative (deaths)
        'accent_yellow': '#f59e0b',   # Warning/Highlight
        'accent_purple': '#8b5cf6',   # Alternative accent
        'border': '#e5e7eb',      # Card borders
        'grid': '#f0f0f0'         # Chart grid lines
    }

# Instantiate configuration
config = Config()

# ============================================================================
# CLIENTE API
# ============================================================================
# ... (CovidAPI class remains the same) ...
class CovidAPI:
    """
    A client class to interact with the COVID-19 FastAPI backend.
    Includes caching for frequently accessed endpoints.
    """
    def __init__(self, base_url: str):
        """
        Initializes the API client.

        Args:
            base_url: The base URL of the FastAPI backend.
        """
        self.base_url: str = base_url

    @lru_cache(maxsize=1) # Cache heavily as this rarely changes
    def get_countries(self) -> List[Dict[str, Any]]:
        """Fetches the list of available countries from the API."""
        endpoint = f"{self.base_url}/covid/countries"
        logger.debug("Fetching countries from %s", endpoint)
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching countries from %s: %s", endpoint, e)
            return [] # Return empty list on error

    @lru_cache(maxsize=2) # Cache for a while
    def get_continent_metrics(self) -> Dict[str, Any]:
        """Fetches the latest pre-calculated metrics for continents."""
        endpoint = f"{self.base_url}/covid/metrics/continents"
        logger.debug("Fetching continent metrics from %s", endpoint)
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching continent metrics from %s: %s", endpoint, e)
            return {}

    # region_metrics endpoint removed

    @lru_cache(maxsize=10) # Cache map data per metric
    def get_map_data(self, metric: str) -> Optional[Dict[str, Any]]:
        """Fetches data suitable for map rendering for a specific metric."""
        endpoint = f"{self.base_url}/covid/map-data"
        params = {'metric': metric}
        logger.debug("Fetching map data from %s with params %s", endpoint, params)
        try:
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching map data from %s: %s", endpoint, e)
            return None

    @lru_cache(maxsize=200) # Cache summaries for many countries
    def get_summary(self, country: str) -> Optional[Dict[str, Any]]:
        """Fetches the latest summary statistics for a specific country."""
        endpoint = f"{self.base_url}/covid/summary"
        params = {'country': country}
        logger.debug("Fetching summary for '%s' from %s", country, endpoint)
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 404:
                logger.warning("Country '%s' not found by API.", country)
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching summary for '%s' from %s: %s", country, endpoint, e)
            return None

    def get_timeseries(self, country: str, metric: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches time series data for a specific country and metric."""
        endpoint = f"{self.base_url}/covid/timeseries"
        params: Dict[str, Any] = {'country': country, 'metric': metric}
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        logger.debug("Fetching timeseries from %s with params %s", endpoint, params)
        try:
            response = requests.get(endpoint, params=params, timeout=15)
            if response.status_code == 404:
                logger.warning("Timeseries data not found for metric '%s' country '%s'. API response: %s", metric, country, response.text)
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching timeseries from %s: %s", endpoint, e)
            return None

    def compare_countries(self, countries: List[str], metric: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches data to compare a metric across multiple countries."""
        endpoint = f"{self.base_url}/covid/compare"
        # FastAPI expects list parameters without index, requests handles this with list values
        params: Dict[str, Any] = {'countries': countries, 'metric': metric}
        if start_date: params['start_date'] = start_date
        if end_date: params['end_date'] = end_date
        logger.debug("Fetching comparison from %s with params %s", endpoint, params)
        try:
            # Use GET with list params
            response = requests.get(endpoint, params=params, timeout=20)
            if response.status_code == 404:
                 logger.warning("Comparison data not found for metric '%s' countries '%s'. API response: %s", metric, countries, response.text)
                 return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("API Error fetching comparison from %s: %s", endpoint, e)
            return None

# Global API client instance
api = CovidAPI(config.API_BASE_URL)

# ============================================================================
# INICIALIZAR DASH
# ============================================================================
# ... (Dash app initialization and index_string remain the same) ...
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
            /* Basic Reset & Font */
            body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            /* Animation for live indicator */
            @keyframes pulse { 0%%, 100%% { opacity: 1; } 50%% { opacity: 0.5; } }
            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #f1f1f1; }
            ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
            /* Metric Table Styling */
            .metric-table { width: 100%; border-collapse: collapse; font-size: 11px; }
            .metric-table th, .metric-table td { padding: 4px 6px; text-align: left; border-bottom: 1px solid #f0f0f0; }
            .metric-table th { font-weight: 600; color: #666; }
            .metric-table td:nth-child(2), .metric-table td:nth-child(3) { text-align: right; font-weight: 500; }
            .metric-table tr:last-child td { border-bottom: none; }
            .metric-table-container { max-height: 200px; overflow-y: auto; } /* Scrollable container */
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
# ESTILOS REUTILIZABLES
# ============================================================================
# ... (Styles remain the same) ...
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
# LAYOUT PRINCIPAL DEL DASHBOARD
# ============================================================================
# ... (Layout remains the same) ...
app.layout = html.Div(id="main-container", children=[
    # Header Section
    html.Header(style={
        'backgroundColor': config.COLORS['bg_card'],
        'padding': '20px 40px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'borderBottom': f"2px solid {config.COLORS['border']}",
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)'
    }, children=[
        html.Div([ # Title and Subtitle
            html.H1("PANEL COVID-19", style={'margin': '0', 'fontSize': '28px', 'fontWeight': '700', 'color': config.COLORS['text_primary'], 'letterSpacing': '2px'}),
            html.P("Análisis Global de la Pandemia en Tiempo Real", style={'margin': '5px 0 0 0', 'fontSize': '14px', 'color': config.COLORS['text_secondary'], 'fontWeight': '300'})
        ]),
        html.Div([ # Status Indicator
            html.Div(id='live-indicator', style={'display': 'inline-block', 'width': '10px', 'height': '10px', 'borderRadius': '50%', 'backgroundColor': config.COLORS['accent_green'], 'marginRight': '8px', 'animation': 'pulse 2s infinite'}),
            html.Span("EN LINEA", style={'fontSize': '12px', 'color': config.COLORS['accent_green'], 'fontWeight': '600', 'letterSpacing': '1px'}),
            html.Span(id='update-time', style={'marginLeft': '15px', 'fontSize': '12px', 'color': config.COLORS['text_secondary']})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ]),

    # Main Content Area (Sidebar + Main View)
    html.Div(style={'display': 'flex', 'padding': '20px', 'gap': '0', 'minHeight': 'calc(100vh - 80px)'}, children=[

        # Left Sidebar (Controls)
        html.Aside(style={'width': '320px', 'flexShrink': '0', 'paddingRight': '20px'}, children=[
            # Country Selector Card
            html.Div(style=CARD_STYLE, children=[
                html.Label("SELECCIONAR PAÍS", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(id='country-select', placeholder="🌍 Busca por País, ISO o Continente...", value=config.DEFAULT_COUNTRY, style={'backgroundColor': config.COLORS['bg_card'], 'color': config.COLORS['text_primary']})
            ]),

            # Main Stats Card (Content updated by callback)
            html.Div(id='main-stats', children=[html.Div("Cargando estadísticas...", style={'color': config.COLORS['text_secondary'], 'textAlign': 'center', 'padding': '40px'})]),

            # Continent Metrics Card
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICAS GLOBALES - CONTINENTES", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Loading(id="loading-continent-metrics", type="circle", color=config.COLORS['accent_blue'], children=html.Div(id='continent-metrics-box'))
            ]),

            # Trend Metric Selector Card
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICA DE TENDENCIA / COMPARACIÓN", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='trend-metric-select',
                    options=[
                        {'label': '📈 Nuevos Casos (Promedio 7 días)', 'value': 'new_cases_smoothed'},
                        {'label': '💀 Nuevas Muertes (Promedio 7 días)', 'value': 'new_deaths_smoothed'},
                        {'label': '🦠 Casos Totales', 'value': 'total_cases'},
                        {'label': '☠️ Muertes Totales', 'value': 'total_deaths'},
                    ],
                    value='new_cases_smoothed', clearable=False,
                    style={'backgroundColor': config.COLORS['bg_card'], 'color': config.COLORS['text_primary']}
                )
            ]),

            # Map Metric Selector Card
            html.Div(style=CARD_STYLE, children=[
                html.Label("MÉTRICA DEL MAPA", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='map-metric',
                    options=[
                        {'label': '🦠 Casos Totales', 'value': 'total_cases'},
                        {'label': '💀 Muertes Totales', 'value': 'total_deaths'},
                        {'label': '💉 Vacunados (Completos)', 'value': 'people_fully_vaccinated'},
                        {'label': '📊 Tasa de Vacunación (%)', 'value': 'vaccination_rate'},
                    ],
                    value='total_cases', clearable=False,
                    style={'backgroundColor': config.COLORS['bg_card'], 'color': config.COLORS['text_primary']}
                )
            ]),

            # Date Range Selector Card
            html.Div(style=CARD_STYLE, children=[
                html.Label("RANGO DE FECHAS (Automático para prom. 7 días)", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date='2020-03-01',
                    end_date='2023-11-30',   # Default end set to Nov 30, 2023
                    min_date_allowed='2019-12-01',
                    max_date_allowed=date.today().isoformat(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'},
                    start_date_placeholder_text='Fecha inicio',
                    end_date_placeholder_text='Fecha fin'
                )
            ]),

            # Compare Countries Card
            html.Div(style=CARD_STYLE, children=[
                html.Label(f"COMPARAR PAÍSES (Máx {config.MAX_COMPARE_COUNTRIES})", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                dcc.Dropdown(
                    id='compare-countries', multi=True,
                    placeholder=f"🔄 Busca y selecciona hasta {config.MAX_COMPARE_COUNTRIES}...",
                    style={'backgroundColor': config.COLORS['bg_card'], 'color': config.COLORS['text_primary']}
                )
            ])
        ]), # End Sidebar

        # Main Content Area (Map and Charts)
        html.Main(style={'flex': '1'}, children=[
            # Top Row: Map
            html.Div(style={'marginBottom': '20px'}, children=[
                html.Section(style={**CARD_STYLE, 'height': '500px'}, children=[
                    html.Label("MAPA MUNDIAL INTERACTIVO (Haz clic en un país)", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px', 'display': 'block'}),
                    dcc.Loading(id="loading-map", type="circle", color=config.COLORS['accent_blue'], children=html.Div(id='map-container', style={'height': 'calc(100% - 30px)'}))
                ])
            ]),

            # Bottom Row: Trend and Comparison Charts
            html.Div(style={'display': 'flex'}, children=[
                # Trend Chart
                html.Section(style={**CARD_STYLE, 'height': '350px', 'marginRight': '20px', 'flex': '1'}, children=[
                    dcc.Loading(id="loading-trend", type="circle", color=config.COLORS['accent_blue'], children=html.Div(id='trend-chart', style={'height': '100%'}))
                ]),
                # Comparison Chart
                html.Section(style={**CARD_STYLE, 'height': '350px', 'flex': '1'}, children=[
                    dcc.Loading(id="loading-comparison", type="circle", color=config.COLORS['accent_blue'], children=html.Div(id='comparison-chart', style={'height': '100%'}))
                ])
            ])
        ]) # End Main Content
    ]), # End Content Area

    # Interval component for periodic updates
    dcc.Interval(id='interval', interval=config.UPDATE_INTERVAL_MS, n_intervals=0)

], style={ # Main container style
    'backgroundColor': config.COLORS['bg_dark'],
    'color': config.COLORS['text_primary'],
    'minHeight': '100vh',
    'margin': '0',
    'padding': '0'
})

# ============================================================================
# CALLBACKS
# ============================================================================

# <-- ¡CALLBACK MODIFICADO! -->
@app.callback(
    [Output('country-select', 'options'),
     Output('compare-countries', 'options'),
     Output('update-time', 'children')],
    [Input('interval', 'n_intervals')]
)
def load_countries(n: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    """
    Periodically fetches the list of countries from the API and updates dropdown options.
    ¡NUEVO! Las etiquetas ahora incluyen ISO code y Continente para búsqueda.
    Also updates the 'last updated' timestamp.

    Args:
        n: The number of times the interval timer has fired.

    Returns:
        A tuple containing options for the country selector, options for the
        comparison selector, and the current timestamp string.
    """
    logger.info("Callback triggered: load_countries (interval=%d)", n)
    start_time = time.time()
    
    # Asumimos que esta llamada ahora devuelve [{'name': 'Ecuador', 'iso_code': 'ECU', 'continent': 'South America'}, ...]
    countries: List[Dict[str, Any]] = api.get_countries()
    timestamp: str = datetime.now().strftime('%H:%M:%S')

    if countries:
        options: List[Dict[str, str]] = []
        
        for c in countries:
            name = c.get('name')
            if not name:
                continue # Saltar países sin nombre
            
            # --- INICIO DE LA MODIFICACIÓN ---
            # Obtenemos los nuevos campos, con 'N/A' como fallback
            iso = c.get('iso_code', 'N/A')
            continent = c.get('continent', 'N/A')
            
            # Creamos la nueva etiqueta. El Dropdown buscará en esta cadena.
            # Ejemplo: "Ecuador [ECU] - South America"
            new_label = f"{name} [{iso}] - {continent}"
            # --- FIN DE LA MODIFICACIÓN ---
            
            # El 'value' sigue siendo solo el nombre, para que los otros callbacks no se rompan
            options.append({'label': new_label, 'value': name})

        # Ordenar alfabéticamente por la nueva etiqueta
        options = sorted(options, key=lambda x: x['label'])
    
    else:
        logger.warning("No countries received from API, using default.")
        options = [{'label': config.DEFAULT_COUNTRY, 'value': config.DEFAULT_COUNTRY}]

    elapsed = time.time() - start_time
    logger.info("Callback finished: load_countries (%.2f s)", elapsed)
    
    # Devolvemos las mismas opciones para ambos dropdowns y el timestamp
    return options, options, f"Actualizado: {timestamp}"


@app.callback(
    Output('main-stats', 'children'),
    [Input('country-select', 'value')]
)
def update_stats(country: Optional[str]) -> html.Div:
    """
    Fetches and displays the main summary statistics for the selected country.

    Args:
        country: The name of the country selected in the dropdown.

    Returns:
        An html.Div containing the formatted statistics cards.
    """
    logger.info("Callback triggered: update_stats (country=%s)", country)
    start_time = time.time()
    if not country:
        logger.warning("No country selected, defaulting to %s", config.DEFAULT_COUNTRY)
        country = config.DEFAULT_COUNTRY

    summary: Optional[Dict[str, Any]] = api.get_summary(country)

    if not summary:
        logger.warning("No summary data found for country: %s", country)
        elapsed_fail = time.time() - start_time
        logger.info("Callback finished: update_stats (%.2f s) - No Data", elapsed_fail)
        return html.Div([
            html.Div("⚠️", style={'fontSize': '36px', 'marginBottom': '10px'}),
            html.Div(f"No hay datos de resumen para {country}", style={'color': config.COLORS['text_primary'], 'fontSize': '13px'})
        ], style={'textAlign': 'center', 'paddingTop': '50px', **CARD_STYLE})

    stats_layout = html.Div([
        # Cases Card
        html.Div(style={**STAT_CARD_STYLE, 'marginBottom': '15px'}, children=[
            html.Div("CASOS CONFIRMADOS", style={'fontSize': '11px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '10px'}),
            html.Div(fmt(summary.get('total_cases')), style={'fontSize': '36px', 'fontWeight': '700', 'color': config.COLORS['accent_blue'], 'marginBottom': '5px', 'lineHeight': '1'}),
            html.Div(f"Población: {fmt(summary.get('population'))}", style={'fontSize': '12px', 'color': config.COLORS['text_secondary']})
        ]),
        # Deaths & Vaccinated Row
        html.Div(style={'display': 'flex', 'marginBottom': '15px'}, children=[
            # Deaths Card
            html.Div(style={**STAT_CARD_STYLE, 'marginRight': '10px', 'flex': '1'}, children=[
                html.Div("MUERTES", style={'fontSize': '10px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '8px'}),
                html.Div(fmt(summary.get('total_deaths')), style={'fontSize': '24px', 'fontWeight': '7G00', 'color': config.COLORS['accent_red'], 'marginBottom': '3px'}),
                html.Div(f"Mortalidad: {summary.get('mortality_rate', 0):.2f}%", style={'fontSize': '11px', 'color': config.COLORS['text_secondary']})
            ]),
            # Vaccinated Card
            html.Div(style={**STAT_CARD_STYLE, 'flex': '1'}, children=[
                html.Div("VACUNADOS (COMPLETO)", style={'fontSize': '10px', 'fontWeight': '600', 'color': config.COLORS['text_secondary'], 'letterSpacing': '1px', 'marginBottom': '8px'}),
                html.Div(f"{summary.get('vaccination_rate', 0):.1f}%", style={'fontSize': '24px', 'fontWeight': '700', 'color': config.COLORS['accent_green'], 'marginBottom': '3px'}),
                html.Div(f"({fmt(summary.get('people_fully_vaccinated'))})", style={'fontSize': '11px', 'color': config.COLORS['text_secondary']})
            ]),
        ]),
        # Additional Info Box
        html.Div(style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#f9fafb', 'borderRadius': '8px', 'border': f"1px solid {config.COLORS['border']}"}, children=[
            html.Div(f"📍 Continente: {summary.get('continent', 'N/A')}", style={'fontSize': '12px', 'color': config.COLORS['text_secondary'], 'marginBottom': '5px'}),
            html.Div(f"🗓️ Última Actualización: {summary.get('latest_date', 'N/A')}", style={'fontSize': '12px', 'color': config.COLORS['text_secondary'], 'marginBottom': '5px'}),
            html.Div(f"🌐 ISO Code: {summary.get('iso_code', 'N/A')}", style={'fontSize': '12px', 'color': config.COLORS['text_secondary']})
        ])
    ])

    elapsed = time.time() - start_time
    logger.info("Callback finished: update_stats (%.2f s)", elapsed)
    return stats_layout


@app.callback(
    Output('continent-metrics-box', 'children'),
    Input('interval', 'n_intervals')
)
def update_continent_metrics(n: int) -> html.Div:
    """
    Fetches and displays the latest metrics for continents in a table.

    Args:
        n: The interval counter.

    Returns:
        An html.Div containing the formatted table or an error message.
    """
    logger.info("Callback triggered: update_continent_metrics (interval=%d)", n)
    start_time = time.time()
    data: Dict[str, Any] = api.get_continent_metrics()

    if not data:
        logger.warning("No continent metrics received from API.")
        elapsed_fail = time.time() - start_time
        logger.info("Callback finished: update_continent_metrics (%.2f s) - No Data", elapsed_fail)
        return html.Div("No hay datos de continentes disponibles.", style={'fontSize': '11px', 'color': config.COLORS['text_secondary'], 'padding': '20px 0'})

    try:
        sorted_items = sorted(
            data.items(),
            key=lambda item: item[1].get('total_cases', 0) if isinstance(item[1], dict) else 0,
            reverse=True
        )
        sorted_data = dict(sorted_items)
    except Exception as sort_err:
        logger.error("Error sorting continent data: %s. Displaying unsorted.", sort_err)
        sorted_data = data

    table = create_aggregate_table(sorted_data)
    elapsed = time.time() - start_time
    logger.info("Callback finished: update_continent_metrics (%.2f s)", elapsed)
    return table


# ... (update_map remains the same) ...
@app.callback(
    Output('map-container', 'children'),
    [Input('map-metric', 'value')]
)
def update_map(metric: Optional[str]) -> Union[dcc.Graph, html.Div]:
    """
    Updates the world map based on the selected metric.
    Now includes an ID for click interactions.

    Args:
        metric: The metric selected in the 'map-metric' dropdown.

    Returns:
        A dcc.Graph component with the map or an html.Div with an error message.
    """
    logger.info("Callback triggered: update_map (metric=%s)", metric)
    start_time = time.time()
    if not metric:
        logger.warning("No metric selected for map, defaulting to 'total_cases'.")
        metric = 'total_cases'

    try:
        map_data: Optional[Dict[str, Any]] = api.get_map_data(metric)

        if not map_data or not map_data.get('data'):
            logger.warning("No map data received from API for metric: %s", metric)
            elapsed_fail = time.time() - start_time
            logger.info("Callback finished: update_map (%.2f s) - No Data", elapsed_fail)
            return html.Div("⚠️ No hay datos disponibles para mostrar en el mapa para esta métrica.",
                            style={'textAlign': 'center', 'padding': '100px 40px', 'color': config.COLORS['text_secondary'], 'fontSize': '14px'})

        locations: List[str] = []
        values: List[float] = []
        country_names_data: List[str] = [] # Almacenará los nombres de los países

        for item in map_data['data']:
            iso = item.get('iso_code')
            val = item.get('value')
            country_name = item.get('country', 'N/A') # Obtener el nombre del país

            if iso and val is not None:
                locations.append(iso)
                values.append(val)
                country_names_data.append(country_name) # Añadir el nombre a customdata
            else:
                logger.debug("Skipping map item due to missing iso_code or value: %s", item)

        if not locations:
            logger.warning("No valid locations with iso_codes found for map metric: %s", metric)
            elapsed_nodata = time.time() - start_time
            logger.info("Callback finished: update_map (%.2f s) - No Valid Data", elapsed_nodata)
            return html.Div("⚠️ No hay países con códigos ISO válidos para mostrar en el mapa.",
                                  style={'textAlign': 'center', 'padding': '100px 40px', 'color': config.COLORS['text_secondary'], 'fontSize': '14px'})

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
                title="", thickness=15, len=0.6, x=0.95, y=0.7,
                tickfont=dict(color=config.COLORS['text_primary'], size=10),
                bgcolor='rgba(255,255,255,0.7)', tickformat=','
            ),
            customdata=country_names_data, # Usar los nombres de los países
            hovertemplate=( # Definir el template de hover aquí
                f'<b>%{{customdata}}</b><br>'
                f'{get_metric_name(metric)}: %{{z:,.0f}}'
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{get_metric_name(metric).upper()} POR PAÍS</b>",
                y=0.95, x=0.5, xanchor='center', yanchor='top',
                font=dict(size=14, color=config.COLORS['text_primary'])
            ),
            geo=dict(
                showframe=False, showcoastlines=True, projection_type='natural earth',
                bgcolor=config.COLORS['bg_card'],
                landcolor='#f9fafb',
                oceancolor='#e0f2fe',
                subunitcolor=config.COLORS['border']
            ),
            paper_bgcolor=config.COLORS['bg_card'],
            margin=dict(t=50, r=10, b=10, l=10),
            height=450,
            font=dict(color=config.COLORS['text_primary'])
        )

        elapsed = time.time() - start_time
        logger.info("Callback finished: update_map (%.2f s)", elapsed)
        
        # Añadir un 'id' al dcc.Graph
        return dcc.Graph(
            id='world-map-graph', 
            figure=fig, 
            config={'displayModeBar': False}, 
            style={'height': 'calc(100% - 10px)'}
        )

    except Exception as e:
        logger.exception("Error fatal generando el mapa para la métrica '%s'", metric)
        elapsed_err = time.time() - start_time
        logger.info("Callback finished: update_map (%.2f s) - Error", elapsed_err)
        return html.Div([
            html.Div("❌ Error al Cargar Mapa", style={'fontSize': '16px', 'fontWeight': '600', 'color': config.COLORS['accent_red'], 'marginBottom': '10px'}),
            html.Div(f"Detalle: {str(e)}", style={'fontSize': '11px', 'color': config.COLORS['text_secondary']})
        ], style={'textAlign': 'center', 'padding': '100px 40px'})


# ... (update_country_from_map remains the same) ...
@app.callback(
    Output('country-select', 'value'),
    Input('world-map-graph', 'clickData'),
    State('country-select', 'value'),
    prevent_initial_call=True
)
def update_country_from_map(clickData: Optional[Dict[str, Any]],
                            current_country: Optional[str]) -> Union[str, no_update]: # type: ignore
    """
    Updates the country-select dropdown when a country is clicked on the map.
    """
    if not clickData:
        logger.debug("Map click callback triggered with no clickData.")
        return no_update

    try:
        # Extrae el nombre del país desde el 'customdata' que definimos en update_map
        country_name: str = clickData['points'][0]['customdata']
        logger.info("Callback triggered: update_country_from_map (clicked=%s)", country_name)

        if country_name and country_name != current_country:
            # Si el país es válido y diferente al actual, actualiza el dropdown
            return country_name
        else:
            # Si se hace clic en el mismo país o el dato es inválido, no hagas nada
            logger.debug("Clicked country is same as current or invalid. No update.")
            return no_update

    except (KeyError, IndexError, TypeError) as e:
        logger.warning("Error processing map clickData: %s. Data: %s", e, clickData)
        return no_update


# <-- ¡NUEVO CALLBACK AÑADIDO! -->
@app.callback(
    Output('date-range', 'end_date'),
    Input('date-range', 'start_date'),
    Input('trend-metric-select', 'value'),
    State('date-range', 'end_date'),
    prevent_initial_call=True
)
def auto_update_end_date(start_date_str: Optional[str], 
                         selected_metric: Optional[str], 
                         current_end_date_str: Optional[str]) -> Union[str, no_update]: # type: ignore
    """
    Automatically updates the end date to be 7 days (start + 6) after the
    start date IF a 'smoothed' (7-day average) metric is selected.
    """
    logger.debug("Callback triggered: auto_update_end_date (start=%s, metric=%s)", start_date_str, selected_metric)
    
    # Solo aplicar esta lógica si se selecciona una métrica de 7 días
    if not selected_metric or 'smoothed' not in selected_metric:
        logger.debug("Metric is not smoothed, skipping auto-date.")
        return no_update

    if not start_date_str:
        logger.debug("No start date provided, skipping auto-date.")
        return no_update

    try:
        start_date_obj = date.fromisoformat(start_date_str)
        
        # Un período de 7 días es start_date + 6 días
        new_end_date_obj = start_date_obj + timedelta(days=6)
        
        # No permitir que la fecha final excede la fecha máxima permitida
        max_allowed_obj = date.today()
        if new_end_date_obj > max_allowed_obj:
            new_end_date_obj = max_allowed_obj
            
        new_end_date_str = new_end_date_obj.isoformat()

        # Solo actualizar si la nueva fecha es diferente de la actual
        if new_end_date_str != current_end_date_str:
            logger.info("Auto-updating end date to %s for 7-day metric.", new_end_date_str)
            return new_end_date_str
        else:
            logger.debug("New end date is same as current, no update.")
            return no_update

    except (ValueError, TypeError) as e:
        logger.warning("Error parsing start date '%s' for auto-update: %s", start_date_str, e)
        return no_update


# ... (update_trend remains the same) ...
@app.callback(
    Output('trend-chart', 'children'),
    [Input('country-select', 'value'),
     Input('trend-metric-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_trend(country: Optional[str], selected_metric: Optional[str],
                 start_date: Optional[str], end_date: Optional[str]) -> Union[dcc.Graph, html.Div]:
    """
    Updates the trend chart based on selected country, metric, and date range.
    """
    logger.info("Callback triggered: update_trend (country=%s, metric=%s, start=%s, end=%s)",
                country, selected_metric, start_date, end_date)
    start_time = time.time()

    if not country: country = config.DEFAULT_COUNTRY
    if not selected_metric: selected_metric = 'new_cases_smoothed'

    data: Optional[Dict[str, Any]] = api.get_timeseries(country, selected_metric, start_date, end_date)

    if not data or not data.get('data'):
        logger.warning("No timeseries data received for country '%s', metric '%s'", country, selected_metric)
        elapsed_fail = time.time() - start_time
        logger.info("Callback finished: update_trend (%.2f s) - No Data", elapsed_fail)
        return html.Div(f"📈 No hay datos de '{get_metric_name(selected_metric)}' para {country}.",
                        style={'textAlign': 'center', 'paddingTop': '120px', 'color': config.COLORS['text_secondary'], 'fontSize': '14px'})

    try:
        dates: List[str] = [d['date'] for d in data['data']]
        values: List[float] = [d.get('value', 0) or 0 for d in data['data']]
    except (KeyError, TypeError) as e:
         logger.error("Error procesando datos de timeseries: %s. Data: %s", e, data)
         elapsed_err = time.time() - start_time
         logger.info("Callback finished: update_trend (%.2f s) - Data Error", elapsed_err)
         return html.Div("❌ Error procesando los datos de la serie temporal.",
                           style={'textAlign': 'center', 'paddingTop': '120px', 'color': config.COLORS['accent_red'], 'fontSize': '14px'})

    fig = go.Figure()
    line_color: str = config.COLORS['accent_blue']
    if 'death' in selected_metric.lower(): line_color = config.COLORS['accent_red']
    elif 'vaccin' in selected_metric.lower() or 'people' in selected_metric.lower(): line_color = config.COLORS['accent_green']
    elif 'case' not in selected_metric.lower(): line_color = config.COLORS['accent_purple']

    fig.add_trace(go.Scatter(
        x=dates, y=values, mode='lines', name=get_metric_name(selected_metric),
        line=dict(color=line_color, width=2.5), fill='tozeroy',
        fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)",
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    metric_title_name: str = get_metric_name(selected_metric)
    subtitle: str = f"{country}" + (" - Promedio Móvil 7 Días" if "smoothed" in selected_metric else "")

    fig.update_layout(
        title=dict(
            text=f"<b>TENDENCIA: {metric_title_name.upper()}</b><br><span style='font-size:11px;color:{config.COLORS['text_secondary']}'>{subtitle}</span>",
            x=0.5, y=0.95, xanchor='center', yanchor='top',
            font=dict(size=14, color=config.COLORS['text_primary'])
        ),
        xaxis=dict(showgrid=True, gridcolor=config.COLORS['grid'], showline=False, tickfont=dict(size=10, color=config.COLORS['text_secondary'])),
        yaxis=dict(showgrid=True, gridcolor=config.COLORS['grid'], showline=False, tickfont=dict(size=10, color=config.COLORS['text_secondary']), tickformat=','),
        plot_bgcolor=config.COLORS['bg_card'], paper_bgcolor=config.COLORS['bg_card'],
        margin=dict(t=65, r=20, b=40, l=60), height=310, hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(color=config.COLORS['text_primary'])
    )

    elapsed = time.time() - start_time
    logger.info("Callback finished: update_trend (%.2f s)", elapsed)
    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%'})


# ... (update_comparison remains the same) ...
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
    """
    Updates the comparison chart based on selected countries, metric, and date range.
    """
    logger.info("Callback triggered: update_comparison (main=%s, compare=%s, metric=%s, start=%s, end=%s)",
                main_country, compare_countries, selected_metric, start_date, end_date)
    start_time = time.time()

    if not main_country: main_country = config.DEFAULT_COUNTRY
    if not compare_countries: compare_countries = []
    if not selected_metric: selected_metric = 'new_cases_smoothed'

    all_countries: List[str] = list(dict.fromkeys([main_country] + compare_countries))[:config.MAX_COMPARE_COUNTRIES+1]

    if len(all_countries) < 2:
        logger.debug("Less than 2 countries selected for comparison.")
        elapsed_skip = time.time() - start_time
        logger.info("Callback finished: update_comparison (%.2f s) - Skipped", elapsed_skip)
        return html.Div("📊 Selecciona al menos un país adicional para comparar.",
                        style={'textAlign': 'center', 'paddingTop': '120px', 'color': config.COLORS['text_secondary'], 'fontSize': '14px'})

    data: Optional[Dict[str, Any]] = api.compare_countries(all_countries, selected_metric, start_date, end_date)

    if not data or not data.get('data'):
        logger.warning("No comparison data received for countries %s, metric '%s'", all_countries, selected_metric)
        elapsed_fail = time.time() - start_time
        logger.info("Callback finished: update_comparison (%.2f s) - No Data", elapsed_fail)
        return html.Div(f"🔄 No hay datos de comparación disponibles para '{get_metric_name(selected_metric)}'.",
                        style={'textAlign': 'center', 'paddingTop': '120px', 'color': config.COLORS['text_secondary'], 'fontSize': '14px'})

    fig = go.Figure()
    colors: List[str] = [config.COLORS['accent_blue'], config.COLORS['accent_green'], config.COLORS['accent_yellow'],
                         config.COLORS['accent_red'], config.COLORS['accent_purple'], '#17a2b8', '#fd7e14']

    try:
        dates: List[str] = [d['date'] for d in data['data']]
        for i, country in enumerate(data.get('countries', [])):
            values: List[float] = [d.get('values', {}).get(country, 0) or 0 for d in data['data']]
            fig.add_trace(go.Scatter(
                x=dates, y=values, mode='lines', name=country,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{country}</b><br>%{{x}}<br>{get_metric_name(selected_metric)}: %{{y:,.0f}}<extra></extra>"
            ))
    except (KeyError, TypeError, IndexError) as e:
         logger.error("Error procesando datos de comparación: %s. Data: %s", e, data)
         elapsed_err = time.time() - start_time
         logger.info("Callback finished: update_comparison (%.2f s) - Data Error", elapsed_err)
         return html.Div("❌ Error procesando los datos de comparación.",
                           style={'textAlign': 'center', 'paddingTop': '120px', 'color': config.COLORS['accent_red'], 'fontSize': '14px'})

    metric_title_name: str = get_metric_name(selected_metric)
    subtitle: str = metric_title_name + (" - Promedio Móvil 7 Días" if "smoothed" in selected_metric else "")

    fig.update_layout(
        title=dict(
            text=f"<b>COMPARACIÓN: {metric_title_name.upper()}</b><br><span style='font-size:11px;color:{config.COLORS['text_secondary']}'>{subtitle}</span>",
            x=0.5, y=0.95, xanchor='center', yanchor='top',
            font=dict(size=14, color=config.COLORS['text_primary'])
        ),
        xaxis=dict(showgrid=True, gridcolor=config.COLORS['grid'], showline=False, tickfont=dict(size=10, color=config.COLORS['text_secondary'])),
        yaxis=dict(showgrid=True, gridcolor=config.COLORS['grid'], showline=False, tickfont=dict(size=10, color=config.COLORS['text_secondary']), tickformat=','),
        plot_bgcolor=config.COLORS['bg_card'], paper_bgcolor=config.COLORS['bg_card'],
        margin=dict(t=65, r=20, b=40, l=60), height=310, hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5,
            font=dict(size=10, color=config.COLORS['text_secondary']),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        font=dict(color=config.COLORS['text_primary'])
    )

    elapsed = time.time() - start_time
    logger.info("Callback finished: update_comparison (%.2f s)", elapsed)
    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%'})

# ============================================================================
# UTILIDADES HELPER
# ============================================================================
# ... (fmt and get_metric_name remain the same) ...
def fmt(num: Optional[Union[float, int, str]]) -> str:
    """
    Formats a number into a human-readable string (e.g., 1.2M, 50K, 1,234).

    Args:
        num: The number to format.

    Returns:
        A formatted string representation of the number, or '0' if input is None/invalid.
    """
    if num is None: return '0'
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000: return f"{num/1_000_000_000:.1f}B"
        if abs(num) >= 1_000_000: return f"{num/1_000_000:.1f}M"
        elif abs(num) >= 10_000: return f"{num/1_000:.0f}K"
        elif abs(num) >= 1_000: return f"{num/1_000:.1f}K"
        elif abs(num) < 10 and not num.is_integer(): return f"{num:.2f}"
        return f"{int(num):,}"
    except (ValueError, TypeError):
        logger.warning("Invalid number format encountered: %s", num)
        return 'N/A'

def get_metric_name(metric: str) -> str:
    """Provides a user-friendly display name for a given metric key."""
    names: Dict[str, str] = {
        'total_cases': 'Casos Totales', 'new_cases': 'Nuevos Casos',
        'new_cases_smoothed': 'Nuevos Casos (Prom. 7 días)',
        'total_deaths': 'Muertes Totales', 'new_deaths': 'Nuevas Muertes',
        'new_deaths_smoothed': 'Nuevas Muertes (Prom. 7 días)',
        'total_tests': 'Pruebas Totales', 'new_tests': 'Nuevas Pruebas',
        'new_tests_smoothed': 'Nuevas Pruebas (Prom. 7 días)',
        'positive_rate': 'Tasa de Positividad (%)',
        'people_vaccinated': 'Personas Vacunadas (Al menos 1 dosis)',
        'people_fully_vaccinated': 'Personas Completamente Vacunadas',
        'total_boosters': 'Dosis de Refuerzo Totales',
        'vaccination_rate': 'Tasa de Vacunación Completa (%)',
        'hosp_patients': 'Pacientes Hospitalizados', 'icu_patients': 'Pacientes en UCI',
        'reproduction_rate': 'Tasa de Reproducción (Rt)',
        'stringency_index': 'Índice de Rigurosidad', 'population': 'Población',
        'population_density': 'Densidad Poblacional', 'median_age': 'Edad Mediana',
        'aged_65_older': '% Mayores de 65', 'gdp_per_capita': 'PIB per cápita',
        'cardiovasc_death_rate': 'Tasa Muerte Cardiovascular',
        'diabetes_prevalence': 'Prevalencia Diabetes (%)',

        'life_expectancy': 'Expectativa de Vida', 'mortality_rate': 'Tasa de Mortalidad (%)'
    }
    return names.get(metric, metric.replace('_', ' ').title())


def create_aggregate_table(data: Dict[str, Any]) -> html.Div:
    """
    Creates an HTML table component displaying aggregate metrics.

    Args:
        data: A dictionary where keys are locations (continents/regions)
              and values are dictionaries of metrics.

    Returns:
        An html.Div containing the styled table within a scrollable container.
    """
    if not data:
        return html.Div("No hay datos disponibles.", style={'fontSize': '11px', 'color': config.COLORS['text_secondary']})

    header = [html.Thead(html.Tr([
        html.Th("Ubicación"),
        html.Th("Casos", style={'textAlign': 'right'}),
        html.Th("Muertes", style={'textAlign': 'right'}),
    ]))]

    rows: List[html.Tr] = []
    for location, metrics in data.items():
        if not isinstance(metrics, dict):
             logger.warning("Formato inesperado para métricas agregadas de '%s'", location)
             continue
        rows.append(html.Tr([
            html.Td(location),
            # <-- ¡CORRECCIÓN DE BUG! config.Icons -> config.COLORS -->
            html.Td(fmt(metrics.get('total_cases')), style={'color': config.COLORS['accent_blue']}),
            html.Td(fmt(metrics.get('total_deaths')), style={'color': config.COLORS['accent_red']}),
        ]))

    body = [html.Tbody(rows)]

    return html.Div(
        html.Table(header + body, className="metric-table"),
        className="metric-table-container"
    )

# ============================================================================
# EJECUTAR APLICACIÓN DASH
# ============================================================================
# ... (Run section remains the same) ...
if __name__ == '__main__':
    print("\n" + "="*70)
    # <-- ¡CAMBIO DE VERSIÓN AQUÍ! -->
    print("🚀 Iniciando Panel COVID-19 Profesional - Tema Claro (v1.5 con Búsqueda ISO/Continente)")
    print("="*70)
    # Correctly get host/port for display message
    run_host = '127.0.0.1'
    run_port = 8050
    print(f"\n🌐 Dashboard disponible en: http://{run_host}:{run_port}")
    print(f"\n💡 Asegúrate de que la API esté corriendo en: {config.API_BASE_URL}")
    print("   Puedes iniciarla con: python covid_api.py")
    print("💡 Asegúrate de haber ejecutado el ETL al menos una vez:")
    print("   python etl_pipeline.py")
    print("\n⏳ Presiona CTRL+C para detener el servidor del dashboard.")
    print("="*70 + "\n")

    try:
        # Run the Dash server
        # debug=True enables hot-reloading and error messages in browser
        app.run(debug=True, host=run_host, port=run_port)
    except KeyboardInterrupt:
        print("\n✅ Servidor del dashboard detenido por el usuario.")
    except Exception as e:
        logger.exception("❌ Error fatal al iniciar o ejecutar el servidor del dashboard.")
        print(f"❌ Error al iniciar el dashboard: {e}")