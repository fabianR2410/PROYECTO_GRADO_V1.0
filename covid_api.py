"""
API REST para Datos COVID-19 - VERSIÓN ULTRA-OPTIMIZADA v3.0.0
============================================================
Sirve datos procesados del pipeline ETL COVID-19 con optimizaciones
de rendimiento como caché, pre-cálculo y ORJSON.
"""

from attr import dataclass
from fastapi import (
    FastAPI, HTTPException, Query, Depends, status, Request, BackgroundTasks,
    Path as PathParam # Alias Path to avoid conflict with pathlib.Path
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, date, timedelta
from functools import lru_cache, wraps
import pandas as pd
import numpy as np
import uvicorn
from enum import Enum
import logging
import sys
from pathlib import Path
import hashlib
from collections import defaultdict
import time
import json

# Attempt to import ORJSON for faster JSON responses
try:
    from fastapi.responses import ORJSONResponse
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
    USING_ORJSON = True
except ImportError:
    DEFAULT_RESPONSE_CLASS = JSONResponse
    USING_ORJSON = False
    print("⚠️  ORJSON no instalado. Usando JSON estándar (menos eficiente).")
    print("💡 Para mejor rendimiento: pip install orjson")

# ============================================================================
# CONFIGURACIÓN DE RUTAS Y LOGGING
# ============================================================================

BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_FILE: Path = LOGS_DIR / 'api.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(API_LOG_FILE, encoding='utf-8')
    ]
)
logger: logging.Logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

@dataclass
class Config:
    """Holds configuration settings for the COVID API."""
    # File Paths
    DATA_FILE: Path = DATA_DIR / "processed_covid.csv.gz"
    CONTINENT_METRICS_FILE: Path = DATA_DIR / "continent_metrics.json"
    REGION_METRICS_FILE: Path = DATA_DIR / "region_metrics.json"

    # API Metadata
    API_TITLE: str = "COVID-19 Data API - Ultra Optimized"
    API_VERSION: str = "3.0.0"

    # Limits
    MAX_COUNTRIES_COMPARE: int = 10
    DEFAULT_PAGE_SIZE: int = 100 # Placeholder, pagination not implemented yet
    MAX_PAGE_SIZE: int = 1000   # Placeholder

    # Cache Settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    ENABLE_CACHE: bool = True
    MAX_CACHE_SIZE: int = 1000     # Max items in LRU cache

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 200 # Requests per window
    RATE_LIMIT_WINDOW_SECONDS: int = 60 # Window duration in seconds

    # Performance Tuning
    PRECOMPUTE_SUMMARIES: bool = True # Precompute summaries at startup?
    USE_INDEXES: bool = True         # Use Pandas index for faster country lookup?
    CHUNK_SIZE: int = 10000          # Placeholder for potential future chunked processing

    # CORS and Middleware
    ALLOWED_ORIGINS: List[str] = ["*"] # Allow all origins for simplicity
    GZIP_MIN_SIZE: int = 500         # Min response size to apply GZip

# Instantiate configuration
config = Config()

# ============================================================================
# SISTEMA DE CACHÉ
# ============================================================================

class OptimizedCache:
    """
    A simple thread-unsafe in-memory cache with Time-To-Live (TTL) and Least
    Recently Used (LRU) eviction policy. Keys are generated based on function
    name and arguments.
    """
    def __init__(self, max_size: int = config.MAX_CACHE_SIZE, ttl: int = config.CACHE_TTL_SECONDS):
        """
        Initializes the cache.

        Args:
            max_size: Maximum number of items to store.
            ttl: Time-to-live for cache entries in seconds.
        """
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size: int = max_size
        self.ttl: int = ttl
        self.hits: int = 0
        self.misses: int = 0

    def _make_key(self, func_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        """Creates a unique hash key based on function name and arguments."""
        key_parts = [func_name]
        key_parts.extend(repr(a) for a in args) # Use repr for better distinction
        key_parts.extend(f"{k}={repr(v)}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache if it exists and hasn't expired.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # Check TTL
        if time.time() - self.access_times[key] > self.ttl:
            self._evict(key) # Remove expired item
            self.misses += 1
            return None

        # Cache hit: Update access time and return value
        self.hits += 1
        self.access_times[key] = time.time()
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Adds or updates an item in the cache. Evicts LRU item if max size is reached.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru() # Make space if needed before adding new item

        self.cache[key] = value
        self.access_times[key] = time.time() # Set/Update access time

    def _evict(self, key: str) -> None:
        """Removes a specific key from the cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]

    def _evict_lru(self) -> None:
        """Evicts the least recently used item from the cache."""
        if not self.access_times:
            return
        # Find the key with the minimum access time
        oldest_key = min(self.access_times, key=self.access_times.get)
        self._evict(oldest_key)

    def clear(self) -> None:
        """Clears the entire cache and resets stats."""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Returns statistics about cache usage."""
        total_accesses = self.hits + self.misses
        hit_rate = (self.hits / total_accesses * 100) if total_accesses > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "ttl_seconds": self.ttl
        }

# Global cache instance
cache = OptimizedCache()

# Cache decorator
def cached(ttl: Optional[int] = None):
    """
    Decorator to cache the result of a function.

    Args:
        ttl: Optional specific TTL for this function, defaults to global TTL.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.ENABLE_CACHE:
                return func(*args, **kwargs)

            cache_key = cache._make_key(func.__name__, args, kwargs)
            # Use the specific TTL if provided, else global TTL (already handled by cache.get)
            # No need to pass TTL here, OptimizedCache handles it internally on get/set
            result = cache.get(cache_key)
            if result is not None:
                return result # Return cached result

            # Cache miss: Execute function, cache result, then return
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator

# ============================================================================
# RATE LIMITING
# ============================================================================

class FastRateLimiter:
    """
    A simple in-memory rate limiter based on client ID (e.g., IP address)
    using a sliding window approach (simplified).
    """
    def __init__(self, requests: int = config.RATE_LIMIT_REQUESTS,
                 window: int = config.RATE_LIMIT_WINDOW_SECONDS):
        """
        Initializes the rate limiter.

        Args:
            requests: Maximum number of requests allowed within the window.
            window: Time window in seconds.
        """
        self.requests: int = requests
        self.window: int = window
        # Stores {'client_id': {'count': int, 'reset_time': float}}
        self.clients: Dict[str, Dict[str, Union[int, float]]] = defaultdict(
            lambda: {"count": 0, "reset_time": time.time() + self.window}
        )

    def is_allowed(self, client_id: str) -> bool:
        """
        Checks if a request from the client is allowed based on the rate limit.

        Args:
            client_id: A unique identifier for the client (e.g., IP address).

        Returns:
            True if the request is allowed, False otherwise.
        """
        now = time.time()
        client_data = self.clients[client_id]

        # Reset count if the window has passed
        if now >= client_data["reset_time"]:
            client_data["count"] = 0
            client_data["reset_time"] = now + self.window

        # Check if limit is exceeded
        if client_data["count"] >= self.requests:
            return False

        # Increment count and allow request
        client_data["count"] += 1
        return True

    def cleanup(self) -> None:
        """Removes inactive client records from memory."""
        now = time.time()
        # Find clients whose reset time is far in the past (e.g., more than one window ago)
        inactive_clients = [
            client_id for client_id, data in self.clients.items()
            if now > data["reset_time"] + self.window
        ]
        for client_id in inactive_clients:
            del self.clients[client_id]
        if inactive_clients:
            logger.debug("Rate limiter cleanup removed %d inactive clients.", len(inactive_clients))

# Global rate limiter instance
rate_limiter = FastRateLimiter()

# ============================================================================
# CARGADOR DE DATOS
# ============================================================================

class OptimizedDataLoader:
    """
    Handles loading, optimizing, and pre-computing COVID data at startup.
    Provides methods to access the data.
    """
    def __init__(self):
        """Initializes the loader and attempts to load data immediately."""
        self.df: Optional[pd.DataFrame] = None
        self.df_indexed: Optional[pd.DataFrame] = None # For faster lookup if USE_INDEXES is True
        self.metadata: Dict[str, Any] = {}
        self.precomputed: Dict[str, Any] = {} # Stores precomputed summaries, lists, etc.
        self.continent_metrics: Dict[str, Any] = {}
        self.region_metrics: Dict[str, Any] = {}
        self.load_data() # Load data on initialization

    def load_data(self) -> None:
        """Loads data from the processed gzipped CSV file specified in Config."""
        logger.info(f"📂 Cargando datos desde {config.DATA_FILE}")
        try:
            if not config.DATA_FILE.exists():
                logger.error(f"❌ Archivo de datos no encontrado: {config.DATA_FILE}")
                logger.error("💡 Por favor, ejecuta 'python etl_pipeline.py' primero.")
                return # Stop loading if file doesn't exist

            # Define dtypes for memory optimization during load
            dtype_dict = {
                'iso_code': 'category',
                'continent': 'category',
                'location': 'category'
                # Let pandas infer numeric types initially, will optimize later
            }

            self.df = pd.read_csv(
                config.DATA_FILE,
                compression="gzip",
                parse_dates=["date"], # Parse date column during read
                dtype=dtype_dict,
                low_memory=False # Recommended for mixed dtypes
            )

            # Further optimize numeric dtypes
            self.df = self._optimize_dtypes(self.df)
            # Sort for consistency and potential time-series operations
            self.df = self.df.sort_values(['location', 'date']).reset_index(drop=True)

            # Create indexed version if configured
            if config.USE_INDEXES:
                # Use verify_integrity=True to check for duplicate index entries (shouldn't happen if ETL is correct)
                try:
                    self.df_indexed = self.df.set_index('location', verify_integrity=False) # Skip verify for speed if confident
                    logger.info("⚡ Índices activados para búsqueda rápida por país.")
                except Exception as index_err:
                    logger.warning("⚠️ No se pudo crear el índice por país, usando búsqueda estándar: %s", index_err)
                    self.df_indexed = None # Fallback

            # Populate metadata
            self._populate_metadata()

            # Load aggregate metrics from JSON
            self._load_aggregate_metrics()

            # Precompute summaries if configured
            if config.PRECOMPUTE_SUMMARIES:
                self._precompute_data()

            logger.info(f"✅ Datos cargados: {self.metadata.get('total_records', 0):,} registros")
            logger.info(f"💾 Memoria usada: {self.metadata.get('memory_usage_mb', 0):.2f} MB")

        except FileNotFoundError:
             # This case should be caught earlier, but added for safety
             logger.error(f"❌ Error crítico: Archivo de datos no encontrado en la ruta esperada: {config.DATA_FILE}")
        except Exception as e:
            logger.error(f"❌ Error fatal al cargar o procesar datos iniciales: {str(e)}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback for debugging
            self.df = None # Ensure df is None on failure


    def _populate_metadata(self) -> None:
        """Calculates and stores metadata about the loaded DataFrame."""
        if self.df is None: return

        date_col = self.df['date']
        start_date = date_col.min()
        end_date = date_col.max()

        self.metadata = {
            'total_records': len(self.df),
            'countries': self.df['location'].nunique(),
            'columns': len(self.df.columns),
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d') if pd.notna(start_date) else None,
                'end': end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else None,
            },
            'metrics': [col for col in self.df.columns
                       if col not in ['location', 'date', 'iso_code', 'continent']],
            'loaded_at': datetime.now().isoformat(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

    def _load_aggregate_metrics(self) -> None:
        """Loads pre-calculated continent and region metrics from JSON files."""
        if config.CONTINENT_METRICS_FILE.exists():
            try:
                with open(config.CONTINENT_METRICS_FILE, 'r', encoding='utf-8') as f:
                    self.continent_metrics = json.load(f)
                logger.info(f"✅ Métricas de continentes cargadas: {len(self.continent_metrics)} registros")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️ No se pudieron cargar o parsear las métricas de continentes: {e}")
        else:
            logger.warning(f"⚠️ Archivo de métricas de continentes no encontrado: {config.CONTINENT_METRICS_FILE}")

        if config.REGION_METRICS_FILE.exists():
            try:
                with open(config.REGION_METRICS_FILE, 'r', encoding='utf-8') as f:
                    self.region_metrics = json.load(f)
                logger.info(f"✅ Métricas de regiones cargadas: {len(self.region_metrics)} registros")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️ No se pudieron cargar o parsear las métricas de regiones: {e}")
        else:
            logger.warning(f"⚠️ Archivo de métricas de regiones no encontrado: {config.REGION_METRICS_FILE}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcasts numeric columns to save memory."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
             # Use integer downcast only if safe (no NaNs or fits in smaller int)
             try:
                 df[col] = pd.to_numeric(df[col], downcast='integer')
             except (TypeError, ValueError): # Keep as original if downcast fails (e.g., due to NaNs)
                 pass
        return df

    def _precompute_data(self) -> None:
        """Pre-calculates common queries like country lists, summaries, and map data."""
        if self.df is None: return
        logger.info("🔄 Precalculando resúmenes y datos comunes...")
        start_time = time.time()

        # Precompute Countries List
        countries_list: List[Dict[str, Any]] = []
        # Group by location and get the last (latest) entry for each
        latest_entries = self.df.groupby('location').last().reset_index()
        # Get counts separately
        counts = self.df['location'].value_counts().to_dict()

        for _, row in latest_entries.iterrows():
            country = row['location']
            countries_list.append({
                'name': country,
                'iso_code': row.get('iso_code'),
                'continent': row.get('continent'),
                'population': self._clean_float(row.get('population')),
                'data_points': counts.get(country, 0),
                'latest_date': row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None
            })
        self.precomputed['countries_list'] = countries_list

        # Precompute Summaries
        summaries: Dict[str, Dict[str, Any]] = {}
        for _, row in latest_entries.iterrows():
            country = row['location']
            summaries[country] = {
                'country': country,
                'iso_code': row.get('iso_code'),
                'continent': row.get('continent'),
                'latest_date': row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None,
                'population': self._clean_float(row.get('population')),
                'total_cases': self._clean_float(row.get('total_cases')),
                'total_deaths': self._clean_float(row.get('total_deaths')),
                'people_fully_vaccinated': self._clean_float(row.get('people_fully_vaccinated')),
                'vaccination_rate': self._clean_float(row.get('vaccination_rate')),
                'mortality_rate': self._clean_float(row.get('mortality_rate'))
            }
        self.precomputed['summaries'] = summaries

        # Precompute Map Data
        map_data: Dict[str, List[Dict[str, Any]]] = {}
        map_metrics = ['total_cases', 'total_deaths', 'people_fully_vaccinated', 'vaccination_rate']
        for metric in map_metrics:
            if metric not in latest_entries.columns: continue
            metric_data = []
            for _, row in latest_entries.iterrows():
                value = self._clean_float(row.get(metric))
                # Only include countries with valid metric value and iso_code for the map
                if value is not None and value > 0 and pd.notna(row.get('iso_code')):
                    metric_data.append({
                        'country': row['location'],
                        'value': value,
                        'population': self._clean_float(row.get('population')),
                        'iso_code': row.get('iso_code')
                    })
            map_data[metric] = metric_data
        self.precomputed['map_data'] = map_data

        # Add loaded aggregate metrics to precomputed dictionary
        self.precomputed['continent_metrics'] = self.continent_metrics
        self.precomputed['region_metrics'] = self.region_metrics

        elapsed = time.time() - start_time
        logger.info(f"✅ Precálculos completados en {elapsed:.2f} segundos: {len(self.precomputed)} conjuntos")

    def _clean_float(self, value: Any) -> Optional[float]:
        """Safely converts a value to float, returning None for invalid inputs."""
        if value is None or pd.isna(value) or (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def is_ready(self) -> bool:
        """Checks if the data has been loaded successfully."""
        return isinstance(self.df, pd.DataFrame) and not self.df.empty

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the main processed DataFrame, raising an error if not ready."""
        if not self.is_ready():
            logger.error("Intento de acceso a datos antes de la carga.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Los datos COVID no están disponibles o no se cargaron correctamente. Inténtalo más tarde o ejecuta el ETL."
            )
        # Ensure self.df is not None before returning
        if self.df is None:
             # This should ideally not happen if is_ready() is True, but added for type safety
             raise HTTPException(status_code=500, detail="Error interno: DataFrame es None inesperadamente.")
        return self.df

    def get_country_data_fast(self, country: str) -> pd.DataFrame:
        """
        Retrieves data for a specific country, using index if available.

        Args:
            country: The name of the country.

        Returns:
            A DataFrame containing data for the specified country.

        Raises:
            HTTPException: If the country is not found or data is not ready.
        """
        if not self.is_ready():
             raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Datos no disponibles.")

        # Try indexed lookup first if available and configured
        if config.USE_INDEXES and isinstance(self.df_indexed, pd.DataFrame):
            try:
                # .loc can return a Series if only one row matches, or DataFrame
                data = self.df_indexed.loc[[country]] # Use list to force DataFrame return
                # If index lookup worked, reset index to match standard format
                return data.reset_index()
            except KeyError: # Country not found in index
                 raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"País '{country}' no encontrado.")
            except Exception as e:
                 logger.warning("Error inesperado durante búsqueda indexada para '%s', recurriendo a búsqueda normal: %s", country, e)
                 # Fall through to standard filtering if index lookup fails unexpectedly

        # Fallback to standard boolean filtering
        # Ensure self.df is not None
        if self.df is None:
             raise HTTPException(status_code=500, detail="Error interno: DataFrame es None inesperadamente.")

        country_df = self.df[self.df['location'] == country]
        if country_df.empty:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"País '{country}' no encontrado.")
        return country_df.copy() # Return a copy to avoid SettingWithCopyWarning downstream


# Global data loader instance
data_loader = OptimizedDataLoader()

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================
# Descriptions added for better OpenAPI docs

class TimeSeriesPoint(BaseModel):
    """Represents a single data point in a time series."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: Optional[float] = Field(None, description="Metric value for the date")

class TimeSeriesResponse(BaseModel):
    """Response model for time series data of a specific metric for a country."""
    country: str = Field(..., description="Name of the country")
    metric: str = Field(..., description="Name of the requested metric")
    total_points: int = Field(..., description="Number of data points returned")
    data: List[TimeSeriesPoint] = Field(..., description="List of time series data points")

class SummaryStats(BaseModel):
    """Response model for summary statistics of a country."""
    country: str = Field(..., description="Name of the country")
    iso_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-3 code")
    continent: Optional[str] = Field(None, description="Continent name")
    latest_date: str = Field(..., description="Most recent date for data (YYYY-MM-DD)")
    population: Optional[float] = Field(None, description="Estimated population")
    total_cases: Optional[float] = Field(None, description="Cumulative number of cases")
    total_deaths: Optional[float] = Field(None, description="Cumulative number of deaths")
    # Added people_fully_vaccinated to match precomputation
    people_fully_vaccinated: Optional[float] = Field(None, description="Cumulative number of people fully vaccinated")
    vaccination_rate: Optional[float] = Field(None, description="Percentage of population fully vaccinated")
    mortality_rate: Optional[float] = Field(None, description="Calculated mortality rate (total_deaths / total_cases)")

class MapDataPoint(BaseModel):
    """Represents data for a single country on the map."""
    country: str = Field(..., description="Name of the country")
    value: float = Field(..., description="Metric value for the country")
    population: Optional[float] = Field(None, description="Estimated population")
    iso_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-3 code for map rendering")

class MapDataResponse(BaseModel):
    """Response model for data suitable for rendering a choropleth map."""
    metric: str = Field(..., description="Name of the metric displayed")
    total_countries: int = Field(..., description="Number of countries with data")
    data: List[MapDataPoint] = Field(..., description="List of data points for the map")

# ============================================================================
# UTILIDADES Y DEPENDENCIAS DE ENDPOINTS
# ============================================================================

def clean_float(value: Any) -> Optional[float]:
    """
    Safely converts input to float, handling None, NaN, and Inf.

    Args:
        value: The value to convert.

    Returns:
        The float value, or None if conversion is not possible or invalid.
    """
    # Reuse the optimized data loader's cleaner function
    return data_loader._clean_float(value)

@cached(ttl=config.CACHE_TTL_SECONDS) # Apply cache decorator
def get_country_data_cached(country: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Cached retrieval of country data, optionally filtered by date.

    Args:
        country: The name of the country.
        start_date: Optional start date string (YYYY-MM-DD).
        end_date: Optional end date string (YYYY-MM-DD).

    Returns:
        A DataFrame filtered for the country and date range.

    Raises:
        HTTPException: If data is unavailable or country not found (delegated).
    """
    logger.debug("Cache miss or disabled for get_country_data_cached(country=%s, start=%s, end=%s)",
                 country, start_date, end_date)
    df = data_loader.get_country_data_fast(country) # Fetches or raises 404/503

    # Apply date filtering if requested
    if start_date:
        try:
            start_dt = pd.Timestamp(start_date)
            df = df[df['date'] >= start_dt]
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha de inicio inválido. Use YYYY-MM-DD.")
    if end_date:
        try:
            end_dt = pd.Timestamp(end_date)
            df = df[df['date'] <= end_dt]
        except ValueError:
             raise HTTPException(status_code=400, detail="Formato de fecha de fin inválido. Use YYYY-MM-DD.")

    # Return a copy to prevent modifying the cached DataFrame implicitly
    return df.copy()

async def get_data() -> pd.DataFrame:
    """Dependency to get the main DataFrame, ensuring data is loaded."""
    return data_loader.get_dataframe() # Raises 503 if not ready

async def rate_limit_check(request: Request) -> None:
    """Dependency to enforce rate limiting based on client IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit excedido para el cliente: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Límite de peticiones excedido. Inténtalo de nuevo en {config.RATE_LIMIT_WINDOW_SECONDS} segundos."
        )
    # No return value needed, just raises exception if limit exceeded

# ============================================================================
# INICIALIZAR FASTAPI
# ============================================================================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="API optimizada para servir datos procesados de COVID-19.",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=DEFAULT_RESPONSE_CLASS # Use ORJSON if available
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"], # Allow only GET requests
    allow_headers=["*"],
)

# Configure GZip compression for larger responses
app.add_middleware(GZipMiddleware, minimum_size=config.GZIP_MIN_SIZE)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"], summary="Root endpoint providing API status and info")
async def root() -> Dict[str, Any]:
    """Provides basic information about the API status and features."""
    return {
        "api_name": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "operational" if data_loader.is_ready() else "degraded (datos no cargados)",
        "using_orjson": USING_ORJSON,
        "cache_enabled": config.ENABLE_CACHE,
        "precomputation_enabled": config.PRECOMPUTE_SUMMARIES,
        "indexed_search": config.USE_INDEXES and data_loader.df_indexed is not None,
        "data_status": data_loader.metadata if data_loader.is_ready() else "Data not loaded",
        "documentation": "/docs"
    }

@app.get("/health", tags=["General"], summary="Health check endpoint")
async def health_check() -> Dict[str, Any]:
    """Returns the current health status of the API, including cache stats."""
    is_healthy = data_loader.is_ready()
    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": is_healthy,
        "cache_stats": cache.stats() if config.ENABLE_CACHE else "Cache disabled"
    }

@app.get("/covid/metrics/continents", tags=["Métricas Globales"],
         summary="Get latest metrics for continents",
         response_model=Dict[str, Dict[str, Any]]) # Define response model
async def get_continent_metrics() -> Dict[str, Any]:
    """
    Retrieves the latest pre-calculated summary metrics for each continent.
    Data is sourced from the `continent_metrics.json` file generated by the ETL.
    """
    # Use precomputed data which includes loaded metrics
    metrics = data_loader.precomputed.get('continent_metrics')
    if metrics:
        return metrics
    logger.warning("Solicitud de métricas de continentes, pero no están precargadas/encontradas.")
    raise HTTPException(status_code=404, detail="Métricas de continentes no encontradas o no cargadas.")

@app.get("/covid/metrics/regions", tags=["Métricas Globales"],
         summary="Get latest metrics for global regions/groups",
         response_model=Dict[str, Dict[str, Any]])
async def get_region_metrics() -> Dict[str, Any]:
    """
    Retrieves the latest pre-calculated summary metrics for global regions
    (e.g., World, High income). Data is sourced from the `region_metrics.json`
    file generated by the ETL.
    """
    metrics = data_loader.precomputed.get('region_metrics')
    if metrics:
        return metrics
    logger.warning("Solicitud de métricas de regiones, pero no están precargadas/encontradas.")
    raise HTTPException(status_code=404, detail="Métricas de regiones no encontradas o no cargadas.")

@app.get("/covid/countries", tags=["Información"],
         summary="Get a list of available countries",
         response_model=List[Dict[str, Any]])
async def get_countries() -> List[Dict[str, Any]]:
    """
    Returns a list of all unique countries available in the dataset,
    along with basic metadata like ISO code and continent. Uses precomputed list if available.
    """
    countries = data_loader.precomputed.get('countries_list')
    if countries:
        return countries

    # Fallback if precomputation failed or is disabled
    logger.warning("Recurriendo a cálculo de lista de países (precomputación falló o deshabilitada).")
    df = await get_data() # Ensure data is loaded
    countries_fallback: List[Dict[str, Any]] = []
    # Use groupby().last() for efficiency if many countries
    latest_data = df.groupby('location').last().reset_index()
    for _, row in latest_data.iterrows():
        countries_fallback.append({
            'name': row['location'],
            'iso_code': row.get('iso_code'),
            'continent': row.get('continent')
            # Add other relevant static fields if needed
        })
    return countries_fallback

@app.get("/covid/map-data", response_model=MapDataResponse, tags=["Datos"],
         summary="Get latest data suitable for a world map",
         dependencies=[Depends(rate_limit_check)])
async def get_map_data(
    metric: str = Query("total_cases", description="Metric to display on the map (e.g., total_cases, vaccination_rate)")
) -> MapDataResponse:
    """
    Provides the latest value for a given metric for all countries,
    optimized for rendering a choropleth map. Uses precomputed data if available.
    Requires ISO codes for map rendering.
    """
    precomputed_map_data = data_loader.precomputed.get('map_data', {}).get(metric)
    if precomputed_map_data:
        # Pydantic will validate the structure here
        return MapDataResponse(
            metric=metric,
            total_countries=len(precomputed_map_data),
            data=precomputed_map_data # Already in correct format
        )

    # Fallback if metric not precomputed or precomputation disabled
    logger.warning("Recurriendo a cálculo de map_data para '%s'.", metric)
    df = await get_data()
    if metric not in df.columns:
        raise HTTPException(status_code=400, detail=f"Métrica inválida para el mapa: '{metric}'")

    latest_data = df.groupby('location').last().reset_index()
    map_data_list: List[Dict[str, Any]] = []
    for _, row in latest_data.iterrows():
        value = clean_float(row.get(metric))
        iso_code = row.get('iso_code')
        # Include only if value is valid and iso_code exists
        if value is not None and value > 0 and pd.notna(iso_code):
            map_data_list.append({
                'country': row['location'],
                'value': value,
                'population': clean_float(row.get('population')),
                'iso_code': iso_code
            })

    if not map_data_list:
        # Return empty list instead of 404 if metric exists but yields no map data
         logger.warning("Map data para '%s' resultó vacía.", metric)

    # Pydantic validation happens on return
    return MapDataResponse(
        metric=metric,
        total_countries=len(map_data_list),
        data=map_data_list
    )


@app.get("/covid/summary", response_model=SummaryStats, tags=["Datos"],
         summary="Get latest summary statistics for a country",
         dependencies=[Depends(rate_limit_check)])
async def get_summary(
    country: str = Query(..., description="Name of the country (e.g., Ecuador, Spain)")
) -> SummaryStats:
    """
    Returns the latest summary statistics (total cases, deaths, vaccination rate, etc.)
    for a specific country. Uses precomputed summary if available.
    """
    precomputed_summary = data_loader.precomputed.get('summaries', {}).get(country)
    if precomputed_summary:
        # Validate structure with Pydantic model on return
        return SummaryStats(**precomputed_summary)

    # Fallback if not precomputed
    logger.warning("Recurriendo a cálculo de summary para '%s'.", country)
    # get_country_data_fast handles 404/503 errors
    country_df = data_loader.get_country_data_fast(country)
    if country_df.empty: # Should be caught by get_country_data_fast, but double-check
        raise HTTPException(status_code=404, detail=f"No data found for country '{country}' after fallback.")

    latest = country_df.iloc[-1] # Get the last row (latest data)

    # Manually construct the response matching the Pydantic model
    summary_data = {
        'country': country,
        'iso_code': latest.get('iso_code'),
        'continent': latest.get('continent'),
        'latest_date': latest['date'].strftime('%Y-%m-%d') if pd.notna(latest.get('date')) else None,
        'population': clean_float(latest.get('population')),
        'total_cases': clean_float(latest.get('total_cases')),
        'total_deaths': clean_float(latest.get('total_deaths')),
        'people_fully_vaccinated': clean_float(latest.get('people_fully_vaccinated')),
        'vaccination_rate': clean_float(latest.get('vaccination_rate')),
        'mortality_rate': clean_float(latest.get('mortality_rate'))
    }
     # Filter out None latest_date if date conversion failed somewhere
    if summary_data['latest_date'] is None:
        raise HTTPException(status_code=500, detail="Error interno: fecha inválida encontrada.")

    return SummaryStats(**summary_data)


@app.get("/covid/timeseries", response_model=TimeSeriesResponse, tags=["Datos"],
         summary="Get time series data for a metric and country",
         dependencies=[Depends(rate_limit_check)])
async def get_timeseries(
    country: str = Query(..., description="Name of the country"),
    metric: str = Query(..., description="Metric name (e.g., new_cases_smoothed, total_deaths)"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)", regex=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)", regex=r"^\d{4}-\d{2}-\d{2}$")
) -> TimeSeriesResponse:
    """
    Returns time series data for a specific metric and country, optionally
    filtered by a date range. Uses cached data retrieval.
    """
    df_full = await get_data() # Ensure main df is available
    if metric not in df_full.columns:
        raise HTTPException(status_code=400, detail=f"Métrica inválida: '{metric}'")

    # Use cached function which handles 404/503 and date filtering/validation
    country_df = get_country_data_cached(country, start_date, end_date)

    # Select only necessary columns and filter out rows where the metric is NaN
    ts_data = country_df[['date', metric]].dropna(subset=[metric])

    # Convert to list of dictionaries (or Pydantic models) efficiently
    data_points: List[TimeSeriesPoint] = [
        TimeSeriesPoint(date=row['date'].strftime('%Y-%m-%d'), value=clean_float(row[metric]))
        for _, row in ts_data.iterrows()
    ]

    return TimeSeriesResponse(
        country=country,
        metric=metric,
        total_points=len(data_points),
        data=data_points
    )


@app.get("/covid/compare", tags=["Datos"],
         summary="Compare time series of a metric across multiple countries",
         response_model=Dict[str, Any], # Define a more specific model if needed
         dependencies=[Depends(rate_limit_check)])
async def compare_countries(
    countries: List[str] = Query(..., min_length=2, max_length=config.MAX_COUNTRIES_COMPARE, description="List of 2 to 10 country names"),
    metric: str = Query(..., description="Metric name to compare"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)", regex=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)", regex=r"^\d{4}-\d{2}-\d{2}$")
) -> Dict[str, Any]:
    """
    Returns time series data for a specific metric, comparing values across
    a list of specified countries within an optional date range.
    """
    df = await get_data()
    if metric not in df.columns:
        raise HTTPException(status_code=400, detail=f"Métrica inválida: '{metric}'")

    # Check if all requested countries exist (optional, but good practice)
    available_countries = set(df['location'].unique())
    missing_countries = [c for c in countries if c not in available_countries]
    if missing_countries:
        raise HTTPException(status_code=404, detail=f"Países no encontrados: {', '.join(missing_countries)}")

    # Filter data for the selected countries
    compare_df = df[df['location'].isin(countries)].copy()

    # Apply date filters
    if start_date:
        try:
            compare_df = compare_df[compare_df['date'] >= pd.Timestamp(start_date)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Formato de fecha de inicio inválido.")
    if end_date:
        try:
             compare_df = compare_df[compare_df['date'] <= pd.Timestamp(end_date)]
        except ValueError:
             raise HTTPException(status_code=400, detail="Formato de fecha de fin inválido.")

    # Pivot the table for comparison format: date index, country columns
    # Use pivot_table to handle potential duplicate date/location pairs gracefully (takes first)
    pivot_data = compare_df.pivot_table(
        index='date',
        columns='location',
        values=metric,
        aggfunc='first' # Or 'mean'/'sum' if aggregation is needed, though 'first' is likely correct here
    ).reset_index() # Reset index to have 'date' as a column

    # Format the output data
    comparison_data: List[Dict[str, Any]] = []
    for _, row in pivot_data.iterrows():
        point: Dict[str, Any] = {
            'date': row['date'].strftime('%Y-%m-%d'),
            'values': {
                # Include value only if it exists for that country on that date
                country: clean_float(row.get(country))
                for country in countries if country in row and pd.notna(row.get(country))
            }
        }
        comparison_data.append(point)

    return {
        'countries': countries,
        'metric': metric,
        'total_points': len(comparison_data),
        'data': comparison_data
    }

@app.get("/covid/latest/{country}", tags=["Datos"],
         summary="Get all latest metric values for a specific country",
         response_model=Dict[str, Any], # Define a more specific model if needed
         dependencies=[Depends(rate_limit_check)])
async def get_latest_data(
    country: str = PathParam(..., description="Name of the country")
) -> Dict[str, Any]:
    """Retrieves the most recent record with all available metrics for a single country."""
    # get_country_data_fast handles 404/503
    country_df = data_loader.get_country_data_fast(country)
    if country_df.empty: # Should be caught, but double-check
        raise HTTPException(status_code=404, detail=f"No data found for country '{country}'.")

    latest: pd.Series = country_df.iloc[-1] # Get the last row

    result: Dict[str, Any] = {
        'country': country,
        'date': latest['date'].strftime('%Y-%m-%d') if pd.notna(latest.get('date')) else None,
        'data': {}
    }
    if result['date'] is None:
         raise HTTPException(status_code=500, detail="Error interno: fecha inválida encontrada en los últimos datos.")

    # Iterate through the Series index (column names)
    for col_name, value in latest.items():
        if col_name not in ['location', 'date', 'index']: # Exclude keys/index
             # Store cleaned float or original value if not float/int
             cleaned_value = clean_float(value)
             # Store only non-null values for cleaner output
             if cleaned_value is not None:
                 result['data'][col_name] = cleaned_value
             # Optionally handle categorical data explicitly if needed
             elif isinstance(value, str) and pd.notna(value):
                 result['data'][col_name] = value

    return result

@app.get("/covid/batch-summaries", tags=["Datos"],
         summary="Get basic summary stats for multiple countries in one request",
         response_model=Dict[str, Dict[str, Any]], # Dict keys are country names
         dependencies=[Depends(rate_limit_check)])
async def batch_summaries(
    countries: List[str] = Query(..., description="List of country names (max 50)", max_length=50)
) -> Dict[str, Dict[str, Any]]:
    """
    Efficiently retrieves basic summary statistics (cases, deaths, population)
    for a list of countries. Uses precomputed summaries if available.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # Try precomputed first
    if 'summaries' in data_loader.precomputed:
        precomputed_summaries = data_loader.precomputed['summaries']
        for country in countries:
            summary = precomputed_summaries.get(country)
            if summary:
                # Extract only basic stats if needed, or return full precomputed summary
                results[country] = {
                    'total_cases': summary.get('total_cases'),
                    'total_deaths': summary.get('total_deaths'),
                    'population': summary.get('population'),
                    'latest_date': summary.get('latest_date')
                }
            else:
                results[country] = {"error": "País no encontrado"}
        return results

    # Fallback if precomputation failed or is disabled
    logger.warning("Recurriendo a cálculo de batch_summaries.")
    df = await get_data()
    # Group by country and get the last entry for efficiency
    latest_data = df[df['location'].isin(countries)].groupby('location').last()

    for country in countries:
        if country in latest_data.index:
            row = latest_data.loc[country]
            results[country] = {
                'total_cases': clean_float(row.get('total_cases')),
                'total_deaths': clean_float(row.get('total_deaths')),
                'population': clean_float(row.get('population')),
                'latest_date': row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None
            }
        else:
            results[country] = {"error": "País no encontrado"}
    return results

# ============================================================================
# BACKGROUND TASKS & STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event() -> None:
    """Logs API startup information and data loading status."""
    logger.info("="*70)
    logger.info(f"🚀 Iniciando {config.API_TITLE} v{config.API_VERSION}")
    logger.info("="*70)

    if data_loader.is_ready() and data_loader.metadata:
        meta = data_loader.metadata
        logger.info(f"✅ Estado de datos: CARGADOS")
        logger.info(f"📊 Registros: {meta.get('total_records', 0):,}")
        logger.info(f"💾 Memoria: {meta.get('memory_usage_mb', 0):.2f} MB")
        logger.info(f"🌍 Países: {meta.get('countries', 0)}")
        logger.info(f"📅 Rango: {meta.get('date_range', {}).get('start', 'N/A')} a {meta.get('date_range', {}).get('end', 'N/A')}")
        logger.info(f"⚡ Opciones: ORJSON={'Sí' if USING_ORJSON else 'No'}, Cache={'Sí' if config.ENABLE_CACHE else 'No'}, Precálculo={'Sí' if config.PRECOMPUTE_SUMMARIES else 'No'}, Índices={'Sí' if config.USE_INDEXES and data_loader.df_indexed is not None else 'No'}")
    else:
        logger.error("❌ DATOS NO CARGADOS o error en carga. La API operará en modo DEGRADADO.")
        logger.error(f"💡 Verifica que el archivo '{config.DATA_FILE.name}' exista en '{config.DATA_DIR}'.")
        logger.error("💡 Ejecuta el script 'etl_pipeline.py' para generar/actualizar los datos.")

    logger.info("="*70)

# Optional: Add shutdown event if needed for cleanup
# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("🛑 Deteniendo la API...")

@app.post("/cache/clear", tags=["General"], status_code=status.HTTP_202_ACCEPTED,
          summary="Schedule background tasks to clear cache and cleanup rate limiter")
async def clear_cache_endpoint(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Triggers background tasks to clear the application cache and clean up
    stale rate limiter entries. Returns immediately.
    """
    if config.ENABLE_CACHE:
        background_tasks.add_task(cache.clear)
        logger.info("Tarea de limpieza de caché programada.")
    else:
         logger.info("Limpieza de caché solicitada, pero el caché está deshabilitado.")

    background_tasks.add_task(rate_limiter.cleanup)
    logger.info("Tarea de limpieza de rate limiter programada.")

    return {"message": "Tareas de limpieza programadas en segundo plano."}

# ============================================================================
# RUN WITH UVICORN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    # Log startup info also to console for direct run
    print("\n" + "="*70)
    print(f"🚀 Ejecutando {config.API_TITLE} v{config.API_VERSION} directamente con Uvicorn...")
    print("="*70)

    if data_loader.is_ready():
        meta = data_loader.metadata
        print(f"\n✅ Estado: OPERACIONAL")
        print(f"📊 Registros: {meta.get('total_records', 0):,}")
        print(f"💾 Memoria: {meta.get('memory_usage_mb', 0):.2f} MB")
        print(f"🌍 Países: {meta.get('countries', 0)}")
        print(f"\n📖 Documentación disponible en: http://127.0.0.1:8000/docs")
        print("="*70 + "\n")

        uvicorn.run(
            "covid_api:app", # Use "filename:app_instance" format
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False, # Enable reload for development

            # === IMPORTANT: Reload configuration to avoid loops ===
            reload_dirs=[str(BASE_DIR)], # Watch only the base directory
            reload_excludes=[
                str(LOGS_DIR),        # Ignore changes in the logs directory
                str(DATA_DIR),        # Ignore changes in the data directory
                ".*",                 # Ignore hidden files/dirs (like .venv, .git)
                "*.pyc",              # Ignore Python cache files
                "__pycache__"         # Ignore Python cache directory
            ]
        )
    else:
        # Critical error message if data didn't load
        print("\n" + "="*70)
        print("❌ ERROR CRÍTICO: Los datos COVID no se cargaron correctamente.")
        print(f"   Asegúrate de que el archivo '{config.DATA_FILE.name}' exista en '{config.DATA_DIR}'.")
        print("   Ejecuta 'python etl_pipeline.py' para generar los datos.")
        print("   La API no puede iniciarse sin datos.")
        print("="*70 + "\n")
        sys.exit(1) # Exit with error code