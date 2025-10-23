"""
COVID-19 REST API - SIMPLIFIED VERSION (Countries Only)
======================================================
API REST enfocada en métricas por país.
"""

import os
import sys
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path  # Ensure Path is imported
from typing import List, Optional, Dict, Any, Union
from functools import wraps

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, Request, BackgroundTasks, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

# --- START CORRECTION: Define env_config first ---
try:
    from decouple import config as env_config
except ImportError:
    class DummyConfig:
        def __call__(self, key, default=None, cast=None):
            value = os.getenv(key, default)
            if cast and value is not None:
                return cast(value)
            return value
    env_config = DummyConfig()
# --- END CORRECTION ---

# Redis (opcional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis no disponible. Usando caché en memoria.")

# ORJSON (opcional)
try:
    from fastapi.responses import ORJSONResponse
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
except ImportError:
    DEFAULT_RESPONSE_CLASS = JSONResponse

# ============================================================================
# CONFIGURACIÓN DE PATHS (Moved earlier)
# ============================================================================

# --- START CORRECTION: Define Paths after env_config ---
BASE_DIR = Path(__file__).resolve().parent # Use resolve() for better path handling
DATA_DIR = BASE_DIR / env_config("DATA_DIR", default="data")
LOGS_DIR = BASE_DIR / env_config("LOGS_DIR", default="logs")
# --- END CORRECTION ---

# ============================================================================
# CONFIGURACIÓN DE LOGGING (Now uses defined LOGS_DIR)
# ============================================================================
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=env_config("LOG_LEVEL", default="INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / 'api_paises.log', encoding='utf-8') # Now LOGS_DIR is defined
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA API (Now uses defined DATA_DIR and env_config)
# ============================================================================

class APIConfig:
    DATA_FILE = DATA_DIR / "processed_covid.csv.gz" # Now DATA_DIR is defined
    API_TITLE = env_config("API_TITLE", "COVID-19 Data API (Countries)")
    API_VERSION = env_config("API_VERSION", "5.2.0") # Versión actualizada
    ENVIRONMENT = env_config("ENVIRONMENT", "development")

    _api_keys_str = env_config("API_KEYS", default="")
    API_KEYS = [key.strip() for key in _api_keys_str.split(',') if key.strip()]

    _admin_api_keys_str = env_config("ADMIN_API_KEYS", default="")
    ADMIN_API_KEYS = [key.strip() for key in _admin_api_keys_str.split(',') if key.strip()]

    ALLOWED_HOSTS = env_config("ALLOWED_HOSTS", default="*").split(",")
    CORS_ORIGINS = env_config("CORS_ORIGINS", default="*").split(",")
    RATE_LIMIT_PER_MINUTE = int(env_config("RATE_LIMIT_PER_MINUTE", default="60"))
    REDIS_URL = env_config("REDIS_URL", default="redis://localhost:6379/0")
    REDIS_ENABLED = env_config("REDIS_ENABLED", default="false").lower() == "true"
    CACHE_TTL = int(env_config("CACHE_TTL_SECONDS", default="3600"))
    ENABLE_CACHE = env_config("ENABLE_CACHE", default="true").lower() == "true"
    PRECOMPUTE_SUMMARIES = env_config("PRECOMPUTE_SUMMARIES", default="true").lower() == "true"
    USE_INDEXES = env_config("USE_INDEXES", default="true").lower() == "true"
    GZIP_MIN_SIZE = int(env_config("GZIP_MIN_SIZE", default="500"))

config = APIConfig()

# ============================================================================
# SEGURIDAD, CACHÉ
# ============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "No autorizado"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    if config.ENVIRONMENT == "development" and not config.API_KEYS: return "dev-key"
    if not api_key: raise UnauthorizedException("API key requerida en header X-API-Key")
    if api_key not in config.API_KEYS:
        logger.warning(f"Intento de acceso con API key inválida")
        raise UnauthorizedException("API key inválida")
    return api_key

def verify_admin_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    if config.ENVIRONMENT == "development" and not config.ADMIN_API_KEYS: return "dev-admin-key"
    if not api_key: raise UnauthorizedException("API key de administrador requerida")
    if api_key not in config.ADMIN_API_KEYS:
        logger.warning("Intento de acceso admin con key inválida")
        raise UnauthorizedException("No tiene permisos de administrador")
    return api_key

class CacheBackend:
    def get(self, key: str) -> Optional[Any]: raise NotImplementedError
    def set(self, key: str, value: Any, ttl: int = 3600) -> None: raise NotImplementedError
    def delete(self, key: str) -> None: raise NotImplementedError
    def clear(self) -> None: raise NotImplementedError
    def stats(self) -> Dict[str, Any]: raise NotImplementedError

class RedisCache(CacheBackend):
    def __init__(self, redis_url: str):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.hits = 0; self.misses = 0
            logger.info(f"✅ Redis conectado: {redis_url}")
        except Exception as e:
            raise ConnectionError(f"No se pudo conectar a Redis: {e}")
    def get(self, key: str) -> Optional[Any]:
        try: import pickle; data = self.redis_client.get(key);
        except Exception as e: logger.error(f"Error leyendo de Redis: {e}"); self.misses += 1; return None
        if data: self.hits += 1; return pickle.loads(data)
        self.misses += 1; return None
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        try: import pickle; self.redis_client.setex(key, ttl, pickle.dumps(value))
        except Exception as e: logger.error(f"Error escribiendo en Redis: {e}")
    def delete(self, key: str) -> None:
        try: self.redis_client.delete(key)
        except Exception as e: logger.error(f"Error eliminando de Redis: {e}")
    def clear(self) -> None:
        try: self.redis_client.flushdb(); logger.info("✅ Caché Redis limpiado")
        except Exception as e: logger.error(f"Error limpiando Redis: {e}")
    def stats(self) -> Dict[str, Any]:
        try:
            info = self.redis_client.info()
            return {'backend': 'redis', 'hits': self.hits, 'misses': self.misses,
                    'hit_rate': round(self.hits / max(self.hits + self.misses, 1), 3),
                    'used_memory_mb': round(info.get('used_memory', 0) / 1024**2, 2),
                    'connected_clients': info.get('connected_clients', 0)}
        except: return {'backend': 'redis', 'error': 'No se pudo obtener stats'}

class MemoryCache(CacheBackend):
    def __init__(self): self.cache = {}; self.hits = 0; self.misses = 0; logger.info("✅ Caché en memoria inicializado")
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires_at'] > time.time(): self.hits += 1; return entry['value']
            else: del self.cache[key]
        self.misses += 1; return None
    def set(self, key: str, value: Any, ttl: int = 3600) -> None: self.cache[key] = {'value': value, 'expires_at': time.time() + ttl}
    def delete(self, key: str) -> None: self.cache.pop(key, None)
    def clear(self) -> None: self.cache.clear(); logger.info("✅ Caché en memoria limpiado")
    def stats(self) -> Dict[str, Any]:
        now = time.time(); expired_keys = [k for k, v in self.cache.items() if v['expires_at'] <= now];
        for k in expired_keys: del self.cache[k]
        return {'backend': 'memory', 'hits': self.hits, 'misses': self.misses,
                'hit_rate': round(self.hits / max(self.hits + self.misses, 1), 3),
                'entries': len(self.cache)}

if REDIS_AVAILABLE and config.REDIS_ENABLED:
    try: cache = RedisCache(config.REDIS_URL)
    except Exception as e: logger.warning(f"No se pudo conectar a Redis: {e}. Usando caché en memoria"); cache = MemoryCache()
else: cache = MemoryCache()

# ============================================================================
# DATA LOADER (SIMPLIFICADO)
# ============================================================================

class DataLoader:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.precomputed: Dict[str, Any] = {}
        self.load_data()

    def is_ready(self) -> bool: return self.df is not None and not self.df.empty
    def get_dataframe(self) -> pd.DataFrame:
        if not self.is_ready(): raise HTTPException(503, detail="Datos no disponibles")
        return self.df

    @staticmethod
    def _clean_float(val: Any) -> Optional[float]:
        if pd.isna(val) or val is None: return None
        try: return float(val)
        except (ValueError, TypeError): return None
    @staticmethod
    def _clean_string(val: Any) -> Optional[str]:
        if pd.isna(val) or val is None or val == '': return None
        return str(val).strip()

    def load_data(self):
        try:
            data_file = config.DATA_FILE
            if not data_file.exists(): logger.error(f"Archivo no encontrado: {data_file}"); return
            logger.info(f"Cargando datos desde {data_file}...")
            self.df = pd.read_csv(data_file, parse_dates=['date'], compression='gzip' if str(data_file).endswith('.gz') else None)

            required_cols = ['location', 'date']
            if not all(col in self.df.columns for col in required_cols):
                 missing = [col for col in required_cols if col not in self.df.columns]
                 logger.error(f"Columnas esenciales faltantes en el CSV: {missing}")
                 self.df = None
                 return

            if config.USE_INDEXES:
                self.df.set_index(['location', 'date'], inplace=True); self.df.sort_index(inplace=True)

            df_reset = self.df.reset_index() if config.USE_INDEXES else self.df
            self.metadata = {
                'total_records': len(self.df),
                'countries': len(df_reset['location'].unique()),
                'date_range': {'start': df_reset['date'].min().strftime('%Y-%m-%d'), 'end': df_reset['date'].max().strftime('%Y-%m-%d')},
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
                'columns': list(self.df.columns),
                'loaded_at': datetime.now().isoformat()
            }
            if config.PRECOMPUTE_SUMMARIES: self._precompute_summaries()
            logger.info(f"✅ Datos cargados: {self.metadata['total_records']:,} registros")
        except Exception as e: logger.error(f"Error cargando datos: {e}", exc_info=True); self.df = None

    def _precompute_summaries(self):
        if self.df is None: return
        logger.info("Precomputando métricas...")
        df = self.df.reset_index() if config.USE_INDEXES else self.df.copy()
        latest = df.groupby('location', observed=False).last().reset_index()
        available_cols = set(latest.columns)

        # Global summary
        global_summary = {}
        if 'total_cases' in available_cols: global_summary['total_cases'] = self._clean_float(latest['total_cases'].sum())
        if 'total_deaths' in available_cols: global_summary['total_deaths'] = self._clean_float(latest['total_deaths'].sum())
        if 'total_vaccinations' in available_cols: global_summary['total_vaccinations'] = self._clean_float(latest['total_vaccinations'].sum())
        
        # ====================================================================
        # >>>>> INICIO DE MODIFICACIÓN (API) <<<<<
        # ====================================================================
        if 'population' in available_cols: global_summary['total_population'] = self._clean_float(latest['population'].sum())
        # ====================================================================
        # >>>>> FIN DE MODIFICACIÓN (API) <<<<<
        # ====================================================================

        global_summary['countries_affected'] = int(latest['location'].nunique())
        global_summary['last_updated'] = latest['date'].max().strftime('%Y-%m-%d')
        self.precomputed['global_summary'] = global_summary

        # Map data
        possible_metrics = ['total_cases', 'total_deaths', 'total_vaccinations', 'people_fully_vaccinated', 'total_cases_per_million', 'total_deaths_per_million', 'population']
        available_map_metrics = [m for m in possible_metrics if m in available_cols]
        self.precomputed['map_data'] = {}
        for metric in available_map_metrics:
            map_data = []
            for _, row in latest.iterrows():
                value = self._clean_float(row.get(metric))
                iso_code = self._clean_string(row.get('iso_code'))
                if value is not None and value >= 0 and iso_code: # Allow 0 for maps
                    map_data.append({'country': row['location'], 'value': value,
                                     'population': self._clean_float(row.get('population')), 'iso_code': iso_code})
            self.precomputed['map_data'][metric] = map_data
        logger.info("✅ Métricas (global, map) precomputadas")

data_loader = DataLoader()

# ============================================================================
# DECORADOR CACHÉ
# ============================================================================
def cached_endpoint(ttl: int = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not config.ENABLE_CACHE: return await func(*args, **kwargs)
            request = kwargs.get('request'); cache_key = f"{func.__name__}:{request.url.path}:{str(sorted(request.query_params.items()))}"
            cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cached_response = cache.get(cache_key_hash)
            if cached_response is not None: return cached_response
            response = await func(*args, **kwargs)
            cache.set(cache_key_hash, response, ttl or config.CACHE_TTL)
            return response
        return wrapper
    return decorator

# ============================================================================
# LIFESPAN (SIMPLIFICADO)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70); logger.info(f"🚀 Iniciando {config.API_TITLE} v{config.API_VERSION} ({config.ENVIRONMENT})"); logger.info("=" * 70)
    if data_loader.is_ready():
        meta = data_loader.metadata
        logger.info(f"✅ Datos: {meta['total_records']:,} registros | {meta['countries']} países | Rango: {meta['date_range']['start']} a {meta['date_range']['end']}")
        logger.info(f"💾 Memoria: {meta['memory_usage_mb']:.2f} MB | ⚡ Cache: {cache.stats()['backend']} | 🔑 Auth: {'Enabled' if config.API_KEYS else 'Disabled (dev)'}")
    else: logger.error("❌ DATOS NO CARGADOS - API EN MODO DEGRADADO")
    logger.info("=" * 70)
    yield
    logger.info("🛑 Deteniendo API...")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title=config.API_TITLE, version=config.API_VERSION, description="API REST para datos COVID-19 por país",
              docs_url="/docs", redoc_url="/redoc", default_response_class=DEFAULT_RESPONSE_CLASS, lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address); app.state.limiter = limiter; app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=config.GZIP_MIN_SIZE)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])
if config.ALLOWED_HOSTS != ["*"]: app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)
app_start_time = time.time()

# ============================================================================
# MODELOS PYDANTIC (SIMPLIFICADOS)
# ============================================================================
class HealthResponse(BaseModel): status: str; version: str; uptime_seconds: float; data_loaded: bool; cache_backend: str
class TimeSeriesPoint(BaseModel): date: str; value: Optional[float]
# ====================================================================
# >>>>> INICIO DE MODIFICACIÓN (API) <<<<<
# ====================================================================
class GlobalSummary(BaseModel): 
    total_cases: Optional[float]
    total_deaths: Optional[float]
    total_vaccinations: Optional[float] = None
    total_population: Optional[float] = None # AÑADIDO
    countries_affected: int
    last_updated: str
# ====================================================================
# >>>>> FIN DE MODIFICACIÓN (API) <<<<<
# ====================================================================
class MapDataPoint(BaseModel): country: str; value: Optional[float]; population: Optional[float]; iso_code: str
class MapDataResponse(BaseModel): metric: str; total_countries: int; data: List[MapDataPoint]
class CountryListResponse(BaseModel): total: int; countries: List[str]
class CountryDataResponse(BaseModel): country: str; total_records: int; data: List[Dict]
class TimeSeriesResponse(BaseModel): country: str; metric: str; date_range: Dict[str, str]; total_points: int; data: List[TimeSeriesPoint]
class CompareResponsePoint(BaseModel): location: str; value: Optional[float]; population: Optional[float]; normalized_value: Optional[float] = None
class CompareResponse(BaseModel): metric: str; normalize: bool; data: List[CompareResponsePoint]
class StatisticsResponse(BaseModel): metric: str; grouping: str = 'global'; include_outliers: bool; statistics: Dict[str, Dict[str, Any]]
class CorrelationResponse(BaseModel): metrics: List[str]; method: str; correlation_matrix: Dict[str, Dict[str, Optional[float]]] # Allow None in matrix

# ============================================================================
# ENDPOINTS (SIMPLIFICADOS)
# ============================================================================

@app.get("/", tags=["Info"])
async def root(): return {"api": config.API_TITLE, "version": config.API_VERSION, "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["Info"], response_model=HealthResponse)
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def health_check(request: Request):
    return HealthResponse(status="healthy" if data_loader.is_ready() else "degraded", version=config.API_VERSION,
                          uptime_seconds=round(time.time() - app_start_time, 2), data_loaded=data_loader.is_ready(),
                          cache_backend=cache.stats()['backend'])

@app.get("/covid/global", tags=["Global Stats"], response_model=GlobalSummary, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint(ttl=1800)
async def get_global_summary(request: Request):
    if not data_loader.is_ready(): raise HTTPException(503, detail="Datos no disponibles")
    summary = data_loader.precomputed.get('global_summary')
    if not summary: # Fallback
        df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
        latest = df_reset.groupby('location', observed=False).last().reset_index()
        summary = {'countries_affected': int(latest['location'].nunique()), 'last_updated': latest['date'].max().strftime('%Y-%m-%d')}
        if 'total_cases' in latest.columns: summary['total_cases'] = data_loader._clean_float(latest['total_cases'].sum())
        if 'total_deaths' in latest.columns: summary['total_deaths'] = data_loader._clean_float(latest['total_deaths'].sum())
        if 'total_vaccinations' in latest.columns: summary['total_vaccinations'] = data_loader._clean_float(latest['total_vaccinations'].sum())
        # ====================================================================
        # >>>>> INICIO DE MODIFICACIÓN (API) <<<<<
        # ====================================================================
        if 'population' in latest.columns: summary['total_population'] = data_loader._clean_float(latest['population'].sum())
        # ====================================================================
        # >>>>> FIN DE MODIFICACIÓN (API) <<<<<
        # ====================================================================
    return summary

# ============================================================================
# >>>>> INICIO DE MODIFICACIÓN (API) <<<<<
# ============================================================================
@app.get("/covid/map-data", tags=["Global Stats"], response_model=MapDataResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def get_map_data(
    request: Request, 
    metric: str = Query("total_cases", description="Métrica para el mapa"),
    top: Optional[int] = Query(None, ge=1, description="Devolver solo los N primeros resultados") # AÑADIDO
):
    map_data = data_loader.precomputed.get('map_data', {}).get(metric)
    if not map_data: # Fallback
        df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
        latest = df_reset.groupby('location', observed=False).last().reset_index()
        if metric not in latest.columns: raise HTTPException(400, detail=f"Métrica '{metric}' no disponible")
        map_data = []
        for _, row in latest.iterrows():
            value = data_loader._clean_float(row.get(metric)); iso_code = data_loader._clean_string(row.get('iso_code'))
            if value is not None and value >= 0 and iso_code: # Allow 0
                map_data.append({'country': row['location'], 'value': value, 'population': data_loader._clean_float(row.get('population')), 'iso_code': iso_code})
    
    # APLICAR LÓGICA DE TOP N
    final_data = map_data
    if top:
        # Ordenar por 'value', manejar Nones por si acaso
        final_data = sorted(
            [d for d in map_data if d.get('value') is not None], 
            key=lambda x: x['value'], 
            reverse=True
        )
        final_data = final_data[:top] # Limitar a los N primeros

    return {'metric': metric, 'total_countries': len(final_data), 'data': final_data}
# ============================================================================
# >>>>> FIN DE MODIFICACIÓN (API) <<<<<
# ============================================================================

@app.get("/covid/countries", tags=["Countries"], response_model=CountryListResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def list_countries(request: Request):
    df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
    countries = df_reset['location'].unique().tolist()
    return {'total': len(countries), 'countries': sorted(countries)}

# ============================================================================
# >>>>> INICIO DE MODIFICACIÓN (API) <<<<<
# ============================================================================
@app.get("/covid/country/{country_name}", tags=["Countries"], response_model=CountryDataResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def get_country_data(
    request: Request, 
    country_name: str, 
    limit: int = Query(5000, ge=1, le=10000),
    # AÑADIR PARÁMETROS DE FECHA
    start_date: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD")
):
    df = data_loader.get_dataframe()
    
    # Obtener *todos* los datos del país primero
    if config.USE_INDEXES:
        if country_name not in df.index.get_level_values('location'): raise HTTPException(404, detail=f"País '{country_name}' no encontrado")
        country_df_full = df.loc[country_name].reset_index() 
    else: 
        country_df_full = df[df['location'] == country_name]
    
    if country_df_full.empty: raise HTTPException(404, detail=f"País '{country_name}' no encontrado")
    
    # --- APLICAR FILTROS DE FECHA AQUÍ ---
    if start_date:
        try:
            country_df_full = country_df_full[country_df_full['date'] >= pd.to_datetime(start_date)]
        except Exception as e:
            logger.warning(f"Error parseando start_date '{start_date}': {e}")
    if end_date:
        try:
            country_df_full = country_df_full[country_df_full['date'] <= pd.to_datetime(end_date)]
        except Exception as e:
            logger.warning(f"Error parseando end_date '{end_date}': {e}")
    # --- FIN DE FILTROS ---

    # Aplicar el limit *después* de filtrar por fecha (obtener los últimos N)
    country_df = country_df_full.tail(limit) 
    
    records = country_df.to_dict(orient='records')
    for record in records:
        record['date'] = record['date'].strftime('%Y-%m-%d') if pd.notna(record.get('date')) else None
        for key, value in list(record.items()):
             if pd.isna(value):
                 record[key] = None
             elif isinstance(value, (np.number, int, float)) and not isinstance(value, bool):
                 record[key] = float(value) if not float(value).is_integer() else int(value)
             elif key != 'date' and key != 'location' and key != 'iso_code':
                 try:
                     num_val = float(value)
                     record[key] = int(num_val) if num_val.is_integer() else num_val
                 except (ValueError, TypeError):
                     if key not in ['location', 'iso_code']:
                        record[key] = str(value)

    return {'country': country_name, 'total_records': len(records), 'data': records}
# ============================================================================
# >>>>> FIN DE MODIFICACIÓN (API) <<<<<
# ============================================================================

@app.get("/covid/timeseries/{country_name}", tags=["Time Series"], response_model=TimeSeriesResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def get_country_timeseries(request: Request, country_name: str, metric: str = Query("total_cases"),
                                 start_date: Optional[str] = Query(None), end_date: Optional[str] = Query(None)):
    df = data_loader.get_dataframe()
    if config.USE_INDEXES:
        if country_name not in df.index.get_level_values('location'): raise HTTPException(404, detail=f"País '{country_name}' no encontrado")
        country_df = df.loc[country_name].reset_index()
    else: country_df = df[df['location'] == country_name]
    if country_df.empty: raise HTTPException(404, detail=f"País '{country_name}' no encontrado")
    if metric not in country_df.columns: raise HTTPException(400, detail=f"Métrica '{metric}' no válida")
    if start_date: country_df = country_df[country_df['date'] >= pd.to_datetime(start_date)]
    if end_date: country_df = country_df[country_df['date'] <= pd.to_datetime(end_date)]
    country_df = country_df.sort_values(by='date')
    timeseries = [{'date': row['date'].strftime('%Y-%m-%d'), 'value': data_loader._clean_float(row[metric])} for _, row in country_df.iterrows()]
    start_ts = country_df['date'].min().strftime('%Y-%m-%d') if not country_df.empty else None
    end_ts = country_df['date'].max().strftime('%Y-%m-%d') if not country_df.empty else None

    return {'country': country_name, 'metric': metric, 'date_range': {'start': start_ts, 'end': end_ts},
            'total_points': len(timeseries), 'data': timeseries}

@app.get("/covid/compare", tags=["Comparison"], response_model=CompareResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def compare_countries(
    request: Request,
    countries: List[str] = Query(..., description="Países a comparar"),
    metric: str = Query("total_cases", description="Métrica para comparar"),
    normalize: bool = Query(False, description="Normalizar por población (por 100k)")
):
    df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
    if metric not in df_reset.columns: raise HTTPException(400, detail=f"Métrica '{metric}' no disponible")
    latest = df_reset.groupby('location', observed=False).last().reset_index()
    comparison = []
    for country in countries:
        country_df = latest[latest['location'] == country]
        if country_df.empty: continue
        row = country_df.iloc[0]
        value = data_loader._clean_float(row.get(metric))
        population = data_loader._clean_float(row.get('population'))
        result = {'location': country, 'value': value, 'population': population}
        if normalize and population and population > 0:
             result['normalized_value'] = round((value / population) * 100000, 2) if value is not None else None
        comparison.append(result)
    return {'metric': metric, 'normalize': normalize, 'data': comparison}

@app.get("/covid/metrics/statistics", tags=["Statistics"], response_model=StatisticsResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def get_statistics(
    request: Request,
    metric: str = Query("total_cases", description="Métrica para análisis"),
    include_outliers: bool = Query(False, description="Incluir outliers")
):
    df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
    if metric not in df_reset.columns: raise HTTPException(400, detail=f"Métrica '{metric}' no disponible")
    latest = df_reset.groupby('location', observed=False).last().reset_index()
    values = latest[metric].dropna()
    statistics = {}
    if not values.empty:
        if not include_outliers and len(values) > 4:
            Q1 = values.quantile(0.25); Q3 = values.quantile(0.75); IQR = Q3 - Q1
            if pd.notna(IQR) and IQR > 0:
                lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
                values = values[(values >= lower_bound) & (values <= upper_bound)]
            elif pd.notna(IQR) and IQR == 0:
                values = values[values == Q1]

        if not values.empty:
            stats_dict = {
                'mean': float(values.mean()), 'median': float(values.median()),
                'std': float(values.std()) if len(values) > 1 else 0.0,
                'min': float(values.min()), 'max': float(values.max()),
                'count': int(len(values))
            }
            if len(values) >= 4:
                 stats_dict['q25'] = float(values.quantile(0.25))
                 stats_dict['q75'] = float(values.quantile(0.75))
            else:
                 stats_dict['q25'] = float(values.median()) if len(values) > 0 else None
                 stats_dict['q75'] = float(values.median()) if len(values) > 0 else None
            statistics['global'] = stats_dict
        else:
             statistics['global'] = {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'q25': None, 'q75': None, 'count': 0}
    else:
        statistics['global'] = {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'q25': None, 'q75': None, 'count': 0}

    return {'metric': metric, 'grouping': 'global', 'include_outliers': include_outliers, 'statistics': statistics}

@app.get("/covid/metrics/correlations", tags=["Statistics"], response_model=CorrelationResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@cached_endpoint()
async def get_correlations(
    request: Request,
    metrics: List[str] = Query(..., min_items=2, description="Métricas para correlacionar"),
    method: str = Query("pearson", regex="^(pearson|spearman)$", description="Método de correlación")
):
    df = data_loader.get_dataframe(); df_reset = df.reset_index() if config.USE_INDEXES else df
    latest = df_reset.groupby('location', observed=False).last().reset_index()
    valid_metrics = []
    for m in metrics:
        if m not in latest.columns: raise HTTPException(400, detail=f"Métrica '{m}' no disponible")
        if not pd.api.types.is_numeric_dtype(latest[m]):
             raise HTTPException(400, detail=f"Métrica '{m}' no es numérica y no puede usarse para correlación.")
        valid_metrics.append(m)

    corr_matrix = latest[valid_metrics].corr(method=method)
    corr_matrix.replace([np.inf, -np.inf], None, inplace=True)
    corr_dict = corr_matrix.where(pd.notnull(corr_matrix), None).to_dict()

    final_corr_dict = {
        outer_key: {inner_key: float(value) if value is not None else None for inner_key, value in inner_values.items()}
        for outer_key, inner_values in corr_dict.items()
    }

    return {'metrics': valid_metrics, 'method': method, 'correlation_matrix': final_corr_dict}

# --- Endpoints /admin/... ---
@app.post("/admin/cache/clear", tags=["Admin"], status_code=202, dependencies=[Depends(verify_admin_key)])
async def clear_cache(background_tasks: BackgroundTasks):
    background_tasks.add_task(cache.clear)
    return {"message": "Limpieza de caché programada"}

@app.get("/admin/stats", tags=["Admin"], dependencies=[Depends(verify_admin_key)])
async def get_admin_stats():
    return {
        "cache": cache.stats(),
        "data": data_loader.metadata,
        "uptime_seconds": round(time.time() - app_start_time, 2),
        "environment": config.ENVIRONMENT
    }

@app.post("/admin/reload-data", tags=["Admin"], status_code=202,
          dependencies=[Depends(verify_admin_key)])
async def reload_data(background_tasks: BackgroundTasks):
    """Recarga datos desde archivo (solo admin)."""
    def reload():
        logger.info("Iniciando recarga de datos en segundo plano...")
        try:
            data_loader.load_data()
            logger.info("Recarga de datos completada exitosamente.")
            cache.clear()
            logger.info("Caché limpiado después de la recarga de datos.")
        except Exception as e:
            logger.error(f"Error durante la recarga de datos: {e}", exc_info=True)

    background_tasks.add_task(reload)
    return {"message": "Recarga de datos programada"}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70); print(f"🚀 Ejecutando {config.API_TITLE} v{config.API_VERSION} ({config.ENVIRONMENT})"); print("=" * 70)
    if data_loader.is_ready():
        print(f"\n✅ Estado: OPERACIONAL | Registros: {data_loader.metadata['total_records']:,}")
        print(f"\n📖 Accede a la API en: http://127.0.0.1:8000")
        print(f"📖 Documentación: http://127.0.0.1:8000/docs")
        if config.API_KEYS: print(f"\n🔑 Requiere API Key (X-API-Key) | Keys: {len(config.API_KEYS)}")
        else: print("\n⚠️ Sin autenticación (modo desarrollo)")
        print("\n" + "=" * 70 + "\n"); uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    else: print("\n❌ ERROR: Datos no cargados. Verifica el archivo CSV y las columnas."); print("=" * 70 + "\n"); sys.exit(1)