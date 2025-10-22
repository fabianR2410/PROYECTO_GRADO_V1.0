"""
COVID-19 REST API - PRODUCTION VERSION
=======================================
API REST segura con autenticación, Redis, rate limiting robusto y monitoring.
"""

import os
import sys
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
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
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Variables de entorno
try:
    from decouple import config as env_config, Csv
except ImportError:
    # Fallback si no está instalado decouple
    class DummyConfig:
        def __call__(self, key, default=None, cast=None):
            value = os.getenv(key, default)
            if cast and value is not None:
                if cast == Csv():
                    return value.split(',') if value else []
                return cast(value)
            return value
    env_config = DummyConfig()
    Csv = lambda: list

# Redis para caché (opcional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis no disponible. Usando caché en memoria.")

# ORJSON para respuestas rápidas
try:
    from fastapi.responses import ORJSONResponse
    DEFAULT_RESPONSE_CLASS = ORJSONResponse
    USING_ORJSON = True
except ImportError:
    DEFAULT_RESPONSE_CLASS = JSONResponse
    USING_ORJSON = False

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
LOGS_DIR = BASE_DIR / os.getenv("LOGS_DIR", "logs")

# Logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / 'api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

class APIConfig:
    """Configuración de la API desde variables de entorno."""
    
    # Archivos
    DATA_FILE = DATA_DIR / "processed_covid.csv.gz"
    CONTINENT_METRICS_FILE = DATA_DIR / "continent_metrics.json"
    REGION_METRICS_FILE = DATA_DIR / "region_metrics.json"
    
    # Metadata
    API_TITLE = os.getenv("API_TITLE", "COVID-19 Data API")
    API_VERSION = os.getenv("API_VERSION", "4.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Seguridad
    API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    ADMIN_API_KEYS = os.getenv("ADMIN_API_KEYS", "").split(",") if os.getenv("ADMIN_API_KEYS") else []
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    
    # Cache
    CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    # Performance
    PRECOMPUTE_SUMMARIES = os.getenv("PRECOMPUTE_SUMMARIES", "true").lower() == "true"
    USE_INDEXES = os.getenv("USE_INDEXES", "true").lower() == "true"
    GZIP_MIN_SIZE = int(os.getenv("GZIP_MIN_SIZE", "500"))


config = APIConfig()


# ============================================================================
# SEGURIDAD - AUTENTICACIÓN
# ============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "No autorizado"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Verifica API key para endpoints públicos."""
    # En desarrollo, permitir acceso sin API key
    if config.ENVIRONMENT == "development" and not config.API_KEYS:
        return "dev-key"
    
    if not api_key:
        raise UnauthorizedException("API key requerida en header X-API-Key")
    
    if api_key not in config.API_KEYS:
        logger.warning(f"Intento de acceso con API key inválida")
        raise UnauthorizedException("API key inválida")
    
    return api_key


def verify_admin_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Verifica API key de administrador."""
    if config.ENVIRONMENT == "development" and not config.ADMIN_API_KEYS:
        return "dev-admin-key"
    
    if not api_key:
        raise UnauthorizedException("API key de administrador requerida")
    
    if api_key not in config.ADMIN_API_KEYS:
        logger.warning("Intento de acceso admin con key inválida")
        raise UnauthorizedException("No tiene permisos de administrador")
    
    return api_key


# ============================================================================
# CACHÉ CON REDIS O MEMORIA
# ============================================================================

class CacheBackend:
    """Backend abstracto de caché."""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        raise NotImplementedError
    
    def clear(self) -> None:
        raise NotImplementedError
    
    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class RedisCache(CacheBackend):
    """Caché usando Redis."""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.hits = 0
        self.misses = 0
        logger.info(f"✅ Redis conectado: {redis_url}")
    
    def get(self, key: str) -> Optional[Any]:
        try:
            import pickle
            data = self.redis_client.get(key)
            if data:
                self.hits += 1
                return pickle.loads(data)
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error leyendo de Redis: {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        try:
            import pickle
            self.redis_client.setex(key, ttl, pickle.dumps(value))
        except Exception as e:
            logger.error(f"Error escribiendo en Redis: {e}")
    
    def delete(self, key: str) -> None:
        self.redis_client.delete(key)
    
    def clear(self) -> None:
        self.redis_client.flushdb()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        info = self.redis_client.info()
        total = self.hits + self.misses
        return {
            "backend": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round((self.hits / total * 100), 2) if total > 0 else 0,
            "keys": info.get("db0", {}).get("keys", 0),
            "memory_used_mb": round(info.get("used_memory", 0) / (1024 ** 2), 2)
        }


class MemoryCache(CacheBackend):
    """Caché en memoria (fallback)."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        logger.info("⚠️ Usando caché en memoria (no persistente)")
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry["expires_at"]:
                self.hits += 1
                return entry["value"]
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
    
    def delete(self, key: str) -> None:
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "memory",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round((self.hits / total * 100), 2) if total > 0 else 0,
            "keys": len(self.cache),
            "memory_used_mb": round(sys.getsizeof(self.cache) / (1024 ** 2), 2)
        }


# Inicializar caché
if config.REDIS_ENABLED and REDIS_AVAILABLE:
    try:
        cache = RedisCache(config.REDIS_URL)
    except Exception as e:
        logger.error(f"No se pudo conectar a Redis: {e}. Usando memoria.")
        cache = MemoryCache()
else:
    cache = MemoryCache()


def cached(ttl: Optional[int] = None):
    """Decorator para cachear respuestas."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.ENABLE_CACHE:
                return func(*args, **kwargs)
            
            # Generar key del caché
            key_parts = [func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Intentar obtener del caché
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Ejecutar función y cachear
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl or config.CACHE_TTL)
            return result
        
        return wrapper
    return decorator


# ============================================================================
# DATA LOADER
# ============================================================================

class OptimizedDataLoader:
    """Cargador de datos optimizado."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.df_indexed: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.precomputed: Dict[str, Any] = {}
        self.continent_metrics: Dict[str, Any] = {}
        self.region_metrics: Dict[str, Any] = {}
        self.load_data()
    
    def load_data(self) -> None:
        """Carga datos al iniciar."""
        logger.info(f"📂 Cargando datos desde {config.DATA_FILE}")
        
        if not config.DATA_FILE.exists():
            logger.error(f"❌ Archivo no encontrado: {config.DATA_FILE}")
            logger.error("💡 Ejecuta 'python etl_pipeline.py' primero")
            return
        
        try:
            # Cargar CSV
            dtype_dict = {
                'iso_code': 'category',
                'continent': 'category',
                'location': 'category'
            }
            
            self.df = pd.read_csv(
                config.DATA_FILE,
                compression="gzip",
                parse_dates=["date"],
                dtype=dtype_dict,
                low_memory=False
            )
            
            # Optimizar tipos
            self.df = self._optimize_dtypes(self.df)
            self.df = self.df.sort_values(['location', 'date']).reset_index(drop=True)
            
            # Crear índice
            if config.USE_INDEXES:
                self.df_indexed = self.df.set_index('location', drop=False)
                logger.info("⚡ Índices activados")
            
            # Metadata
            self._populate_metadata()
            
            # Cargar métricas agregadas
            self._load_aggregate_metrics()
            
            # Pre-calcular
            if config.PRECOMPUTE_SUMMARIES:
                self._precompute_data()
            
            logger.info(f"✅ Datos cargados: {self.metadata['total_records']:,} registros")
            logger.info(f"💾 Memoria: {self.metadata['memory_usage_mb']:.2f} MB")
            
        except Exception as e:
            logger.exception(f"❌ Error cargando datos: {e}")
            self.df = None
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza tipos de datos."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except (TypeError, ValueError):
                pass
        return df
    
    def _populate_metadata(self) -> None:
        """Calcula metadata."""
        if self.df is None:
            return
        
        date_col = self.df['date']
        self.metadata = {
            'total_records': len(self.df),
            'countries': self.df['location'].nunique(),
            'columns': len(self.df.columns),
            'date_range': {
                'start': date_col.min().strftime('%Y-%m-%d') if pd.notna(date_col.min()) else None,
                'end': date_col.max().strftime('%Y-%m-%d') if pd.notna(date_col.max()) else None,
            },
            'loaded_at': datetime.now().isoformat(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _load_aggregate_metrics(self) -> None:
        """Carga métricas de JSON."""
        import json
        
        if config.CONTINENT_METRICS_FILE.exists():
            try:
                with open(config.CONTINENT_METRICS_FILE, 'r') as f:
                    self.continent_metrics = json.load(f)
                logger.info("✅ Métricas continentes cargadas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando métricas continentes: {e}")
        
        if config.REGION_METRICS_FILE.exists():
            try:
                with open(config.REGION_METRICS_FILE, 'r') as f:
                    self.region_metrics = json.load(f)
                logger.info("✅ Métricas regiones cargadas")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando métricas regiones: {e}")
    
    def _precompute_data(self) -> None:
        """Pre-calcula datos comunes."""
        if self.df is None:
            return
        
        logger.info("🔄 Pre-calculando datos...")
        start_time = time.time()
        
        # Lista de países
        latest = self.df.groupby('location').last().reset_index()
        counts = self.df['location'].value_counts().to_dict()
        
        countries_list = []
        for _, row in latest.iterrows():
            countries_list.append({
                'name': row['location'],
                'iso_code': self._clean_string(row.get('iso_code')),
                'continent': self._clean_string(row.get('continent')),
                'population': self._clean_float(row.get('population')),
                'data_points': counts.get(row['location'], 0),
                'latest_date': row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None
            })
        
        self.precomputed['countries_list'] = countries_list
        
        # Resúmenes por país
        summaries = {}
        for _, row in latest.iterrows():
            summaries[row['location']] = {
                'country': row['location'],
                'iso_code': self._clean_string(row.get('iso_code')),
                'continent': self._clean_string(row.get('continent')),
                'latest_date': row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else None,
                'population': self._clean_float(row.get('population')),
                'total_cases': self._clean_float(row.get('total_cases')),
                'total_deaths': self._clean_float(row.get('total_deaths')),
                'people_fully_vaccinated': self._clean_float(row.get('people_fully_vaccinated')),
                'vaccination_rate': self._clean_float(row.get('vaccination_rate')),
                'mortality_rate': self._clean_float(row.get('mortality_rate'))
            }
        
        self.precomputed['summaries'] = summaries
        
        # Datos para mapas
        map_data = {}
        map_metrics = ['total_cases', 'total_deaths', 'people_fully_vaccinated', 'vaccination_rate']
        
        for metric in map_metrics:
            if metric not in latest.columns:
                continue
            
            metric_data = []
            for _, row in latest.iterrows():
                value = self._clean_float(row.get(metric))
                iso = self._clean_string(row.get('iso_code'))
                if value is not None and value > 0 and iso:
                    metric_data.append({
                        'country': row['location'],
                        'value': value,
                        'population': self._clean_float(row.get('population')),
                        'iso_code': iso
                    })
            
            map_data[metric] = metric_data
        
        self.precomputed['map_data'] = map_data
        self.precomputed['continent_metrics'] = self.continent_metrics
        self.precomputed['region_metrics'] = self.region_metrics
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Pre-cálculos completados en {elapsed:.2f}s")
    
    def _clean_string(self, value: Any) -> Optional[str]:
        """Limpia valores string."""
        if value is None or pd.isna(value):
            return None
        return str(value)
    
    def _clean_float(self, value: Any) -> Optional[float]:
        """Limpia valores float."""
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)) and (np.isinf(value) or np.isnan(value)):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def is_ready(self) -> bool:
        """Verifica si los datos están listos."""
        return isinstance(self.df, pd.DataFrame) and not self.df.empty
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame."""
        if not self.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Datos no disponibles. Intenta más tarde."
            )
        return self.df
    
    def get_country_data_fast(self, country: str) -> pd.DataFrame:
        """Obtiene datos de país con índice."""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Datos no disponibles")
        
        if config.USE_INDEXES and self.df_indexed is not None:
            try:
                result = self.df_indexed[self.df_indexed['location'] == country].copy()
                if not result.empty:
                    return result
            except Exception as e:
                logger.warning(f"Error usando índice: {e}")
        
        country_df = self.df[self.df['location'] == country].copy()
        if country_df.empty:
            raise HTTPException(status_code=404, detail=f"País '{country}' no encontrado")
        return country_df


# Global data loader
data_loader = OptimizedDataLoader()


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class TimeSeriesPoint(BaseModel):
    date: str
    value: Optional[float] = None


class TimeSeriesResponse(BaseModel):
    country: str
    metric: str
    total_points: int
    data: List[TimeSeriesPoint]


class SummaryStats(BaseModel):
    country: str
    iso_code: Optional[str] = None
    continent: Optional[str] = None
    latest_date: str
    population: Optional[float] = None
    total_cases: Optional[float] = None
    total_deaths: Optional[float] = None
    people_fully_vaccinated: Optional[float] = None
    vaccination_rate: Optional[float] = None
    mortality_rate: Optional[float] = None


class MapDataPoint(BaseModel):
    country: str
    value: float
    population: Optional[float] = None
    iso_code: Optional[str] = None


class MapDataResponse(BaseModel):
    metric: str
    total_countries: int
    data: List[MapDataPoint]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    data_loaded: bool
    cache_stats: Dict[str, Any]
    uptime_seconds: float


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="API REST para datos COVID-19 con seguridad y caché",
    docs_url="/docs" if config.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if config.ENVIRONMENT == "development" else None,
    default_response_class=DEFAULT_RESPONSE_CLASS
)

# Rate limiter con slowapi
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
if config.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.ALLOWED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=config.GZIP_MIN_SIZE)

# Variables globales
app_start_time = time.time()


# ============================================================================
# UTILIDADES
# ============================================================================

@cached(ttl=config.CACHE_TTL)
def get_country_data_cached(country: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
    """Obtiene datos de país con caché."""
    df = data_loader.get_country_data_fast(country)
    
    if start_date:
        try:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Formato de start_date inválido (usa YYYY-MM-DD)")
    
    if end_date:
        try:
            df = df[df['date'] <= pd.Timestamp(end_date)]
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Formato de end_date inválido (usa YYYY-MM-DD)")
    
    return df.copy()


def clean_float(value: Any) -> Optional[float]:
    """Limpia valores float."""
    return data_loader._clean_float(value)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Información de la API."""
    return {
        "api_name": config.API_TITLE,
        "version": config.API_VERSION,
        "environment": config.ENVIRONMENT,
        "status": "operational" if data_loader.is_ready() else "degraded",
        "documentation": "/docs" if config.ENVIRONMENT == "development" else "disabled in production",
        "authentication": "API Key required (X-API-Key header)" if config.API_KEYS else "No authentication in development mode",
        "endpoints": {
            "health": "/health",
            "countries": "/covid/countries",
            "summary": "/covid/summary?country=Ecuador",
            "timeseries": "/covid/timeseries?country=Ecuador&metric=new_cases",
            "compare": "/covid/compare?countries=Ecuador&countries=Colombia&metric=total_cases",
            "map_data": "/covid/map-data?metric=total_cases"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check con métricas."""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy" if data_loader.is_ready() else "degraded",
        timestamp=datetime.now().isoformat(),
        environment=config.ENVIRONMENT,
        data_loaded=data_loader.is_ready(),
        cache_stats=cache.stats() if config.ENABLE_CACHE else {},
        uptime_seconds=round(uptime, 2)
    )


@app.get("/covid/countries", tags=["Data"], dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def get_countries(request: Request):
    """Lista de países disponibles."""
    countries = data_loader.precomputed.get('countries_list')
    if countries:
        return {"total": len(countries), "countries": countries}
    
    # Fallback
    df = data_loader.get_dataframe()
    latest = df.groupby('location').last().reset_index()
    countries_list = [
        {
            'name': row['location'],
            'iso_code': data_loader._clean_string(row.get('iso_code')),
            'continent': data_loader._clean_string(row.get('continent'))
        }
        for _, row in latest.iterrows()
    ]
    return {"total": len(countries_list), "countries": countries_list}


@app.get("/covid/summary", response_model=SummaryStats, tags=["Data"], 
         dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def get_summary(request: Request, country: str = Query(..., description="Nombre del país")):
    """Resumen de país."""
    summary = data_loader.precomputed.get('summaries', {}).get(country)
    if summary:
        return SummaryStats(**summary)
    
    # Fallback
    country_df = data_loader.get_country_data_fast(country)
    if country_df.empty:
        raise HTTPException(404, detail=f"País '{country}' no encontrado")
    
    latest = country_df.iloc[-1]
    return SummaryStats(
        country=country,
        iso_code=data_loader._clean_string(latest.get('iso_code')),
        continent=data_loader._clean_string(latest.get('continent')),
        latest_date=latest['date'].strftime('%Y-%m-%d'),
        population=clean_float(latest.get('population')),
        total_cases=clean_float(latest.get('total_cases')),
        total_deaths=clean_float(latest.get('total_deaths')),
        people_fully_vaccinated=clean_float(latest.get('people_fully_vaccinated')),
        vaccination_rate=clean_float(latest.get('vaccination_rate')),
        mortality_rate=clean_float(latest.get('mortality_rate'))
    )


@app.get("/covid/timeseries", response_model=TimeSeriesResponse, tags=["Data"],
         dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def get_timeseries(
    request: Request,
    country: str = Query(..., description="Nombre del país"),
    metric: str = Query(..., description="Métrica (ej: new_cases, total_deaths)"),
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Fecha inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Fecha fin (YYYY-MM-DD)")
):
    """Serie temporal."""
    df = data_loader.get_dataframe()
    if metric not in df.columns:
        raise HTTPException(400, detail=f"Métrica '{metric}' inválida. Columnas disponibles: {', '.join(df.columns.tolist())}")
    
    country_df = get_country_data_cached(country, start_date, end_date)
    ts_data = country_df[['date', metric]].dropna(subset=[metric])
    
    data_points = [
        TimeSeriesPoint(
            date=row['date'].strftime('%Y-%m-%d'),
            value=clean_float(row[metric])
        )
        for _, row in ts_data.iterrows()
    ]
    
    return TimeSeriesResponse(
        country=country,
        metric=metric,
        total_points=len(data_points),
        data=data_points
    )


@app.get("/covid/compare", tags=["Data"], dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def compare_countries(
    request: Request,
    countries: List[str] = Query(..., min_length=2, max_length=10, description="Lista de países a comparar"),
    metric: str = Query(..., description="Métrica a comparar"),
    start_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
):
    """Compara métrica entre países."""
    df = data_loader.get_dataframe()
    if metric not in df.columns:
        raise HTTPException(400, detail=f"Métrica '{metric}' inválida")
    
    # Validar países
    available = set(df['location'].unique())
    missing = [c for c in countries if c not in available]
    if missing:
        raise HTTPException(404, detail=f"Países no encontrados: {', '.join(missing)}")
    
    # Filtrar datos
    compare_df = df[df['location'].isin(countries)].copy()
    
    if start_date:
        compare_df = compare_df[compare_df['date'] >= pd.Timestamp(start_date)]
    if end_date:
        compare_df = compare_df[compare_df['date'] <= pd.Timestamp(end_date)]
    
    # Pivotar
    pivot_data = compare_df.pivot_table(
        index='date',
        columns='location',
        values=metric,
        aggfunc='first'
    ).reset_index()
    
    # Formatear salida
    comparison_data = []
    for _, row in pivot_data.iterrows():
        point = {
            'date': row['date'].strftime('%Y-%m-%d'),
            'values': {
                country: clean_float(row.get(country))
                for country in countries if country in row and pd.notna(row.get(country))
            }
        }
        if point['values']:  # Solo incluir si hay valores
            comparison_data.append(point)
    
    return {
        'countries': countries,
        'metric': metric,
        'total_points': len(comparison_data),
        'data': comparison_data
    }


@app.get("/covid/map-data", response_model=MapDataResponse, tags=["Data"],
         dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def get_map_data(request: Request, metric: str = Query("total_cases", description="Métrica para el mapa")):
    """Datos para mapa."""
    map_data = data_loader.precomputed.get('map_data', {}).get(metric)
    if map_data:
        return MapDataResponse(
            metric=metric,
            total_countries=len(map_data),
            data=map_data
        )
    
    # Fallback
    df = data_loader.get_dataframe()
    if metric not in df.columns:
        raise HTTPException(400, detail=f"Métrica '{metric}' inválida")
    
    latest = df.groupby('location').last().reset_index()
    map_data_list = []
    
    for _, row in latest.iterrows():
        value = clean_float(row.get(metric))
        iso_code = data_loader._clean_string(row.get('iso_code'))
        if value and value > 0 and iso_code:
            map_data_list.append({
                'country': row['location'],
                'value': value,
                'population': clean_float(row.get('population')),
                'iso_code': iso_code
            })
    
    return MapDataResponse(
        metric=metric,
        total_countries=len(map_data_list),
        data=map_data_list
    )


@app.get("/covid/metrics/continents", tags=["Metrics"], 
         dependencies=[Depends(verify_api_key)])
async def get_continent_metrics():
    """Métricas por continente."""
    metrics = data_loader.precomputed.get('continent_metrics')
    if metrics:
        return metrics
    raise HTTPException(404, detail="Métricas no disponibles")


@app.get("/covid/metrics/regions", tags=["Metrics"],
         dependencies=[Depends(verify_api_key)])
async def get_region_metrics():
    """Métricas por región."""
    metrics = data_loader.precomputed.get('region_metrics')
    if metrics:
        return metrics
    raise HTTPException(404, detail="Métricas no disponibles")


@app.post("/admin/cache/clear", tags=["Admin"], status_code=202,
          dependencies=[Depends(verify_admin_key)])
async def clear_cache(background_tasks: BackgroundTasks):
    """Limpia caché (solo admin)."""
    background_tasks.add_task(cache.clear)
    return {"message": "Limpieza de caché programada"}


@app.get("/admin/stats", tags=["Admin"], dependencies=[Depends(verify_admin_key)])
async def get_admin_stats():
    """Estadísticas de la API (solo admin)."""
    return {
        "cache": cache.stats(),
        "data": data_loader.metadata,
        "uptime_seconds": round(time.time() - app_start_time, 2),
        "environment": config.ENVIRONMENT,
        "config": {
            "rate_limit_per_minute": config.RATE_LIMIT_PER_MINUTE,
            "cache_ttl": config.CACHE_TTL,
            "precompute_enabled": config.PRECOMPUTE_SUMMARIES
        }
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio."""
    logger.info("=" * 70)
    logger.info(f"🚀 Iniciando {config.API_TITLE} v{config.API_VERSION}")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info("=" * 70)
    
    if data_loader.is_ready():
        meta = data_loader.metadata
        logger.info(f"✅ Datos: {meta['total_records']:,} registros")
        logger.info(f"💾 Memoria: {meta['memory_usage_mb']:.2f} MB")
        logger.info(f"🌍 Países: {meta['countries']}")
        logger.info(f"⚡ Cache: {cache.stats()['backend']}")
        logger.info(f"🔑 Auth: {'Enabled' if config.API_KEYS else 'Disabled (dev mode)'}")
    else:
        logger.error("❌ DATOS NO CARGADOS - API EN MODO DEGRADADO")
    
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre."""
    logger.info("🛑 Deteniendo API...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"🚀 Ejecutando {config.API_TITLE} v{config.API_VERSION}")
    print(f"Environment: {config.ENVIRONMENT}")
    print("=" * 70)
    
    if data_loader.is_ready():
        print(f"\n✅ Estado: OPERACIONAL")
        print(f"📊 Registros: {data_loader.metadata['total_records']:,}")
        print(f"\n📖 Accede a la API en:")
        print(f"   → http://127.0.0.1:8000")
        print(f"   → http://127.0.0.1:8000")
        print(f"\n📖 Documentación:")
        print(f"   → http://127.0.0.1:8000/docs")
        print(f"   → http://127.0.0.1:8000/docs")
        
        if config.API_KEYS:
            print("\n🔑 Requiere API Key en header X-API-Key")
            print(f"   API Keys configuradas: {len(config.API_KEYS)}")
        else:
            print("\n⚠️  Sin autenticación (modo desarrollo)")
        
        print("\n" + "=" * 70 + "\n")
        
        uvicorn.run(
            "covid_api:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False
        )
    else:
        print("\n❌ ERROR: Datos no cargados")
        print("💡 Ejecuta: python etl_pipeline.py")
        print("=" * 70 + "\n")
        sys.exit(1)