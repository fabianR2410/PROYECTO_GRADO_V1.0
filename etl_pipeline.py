"""
ETL Pipeline COVID-19
======================
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import hashlib

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Importar configuración de variables de entorno
try:
    from decouple import config as env_config, Csv
except ImportError:
    print("ERROR: python-decouple no instalado. Ejecuta: pip install python-decouple")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

# ETIQUETA: Clase para formatear los mensajes de log (ej. añadir emojis o formato JSON)
class ProductionFormatter(logging.Formatter):
    """Formateador de logs estructurado para producción con soporte JSON opcional."""
    
    EMOJI_MAP: Dict[str, str] = {
        "🚀": "[START]", "✅": "[OK]", "❌": "[ERROR]", "⚠️": "[WARN]",
        "📊": "[INFO]", "📥": "[DOWNLOAD]", "🔧": "[CONFIG]"
    }

    # ETIQUETA: Constructor de la clase Formatter
    def __init__(self, fmt: str, json_format: bool = False):
        super().__init__(fmt)
        self.json_format = json_format

    # ETIQUETA: Método que aplica el formato (texto o JSON) a un mensaje de log
    def format(self, record: logging.LogRecord) -> str:
        """Formatea logs como texto o JSON según configuración."""
        if self.json_format:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)
        
        # Formato texto con reemplazo de emojis
        text = super().format(record)
        for emoji, replacement in self.EMOJI_MAP.items():
            text = text.replace(emoji, replacement)
        return text

# ETIQUETA: Función para configurar el sistema de logging (logs en consola y en archivo)
def setup_logging(log_file: Path, json_logs: bool = False) -> logging.Logger:
    """
    Configura logging con rotación y múltiples handlers.
    
    Args:
        log_file: Path del archivo de log
        json_logs: Si True, usa formato JSON para logs
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("covid_etl")
    if logger.hasHandlers():
        return logger

    log_level = env_config("LOG_LEVEL", default="INFO")
    logger.setLevel(getattr(logging, log_level))

    # Formato de logs
    if json_logs:
        fmt = "%(message)s"  # JSON se formatea en el formatter
    else:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"

    # Handler de consola
    console = logging.StreamHandler()
    console.setFormatter(ProductionFormatter(fmt, json_format=json_logs))
    logger.addHandler(console)

    # Handler de archivo con rotación
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from logging.handlers import RotatingFileHandler
        # 10MB por archivo, mantener 5 backups
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(ProductionFormatter(fmt, json_format=json_logs))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"No se pudo configurar RotatingFileHandler: {e}")

    return logger


# ============================================================================
# CONFIGURACIÓN CON VARIABLES DE ENTORNO
# ============================================================================

# ETIQUETA: Clase para almacenar todas las variables de configuración (rutas, URLs, parámetros)
@dataclass
class Config:
    """Configuración del pipeline desde variables de entorno."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    
    # Archivos
    INPUT_FILE: Path = field(init=False)
    OUTPUT_RAW: Path = field(init=False)
    OUTPUT_PROCESSED: Path = field(init=False)
    OUTPUT_COMPRESSED: Path = field(init=False)
    OUTPUT_REPORT: Path = field(init=False)
    OUTPUT_SUMMARY: Path = field(init=False)
    OUTPUT_CONTINENT_METRICS: Path = field(init=False)
    OUTPUT_REGION_METRICS: Path = field(init=False)
    LOG_FILE: Path = field(init=False)
    
    # URLs de datos (desde .env)
    DATA_URLS: List[str] = field(init=False)
    
    # Columnas y configuración de procesamiento
    CORE_COLUMNS: List[str] = field(default_factory=lambda: [
        "iso_code", "continent", "location", "date", "population",
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "total_tests", "new_tests", "positive_rate",
        "people_vaccinated", "people_fully_vaccinated", "total_boosters",
        "hosp_patients", "icu_patients", "reproduction_rate",
        "stringency_index", "population_density", "median_age",
        "aged_65_older", "aged_70_older", "gdp_per_capita",
        "cardiovasc_death_rate", "diabetes_prevalence", "life_expectancy"
    ])
    
    NEGATIVE_COLS_TO_CLIP: List[str] = field(default_factory=lambda: [
        "total_cases", "new_cases", "total_deaths", "new_deaths",
        "total_tests", "new_tests", "population", "hosp_patients",
        "icu_patients", "people_vaccinated", "people_fully_vaccinated"
    ])
    
    MONOTONIC_COLS: List[str] = field(default_factory=lambda: [
        "total_cases", "total_deaths", "total_tests",
        "people_vaccinated", "people_fully_vaccinated"
    ])
    
    CUMULATIVE_COLS_TO_FFILL: List[str] = field(default_factory=lambda: [
        "total_cases", "total_deaths", "people_vaccinated",
        "people_fully_vaccinated", "total_boosters", "total_tests"
    ])
    
    DEMOGRAPHIC_COLS_TO_FILL: List[str] = field(default_factory=lambda: [
        "population", "population_density", "median_age",
        "aged_65_older", "gdp_per_capita", "life_expectancy"
    ])
    
    DAILY_COLS_TO_INTERPOLATE: List[str] = field(default_factory=lambda: [
        "new_cases", "new_deaths", "new_tests"
    ])
    
    OUTLIER_COLS: List[str] = field(default_factory=lambda: [
        "new_cases", "new_deaths", "new_tests"
    ])
    
    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        "iso_code", "continent", "location"
    ])
    
    AGGREGATES_TO_EXCLUDE: List[str] = field(default_factory=lambda: [
        "World", "International", "High income", "Low income",
        "Lower middle income", "Upper middle income", "European Union",
        "Africa", "Asia", "Europe", "North America", "South America",
        "Oceania", "Antarctica"
    ])
    
    CONTINENT_NAMES: List[str] = field(default_factory=lambda: [
        "Africa", "Asia", "Europe", "North America", "South America", "Oceania"
    ])
    
    REGION_NAMES: List[str] = field(default_factory=lambda: [
        "World", "International", "High income", "Low income",
        "Lower middle income", "Upper middle income", "European Union"
    ])
    
    # Parámetros de procesamiento
    ROLLING_WINDOW: int = 7
    MIN_DATA_POINTS: int = 30
    MAX_MISSING_RATE: float = 0.95
    OUTLIER_IQR_FACTOR: float = 3.0
    MIN_POPULATION: int = 10_000
    MAX_MORTALITY_RATE: float = 20.0
    CONSISTENCY_CHECKS: bool = True
    REMOVE_NEGATIVES: bool = True
    FIX_MONOTONIC: bool = True
    
    # Configuración de descarga
    DOWNLOAD_TIMEOUT: int = 60
    DOWNLOAD_RETRIES: int = 3
    VERIFY_CHECKSUM: bool = True

    # ETIQUETA: Método especial que se ejecuta después de crear la clase. Carga variables de .env y crea directorios.
    def __post_init__(self):
        """Inicializa paths y carga configuración desde .env"""
        # Configurar directorios
        data_dir_env = env_config("DATA_DIR", default="data")
        self.DATA_DIR = self.BASE_DIR / data_dir_env
        
        logs_dir_env = env_config("LOGS_DIR", default="logs")
        self.LOGS_DIR = self.BASE_DIR / logs_dir_env
        
        # Configurar archivos
        self.INPUT_FILE = self.DATA_DIR / "owid-covid-data.csv"
        self.OUTPUT_RAW = self.DATA_DIR / "raw_covid_data.csv"
        self.OUTPUT_PROCESSED = self.DATA_DIR / "processed_covid.csv"
        self.OUTPUT_COMPRESSED = self.DATA_DIR / "processed_covid.csv.gz"
        self.OUTPUT_REPORT = self.DATA_DIR / "data_quality_report.json"
        self.OUTPUT_SUMMARY = self.BASE_DIR / "pipeline_summary.txt"  # ✅ CORREGIDO: Guardar en raíz
        self.OUTPUT_CONTINENT_METRICS = self.DATA_DIR / "continent_metrics.json"
        self.OUTPUT_REGION_METRICS = self.DATA_DIR / "region_metrics.json"
        self.LOG_FILE = self.LOGS_DIR / "etl_pipeline.log"
        
        # Cargar URLs desde .env (con fallback a defaults)
        urls_str = env_config(
            "DATA_SOURCE_URLS",
            default="https://covid.ourworldindata.org/data/owid-covid-data.csv,"
                    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        )
        self.DATA_URLS = [url.strip() for url in urls_str.split(",")]
        
        # Crear directorios si no existen
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Inicializar configuración y logger
config = Config()
logger = setup_logging(
    config.LOG_FILE, 
    json_logs=env_config("JSON_LOGS", default=False, cast=bool)
)


# ============================================================================
# VALIDADORES DE DATOS
# ============================================================================

# ETIQUETA: Clase que agrupa funciones estáticas para validar la calidad de los datos
class DataValidator:
    """Validadores estáticos para calidad de datos."""

    # ETIQUETA: Función estática para asegurar que no haya valores negativos (los convierte a 0)
    @staticmethod
    def validate_non_negative(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clip valores negativos a cero."""
        for col in columns:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    logger.warning(f"{col}: {int(neg)} valores negativos -> corregidos a 0")
                    df[col] = df[col].clip(lower=0)
        return df

    # ETIQUETA: Función estática para forzar que columnas (ej. total_cases) siempre incrementen
    @staticmethod
    def validate_monotonic_increasing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fuerza monotonicidad en columnas acumulativas."""
        for col in columns:
            if col in df.columns and "location" in df.columns:
                df[col] = df.groupby("location", observed=True)[col].cummax()  # ✅ CORREGIDO
        return df

    # ETIQUETA: Función estática para detectar valores atípicos (outliers) usando el rango intercuartílico (IQR)
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
        """Detecta outliers usando IQR."""
        if series.dropna().empty:
            return pd.Series(False, index=series.index)
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(False, index=series.index)

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return (series < lower_bound) | (series > upper_bound)

    # ETIQUETA: Función estática para calcular la firma digital (hash SHA256) de un archivo
    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """Calcula checksum SHA256 de un archivo."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculando checksum: {e}")
            return ""


# ============================================================================
# CLASE PRINCIPAL ETL
# ============================================================================

# ETIQUETA: Clase principal que define y ejecuta todo el pipeline ETL
class ImprovedCovidETL:
    """Pipeline ETL completo con reintentos y validación mejorada."""

    # ETIQUETA: Constructor de la clase ETL. Inicializa variables.
    def __init__(self, config: Config):
        self.config = config
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.df_continent_metrics: Optional[pd.DataFrame] = None
        self.df_region_metrics: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "raw_shape": None,
            "processed_shape": None,
            "duplicates_removed": 0,
            "outliers_treated": 0,
            "missing_handled": 0,
            "countries_filtered": 0,
            "features_created": 0,
            "checksum": None,
        }

    # ========================================================================
    # EXTRACCIÓN
    # ========================================================================

    # ETIQUETA: Método de la Fase 1 (Extracción). Carga datos desde el archivo CSV.
    def extract_data(self) -> bool:
        """Extrae datos con validación de integridad."""
        logger.info("FASE 1: EXTRACCION DE DATOS")
        
        if not self.config.INPUT_FILE.exists():
            logger.info(f"Archivo no encontrado: {self.config.INPUT_FILE}. Descargando...")
            if not self._download_data():
                logger.error("No se pudo descargar el archivo de datos")
                return False

        try:
            # Validar checksum si está habilitado
            if self.config.VERIFY_CHECKSUM and self.config.INPUT_FILE.exists():
                checksum = DataValidator.calculate_checksum(self.config.INPUT_FILE)
                self.stats["checksum"] = checksum
                logger.info(f"Checksum del archivo: {checksum[:16]}...")

            # Leer CSV
            self.df_raw = pd.read_csv(
                self.config.INPUT_FILE,
                low_memory=False,
                na_values=["", "NA", "N/A", "nan", "NaN", "-", "--", "null"]
            )
            
            if "date" in self.df_raw.columns:
                self.df_raw["date"] = pd.to_datetime(self.df_raw["date"], errors="coerce")

            self.stats["raw_shape"] = self.df_raw.shape
            
            # Guardar copia raw
            self.df_raw.to_csv(self.config.OUTPUT_RAW, index=False)
            
            logger.info(f"Datos crudos cargados: {self.df_raw.shape[0]} filas x {self.df_raw.shape[1]} columnas")
            return True
            
        except Exception as exc:
            logger.exception(f"Error leyendo CSV: {exc}")
            return False

    # ETIQUETA: Método privado para descargar el archivo de datos con reintentos automáticos
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _download_data(self) -> bool:
        """Descarga datos con reintentos automáticos."""
        for url in self.config.DATA_URLS:
            try:
                logger.info(f"Descargando desde: {url}")
                
                response = requests.get(
                    url,
                    timeout=self.config.DOWNLOAD_TIMEOUT,
                    stream=True,
                    headers={"User-Agent": "COVID-ETL-Pipeline/1.0"}
                )
                response.raise_for_status()
                
                # Guardar archivo
                with open(self.config.INPUT_FILE, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
                
                logger.info(f"Descarga exitosa desde: {url}")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Fallo descarga desde {url}: {e}")
                # Limpiar archivo incompleto
                if self.config.INPUT_FILE.exists():
                    try:
                        self.config.INPUT_FILE.unlink()
                    except OSError:
                        pass
                continue
        
        return False

    # ========================================================================
    # TRANSFORMACIÓN
    # ========================================================================

    # ETIQUETA: Método de la Fase 2 (Transformación). Orquesta toda la limpieza de datos.
    def transform_data(self) -> bool:
        """Aplica transformaciones con manejo robusto de errores."""
        logger.info("FASE 2: TRANSFORMACION Y LIMPIEZA")
        
        if self.df_raw is None:
            logger.error("No hay datos crudos para transformar")
            return False

        self.df_processed = self.df_raw.copy()
        
        try:
            # Pipeline de transformación
            self._select_columns()
            self._convert_datatypes()
            self._remove_duplicates()
            self._validate_values()
            self._calculate_aggregate_metrics()
            self._filter_quality()
            self._handle_missing()
            self._handle_outliers()
            self._validate_consistency()
            self._create_features()
            self._final_cleanup()

            self.stats["processed_shape"] = self.df_processed.shape
            logger.info(f"Transformación completada: {self.df_processed.shape[0]} filas x {self.df_processed.shape[1]} columnas")
            return True
            
        except Exception as exc:
            logger.exception(f"Error durante transformación: {exc}")
            return False

    # ETIQUETA: Método privado para seleccionar solo las columnas necesarias
    def _select_columns(self) -> None:
        """Selecciona columnas relevantes."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        available_cols = [c for c in self.config.CORE_COLUMNS if c in self.df_processed.columns]
        cols_to_drop = [
            c for c in available_cols
            if self.df_processed[c].isna().mean() > self.config.MAX_MISSING_RATE
        ]
        selected_cols = [c for c in available_cols if c not in cols_to_drop]
        self.df_processed = self.df_processed[selected_cols]
        
        logger.info(f"Columnas seleccionadas: {len(selected_cols)} (eliminadas por missing > {self.config.MAX_MISSING_RATE*100}%: {len(cols_to_drop)})")

    # ETIQUETA: Método privado para convertir tipos de datos (ej. texto a número, texto a fecha)
    def _convert_datatypes(self) -> None:
        """Convierte tipos de datos."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        # Fecha
        if "date" in self.df_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df_processed['date']):
                self.df_processed["date"] = pd.to_datetime(self.df_processed["date"], errors="coerce")

        # Categóricas - ✅ CORREGIDO
        categorical_cols = [c for c in self.config.CATEGORICAL_COLS if c in self.df_processed.columns]
        for col in categorical_cols:
            if not isinstance(self.df_processed[col].dtype, pd.CategoricalDtype):  # ✅ CORREGIDO
                self.df_processed[col] = self.df_processed[col].astype("category")

        # Numéricas
        numeric_cols = [c for c in self.df_processed.columns if c not in categorical_cols + ["date"]]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors="coerce")
        
        logger.info(f"Tipos convertidos: {len(categorical_cols)} categóricas, {len(numeric_cols)} numéricas")

    # ETIQUETA: Método privado para eliminar filas duplicadas
    def _remove_duplicates(self) -> None:
        """Elimina duplicados."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        before = len(self.df_processed)
        self.df_processed = self.df_processed.drop_duplicates()
        
        if {"location", "date"}.issubset(self.df_processed.columns):
            self.df_processed = self.df_processed.drop_duplicates(subset=["location", "date"], keep="first")

        value_cols = [c for c in self.df_processed.columns if c not in self.config.CATEGORICAL_COLS + ["date"]]
        if value_cols:
            mask_all_nan = self.df_processed[value_cols].isna().all(axis=1)
            self.df_processed = self.df_processed[~mask_all_nan]

        removed = before - len(self.df_processed)
        self.stats["duplicates_removed"] = int(removed)

    # ETIQUETA: Método privado para validar reglas de negocio (ej. no negativos, rangos válidos)
    def _validate_values(self) -> None:
        """Valida valores específicos."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        validator = DataValidator()
        
        if self.config.REMOVE_NEGATIVES:
            self.df_processed = validator.validate_non_negative(
                self.df_processed, 
                self.config.NEGATIVE_COLS_TO_CLIP
            )

        # Validar positive_rate
        if "positive_rate" in self.df_processed.columns:
            if pd.api.types.is_numeric_dtype(self.df_processed["positive_rate"]):
                bad_mask = (self.df_processed["positive_rate"] < 0) | (self.df_processed["positive_rate"] > 1)
                if bad_mask.sum() > 0:
                    logger.warning(f"positive_rate fuera de rango: {int(bad_mask.sum())} valores -> corregidos")
                    self.df_processed.loc[bad_mask, "positive_rate"] = np.nan

        # Validar población mínima
        if "population" in self.df_processed.columns:
            if pd.api.types.is_numeric_dtype(self.df_processed["population"]):
                low_pop_mask = self.df_processed["population"] < self.config.MIN_POPULATION
                if low_pop_mask.sum() > 0:
                    logger.warning(f"Registros con población < {self.config.MIN_POPULATION}: {int(low_pop_mask.sum())} -> eliminando filas")
                    self.df_processed = self.df_processed[~low_pop_mask]

    # ETIQUETA: Método privado para extraer y guardar métricas de agregados (Continentes, Regiones)
    def _calculate_aggregate_metrics(self) -> None:
        """Calcula métricas de continentes y regiones."""
        logger.info("Calculando métricas de agregados (Regiones y Continentes)...")
        
        if not isinstance(self.df_processed, pd.DataFrame):
            return

        aggregate_df = self.df_processed[
            self.df_processed["location"].isin(self.config.AGGREGATES_TO_EXCLUDE)
        ].copy()

        if aggregate_df.empty:
            logger.warning("No se encontraron datos agregados")
            return

        if not pd.api.types.is_datetime64_any_dtype(aggregate_df['date']):
            aggregate_df['date'] = pd.to_datetime(aggregate_df['date'], errors='coerce')
            aggregate_df = aggregate_df.dropna(subset=['date'])

        if aggregate_df.empty:
            return

        latest_aggregates = aggregate_df.sort_values("date").drop_duplicates(
            subset=["location"], 
            keep="last"
        )

        metric_cols = ["location", "date", "total_cases", "total_deaths", 
                       "people_fully_vaccinated", "population"]
        final_cols = [col for col in metric_cols if col in latest_aggregates.columns]
        latest_aggregates = latest_aggregates[final_cols]

        self.df_continent_metrics = latest_aggregates[
            latest_aggregates["location"].isin(self.config.CONTINENT_NAMES)
        ].set_index("location")

        self.df_region_metrics = latest_aggregates[
            latest_aggregates["location"].isin(self.config.REGION_NAMES)
        ].set_index("location")

        logger.info(f"Métricas de Continentes calculadas: {len(self.df_continent_metrics)} filas")
        logger.info(f"Métricas de Regiones calculadas: {len(self.df_region_metrics)} filas")

    # ETIQUETA: Método privado para filtrar datos (eliminar agregados y países con pocos datos)
    def _filter_quality(self) -> None:
        """Filtra por calidad de datos."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return

        initial_locations = self.df_processed["location"].nunique()
        
        self.df_processed = self.df_processed[
            ~self.df_processed["location"].isin(self.config.AGGREGATES_TO_EXCLUDE)
        ]

        counts = self.df_processed.groupby("location", observed=True).size()  # ✅ CORREGIDO
        valid_locations = counts[counts >= self.config.MIN_DATA_POINTS].index
        self.df_processed = self.df_processed[self.df_processed["location"].isin(valid_locations)]

        final_locations = self.df_processed["location"].nunique()
        self.stats["countries_filtered"] = int(initial_locations - final_locations)
        
        logger.info(f"Ubicaciones (países) filtradas (agregados o < {self.config.MIN_DATA_POINTS} puntos): {initial_locations} -> {final_locations}")

    # ETIQUETA: Método privado para manejar (rellenar o interpolar) valores faltantes (NaN)
    def _handle_missing(self) -> None:
        """Maneja valores faltantes."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        before_missing = int(self.df_processed.isna().sum().sum())

        # Forward fill acumulativos - ✅ CORREGIDO
        for col in self.config.CUMULATIVE_COLS_TO_FFILL:
            if col in self.df_processed.columns:
                self.df_processed[col] = self.df_processed.groupby("location", observed=True)[col].ffill()

        # Fill demográficos con mediana - ✅ CORREGIDO
        for col in self.config.DEMOGRAPHIC_COLS_TO_FILL:
            if col in self.df_processed.columns:
                medians = self.df_processed.groupby("location", observed=True)[col].transform('median')
                self.df_processed[col] = self.df_processed[col].fillna(medians)

        # Interpolar diarios - ✅ CORREGIDO
        for col in self.config.DAILY_COLS_TO_INTERPOLATE:
            if col in self.df_processed.columns and pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed[col] = (
                    self.df_processed.groupby("location", observed=True)[col]
                    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
                    .fillna(0)
                    .clip(lower=0)
                )

        after_missing = int(self.df_processed.isna().sum().sum())
        self.stats["missing_handled"] = before_missing - after_missing
        logger.info(f"Valores faltantes manejados: {self.stats['missing_handled']} (restantes: {after_missing})")

    # ETIQUETA: Método privado para detectar y tratar valores atípicos (outliers)
    def _handle_outliers(self) -> None:
        """Detecta y trata outliers."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        validator = DataValidator()
        total_outliers = 0

        for col in self.config.OUTLIER_COLS:
            if col not in self.df_processed.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df_processed[col]):
                continue

            # ETIQUETA: Función interna (anidada) para aplicar la detección de outliers por grupo
            def detect_group_outliers(series: pd.Series) -> pd.Series:
                if series.dropna().shape[0] < 10:
                    return pd.Series(False, index=series.index)
                return validator.detect_outliers_iqr(series, factor=self.config.OUTLIER_IQR_FACTOR)

            # ✅ CORREGIDO
            outlier_mask = self.df_processed.groupby("location", observed=True)[col].transform(detect_group_outliers)
            outlier_mask = outlier_mask.fillna(False).astype(bool)
            col_outliers = int(outlier_mask.sum())

            if col_outliers > 0:
                self.df_processed.loc[outlier_mask, col] = np.nan
                self.df_processed[col] = self.df_processed.groupby("location", observed=True)[col].transform(
                    lambda s: s.interpolate(method="linear", limit=3).fillna(0)
                )
                total_outliers += col_outliers
                logger.info(f"Outliers detectados y tratados en '{col}': {col_outliers}")

        self.stats["outliers_treated"] = total_outliers

    # ETIQUETA: Método privado para validar la consistencia lógica entre columnas (ej. new_cases vs total_cases)
    def _validate_consistency(self) -> None:
        """Valida consistencia lógica."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return

        # Recalcular new_* desde total_* - ✅ CORREGIDO
        for prefix in ("cases", "deaths"):
            total_col, new_col = f"total_{prefix}", f"new_{prefix}"
            if {total_col, new_col}.issubset(self.df_processed.columns):
                calculated_new = self.df_processed.groupby("location", observed=True)[total_col].diff().fillna(0).clip(lower=0)
                self.df_processed[new_col] = calculated_new.where(
                    calculated_new > 0, 
                    self.df_processed[new_col].fillna(0)
                )

        # Forzar monotonicidad
        if self.config.FIX_MONOTONIC:
            self.df_processed = DataValidator.validate_monotonic_increasing(
                self.df_processed, 
                self.config.MONOTONIC_COLS
            )

        # Validar fully_vaccinated <= people_vaccinated
        if {"people_fully_vaccinated", "people_vaccinated"}.issubset(self.df_processed.columns):
            mask = self.df_processed["people_fully_vaccinated"] > self.df_processed["people_vaccinated"]
            if mask.sum() > 0:
                logger.warning(f"people_fully_vaccinated > people_vaccinated: {int(mask.sum())} registros -> ajustando fully_vaccinated al valor de people_vaccinated")
                self.df_processed.loc[mask, "people_fully_vaccinated"] = self.df_processed.loc[mask, "people_vaccinated"]

    # ETIQUETA: Método privado para crear nuevas columnas (features) para el análisis
    def _create_features(self) -> None:
        """Crea features derivados."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return
        
        new_features = []

        # Mortality rate
        if {"total_deaths", "total_cases"}.issubset(self.df_processed.columns):
            valid_mask = self.df_processed["total_cases"] > 0
            rate = (self.df_processed.loc[valid_mask, "total_deaths"] / 
                   self.df_processed.loc[valid_mask, "total_cases"]) * 100
            self.df_processed["mortality_rate"] = rate.reindex(self.df_processed.index)
            self.df_processed["mortality_rate"] = self.df_processed["mortality_rate"].clip(
                upper=self.config.MAX_MORTALITY_RATE
            )
            new_features.append("mortality_rate")

        # Per 100k metrics - ✅ CORREGIDO
        if "population" in self.df_processed.columns:
            valid_pop = self.df_processed["population"] > 0
            for metric in ("total_cases", "total_deaths"):
                if metric in self.df_processed.columns:
                    colname = f"{metric}_per_100k"
                    per_capita = (self.df_processed.loc[valid_pop, metric] / 
                                  self.df_processed.loc[valid_pop, "population"]) * 100_000
                    self.df_processed[colname] = per_capita.reindex(self.df_processed.index)
                    # ✅ CORREGIDO: Usar método correcto
                    self.df_processed[colname] = self.df_processed[colname].replace([np.inf, -np.inf], np.nan)
                    new_features.append(colname)

        # Smoothed metrics - ✅ CORREGIDO
        for col in self.config.DAILY_COLS_TO_INTERPOLATE:
            if col in self.df_processed.columns:
                smoothed_col = f"{col}_smoothed"
                self.df_processed[smoothed_col] = self.df_processed.groupby("location", observed=True)[col].transform(
                    lambda s: s.rolling(window=self.config.ROLLING_WINDOW, min_periods=1).mean()
                )
                new_features.append(smoothed_col)

        # Vaccination rate
        if {"people_fully_vaccinated", "population"}.issubset(self.df_processed.columns):
            valid_mask = self.df_processed["population"] > 0
            rate = (self.df_processed.loc[valid_mask, "people_fully_vaccinated"] / 
                   self.df_processed.loc[valid_mask, "population"]) * 100
            self.df_processed["vaccination_rate"] = rate.reindex(self.df_processed.index)
            self.df_processed["vaccination_rate"] = self.df_processed["vaccination_rate"].clip(upper=100)
            new_features.append("vaccination_rate")

        self.stats["features_created"] = len(new_features)
        logger.info(f"Features creados: {len(new_features)} ({', '.join(new_features)})")

    # ETIQUETA: Método privado para la limpieza final (ordenar, eliminar NaN/Infinitos)
    def _final_cleanup(self) -> None:
        """Limpieza final."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return

        # Eliminar duplicados finales
        if {"location", "date"}.issubset(self.df_processed.columns):
            self.df_processed = self.df_processed.drop_duplicates(subset=["location", "date"], keep='first')
            self.df_processed = self.df_processed[self.df_processed["date"].notna()]
            self.df_processed = self.df_processed.sort_values(["location", "date"]).reset_index(drop=True)

        # Reemplazar infinitos
        self.df_processed = self.df_processed.replace([np.inf, -np.inf], np.nan)  # ✅ CORREGIDO

        # Eliminar columnas todas NaN
        all_nan_cols = self.df_processed.columns[self.df_processed.isna().all()].tolist()
        if all_nan_cols:
            self.df_processed = self.df_processed.drop(columns=all_nan_cols)

    # ========================================================================
    # CARGA - ✅ COMPLETAMENTE CORREGIDO
    # ========================================================================

    # ETIQUETA: Método de la Fase 3 (Carga). Guarda los datos procesados en archivos.
    def load_data(self) -> bool:
        """Guarda datos procesados con manejo robusto de errores."""
        logger.info("FASE 3: CARGA Y ALMACENAMIENTO")
        
        if not isinstance(self.df_processed, pd.DataFrame) or self.df_processed.empty:
            logger.error("No hay datos procesados para guardar")
            return False

        try:
            # Backup de archivos existentes
            self._backup_existing_files()

            # Guardar archivo principal procesado
            logger.info("Guardando archivo principal procesado...")
            self._save_csv_safely(self.df_processed, self.config.OUTPUT_PROCESSED)
            
            # Guardar comprimido
            logger.info("Guardando archivo comprimido...")
            self._save_csv_safely(self.df_processed, self.config.OUTPUT_COMPRESSED, compression='gzip')

            # Guardar por continente
            if "continent" in self.df_processed.columns:
                logger.info("Separando datos por continente...")
                for continent in self.df_processed['continent'].dropna().unique():
                    continent_safe = str(continent).replace(' ', '_').replace('/', '_')
                    continent_file = self.config.DATA_DIR / f"processed_covid_{continent_safe}.csv.gz"
                    df_continent = self.df_processed[self.df_processed['continent'] == continent]
                    if not df_continent.empty:
                        self._save_csv_safely(df_continent, continent_file, compression='gzip')

            # Guardar métricas agregadas
            if isinstance(self.df_continent_metrics, pd.DataFrame) and not self.df_continent_metrics.empty:
                logger.info("Guardando métricas de continentes...")
                self._save_json_safely(
                    self.df_continent_metrics.to_dict(orient="index"),
                    self.config.OUTPUT_CONTINENT_METRICS
                )

            if isinstance(self.df_region_metrics, pd.DataFrame) and not self.df_region_metrics.empty:
                logger.info("Guardando métricas de regiones...")
                self._save_json_safely(
                    self.df_region_metrics.to_dict(orient="index"),
                    self.config.OUTPUT_REGION_METRICS
                )

            # Guardar reporte
            logger.info("Guardando reporte de calidad...")
            self._save_json_safely(self.stats, self.config.OUTPUT_REPORT)

            # Log tamaños
            if self.config.OUTPUT_COMPRESSED.exists():
                size_gz = self.config.OUTPUT_COMPRESSED.stat().st_size / (1024 ** 2)
                logger.info(f"Archivo comprimido guardado: {size_gz:.2f} MB")

            logger.info("Todos los archivos guardados exitosamente")
            return True

        except Exception as exc:
            logger.exception(f"Error guardando archivos: {exc}")
            return False

    # ETIQUETA: Método privado para guardar un DataFrame en CSV de forma segura (evita errores de Windows)
    def _save_csv_safely(self, df: pd.DataFrame, filepath: Path, compression: Optional[str] = None) -> None:
        """Guarda CSV con manejo robusto de errores en Windows."""
        try:
            # Asegurar que el directorio existe
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Convertir Path a string para evitar problemas en Windows
            filepath_str = str(filepath.resolve())
            
            # Parámetros de guardado seguros
            save_params = {
                'index': False,
                'encoding': 'utf-8',
                'date_format': '%Y-%m-%d',
                'na_rep': ''
            }
            
            if compression:
                save_params['compression'] = compression
            
            # Intentar guardar
            df.to_csv(filepath_str, **save_params)
            
        except OSError as e:
            # Manejo específico de errores de Windows
            if "Invalid argument" in str(e) or e.errno == 22:
                logger.error(f"Error de Windows al guardar {filepath.name}")
                logger.error("Posibles causas:")
                logger.error("  1. El archivo está abierto en Excel u otro programa")
                logger.error("  2. Problemas de permisos en el directorio")
                logger.error("  3. Nombre de archivo inválido")
                logger.error("\nIntentando método alternativo...")
                
                # Método alternativo: guardar con nombre temporal y renombrar
                temp_file = filepath.with_suffix('.tmp')
                try:
                    df.to_csv(str(temp_file.resolve()), **save_params)
                    if filepath.exists():
                        filepath.unlink()
                    temp_file.rename(filepath)
                    logger.info(f"Guardado exitoso usando método alternativo: {filepath.name}")
                except Exception as e2:
                    logger.error(f"Método alternativo también falló: {e2}")
                    raise
            else:
                raise

    # ETIQUETA: Método privado para guardar datos en JSON de forma segura (maneja tipos de datos de Pandas/Numpy)
    def _save_json_safely(self, data: dict, filepath: Path) -> None:
        """Guarda JSON con manejo robusto de errores."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # ETIQUETA: Función interna para convertir tipos de datos no serializables a JSON
            def convert_for_json(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            # ETIQUETA: Función interna recursiva para serializar diccionarios
            def serialize_dict(d):
                if isinstance(d, dict):
                    return {k: serialize_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [serialize_dict(item) for item in d]
                else:
                    return convert_for_json(d)
            
            serialized_data = serialize_dict(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error guardando JSON {filepath.name}: {e}")
            raise

    # ETIQUETA: Método privado para crear una copia de seguridad de los archivos de salida antiguos
    def _backup_existing_files(self) -> None:
        """Crea backup de archivos existentes."""
        files_to_backup = [
            self.config.OUTPUT_PROCESSED,
            self.config.OUTPUT_COMPRESSED,
            self.config.OUTPUT_CONTINENT_METRICS,
            self.config.OUTPUT_REGION_METRICS,
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config.DATA_DIR / "backups" / timestamp
        
        any_backed_up = False
        for file_path in files_to_backup:
            if file_path.exists():
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / file_path.name
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    any_backed_up = True
                except Exception as e:
                    logger.warning(f"No se pudo hacer backup de {file_path.name}: {e}")
        
        if any_backed_up:
            logger.info(f"Backups creados en: {backup_dir}")

    # ETIQUETA: Método para imprimir un reporte de calidad de datos en el log
    def generate_report(self) -> None:
        """Genera reporte de calidad."""
        if not isinstance(self.df_processed, pd.DataFrame):
            return

        logger.info("=" * 60)
        logger.info("REPORTE DE CALIDAD DE DATOS")
        logger.info("=" * 60)
        
        rows, cols = self.df_processed.shape
        unique_countries = int(self.df_processed["location"].nunique()) if "location" in self.df_processed.columns else 0
        missing = int(self.df_processed.isna().sum().sum())
        total_cells = rows * cols
        completeness = ((total_cells - missing) / total_cells) * 100 if total_cells > 0 else 0

        logger.info(f"Dimensiones finales: {rows:,} filas x {cols} columnas")
        logger.info(f"Países únicos: {unique_countries}")
        logger.info(f"Duplicados removidos: {self.stats.get('duplicates_removed', 0):,}")
        logger.info(f"Outliers tratados: {self.stats.get('outliers_treated', 0):,}")
        logger.info(f"Valores faltantes manejados: {self.stats.get('missing_handled', 0):,}")
        logger.info(f"Features creados: {self.stats.get('features_created', 0)}")
        logger.info(f"Completitud de datos: {completeness:.2f}%")
        logger.info("=" * 60)

    # ETIQUETA: Método principal que ejecuta todo el pipeline (Extract, Transform, Load)
    def run(self) -> bool:
        """Ejecuta pipeline completo."""
        self.stats["start_time"] = datetime.now()
        success = False

        try:
            logger.info("=" * 70)
            logger.info("INICIANDO PIPELINE COVID-19 ETL")
            logger.info(f"Directorio base: {self.config.BASE_DIR}")
            logger.info(f"Directorio de datos: {self.config.DATA_DIR}")
            logger.info("=" * 70)
            
            # Ejecutar fases
            if not self.extract_data():
                raise Exception("Fase de extracción falló")
            
            if not self.transform_data():
                raise Exception("Fase de transformación falló")
            
            if not self.load_data():
                raise Exception("Fase de carga falló")
            
            success = True

        except KeyboardInterrupt:
            logger.warning("Pipeline interrumpido por el usuario")
            success = False
        except Exception as e:
            logger.exception(f"Pipeline falló: {e}")
            success = False
        finally:
            self.stats["end_time"] = datetime.now()
            elapsed = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            if success:
                self.generate_report()
                logger.info("=" * 70)
                logger.info(f"PIPELINE COMPLETADO EXITOSAMENTE en {elapsed:.2f} segundos")
                logger.info("=" * 70)
                self._save_summary(elapsed, success=True)
            else:
                logger.error("=" * 70)
                logger.error(f"PIPELINE FALLÓ después de {elapsed:.2f} segundos")
                logger.error("Revisa el log para más detalles")
                logger.error("=" * 70)
                self._save_summary(elapsed, success=False)

        return success

    # ETIQUETA: Método privado para guardar un resumen de la ejecución en un archivo de texto
    def _save_summary(self, elapsed: float, success: bool = True) -> None:
        """Guarda resumen de ejecución."""
        try:
            summary_lines = []
            summary_lines.append("=" * 70)
            summary_lines.append("RESUMEN DE EJECUCIÓN - PIPELINE COVID-19 ETL")
            summary_lines.append("=" * 70)
            summary_lines.append(f"Estado: {'✅ EXITOSO' if success else '❌ FALLIDO'}")
            summary_lines.append(f"Inicio: {self.stats.get('start_time', 'N/A')}")
            summary_lines.append(f"Fin: {self.stats.get('end_time', 'N/A')}")
            summary_lines.append(f"Duración: {elapsed:.2f} segundos")
            summary_lines.append("")
            summary_lines.append("--- ESTADÍSTICAS ---")
            
            raw_shape = self.stats.get('raw_shape', (None, None))
            proc_shape = self.stats.get('processed_shape', (None, None))
            
            if raw_shape[0]:
                summary_lines.append(f"Datos crudos: {raw_shape[0]:,} filas x {raw_shape[1]} columnas")
            if proc_shape[0]:
                summary_lines.append(f"Datos procesados: {proc_shape[0]:,} filas x {proc_shape[1]} columnas")
            
            summary_lines.append(f"Duplicados removidos: {self.stats.get('duplicates_removed', 0):,}")
            summary_lines.append(f"Outliers tratados: {self.stats.get('outliers_treated', 0):,}")
            summary_lines.append(f"Valores faltantes manejados: {self.stats.get('missing_handled', 0):,}")
            summary_lines.append(f"Features creados: {self.stats.get('features_created', 0)}")
            
            if self.stats.get('checksum'):
                summary_lines.append(f"Checksum: {self.stats['checksum']}")
            
            summary_lines.append("=" * 70)
            
            # Guardar en archivo
            with open(self.config.OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info(f"Resumen de ejecución guardado en {self.config.OUTPUT_SUMMARY}")

        except Exception as e:
            logger.error(f"No se pudo guardar resumen: {e}")


# ============================================================================
# MAIN
# ============================================================================

# ETIQUETA: Función principal que se ejecuta al iniciar el script
def main() -> None:
    """Función principal."""
    try:
        # Mostrar configuración
        logger.info("Cargando configuración desde variables de entorno...")
        logger.info(f"Environment: {env_config('ENVIRONMENT', default='development')}")
        logger.info(f"Directorio de datos: {config.DATA_DIR}")
        logger.info(f"Log level: {env_config('LOG_LEVEL', default='INFO')}")
        
        # Ejecutar pipeline
        pipeline = ImprovedCovidETL(config)
        success = pipeline.run()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Error fatal: {e}")
        sys.exit(1)


# ETIQUETA: Punto de entrada del script. Llama a la función main() si el archivo se ejecuta directamente.
if __name__ == "__main__":
    main()
# %%