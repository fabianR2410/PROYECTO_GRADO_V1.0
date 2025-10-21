#%%
from __future__ import annotations # Keep this for future compatibility

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

# -----------------------------
# Configuración de logging
# -----------------------------
class WindowsFormatter(logging.Formatter):
    """
    A custom logging formatter that replaces emojis with text representations
    for compatibility with Windows terminals that might not display emojis correctly.
    """
    EMOJI_MAP: Dict[str, str] = {
        "🚀": "[START]", "✅": "[OK]", "❌": "[ERROR]", "⚠️": "[WARN]",
        "📊": "[INFO]", "📥": "[DOWNLOAD]"
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing emojis."""
        text = super().format(record)
        for emoji, replacement in self.EMOJI_MAP.items():
            text = text.replace(emoji, replacement)
        return text

def setup_logging(log_file: Path) -> logging.Logger:
    """
    Sets up and configures the logger for the ETL pipeline.

    Configures a logger named 'covid_etl' with both console and file handlers.
    Ensures the log directory exists. Uses WindowsFormatter for emoji compatibility.

    Args:
        log_file: The Path object representing the log file destination.

    Returns:
        The configured Logger instance.
    """
    logger = logging.getLogger("covid_etl")
    if logger.hasHandlers(): # Avoid adding multiple handlers if called again
        return logger

    logger.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    console = logging.StreamHandler()
    console.setFormatter(WindowsFormatter(fmt))

    # Ensure the log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fileh = logging.FileHandler(log_file, encoding="utf-8")
    fileh.setFormatter(WindowsFormatter(fmt))

    logger.addHandler(console)
    logger.addHandler(fileh)
    return logger

# -----------------------------
# Configuración del pipeline
# -----------------------------
@dataclass
class Config:
    """
    Holds configuration parameters for the ETL pipeline, including file paths,
    URLs, column lists, and processing parameters. Uses pathlib for path management.
    """
    # --- Paths ---
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Data files
    INPUT_FILE: Path = DATA_DIR / "owid-covid-data.csv"
    OUTPUT_RAW: Path = DATA_DIR / "raw_covid_data.csv"
    OUTPUT_PROCESSED: Path = DATA_DIR / "processed_covid.csv"
    OUTPUT_COMPRESSED: Path = DATA_DIR / "processed_covid.csv.gz"
    OUTPUT_REPORT: Path = DATA_DIR / "data_quality_report.json"
    OUTPUT_SUMMARY: Path = DATA_DIR / "pipeline_summary.txt"
    OUTPUT_CONTINENT_METRICS: Path = DATA_DIR / "continent_metrics.json"
    OUTPUT_REGION_METRICS: Path = DATA_DIR / "region_metrics.json"
    LOG_FILE: Path = LOGS_DIR / "etl_pipeline.log"

    # Data Source URLs
    DATA_URLS: List[str] = field(default_factory=lambda: [
        "https://covid.ourworldindata.org/data/owid-covid-data.csv",
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
    ])

    # --- Column Lists ---
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
        "Oceania", "Antarctica" # Include Antarctica if present
    ])
    CONTINENT_NAMES: List[str] = field(default_factory=lambda: [
        "Africa", "Asia", "Europe", "North America", "South America", "Oceania", "Antarctica"
    ])
    REGION_NAMES: List[str] = field(default_factory=lambda: [
        "World", "International", "High income", "Low income",
        "Lower middle income", "Upper middle income", "European Union"
    ])

    # --- Quality/Transformation Parameters ---
    ROLLING_WINDOW: int = 7
    MIN_DATA_POINTS: int = 30
    MAX_MISSING_RATE: float = 0.95
    OUTLIER_IQR_FACTOR: float = 3.0
    MIN_POPULATION: int = 10_000
    MAX_MORTALITY_RATE: float = 20.0
    CONSISTENCY_CHECKS: bool = True
    REMOVE_NEGATIVES: bool = True
    FIX_MONOTONIC: bool = True

# -----------------------------
# Initialize Config and Logger
# -----------------------------
config = Config()
logger = setup_logging(config.LOG_FILE)


# -----------------------------
# Data Validation Utilities
# -----------------------------
class DataValidator:
    """Provides static methods for validating common data quality issues."""

    @staticmethod
    def validate_non_negative(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Clips negative values in specified columns to zero.

        Args:
            df: The DataFrame to validate.
            columns: A list of column names to check for negative values.

        Returns:
            The DataFrame with negative values clipped to 0.
        """
        for col in columns:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0: # Check if neg is greater than 0 before logging
                    logger.warning("%s: %d valores negativos -> corregidos a 0", col, int(neg))
                    df[col] = df[col].clip(lower=0)
        return df

    @staticmethod
    def validate_monotonic_increasing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Ensures specified columns are monotonically increasing within each 'location' group.

        Uses cummax() grouped by 'location'.

        Args:
            df: The DataFrame to validate.
            columns: A list of column names expected to be monotonically increasing.

        Returns:
            The DataFrame with monotonicity enforced.
        """
        for col in columns:
            if col in df.columns and "location" in df.columns: # Check location exists
                df[col] = df.groupby("location")[col].cummax()
        return df

    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
        """
        Detects outliers in a Series using the Interquartile Range (IQR) method.

        Args:
            series: The pandas Series to check for outliers.
            factor: The multiplication factor for the IQR to determine outlier bounds.

        Returns:
            A boolean Series indicating which values are outliers (True) or not (False).
        """
        if series.dropna().empty: # Handle empty series after dropping NaNs
             return pd.Series(False, index=series.index)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return (series < lower_bound) | (series > upper_bound)

# -----------------------------
# Main ETL Pipeline Class
# -----------------------------
class ImprovedCovidETL:
    """
    Orchestrates the COVID-19 data ETL process: Extract, Transform, Load.

    Attributes:
        config: A Config object holding pipeline settings.
        df_raw: DataFrame holding the raw data after extraction.
        df_processed: DataFrame holding the data after transformation.
        df_continent_metrics: DataFrame holding latest metrics for continents.
        df_region_metrics: DataFrame holding latest metrics for regions/groups.
        stats: A dictionary to store pipeline execution statistics.
    """
    def __init__(self, config: Config):
        """Initializes the ETL pipeline with the given configuration."""
        self.config: Config = config
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.df_continent_metrics: Optional[pd.DataFrame] = None
        self.df_region_metrics: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {
            "start_time": None, "end_time": None, "raw_shape": None,
            "processed_shape": None, "duplicates_removed": 0,
            "outliers_treated": 0, "missing_handled": 0,
            "countries_filtered": 0, "features_created": 0,
        }
        # Ensure data directories exist
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Extraction ----------
    def extract_data(self) -> bool:
        """
        Extracts data from the source CSV file. Downloads if not found locally.

        Reads the CSV specified in config.INPUT_FILE, handling potential download
        and read errors. Converts the 'date' column to datetime. Saves a raw copy.

        Returns:
            True if extraction was successful, False otherwise.
        """
        logger.info("FASE 1: EXTRACCION DE DATOS")
        if not self.config.INPUT_FILE.exists():
            logger.info("Archivo no encontrado localmente: %s. Intentando descargar...", self.config.INPUT_FILE)
            if not self._download_data():
                logger.error("No se pudo obtener el archivo de ninguna fuente.")
                return False

        try:
            self.df_raw = pd.read_csv(
                self.config.INPUT_FILE,
                low_memory=False,
                na_values=["", "NA", "N/A", "nan", "NaN", "-", "--", "null"]
            )
            if "date" in self.df_raw.columns:
                # Specify format for potential speedup if known, otherwise let pandas infer
                self.df_raw["date"] = pd.to_datetime(self.df_raw["date"], errors="coerce")

            self.stats["raw_shape"] = self.df_raw.shape
            self.df_raw.to_csv(self.config.OUTPUT_RAW, index=False) # Save raw copy
            logger.info("Datos crudos cargados: %d filas x %d columnas", *self.df_raw.shape)
            return True
        except FileNotFoundError:
            logger.error("Archivo CSV no encontrado después del intento de descarga: %s", self.config.INPUT_FILE)
            return False
        except pd.errors.EmptyDataError:
            logger.error("El archivo CSV está vacío: %s", self.config.INPUT_FILE)
            return False
        except Exception as exc:
            logger.exception("Error inesperado leyendo el CSV: %s", exc)
            return False

    def _download_data(self) -> bool:
        """
        Attempts to download the data file from URLs specified in the config.

        Iterates through config.DATA_URLS and tries to download the file,
        saving it to config.INPUT_FILE upon success.

        Returns:
            True if download was successful from any URL, False otherwise.
        """
        for url in self.config.DATA_URLS:
            try:
                logger.info("Descargando: %s", url)
                resp = requests.get(url, timeout=60, stream=True) # Use stream for large files
                resp.raise_for_status()
                with open(self.config.INPUT_FILE, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)
                logger.info("Descarga exitosa desde: %s", url)
                return True
            except requests.exceptions.RequestException as e:
                logger.warning("Fallo descarga desde %s: %s", url, e)
                # Clean up potentially incomplete file
                if self.config.INPUT_FILE.exists():
                    try:
                        os.remove(self.config.INPUT_FILE)
                    except OSError:
                        logger.warning("No se pudo eliminar el archivo incompleto: %s", self.config.INPUT_FILE)
                continue # Try next URL
        return False

    # ---------- Transformation ----------
    def transform_data(self) -> bool:
        """
        Applies a series of transformations and cleaning steps to the raw data.

        Executes private methods for column selection, type conversion, duplicate removal,
        value validation, quality filtering, missing data handling, outlier treatment,
        consistency checks, feature creation, and final cleanup. Also calculates
        aggregate metrics.

        Returns:
            True if all transformation steps were successful, False otherwise.
        """
        logger.info("FASE 2: TRANSFORMACION Y LIMPIEZA")
        if self.df_raw is None:
            logger.error("No hay datos crudos para transformar. Ejecuta extract_data() primero.")
            return False

        self.df_processed = self.df_raw.copy()
        try:
            self._select_columns()
            self._convert_datatypes()
            self._remove_duplicates()
            self._validate_values()
            self._calculate_aggregate_metrics() # Calculate before filtering them out
            self._filter_quality()
            self._handle_missing()
            self._handle_outliers()
            self._validate_consistency()
            self._create_features()
            self._final_cleanup()

            # Final check if df_processed exists and is DataFrame
            if not isinstance(self.df_processed, pd.DataFrame):
                logger.error("df_processed no es un DataFrame después de la transformación.")
                return False

            self.stats["processed_shape"] = self.df_processed.shape
            logger.info("Transformación completada: %d filas x %d columnas", *self.df_processed.shape)
            return True
        except Exception as exc:
            logger.exception("Error inesperado durante la transformación de datos: %s", exc)
            return False

    def _calculate_aggregate_metrics(self) -> None:
        """
        Calculates the latest metrics for continents and regions/groups.

        Filters rows corresponding to continents and regions, selects the most
        recent entry for each, and stores them in dedicated DataFrames.
        Should be run before _filter_quality removes these aggregate rows.
        """
        logger.info("Calculando métricas de agregados (Regiones y Continentes)...")
        if not isinstance(self.df_processed, pd.DataFrame) or \
           "location" not in self.df_processed.columns or \
           "date" not in self.df_processed.columns:
            logger.warning("No se pueden calcular métricas de agregados, DataFrame no válido o faltan columnas.")
            return

        aggregate_df = self.df_processed[
            self.df_processed["location"].isin(self.config.AGGREGATES_TO_EXCLUDE)
        ].copy()

        if aggregate_df.empty:
            logger.warning("No se encontraron datos de agregados para procesar.")
            return

        # Ensure 'date' is datetime before sorting
        if not pd.api.types.is_datetime64_any_dtype(aggregate_df['date']):
            aggregate_df['date'] = pd.to_datetime(aggregate_df['date'], errors='coerce')
            aggregate_df = aggregate_df.dropna(subset=['date']) # Remove rows where date conversion failed

        if aggregate_df.empty:
            logger.warning("No se encontraron datos de agregados válidos después de la conversión de fecha.")
            return

        latest_aggregates = aggregate_df.sort_values("date").drop_duplicates(subset=["location"], keep="last")

        metric_cols: List[str] = ["location", "date", "total_cases", "total_deaths", "people_fully_vaccinated", "population"]
        final_metric_cols: List[str] = [col for col in metric_cols if col in latest_aggregates.columns]
        latest_aggregates = latest_aggregates[final_metric_cols]

        self.df_continent_metrics = latest_aggregates[
            latest_aggregates["location"].isin(self.config.CONTINENT_NAMES)
        ].set_index("location")

        self.df_region_metrics = latest_aggregates[
            latest_aggregates["location"].isin(self.config.REGION_NAMES)
        ].set_index("location")

        logger.info("Métricas de Continentes calculadas: %d filas", len(self.df_continent_metrics))
        logger.info("Métricas de Regiones calculadas: %d filas", len(self.df_region_metrics))

    def _select_columns(self) -> None:
        """Selects core columns and drops those with excessive missing values."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        available_cols = [c for c in self.config.CORE_COLUMNS if c in self.df_processed.columns]
        cols_to_drop = [
            c for c in available_cols
            if self.df_processed[c].isna().mean() > self.config.MAX_MISSING_RATE
        ]
        selected_cols = [c for c in available_cols if c not in cols_to_drop]
        self.df_processed = self.df_processed[selected_cols]
        logger.info("Columnas seleccionadas: %d (eliminadas por missing > %.1f%%: %d)",
                    len(selected_cols), self.config.MAX_MISSING_RATE * 100, len(cols_to_drop))

    def _convert_datatypes(self) -> None:
        """Converts columns to appropriate data types (datetime, category, numeric)."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        if "date" in self.df_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df_processed['date']):
                self.df_processed["date"] = pd.to_datetime(self.df_processed["date"], errors="coerce")

        categorical_cols = [c for c in self.config.CATEGORICAL_COLS if c in self.df_processed.columns]
        for col in categorical_cols:
            if not pd.api.types.is_categorical_dtype(self.df_processed[col]):
                self.df_processed[col] = self.df_processed[col].astype("category")

        numeric_cols = [c for c in self.df_processed.columns if c not in categorical_cols + ["date"]]
        for col in numeric_cols:
            # Check if not already numeric to avoid unnecessary conversion
            if not pd.api.types.is_numeric_dtype(self.df_processed[col]):
                self.df_processed[col] = pd.to_numeric(self.df_processed[col], errors="coerce")
        logger.info("Tipos convertidos: %d categóricas, %d numéricas", len(categorical_cols), len(numeric_cols))

    def _remove_duplicates(self) -> None:
        """Removes duplicate rows and rows where all numeric values are NaN."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        before = len(self.df_processed)
        # Remove exact duplicates first
        self.df_processed = self.df_processed.drop_duplicates()

        # Remove duplicates based on location and date, keeping the first
        if {"location", "date"}.issubset(self.df_processed.columns):
            self.df_processed = self.df_processed.drop_duplicates(subset=["location", "date"], keep="first")

        # Remove rows where all non-key columns are NaN
        value_cols = [c for c in self.df_processed.columns if c not in self.config.CATEGORICAL_COLS + ["date"]]
        if value_cols: # Only proceed if there are value columns
            mask_all_nan = self.df_processed[value_cols].isna().all(axis=1)
            self.df_processed = self.df_processed[~mask_all_nan]

        removed = before - len(self.df_processed)
        self.stats["duplicates_removed"] = int(removed)
        if removed > 0:
            logger.info("Duplicados/filas vacías removidas: %d", removed)

    def _validate_values(self) -> None:
        """Validates specific column values (negatives, ranges, minimum population)."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        validator = DataValidator()
        if self.config.REMOVE_NEGATIVES:
            self.df_processed = validator.validate_non_negative(self.df_processed, self.config.NEGATIVE_COLS_TO_CLIP)

        if "positive_rate" in self.df_processed.columns:
            # Ensure it's numeric before comparison
            if pd.api.types.is_numeric_dtype(self.df_processed["positive_rate"]):
                bad_mask = (self.df_processed["positive_rate"] < 0) | (self.df_processed["positive_rate"] > 1)
                bad_count = bad_mask.sum()
                if bad_count > 0:
                    logger.warning("positive_rate fuera de rango [0, 1]: %d -> convertido a NaN", int(bad_count))
                    self.df_processed.loc[bad_mask, "positive_rate"] = np.nan

        if "population" in self.df_processed.columns:
            # Ensure it's numeric before comparison
             if pd.api.types.is_numeric_dtype(self.df_processed["population"]):
                low_pop_mask = self.df_processed["population"] < self.config.MIN_POPULATION
                low_pop_count = low_pop_mask.sum()
                if low_pop_count > 0:
                    logger.warning("Registros con población < %d: %d -> eliminando filas", self.config.MIN_POPULATION, int(low_pop_count))
                    self.df_processed = self.df_processed[~low_pop_mask]


    def _filter_quality(self) -> None:
        """Filters out aggregate locations and locations with too few data points."""
        if not isinstance(self.df_processed, pd.DataFrame) or "location" not in self.df_processed.columns:
            return

        initial_locations = self.df_processed["location"].nunique()

        # Filter out aggregates (continents, regions, etc.)
        self.df_processed = self.df_processed[~self.df_processed["location"].isin(self.config.AGGREGATES_TO_EXCLUDE)]

        # Filter out locations with fewer than MIN_DATA_POINTS
        counts = self.df_processed.groupby("location").size()
        valid_locations = counts[counts >= self.config.MIN_DATA_POINTS].index
        self.df_processed = self.df_processed[self.df_processed["location"].isin(valid_locations)]

        final_locations = self.df_processed["location"].nunique()
        filtered_count = initial_locations - final_locations
        self.stats["countries_filtered"] = int(filtered_count) # Assuming filtered locations are countries now
        if filtered_count > 0:
            logger.info("Ubicaciones (países) filtradas (agregados o < %d puntos): %d -> %d",
                        self.config.MIN_DATA_POINTS, initial_locations, final_locations)

    def _handle_missing(self) -> None:
        """Fills missing values using appropriate strategies (ffill, median, interpolation)."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        before_missing = int(self.df_processed.isna().sum().sum())

        # Forward fill cumulative columns within each location group
        for col in self.config.CUMULATIVE_COLS_TO_FFILL:
            if col in self.df_processed.columns:
                self.df_processed[col] = self.df_processed.groupby("location")[col].ffill()

        # Fill demographic columns with the median for that location
        for col in self.config.DEMOGRAPHIC_COLS_TO_FILL:
            if col in self.df_processed.columns:
                # Calculate medians once per group for efficiency
                medians = self.df_processed.groupby("location")[col].transform('median')
                self.df_processed[col] = self.df_processed[col].fillna(medians)

        # Interpolate daily columns linearly within each location group, then fill remaining NaNs with 0
        for col in self.config.DAILY_COLS_TO_INTERPOLATE:
            if col in self.df_processed.columns:
                # Ensure column is numeric before interpolating
                if pd.api.types.is_numeric_dtype(self.df_processed[col]):
                    self.df_processed[col] = (
                        self.df_processed.groupby("location")[col]
                        .transform(lambda s: s.interpolate(method="linear", limit_direction="both", limit_area='inside')) # Limit interpolation area
                        .fillna(0) # Fill NaNs at ends or where interpolation failed
                        .clip(lower=0) # Ensure no negative values after fillna(0)
                    )

        after_missing = int(self.df_processed.isna().sum().sum())
        missing_handled_count = before_missing - after_missing
        self.stats["missing_handled"] = missing_handled_count
        if missing_handled_count > 0 or after_missing > 0:
            logger.info("Valores faltantes manejados: %d (restantes: %d)", missing_handled_count, after_missing)

    def _handle_outliers(self) -> None:
        """Detects and treats outliers using IQR, followed by interpolation."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        validator = DataValidator()
        total_outliers_treated = 0

        for col in self.config.OUTLIER_COLS:
            if col not in self.df_processed.columns or not pd.api.types.is_numeric_dtype(self.df_processed[col]):
                continue

            # Define the outlier detection function for transform
            def detect_group_outliers(series: pd.Series) -> pd.Series:
                if series.dropna().shape[0] < 10: # Minimum data points to detect outliers
                    return pd.Series(False, index=series.index)
                return validator.detect_outliers_iqr(series, factor=self.config.OUTLIER_IQR_FACTOR)

            # Apply outlier detection within each group
            outlier_mask = self.df_processed.groupby("location")[col].transform(detect_group_outliers)
            # Ensure the mask is boolean and handle potential NaNs from transform
            outlier_mask = outlier_mask.fillna(False).astype(bool)
            col_outliers_count = int(outlier_mask.sum())

            if col_outliers_count > 0:
                # Set outliers to NaN
                self.df_processed.loc[outlier_mask, col] = np.nan
                # Interpolate the created NaNs (outliers)
                self.df_processed[col] = self.df_processed.groupby("location")[col].transform(
                    lambda s: s.interpolate(method="linear", limit=3, limit_direction="both", limit_area='inside').fillna(0) # Limit interpolation, fill ends
                )
                logger.info("Outliers detectados y tratados en '%s': %d", col, col_outliers_count)
                total_outliers_treated += col_outliers_count

        self.stats["outliers_treated"] = total_outliers_treated

    def _validate_consistency(self) -> None:
        """Performs consistency checks (e.g., total vs new cases, vaccination totals)."""
        if not isinstance(self.df_processed, pd.DataFrame) or not self.config.CONSISTENCY_CHECKS:
            return

        # Recalculate 'new_' columns from 'total_' columns where possible
        for prefix in ("cases", "deaths"):
            total_col, new_col = f"total_{prefix}", f"new_{prefix}"
            if {total_col, new_col}.issubset(self.df_processed.columns) and "location" in self.df_processed.columns:
                # Calculate daily difference from total, fill initial NaN with 0, clip negatives
                calculated_new = self.df_processed.groupby("location")[total_col].diff().fillna(0).clip(lower=0)
                # Where calculated difference is positive, use it. Otherwise, keep original (potentially 0 or imputed)
                self.df_processed[new_col] = calculated_new.where(calculated_new > 0, self.df_processed[new_col].fillna(0))
                logger.debug("Recalculado %s desde %s", new_col, total_col)

        # Enforce monotonicity for cumulative columns if enabled
        if self.config.FIX_MONOTONIC:
            self.df_processed = DataValidator.validate_monotonic_increasing(self.df_processed, self.config.MONOTONIC_COLS)
            logger.debug("Monotonicidad forzada para: %s", self.config.MONOTONIC_COLS)

        # Ensure fully vaccinated <= people vaccinated
        if {"people_fully_vaccinated", "people_vaccinated"}.issubset(self.df_processed.columns):
            # Check for numeric types before comparison
            if pd.api.types.is_numeric_dtype(self.df_processed["people_fully_vaccinated"]) and \
               pd.api.types.is_numeric_dtype(self.df_processed["people_vaccinated"]):
                mask = self.df_processed["people_fully_vaccinated"] > self.df_processed["people_vaccinated"]
                inconsistent_count = mask.sum()
                if inconsistent_count > 0:
                    logger.warning("people_fully_vaccinated > people_vaccinated: %d registros -> ajustando fully_vaccinated al valor de people_vaccinated",
                                   int(inconsistent_count))
                    self.df_processed.loc[mask, "people_fully_vaccinated"] = self.df_processed.loc[mask, "people_vaccinated"]


    def _create_features(self) -> None:
        """Creates new derived features like mortality rate, per capita metrics, etc."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        new_features_created: List[str] = []

        # Mortality Rate
        if {"total_deaths", "total_cases"}.issubset(self.df_processed.columns):
            # Calculate only where total_cases > 0 to avoid division by zero
            valid_mask = self.df_processed["total_cases"] > 0
            rate = (self.df_processed.loc[valid_mask, "total_deaths"] / self.df_processed.loc[valid_mask, "total_cases"]) * 100
            # Assign calculated rate, fill others with NaN, then clip
            self.df_processed["mortality_rate"] = rate.reindex(self.df_processed.index)
            self.df_processed["mortality_rate"] = self.df_processed["mortality_rate"].clip(upper=self.config.MAX_MORTALITY_RATE)
            new_features_created.append("mortality_rate")

        # Per 100k Metrics
        if "population" in self.df_processed.columns:
            valid_pop_mask = self.df_processed["population"] > 0
            for metric in ("total_cases", "total_deaths"):
                if metric in self.df_processed.columns:
                    colname = f"{metric}_per_100k"
                    per_capita = (self.df_processed.loc[valid_pop_mask, metric] / self.df_processed.loc[valid_pop_mask, "population"]) * 100_000
                    # Assign calculated rate, fill others with NaN
                    self.df_processed[colname] = per_capita.reindex(self.df_processed.index)
                    # Replace potential inf/-inf with NaN (shouldn't happen with valid_pop_mask, but safe)
                    self.df_processed[colname].replace([np.inf, -np.inf], np.nan, inplace=True)
                    new_features_created.append(colname)

        # Smoothed Daily Metrics (Rolling Average)
        for col in self.config.DAILY_COLS_TO_INTERPOLATE:
            if col in self.df_processed.columns and "location" in self.df_processed.columns:
                smoothed_col = f"{col}_smoothed"
                self.df_processed[smoothed_col] = self.df_processed.groupby("location")[col].transform(
                    lambda s: s.rolling(window=self.config.ROLLING_WINDOW, min_periods=1).mean()
                )
                new_features_created.append(smoothed_col)

        # Vaccination Rate
        if {"people_fully_vaccinated", "population"}.issubset(self.df_processed.columns):
            valid_mask = self.df_processed["population"] > 0
            rate = (self.df_processed.loc[valid_mask, "people_fully_vaccinated"] / self.df_processed.loc[valid_mask, "population"]) * 100
            self.df_processed["vaccination_rate"] = rate.reindex(self.df_processed.index)
            self.df_processed["vaccination_rate"] = self.df_processed["vaccination_rate"].clip(upper=100) # Rate cannot exceed 100%
            new_features_created.append("vaccination_rate")

        self.stats["features_created"] = len(new_features_created)
        if new_features_created:
            logger.info("Features creados: %d (%s)", len(new_features_created), ", ".join(new_features_created))

    def _final_cleanup(self) -> None:
        """Performs final cleanup steps: drop duplicates, sort, reset index, handle infinities."""
        if not isinstance(self.df_processed, pd.DataFrame): return
        before_rows = len(self.df_processed)

        # Drop duplicates by location/date again just in case transformations introduced any
        if {"location", "date"}.issubset(self.df_processed.columns):
            self.df_processed = self.df_processed.drop_duplicates(subset=["location", "date"], keep='first')
            # Ensure date column is valid and sort
            self.df_processed = self.df_processed[self.df_processed["date"].notna()]
            self.df_processed = self.df_processed.sort_values(["location", "date"]).reset_index(drop=True)

        # Replace any remaining infinities with NaN
        self.df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns that are entirely NaN after all processing
        all_nan_cols = self.df_processed.columns[self.df_processed.isna().all()].tolist()
        if all_nan_cols:
            self.df_processed.drop(columns=all_nan_cols, inplace=True)
            logger.info("Columnas completamente NaN eliminadas después del procesamiento: %s", all_nan_cols)

        after_rows = len(self.df_processed)
        if before_rows != after_rows or all_nan_cols:
            logger.info("Limpieza final: %d -> %d filas", before_rows, after_rows)

    # ---------- Load ----------
    def load_data(self) -> bool:
        """
        Saves the processed data, aggregate metrics, and continent-specific files.

        Saves the main processed DataFrame to CSV and compressed Gzip CSV.
        Saves continent-specific data to separate compressed Gzip CSV files. # <-- NUEVO
        Saves continent and region metrics to JSON files.
        Saves pipeline statistics to a JSON report. Calculates final file sizes.

        Returns:
            True if all save operations were successful, False otherwise.
        """
        logger.info("FASE 3: CARGA Y ALMACENAMIENTO")
        if not isinstance(self.df_processed, pd.DataFrame) or self.df_processed.empty:
            logger.error("No hay datos procesados válidos para guardar.")
            return False

        try:
            float_format = '%.2f' # Format floats to 2 decimal places in CSV

            # --- Save main processed data ---
            logger.info("Guardando archivo principal procesado...")
            self.df_processed.to_csv(
                self.config.OUTPUT_PROCESSED,
                index=False,
                float_format=float_format
            )
            self.df_processed.to_csv(
                self.config.OUTPUT_COMPRESSED,
                index=False,
                compression="gzip",
                float_format=float_format
            )
            logger.info("Archivos principales guardados: %s, %s", 
                        self.config.OUTPUT_PROCESSED.name, self.config.OUTPUT_COMPRESSED.name)

            # --- NUEVO: Save data separated by continent ---
            if "continent" in self.df_processed.columns:
                unique_continents = self.df_processed['continent'].dropna().unique()
                logger.info("Separando y guardando datos por continente para: %s", ', '.join(unique_continents))
                
                for continent in unique_continents:
                    # Crear nombre de archivo específico para el continente
                    continent_filename = self.config.DATA_DIR / f"processed_covid_{continent.replace(' ', '_')}.csv.gz"
                    
                    # Filtrar el DataFrame por el continente actual
                    df_continent = self.df_processed[self.df_processed['continent'] == continent]
                    
                    if not df_continent.empty:
                        logger.debug("Guardando datos para %s (%d filas) en %s...", 
                                     continent, len(df_continent), continent_filename.name)
                        df_continent.to_csv(
                            continent_filename,
                            index=False,
                            compression="gzip",
                            float_format=float_format
                        )
                    else:
                         logger.warning("No se encontraron datos para el continente %s, se omite el guardado.", continent)
                logger.info("Datos por continente guardados en el directorio: %s", self.config.DATA_DIR)
            else:
                 logger.warning("La columna 'continent' no existe, no se pueden separar los datos por continente.")
            # --- FIN NUEVO ---


            # --- Save aggregate metrics ---
            if isinstance(self.df_continent_metrics, pd.DataFrame) and not self.df_continent_metrics.empty:
                self.df_continent_metrics.to_json(
                    self.config.OUTPUT_CONTINENT_METRICS,
                    orient="index",
                    indent=2,
                    date_format="iso" # Ensure dates are ISO formatted
                )
                logger.info("Métricas de continentes guardadas en: %s", self.config.OUTPUT_CONTINENT_METRICS)

            if isinstance(self.df_region_metrics, pd.DataFrame) and not self.df_region_metrics.empty:
                self.df_region_metrics.to_json(
                    self.config.OUTPUT_REGION_METRICS,
                    orient="index",
                    indent=2,
                    date_format="iso"
                )
                logger.info("Métricas de regiones guardadas en: %s", self.config.OUTPUT_REGION_METRICS)

            # --- Save run statistics report ---
            with open(self.config.OUTPUT_REPORT, "w", encoding="utf-8") as fh:
                json.dump(self.stats, fh, indent=2, default=str) # Use default=str for datetime etc.

            logger.info("Reporte JSON guardado en: %s", self.config.OUTPUT_REPORT)

            # --- Calculate and log file sizes ---
            try:
                size_csv = self.config.OUTPUT_PROCESSED.stat().st_size / (1024 ** 2) if self.config.OUTPUT_PROCESSED.exists() else 0
                size_gz = self.config.OUTPUT_COMPRESSED.stat().st_size / (1024 ** 2) if self.config.OUTPUT_COMPRESSED.exists() else 0
                logger.info("Tamaño final - CSV Principal: %.2f MB, GZ Principal: %.2f MB", size_csv, size_gz)
                # Opcional: Calcular tamaño total de archivos de continentes
                total_continent_size = 0
                for continent_file in self.config.DATA_DIR.glob("processed_covid_*.csv.gz"):
                    total_continent_size += continent_file.stat().st_size
                logger.info("Tamaño total de archivos por continente (GZ): %.2f MB", total_continent_size / (1024**2))

            except Exception as e:
                logger.warning("No se pudieron calcular los tamaños de archivo finales: %s", e)

            return True
        except IOError as io_exc:
            logger.exception("Error de I/O guardando archivos: %s", io_exc)
            return False
        except Exception as exc:
            logger.exception("Error inesperado guardando archivos: %s", exc)
            return False

    def generate_report(self) -> None:
        """Logs a summary report of the processed data quality to the console."""
        if not isinstance(self.df_processed, pd.DataFrame):
            logger.warning("No hay datos procesados para generar reporte.")
            return

        logger.info("--- REPORTE DE CALIDAD DE DATOS PROCESADOS ---")
        rows, cols = self.df_processed.shape
        unique_countries = 0
        if "location" in self.df_processed.columns:
            unique_countries = int(self.df_processed["location"].nunique())

        missing_values = int(self.df_processed.isna().sum().sum())
        total_cells = rows * cols
        completeness = ((total_cells - missing_values) / total_cells) * 100 if total_cells > 0 else 0.0

        logger.info("Dimensiones finales: %d filas x %d columnas", rows, cols)
        logger.info("Países únicos procesados: %d", unique_countries)
        logger.info("Duplicados/filas vacías removidas: %d", self.stats.get("duplicates_removed", 0))
        logger.info("Outliers tratados (valores reemplazados): %d", self.stats.get("outliers_treated", 0))
        logger.info("Valores faltantes manejados (imputados/rellenados): %d", self.stats.get("missing_handled", 0))
        logger.info("Completitud general de datos: %.2f%% (%d celdas no nulas de %d totales)",
                    completeness, total_cells - missing_values, total_cells)
        logger.info("-------------------------------------------------")


    def run(self) -> bool:
        """
        Executes the full ETL pipeline: Extract, Transform, Load.

        Records start and end times, manages overall success status,
        generates reports, and saves a final summary text file.

        Returns:
            True if the entire pipeline completed successfully, False otherwise.
        """
        self.stats["start_time"] = datetime.now()
        run_success: bool = False
        try:
            # Chain the steps: proceed only if the previous one succeeded
            extract_ok = self.extract_data()
            transform_ok = extract_ok and self.transform_data()
            load_ok = transform_ok and self.load_data()
            run_success = load_ok # Overall success depends on the last step completing

        except Exception as e:
            logger.exception("PIPELINE FALLÓ con una excepción inesperada no capturada: %s", e)
            run_success = False # Ensure failure state
        finally:
            self.stats["end_time"] = datetime.now()
            # Calculate elapsed time safely, even if start_time is None (though unlikely)
            start_time = self.stats.get("start_time")
            end_time = self.stats.get("end_time")
            elapsed_seconds = (end_time - start_time).total_seconds() if start_time and end_time else 0.0

            if run_success:
                self.generate_report() # Log quality report to console
                logger.info("🚀 PIPELINE COMPLETADO EXITOSAMENTE en %.2f segundos", elapsed_seconds)
                self._save_summary(elapsed_seconds, success=True)
            else:
                logger.error("❌ PIPELINE FALLÓ. Revisa el log '%s' para más detalles.", self.config.LOG_FILE.name)
                self._save_summary(elapsed_seconds, success=False)
        return run_success

    def _save_summary(self, elapsed: float, success: bool = True) -> None:
        """Saves a brief text summary of the pipeline execution."""
        try:
            with open(self.config.OUTPUT_SUMMARY, "w", encoding="utf-8") as fh:
                fh.write(f"RESUMEN PIPELINE COVID-19 ETL\n")
                fh.write("=" * 60 + "\n")
                fh.write(f"Estado: {'EXITOSO' if success else 'FALLIDO'}\n")
                fh.write(f"Inicio: {self.stats.get('start_time', 'N/A')}\n")
                fh.write(f"Fin:    {self.stats.get('end_time', 'N/A')}\n")
                fh.write(f"Duración: {elapsed:.2f} segundos\n\n")

                fh.write("--- Estadísticas Clave ---\n")
                raw_shape: Tuple[Optional[int], Optional[int]] = self.stats.get('raw_shape', (None, None))
                proc_shape: Tuple[Optional[int], Optional[int]] = self.stats.get('processed_shape', (None, None))

                fh.write(f"Datos Crudos:   {raw_shape[0]:,} filas x {raw_shape[1]} columnas\n" if raw_shape[0] is not None else "Datos Crudos: N/A\n")
                fh.write(f"Datos Procesados: {proc_shape[0]:,} filas x {proc_shape[1]} columnas\n\n" if proc_shape[0] is not None else "Datos Procesados: N/A\n")

                fh.write(f"Duplicados Removidos: {self.stats.get('duplicates_removed', 'N/A')}\n")
                fh.write(f"Outliers Tratados:    {self.stats.get('outliers_treated', 'N/A')}\n")
                fh.write(f"Missing Manejados:  {self.stats.get('missing_handled', 'N/A')}\n")
                fh.write(f"Países Filtrados:   {self.stats.get('countries_filtered', 'N/A')}\n")
                fh.write(f"Features Creados:   {self.stats.get('features_created', 'N/A')}\n\n")

                fh.write(f"Reporte JSON detallado en: {self.config.OUTPUT_REPORT.name}\n")
                fh.write(f"Log completo en: {self.config.LOG_FILE.name}\n")

            logger.info("Resumen de ejecución guardado en %s", self.config.OUTPUT_SUMMARY.name)
        except IOError as e:
            logger.error("No se pudo guardar el archivo de resumen de texto: %s", e)
        except Exception as e:
            logger.exception("Error inesperado guardando el resumen de texto: %s", e)


# -----------------------------
# Script Execution
# -----------------------------
def main() -> None:
    """Main function to run the ETL pipeline."""
    # Config is already initialized globally
    pipeline = ImprovedCovidETL(config)
    success = pipeline.run()
    if not success:
        # Exit with a non-zero code to indicate failure, useful for scripting
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido manualmente por el usuario (Ctrl+C).")
        raise SystemExit(130) # Standard exit code for Ctrl+C
    except SystemExit as sysexit:
        # Propagate SystemExit with its code (e.g., from main())
        raise sysexit
    except Exception:
        # Catch any other unexpected error during script execution
        logger.exception("Ejecución del script terminada debido a un error no manejado.")
        raise # Re-raise the exception after logging
# %%