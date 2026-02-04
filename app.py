# ============================================================================
# PHARMAINTELLIGENCE PRO - ENTERPRISE PHARMACEUTICAL ANALYTICS PLATFORM
# ============================================================================
# Version: 7.0 - PROFESSIONAL EDITION
# Lines: 4000+
# Author: Enterprise Analytics Team
# License: Proprietary
# Copyright: © 2024 PharmaIntelligence Inc.
# 
# Features:
# - Advanced Machine Learning (RF, GBM, XGBoost-style)
# - Multi-dimensional Clustering & Segmentation
# - Anomaly Detection with Severity Scoring
# - Predictive Analytics & Forecasting
# - Interactive Dashboards & Visualizations
# - Geographic Market Analysis
# - Competitive Intelligence
# - What-If Scenario Analysis
# - Comprehensive Reporting & Export
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning & Statistics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    IsolationForest, 
    RandomForestRegressor, 
    GradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error, 
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    GridSearchCV,
    KFold
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO, StringIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import re
from collections import defaultdict, Counter, OrderedDict
import hashlib
import base64

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

class Config:
    """
    Centralized application configuration management
    
    This class contains all application-wide settings including:
    - Display configurations
    - Performance parameters
    - ML hyperparameters
    - UI styling constants
    """
    
    # Application metadata
    APP_NAME = "PharmaIntelligence Pro"
    VERSION = "7.0"
    BUILD_DATE = "2024-02-04"
    ORGANIZATION = "Enterprise Analytics Team"
    
    # Display settings
    MAX_ROWS_DISPLAY = 1000
    MAX_ROWS_EXPORT = 100000
    CHART_HEIGHT_STANDARD = 500
    CHART_HEIGHT_LARGE = 700
    CHART_HEIGHT_SMALL = 400
    CHART_HEIGHT_XLARGE = 900
    
    # Performance settings
    SAMPLE_SIZE_DEFAULT = 5000
    SAMPLE_SIZE_MIN = 1000
    SAMPLE_SIZE_MAX = 50000
    CACHE_TTL = 3600  # 1 hour
    CHUNK_SIZE = 10000
    
    # ML parameters
    ML_MIN_SAMPLES = 50
    ML_MIN_SAMPLES_CLUSTERING = 100
    ML_MIN_SAMPLES_ANOMALY = 30
    RANDOM_STATE = 42
    N_JOBS = -1
    CV_FOLDS = 5
    
    # Clustering configuration
    DEFAULT_N_CLUSTERS = 4
    MAX_N_CLUSTERS = 15
    MIN_N_CLUSTERS = 2
    MIN_SAMPLES_PER_CLUSTER = 10
    
    # Anomaly detection
    DEFAULT_CONTAMINATION = 0.1
    MIN_CONTAMINATION = 0.01
    MAX_CONTAMINATION = 0.3
    
    # Forecasting
    MAX_FORECAST_YEARS = 10
    MIN_FORECAST_YEARS = 1
    DEFAULT_FORECAST_YEARS = 3
    
    # Color schemes - Professional palette
    COLOR_PRIMARY = '#2d7dd2'
    COLOR_SECONDARY = '#2acaea'
    COLOR_SUCCESS = '#2dd4a3'
    COLOR_WARNING = '#fbbf24'
    COLOR_DANGER = '#ff4444'
    COLOR_INFO = '#60a5fa'
    COLOR_PURPLE = '#a855f7'
    COLOR_PINK = '#ec4899'
    COLOR_ORANGE = '#f97316'
    COLOR_TEAL = '#14b8a6'
    
    # Chart color palettes
    PALETTE_CATEGORICAL = [
        '#2d7dd2', '#2acaea', '#2dd4a3', '#fbbf24', 
        '#ff4444', '#a855f7', '#ec4899', '#f97316'
    ]
    
    PALETTE_SEQUENTIAL = [
        '#0f172a', '#1e293b', '#334155', '#475569',
        '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0'
    ]
    
    # Statistical thresholds
    SIGNIFICANCE_LEVEL = 0.05
    CONFIDENCE_LEVEL = 0.95
    OUTLIER_STD_THRESHOLD = 3
    
    # Export settings
    EXPORT_FORMATS = ['csv', 'json', 'xlsx', 'html']
    EXPORT_ENCODING = 'utf-8'
    
    # Data quality thresholds
    MIN_DATA_QUALITY_SCORE = 0.7
    MAX_MISSING_PERCENTAGE = 30
    
    # UI Settings
    SIDEBAR_WIDTH = 300
    ANIMATION_DURATION = 500
    TOAST_DURATION = 3000

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

PROFESSIONAL_CSS = """
<style>
    /* ========== IMPORTS & FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        letter-spacing: -0.01em;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* ========== GLOBAL STYLES ========== */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* ========== HEADERS ========== */
    .section-header {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(42, 202, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 15s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .section-header h1, .section-header h2 {
        color: #f8fafc;
        margin: 0;
        font-weight: 800;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .section-header h1 {
        font-size: 3rem;
        background: linear-gradient(to right, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header h2 {
        font-size: 2rem;
    }
    
    .section-header p {
        color: #cbd5e1;
        margin: 0.75rem 0 0 0;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    .section-header .version-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .subsection-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin: 2rem 0 1.5rem 0;
        border-left: 5px solid #2acaea;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .subsection-header::after {
        content: '';
        position: absolute;
        right: -50px;
        top: -50px;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(42, 202, 234, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .subsection-header h3 {
        color: #2acaea;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .subsection-header p {
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        position: relative;
        z-index: 1;
    }
    
    /* ========== KPI CARDS ========== */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #2acaea;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(42, 202, 234, 0.05) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .kpi-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(42, 202, 234, 0.6);
        border-left-color: #60a5fa;
    }
    
    .kpi-card:hover::before {
        opacity: 1;
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .kpi-value {
        color: #f8fafc;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.75rem;
        line-height: 1;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        background: linear-gradient(to right, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .kpi-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .kpi-delta {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 0.75rem;
        backdrop-filter: blur(10px);
    }
    
    .kpi-delta.positive {
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.3), rgba(45, 212, 163, 0.1));
        color: #2dd4a3;
        border: 1px solid rgba(45, 212, 163, 0.5);
    }
    
    .kpi-delta.negative {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.3), rgba(255, 68, 68, 0.1));
        color: #ff4444;
        border: 1px solid rgba(255, 68, 68, 0.5);
    }
    
    .kpi-delta.neutral {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.3), rgba(96, 165, 250, 0.1));
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.5);
    }
    
    /* ========== INSIGHT CARDS ========== */
    .insight-card {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.15) 0%, rgba(45, 125, 210, 0.05) 100%);
        border-left: 5px solid #2d7dd2;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .insight-card:hover {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.25) 0%, rgba(45, 125, 210, 0.1) 100%);
        transform: translateX(8px);
        box-shadow: 0 8px 20px rgba(42, 202, 234, 0.3);
    }
    
    .insight-card.success {
        border-left-color: #2dd4a3;
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.15) 0%, rgba(45, 212, 163, 0.05) 100%);
    }
    
    .insight-card.warning {
        border-left-color: #fbbf24;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%);
    }
    
    .insight-card.danger {
        border-left-color: #ff4444;
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 68, 68, 0.05) 100%);
    }
    
    .insight-card.info {
        border-left-color: #60a5fa;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(96, 165, 250, 0.05) 100%);
    }
    
    .insight-card h4 {
        color: #2acaea;
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-card.success h4 { color: #2dd4a3; }
    .insight-card.warning h4 { color: #fbbf24; }
    .insight-card.danger h4 { color: #ff4444; }
    .insight-card.info h4 { color: #60a5fa; }
    
    .insight-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* ========== METRIC BADGES ========== */
    .metric-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(42, 202, 234, 0.2), rgba(42, 202, 234, 0.1));
        color: #2acaea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        border: 1px solid rgba(42, 202, 234, 0.3);
    }
    
    .metric-badge.success {
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.2), rgba(45, 212, 163, 0.1));
        color: #2dd4a3;
        border-color: rgba(45, 212, 163, 0.3);
    }
    
    .metric-badge.warning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.1));
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.3);
    }
    
    .metric-badge.danger {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.2), rgba(255, 68, 68, 0.1));
        color: #ff4444;
        border-color: rgba(255, 68, 68, 0.3);
    }
    
    /* ========== FILTER BADGE ========== */
    .filter-badge {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 40px;
        display: inline-block;
        margin: 1.5rem 0;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(42, 202, 234, 0.5);
        animation: pulse-glow 3s infinite;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 8px 20px rgba(42, 202, 234, 0.5);
        }
        50% {
            box-shadow: 0 8px 30px rgba(42, 202, 234, 0.8);
        }
    }
    
    /* ========== NOTIFICATION BOXES ========== */
    .success-box {
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.2) 0%, rgba(45, 212, 163, 0.05) 100%);
        border-left: 6px solid #2dd4a3;
        padding: 1.5rem;
        border-radius: 16px;
        color: #cbd5e1;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(45, 212, 163, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 6px solid #fbbf24;
        padding: 1.5rem;
        border-radius: 16px;
        color: #cbd5e1;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(251, 191, 36, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.2) 0%, rgba(96, 165, 250, 0.05) 100%);
        border-left: 6px solid #60a5fa;
        padding: 1.5rem;
        border-radius: 16px;
        color: #cbd5e1;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(96, 165, 250, 0.3);
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.2) 0%, rgba(255, 68, 68, 0.05) 100%);
        border-left: 6px solid #ff4444;
        padding: 1.5rem;
        border-radius: 16px;
        color: #cbd5e1;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(255, 68, 68, 0.3);
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 2.5rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 16px rgba(42, 202, 234, 0.4);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 28px rgba(42, 202, 234, 0.6);
        background: linear-gradient(135deg, #3d8de2 0%, #3adcfa 100%);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.6), rgba(30, 58, 95, 0.4));
        padding: 0.75rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 700;
        padding: 1rem 2rem;
        transition: all 0.3s;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(42, 202, 234, 0.2), rgba(42, 202, 234, 0.1));
        color: #2acaea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white !important;
        box-shadow: 0 6px 16px rgba(42, 202, 234, 0.5);
    }
    
    /* ========== METRICS ========== */
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(42, 202, 234, 0.2);
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    /* ========== DATAFRAME ========== */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(42, 202, 234, 0.5);
    }
    
    /* ========== SIDEBAR ========== */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(45, 125, 210, 0.1));
        border-radius: 12px;
        font-weight: 700;
        color: #2acaea;
        padding: 1rem !important;
        border: 1px solid rgba(42, 202, 234, 0.3);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.3), rgba(45, 125, 210, 0.15));
        border-color: rgba(42, 202, 234, 0.5);
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 12px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        border-radius: 12px;
        border: 2px solid #1e293b;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3d8de2 0%, #3adcfa 100%);
    }
    
    /* ========== SELECT BOXES ========== */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.6), rgba(30, 58, 95, 0.4));
        border-radius: 12px;
        color: #f8fafc;
        border: 1px solid rgba(42, 202, 234, 0.3);
    }
    
    /* ========== MULTISELECT ========== */
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.6), rgba(30, 58, 95, 0.4));
        border-radius: 12px;
        border: 1px solid rgba(42, 202, 234, 0.3);
    }
    
    /* ========== TEXT INPUT ========== */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.6), rgba(30, 58, 95, 0.4));
        border-radius: 12px;
        color: #f8fafc;
        border: 2px solid rgba(42, 202, 234, 0.3);
        padding: 0.75rem 1rem;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2acaea;
        box-shadow: 0 0 0 3px rgba(42, 202, 234, 0.2);
    }
    
    /* ========== SLIDER ========== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
        border-radius: 12px;
    }
    
    /* ========== STATISTICS TABLE ========== */
    .stats-table {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(30, 58, 95, 0.2));
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(42, 202, 234, 0.2);
    }
    
    .stats-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .stats-table th {
        color: #2acaea;
        font-weight: 700;
        padding: 1rem;
        text-align: left;
        border-bottom: 2px solid rgba(42, 202, 234, 0.3);
    }
    
    .stats-table td {
        color: #cbd5e1;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.6s ease-out;
    }
    
    .scale-in {
        animation: scaleIn 0.5s ease-out;
    }
    
    /* ========== LOADING SPINNER ========== */
    .stSpinner > div {
        border-top-color: #2acaea !important;
        border-right-color: #2acaea !important;
    }
    
    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: 768px) {
        .section-header h1 {
            font-size: 2rem;
        }
        
        .section-header h2 {
            font-size: 1.5rem;
        }
        
        .kpi-value {
            font-size: 2rem;
        }
        
        .stButton > button {
            padding: 0.75rem 1.5rem;
            font-size: 0.9rem;
        }
    }
    
    /* ========== CUSTOM COMPONENTS ========== */
    .feature-showcase {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.4), rgba(30, 58, 95, 0.2));
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s;
        border: 1px solid rgba(42, 202, 234, 0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(42, 202, 234, 0.3);
        border-color: rgba(42, 202, 234, 0.5);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #2acaea;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }
</style>
"""

# Page configuration
st.set_page_config(
    page_title=f"{Config.APP_NAME} v{Config.VERSION}",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/support',
        'Report a bug': 'https://pharmaintelligence.com/bug-report',
        'About': f"""
        ### {Config.APP_NAME} v{Config.VERSION}
        
        **Enterprise-Grade Pharmaceutical Analytics Platform**
        
        **Build Date:** {Config.BUILD_DATE}
        **Organization:** {Config.ORGANIZATION}
        
        **Features:**
        - 🤖 Advanced Machine Learning & AI
        - 📊 Predictive Analytics & Forecasting
        - 🎯 Multi-dimensional Clustering
        - ⚠️ Intelligent Anomaly Detection
        - 🌍 Geographic Market Analysis
        - 🏆 Competitive Intelligence
        - 📈 What-If Scenario Modeling
        - 📑 Comprehensive Reporting
        
        © 2024 PharmaIntelligence Inc. All rights reserved.
        """
    }
)

# Apply CSS
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

class Logger:
    """
    Professional logging utility for application events
    """
    
    @staticmethod
    def info(message: str) -> None:
        """Log info message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[INFO] [{timestamp}] {message}")
    
    @staticmethod
    def warning(message: str) -> None:
        """Log warning message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[WARNING] [{timestamp}] {message}")
    
    @staticmethod
    def error(message: str, exception: Optional[Exception] = None) -> None:
        """Log error message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[ERROR] [{timestamp}] {message}")
        if exception:
            print(f"[ERROR] Exception: {str(exception)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")


class Utils:
    """
    Comprehensive utility functions for data formatting and manipulation
    """
    
    @staticmethod
    def format_number(num: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        """
        Format number with intelligent abbreviation
        
        Args:
            num: Number to format
            prefix: Prefix string (e.g., '$', '€')
            suffix: Suffix string (e.g., '%', 'x')
            decimals: Number of decimal places
        
        Returns:
            Formatted string
        """
        if pd.isna(num) or num is None:
            return "N/A"
        
        try:
            num = float(num)
        except:
            return "N/A"
        
        abs_num = abs(num)
        
        if abs_num >= 1e12:
            return f"{prefix}{num/1e12:.{decimals}f}T{suffix}"
        elif abs_num >= 1e9:
            return f"{prefix}{num/1e9:.{decimals}f}B{suffix}"
        elif abs_num >= 1e6:
            return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
        elif abs_num >= 1e3:
            return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
        else:
            return f"{prefix}{num:.{decimals}f}{suffix}"
    
    @staticmethod
    def format_percentage(num: float, decimals: int = 1, include_sign: bool = True) -> str:
        """Format percentage with optional sign"""
        if pd.isna(num) or num is None:
            return "N/A"
        
        try:
            num = float(num)
        except:
            return "N/A"
        
        sign = "+" if num > 0 and include_sign else ""
        return f"{sign}{num:.{decimals}f}%"
    
    @staticmethod
    def format_currency(num: float, currency: str = "USD", decimals: int = 2) -> str:
        """Format currency with symbol"""
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'TRY': '₺'
        }
        
        symbol = currency_symbols.get(currency, currency)
        return Utils.format_number(num, prefix=symbol, decimals=decimals)
    
    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            start_value: Starting value
            end_value: Ending value
            periods: Number of periods
        
        Returns:
            CAGR as percentage
        """
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return np.nan
        
        try:
            cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100
            return cagr
        except:
            return np.nan
    
    @staticmethod
    def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default fallback"""
        if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
            return default
        try:
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def get_color_scale(value: float, min_val: float, max_val: float, 
                       reverse: bool = False) -> str:
        """
        Get color based on value position in range
        
        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            reverse: Reverse color scale
        
        Returns:
            Color hex code
        """
        if pd.isna(value):
            return Config.COLOR_INFO
        
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        
        if reverse:
            normalized = 1 - normalized
        
        if normalized < 0.25:
            return Config.COLOR_DANGER
        elif normalized < 0.5:
            return Config.COLOR_WARNING
        elif normalized < 0.75:
            return Config.COLOR_INFO
        else:
            return Config.COLOR_SUCCESS
    
    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, 
                           file_format: str = 'csv') -> None:
        """
        Create download button for dataframe with duplicate column handling
        
        Args:
            df: DataFrame to download
            filename: Base filename
            file_format: Format ('csv', 'json', 'xlsx', 'html')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"{filename}_{timestamp}.{file_format}"
        
        # Remove duplicate columns
        df_export = df.loc[:, ~df.columns.duplicated()]
        
        try:
            if file_format == 'csv':
                data = df_export.to_csv(index=False, encoding=Config.EXPORT_ENCODING)
                mime = 'text/csv'
            elif file_format == 'json':
                data = df_export.to_json(orient='records', indent=2)
                mime = 'application/json'
            elif file_format == 'xlsx':
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='Data')
                data = output.getvalue()
                mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif file_format == 'html':
                data = df_export.to_html(index=False)
                mime = 'text/html'
            else:
                data = df_export.to_csv(index=False)
                mime = 'text/csv'
            
            st.download_button(
                label=f"⬇️ Download {file_format.upper()}",
                data=data,
                file_name=full_filename,
                mime=mime,
                use_container_width=True
            )
            
        except Exception as e:
            Logger.error(f"Download creation failed: {str(e)}", e)
            st.error(f"❌ Could not create download: {str(e)}")
    
    @staticmethod
    def calculate_statistics(series: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a series
        
        Args:
            series: Pandas series
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        try:
            clean_series = series.dropna()
            
            if len(clean_series) == 0:
                return stats
            
            stats['count'] = len(clean_series)
            stats['mean'] = float(clean_series.mean())
            stats['median'] = float(clean_series.median())
            stats['std'] = float(clean_series.std())
            stats['min'] = float(clean_series.min())
            stats['max'] = float(clean_series.max())
            stats['q1'] = float(clean_series.quantile(0.25))
            stats['q3'] = float(clean_series.quantile(0.75))
            stats['iqr'] = stats['q3'] - stats['q1']
            stats['range'] = stats['max'] - stats['min']
            stats['cv'] = (stats['std'] / stats['mean'] * 100) if stats['mean'] != 0 else 0
            stats['skewness'] = float(clean_series.skew())
            stats['kurtosis'] = float(clean_series.kurtosis())
            
            # Outlier detection
            lower_bound = stats['q1'] - 1.5 * stats['iqr']
            upper_bound = stats['q3'] + 1.5 * stats['iqr']
            stats['outliers_count'] = int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())
            stats['outliers_percentage'] = (stats['outliers_count'] / len(clean_series)) * 100
            
        except Exception as e:
            Logger.warning(f"Statistics calculation failed: {str(e)}")
        
        return stats


class PerformanceMonitor:
    """
    Monitor and optimize application performance
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str) -> None:
        """Record a performance checkpoint"""
        self.checkpoints[name] = time.time() - self.start_time
    
    def get_elapsed(self, name: Optional[str] = None) -> float:
        """Get elapsed time for checkpoint or total"""
        if name and name in self.checkpoints:
            return self.checkpoints[name]
        return time.time() - self.start_time
    
    def report(self) -> Dict[str, float]:
        """Get performance report"""
        return {
            'total_elapsed': self.get_elapsed(),
            **self.checkpoints
        }


# ============================================================================
# DATA MANAGEMENT CLASS
# ============================================================================

class DataManager:
    """
    Advanced data processing and management with quality assurance
    """
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def load_data(file, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load and validate data from uploaded file
        
        Args:
            file: Uploaded file object
            sample_size: Optional sample size for large datasets
        
        Returns:
            Processed DataFrame or None if error
        """
        monitor = PerformanceMonitor()
        
        try:
            Logger.info(f"Loading data from {file.name}")
            
            # Determine file type and load
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, nrows=sample_size, low_memory=False, encoding='utf-8')
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
            else:
                st.error("❌ Unsupported file format! Please upload CSV or Excel file.")
                return None
            
            monitor.checkpoint('load')
            
            if df is None or len(df) == 0:
                st.error("❌ File is empty or could not be read!")
                return None
            
            # Clean column names
            df.columns = DataManager.clean_column_names(df.columns)
            monitor.checkpoint('clean_columns')
            
            # Remove duplicate columns immediately
            df = df.loc[:, ~df.columns.duplicated()]
            monitor.checkpoint('remove_duplicates')
            
            # Optimize memory usage
            df = DataManager.optimize_dataframe(df)
            monitor.checkpoint('optimize')
            
            # Calculate quality metrics
            quality_score = DataManager.assess_data_quality(df)
            
            # Performance report
            perf_report = monitor.report()
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            
            st.success(
                f"✅ **Data loaded successfully!**\n\n"
                f"📊 {len(df):,} rows × {len(df.columns)} columns\n\n"
                f"⏱️ Load time: {perf_report['total_elapsed']:.2f}s\n\n"
                f"💾 Memory: {memory_mb:.1f} MB\n\n"
                f"🎯 Quality Score: {quality_score:.1%}"
            )
            
            Logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            Logger.error(f"Data loading failed: {str(e)}", e)
            st.error(f"❌ **Data loading error:**\n\n{str(e)}")
            with st.expander("🔍 View detailed error"):
                st.code(traceback.format_exc())
            return None
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """
        Intelligently clean and standardize column names
        
        Args:
            columns: List of column names
        
        Returns:
            List of cleaned column names
        """
        cleaned = []
        seen = {}
        
        # Turkish to English character mapping
        tr_to_en = {
            'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
            'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
            'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
        }
        
        for col in columns:
            # Convert to string
            col = str(col) if not isinstance(col, str) else col
            
            # Remove Turkish characters
            for tr_char, en_char in tr_to_en.items():
                col = col.replace(tr_char, en_char)
            
            # Clean whitespace and special characters
            col = ' '.join(col.split()).strip()
            col = re.sub(r'[^\w\s-]', '', col)
            
            # Standardize pharmaceutical data patterns
            col = DataManager.standardize_pharma_column(col)
            
            # Handle duplicates by adding numeric suffix
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def standardize_pharma_column(col: str) -> str:
        """
        Standardize pharmaceutical-specific column patterns
        
        Args:
            col: Column name
        
        Returns:
            Standardized column name
        """
        # Sales columns
        if 'USD' in col and 'MAT' in col:
            year_match = re.search(r'(\d{4})', col)
            year = year_match.group(1) if year_match else '2024'
            
            if 'Units' in col or 'Unit' in col:
                return f'Birim_{year}'
            elif 'Avg Price' in col or 'Price' in col:
                return f'Ort_Fiyat_{year}'
            else:
                return f'Satis_{year}'
        
        # Growth columns
        if 'Growth' in col or 'Buyume' in col:
            year_match = re.findall(r'(\d{4})', col)
            if len(year_match) >= 2:
                return f'Buyume_{year_match[0]}_{year_match[1]}'
        
        return col
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through intelligent type conversion
        
        Args:
            df: Input DataFrame
        
        Returns:
            Optimized DataFrame
        """
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Optimize categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df)
                
                # Convert to category if cardinality is low
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(df[col]):
                    # Unsigned integers
                    if col_min >= 0:
                        if col_max <= 255:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max <= 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif col_max <= 4294967295:
                            df[col] = df[col].astype(np.uint32)
                    # Signed integers
                    else:
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype(np.int16)
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            df[col] = df[col].astype(np.int32)
                # Float to float32
                else:
                    df[col] = df[col].astype(np.float32)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            savings = original_memory - optimized_memory
            
            if savings > 0:
                st.info(
                    f"💾 **Memory Optimization:**\n\n"
                    f"Before: {original_memory:.1f} MB\n\n"
                    f"After: {optimized_memory:.1f} MB\n\n"
                    f"Saved: {savings:.1f} MB ({savings/original_memory*100:.1f}%)"
                )
                Logger.info(f"Memory optimized: {savings:.1f} MB saved")
            
            return df
            
        except Exception as e:
            Logger.warning(f"Memory optimization failed: {str(e)}")
            return df
    
    @staticmethod
    def assess_data_quality(df: pd.DataFrame) -> float:
        """
        Assess overall data quality score
        
        Args:
            df: DataFrame to assess
        
        Returns:
            Quality score between 0 and 1
        """
        try:
            scores = []
            
            # Completeness (missing data)
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            completeness = max(0, 1 - (missing_pct / 100))
            scores.append(completeness)
            
            # Consistency (duplicate rows)
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            consistency = max(0, 1 - (duplicate_pct / 100))
            scores.append(consistency)
            
            # Validity (numeric columns within reasonable ranges)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                validity_scores = []
                for col in numeric_cols:
                    # Check for extreme outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
                    validity_scores.append(max(0, 1 - (outliers / len(df))))
                
                validity = np.mean(validity_scores) if validity_scores else 1.0
                scores.append(validity)
            
            # Overall score
            quality_score = np.mean(scores)
            
            return quality_score
            
        except Exception as e:
            Logger.warning(f"Quality assessment failed: {str(e)}")
            return 0.7  # Default acceptable quality
    
    @staticmethod
    def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering for pharmaceutical data analysis
        
        Args:
            df: Input DataFrame
        
        Returns:
            Enhanced DataFrame with calculated features
        """
        try:
            Logger.info("Starting feature engineering")
            monitor = PerformanceMonitor()
            
            # Identify sales columns
            sales_cols = {}
            for col in df.columns:
                if 'Satis_' in col or 'Sales_' in col:
                    year_match = re.search(r'(\d{4})', col)
                    if year_match:
                        year = year_match.group(1)
                        if year not in sales_cols:
                            sales_cols[year] = col
            
            if not sales_cols:
                Logger.warning("No sales columns found")
                return df
            
            years = sorted([int(y) for y in sales_cols.keys()])
            Logger.info(f"Found {len(years)} years of sales data: {years}")
            
            # ========== PRICE CALCULATION ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col or 'Unit' in col]
            
            if not price_cols and sales_cols and unit_cols:
                Logger.info("Calculating prices from sales and units")
                for year in sales_cols.keys():
                    price_col_name = f'Ort_Fiyat_{year}'
                    
                    if price_col_name not in df.columns:
                        sales_col = sales_cols[year]
                        unit_col = f"Birim_{year}"
                        
                        if unit_col in df.columns and sales_col in df.columns:
                            df[price_col_name] = np.where(
                                df[unit_col] != 0,
                                df[sales_col] / df[unit_col],
                                np.nan
                            )
                            Logger.info(f"Created price column: {price_col_name}")
            
            monitor.checkpoint('prices')
            
            # ========== GROWTH RATES (YoY) ==========
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                growth_col_name = f'Buyume_{prev_year}_{curr_year}'
                change_col_name = f'Degisim_{prev_year}_{curr_year}'
                
                if growth_col_name not in df.columns:
                    if prev_year in sales_cols and curr_year in sales_cols:
                        prev_col = sales_cols[prev_year]
                        curr_col = sales_cols[curr_year]
                        
                        # YoY Growth %
                        df[growth_col_name] = (
                            (df[curr_col] - df[prev_col]) / 
                            df[prev_col].replace(0, np.nan)
                        ) * 100
                        
                        # Absolute change
                        df[change_col_name] = df[curr_col] - df[prev_col]
                        
                        Logger.info(f"Calculated growth: {growth_col_name}")
            
            monitor.checkpoint('growth')
            
            # ========== PRICE ANALYSIS ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            
            if price_cols and 'Ort_Fiyat_Genel' not in df.columns:
                # Average price across years
                df['Ort_Fiyat_Genel'] = df[price_cols].mean(axis=1, skipna=True)
                
                # Price volatility
                df['Fiyat_Volatilite'] = df[price_cols].std(axis=1, skipna=True)
                
                # Price trend (slope)
                if len(price_cols) >= 2:
                    def calc_price_trend(row):
                        prices = row[price_cols].dropna()
                        if len(prices) < 2:
                            return np.nan
                        x = np.arange(len(prices))
                        y = prices.values
                        try:
                            slope = np.polyfit(x, y, 1)[0]
                            return slope
                        except:
                            return np.nan
                    
                    df['Fiyat_Trend'] = df.apply(calc_price_trend, axis=1)
                
                Logger.info("Price analysis features created")
            
            monitor.checkpoint('price_analysis')
            
            # ========== CAGR ==========
            if len(years) >= 2 and 'CAGR' not in df.columns:
                first_year = str(years[0])
                last_year = str(years[-1])
                
                if first_year in sales_cols and last_year in sales_cols:
                    num_years = len(years) - 1
                    df['CAGR'] = (
                        (df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan))
                        ** (1/num_years) - 1
                    ) * 100
                    Logger.info("CAGR calculated")
            
            monitor.checkpoint('cagr')
            
            # ========== MARKET SHARE ==========
            if years and 'Pazar_Payi' not in df.columns:
                last_year = str(years[-1])
                if last_year in sales_cols:
                    last_sales_col = sales_cols[last_year]
                    total_sales = df[last_sales_col].sum()
                    
                    if total_sales > 0:
                        df['Pazar_Payi'] = (df[last_sales_col] / total_sales) * 100
                        
                        # Cumulative market share (for Pareto analysis)
                        df_sorted = df.sort_values(last_sales_col, ascending=False).reset_index(drop=True)
                        df_sorted['Kumulatif_Pazar_Payi'] = df_sorted['Pazar_Payi'].cumsum()
                        
                        df = df.merge(
                            df_sorted[['Kumulatif_Pazar_Payi']],
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                        
                        Logger.info("Market share calculated")
            
            monitor.checkpoint('market_share')
            
            # ========== SALES MOMENTUM ==========
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            
            if growth_cols and 'Momentum_Ortalama' not in df.columns:
                df['Momentum_Ortalama'] = df[growth_cols].mean(axis=1, skipna=True)
                df['Momentum_Volatilite'] = df[growth_cols].std(axis=1, skipna=True)
                
                # Growth acceleration
                if len(growth_cols) >= 2:
                    df['Ivmelenme'] = df[growth_cols[-1]] - df[growth_cols[-2]]
                
                Logger.info("Momentum features created")
            
            monitor.checkpoint('momentum')
            
            # ========== VOLUME ANALYSIS ==========
            unit_cols = [col for col in df.columns if 'Birim_' in col]
            
            if unit_cols and 'Toplam_Hacim' not in df.columns:
                df['Toplam_Hacim'] = df[unit_cols].sum(axis=1, skipna=True)
                df['Ortalama_Hacim'] = df[unit_cols].mean(axis=1, skipna=True)
                
                # Volume growth
                if len(unit_cols) >= 2:
                    df['Hacim_Buyume'] = (
                        (df[unit_cols[-1]] - df[unit_cols[-2]]) / 
                        df[unit_cols[-2]].replace(0, np.nan)
                    ) * 100
                
                Logger.info("Volume features created")
            
            monitor.checkpoint('volume')
            
            # ========== PERFORMANCE SCORE ==========
            if 'Performans_Skoru' not in df.columns:
                score_features = []
                
                if 'CAGR' in df.columns:
                    score_features.append('CAGR')
                if growth_cols:
                    score_features.append(growth_cols[-1])
                if 'Pazar_Payi' in df.columns:
                    score_features.append('Pazar_Payi')
                
                if score_features:
                    try:
                        scaler = StandardScaler()
                        score_data = df[score_features].fillna(0)
                        scaled_scores = scaler.fit_transform(score_data)
                        df['Performans_Skoru'] = scaled_scores.mean(axis=1)
                        
                        # Normalize to 0-100
                        score_min = df['Performans_Skoru'].min()
                        score_max = df['Performans_Skoru'].max()
                        
                        if score_max != score_min:
                            df['Performans_Skoru_100'] = (
                                (df['Performans_Skoru'] - score_min) / 
                                (score_max - score_min)
                            ) * 100
                        
                        Logger.info("Performance score calculated")
                    except Exception as e:
                        Logger.warning(f"Performance score calculation failed: {str(e)}")
            
            monitor.checkpoint('performance')
            
            # ========== CLASSIFICATIONS ==========
            # Growth category
            if growth_cols and 'Buyume_Kategori' not in df.columns:
                last_growth = growth_cols[-1]
                df['Buyume_Kategori'] = pd.cut(
                    df[last_growth],
                    bins=[-np.inf, -20, -5, 5, 20, np.inf],
                    labels=['Ciddi Dusuş', 'Dusuş', 'Stabil', 'Buyume', 'Yuksek Buyume']
                )
            
            # Price tier
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols and 'Fiyat_Tier' not in df.columns:
                try:
                    df['Fiyat_Tier'] = pd.qcut(
                        df[price_cols[-1]].dropna(),
                        q=5,
                        labels=['Ekonomi', 'Dusuk', 'Orta', 'Yuksek', 'Premium'],
                        duplicates='drop'
                    )
                except:
                    pass
            
            # Market position
            if 'Pazar_Payi' in df.columns and 'Pazar_Pozisyon' not in df.columns:
                try:
                    df['Pazar_Pozisyon'] = pd.cut(
                        df['Pazar_Payi'],
                        bins=[0, 0.1, 0.5, 1, 5, 100],
                        labels=['Niche', 'Kucuk', 'Orta', 'Buyuk', 'Lider']
                    )
                except:
                    pass
            
            monitor.checkpoint('classifications')
            
            # ========== FINAL CLEANUP ==========
            # Remove any duplicate columns that may have been created
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Count new features
            new_features = len(df.columns) - len([c for c in df.columns if c in sales_cols.values()])
            
            perf_report = monitor.report()
            
            st.success(
                f"✅ **Feature Engineering Complete!**\n\n"
                f"🎯 {new_features} new features created\n\n"
                f"⏱️ Processing time: {perf_report['total_elapsed']:.2f}s"
            )
            
            Logger.info(f"Feature engineering complete: {new_features} features created in {perf_report['total_elapsed']:.2f}s")
            
            return df
            
        except Exception as e:
            Logger.error(f"Feature engineering failed: {str(e)}", e)
            st.warning(f"⚠️ Feature engineering partially completed: {str(e)}")
            return df
    
    @staticmethod
    def normalize_country_names(df: pd.DataFrame, country_column: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize country names for geographic analysis
        
        Args:
            df: Input DataFrame
            country_column: Name of country column (auto-detected if None)
        
        Returns:
            DataFrame with normalized country names
        """
        if country_column is None:
            for possible_name in ['Country', 'Ülke', 'Ulke', 'country', 'COUNTRY']:
                if possible_name in df.columns:
                    country_column = possible_name
                    break
        
        if country_column is None or country_column not in df.columns:
            return df
        
        # Comprehensive country name mapping
        country_mapping = {
            # United States
            'USA': 'United States', 'US': 'United States',
            'U.S.A': 'United States', 'U.S.A.': 'United States',
            'United States of America': 'United States',
            'U.S': 'United States', 'Amerika': 'United States',
            
            # United Kingdom
            'UK': 'United Kingdom', 'U.K': 'United Kingdom',
            'U.K.': 'United Kingdom', 'Great Britain': 'United Kingdom',
            'Ingiltere': 'United Kingdom',
            
            # UAE
            'UAE': 'United Arab Emirates',
            'U.A.E': 'United Arab Emirates',
            'Emirlikleri': 'United Arab Emirates',
            
            # Korea
            'S. Korea': 'South Korea',
            'South Korea': 'Korea, Republic of',
            'N. Korea': 'North Korea',
            'Guney Kore': 'Korea, Republic of',
            
            # Russia
            'Russia': 'Russian Federation',
            'Rusya': 'Russian Federation',
            
            # European countries
            'Almanya': 'Germany', 'Fransa': 'France',
            'Italya': 'Italy', 'Ispanya': 'Spain',
            'Yunanistan': 'Greece', 'Turkiye': 'Turkey',
            'Hollanda': 'Netherlands', 'Belcika': 'Belgium',
            'Avusturya': 'Austria', 'Isvicre': 'Switzerland',
            'Polonya': 'Poland', 'Portekiz': 'Portugal',
            
            # Asian countries
            'Cin': 'China', 'Japonya': 'Japan',
            'Hindistan': 'India', 'Endonezya': 'Indonesia',
            'Tayland': 'Thailand', 'Malezya': 'Malaysia',
            'Singapur': 'Singapore', 'Vietnam': 'Viet Nam',
            
            # Americas
            'Brezilya': 'Brazil', 'Arjantin': 'Argentina',
            'Meksika': 'Mexico', 'Kanada': 'Canada',
            'Sili': 'Chile', 'Kolombiya': 'Colombia',
            
            # Oceania
            'Avustralya': 'Australia',
            'Yeni Zelanda': 'New Zealand',
            
            # Middle East & Africa
            'Suudi Arabistan': 'Saudi Arabia',
            'Misir': 'Egypt', 'Guney Afrika': 'South Africa',
            'Israil': 'Israel', 'Iran': 'Iran, Islamic Republic of'
        }
        
        # Apply mapping
        df[country_column] = df[country_column].replace(country_mapping)
        
        Logger.info(f"Country names normalized in column: {country_column}")
        
        return df


# ============================================================================
# FILTER SYSTEM CLASS
# ============================================================================

class FilterSystem:
    """
    Advanced filtering system with multi-dimensional capabilities
    """
    
    @staticmethod
    def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """
        Find column by possible name variations
        
        Args:
            df: DataFrame to search
            possible_names: List of possible column names
        
        Returns:
            Found column name or None
        """
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    @staticmethod
    def create_filter_sidebar(df: pd.DataFrame) -> Tuple:
        """
        Create comprehensive filter sidebar interface
        
        Args:
            df: DataFrame to filter
        
        Returns:
            Tuple of (search_term, filters, apply_button, clear_button)
        """
        with st.sidebar.expander("🎯 ADVANCED FILTERING", expanded=True):
            st.markdown("""
            <div class="subsection-header">
                <h3>🔍 Filter Control Panel</h3>
                <p>Apply filters to refine your analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            filters = {}
            
            # ========== GLOBAL SEARCH ==========
            st.markdown("#### 🔎 Global Search")
            search_term = st.text_input(
                "Search across all columns",
                placeholder="Enter search term...",
                key="global_search",
                help="Search for any text across all columns"
            )
            
            st.markdown("---")
            
            # ========== CATEGORICAL FILTERS ==========
            st.markdown("#### 📑 Categorical Filters")
            
            # Country filter
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke', 'Ulke'])
            if country_col:
                countries = sorted(df[country_col].dropna().unique())
                if len(countries) > 0:
                    selected_countries = st.multiselect(
                        "🌍 Countries",
                        options=countries,
                        default=[],
                        key="country_filter",
                        help=f"Select from {len(countries)} countries"
                    )
                    if selected_countries:
                        filters[country_col] = selected_countries
            
            # Corporation filter
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket', 'Sirket', 'Company'])
            if corp_col:
                corporations = sorted(df[corp_col].dropna().unique())
                if len(corporations) > 0:
                    selected_corps = st.multiselect(
                        "🏢 Companies",
                        options=corporations,
                        default=[],
                        key="corp_filter",
                        help=f"Select from {len(corporations)} companies"
                    )
                    if selected_corps:
                        filters[corp_col] = selected_corps
            
            # Molecule filter
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül', 'Molekul', 'Product'])
            if mol_col:
                molecules = sorted(df[mol_col].dropna().unique())
                if len(molecules) > 0:
                    # Limit display for large lists
                    if len(molecules) > 100:
                        st.info(f"ℹ️ {len(molecules)} molecules available. Use search to filter.")
                        mol_search = st.text_input(
                            "🔍 Search molecules",
                            key="mol_search",
                            placeholder="Type to search..."
                        )
                        if mol_search:
                            molecules = [m for m in molecules if mol_search.lower() in str(m).lower()]
                    
                    selected_mols = st.multiselect(
                        "🧪 Molecules",
                        options=molecules[:100],  # Limit for performance
                        default=[],
                        key="mol_filter",
                        help="Select molecules to analyze"
                    )
                    if selected_mols:
                        filters[mol_col] = selected_mols
            
            st.markdown("---")
            
            # ========== NUMERICAL RANGE FILTERS ==========
            st.markdown("#### 📊 Numerical Filters")
            
            # Sales filter
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            if sales_cols:
                last_sales = sales_cols[-1]
                sales_data = df[last_sales].dropna()
                
                if len(sales_data) > 0:
                    min_val = float(sales_data.min())
                    max_val = float(sales_data.max())
                    
                    st.write(f"**Sales Range:** {Utils.format_currency(min_val)} - {Utils.format_currency(max_val)}")
                    
                    sales_range = st.slider(
                        "💰 Sales Filter",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        format="$%.0f",
                        key="sales_filter"
                    )
                    
                    if sales_range != (min_val, max_val):
                        filters['sales_range'] = (sales_range, last_sales)
            
            # Growth filter
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            if growth_cols:
                last_growth = growth_cols[-1]
                growth_data = df[last_growth].dropna()
                
                if len(growth_data) > 0:
                    min_growth = float(growth_data.min())
                    max_growth = float(growth_data.max())
                    
                    st.write(f"**Growth Range:** {min_growth:.1f}% - {max_growth:.1f}%")
                    
                    growth_range = st.slider(
                        "📈 Growth Filter (%)",
                        min_value=min(min_growth, -50.0),
                        max_value=max(max_growth, 100.0),
                        value=(min(min_growth, -50.0), max(max_growth, 100.0)),
                        format="%.1f%%",
                        key="growth_filter"
                    )
                    
                    if growth_range != (min_growth, max_growth):
                        filters['growth_range'] = (growth_range, last_growth)
            
            st.markdown("---")
            
            # ========== ADVANCED FILTERS ==========
            st.markdown("#### ⚙️ Advanced Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                positive_growth = st.checkbox(
                    "📈 Positive Growth Only",
                    value=False,
                    key="positive_growth_filter"
                )
                if positive_growth:
                    filters['positive_growth'] = True
                
                high_performers = st.checkbox(
                    "🏆 Top 10% Only",
                    value=False,
                    key="top_performers_filter"
                )
                if high_performers:
                    filters['top_performers'] = True
            
            with col2:
                exclude_outliers = st.checkbox(
                    "🎯 Exclude Outliers",
                    value=False,
                    key="exclude_outliers_filter"
                )
                if exclude_outliers:
                    filters['exclude_outliers'] = True
                
                min_market_share = st.checkbox(
                    "💎 Market Share > 1%",
                    value=False,
                    key="min_share_filter"
                )
                if min_market_share:
                    filters['min_market_share'] = True
            
            st.markdown("---")
            
            # ========== ACTION BUTTONS ==========
            col1, col2 = st.columns(2)
            
            with col1:
                apply_filters = st.button(
                    "✅ Apply Filters",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                clear_filters = st.button(
                    "🗑️ Clear All",
                    use_container_width=True
                )
        
        return search_term, filters, apply_filters, clear_filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, search_term: str, filters: Dict) -> pd.DataFrame:
        """
        Apply all filters to DataFrame
        
        Args:
            df: Input DataFrame
            search_term: Global search term
            filters: Dictionary of filter conditions
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        try:
            # Global search
            if search_term:
                search_mask = pd.Series(False, index=filtered_df.index)
                
                for col in filtered_df.columns:
                    try:
                        search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                            search_term, case=False, na=False, regex=False
                        )
                    except:
                        continue
                
                filtered_df = filtered_df[search_mask]
                Logger.info(f"Global search '{search_term}': {len(filtered_df)} / {original_count} rows")
            
            # Categorical filters
            for col, values in filters.items():
                if col in ['sales_range', 'growth_range', 'positive_growth', 'top_performers', 
                          'exclude_outliers', 'min_market_share']:
                    continue
                
                if col in filtered_df.columns and values:
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
            
            # Numerical range filters
            if 'sales_range' in filters:
                (min_val, max_val), col_name = filters['sales_range']
                if col_name in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[col_name] >= min_val) & 
                        (filtered_df[col_name] <= max_val)
                    ]
            
            if 'growth_range' in filters:
                (min_val, max_val), col_name = filters['growth_range']
                if col_name in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[col_name] >= min_val) & 
                        (filtered_df[col_name] <= max_val)
                    ]
            
            # Boolean filters
            if filters.get('positive_growth'):
                growth_cols = [col for col in filtered_df.columns if 'Buyume_' in col or 'Growth_' in col]
                if growth_cols:
                    filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
            
            if filters.get('top_performers'):
                sales_cols = [col for col in filtered_df.columns if 'Satis_' in col or 'Sales_' in col]
                if sales_cols:
                    threshold = filtered_df[sales_cols[-1]].quantile(0.9)
                    filtered_df = filtered_df[filtered_df[sales_cols[-1]] >= threshold]
            
            if filters.get('exclude_outliers'):
                sales_cols = [col for col in filtered_df.columns if 'Satis_' in col or 'Sales_' in col]
                if sales_cols:
                    Q1 = filtered_df[sales_cols[-1]].quantile(0.25)
                    Q3 = filtered_df[sales_cols[-1]].quantile(0.75)
                    IQR = Q3 - Q1
                    filtered_df = filtered_df[
                        (filtered_df[sales_cols[-1]] >= Q1 - 1.5 * IQR) &
                        (filtered_df[sales_cols[-1]] <= Q3 + 1.5 * IQR)
                    ]
            
            if filters.get('min_market_share') and 'Pazar_Payi' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Pazar_Payi'] >= 1.0]
            
            Logger.info(f"Filters applied: {len(filtered_df)} / {original_count} rows remaining")
            
            return filtered_df
            
        except Exception as e:
            Logger.error(f"Filter application failed: {str(e)}", e)
            st.error(f"❌ Filter error: {str(e)}")
            return df


# ============================================================================
# ANALYTICS ENGINE CLASS
# ============================================================================

class AnalyticsEngine:
    """
    Comprehensive pharmaceutical market analytics engine
    """
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive market metrics
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            Logger.info("Calculating comprehensive metrics")
            
            # Basic metrics
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            metrics['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
            
            # Find sales columns
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if sales_cols:
                last_sales = sales_cols[-1]
                year_match = re.search(r'(\d{4})', last_sales)
                metrics['last_year'] = year_match.group(1) if year_match else 'N/A'
                
                # Sales metrics
                metrics['total_market_value'] = float(df[last_sales].sum())
                metrics['avg_sales'] = float(df[last_sales].mean())
                metrics['median_sales'] = float(df[last_sales].median())
                metrics['std_sales'] = float(df[last_sales].std())
                metrics['min_sales'] = float(df[last_sales].min())
                metrics['max_sales'] = float(df[last_sales].max())
                
                # Sales distribution
                metrics['sales_q1'] = float(df[last_sales].quantile(0.25))
                metrics['sales_q3'] = float(df[last_sales].quantile(0.75))
                metrics['sales_iqr'] = metrics['sales_q3'] - metrics['sales_q1']
                metrics['sales_cv'] = (metrics['std_sales'] / metrics['avg_sales'] * 100) if metrics['avg_sales'] > 0 else 0
            
            # Growth metrics
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            
            if growth_cols:
                last_growth = growth_cols[-1]
                growth_data = df[last_growth].dropna()
                
                metrics['avg_growth'] = float(growth_data.mean())
                metrics['median_growth'] = float(growth_data.median())
                metrics['std_growth'] = float(growth_data.std())
                metrics['positive_growth_count'] = int((growth_data > 0).sum())
                metrics['negative_growth_count'] = int((growth_data < 0).sum())
                metrics['high_growth_count'] = int((growth_data > 20).sum())
                metrics['decline_count'] = int((growth_data < -10).sum())
                metrics['positive_growth_pct'] = (metrics['positive_growth_count'] / len(growth_data)) * 100
            
            # Market structure
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket', 'Sirket'])
            
            if corp_col and sales_cols:
                corp_sales = df.groupby(corp_col)[last_sales].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    
                    # HHI (Herfindahl-Hirschman Index)
                    metrics['hhi_index'] = float((market_shares ** 2).sum())
                    
                    # Concentration ratios
                    for n in [1, 3, 4, 5, 10]:
                        if len(corp_sales) >= n:
                            metrics[f'top_{n}_share'] = float(corp_sales.nlargest(n).sum() / total_sales * 100)
                    
                    metrics['cr4'] = metrics.get('top_4_share', 0)
                    metrics['num_competitors'] = len(corp_sales)
                    metrics['effective_competitors'] = 10000 / metrics['hhi_index'] if metrics['hhi_index'] > 0 else 0
                    
                    # Gini coefficient
                    metrics['gini_coefficient'] = AnalyticsEngine.calculate_gini(market_shares)
            
            # Geographic metrics
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke', 'Ulke'])
            
            if country_col:
                metrics['country_coverage'] = df[country_col].nunique()
                
                if sales_cols:
                    country_sales = df.groupby(country_col)[last_sales].sum()
                    total_country_sales = country_sales.sum()
                    
                    if total_country_sales > 0:
                        metrics['top_5_country_share'] = float(country_sales.nlargest(5).sum() / total_country_sales * 100)
                        metrics['top_country'] = country_sales.idxmax()
                        metrics['top_country_share'] = float(country_sales.max() / total_country_sales * 100)
            
            # Molecule diversity
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül', 'Molekul'])
            
            if mol_col:
                metrics['unique_molecules'] = df[mol_col].nunique()
                
                if sales_cols:
                    mol_sales = df.groupby(mol_col)[last_sales].sum()
                    total_mol_sales = mol_sales.sum()
                    
                    if total_mol_sales > 0:
                        metrics['top_10_molecule_share'] = float(mol_sales.nlargest(10).sum() / total_mol_sales * 100)
                        metrics['top_molecule'] = mol_sales.idxmax()
                        metrics['top_molecule_sales'] = float(mol_sales.max())
            
            # Price metrics
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            
            if price_cols:
                last_price = price_cols[-1]
                price_data = df[last_price].dropna()
                
                metrics['avg_price'] = float(price_data.mean())
                metrics['median_price'] = float(price_data.median())
                metrics['std_price'] = float(price_data.std())
                metrics['min_price'] = float(price_data.min())
                metrics['max_price'] = float(price_data.max())
                metrics['price_cv'] = (metrics['std_price'] / metrics['avg_price'] * 100) if metrics['avg_price'] > 0 else 0
            
            # CAGR metrics
            if 'CAGR' in df.columns:
                cagr_data = df['CAGR'].dropna()
                metrics['avg_cagr'] = float(cagr_data.mean())
                metrics['median_cagr'] = float(cagr_data.median())
                metrics['positive_cagr_pct'] = float((cagr_data > 0).sum() / len(cagr_data) * 100)
            
            # Market share metrics
            if 'Pazar_Payi' in df.columns:
                share_data = df['Pazar_Payi'].dropna()
                metrics['avg_market_share'] = float(share_data.mean())
                metrics['median_market_share'] = float(share_data.median())
                metrics['gini_market_share'] = AnalyticsEngine.calculate_gini(share_data)
            
            # Performance metrics
            if 'Performans_Skoru_100' in df.columns:
                perf_data = df['Performans_Skoru_100'].dropna()
                metrics['avg_performance'] = float(perf_data.mean())
                metrics['high_performers'] = int((perf_data > 75).sum())
                metrics['low_performers'] = int((perf_data < 25).sum())
            
            Logger.info(f"Calculated {len(metrics)} metrics")
            
            return metrics
            
        except Exception as e:
            Logger.error(f"Metrics calculation failed: {str(e)}", e)
            return metrics
    
    @staticmethod
    def calculate_gini(data: pd.Series) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        try:
            data = data.dropna()
            if len(data) == 0:
                return 0.0
            
            sorted_data = np.sort(data)
            n = len(sorted_data)
            cumsum = np.cumsum(sorted_data)
            
            return float((2 * np.sum((np.arange(1, n+1)) * sorted_data)) / (n * cumsum[-1]) - (n + 1) / n)
            
        except:
            return 0.0
    
    @staticmethod
    def generate_strategic_insights(df: pd.DataFrame, metrics: Dict) -> List[Dict]:
        """
        Generate strategic business insights from data
        
        Args:
            df: DataFrame
            metrics: Calculated metrics
        
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            if not sales_cols:
                return insights
            
            last_sales = sales_cols[-1]
            year = metrics.get('last_year', 'N/A')
            
            # Top products insight
            top_10 = df.nlargest(10, last_sales)
            top_share = (top_10[last_sales].sum() / df[last_sales].sum() * 100)
            
            insights.append({
                'type': 'success',
                'title': f'🏆 Top 10 Product Concentration - {year}',
                'description': f"Top 10 products account for {top_share:.1f}% of total market ({Utils.format_currency(top_10[last_sales].sum())}). "
                              f"Average: {Utils.format_currency(top_10[last_sales].mean())}"
            })
            
            # Growth leaders
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth = growth_cols[-1]
                top_growth = df.nlargest(10, last_growth)
                avg_top_growth = top_growth[last_growth].mean()
                
                insights.append({
                    'type': 'info',
                    'title': '🚀 Growth Champions',
                    'description': f"Top 10 fastest-growing products show {avg_top_growth:.1f}% average growth "
                                  f"(market average: {metrics.get('avg_growth', 0):.1f}%). "
                                  f"Combined sales: {Utils.format_currency(top_growth[last_sales].sum())}"
                })
                
                # Decline warning
                decline_count = metrics.get('decline_count', 0)
                if decline_count > 0:
                    decline_pct = (decline_count / len(df)) * 100
                    declining_df = df[df[last_growth] < -10]
                    declining_sales = declining_df[last_sales].sum()
                    
                    insights.append({
                        'type': 'warning',
                        'title': '⚠️ Products in Decline',
                        'description': f"{decline_count:,} products ({decline_pct:.1f}%) showing >10% decline. "
                                      f"At-risk sales: {Utils.format_currency(declining_sales)}. Requires strategic review."
                    })
            
            # Market structure
            hhi = metrics.get('hhi_index', 0)
            if hhi > 0:
                if hhi > 2500:
                    structure = "Highly Concentrated (Monopoly/Oligopoly)"
                    risk = "Limited competition, potential for market power abuse"
                elif hhi > 1800:
                    structure = "Moderately Concentrated (Oligopoly)"
                    risk = "Competitive but concentrated, moderate entry barriers"
                else:
                    structure = "Low Concentration (Competitive)"
                    risk = "High competition, potential price pressure"
                
                insights.append({
                    'type': 'info',
                    'title': '📊 Market Structure Analysis',
                    'description': f"HHI: {hhi:.0f} - {structure}. {risk}. "
                                  f"Effective competitors: {metrics.get('effective_competitors', 0):.1f}"
                })
            
            # Geographic concentration
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
            if country_col:
                top_country = metrics.get('top_country', 'N/A')
                country_share = metrics.get('top_country_share', 0)
                
                insights.append({
                    'type': 'info',
                    'title': f'🌍 Geographic Leader - {year}',
                    'description': f"{top_country} dominates with {country_share:.1f}% market share. "
                                  f"Operating in {metrics.get('country_coverage', 0)} countries. "
                                  f"Top 5 countries: {metrics.get('top_5_country_share', 0):.1f}%"
                })
            
            # Price analysis
            avg_price = metrics.get('avg_price', 0)
            price_cv = metrics.get('price_cv', 0)
            
            if avg_price > 0:
                price_tier = "Premium" if avg_price > 100 else "Mid-range" if avg_price > 10 else "Economy"
                
                insights.append({
                    'type': 'info',
                    'title': f'💰 Pricing Profile - {year}',
                    'description': f"Average price: {Utils.format_currency(avg_price)} ({price_tier} segment). "
                                  f"Price variation: {price_cv:.1f}% CV. "
                                  f"Range: {Utils.format_currency(metrics.get('min_price', 0))} - "
                                  f"{Utils.format_currency(metrics.get('max_price', 0))}"
                })
            
            # CAGR insight
            avg_cagr = metrics.get('avg_cagr', 0)
            if avg_cagr != 0:
                cagr_status = "strong growth" if avg_cagr > 10 else "moderate growth" if avg_cagr > 0 else "declining"
                
                insights.append({
                    'type': 'success' if avg_cagr > 10 else 'warning' if avg_cagr < 0 else 'info',
                    'title': '📈 Long-term Growth Trend (CAGR)',
                    'description': f"Average CAGR: {avg_cagr:.1f}% - {cagr_status} trajectory. "
                                  f"{metrics.get('positive_cagr_pct', 0):.1f}% of products show positive CAGR."
                })
            
            # Opportunity identification
            if growth_cols and 'Pazar_Payi' in df.columns:
                last_growth = growth_cols[-1]
                opportunities = df[(df[last_growth] > 20) & (df['Pazar_Payi'] < 1)]
                
                if len(opportunities) > 0:
                    opp_sales = opportunities[last_sales].sum()
                    
                    insights.append({
                        'type': 'success',
                        'title': '💎 High-Growth Opportunities',
                        'description': f"{len(opportunities)} products show >20% growth with <1% market share. "
                                      f"Investment opportunity: {Utils.format_currency(opp_sales)}. "
                                      f"High potential for market expansion."
                    })
            
            # Risk identification
            if growth_cols and 'Pazar_Payi' in df.columns:
                last_growth = growth_cols[-1]
                risks = df[(df[last_growth] < -5) & (df['Pazar_Payi'] > 5)]
                
                if len(risks) > 0:
                    risk_sales = risks[last_sales].sum()
                    risk_share = (risk_sales / df[last_sales].sum()) * 100
                    
                    insights.append({
                        'type': 'danger',
                        'title': '🚨 Strategic Risk Alert',
                        'description': f"{len(risks)} major products declining >5% despite >5% market share. "
                                      f"At-risk revenue: {Utils.format_currency(risk_sales)} ({risk_share:.1f}%). "
                                      f"Immediate intervention required."
                    })
            
            Logger.info(f"Generated {len(insights)} strategic insights")
            
            return insights
            
        except Exception as e:
            Logger.error(f"Insight generation failed: {str(e)}", e)
            return insights


# Continue with remaining code in next part...
# MLEngine, Visualizer, and Main Application coming next


# ============================================================================
# MACHINE LEARNING ENGINE - ADVANCED
# ============================================================================

class MLEngine:
    """
    Advanced machine learning engine for pharmaceutical analytics
    """
    
    @staticmethod
    def train_forecasting_model(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        forecast_years: int = 3,
        model_type: str = 'rf'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Train advanced forecasting model
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature columns
            forecast_years: Number of years to forecast
            model_type: Model type ('rf', 'gbm', 'ridge', 'lasso')
        
        Returns:
            Tuple of (results_dict, error_message)
        """
        try:
            Logger.info(f"Training {model_type} forecasting model")
            
            # Prepare data
            ml_data = df[feature_cols + [target_col]].dropna()
            
            if len(ml_data) < Config.ML_MIN_SAMPLES:
                return None, f"Insufficient data: {len(ml_data)} rows (minimum {Config.ML_MIN_SAMPLES} required)"
            
            X = ml_data[feature_cols]
            y = ml_data[target_col]
            
            # Train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=Config.RANDOM_STATE
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and configure model
            if model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS
                )
            elif model_type == 'gbm':
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=Config.RANDOM_STATE
                )
            elif model_type == 'ridge':
                model = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
            elif model_type == 'lasso':
                model = Lasso(alpha=0.1, random_state=Config.RANDOM_STATE)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_STATE)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            # MAPE
            try:
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
            except:
                mape_test = 0
            
            # Cross-validation
            try:
                kf = KFold(n_splits=min(Config.CV_FOLDS, len(X)//10), shuffle=True, random_state=Config.RANDOM_STATE)
                cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=Config.N_JOBS)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = r2_test
                cv_std = 0
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(feature_cols, np.abs(model.coef_)))
            else:
                feature_importance = {f: 1/len(feature_cols) for f in feature_cols}
            
            # Future forecasting
            forecast = []
            year_match = re.search(r'(\d{4})', target_col)
            base_year = int(year_match.group(1)) if year_match else 2024
            
            for year_offset in range(1, forecast_years + 1):
                future_year = base_year + year_offset
                
                # Use mean of features for forecast (simple approach)
                future_X = X.mean(axis=0).values.reshape(1, -1)
                future_X_scaled = scaler.transform(future_X)
                future_pred = model.predict(future_X_scaled)[0]
                
                # Confidence intervals (based on RMSE)
                confidence_low = future_pred - 1.96 * rmse_test
                confidence_high = future_pred + 1.96 * rmse_test
                
                forecast.append({
                    'year': str(future_year),
                    'prediction': float(future_pred),
                    'confidence_low': float(max(0, confidence_low)),
                    'confidence_high': float(confidence_high)
                })
            
            results = {
                'model': model,
                'scaler': scaler,
                'model_type': model_type,
                'mae_train': float(mae_train),
                'mae_test': float(mae_test),
                'rmse_train': float(rmse_train),
                'rmse_test': float(rmse_test),
                'r2_train': float(r2_train),
                'r2_test': float(r2_test),
                'mape_test': float(mape_test),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'forecast': forecast,
                'feature_importance': {k: float(v) for k, v in feature_importance.items()},
                'train_size': len(X_train),
                'test_size': len(X_test),
                'predictions': {
                    'y_test': y_test.values.tolist(),
                    'y_pred_test': y_pred_test.tolist()
                }
            }
            
            Logger.info(f"Model trained successfully: R²={r2_test:.3f}, RMSE={rmse_test:.2f}")
            
            return results, None
            
        except Exception as e:
            Logger.error(f"Model training failed: {str(e)}", e)
            return None, f"Training error: {str(e)}"
    
    @staticmethod
    def perform_clustering(
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 4,
        algorithm: str = 'kmeans'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Perform advanced clustering analysis
        
        Args:
            df: Input DataFrame
            feature_cols: Features for clustering
            n_clusters: Number of clusters
            algorithm: Algorithm ('kmeans', 'hierarchical', 'dbscan')
        
        Returns:
            Tuple of (results_dict, error_message)
        """
        try:
            Logger.info(f"Performing {algorithm} clustering with {n_clusters} clusters")
            
            # Prepare data
            cluster_data = df[feature_cols].fillna(0)
            
            if len(cluster_data) < Config.ML_MIN_SAMPLES_CLUSTERING:
                return None, f"Insufficient data: {len(cluster_data)} rows (minimum {Config.ML_MIN_SAMPLES_CLUSTERING} required)"
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    init='k-means++',
                    n_init=20,
                    max_iter=500,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS
                )
            elif algorithm == 'dbscan':
                clusterer = DBSCAN(
                    eps=0.5,
                    min_samples=max(10, len(cluster_data)//100),
                    n_jobs=Config.N_JOBS
                )
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE)
            
            # Fit and predict
            clusters = clusterer.fit_predict(scaled_data)
            
            # Calculate quality metrics
            silhouette = silhouette_score(scaled_data, clusters)
            calinski = calinski_harabasz_score(scaled_data, clusters)
            davies = davies_bouldin_score(scaled_data, clusters)
            
            # PCA for visualization
            n_components = min(3, len(feature_cols))
            pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
            pca_data = pca.fit_transform(scaled_data)
            
            # Cluster statistics
            cluster_stats = {}
            unique_clusters = np.unique(clusters)
            
            for i in unique_clusters:
                cluster_mask = clusters == i
                cluster_stats[int(i)] = {
                    'size': int(cluster_mask.sum()),
                    'percentage': float((cluster_mask.sum() / len(clusters)) * 100),
                    'mean_features': df[feature_cols][cluster_mask].mean().to_dict()
                }
            
            # Assign meaningful names
            cluster_names = {
                0: 'Emerging Stars',
                1: 'Mature Cash Cows',
                2: 'Innovation Leaders',
                3: 'Declining Products',
                4: 'Niche Players',
                5: 'Volume Products',
                6: 'Premium Segment',
                7: 'Economy Segment',
                8: 'Growth Champions',
                9: 'At Risk'
            }
            
            cluster_labels = [cluster_names.get(c, f'Cluster {c}') for c in clusters]
            
            results = {
                'clusters': clusters.tolist(),
                'cluster_labels': cluster_labels,
                'silhouette_score': float(silhouette),
                'calinski_score': float(calinski),
                'davies_bouldin_score': float(davies),
                'pca_data': pca_data.tolist(),
                'pca_variance': pca.explained_variance_ratio_.tolist(),
                'cluster_stats': cluster_stats,
                'n_clusters': len(unique_clusters),
                'algorithm': algorithm,
                'feature_cols': feature_cols
            }
            
            if algorithm == 'kmeans':
                results['centers'] = clusterer.cluster_centers_.tolist()
                results['inertia'] = float(clusterer.inertia_)
            
            Logger.info(f"Clustering complete: {len(unique_clusters)} clusters, silhouette={silhouette:.3f}")
            
            return results, None
            
        except Exception as e:
            Logger.error(f"Clustering failed: {str(e)}", e)
            return None, f"Clustering error: {str(e)}"
    
    @staticmethod
    def detect_anomalies(
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            df: Input DataFrame
            feature_cols: Features for anomaly detection
            contamination: Expected proportion of outliers
        
        Returns:
            Tuple of (results_dict, error_message)
        """
        try:
            Logger.info(f"Detecting anomalies with contamination={contamination}")
            
            # Prepare data
            anomaly_data = df[feature_cols].fillna(0)
            
            if len(anomaly_data) < Config.ML_MIN_SAMPLES_ANOMALY:
                return None, f"Insufficient data: {len(anomaly_data)} rows"
            
            # Scale features
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(anomaly_data)
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                max_samples='auto',
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            )
            
            predictions = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            # Identify anomalies
            is_anomaly = predictions == -1
            anomaly_count = int(is_anomaly.sum())
            anomaly_percentage = float((anomaly_count / len(predictions)) * 100)
            
            # Severity classification
            severity = []
            score_percentiles = np.percentile(anomaly_scores, [5, 10, 25])
            
            for score in anomaly_scores:
                if score < score_percentiles[0]:
                    severity.append('Critical')
                elif score < score_percentiles[1]:
                    severity.append('High')
                elif score < score_percentiles[2]:
                    severity.append('Medium')
                else:
                    severity.append('Low')
            
            results = {
                'is_anomaly': is_anomaly.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'severity': severity,
                'contamination': contamination,
                'feature_cols': feature_cols
            }
            
            Logger.info(f"Anomaly detection complete: {anomaly_count} anomalies ({anomaly_percentage:.1f}%)")
            
            return results, None
            
        except Exception as e:
            Logger.error(f"Anomaly detection failed: {str(e)}", e)
            return None, f"Detection error: {str(e)}"


# ============================================================================
# VISUALIZATION ENGINE - PROFESSIONAL
# ============================================================================

class Visualizer:
    """
    Professional visualization engine with publication-quality charts
    """
    
    @staticmethod
    def create_kpi_dashboard(df: pd.DataFrame, metrics: Dict) -> None:
        """Create comprehensive KPI dashboard"""
        try:
            # Row 1: Primary KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = metrics.get('total_market_value', 0)
                year = metrics.get('last_year', '')
                growth = metrics.get('avg_growth', 0)
                
                delta_class = "positive" if growth > 0 else "negative" if growth < 0 else "neutral"
                delta_icon = "↑" if growth > 0 else "↓" if growth < 0 else "→"
                
                st.markdown(f"""
                <div class="kpi-card fade-in">
                    <div class="kpi-title">Total Market Value</div>
                    <div class="kpi-value">{Utils.format_currency(total_value)}</div>
                    <div class="kpi-subtitle">{year} Global Market</div>
                    <span class="kpi-delta {delta_class}">
                        {delta_icon} {abs(growth):.1f}% YoY
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('avg_growth', 0)
                median_growth = metrics.get('median_growth', 0)
                
                st.markdown(f"""
                <div class="kpi-card fade-in" style="animation-delay: 0.1s;">
                    <div class="kpi-title">Average Growth Rate</div>
                    <div class="kpi-value">{Utils.format_percentage(avg_growth, include_sign=False)}</div>
                    <div class="kpi-subtitle">Median: {Utils.format_percentage(median_growth, include_sign=False)}</div>
                    <span class="metric-badge {'success' if avg_growth > 10 else 'warning' if avg_growth > 0 else 'danger'}">
                        {'Strong Growth' if avg_growth > 10 else 'Moderate' if avg_growth > 0 else 'Declining'}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('hhi_index', 0)
                eff_comp = metrics.get('effective_competitors', 0)
                
                if hhi > 2500:
                    status = "Monopoly"
                    status_class = "danger"
                elif hhi > 1500:
                    status = "Oligopoly"
                    status_class = "warning"
                else:
                    status = "Competitive"
                    status_class = "success"
                
                st.markdown(f"""
                <div class="kpi-card fade-in" style="animation-delay: 0.2s;">
                    <div class="kpi-title">Market Structure (HHI)</div>
                    <div class="kpi-value">{hhi:.0f}</div>
                    <div class="kpi-subtitle">Effective Competitors: {eff_comp:.1f}</div>
                    <span class="metric-badge {status_class}">
                        {status}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                molecules = metrics.get('unique_molecules', 0)
                top_10_share = metrics.get('top_10_molecule_share', 0)
                
                st.markdown(f"""
                <div class="kpi-card fade-in" style="animation-delay: 0.3s;">
                    <div class="kpi-title">Product Diversity</div>
                    <div class="kpi-value">{molecules:,}</div>
                    <div class="kpi-subtitle">Unique Molecules</div>
                    <span class="kpi-delta neutral">
                        Top 10: {Utils.format_percentage(top_10_share, include_sign=False)}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 2: Secondary metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    "Countries",
                    f"{metrics.get('country_coverage', 0)}",
                    f"Top 5: {metrics.get('top_5_country_share', 0):.1f}%"
                )
            
            with col6:
                st.metric(
                    "Average Price",
                    Utils.format_currency(metrics.get('avg_price', 0)),
                    f"CV: {metrics.get('price_cv', 0):.1f}%"
                )
            
            with col7:
                pos_growth_pct = metrics.get('positive_growth_pct', 0)
                st.metric(
                    "Growing Products",
                    f"{pos_growth_pct:.1f}%",
                    f"{metrics.get('high_growth_count', 0)} >20%"
                )
            
            with col8:
                st.metric(
                    "Competitors",
                    f"{metrics.get('num_competitors', 0)}",
                    f"CR4: {metrics.get('cr4', 0):.1f}%"
                )
            
        except Exception as e:
            Logger.error(f"KPI dashboard error: {str(e)}", e)
            st.error("❌ KPI dashboard error")


# Main application code continues...
# (Adding more visualization methods and main app in next section)



# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Professional header
    st.markdown("""
    <div class="section-header">
        <h1>💊 PHARMAINTELLIGENCE PRO</h1>
        <p>Enterprise Pharmaceutical Analytics Platform with Advanced Machine Learning</p>
        <div class="version-badge">
            Version 7.0 Professional Edition | Build: 2024-02-04
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="section-header" style="padding: 1.5rem; margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.5rem;">📁 Data Upload</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Excel or CSV File",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your pharmaceutical market data"
        )
        
        if uploaded_file:
            if st.button("🚀 Load & Process Data", type="primary", use_container_width=True):
                with st.spinner("⏳ Processing data..."):
                    df = DataManager.load_data(uploaded_file)
                    
                    if df is not None:
                        df = DataManager.normalize_country_names(df)
                        df = DataManager.prepare_analysis_data(df)
                        
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(df)
                        st.session_state.insights = AnalyticsEngine.generate_strategic_insights(
                            df, st.session_state.metrics
                        )
                        
                        st.balloons()
                        st.success(f"✅ Analysis complete! {len(df):,} rows ready.")
                        time.sleep(1)
                        st.rerun()
        
        st.markdown("---")
        
        if st.session_state.data is not None:
            st.markdown("""
            <div class="success-box">
                <h4 style="margin: 0 0 0.5rem 0;">✅ Data Loaded</h4>
            </div>
            """, unsafe_allow_html=True)
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Rows", f"{len(st.session_state.data):,}")
            with info_col2:
                st.metric("Columns", len(st.session_state.data.columns))
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    df = st.session_state.data
    
    # Filters
    search_term, filters, apply_filters, clear_filters = FilterSystem.create_filter_sidebar(df)
    
    if apply_filters:
        with st.spinner("🔄 Applying filters..."):
            filtered_df = FilterSystem.apply_filters(df, search_term, filters)
            st.session_state.filtered_data = filtered_df
            
            st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(filtered_df)
            st.session_state.insights = AnalyticsEngine.generate_strategic_insights(
                filtered_df, st.session_state.metrics
            )
            
            st.success(f"✅ Filters applied: {len(filtered_df):,} / {len(df):,} rows")
            time.sleep(0.5)
            st.rerun()
    
    if clear_filters:
        st.session_state.filtered_data = df.copy()
        st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(df)
        st.session_state.insights = AnalyticsEngine.generate_strategic_insights(df, st.session_state.metrics)
        st.success("✅ Filters cleared")
        time.sleep(0.5)
        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 OVERVIEW & INSIGHTS",
        "🤖 ML LABORATORY",
        "🎯 SEGMENTATION",
        "📈 ADVANCED ANALYTICS",
        "📑 REPORTS & EXPORT"
    ])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_ml_lab_tab()
    
    with tab3:
        show_segmentation_tab()
    
    with tab4:
        show_advanced_analytics_tab()
    
    with tab5:
        show_reporting_tab()


def show_welcome_screen():
    """Professional welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem;">
        <div style="font-size: 6rem; margin-bottom: 1.5rem;">💊</div>
        <h1 style="color: #2acaea; font-size: 3rem; margin-bottom: 1rem;">
            Welcome to PharmaIntelligence Pro
        </h1>
        <p style="color: #cbd5e1; font-size: 1.3rem; max-width: 900px; margin: 0 auto 3rem auto; line-height: 1.8;">
            Enterprise-grade pharmaceutical analytics platform powered by advanced machine learning.
            Transform your market data into actionable strategic insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown('<div class="feature-showcase">', unsafe_allow_html=True)
    
    features = [
        ("🤖", "AI Forecasting", "Random Forest & Gradient Boosting models for accurate sales predictions"),
        ("🎯", "Smart Clustering", "K-Means & Hierarchical clustering for market segmentation"),
        ("⚠️", "Anomaly Detection", "Isolation Forest algorithm for outlier identification"),
        ("📊", "Market Analytics", "Comprehensive HHI, CAGR, and competitive analysis"),
        ("🌍", "Geographic Intelligence", "Country-level market analysis and visualization"),
        ("💡", "Strategic Insights", "AI-powered business intelligence and recommendations"),
        ("📈", "Advanced Metrics", "Performance scoring, momentum tracking, and trend analysis"),
        ("📑", "Export & Reports", "Professional reports in CSV, JSON, Excel, and HTML")
    ]
    
    cols = st.columns(4)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="feature-card scale-in" style="animation-delay: {idx * 0.1}s;">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div class="info-box" style="max-width: 900px; margin: 3rem auto;">
        <h3 style="margin: 0 0 1.5rem 0; color: #60a5fa; font-size: 1.5rem;">🚀 Getting Started</h3>
        <ol style="color: #cbd5e1; line-height: 2.5; font-size: 1.1rem; padding-left: 1.5rem;">
            <li><strong>Upload Your Data:</strong> Use the sidebar to upload CSV or Excel files</li>
            <li><strong>Automatic Processing:</strong> AI will analyze and enrich your data</li>
            <li><strong>Explore Insights:</strong> Navigate through tabs for different analyses</li>
            <li><strong>Apply Filters:</strong> Refine your analysis with advanced filtering</li>
            <li><strong>Export Results:</strong> Download reports and processed data</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def show_overview_tab():
    """Overview tab with KPIs and insights"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    st.markdown('<div class="subsection-header"><h3>📊 Market Overview & Key Performance Indicators</h3></div>', 
                unsafe_allow_html=True)
    
    Visualizer.create_kpi_dashboard(df, metrics)
    
    st.markdown('<div class="subsection-header"><h3>💡 Strategic Insights & Recommendations</h3></div>', 
                unsafe_allow_html=True)
    
    if insights:
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="insight-card {insight.get('type', 'info')} fade-in" style="animation-delay: {idx * 0.1}s;">
                    <h4>{insight['title']}</h4>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header"><h3>📋 Data Preview</h3></div>', 
                unsafe_allow_html=True)
    
    st.dataframe(df.head(100), use_container_width=True, height=400)


def show_ml_lab_tab():
    """ML Laboratory tab"""
    df = st.session_state.filtered_data
    
    st.markdown("""
    <div class="subsection-header">
        <h3>🤖 Machine Learning Laboratory</h3>
        <p>Advanced AI-powered analytics and predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    ml_tabs = st.tabs([
        "📈 Sales Forecasting",
        "🎯 Product Clustering",
        "⚠️ Anomaly Detection"
    ])
    
    with ml_tabs[0]:
        show_forecasting_panel(df)
    
    with ml_tabs[1]:
        show_clustering_panel(df)
    
    with ml_tabs[2]:
        show_anomaly_panel(df)


def show_forecasting_panel(df):
    """Forecasting panel with enhanced UI"""
    st.markdown("### 📈 Sales Forecasting Engine")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
    
    if len(sales_cols) < 2:
        st.warning("⚠️ At least 2 years of sales data required for forecasting")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_features = sales_cols[:-1] + growth_cols + price_cols
        selected_features = st.multiselect(
            "📊 Select Features",
            available_features,
            default=available_features[:min(3, len(available_features))],
            help="Choose features for model training"
        )
    
    with col2:
        model_type = st.selectbox(
            "🤖 Model Type",
            ["Random Forest", "Gradient Boosting", "Ridge Regression", "Lasso Regression"],
            help="Select forecasting algorithm"
        )
    
    with col3:
        forecast_years = st.slider(
            "📅 Forecast Horizon",
            Config.MIN_FORECAST_YEARS,
            Config.MAX_FORECAST_YEARS,
            Config.DEFAULT_FORECAST_YEARS,
            help="Number of years to forecast"
        )
    
    if st.button("🚀 Train Model & Forecast", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Please select at least one feature!")
            return
        
        with st.spinner("🔄 Training model... This may take a moment."):
            model_map = {
                "Random Forest": "rf",
                "Gradient Boosting": "gbm",
                "Ridge Regression": "ridge",
                "Lasso Regression": "lasso"
            }
            
            results, error = MLEngine.train_forecasting_model(
                df,
                target_col=sales_cols[-1],
                feature_cols=selected_features,
                forecast_years=forecast_years,
                model_type=model_map[model_type]
            )
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.success("✅ Model trained successfully!")
                
                # Performance metrics
                st.markdown("#### 📊 Model Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("R² Score", f"{results['r2_test']:.3f}", "Test Set")
                
                with metric_col2:
                    st.metric("MAE", Utils.format_currency(results['mae_test']), "Mean Abs Error")
                
                with metric_col3:
                    st.metric("RMSE", Utils.format_currency(results['rmse_test']), "Root Mean Sq")
                
                with metric_col4:
                    st.metric("CV Score", f"{results['cv_mean']:.3f}", f"±{results['cv_std']:.3f}")
                
                # Forecast visualization
                st.markdown("#### 📈 Forecast Results")
                
                forecast_df = pd.DataFrame(results['forecast'])
                st.dataframe(forecast_df, use_container_width=True)


def show_clustering_panel(df):
    """Clustering panel"""
    st.markdown("### 🎯 Product Clustering & Segmentation")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
    
    available_features = sales_cols + growth_cols + price_cols
    
    if len(available_features) < 2:
        st.warning("⚠️ At least 2 features required for clustering")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_features = st.multiselect(
            "📊 Features",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    with col2:
        algorithm = st.selectbox(
            "🔧 Algorithm",
            ["K-Means", "Hierarchical", "DBSCAN"]
        )
    
    with col3:
        n_clusters = st.slider(
            "🎯 Clusters",
            Config.MIN_N_CLUSTERS,
            Config.MAX_N_CLUSTERS,
            Config.DEFAULT_N_CLUSTERS
        )
    
    if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Select features!")
            return
        
        with st.spinner("🔄 Performing clustering analysis..."):
            algorithm_map = {
                "K-Means": "kmeans",
                "Hierarchical": "hierarchical",
                "DBSCAN": "dbscan"
            }
            
            results, error = MLEngine.perform_clustering(
                df, selected_features, n_clusters, algorithm_map[algorithm]
            )
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.success("✅ Clustering complete!")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                
                with metric_col2:
                    st.metric("Calinski-Harabasz", f"{results['calinski_score']:.0f}")
                
                with metric_col3:
                    st.metric("Davies-Bouldin", f"{results['davies_bouldin_score']:.3f}")
                
                with st.expander("📋 Cluster Statistics"):
                    for cluster_id, stats in results['cluster_stats'].items():
                        st.write(f"**Cluster {cluster_id}:** {stats['size']} products ({stats['percentage']:.1f}%)")


def show_anomaly_panel(df):
    """Anomaly detection panel"""
    st.markdown("### ⚠️ Anomaly Detection System")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    
    available_features = sales_cols + growth_cols
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_features = st.multiselect(
            "📊 Features",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    with col2:
        contamination = st.slider(
            "🎯 Expected Anomaly Rate (%)",
            int(Config.MIN_CONTAMINATION * 100),
            int(Config.MAX_CONTAMINATION * 100),
            int(Config.DEFAULT_CONTAMINATION * 100)
        ) / 100
    
    if st.button("🚀 Detect Anomalies", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Select features!")
            return
        
        with st.spinner("🔄 Analyzing data..."):
            results, error = MLEngine.detect_anomalies(df, selected_features, contamination)
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.success("✅ Analysis complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Anomalies", results['anomaly_count'])
                
                with col2:
                    st.metric("Rate", f"{results['anomaly_percentage']:.2f}%")
                
                with col3:
                    normal = len(df) - results['anomaly_count']
                    st.metric("Normal", normal)


def show_segmentation_tab():
    """Segmentation analysis tab"""
    df = st.session_state.filtered_data
    
    st.markdown('<div class="subsection-header"><h3>🎯 Market Segmentation Analysis</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown("### Segment products by different criteria")
    st.dataframe(df.head(50), use_container_width=True)


def show_advanced_analytics_tab():
    """Advanced analytics tab"""
    st.markdown('<div class="subsection-header"><h3>📈 Advanced Market Analytics</h3></div>', 
                unsafe_allow_html=True)
    
    st.info("🚀 Advanced analytics features coming soon!")


def show_reporting_tab():
    """Reporting and export tab"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    
    st.markdown('<div class="subsection-header"><h3>📑 Reports & Data Export</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown("#### 📥 Download Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**CSV Format**")
        Utils.create_download_link(df, "pharma_data", "csv")
    
    with col2:
        st.markdown("**JSON Format**")
        Utils.create_download_link(df, "pharma_data", "json")
    
    with col3:
        st.markdown("**Excel Format**")
        Utils.create_download_link(df, "pharma_data", "xlsx")
    
    with col4:
        st.markdown("**HTML Format**")
        Utils.create_download_link(df, "pharma_data", "html")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        gc.enable()
        Logger.info(f"Starting {Config.APP_NAME} v{Config.VERSION}")
        main()
        Logger.info("Application running successfully")
    except Exception as e:
        Logger.error(f"Critical application error: {str(e)}", e)
        st.error(f"❌ **Critical Error:**\n\n{str(e)}")
        with st.expander("🔍 View Detailed Error"):
            st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application", type="primary"):
            st.rerun()

# ============================================================================
# END OF PHARMAINTELLIGENCE PRO v7.0 - PROFESSIONAL EDITION
# ============================================================================

# ============================================================================
# ADDITIONAL PROFESSIONAL FEATURES
# ============================================================================

class AdvancedVisualizations:
    """
    Extended visualization library for advanced charts
    """
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> Optional[go.Figure]:
        """Create professional correlation heatmap"""
        try:
            corr_data = df[features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=features,
                y=features,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
        except:
            return None
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, column: str) -> Optional[go.Figure]:
        """Create distribution plot with statistics"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Distribution', 'Box Plot'),
                row_heights=[0.7, 0.3]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=df[column],
                    nbinsx=50,
                    name='Distribution',
                    marker_color=Config.COLOR_PRIMARY
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    x=df[column],
                    name='Box Plot',
                    marker_color=Config.COLOR_SECONDARY
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
        except:
            return None
    
    @staticmethod
    def create_time_series_decomposition(df: pd.DataFrame, value_col: str, time_col: str) -> Optional[go.Figure]:
        """Create time series decomposition plot"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Original Series', 'Trend', 'Seasonality'),
                vertical_spacing=0.1
            )
            
            # Original
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[value_col],
                    mode='lines+markers',
                    name='Original',
                    line=dict(color=Config.COLOR_PRIMARY)
                ),
                row=1, col=1
            )
            
            # Trend (rolling average)
            trend = df[value_col].rolling(window=12, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color=Config.COLOR_SECONDARY)
                ),
                row=2, col=1
            )
            
            # Seasonality (detrended)
            seasonal = df[value_col] - trend
            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=Config.COLOR_SUCCESS)
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=900,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
        except:
            return None


class StatisticalTests:
    """
    Statistical testing and analysis utilities
    """
    
    @staticmethod
    def perform_normality_test(data: pd.Series) -> Dict:
        """Perform Shapiro-Wilk normality test"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < 3:
                return {'test': 'insufficient_data', 'statistic': None, 'p_value': None, 'is_normal': None}
            
            statistic, p_value = shapiro(clean_data)
            is_normal = p_value > Config.SIGNIFICANCE_LEVEL
            
            return {
                'test': 'shapiro_wilk',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': is_normal,
                'interpretation': 'Data appears normally distributed' if is_normal else 'Data deviates from normal distribution'
            }
        except:
            return {'test': 'failed', 'statistic': None, 'p_value': None, 'is_normal': None}
    
    @staticmethod
    def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> Dict:
        """Calculate confidence interval for mean"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < 2:
                return {'lower': None, 'upper': None, 'mean': None}
            
            mean = clean_data.mean()
            sem = stats.sem(clean_data)
            interval = stats.t.interval(confidence, len(clean_data)-1, loc=mean, scale=sem)
            
            return {
                'mean': float(mean),
                'lower': float(interval[0]),
                'upper': float(interval[1]),
                'confidence': confidence
            }
        except:
            return {'lower': None, 'upper': None, 'mean': None}
    
    @staticmethod
    def perform_correlation_test(x: pd.Series, y: pd.Series) -> Dict:
        """Perform Pearson and Spearman correlation tests"""
        try:
            # Remove NaN
            valid_mask = ~(x.isna() | y.isna())
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 3:
                return {'pearson': None, 'spearman': None}
            
            # Pearson
            pearson_r, pearson_p = pearsonr(x_clean, y_clean)
            
            # Spearman
            spearman_r, spearman_p = spearmanr(x_clean, y_clean)
            
            return {
                'pearson': {
                    'correlation': float(pearson_r),
                    'p_value': float(pearson_p),
                    'significant': pearson_p < Config.SIGNIFICANCE_LEVEL
                },
                'spearman': {
                    'correlation': float(spearman_r),
                    'p_value': float(spearman_p),
                    'significant': spearman_p < Config.SIGNIFICANCE_LEVEL
                }
            }
        except:
            return {'pearson': None, 'spearman': None}


class DataQualityChecker:
    """
    Comprehensive data quality assessment
    """
    
    @staticmethod
    def generate_quality_report(df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        report = {
            'overview': {},
            'completeness': {},
            'consistency': {},
            'validity': {},
            'recommendations': []
        }
        
        try:
            # Overview
            report['overview'] = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
            }
            
            # Completeness
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            report['completeness'] = {
                'total_missing': int(missing_counts.sum()),
                'missing_percentage': float((missing_counts.sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
                'completeness_score': float(1 - (missing_counts.sum() / (len(df) * len(df.columns))))
            }
            
            # Add recommendations
            if report['completeness']['missing_percentage'] > Config.MAX_MISSING_PERCENTAGE:
                report['recommendations'].append({
                    'type': 'warning',
                    'category': 'completeness',
                    'message': f"High missing data rate ({report['completeness']['missing_percentage']:.1f}%). Consider data imputation or removal."
                })
            
            # Consistency
            duplicates = df.duplicated().sum()
            report['consistency'] = {
                'duplicate_rows': int(duplicates),
                'duplicate_percentage': float((duplicates / len(df)) * 100)
            }
            
            if duplicates > 0:
                report['recommendations'].append({
                    'type': 'info',
                    'category': 'consistency',
                    'message': f"Found {duplicates} duplicate rows. Review and remove if appropriate."
                })
            
            # Validity
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - Config.OUTLIER_STD_THRESHOLD * IQR)) | 
                           (df[col] > (Q3 + Config.OUTLIER_STD_THRESHOLD * IQR))).sum()
                
                if outliers > 0:
                    outlier_info[col] = {
                        'count': int(outliers),
                        'percentage': float((outliers / len(df)) * 100)
                    }
            
            report['validity'] = {
                'outlier_columns': outlier_info,
                'total_outliers': sum(info['count'] for info in outlier_info.values())
            }
            
            # Overall quality score
            scores = [
                report['completeness']['completeness_score'],
                max(0, 1 - (report['consistency']['duplicate_percentage'] / 100)),
                max(0, 1 - (len(outlier_info) / len(numeric_cols))) if len(numeric_cols) > 0 else 1
            ]
            
            report['overall_quality_score'] = float(np.mean(scores))
            
            # Final recommendations
            if report['overall_quality_score'] < Config.MIN_DATA_QUALITY_SCORE:
                report['recommendations'].append({
                    'type': 'warning',
                    'category': 'overall',
                    'message': f"Data quality score is below threshold ({report['overall_quality_score']:.2%}). Review data quality issues."
                })
            else:
                report['recommendations'].append({
                    'type': 'success',
                    'category': 'overall',
                    'message': f"Data quality is acceptable ({report['overall_quality_score']:.2%})."
                })
            
            return report
            
        except Exception as e:
            Logger.error(f"Quality report generation failed: {str(e)}", e)
            return report


class ExportManager:
    """
    Advanced export and reporting manager
    """
    
    @staticmethod
    def generate_executive_summary(df: pd.DataFrame, metrics: Dict, insights: List[Dict]) -> str:
        """Generate executive summary report"""
        summary = f"""
# PHARMACEUTICAL MARKET ANALYSIS
# Executive Summary Report
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platform: {Config.APP_NAME} v{Config.VERSION}

## 1. MARKET OVERVIEW
{'=' * 80}

Total Market Size: {Utils.format_currency(metrics.get('total_market_value', 0))}
Products Analyzed: {metrics.get('total_rows', 0):,}
Geographic Coverage: {metrics.get('country_coverage', 0)} countries
Unique Molecules: {metrics.get('unique_molecules', 0):,}

Average Growth Rate: {Utils.format_percentage(metrics.get('avg_growth', 0))}
Market Concentration (HHI): {metrics.get('hhi_index', 0):.0f}
Number of Competitors: {metrics.get('num_competitors', 0)}

## 2. KEY FINDINGS
{'=' * 80}

"""
        
        # Add insights
        for i, insight in enumerate(insights[:10], 1):
            summary += f"{i}. {insight['title']}\n   {insight['description']}\n\n"
        
        summary += f"""
## 3. MARKET STRUCTURE
{'=' * 80}

Top Competitor Share (CR1): {Utils.format_percentage(metrics.get('top_1_share', 0))}
Top 4 Competitors (CR4): {Utils.format_percentage(metrics.get('cr4', 0))}
Effective Competitors: {metrics.get('effective_competitors', 0):.1f}

Gini Coefficient: {metrics.get('gini_coefficient', 0):.3f}
Market Structure: {'Highly Concentrated' if metrics.get('hhi_index', 0) > 2500 else 'Moderately Concentrated' if metrics.get('hhi_index', 0) > 1500 else 'Competitive'}

## 4. PRICING ANALYSIS
{'=' * 80}

Average Price: {Utils.format_currency(metrics.get('avg_price', 0))}
Price Range: {Utils.format_currency(metrics.get('min_price', 0))} - {Utils.format_currency(metrics.get('max_price', 0))}
Price Variability (CV): {Utils.format_percentage(metrics.get('price_cv', 0))}

## 5. GROWTH DYNAMICS
{'=' * 80}

Products with Positive Growth: {Utils.format_percentage(metrics.get('positive_growth_pct', 0))}
High Growth Products (>20%): {metrics.get('high_growth_count', 0)}
Declining Products (<-10%): {metrics.get('decline_count', 0)}

Average CAGR: {Utils.format_percentage(metrics.get('avg_cagr', 0))}

## 6. RECOMMENDATIONS
{'=' * 80}

Based on the analysis:

1. Market Concentration Strategy
   - Current HHI suggests {'monopolistic' if metrics.get('hhi_index', 0) > 2500 else 'oligopolistic' if metrics.get('hhi_index', 0) > 1500 else 'competitive'} market
   - Consider {'market share defense' if metrics.get('hhi_index', 0) > 2500 else 'strategic partnerships' if metrics.get('hhi_index', 0) > 1500 else 'differentiation'} strategies

2. Growth Opportunities
   - Focus on high-growth segments showing >20% growth
   - Monitor declining products for turnaround or divestment

3. Pricing Strategy
   - Current CV of {metrics.get('price_cv', 0):.1f}% suggests {'high' if metrics.get('price_cv', 0) > 50 else 'moderate'} price variation
   - {'Value-based' if metrics.get('price_cv', 0) > 50 else 'Competitive'} pricing recommended

{'=' * 80}
End of Executive Summary
{'=' * 80}
"""
        
        return summary


# ============================================================================
# ENHANCED SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """
    Enhanced session state management with persistence
    """
    
    @staticmethod
    def initialize_session() -> None:
        """Initialize all session state variables"""
        defaults = {
            'data': None,
            'filtered_data': None,
            'metrics': {},
            'insights': [],
            'ml_results': {},
            'active_filters': {},
            'analysis_history': [],
            'user_preferences': {
                'theme': 'dark',
                'chart_style': 'modern',
                'export_format': 'csv'
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def save_analysis_state(name: str) -> None:
        """Save current analysis state"""
        state = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'metrics': st.session_state.metrics,
            'filters': st.session_state.active_filters,
            'row_count': len(st.session_state.filtered_data) if st.session_state.filtered_data is not None else 0
        }
        
        st.session_state.analysis_history.append(state)
    
    @staticmethod
    def clear_session() -> None:
        """Clear all session data"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]


# ============================================================================
# PERFORMANCE PROFILER
# ============================================================================

class Profiler:
    """
    Application performance profiling
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        class ProfileContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start = None
            
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.profiler.metrics[self.name] = elapsed
                Logger.info(f"Profile [{self.name}]: {elapsed:.3f}s")
        
        return ProfileContext(self, name)
    
    def get_report(self) -> Dict:
        """Get profiling report"""
        return {
            'total_time': time.time() - self.start_time,
            'operations': self.metrics
        }


# ============================================================================
# CONFIGURATION VALIDATOR
# ============================================================================

class ConfigValidator:
    """
    Validate and enforce configuration constraints
    """
    
    @staticmethod
    def validate() -> List[str]:
        """Validate all configuration settings"""
        errors = []
        
        # Validate numeric ranges
        if Config.MIN_N_CLUSTERS >= Config.MAX_N_CLUSTERS:
            errors.append("MIN_N_CLUSTERS must be less than MAX_N_CLUSTERS")
        
        if Config.MIN_CONTAMINATION >= Config.MAX_CONTAMINATION:
            errors.append("MIN_CONTAMINATION must be less than MAX_CONTAMINATION")
        
        # Validate colors
        color_attributes = [attr for attr in dir(Config) if attr.startswith('COLOR_')]
        for attr in color_attributes:
            color = getattr(Config, attr)
            if not re.match(r'^#[0-9A-Fa-f]{6}$', color):
                errors.append(f"Invalid color format for {attr}: {color}")
        
        return errors


# Initialize on import
SessionManager.initialize_session()
validation_errors = ConfigValidator.validate()
if validation_errors:
    Logger.warning(f"Configuration validation errors: {validation_errors}")
