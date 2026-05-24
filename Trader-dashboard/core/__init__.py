# Sniper Terminal — Core Engine
# Institutional-Grade Swing Trading Platform for Indian Markets

from core.indicators import (
    compute_indicators, compute_rsi,
    compute_institutional_score, generate_swing_signal,
    detect_vcp, calculate_avwap,
    compute_delivery_enhanced_score, generate_fo_enhanced_signal,
)
from core.scanner import scan_stock, scan_universe, filter_by_strategy
from core.backtest import run_backtest, run_backtest_simple
from core.market_regime import get_market_regime
from core.relative_strength import get_relative_strength, calculate_rs_score
from core.risk_engine import calculate_position_size, portfolio_heat
from core.charts import plot_chart
from core.utils import load_data, clean_columns, clear_cache, pct_change, safe_round
from core.data_providers import (
    get_market_overview, get_fo_chain, get_india_vix,
    get_fii_dii_data, get_market_breadth, get_delivery_summary,
    get_stock_fo_data, get_expiry_calendar, get_block_deals,
    clear_data_provider_cache
)
from core.patterns import (
    run_all_patterns, get_best_pattern,
    detect_bull_flag, detect_double_bottom, detect_cup_handle,
    detect_consolidation_breakout, get_52w_proximity
)
from core.scan_engine import (
    CONDITION_REGISTRY, STRATEGIES,
    evaluate_conditions, evaluate_strategy, evaluate_all_strategies
)
from core.watchlist import (
    get_watchlists, get_watchlist, create_watchlist, delete_watchlist,
    add_to_watchlist, remove_from_watchlist, get_watchlist_with_prices,
    rename_watchlist, export_watchlist
)
from core.trade_journal import (
    add_trade, close_trade, delete_trade, update_trade,
    get_all_trades, get_open_trades, get_closed_trades,
    get_portfolio_summary, get_monthly_returns, export_journal
)
from core.risk_engine import (
    calculate_position_size, calculate_position_with_target,
    portfolio_heat, suggest_stop_loss, calculate_pyramid_size
)
from core.support_resistance import (
    detect_support_resistance, fibonacci_levels, pivot_points,
    detect_breakout_levels
)
from core.strategy_optimizer import (
    run_strategy_comparison, walk_forward_optimize,
    get_rule_based_score, train_ai_scorer, score_signal_ai,
    _extract_features, STRATEGY_DEFINITIONS, SKLEARN_AVAILABLE
)
from core.volume_profile import (
    volume_profile, vwap_with_bands, market_profile, get_volume_metrics
)
from core.charts import (
    plot_chart_with_drawings, plot_volume_profile,
    plot_comparison_chart, create_drawing_shape
)
from core.shoonya_bridge import (
    ShoonyaClient, get_client, connect as shoonya_connect,
    disconnect as shoonya_disconnect, load_data_shoonya,
    load_credentials, save_credentials, clear_credentials
)

__all__ = [
    "compute_indicators", "compute_rsi", "compute_institutional_score",
    "generate_swing_signal", "detect_vcp", "calculate_avwap",
    "compute_delivery_enhanced_score", "generate_fo_enhanced_signal",
    "scan_stock", "scan_universe", "filter_by_strategy",
    "run_backtest", "run_backtest_simple",
    "get_market_regime",
    "get_relative_strength", "calculate_rs_score",
    "calculate_position_size", "portfolio_heat",
    "plot_chart",
    "load_data", "clean_columns", "clear_cache", "pct_change", "safe_round",
    "get_market_overview", "get_fo_chain", "get_india_vix",
    "get_fii_dii_data", "get_market_breadth", "get_delivery_summary",
    "get_stock_fo_data", "get_expiry_calendar", "get_block_deals",
    "clear_data_provider_cache",
    "run_all_patterns", "get_best_pattern",
    "detect_bull_flag", "detect_double_bottom", "detect_cup_handle",
    "detect_consolidation_breakout", "get_52w_proximity",
    "CONDITION_REGISTRY", "STRATEGIES",
    "evaluate_conditions", "evaluate_strategy", "evaluate_all_strategies"
]

