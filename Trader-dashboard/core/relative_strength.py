from core.utils import load_data
from core.indicators import compute_indicators


def calculate_rs_score(stock_df, index_df):

    if len(stock_df) < 100 or len(index_df) < 100:
        return 0

    stock_return = (
        stock_df["Close"].iloc[-1] /
        stock_df["Close"].iloc[-60]
    ) - 1

    index_return = (
        index_df["Close"].iloc[-1] /
        index_df["Close"].iloc[-60]
    ) - 1

    rs = (stock_return - index_return) * 100

    return round(rs, 2)


def get_relative_strength(ticker):

    stock_df = load_data(ticker)

    nifty_df = load_data("^NSEI")

    if stock_df.empty or nifty_df.empty:
        return 0

    stock_df = compute_indicators(stock_df)

    nifty_df = compute_indicators(nifty_df)

    return calculate_rs_score(stock_df, nifty_df)
