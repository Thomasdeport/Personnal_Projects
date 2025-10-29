import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_candlestick_with_indices(udl ,indices_dict= {}, other = None):
    """
    Affiche AAPL en chandeliers + indices en lignes.
    """
    fig = go.Figure()
    
    other = [sym for sym in indices_dict.keys() if sym != udl] if other == None else other
    # Autres indices en ligne
    for symbol, df in indices_dict.items():
        if symbol == udl: 
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name= udl,
                increasing_line_color='green',
                decreasing_line_color='red',
                opacity=0.8
            ))
        elif symbol in other:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name=symbol,
                line=dict(width=1.5)
            ))

    fig.update_layout(
        title=f"{udl} (candlestick) + US Benchmarks",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()


def return_computation(px, TICKERS= [],  log_tickers= ["AAPL", "SPY", "QQQ", "XLK", "TLT", "GLD", "UUP"], pct_tickers=  [ ], unchanged_tickers= ["VIXY",] ):
    retn = pd.DataFrame(index=px.index)
    for t in TICKERS:
        if t in log_tickers:
            retn[t] = np.log(px[t] / px[t].shift(1)) * 100
        elif t in pct_tickers:
            retn[t] = px[t].pct_change() * 100
        elif unchanged_tickers: 
            retn[t] = px[t]
    return retn.dropna()