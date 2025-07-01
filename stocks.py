import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from itertools import combinations
import warnings
import time
from sklearn.manifold import TSNE
import plotly.express as px

warnings.filterwarnings('ignore')

#define S&P 500 tickers and sectors
def get_sp500_tickers_and_sectors():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        #fetch html content with ssl verification disabled
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        tickers = []
        sectors = []
        for row in table.findAll('tr')[1:]:
            cells = row.findAll('td')
            ticker = cells[0].text.strip().replace('.', '-')
            sector = cells[3].text.strip()  #GICS sector
            tickers.append(ticker)
            sectors.append(sector)
        ticker_sector_dict = dict(zip(tickers, sectors))
        print(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers, ticker_sector_dict
    except Exception as e:
        print(f"Failed to fetch S&P 500 tickers: {e}")
        return [], {}

#fetch tickers and sectors
tickers, ticker_sector_dict = get_sp500_tickers_and_sectors()
if not tickers:
    raise ValueError("Failed to fetch S&P 500 tickers. No fallback available.")
start_date = "2024-01-01"
end_date = "2025-06-24"

#download data sequentially
data = pd.DataFrame()
failed_tickers = []
for ticker in tickers:
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                raise ValueError("No data returned")
            print(f"Successfully fetched data for {ticker} ({len(df)} rows)")
            data[ticker] = df['Close']
            time.sleep(2)  #avoid rate limits
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/3 failed for {ticker}: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"Failed to fetch data for {ticker} after 3 attempts")
                failed_tickers.append(ticker)

if failed_tickers:
    print(f"Failed to fetch data for {len(failed_tickers)} tickers: {failed_tickers}")

#check for valid data
if data.empty:
    raise ValueError("No data retrieved for any tickers. Check API, network, or date range.")
if data.isna().any().any():
    print("Warning: Missing data detected for some tickers.")
    print(data.isna().sum())
data = data.dropna(axis=1, how='any')
tickers = data.columns  #update tickers list
if len(tickers) == 0:
    raise ValueError("No valid tickers with data. Cannot proceed.")
print(f"Proceeding with {len(tickers)} tickers: {list(tickers)}")
returns = data.pct_change().dropna()

#update sector dictionary for valid tickers
ticker_sector_dict = {ticker: ticker_sector_dict.get(ticker, 'Unknown') for ticker in tickers}

#encode returns as symbols
def encode_return(r, threshold=0.01):
    if r > threshold:
        return "+"
    elif r < -threshold:
        return "-"
    else:
        return "0"

symbolic_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
symbolic_returns[returns > 0.01] = "+"
symbolic_returns[returns < -0.01] = "-"
symbolic_returns[returns.abs() <= 0.01] = "0"
symbolic_returns = symbolic_returns.fillna("0")

#perform shingling
def shingle_sequence(seq, k=4):
    if len(seq) < k:
        print(f"Warning: Insufficient data for shingling (length={len(seq)} < k={k})")
        return set()
    return set("".join(seq[i:i+k]) for i in range(len(seq) - k + 1))

shingles = {ticker: shingle_sequence(symbolic_returns[ticker].tolist(), k=4) for ticker in tickers}
for ticker in tickers:
    if not shingles[ticker]:
        print(f"Warning: No shingles generated for {ticker}. Check data or reduce k.")

#compute jaccard similarity matrix
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

jaccard_matrix = pd.DataFrame(0.0, index=tickers, columns=tickers)
for t1, t2 in combinations(tickers, 2):
    sim = jaccard_similarity(shingles[t1], shingles[t2])
    jaccard_matrix.loc[t1, t2] = sim
    jaccard_matrix.loc[t2, t1] = sim
np.fill_diagonal(jaccard_matrix.values, 1.0)

#save similarity matrix
jaccard_matrix.to_csv("matrix.csv")

#convert similarity to distance
distance_matrix = 1 - jaccard_matrix
tsne = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=min(30, len(tickers)-1), init='random')
embeddings = tsne.fit_transform(distance_matrix)

#create dataframe for plotly
df_plot = pd.DataFrame({
    'x': embeddings[:, 0],
    'y': embeddings[:, 1],
    'ticker': tickers,
    'sector': [ticker_sector_dict[t] for t in tickers]
})

#plot with plotly
fig = px.scatter(
    df_plot,
    x='x',
    y='y',
    color='sector',
    text='ticker',
    title="t-SNE of Jaccard Similarity for All S&P 500 Stocks",
    labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
    hover_data=['ticker', 'sector']
)
fig.update_traces(textposition='middle right', textfont_size=6)
fig.update_layout(
    width=1200,
    height=900,
    margin=dict(l=120, r=120, t=120, b=120),
    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
    showlegend=False
)
fig.write_html('plotly_tsne.html')
fig.show()
