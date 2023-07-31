# Asterisc-Technocrat
Bitcoin price prediction
pip install kaleido
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import kaleido.scopes.plotly
import plotly.offline as pyo
from kaleido.scopes.plotly import PlotlyScope
df = yf.download('BTC-USD')
df.head()
df.isnull().sum()
df.info()
df.describe()
df.reset_index(inplace = True)
df['Date'].dtype
plt.figure(figsize = (12,8))
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Price Distribution')
plt.figure(figsize = (12,8))
plt.boxplot(df['Close'])
plt.title('Bitcoin Price Distribution')
df['Day'] = df['Date'].dt.day_name()
df
data = df.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume'])
data.tail()
day_stats = data.groupby('Day')['Close'].agg(['mean', 'median', 'std']).reset_index()
fig = go.Figure()

fig.add_trace(go.Bar(x = day_stats['Day'], y = day_stats['mean'], name = 'Mean'))
fig.add_trace(go.Bar(x = day_stats['Day'], y = day_stats['median'], name = 'Median'))
fig.add_trace(go.Bar(x = day_stats['Day'], y = day_stats['std'], name = 'Standard Division'))
fig.update_layout(title = 'Bitcoin Price Analysis Day wise', xaxis_title = 'Day',  yaxis_title = 'Bitcoin Price')


kaleido_scope = kaleido.scopes.plotly.PlotlyScope(mathjax=None)


pio.write_image(fig, "stats.png", engine = "kaleido")
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data['Close'],
                            model='multiplicative',
                            period=100)

fig = plt.figure()
fig = result.plot()

fig = mpl_to_plotly(fig)

kaleido_scope = kaleido.scopes.plotly,PlotlyScope(mathjax=None)


pio.write_image(fig, "diag.png", engine = "kaleido")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(211)
plot_acf(data['Close'], lags=100, ax=ax1)

ax2 = fig.add_subplot(212)
plot_pacf(data['Close'], lags=100, ax=ax2)

plt.show()
p, d, q = 3, 1, 1
from statsmodels.tsa.arima.model import ARIMA
model_a = ARIMA(data['Close'], order=(p, d, q))
model_a = model_a.fit()
print(model_a.summary())
pred = model_a.predict(len(data) - 50, len(data) + 100)
fig = go.Figure()
fig.add_trace(go.Scatter(x = data.index, y = data['Close'], mode = 'lines', name = 'Bitcoin Actual Prices'))
fig.add_trace(go.Scatter(x = pred.index, y = pred, mode = 'lines', name = 'Bitcoin Predictions'))
fig.update_layout(title = 'Bitcoin Actual Values vs Prediction', xaxis_title = 'Date', yaxis_title = 'Closing Price')


kaleido_scope = kaleido.scopes.plotly.PlotlyScope(mathjax=None)


pio.write_image(fig, "pred1.png", engine = "kaleido")
