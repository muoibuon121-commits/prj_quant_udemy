# Forecasting Financial Volatility: A Comparative Study of GARCH and LSTM
> [!NOTE]
> **Abstract**
> This research project investigates the predictive power of Deep Learning architectures versus classical Econometric models in forecasting financial asset volatility. The study specifically evaluates whether Long Short-Term Memory (LSTM) networks, capable of capturing non-linear dependencies and long-term temporal patterns, outperform the industry-standard GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models.

---

## 1. Mathematical Framework

### 1.1 Target Variable: Realized Volatility
The project targets the annualized **Realized Volatility** ($\sigma$), derived from the log returns ($r_t$) of the asset:

$$r_t = \ln \left( \frac{P_t}{P_{t-1}} \right)$$

The volatility over a rolling window of $N$ days is calculated as:

$$\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (r_{t-i} - \bar{r})^2}$$

### 1.2 Baseline Model: GARCH(1,1)
The benchmark estimates the conditional variance $\sigma_t^2$ as a linear function of past squared residuals and past variances:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### 1.3 Proposed Model: LSTM with QLIKE Loss
To capture non-linear market dynamics, an LSTM network is implemented. The model is optimized using the **Quasi-Likelihood (QLIKE)** loss function, which is statistically robust for variance-based proxies:

$$L(\sigma^2, \hat{\sigma}^2) = \frac{\sigma^2}{\hat{\sigma}^2} - \ln \left( \frac{\sigma^2}{\hat{\sigma}^2} \right) - 1$$

---

## 2. Methodology

### 2.1 Data Pipeline
- **Input Transformation:** Raw price data is converted into 3D Tensors with a shape of $[Batch, Sequence, Features]$.
- **Lookback Window:** A 60-day historical window is used to predict the one-step-ahead volatility.
- **Feature Scaling:** Min-Max Scaling is applied to the training set to ensure gradient stability in the LSTM.

### 2.2 Validation Strategy
To prevent **Look-ahead Bias**, we employ a **Walk-Forward Validation** (Rolling Window) approach:
1. Train the model on data from $T_0$ to $T_n$.
2. Test on $T_{n+1}$.
3. Slide the window forward and retrain/re-evaluate.

---

## 3. Project Structure

- `data_processor.py`: Handles data acquisition via `yfinance` and tensor construction.
- `model_engine.py`: Contains the `VolatilityLSTM` class and the custom `QLIKE` loss implementation.
- `baselines.py`: Fits the GARCH(1,1) parameters using Maximum Likelihood Estimation.
- `evaluator.py`: Generates comparative metrics (MAE, RMSE, QLIKE) and visualization plots.

---

## 4. Technical Implementation

### Prerequisites
- Python 3.9+
- PyTorch
- `arch` library (for Econometrics)
- `yfinance`

### Execution
```bash
# Installation
pip install torch arch pandas numpy yfinance

# Run experiment
python main.py --symbol BTC-USD
