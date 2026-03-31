import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import os
from datetime import timedelta

# Cấu hình trang tổng
st.set_page_config(
    page_title="L-GRU AI Forecast Dashboard",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------------------
# 1. KIẾN TRÚC MẠNG TÙY CHỈNH L-GRU (DÙNG CHUNG)
# -------------------------------------------------------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# --- Kiến trúc L-GRU CŨ (dùng cho model tài chính) ---
class CustomRNNAILayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        self.W_cr = nn.Parameter(torch.Tensor(hidden_size))
        self.W_cz = nn.Parameter(torch.Tensor(hidden_size))
        self.W_co = nn.Parameter(torch.Tensor(hidden_size))

        self.ln_r = nn.LayerNorm(hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)

        self.mish = Mish()

    def forward(self, x_t, states):
        h_prev, c_prev = states
        hx = torch.cat([x_t, h_prev], dim=1)

        r_t = torch.sigmoid(self.ln_r(self.W_r(hx) + self.W_cr * c_prev))
        r_hx = torch.cat([x_t, r_t * h_prev], dim=1)
        c_tilde = self.mish(self.W_c(r_hx))
        z_t = torch.sigmoid(self.ln_z(self.W_z(hx) + self.W_cz * c_prev))
        c_t = z_t * c_prev + (1 - z_t) * c_tilde
        o_t = torch.sigmoid(self.ln_o(self.W_o(hx) + self.W_co * c_t))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class CustomAIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            CustomRNNAILayer(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        states = [(torch.zeros(batch_size, self.hidden_size).to(x.device),
                   torch.zeros(batch_size, self.hidden_size).to(x.device))
                  for _ in range(self.num_layers)]
        out = None
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, states[i])
                states[i] = (h_t, c_t)
                x_t = h_t
            out = h_t
        return self.fc(out)

# ─────────────────────────────────────────────────────────────────────────────
# KIẾN TRÚC MODEL MỚI — Custom GRU-LSTM Hybrid + TemporalAttention
# Khớp hoàn toàn với train_model_weather.py (best_weather_model(release).pth)
# input_size=10 (4 thời tiết + 6 time features), hidden=256, layers=3, seq_len=96
# ─────────────────────────────────────────────────────────────────────────────
class MishActivation(nn.Module):
    """Hàm kích hoạt Mish: f(x) = x * tanh(softplus(x))."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class CustomRNNCell(nn.Module):
    """
    Custom RNN Cell — lai GRU–LSTM với LayerNorm và Mish.
    Khớp với state_dict keys: rnn.cells.N.W_rh, W_rx, W_Cr, ln_r, W_c, W_Cz, W_zh, W_zx,
    ln_z, W_Co, W_oh, W_ox, ln_o
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H, I = hidden_size, input_size

        # Cổng Reset (r)
        self.W_rh = nn.Linear(H, H, bias=False)
        self.W_rx = nn.Linear(I, H, bias=True)
        self.W_Cr = nn.Parameter(torch.Tensor(H))
        self.ln_r = nn.LayerNorm(H)

        # Trạng thái ứng viên (C~)
        self.W_c  = nn.Linear(H + I, H, bias=True)
        self.mish = MishActivation()

        # Cổng Update (z)
        self.W_zh = nn.Linear(H, H, bias=False)
        self.W_zx = nn.Linear(I, H, bias=True)
        self.W_Cz = nn.Parameter(torch.Tensor(H))
        self.ln_z = nn.LayerNorm(H)

        # Cổng Output (o)
        self.W_oh = nn.Linear(H, H, bias=False)
        self.W_ox = nn.Linear(I, H, bias=True)
        self.W_Co = nn.Parameter(torch.Tensor(H))
        self.ln_o = nn.LayerNorm(H)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        r_t     = torch.sigmoid(self.ln_r(self.W_rh(h) + self.W_rx(x) + self.W_Cr * c))
        c_tilde = self.mish(self.W_c(torch.cat([r_t * h, x], dim=-1)))
        z_t     = torch.sigmoid(self.ln_z(self.W_zh(h) + self.W_zx(x) + self.W_Cz * c))
        c_t     = z_t * c + (1.0 - z_t) * c_tilde
        o_t     = torch.sigmoid(self.ln_o(self.W_oh(h) + self.W_ox(x) + self.W_Co * c_t))
        h_t     = o_t * torch.tanh(c_t)
        return h_t, c_t


class CustomRNN(nn.Module):
    """Nhiều lớp CustomRNNCell chồng nhau với Dropout giữa các lớp."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.cells = nn.ModuleList([
            CustomRNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, h_init=None, c_init=None):
        batch_size = x.size(0)
        device     = x.device
        if h_init is None:
            h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if c_init is None:
            c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        h_list = list(h_init.unbind(0))
        c_list = list(c_init.unbind(0))
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, h_list[layer_idx], c_list[layer_idx])
                h_list[layer_idx] = h_t
                c_list[layer_idx] = c_t
                x_t = self.dropout(h_t) if layer_idx < self.num_layers - 1 else h_t
            outputs.append(h_t)
        output = torch.stack(outputs, dim=1)   # (batch, seq_len, hidden)
        h_n    = torch.stack(h_list, dim=0)
        c_n    = torch.stack(c_list, dim=0)
        return output, (h_n, c_n)


class TemporalAttention(nn.Module):
    """Temporal Attention — tổng hợp thông tin thông minh từ toàn bộ chuỗi."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn_layer(rnn_out).squeeze(-1)      # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)              # (batch, seq_len)
        context = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context


class WeatherForecastModel(nn.Module):
    """
    Model dự báo thời tiết mới:
        Input (batch, seq_len=96, input_size=10)
          → CustomRNN (3 lớp, hidden=256)
          → TemporalAttention
          → Dropout → MLP → Output (batch, 4)
    Khớp hoàn toàn với checkpoint best_weather_model(release).pth
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 256,
                 num_layers: int = 3, output_size: int = 4, dropout: float = 0.25):
        super().__init__()
        self.rnn       = CustomRNN(input_size, hidden_size, num_layers, dropout)
        self.attention = TemporalAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            MishActivation(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)              # (batch, seq_len, hidden)
        context    = self.attention(rnn_out)   # (batch, hidden)
        context    = self.dropout(context)
        return self.output_layer(context)      # (batch, output_size)


# -----------------------------------------------------------------------
# HÀM HỖ TRỢ: MÃ HÓA THỜI GIAN VÀO FEATURES (Cyclical Encoding)
# -----------------------------------------------------------------------
def make_time_features(timestamp: pd.Timestamp) -> np.ndarray:
    """
    Tạo 6 đặc trưng thời gian tuần hoàn từ một timestamp:
      [hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos]
    Khớp hoàn toàn với cách mã hóa trong train_model_weather.py.
    """
    hour_sin  = np.sin(2 * np.pi * timestamp.hour / 24)
    hour_cos  = np.cos(2 * np.pi * timestamp.hour / 24)
    day_sin   = np.sin(2 * np.pi * timestamp.dayofyear / 365.25)
    day_cos   = np.cos(2 * np.pi * timestamp.dayofyear / 365.25)
    week_sin  = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    week_cos  = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    return np.array([hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos], dtype=np.float32)


def build_input_sequence_with_time(
    weather_scaled: np.ndarray,   # shape (N, 4): [Scaled_temp, Scaled_hum, Scaled_pres, Scaled_prec]
    timestamps: list,              # list of pd.Timestamp, len=N
    seq_len: int = 96
) -> np.ndarray:
    """
    Ghép weather_scaled (N,4) với time_features (N,6) → (N,10).
    Cắt seq_len bước cuối cùng → (seq_len, 10).
    """
    n = len(timestamps)
    time_feats = np.array([make_time_features(t) for t in timestamps], dtype=np.float32)  # (N,6)
    combined   = np.concatenate([weather_scaled, time_feats], axis=1)   # (N, 10)
    # Lấy seq_len bước cuối (pad đầu nếu không đủ)
    if n >= seq_len:
        return combined[-seq_len:]
    else:
        pad = np.zeros((seq_len - n, 10), dtype=np.float32)
        return np.concatenate([pad, combined], axis=0)


# -----------------------------------------------------------------------
# HÀM DỰ BÁO AUTOREGRESSIVE (MODEL MỚI — 10 INPUT FEATURES)
# -----------------------------------------------------------------------
def predict_autoregressive_weather(
    model, input_seq, steps, device,
    scaler=None,              # MinMaxScaler cho 4 cột thời tiết
    start_timestamp=None      # pd.Timestamp của bước dự báo đầu tiên (t+1)
):
    """
    Dự báo autoregressive nhiều bước với model mới (input 10 features).

    Tại mỗi bước:
      1. Dự báo 4 Scaled thời tiết từ chuỗi hiện tại (seq_len, 10)
      2. Inverse-transform → clip vật lý → re-scale
      3. Tạo time features cho timestamp tiếp theo
      4. Ghép [4 scaled weather | 6 time feats] → nối vào chuỗi trượt
    """
    current_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device)  # (1, seq_len, 10)
    predictions_weather = []  # chỉ lưu 4 cột thời tiết (scaled)

    # Timestamp dự báo bắt đầu từ step tiếp theo
    current_ts = start_timestamp if start_timestamp is not None else pd.Timestamp.now()

    model.eval()  # Đảm bảo model ở eval mode trong suốt inference
    with torch.no_grad():
        for step in range(steps):
            pred = model(current_seq)           # (1, 4)
            pred_np = pred.cpu().numpy()[0]     # (4,)

            if scaler is not None:
                # Inverse → clip vật lý → re-scale
                pred_real = scaler.inverse_transform(pred_np.reshape(1, -1))[0]
                pred_real[0] = np.clip(pred_real[0], 5.0,   44.0)    # nhiệt độ
                pred_real[1] = np.clip(pred_real[1], 0.0,   100.0)   # độ ẩm
                pred_real[2] = np.clip(pred_real[2], 960.0, 1050.0)  # áp suất
                pred_real[3] = np.clip(pred_real[3], 0.0,   None)    # lượng mưa
                pred_scaled  = scaler.transform(pred_real.reshape(1, -1))[0]
            else:
                pred_scaled = pred_np.copy()
                pred_scaled[3] = max(pred_scaled[3], 0.0)

            predictions_weather.append(pred_scaled.copy())

            # FIX: Tạo time features cho CHÍNH timestamp hiện tại (current_ts),
            # KHÔNG phải next_ts. Row mới thêm vào chuỗi phải khớp
            # [weather dự báo tại t] với [time features tại t].
            # Lỗi cũ: dùng next_ts → time features lệch +1 giờ so với weather features.
            time_feat   = make_time_features(current_ts)               # (6,) — khớp với pred_scaled
            next_row    = np.concatenate([pred_scaled, time_feat])     # (10,)
            next_tensor = torch.FloatTensor(next_row).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,10)

            # Trượt chuỗi: bỏ bước đầu, thêm bước mới
            current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)

            # Tăng timestamp SAU khi đã tạo row — không phải trước
            current_ts  = current_ts + pd.Timedelta(hours=1)

    return np.array(predictions_weather)  # (steps, 4) — scaled


def predict_autoregressive_finance(model, input_seq, steps, device):
    """Dự báo autoregressive nhiều bước cho tài chính.

    Loại bỏ tính năng làm mượt (EMA) để giữ nguyên mức độ biến động (volatility)
    đặc trưng của thị trường tài chính.
    """
    current_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    predictions = []

    with torch.no_grad():
        for step in range(steps):
            pred = model(current_seq)
            pred_np = pred.cpu().numpy()[0]  # shape: (1,)

            predictions.append(pred_np.copy())

            pred_tensor = torch.FloatTensor(pred_np).unsqueeze(0).unsqueeze(0).to(device)
            current_seq = torch.cat([current_seq[:, 1:, :], pred_tensor], dim=1)

    return np.array(predictions)


# -------------------------------------------------------------
# 2. HÀM XỬ LÝ CACHE MODEL VÀ SCALER
# -------------------------------------------------------------
@st.cache_resource
def load_weather_model_and_scaler():
    """
    Load model thời tiết mới: WeatherForecastModel (input=10, hidden=256, layers=3, seq_len=96).
    Đọc config từ checkpoint để tự động cấu hình tham số.
    Scaler chỉ dùng cho 4 cột thời tiết gốc (temperature, humidity, pressure, precipitation).
    """
    from sklearn.preprocessing import MinMaxScaler

    model_path = "best_weather_model(release).pth"
    if not os.path.exists(model_path):
        return None, None, None, 96  # model, device, scaler, seq_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Đọc checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Đọc cấu hình từ checkpoint (có fallback an toàn)
    input_size  = ckpt.get('input_size',  10)
    hidden_size = ckpt.get('hidden_size', 256)
    num_layers  = ckpt.get('num_layers',  3)
    output_size = ckpt.get('output_size', 4)
    seq_len     = ckpt.get('seq_len',     96)

    cfg = ckpt.get('config', {})
    dropout = cfg.get('dropout', 0.25)

    # Khởi tạo model với đúng kiến trúc
    model = WeatherForecastModel(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        output_size = output_size,
        dropout     = dropout,
    ).to(device)

    # Load weights (xử lý prefix '_orig_mod.' nếu có từ torch.compile)
    state_dict = ckpt['model_state']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ── Scaler cho 4 cột thời tiết gốc ──
    scaler   = MinMaxScaler()
    features = ['temperature', 'humidity', 'pressure', 'precipitation']
    train_path = "weather_trainn.csv"

    # Dải hợp lý cho Hà Tĩnh — dùng khi không đọc được CSV
    # humidity_min = 0.0 (an toàn hơn 20.0 vì mùa khô Hà Tĩnh có thể xuống rất thấp)
    _fallback_min = np.array([5.0,  0.0,   960.0,  0.0])
    _fallback_max = np.array([43.0, 100.0, 1045.0, 300.0])

    if os.path.exists(train_path):
        try:
            # Đọc theo chunks để tiết kiệm RAM (file > 700k dòng)
            # File CSV phải có cột: temperature, humidity, pressure, precipitation
            data_min = None
            data_max = None
            for chunk in pd.read_csv(train_path, usecols=features, chunksize=50000):
                chunk['precipitation'] = chunk['precipitation'].clip(lower=0)
                c_min = chunk[features].min().values
                c_max = chunk[features].max().values
                data_min = c_min if data_min is None else np.minimum(data_min, c_min)
                data_max = c_max if data_max is None else np.maximum(data_max, c_max)
            scaler.fit(np.vstack([data_min, data_max]))
        except (ValueError, KeyError) as e:
            # Xảy ra khi CSV chỉ có cột Scaled_* mà không có cột raw
            # → dùng fallback hợp lý cho vùng Hà Tĩnh
            import warnings
            warnings.warn(
                f"[Scaler] Không đọc được cột raw từ {train_path} ({e}). "
                "Dùng dải hợp lý cho Hà Tĩnh làm fallback."
            )
            scaler.fit(np.vstack([_fallback_min, _fallback_max]))
    else:
        # Fallback: dải hợp lý cho Hà Tĩnh khi không có file training
        scaler.fit(np.vstack([_fallback_min, _fallback_max]))

    return model, device, scaler, seq_len


@st.cache_resource
def load_finance_scaler(ticker, train_end_date="2025-01-01"):
    """Load và fit MinMaxScaler cho tài chính trên dữ liệu TRƯỚC mốc train.
    
    FIX DATA LEAKAGE: Scaler chỉ được fit trên dữ liệu training (trước train_end_date),
    KHÔNG fit trên dữ liệu từ tương lai (sau thời điểm train), vì sẽ thay đổi không gian
    normalisation và làm sai lệch kết quả inference.
    """
    # CHUYỂN LỆNH IMPORT VÀO ĐÂY ĐỂ TRÁNH LỖI ATEXIT CỦA JOBLIB
    from sklearn.preprocessing import MinMaxScaler
    import yfinance as yf

    # Tải dữ liệu ĐẾN mốc cố định — tương đương dữ liệu training của model
    df_train = yf.download(ticker, start="2010-01-01", end=train_end_date, progress=False)
    if isinstance(df_train.columns, pd.MultiIndex):
        df_train.columns = df_train.columns.droplevel(1)

    if df_train.empty or 'Close' not in df_train.columns:
        # Fallback: nếu không tải được, dùng scaler mặc định với khoảng rộng
        scaler = MinMaxScaler()
        # Dùng range tương đối để fit (sẽ được override khi có data)
        scaler.fit([[0], [1]])
        return scaler

    scaler = MinMaxScaler()
    scaler.fit(df_train[['Close']].values)
    return scaler


@st.cache_resource
def load_finance_model(model_path):
    if not os.path.exists(model_path):
        return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomAIModel(input_size=1, hidden_size=256, output_size=1, num_layers=3).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, device


# -------------------------------------------------------------
# 3. PHÂN HỆ THỜI TIẾT
# -------------------------------------------------------------
def run_weather_app():
    col1, col2 = st.columns([1, 4])
    with col1:
        logo_path = "LogoHTU.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
    with col2:
        st.markdown("<h1 style='color: #1E3A8A;'>🌤️ L-GRU: Dự báo Thời tiết</h1>", unsafe_allow_html=True)
        st.write("**Bản thử nghiệm Beta - Dự báo Thời tiết Khu vực Hà Tĩnh (Dữ liệu 1940-2026)**")

    st.divider()

    # Load model mới (WeatherForecastModel, input=10, hidden=256, layers=3, seq_len=96)
    model, device, scaler, SEQ_LEN = load_weather_model_and_scaler()

    if model is None:
        st.error("❌ Không tìm thấy file model thời tiết `best_weather_model(release).pth`.")
        st.stop()
    if scaler is None:
        st.error("❌ Không thể khởi tạo Scaler cho dữ liệu thời tiết.")
        st.stop()

    forecast_hours = st.sidebar.slider("Dự báo bao nhiêu giờ tiếp theo?", min_value=1, max_value=72, value=24)

    st.markdown("### 🗓️ 1. Chọn thời điểm bắt đầu dự báo")
    st.info("💡 Không cần tải lên file. Hệ thống sẽ **tự động kết nối vệ tinh và tải dữ liệu thời tiết** 48 giờ trước thời điểm bạn chọn.")

    col_date, col_time = st.columns(2)
    with col_date:
        today = pd.Timestamp.now('Asia/Bangkok').date()
        min_date = today - timedelta(days=365*2)  # Cho phép lùi 2 năm
        selected_date = st.date_input("Ngày dự báo:", value=today, min_value=min_date, max_value=today)
    with col_time:
        if "default_time" not in st.session_state:
            st.session_state["default_time"] = pd.Timestamp.now('Asia/Bangkok').replace(second=0, microsecond=0).time()
        selected_time = st.time_input("Giờ dự báo:", value=st.session_state["default_time"])

    if st.button("🚀 TẢI DỮ LIỆU TỰ ĐỘNG & BẮT ĐẦU DỰ BÁO", use_container_width=True, type="primary", key="btn_weather"):
        with st.spinner("⏳ Đang tải dữ liệu thời tiết thực tế từ trạm khí tượng (Open-Meteo API)..."):
            import urllib.request
            import json
            import datetime

            target_dt = datetime.datetime.combine(selected_date, selected_time)
            # Tải đủ dữ liệu = SEQ_LEN (96h) + buffer 24h để bù độ trễ dữ liệu API
            # Tổng = 120h → sau khi lọc df_past sẽ còn ≥ SEQ_LEN hàng
            hours_needed = SEQ_LEN + 24  # đủ dữ liệu + buffer
            start_dt = target_dt - timedelta(hours=hours_needed)

            start_str = start_dt.strftime('%Y-%m-%d')
            end_str = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')

            # Tính số ngày từ hiện tại
            days_diff = (today - selected_date).days
            if days_diff <= 3:
                # API forecast: past_days=7 + forecast_days=1 để bao phủ target_dt
                # Thêm &forecast_days=1 để API trả về cả bản ghi giờ gần nhất hôm nay
                url = ("https://api.open-meteo.com/v1/forecast?"
                       "latitude=18.3333&longitude=105.9000"
                       f"&past_days=7&forecast_days=1"
                       "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation"
                       "&timezone=Asia%2FBangkok")
            else:
                url = (f"https://archive-api.open-meteo.com/v1/archive?"
                       f"latitude=18.3333&longitude=105.9000"
                       f"&start_date={start_str}&end_date={end_str}"
                       f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation"
                       f"&timezone=Asia%2FBangkok")

            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data_api = json.loads(response.read().decode())

                hourly = data_api.get('hourly', {})
                if not hourly:
                    st.error("❌ Không thể lấy dữ liệu từ API cho ngày này.")
                    st.stop()

                df_all = pd.DataFrame({
                    'time': hourly['time'],
                    'temperature': hourly['temperature_2m'],
                    'humidity': hourly['relative_humidity_2m'],
                    'pressure': hourly['surface_pressure'],
                    'precipitation': hourly['precipitation']
                })

                # Cắt đúng SEQ_LEN giờ trước target_dt
                target_dt_str = target_dt.strftime('%Y-%m-%dT%H:00')
                df_past = df_all[df_all['time'] <= target_dt_str]

                if len(df_past) >= SEQ_LEN:
                    df = df_past.tail(SEQ_LEN).reset_index(drop=True)
                elif len(df_past) > 0:
                    df = df_past.reset_index(drop=True)   # dùng tất cả những gì có
                else:
                    df = df_all.head(SEQ_LEN).reset_index(drop=True)

                # Xử lý missing data: nội suy liên tục, mưa điền 0
                df[['temperature', 'humidity', 'pressure']] = (
                    df[['temperature', 'humidity', 'pressure']]
                    .interpolate(method='linear').bfill().ffill()
                )
                df['precipitation'] = df['precipitation'].fillna(0)

                n_rows = len(df)
                st.success(
                    f"✅ Đã tải thành công {n_rows} giờ dữ liệu "
                    f"(cần {SEQ_LEN}) tính đến {target_dt.strftime('%H:%M %d/%m/%Y')}!"
                )
                with st.expander("👁️ Xem trước dữ liệu tự động tải"):
                    st.dataframe(df.tail(10), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Lỗi tự động tải dữ liệu: {e}")
                st.stop()

        # ── Bắt đầu dự báo ──
        with st.spinner("🤖 Bộ AI L-GRU (seq=96h, input=10) đang xử lý mô phỏng..."):
            weather_features = ['temperature', 'humidity', 'pressure', 'precipitation']
            data_raw = df[weather_features].values.astype(np.float32)

            # Clip vật lý trước khi scale
            data_raw[:, 0] = np.clip(data_raw[:, 0], 5.0,   44.0)    # nhiệt độ
            data_raw[:, 1] = np.clip(data_raw[:, 1], 0.0,   100.0)   # độ ẩm
            data_raw[:, 2] = np.clip(data_raw[:, 2], 960.0, 1050.0)  # áp suất
            data_raw[:, 3] = np.clip(data_raw[:, 3], 0.0,   None)    # mưa
            data_scaled = scaler.transform(data_raw)  # (N, 4)

            # Tạo danh sách timestamps tương ứng với từng hàng dữ liệu
            timestamps_list = pd.to_datetime(df['time']).tolist()

            # Xây dựng input sequence (seq_len, 10) = [4 weather | 6 time features]
            input_seq = build_input_sequence_with_time(
                data_scaled, timestamps_list, seq_len=SEQ_LEN
            )  # (SEQ_LEN, 10)

            # Timestamp của bước dự báo đầu tiên (= cuối input + 1 giờ)
            last_ts = pd.to_datetime(df['time'].iloc[-1])
            first_forecast_ts = last_ts + pd.Timedelta(hours=1)

            # Dự báo autoregressive — có tích hợp time features trong vòng lặp
            preds_scaled = predict_autoregressive_weather(
                model, input_seq, steps=forecast_hours,
                device=device, scaler=scaler,
                start_timestamp=first_forecast_ts
            )  # (forecast_hours, 4) scaled

            preds = scaler.inverse_transform(preds_scaled)  # (forecast_hours, 4) real units

            # =========================================================
            # HẬU XỬ LÝ KẾT QUẢ DỰ BÁO
            # =========================================================

            # 1. CLIP LƯỢNG MƯA ÂM: Lượng mưa về mặt vật lý không thể âm
            preds[:, 3] = np.clip(preds[:, 3], 0, None)

            # 2. CLIP ĐỘ ẨM vào dải [0, 100]%
            preds[:, 1] = np.clip(preds[:, 1], 0, 100)

            # 3. Clip nhiệt độ vào dải hợp lý cho Hà Tĩnh
            preds[:, 0] = np.clip(preds[:, 0], 5.0, 44.0)

            # 4. Clip áp suất vào dải hợp lý
            preds[:, 2] = np.clip(preds[:, 2], 960.0, 1050.0)

            # 5. CHUẨN BỊ THỜI GIAN TƯƠNG LAI
            try:
                last_time = pd.to_datetime(df['time'].iloc[-1])
            except Exception:
                last_time = pd.Timestamp.now()

            future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]

            # =========================================================
            past_times = pd.to_datetime(df['time'].tolist()) if 'time' in df.columns else \
                [last_time - timedelta(hours=len(df)-i-1) for i in range(len(df))]
            # Hiển thị tối đa 96h gần nhất để biểu đồ không quá dày
            display_hours = min(96, len(df))
            past_times = past_times[-display_hours:]
            past_temp = df['temperature'].values[-display_hours:]

            st.divider()
            st.markdown("### 🌟 Tổng quan Dự báo")

            current_temp = past_temp[-1]
            max_pred_temp = np.max(preds[:, 0])
            min_pred_temp = np.min(preds[:, 0])
            avg_pred_temp = np.mean(preds[:, 0])

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("🌡️ Nhiệt độ Hiện tại", f"{current_temp:.1f} °C")
            with col2: st.metric("🔥 Cao nhất dự báo", f"{max_pred_temp:.1f} °C", f"{max_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
            with col3: st.metric("❄️ Thấp nhất dự báo", f"{min_pred_temp:.1f} °C", f"{min_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
            with col4: st.metric("💧 Độ ẩm Trung bình", f"{np.mean(preds[:, 1]):.0f} %")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=avg_pred_temp, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Nhiệt độ Trung bình Dự báo", 'font': {'size': 20}}, number={'suffix': " °C", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [10, 45], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': "rgba(0,0,0,0.3)"},
                    'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                    'steps': [{'range': [10, 20], 'color': '#60A5FA'}, {'range': [20, 28], 'color': '#34D399'},
                              {'range': [28, 35], 'color': '#fbbf24'}, {'range': [35, 45], 'color': '#F87171'}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_pred_temp}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()
            st.markdown("### 📈 Phân tích Xu hướng Chi tiết")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=past_times, y=past_temp, mode='lines+markers',
                name='Thực tế', line=dict(color='#2563EB', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[past_times[-1]] + list(future_times),
                y=[past_temp[-1]] + list(preds[:, 0]),
                mode='lines+markers', name='Dự báo L-GRU',
                line=dict(color='#EF4444', width=3, dash='dash')
            ))
            fig.update_layout(
                title="Biểu đồ Dự báo Nhiệt độ Hà Tĩnh",
                xaxis_title="Thời gian (Giờ)", yaxis_title="Nhiệt độ (°C)",
                hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Biểu đồ lượng mưa (đã clip về 0 từ trước)
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=future_times, y=preds[:, 3],
                name='Lượng mưa dự báo', marker_color='#3B82F6'
            ))
            fig_rain.update_layout(
                title="Biểu đồ Dự báo Lượng mưa Hà Tĩnh",
                xaxis_title="Thời gian (Giờ)", yaxis_title="Lượng mưa (mm)",
                yaxis=dict(rangemode='nonnegative'),  # Trục Y không âm
                template="plotly_white"
            )
            st.plotly_chart(fig_rain, use_container_width=True)

            with st.expander("📊 Bảng dữ liệu chi tiết từng giờ"):
                res_df = pd.DataFrame({
                    "Thời gian": [t.strftime('%H:%M %d/%m') for t in future_times],
                    "Nhiệt độ (°C)": preds[:, 0].round(1),
                    "Độ ẩm (%)": preds[:, 1].round(1),
                    "Áp suất (hPa)": preds[:, 2].round(1),
                    "Lượng mưa (mm)": preds[:, 3].round(2)
                })
                st.dataframe(res_df, use_container_width=True)


# -------------------------------------------------------------
# 4. PHÂN HỆ TÀI CHÍNH
# -------------------------------------------------------------
def run_finance_app():
    st.markdown("<h1 style='color: #1E3A8A;'>📈 L-GRU: Dự báo Tài chính</h1>", unsafe_allow_html=True)
    st.write("**Áp dụng thuật toán L-GRU cho chuỗi thời gian giá cổ phiếu & tiền điện tử**")
    st.divider()

    st.sidebar.header("🔧 Cài đặt Model Tài chính")
    model_choice = st.sidebar.selectbox("Chọn mô hình dự báo:", ["Bitcoin", "SP500", "Tesla"])

    model_files = {
        "Bitcoin": "modeltaichinh/best_bitcoin_model.pth",
        "SP500": "modeltaichinh/best_sp500_model.pth",
        "Tesla": "modeltaichinh/best_tesla_model.pth"
    }

    # Mapping ticker và mốc train_end tương ứng
    ticker_map = {
        "Bitcoin": "BTC-USD",
        "SP500": "^GSPC",
        "Tesla": "TSLA"
    }
    # Mốc ngày cuối của tập training (tương đương split khi huấn luyện model)
    train_end_map = {
        "Bitcoin": "2025-01-01",
        "SP500": "2025-01-01",
        "Tesla": "2025-01-01"
    }
    # Bitcoin giao dịch 7 ngày/tuần, SP500/Tesla chỉ giao dịch ngày làm việc
    is_crypto = {"Bitcoin": True, "SP500": False, "Tesla": False}

    forecast_days = st.sidebar.slider("Dự báo bao nhiêu ngày tới?", min_value=1, max_value=30, value=7)

    st.markdown(f"### 🗓️ 1. Chọn ngày kết thúc dữ liệu cho {model_choice}")
    st.info("💡 Hệ thống sẽ **tự động kết nối và tải chuỗi dữ liệu giao dịch 60 ngày** trước đó từ Internet.")

    today = pd.Timestamp.now('Asia/Bangkok').date()
    min_date = today - timedelta(days=365*5)  # Lùi 5 năm
    selected_date = st.date_input("Ngày kết thúc dữ liệu muốn phân tích:", value=today, min_value=min_date, max_value=today)

    if st.button("🚀 TẢI DỮ LIỆU TỰ ĐỘNG & BẮT ĐẦU DỰ BÁO", use_container_width=True, type="primary", key="btn_finance"):
        with st.spinner(f"⏳ Đang kết nối thị trường tài chính để tải dữ liệu {model_choice}..."):
            import yfinance as yf

            ticker = ticker_map.get(model_choice)

            try:
                # Lấy số ngày đủ nhiều để bù ngày nghỉ lễ, cuối tuần
                start_date = selected_date - timedelta(days=100)
                st_str = start_date.strftime('%Y-%m-%d')
                en_str = (selected_date + timedelta(days=1)).strftime('%Y-%m-%d')

                df_yf = yf.download(ticker, start=st_str, end=en_str, progress=False)

                if df_yf.empty:
                    st.error(f"❌ Không có dữ liệu cho {model_choice} trong khoảng thời gian này.")
                    st.stop()

                df = df_yf.reset_index()

                # FIX LỖI 8: Flatten multi-index an toàn hơn để tránh duplicate column
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

                if 'Date' not in df.columns:
                    for col in ['Datetime', 'index', 'level_0']:
                        if col in df.columns:
                            df.rename(columns={col: 'Date'}, inplace=True)
                            break

                df = df[['Date', 'Close']]
                # Sử dụng ffill/bfill để giữ tính liên tục của time-series
                df.ffill(inplace=True)
                df.bfill(inplace=True)

                # Cắt 60 phiên cuối
                seq_len = 60
                if len(df) >= seq_len:
                    df = df.tail(seq_len).reset_index(drop=True)
                else:
                    df = df.reset_index(drop=True)

                st.success(f"✅ Đã tải thành công {len(df)} phiên giao dịch gần nhất!")
                with st.expander("👁️ Xem trước dữ liệu tự động tải"):
                    st.dataframe(df.tail(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Lỗi tải dữ liệu tài chính: {e}")
                st.stop()

        model_path = model_files[model_choice]
        model_data = load_finance_model(model_path)

        if model_data[0] is None:
            st.error(f"❌ Không tìm thấy file model AI tại {model_path}")
            st.stop()

        model, device = model_data

        # FIX LỖI 6: Load scaler đúng — chỉ fit đến mốc train_end, không bao gồm dữ liệu tương lai
        train_end = train_end_map[model_choice]
        scaler = load_finance_scaler(ticker, train_end_date=train_end)

        with st.spinner("🤖 AI L-GRU đang phân tích chuỗi dữ liệu..."):
            seq_len_actual = min(60, len(df))
            recent_data = df['Close'].values[-seq_len_actual:].reshape(-1, 1).astype(np.float32)

            data_scaled = scaler.transform(recent_data)

            # FIX LỖI 7: Dùng hàm dự báo riêng cho tài chính — không kéo về anchor cố định
            preds_scaled = predict_autoregressive_finance(model, data_scaled, steps=forecast_days, device=device)
            preds = scaler.inverse_transform(preds_scaled)

            st.markdown("### 🌟 Tổng quan Dự báo L-GRU")

            # Tính toán và hiển thị Metric Cards
            current_price = float(df['Close'].iloc[-1])
            final_predicted_price = float(preds[-1, 0])
            price_change = final_predicted_price - current_price
            pct_change = (price_change / current_price) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💵 Giá hiện tại", f"${current_price:,.2f}")
            with col2:
                st.metric(f"🎯 Dự báo (sau {forecast_days} ngày)", f"${final_predicted_price:,.2f}", f"{price_change:+,.2f} ({pct_change:+.2f}%)")
            with col3:
                if pct_change > 2:
                    st.info("🚀 **Tín hiệu AI:** Xu hướng TĂNG RÕ RỆT (Tích cực)")
                elif pct_change < -2:
                    st.error("📉 **Tín hiệu AI:** Xu hướng GIẢM (Thận trọng)")
                else:
                    st.warning("⚖️ **Tín hiệu AI:** Đi ngang / Biến động nhẹ")

            st.divider()
            st.markdown("### 📊 Phân tích Biểu đồ Trực quan")

            # Chuẩn bị dữ liệu thời gian tương lai
            last_date = pd.to_datetime(df['Date'].iloc[-1])

            # FIX LỖI 9: Dùng pd.bdate_range cho cổ phiếu (skip cuối tuần),
            # dùng timedelta thông thường cho Bitcoin (giao dịch 7 ngày/tuần)
            if is_crypto[model_choice]:
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            else:
                future_dates = pd.bdate_range(
                    start=last_date + timedelta(days=1), periods=forecast_days
                ).tolist()

            past_dates = pd.to_datetime(df['Date'].iloc[-30:])
            past_prices = df['Close'].iloc[-30:].values

            fig = go.Figure()

            # Line quá khứ
            fig.add_trace(go.Scatter(
                x=past_dates, y=past_prices,
                mode='lines+markers', name='Dữ liệu Thực tế',
                line=dict(color='#64748B', width=2)
            ))

            # Line dự báo với fill bóng mờ
            fig.add_trace(go.Scatter(
                x=[past_dates.iloc[-1]] + future_dates,
                y=[past_prices[-1]] + list(preds[:, 0]),
                mode='lines+markers', name='Dự báo AI (L-GRU)',
                line=dict(color='#10B981' if pct_change >= 0 else '#EF4444', width=3),
                fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)' if pct_change >= 0 else 'rgba(239, 68, 68, 0.1)'
            ))

            # Vạch kẻ dọc ngăn cách Hiện tại và Tương lai
            fig.add_vline(
                x=last_date.timestamp() * 1000,
                line_width=2, line_dash="dash", line_color="gray",
                annotation_text="← Thực tế | Dự báo →", annotation_position="top"
            )

            fig.update_layout(
                title=f"Lộ trình giá {model_choice} trong {forecast_days} ngày tới",
                xaxis_title="Thời gian (Ngày)",
                yaxis_title="Giá ($)",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Bảng chi tiết
            with st.expander("🧮 Xem chi tiết bảng giá dự báo từng ngày"):
                res_df = pd.DataFrame({
                    "Ngày": [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in future_dates],
                    "Giá dự báo ($)": preds[:, 0].round(2)
                })
                st.dataframe(res_df, use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# 5. KHỞI CHẠY ỨNG DỤNG
# -------------------------------------------------------------
def main():
    st.sidebar.title("🤖 Trợ lý L-GRU AI")
    st.sidebar.write("Chọn module phân tích bên dưới:")
    app_mode = st.sidebar.radio("Phân hệ:", ["🌦️ Dự báo Thời tiết", "📈 Dự báo Tài chính"])

    st.sidebar.divider()

    if app_mode == "🌦️ Dự báo Thời tiết":
        run_weather_app()
    elif app_mode == "📈 Dự báo Tài chính":
        run_finance_app()


if __name__ == '__main__':
    main()