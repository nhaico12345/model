import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import os
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*X không có tên các đặc trưng hợp lệ.*")
from datetime import timedelta

# Cấu hình trang tổng
st.set_page_config(
    page_title="Bảng điều khiển Dự báo AI L-GRU",
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
def _apply_physics_clip(pred_real: np.ndarray, month: int) -> np.ndarray:
    """
    Clip vật lý theo mùa cho Hà Tĩnh:
    - Mùa đông (11-3): nhiệt độ max thấp hơn do gió Bắc
    - Mùa hè  (4-10): nhiệt độ có thể lên tới 42°C (gió Lào)
    """
    if month in (11, 12, 1, 2, 3):   # mùa đông
        t_max = 32.0
    elif month in (6, 7, 8):          # đỉnh hè (gió Lào)
        t_max = 42.0
    else:
        t_max = 38.0
    pred_real[0] = np.clip(pred_real[0],  5.0,   t_max)    # nhiệt độ
    pred_real[1] = np.clip(pred_real[1],  0.0,  100.0)     # độ ẩm
    pred_real[2] = np.clip(pred_real[2], 960.0, 1050.0)    # áp suất
    pred_real[3] = np.clip(pred_real[3],  0.0,  None)      # mưa
    return pred_real


def predict_autoregressive_weather(
    model, input_seq, steps, device,
    scaler=None,
    start_timestamp=None,
    n_samples: int = 15,          # số lần MC Dropout — tăng lên cho accuracy tốt hơn
):
    """
    Dự báo autoregressive với Monte Carlo Dropout ensemble.

    Chạy n_samples lần với dropout bật → lấy mean ± std.
    mean  : ước lượng điểm tốt nhất
    std   : uncertainty (độ không chắc chắn)

    Trả về: (mean_scaled, std_scaled) — mỗi array shape (steps, 4)
    """
    current_ts = start_timestamp if start_timestamp is not None else pd.Timestamp.now()
    base_seq   = torch.FloatTensor(input_seq).unsqueeze(0).to(device)  # (1, seq_len, 10)

    # Kích hoạt dropout khi inference (MC Dropout)
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    all_runs = []  # (n_samples, steps, 4) — scaled

    for _ in range(n_samples):
        model.eval()
        model.apply(enable_dropout)   # chỉ bật dropout, giữ BatchNorm eval

        current_seq = base_seq.clone()
        ts          = current_ts
        run_preds   = []

        with torch.no_grad():
            for _step in range(steps):
                pred_t  = model(current_seq)               # (1, 4)
                pred_np = pred_t.cpu().numpy()[0]           # (4,)

                if scaler is not None:
                    pred_real = scaler.inverse_transform(pred_np.reshape(1, -1))[0]
                    pred_real = _apply_physics_clip(pred_real, ts.month)
                    pred_scaled = scaler.transform(pred_real.reshape(1, -1))[0]
                else:
                    pred_scaled = pred_np.copy()
                    pred_scaled[3] = max(pred_scaled[3], 0.0)

                run_preds.append(pred_scaled.copy())

                time_feat   = make_time_features(ts)
                next_row    = np.concatenate([pred_scaled, time_feat])
                next_tensor = torch.FloatTensor(next_row).unsqueeze(0).unsqueeze(0).to(device)
                current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)
                ts += pd.Timedelta(hours=1)

        all_runs.append(np.array(run_preds))  # (steps, 4)

    model.eval()  # khôi phục eval hoàn toàn
    all_runs = np.array(all_runs)          # (n_samples, steps, 4)
    mean_scaled = all_runs.mean(axis=0)    # (steps, 4)
    std_scaled  = all_runs.std(axis=0)     # (steps, 4)
    return mean_scaled, std_scaled


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
def load_station_meta():
    """Đọc tọa độ và timezone trạm đo từ station_meta.json."""
    import json
    for p in ["station_meta.json", "model/station_meta.json"]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    # Fallback mặc định Hà Tĩnh
    return {"latitude": 18.3333, "longitude": 105.9000,
            "timezone": "Asia/Ho_Chi_Minh",
            "station_name": "Hà Tĩnh (fallback)"}


@st.cache_resource
def load_model_config():
    """Đọc hyperparameters từ config.json (ưu tiên hơn hard-code)."""
    import json
    for p in ["config.json", "model/config.json"]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    # Fallback cứng nếu không có file
    return {"input_size": 10, "hidden_size": 256, "num_layers": 3,
            "output_size": 4, "dropout": 0.25, "seq_len": 96}


@st.cache_resource
def load_weather_model_and_scaler():
    """
    Load model + scaler thời tiết.
    Ưu tiên đọc cấu hình từ config.json, tìm file theo nhiều đường dẫn.
    Trả về: (model, device, scaler, seq_len)
    """
    import joblib

    # Tìm file model theo thứ tự ưu tiên
    model_candidates = [
        "best_weather_model(release).pth",
        "model/best_weather_model.pth",
        "model/best_weather_model(release).pth",
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        return None, None, None, 96

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Đọc config từ file JSON trước (nguồn đáng tin cậy nhất)
    cfg_file = load_model_config()

    # Đọc checkpoint và merge (checkpoint override config nếu có)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get('config', {})

    input_size  = ckpt.get('input_size',  cfg_file.get('input_size',  10))
    hidden_size = ckpt.get('hidden_size', cfg_file.get('hidden_size', 256))
    num_layers  = ckpt.get('num_layers',  cfg_file.get('num_layers',  3))
    output_size = ckpt.get('output_size', cfg_file.get('output_size', 4))
    seq_len     = ckpt.get('seq_len',     cfg_file.get('seq_len',     96))
    dropout     = ckpt_cfg.get('dropout', cfg_file.get('dropout',     0.25))

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

    # Tìm scaler theo thứ tự ưu tiên
    scaler_candidates = [
        "weather_scaler.pkl",
        "model/weather_scaler.pkl",
        "weather_scaler.joblib",
    ]
    scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        st.warning("⚠️ Không tìm thấy weather_scaler.pkl — dùng Fallback Hà Tĩnh. Kết quả có thể kém chính xác.")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        _min = np.array([5.0,   0.0,   960.0,  0.0])
        _max = np.array([45.0, 100.0, 1045.0, 600.0])
        scaler.fit(np.vstack([_min, _max]))

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
    st.info("💡Hệ thống sẽ **tự động kết nối vệ tinh và tải dữ liệu thời tiết** 48 giờ trước thời điểm bạn chọn.")

    col_date, col_time = st.columns(2)
    with col_date:
        today = pd.Timestamp.now('Asia/Bangkok').date()
        min_date = today - timedelta(days=365*2)  # Cho phép lùi 2 năm
        selected_date = st.date_input("Ngày dự báo:", value=today, min_value=min_date, max_value=today)
    with col_time:
        if "default_time" not in st.session_state:
            st.session_state["default_time"] = pd.Timestamp.now('Asia/Bangkok').replace(second=0, microsecond=0).time()
        selected_time = st.time_input("Giờ dự báo:", value=st.session_state["default_time"])

    if st.button("🚀 TẢI DỮ LIỆU TỰ ĐỘNG & BẮT ĐẦU DỰ BÁO", width="stretch", type="primary", key="btn_weather"):
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
            # Đọc tọa độ từ station_meta.json (nhất quán với training)
            _meta = load_station_meta()
            _lat  = _meta.get('latitude',  18.3333)
            _lon  = _meta.get('longitude', 105.9000)
            _tz   = _meta.get('timezone',  'Asia/Ho_Chi_Minh').replace('/', '%2F')

            days_diff = (today - selected_date).days
            if days_diff <= 3:
                url = (f"https://api.open-meteo.com/v1/forecast?"
                       f"latitude={_lat}&longitude={_lon}"
                       f"&past_days=7&forecast_days=1"
                       f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation"
                       f"&timezone={_tz}")
            else:
                url = (f"https://archive-api.open-meteo.com/v1/archive?"
                       f"latitude={_lat}&longitude={_lon}"
                       f"&start_date={start_str}&end_date={end_str}"
                       f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation"
                       f"&timezone={_tz}")

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
                # Chuyển đổi toàn bộ cột time sang đối tượng datetime
                df_all['time'] = pd.to_datetime(df_all['time'])
                # Lọc bằng đối tượng datetime trực tiếp
                df_past = df_all[df_all['time'] <= pd.to_datetime(target_dt)]

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
                    st.dataframe(df.tail(10), width="stretch")

            except Exception as e:
                st.error(f"❌ Lỗi tự động tải dữ liệu: {e}")
                st.stop()

        # ── Bắt đầu dự báo ──
        with st.spinner("🤖 AI L-GRU đang tính toán ước lượng tốt nhất..."):
            weather_features = ['temperature', 'humidity', 'pressure', 'precipitation']
            data_raw = df[weather_features].values.astype(np.float32)

            # Clip vật lý trước khi scale
            data_raw[:, 0] = np.clip(data_raw[:, 0],  5.0,  44.0)
            data_raw[:, 1] = np.clip(data_raw[:, 1],  0.0, 100.0)
            data_raw[:, 2] = np.clip(data_raw[:, 2], 960.0, 1050.0)
            data_raw[:, 3] = np.clip(data_raw[:, 3],  0.0,  None)
            data_scaled = scaler.transform(data_raw)  # (N, 4)

            timestamps_list = pd.to_datetime(df['time']).tolist()
            input_seq = build_input_sequence_with_time(
                data_scaled, timestamps_list, seq_len=SEQ_LEN
            )  # (SEQ_LEN, 10)

            last_ts = pd.to_datetime(df['time'].iloc[-1])
            first_forecast_ts = last_ts + pd.Timedelta(hours=1)

            # MC Dropout ensemble → mean + std (uncertainty)
            preds_scaled_mean, preds_scaled_std = predict_autoregressive_weather(
                model, input_seq, steps=forecast_hours,
                device=device, scaler=scaler,
                start_timestamp=first_forecast_ts,
                n_samples=15,
            )

            # Inverse transform để ra đơn vị thật
            preds      = scaler.inverse_transform(preds_scaled_mean)  # (steps, 4)
            preds_std  = scaler.inverse_transform(
                np.clip(preds_scaled_std, 0, None)
            )  # std cũng inverse để cùng đơn vị

            # ── Hậu xử lý: clip vật lý theo mùa ──
            for i, ft in enumerate([first_forecast_ts + pd.Timedelta(hours=h)
                                     for h in range(forecast_hours)]):
                preds[i] = _apply_physics_clip(preds[i].copy(), ft.month)
            preds_std[:, 3] = np.clip(preds_std[:, 3], 0, None)  # std mưa không âm

            # ── Thống kê bổ sung ──
            try:
                last_time = pd.to_datetime(df['time'].iloc[-1])
            except Exception:
                last_time = pd.Timestamp.now()

            future_times  = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
            future_hours  = [t.hour for t in future_times]

            # Ngày / Đêm: ngày = 6h-18h, đêm còn lại
            day_mask   = np.array([6 <= h <= 18 for h in future_hours])
            night_mask = ~day_mask

            avg_day_temp   = np.mean(preds[day_mask,   0]) if day_mask.any()   else np.nan
            avg_night_temp = np.mean(preds[night_mask, 0]) if night_mask.any() else np.nan

            # Heat Index (Steadman, đơn giản hóa) — chỉ có nghĩa khi T > 27°C
            T_arr = preds[:, 0]
            RH_arr = preds[:, 1]
            hi_arr = np.where(
                T_arr >= 27,
                -8.784695 + 1.61139411*T_arr + 2.338549*RH_arr
                - 0.14611605*T_arr*RH_arr - 0.012308094*T_arr**2
                - 0.016424828*RH_arr**2 + 0.002211732*T_arr**2*RH_arr
                + 0.00072546*T_arr*RH_arr**2 - 0.000003582*T_arr**2*RH_arr**2,
                T_arr
            )

            # Xác suất mưa: nếu std lớn hoặc mean > 0 → mưa có khả năng cao
            rain_prob = np.clip(
                (preds[:, 3] / (preds_std[:, 3] + 1e-3)).clip(0, 1) * 100, 0, 100
            )
            # Đơn giản hơn: tỉ lệ giờ có mưa (mean > 0.1mm)
            pct_rain_hours = np.mean(preds[:, 3] > 0.1) * 100
            total_rain     = preds[:, 3].sum()

            past_times = pd.to_datetime(df['time'].tolist()) if 'time' in df.columns else \
                [last_time - timedelta(hours=len(df)-i-1) for i in range(len(df))]
            display_hours = min(96, len(df))
            past_times    = past_times[-display_hours:]
            past_temp     = df['temperature'].values[-display_hours:]
            past_humidity = df['humidity'].values[-display_hours:]
            past_pressure = df['pressure'].values[-display_hours:]

            st.divider()
            st.markdown("### 🌟 Tổng quan Dự báo")

            current_temp  = past_temp[-1]
            max_pred_temp = np.max(preds[:, 0])
            min_pred_temp = np.min(preds[:, 0])
            avg_pred_temp = np.mean(preds[:, 0])
            avg_hi        = np.mean(hi_arr[T_arr >= 27]) if (T_arr >= 27).any() else avg_pred_temp

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌡️ Nhiệt độ Hiện tại", f"{current_temp:.1f} °C")
            with col2:
                st.metric("🔥 Cao nhất dự báo",
                          f"{max_pred_temp:.1f} °C",
                          f"{max_pred_temp - current_temp:+.1f} °C",
                          delta_color="inverse")
            with col3:
                st.metric("❄️ Thấp nhất dự báo",
                          f"{min_pred_temp:.1f} °C",
                          f"{min_pred_temp - current_temp:+.1f} °C",
                          delta_color="inverse")
            with col4:
                st.metric("💧 Độ ẩm Trung bình", f"{np.mean(preds[:, 1]):.0f} %")

            # Hàng 2: thống kê bổ sung
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("☀️ Nhiệt độ TB Ngày",
                          f"{avg_day_temp:.1f} °C" if not np.isnan(avg_day_temp) else "N/A")
            with col6:
                st.metric("🌙 Nhiệt độ TB Đêm",
                          f"{avg_night_temp:.1f} °C" if not np.isnan(avg_night_temp) else "N/A")
            with col7:
                st.metric("🌧️ Tổng lượng mưa", f"{total_rain:.1f} mm",
                          f"{pct_rain_hours:.0f}% giờ có mưa")
            with col8:
                hi_val = float(avg_hi)
                hi_label = ("🟢 Thoải mái" if hi_val < 32
                            else "🟡 Nóng" if hi_val < 38
                            else "🔴 Rất nóng/Nguy hiểm")
                st.metric("🌡️ Heat Index TB", f"{hi_val:.1f} °C", hi_label)

            # Đồng hồ đo nhiệt độ
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=avg_pred_temp,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Nhiệt độ Trung bình Dự báo", 'font': {'size': 20}},
                number={'suffix': " °C", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [10, 45], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0.3)"},
                    'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                    'steps': [
                        {'range': [10, 20], 'color': '#60A5FA'},
                        {'range': [20, 28], 'color': '#34D399'},
                        {'range': [28, 35], 'color': '#fbbf24'},
                        {'range': [35, 45], 'color': '#F87171'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': max_pred_temp}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

            st.divider()
            st.markdown("### 📈 Phân tích Xu hướng Chi tiết")

            # Uncertainty band (±1σ) cho nhiệt độ
            temp_upper = list(preds[:, 0] + preds_std[:, 0])
            temp_lower = list(preds[:, 0] - preds_std[:, 0])
            future_times_str = list(future_times)

            fig = go.Figure()
            # Dải uncertainty (fill giữa upper và lower)
            fig.add_trace(go.Scatter(
                x=future_times_str + future_times_str[::-1],
                y=temp_upper + temp_lower[::-1],
                fill='toself', fillcolor='rgba(239,68,68,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Khoảng tin cậy ±1σ', showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=past_times, y=past_temp, mode='lines+markers',
                name='Thực tế', line=dict(color='#2563EB', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_temp[-1]] + list(preds[:, 0]),
                mode='lines+markers', name='Dự báo L-GRU (mean)',
                line=dict(color='#EF4444', width=3, dash='dash')
            ))
            fig.update_layout(
                title="Biểu đồ Dự báo Nhiệt độ Hà Tĩnh (có dải bất định)",
                xaxis_title="Thời gian", yaxis_title="Nhiệt độ (°C)",
                hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch")

            # Biểu đồ lượng mưa với uncertainty
            rain_upper = np.clip(preds[:, 3] + preds_std[:, 3], 0, None)
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=future_times, y=preds[:, 3],
                name='Lượng mưa (mean)', marker_color='#3B82F6',
                error_y=dict(type='data', array=preds_std[:, 3], visible=True)
            ))
            fig_rain.update_layout(
                title="Biểu đồ Dự báo Lượng mưa Hà Tĩnh (có sai số)",
                xaxis_title="Thời gian", yaxis_title="Lượng mưa (mm)",
                yaxis=dict(rangemode='nonnegative'),
                template="plotly_white"
            )
            st.plotly_chart(fig_rain, width="stretch")

            # Độ ẩm và Áp suất
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Scatter(
                x=past_times, y=past_humidity, mode='lines+markers',
                name='Thực tế', line=dict(color='#10B981', width=3)
            ))
            fig_hum.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_humidity[-1]] + list(preds[:, 1]),
                mode='lines+markers', name='Dự báo',
                line=dict(color='#FBBF24', width=3, dash='dash')
            ))
            fig_hum.update_layout(
                title="Dự báo Độ ẩm",
                xaxis_title="Thời gian", yaxis_title="Độ ẩm (%)",
                hovermode="x unified", template="plotly_white"
            )

            fig_pres = go.Figure()
            fig_pres.add_trace(go.Scatter(
                x=past_times, y=past_pressure, mode='lines+markers',
                name='Thực tế', line=dict(color='#8B5CF6', width=3)
            ))
            fig_pres.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_pressure[-1]] + list(preds[:, 2]),
                mode='lines+markers', name='Dự báo',
                line=dict(color='#A78BFA', width=3, dash='dash')
            ))
            fig_pres.update_layout(
                title="Dự báo Áp suất",
                xaxis_title="Thời gian", yaxis_title="Áp suất (hPa)",
                hovermode="x unified", template="plotly_white"
            )

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.plotly_chart(fig_hum, width="stretch")
            with col_chart2:
                st.plotly_chart(fig_pres, width="stretch")

            # Heat Index chart
            fig_hi = go.Figure()
            fig_hi.add_trace(go.Scatter(
                x=future_times, y=hi_arr, mode='lines',
                name='Heat Index', line=dict(color='#F97316', width=2),
                fill='tozeroy', fillcolor='rgba(249,115,22,0.1)'
            ))
            fig_hi.add_hline(y=32, line_dash="dash", line_color="orange",
                             annotation_text="Cảnh báo nóng (32°C)")
            fig_hi.add_hline(y=38, line_dash="dash", line_color="red",
                             annotation_text="Nguy hiểm (38°C)")
            fig_hi.update_layout(
                title="Chỉ số Heat Index (Cảm giác nóng thực tế)",
                xaxis_title="Thời gian", yaxis_title="Heat Index (°C)",
                template="plotly_white"
            )
            st.plotly_chart(fig_hi, width="stretch")

            with st.expander("📊 Bảng dữ liệu chi tiết từng giờ"):
                res_df = pd.DataFrame({
                    "Thời gian"       : [t.strftime('%H:%M %d/%m') for t in future_times],
                    "Nhiệt độ (°C)"   : preds[:, 0].round(1),
                    "±σ Nhiệt độ"     : preds_std[:, 0].round(2),
                    "Heat Index (°C)" : hi_arr.round(1),
                    "Độ ẩm (%)"       : preds[:, 1].round(1),
                    "Áp suất (hPa)"   : preds[:, 2].round(1),
                    "Lượng mưa (mm)"  : preds[:, 3].round(2),
                    "±σ Mưa"          : preds_std[:, 3].round(2),
                })
                st.dataframe(res_df, width="stretch")


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

    if st.button("🚀 TẢI DỮ LIỆU TỰ ĐỘNG & BẮT ĐẦU DỰ BÁO", width="stretch", type="primary", key="btn_finance"):
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

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

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
                    st.dataframe(df.tail(), width="stretch")

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

            st.plotly_chart(fig, width="stretch")

            # Bảng chi tiết
            with st.expander("🧮 Xem chi tiết bảng giá dự báo từng ngày"):
                res_df = pd.DataFrame({
                    "Ngày": [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in future_dates],
                    "Giá dự báo ($)": preds[:, 0].round(2)
                })
                st.dataframe(res_df, width="stretch", hide_index=True)


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