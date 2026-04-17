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

# Cấu hình trang
st.set_page_config(
    page_title="Bảng điều khiển Dự báo AI L-GRU",
    page_icon="🤖",
    layout="wide"
)


# =============================================================================
# 1. KIẾN TRÚC L-GRU (khớp chính xác train_model_weatherr.py)
#    input_size=11, hidden_size=256, num_layers=3, output_size=7, seq_len=168
# =============================================================================
class LGRUCell(nn.Module):
    """
    Ô L-GRU (Layer-normalized GRU with Cell State + Peephole).
    Công thức:
      r_t = σ(LN(W_rh·h + W_rx·x + W_Cr⊙c + b_r))
      C̃_t = Mish(W_c·[r_t⊙h, x])
      z_t = σ(LN(W_zh·h + W_zx·x + W_Cz⊙c + b_z))
      C_t = z_t⊙c + (1-z_t)⊙C̃_t
      o_t = σ(LN(W_oh·h + W_ox·x + W_Co⊙C_t + b_o))
      h_t = o_t⊙tanh(C_t)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Cổng liên quan (Relevance Gate)
        self.W_rh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_rx = nn.Linear(input_size, hidden_size, bias=False)
        self.W_Cr = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.b_r  = nn.Parameter(torch.zeros(hidden_size))
        self.ln_r = nn.LayerNorm(hidden_size)

        # Cổng cập nhật (Update Gate)
        self.W_zh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_zx = nn.Linear(input_size, hidden_size, bias=False)
        self.W_Cz = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.b_z  = nn.Parameter(torch.zeros(hidden_size))
        self.ln_z = nn.LayerNorm(hidden_size)

        # Trạng thái ứng viên (Candidate State)
        self.W_c = nn.Linear(hidden_size + input_size, hidden_size)

        # Cổng đầu ra (Output Gate)
        self.W_oh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ox = nn.Linear(input_size, hidden_size, bias=False)
        self.W_Co = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.b_o  = nn.Parameter(torch.zeros(hidden_size))
        self.ln_o = nn.LayerNorm(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        r_t = torch.sigmoid(self.ln_r(
            self.W_rh(h_prev) + self.W_rx(x_t) + self.W_Cr * c_prev + self.b_r
        ))
        combined = torch.cat([r_t * h_prev, x_t], dim=-1)
        c_tilde = F.mish(self.W_c(combined))
        z_t = torch.sigmoid(self.ln_z(
            self.W_zh(h_prev) + self.W_zx(x_t) + self.W_Cz * c_prev + self.b_z
        ))
        c_t = z_t * c_prev + (1.0 - z_t) * c_tilde
        o_t = torch.sigmoid(self.ln_o(
            self.W_oh(h_prev) + self.W_ox(x_t) + self.W_Co * c_t + self.b_o
        ))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LGRU(nn.Module):
    """Mạng L-GRU xếp chồng nhiều tầng với Dropout giữa các tầng."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LGRUCell(in_size, hidden_size))
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else nn.Identity()

    def forward(self, x, initial_states=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        if initial_states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=device)
                 for _ in range(self.num_layers)]
        else:
            h, c = initial_states

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    input_t, h[layer_idx], c[layer_idx]
                )
                input_t = h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    input_t = self.dropout(input_t)
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c)


class TemporalAttention(nn.Module):
    """Cơ chế Attention cho chuỗi thời gian — tập trung các bước quan trọng."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1, bias=False)
        )

    def forward(self, lstm_output):
        scores = self.attn_score(lstm_output).squeeze(-1)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        context = (lstm_output * weights).sum(dim=1)
        return context


class WeatherForecastModel(nn.Module):
    """
    Model dự báo thời tiết L-GRU:
      Input (batch, seq_len=168, 11) → LGRU 3 tầng → concat[last_hidden, attention]
      → FC Head (512→256→128→7) → Output (batch, 7)
    Khớp chính xác với best_weather_model.pth (69 state keys)
    """

    def __init__(self, input_size=11, hidden_size=256,
                 num_layers=3, output_size=7, dropout=0.15):
        super().__init__()
        self.lgru = LGRU(input_size, hidden_size, num_layers, dropout)
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Mish(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        outputs, _ = self.lgru(x)
        last_out = outputs[:, -1, :]
        attn_out = self.attention(outputs)
        combined = torch.cat([last_out, attn_out], dim=-1)
        return self.fc(combined)


# =============================================================================
# 2. HÀM HỖ TRỢ: TIME FEATURES & INPUT BUILDER
# =============================================================================
def make_time_features(timestamp: pd.Timestamp) -> np.ndarray:
    """
    Tạo 4 đặc trưng thời gian tuần hoàn (khớp training):
      [hour_sin, hour_cos, day_sin, day_cos]
    """
    hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
    hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
    day_sin  = np.sin(2 * np.pi * timestamp.dayofyear / 365.25)
    day_cos  = np.cos(2 * np.pi * timestamp.dayofyear / 365.25)
    return np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)


def build_input_sequence(
    weather_scaled: np.ndarray,   # (N, 5): Scaled [temp, hum, pres, prec, wind_speed]
    wind_dir_sin: np.ndarray,     # (N,)
    wind_dir_cos: np.ndarray,     # (N,)
    timestamps: list,             # list of pd.Timestamp, len=N
    seq_len: int = 168
) -> np.ndarray:
    """
    Ghép: weather_scaled (N,5) + wind_sincos (N,2) + time_features (N,4) → (N,11).
    Cắt seq_len bước cuối → (seq_len, 11).
    """
    n = len(timestamps)
    time_feats = np.array([make_time_features(t) for t in timestamps], dtype=np.float32)
    wind_sincos = np.column_stack([wind_dir_sin, wind_dir_cos]).astype(np.float32)
    combined = np.concatenate([weather_scaled, wind_sincos, time_feats], axis=1)

    if n >= seq_len:
        return combined[-seq_len:]
    else:
        pad = np.zeros((seq_len - n, 11), dtype=np.float32)
        return np.concatenate([pad, combined], axis=0)


# =============================================================================
# 3. PHYSICS CLIP & AUTOREGRESSIVE PREDICTION
# =============================================================================
def _clip_to_scaler_range(pred_real: np.ndarray, scaler) -> np.ndarray:
    """
    Clip dự báo về phạm vi mà scaler đã học [data_min, data_max].
    Đảm bảo giá trị scaled luôn trong [0, 1] — tránh input ngoài phân phối
    khi chạy autoregressive. Nếu không clip, model nhận input nó chưa bao giờ
    thấy khi training → drift tích lũy qua nhiều bước dự báo.
    """
    for i in range(len(pred_real)):
        pred_real[i] = np.clip(pred_real[i], scaler.data_min_[i], scaler.data_max_[i])
    pred_real[3] = max(pred_real[3], 0.0)  # mưa ≥ 0
    pred_real[4] = max(pred_real[4], 0.0)  # gió ≥ 0
    return pred_real


def _apply_seasonal_clip(pred_real: np.ndarray, month: int) -> np.ndarray:
    """
    Clip vật lý theo mùa cho Hà Tĩnh — chỉ dùng cho hiển thị kết quả cuối.
    Bounds nới rộng hơn scaler range để cho phép các giá trị cực đoan hợp lý.
    pred_real: [temp, humidity, pressure, precipitation, wind_speed]
    """
    if month in (11, 12, 1, 2, 3):
        t_max = 35.0
    elif month in (6, 7, 8):
        t_max = 43.0
    else:
        t_max = 40.0
    pred_real[0] = np.clip(pred_real[0],  5.0,   t_max)   # nhiệt độ
    pred_real[1] = np.clip(pred_real[1],  0.0, 100.0)     # độ ẩm
    pred_real[2] = np.clip(pred_real[2], 960.0, 1050.0)   # áp suất
    pred_real[3] = np.clip(pred_real[3],  0.0,  None)     # mưa ≥ 0
    pred_real[4] = np.clip(pred_real[4],  0.0,  60.0)     # gió
    return pred_real


def predict_autoregressive_weather(
    model, input_seq, steps, device,
    scaler=None,
    start_timestamp=None,
    n_samples: int = 15,
):
    """
    Dự báo autoregressive với MC Dropout ensemble.
    Output model: 7 giá trị [5 Scaled weather + 2 wind_dir_sin/cos]
    Trả về: (mean_preds, std_preds) — shape (steps, 7) trong scaled space
    """
    current_ts = start_timestamp if start_timestamp is not None else pd.Timestamp.now()
    base_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device)

    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    all_runs = []

    for _ in range(n_samples):
        model.eval()
        model.apply(enable_dropout)

        current_seq = base_seq.clone()
        ts = current_ts
        run_preds = []

        with torch.no_grad():
            for _step in range(steps):
                pred_t = model(current_seq)
                pred_np = pred_t.cpu().numpy()[0]        # (7,)

                pred_scaled_5 = pred_np[:5].copy()
                pred_wind_sincos = pred_np[5:7].copy()

                # Inverse → clip vật lý → re-scale (5 biến weather)
                if scaler is not None:
                    pred_real = scaler.inverse_transform(pred_scaled_5.reshape(1, -1))[0]
                    pred_real = _clip_to_scaler_range(pred_real, scaler)
                    pred_scaled_5 = scaler.transform(pred_real.reshape(1, -1))[0]
                else:
                    pred_scaled_5[3] = max(pred_scaled_5[3], 0.0)
                    pred_scaled_5[4] = max(pred_scaled_5[4], 0.0)

                # Normalize wind sin/cos → unit circle
                norm = np.sqrt(pred_wind_sincos[0]**2 + pred_wind_sincos[1]**2) + 1e-8
                pred_wind_sincos = pred_wind_sincos / norm

                pred_combined = np.concatenate([pred_scaled_5, pred_wind_sincos])
                run_preds.append(pred_combined.copy())

                # Build next input (11 features)
                time_feat = make_time_features(ts)
                next_row = np.concatenate([pred_scaled_5, pred_wind_sincos, time_feat])
                next_tensor = torch.FloatTensor(next_row).unsqueeze(0).unsqueeze(0).to(device)
                current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)
                ts += pd.Timedelta(hours=1)

        all_runs.append(np.array(run_preds))

    model.eval()
    all_runs = np.array(all_runs)
    mean_scaled = all_runs.mean(axis=0)
    std_scaled  = all_runs.std(axis=0)
    return mean_scaled, std_scaled


# =============================================================================
# 4. TIỆN ÍCH HIỂN THỊ
# =============================================================================
def degrees_to_cardinal(deg):
    """Chuyển góc (°) sang hướng la bàn 16 phương vị."""
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = round(deg / 22.5) % 16
    return dirs[idx]


CARDINAL_VN = {
    'N': 'Bắc', 'NNE': 'Bắc Đông Bắc', 'NE': 'Đông Bắc', 'ENE': 'Đông Đông Bắc',
    'E': 'Đông', 'ESE': 'Đông Đông Nam', 'SE': 'Đông Nam', 'SSE': 'Nam Đông Nam',
    'S': 'Nam', 'SSW': 'Nam Tây Nam', 'SW': 'Tây Nam', 'WSW': 'Tây Tây Nam',
    'W': 'Tây', 'WNW': 'Tây Tây Bắc', 'NW': 'Tây Bắc', 'NNW': 'Bắc Tây Bắc',
}


def beaufort_description(speed_ms):
    """Mô tả gió theo thang Beaufort (tiếng Việt)."""
    if speed_ms < 0.3:  return "Lặng gió"
    if speed_ms < 1.6:  return "Gió rất nhẹ"
    if speed_ms < 3.4:  return "Gió nhẹ"
    if speed_ms < 5.5:  return "Gió nhẹ vừa"
    if speed_ms < 8.0:  return "Gió vừa"
    if speed_ms < 10.8: return "Gió khá mạnh"
    if speed_ms < 13.9: return "Gió mạnh"
    if speed_ms < 17.2: return "Gió rất mạnh"
    if speed_ms < 20.8: return "Gió bão"
    if speed_ms < 24.5: return "Bão mạnh"
    if speed_ms < 28.5: return "Bão rất mạnh"
    if speed_ms < 32.7: return "Bão dữ dội"
    return "Siêu bão"


def get_weather_emoji(temp, rain_mm, wind_ms):
    """Emoji mô tả điều kiện thời tiết tổng quát."""
    if wind_ms > 17:   return "🌪️"
    if rain_mm > 10:   return "⛈️"
    if rain_mm > 2:    return "🌧️"
    if rain_mm > 0.2:  return "🌦️"
    if temp > 37:      return "🔥"
    if temp > 30:      return "☀️"
    if temp > 22:      return "⛅"
    if temp > 15:      return "🌤️"
    return "❄️"


# =============================================================================
# 5. LOAD MODEL & SCALER (CACHE)
# =============================================================================
@st.cache_resource
def load_station_meta():
    """Đọc tọa độ và timezone trạm đo từ station_meta.json."""
    import json
    for p in ["model/model/station_meta.json"]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"latitude": 18.3428, "longitude": 105.9056,
            "timezone": "Asia/Bangkok",
            "station_name": "Hà Tĩnh (fallback)"}


@st.cache_resource
def load_model_config():
    """Đọc hyperparameters từ config.json."""
    import json
    for p in ["model/model/config.json"]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"input_size": 11, "hidden_size": 256, "num_layers": 3,
            "output_size": 7, "dropout": 0.15, "seq_len": 168}


@st.cache_resource
def load_weather_model_and_scaler():
    """
    Load model L-GRU + scaler thời tiết.
    Trả về: (model, device, scaler, seq_len)
    """
    import joblib

    model_candidates = [
        "model/model/best_weather_model.pth",
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        return None, None, None, 168

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_model_config()

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    input_size  = cfg.get('input_size',  11)
    hidden_size = cfg.get('hidden_size', 256)
    num_layers  = cfg.get('num_layers',  3)
    output_size = cfg.get('output_size', 7)
    seq_len     = cfg.get('seq_len',     168)
    dropout     = cfg.get('dropout',     0.15)

    model = WeatherForecastModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    ).to(device)

    state_dict = ckpt['model_state']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Scaler (5 features: temp, hum, pres, prec, wind_speed)
    scaler_candidates = [
        "model/model/weather_scaler.pkl",
    ]
    scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        st.warning("⚠️ Không tìm thấy weather_scaler.pkl — dùng Fallback.")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        _min = np.array([  5.0,   0.0,  960.0,  0.0,  0.0])
        _max = np.array([ 45.0, 100.0, 1045.0, 80.0, 92.0])
        scaler.fit(np.vstack([_min, _max]))

    return model, device, scaler, seq_len


# =============================================================================
# 6. ỨNG DỤNG DỰ BÁO THỜI TIẾT
# =============================================================================
def run_weather_app():
    col1, col2 = st.columns([1, 4])
    with col1:
        logo_path = "LogoHTU.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
    with col2:
        st.markdown("<h1 style='color: #1E3A8A;'>🌤️ L-GRU: Dự báo Thời tiết Hà Tĩnh</h1>", unsafe_allow_html=True)
        st.write("**Bản thử nghiệm Beta — Mô hình L-GRU (Dữ liệu 1990-2026, 6 biến thời tiết)**")

    st.divider()

    # Load model
    model, device, scaler, SEQ_LEN = load_weather_model_and_scaler()

    if model is None:
        st.error("❌ Không tìm thấy file model `model/model/best_weather_model.pth`.")
        st.stop()
    if scaler is None:
        st.error("❌ Không thể khởi tạo Scaler cho dữ liệu thời tiết.")
        st.stop()

    st.markdown('### 🗓️ 1. Thiết lập Dự báo')
    st.info(
        'Hệ thống AI L-GRU sẽ tự động tải dữ liệu quan trắc thực tế '
        f'({SEQ_LEN} giờ = {SEQ_LEN // 24} ngày gần nhất), '
        'phân tích 6 biến thời tiết (nhiệt độ, độ ẩm, áp suất, lượng mưa, tốc độ gió, hướng gió) '
        'và đưa ra dự báo tương lai.'
    )
    forecast_hours = st.slider(
        'Dự báo bao nhiêu giờ tiếp theo (tính từ thời điểm hiện tại)?',
        min_value=1, max_value=72, value=24
    )

    if st.button("🚀 TẢI DỮ LIỆU TỰ ĐỘNG & BẮT ĐẦU DỰ BÁO", width="stretch",
                 type="primary", key="btn_weather"):

        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 1: TẢI DỮ LIỆU TỪ API
        # ═══════════════════════════════════════════════════════════════
        with st.spinner("⏳ Đang tải dữ liệu thời tiết thực tế từ Open-Meteo API..."):
            import urllib.request
            import json
            import datetime

            now_bkk = pd.Timestamp.now('Asia/Bangkok').replace(minute=0, second=0, microsecond=0)
            target_dt = now_bkk.to_pydatetime()
            today = now_bkk.date()
            hours_needed = SEQ_LEN + 24
            start_dt = target_dt - timedelta(hours=hours_needed)

            start_str = start_dt.strftime('%Y-%m-%d')
            end_str = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')

            _meta = load_station_meta()
            _lat = _meta.get('latitude',  18.3428)
            _lon = _meta.get('longitude', 105.9056)
            _tz  = _meta.get('timezone',  'Asia/Bangkok').replace('/', '%2F')

            api_vars = ("temperature_2m,relative_humidity_2m,surface_pressure,"
                        "precipitation,wind_speed_10m,wind_direction_10m")

            days_diff = 0  # Luôn dùng Forecast API (realtime)
            if days_diff <= 3:
                url = (f"https://api.open-meteo.com/v1/forecast?"
                       f"latitude={_lat}&longitude={_lon}"
                       f"&past_days=10&forecast_days=1"
                       f"&hourly={api_vars}"
                       f"&timezone={_tz}")
            else:
                url = (f"https://archive-api.open-meteo.com/v1/archive?"
                       f"latitude={_lat}&longitude={_lon}"
                       f"&start_date={start_str}&end_date={end_str}"
                       f"&hourly={api_vars}"
                       f"&timezone={_tz}")

            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data_api = json.loads(response.read().decode())

                hourly = data_api.get('hourly', {})
                if not hourly:
                    st.error("❌ Không thể lấy dữ liệu từ API.")
                    st.stop()

                df_all = pd.DataFrame({
                    'time':          hourly['time'],
                    'temperature':   hourly['temperature_2m'],
                    'humidity':      hourly['relative_humidity_2m'],
                    'pressure':      hourly['surface_pressure'],
                    'precipitation': hourly['precipitation'],
                    'wind_speed':    hourly['wind_speed_10m'],
                    'wind_direction': hourly['wind_direction_10m'],
                })

                df_all['time'] = pd.to_datetime(df_all['time'])
                df_past = df_all[df_all['time'] <= pd.to_datetime(target_dt).tz_localize(None)]

                if len(df_past) >= SEQ_LEN:
                    df = df_past.tail(SEQ_LEN).reset_index(drop=True)
                elif len(df_past) > 0:
                    df = df_past.reset_index(drop=True)
                else:
                    df = df_all.head(SEQ_LEN).reset_index(drop=True)

                # Xử lý missing data
                df[['temperature', 'humidity', 'pressure']] = (
                    df[['temperature', 'humidity', 'pressure']]
                    .interpolate(method='linear').bfill().ffill()
                )
                df['precipitation'] = df['precipitation'].fillna(0)
                df['wind_speed'] = df['wind_speed'].interpolate(method='linear').bfill().ffill().fillna(0)
                # Wind direction: nội suy circular qua sin/cos (tránh artifact 350°→10° bị thành 180°)
                _wd = df['wind_direction'].copy()
                _nan_wd = _wd.isna()
                if _nan_wd.any():
                    _rad_wd = np.radians(_wd.values.astype(float))
                    _sin_wd = pd.Series(np.where(_nan_wd, np.nan, np.sin(_rad_wd))).interpolate('linear').bfill().ffill().fillna(0)
                    _cos_wd = pd.Series(np.where(_nan_wd, np.nan, np.cos(_rad_wd))).interpolate('linear').bfill().ffill().fillna(1)
                    df['wind_direction'] = (np.degrees(np.arctan2(_sin_wd.values, _cos_wd.values)) + 360) % 360
                df['wind_direction'] = df['wind_direction'].fillna(0)

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

        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 2: CHUẨN BỊ INPUT & DỰ BÁO
        # ═══════════════════════════════════════════════════════════════
        with st.spinner("🤖 AI L-GRU v2 đang tính toán dự báo (MC Dropout ensemble)..."):
            # Scale 5 biến weather
            raw_cols = ['temperature', 'humidity', 'pressure', 'precipitation', 'wind_speed']
            data_raw = df[raw_cols].values.astype(np.float32)

            # Clip vật lý trước scale
            data_raw[:, 0] = np.clip(data_raw[:, 0],  5.0,  44.0)
            data_raw[:, 1] = np.clip(data_raw[:, 1],  0.0, 100.0)
            data_raw[:, 2] = np.clip(data_raw[:, 2], 960.0, 1050.0)
            data_raw[:, 3] = np.clip(data_raw[:, 3],  0.0,  None)
            data_raw[:, 4] = np.clip(data_raw[:, 4],  0.0,  60.0)
            data_scaled = scaler.transform(data_raw)  # (N, 5)

            # Wind direction → sin/cos
            wind_dir_rad = np.radians(df['wind_direction'].values)
            wind_dir_sin = np.sin(wind_dir_rad).astype(np.float32)
            wind_dir_cos = np.cos(wind_dir_rad).astype(np.float32)

            timestamps_list = pd.to_datetime(df['time']).tolist()
            input_seq = build_input_sequence(
                data_scaled, wind_dir_sin, wind_dir_cos,
                timestamps_list, seq_len=SEQ_LEN
            )  # (SEQ_LEN, 11)

            last_ts = pd.to_datetime(df['time'].iloc[-1])
            first_forecast_ts = last_ts + pd.Timedelta(hours=1)

            # MC Dropout ensemble
            preds_scaled_mean, preds_scaled_std = predict_autoregressive_weather(
                model, input_seq, steps=forecast_hours,
                device=device, scaler=scaler,
                start_timestamp=first_forecast_ts,
                n_samples=15,
            )

            # ── Inverse transform ra đơn vị thật ──
            # 5 biến weather (cols 0-4)
            preds_real = scaler.inverse_transform(preds_scaled_mean[:, :5])   # (steps, 5)
            # Std: MinMaxScaler là tuyến tính → std_real = std_scaled × (max - min)
            scale_range = scaler.data_max_ - scaler.data_min_
            preds_std_real = preds_scaled_std[:, :5] * scale_range            # (steps, 5)

            # Wind direction từ sin/cos
            wind_dir_pred = np.degrees(np.arctan2(
                preds_scaled_mean[:, 5], preds_scaled_mean[:, 6]
            ))
            wind_dir_pred = (wind_dir_pred + 360) % 360   # [0, 360)

            # Hậu xử lý: clip vật lý theo mùa (chỉ cho hiển thị, KHÔNG ảnh hưởng AR loop)
            for i, ft in enumerate([first_forecast_ts + pd.Timedelta(hours=h)
                                     for h in range(forecast_hours)]):
                preds_real[i] = _apply_seasonal_clip(preds_real[i].copy(), ft.month)
            preds_std_real[:, 3] = np.clip(preds_std_real[:, 3], 0, None)
            preds_std_real[:, 4] = np.clip(preds_std_real[:, 4], 0, None)

            # ── Trích biến dễ dùng ──
            pred_temp  = preds_real[:, 0]
            pred_hum   = preds_real[:, 1]
            pred_pres  = preds_real[:, 2]
            pred_prec  = preds_real[:, 3]
            pred_wspd  = preds_real[:, 4]
            pred_wdir  = wind_dir_pred

            std_temp   = preds_std_real[:, 0]
            std_prec   = preds_std_real[:, 3]
            std_wspd   = preds_std_real[:, 4]

            # ── Timeline ──
            try:
                last_time = pd.to_datetime(df['time'].iloc[-1])
            except Exception:
                last_time = pd.Timestamp.now()

            future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
            future_hours = [t.hour for t in future_times]

            # ── Thống kê ──
            day_mask   = np.array([6 <= h <= 18 for h in future_hours])
            night_mask = ~day_mask

            avg_day_temp   = np.mean(pred_temp[day_mask])   if day_mask.any()   else np.nan
            avg_night_temp = np.mean(pred_temp[night_mask]) if night_mask.any() else np.nan

            # Heat Index (Steadman)
            T_arr = pred_temp
            RH_arr = pred_hum
            hi_arr = np.where(
                T_arr >= 27,
                -8.784695 + 1.61139411*T_arr + 2.338549*RH_arr
                - 0.14611605*T_arr*RH_arr - 0.012308094*T_arr**2
                - 0.016424828*RH_arr**2 + 0.002211732*T_arr**2*RH_arr
                + 0.00072546*T_arr*RH_arr**2 - 0.000003582*T_arr**2*RH_arr**2,
                T_arr
            )

            pct_rain_hours = np.mean(pred_prec > 0.1) * 100
            total_rain     = pred_prec.sum()
            avg_wspd       = np.mean(pred_wspd)
            max_wspd       = np.max(pred_wspd)

            # Hướng gió chủ đạo (circular mean)
            mean_sin = np.mean(preds_scaled_mean[:, 5])
            mean_cos = np.mean(preds_scaled_mean[:, 6])
            dominant_dir_deg = (np.degrees(np.arctan2(mean_sin, mean_cos)) + 360) % 360
            dominant_cardinal = degrees_to_cardinal(dominant_dir_deg)
            dominant_cardinal_vn = CARDINAL_VN.get(dominant_cardinal, dominant_cardinal)

            # Past data cho biểu đồ
            past_times    = pd.to_datetime(df['time'].tolist())
            display_hours = min(168, len(df))
            past_times    = past_times[-display_hours:]
            past_temp     = df['temperature'].values[-display_hours:]
            past_humidity = df['humidity'].values[-display_hours:]
            past_pressure = df['pressure'].values[-display_hours:]
            past_wspd     = df['wind_speed'].values[-display_hours:]
            past_wdir     = df['wind_direction'].values[-display_hours:]

        # ═══════════════════════════════════════════════════════════════
        # BƯỚC 3: HIỂN THỊ KẾT QUẢ
        # ═══════════════════════════════════════════════════════════════
        st.divider()
        st.markdown("### 🌟 Tổng quan Dự báo")

        current_temp = past_temp[-1]
        max_pred_temp = np.max(pred_temp)
        min_pred_temp = np.min(pred_temp)
        avg_pred_temp = np.mean(pred_temp)
        avg_hi = np.mean(hi_arr[T_arr >= 27]) if (T_arr >= 27).any() else avg_pred_temp

        weather_icon = get_weather_emoji(avg_pred_temp, total_rain / max(forecast_hours, 1), avg_wspd)

        # ── Hàng 1: Nhiệt độ & Độ ẩm ──
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
            st.metric("💧 Độ ẩm Trung bình", f"{np.mean(pred_hum):.0f} %")

        # ── Hàng 2: Ngày/Đêm & Mưa ──
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

        # ── Hàng 3: Gió ──
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            beaufort = beaufort_description(avg_wspd)
            st.metric("💨 Tốc độ gió TB", f"{avg_wspd:.1f} m/s", beaufort)
        with col10:
            st.metric("🌪️ Gió mạnh nhất", f"{max_wspd:.1f} m/s",
                      beaufort_description(max_wspd))
        with col11:
            st.metric("🧭 Hướng gió chủ đạo",
                      f"{dominant_cardinal_vn}",
                      f"{dominant_dir_deg:.0f}°")
        with col12:
            st.metric(f"{weather_icon} Nhận định chung",
                      beaufort_description(max_wspd) if max_wspd > 10
                      else ("Mưa" if total_rain > 5 else "Ổn định"))

        # ── Đồng hồ đo nhiệt độ ──
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

        # ═══════════════════════════════════════════════════════════════
        # BIỂU ĐỒ 1: NHIỆT ĐỘ (kèm dải bất định ±1σ)
        # ═══════════════════════════════════════════════════════════════
        temp_upper = list(pred_temp + std_temp)
        temp_lower = list(pred_temp - std_temp)
        future_times_str = list(future_times)

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=future_times_str + future_times_str[::-1],
            y=temp_upper + temp_lower[::-1],
            fill='toself', fillcolor='rgba(239,68,68,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Khoảng tin cậy ±1σ', showlegend=True
        ))
        fig_temp.add_trace(go.Scatter(
            x=list(past_times), y=list(past_temp), mode='lines+markers',
            name='Thực tế', line=dict(color='#2563EB', width=2),
            marker=dict(size=3)
        ))
        fig_temp.add_trace(go.Scatter(
            x=[past_times[-1]] + future_times_str,
            y=[past_temp[-1]] + list(pred_temp),
            mode='lines+markers', name='Dự báo L-GRU (mean)',
            line=dict(color='#EF4444', width=3, dash='dash'),
            marker=dict(size=4)
        ))
        fig_temp.update_layout(
            title="🌡️ Dự báo Nhiệt độ Hà Tĩnh (có dải bất định)",
            xaxis_title="Thời gian", yaxis_title="Nhiệt độ (°C)",
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_temp, width="stretch")

        # ═══════════════════════════════════════════════════════════════
        # BIỂU ĐỒ 2: LƯỢNG MƯA (bar chart + sai số)
        # ═══════════════════════════════════════════════════════════════
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=future_times, y=pred_prec,
            name='Lượng mưa (mean)', marker_color='#3B82F6',
            error_y=dict(type='data', array=std_prec, visible=True)
        ))
        fig_rain.update_layout(
            title="🌧️ Dự báo Lượng mưa Hà Tĩnh (có sai số)",
            xaxis_title="Thời gian", yaxis_title="Lượng mưa (mm)",
            yaxis=dict(rangemode='nonnegative'),
            template="plotly_white"
        )
        st.plotly_chart(fig_rain, width="stretch")

        # ═══════════════════════════════════════════════════════════════
        # BIỂU ĐỒ 3-4: ĐỘ ẨM & ÁP SUẤT (2 cột)
        # ═══════════════════════════════════════════════════════════════
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Scatter(
                x=list(past_times), y=list(past_humidity), mode='lines+markers',
                name='Thực tế', line=dict(color='#10B981', width=2),
                marker=dict(size=3)
            ))
            fig_hum.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_humidity[-1]] + list(pred_hum),
                mode='lines+markers', name='Dự báo',
                line=dict(color='#FBBF24', width=3, dash='dash')
            ))
            fig_hum.update_layout(
                title="💧 Dự báo Độ ẩm",
                xaxis_title="Thời gian", yaxis_title="Độ ẩm (%)",
                hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig_hum, width="stretch")

        with col_chart2:
            fig_pres = go.Figure()
            fig_pres.add_trace(go.Scatter(
                x=list(past_times), y=list(past_pressure), mode='lines+markers',
                name='Thực tế', line=dict(color='#8B5CF6', width=2),
                marker=dict(size=3)
            ))
            fig_pres.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_pressure[-1]] + list(pred_pres),
                mode='lines+markers', name='Dự báo',
                line=dict(color='#A78BFA', width=3, dash='dash')
            ))
            fig_pres.update_layout(
                title="☁️ Dự báo Áp suất",
                xaxis_title="Thời gian", yaxis_title="Áp suất (hPa)",
                hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig_pres, width="stretch")

        # ═══════════════════════════════════════════════════════════════
        # BIỂU ĐỒ 5-6: TỐC ĐỘ GIÓ & HƯỚNG GIÓ (2 cột)
        # ═══════════════════════════════════════════════════════════════
        col_wind1, col_wind2 = st.columns(2)

        with col_wind1:
            wspd_upper = list(pred_wspd + std_wspd)
            wspd_lower = list(np.clip(pred_wspd - std_wspd, 0, None))

            fig_wind = go.Figure()
            # Uncertainty band
            fig_wind.add_trace(go.Scatter(
                x=future_times_str + future_times_str[::-1],
                y=wspd_upper + wspd_lower[::-1],
                fill='toself', fillcolor='rgba(59,130,246,0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1σ', showlegend=True
            ))
            fig_wind.add_trace(go.Scatter(
                x=list(past_times), y=list(past_wspd), mode='lines+markers',
                name='Thực tế', line=dict(color='#0EA5E9', width=2),
                marker=dict(size=3)
            ))
            fig_wind.add_trace(go.Scatter(
                x=[past_times[-1]] + future_times_str,
                y=[past_wspd[-1]] + list(pred_wspd),
                mode='lines+markers', name='Dự báo',
                line=dict(color='#F97316', width=3, dash='dash'),
                marker=dict(size=4)
            ))
            # Ngưỡng cảnh báo
            fig_wind.add_hline(y=10.8, line_dash="dash", line_color="orange",
                               annotation_text="Gió mạnh (10.8 m/s)")
            fig_wind.add_hline(y=17.2, line_dash="dash", line_color="red",
                               annotation_text="Bão (17.2 m/s)")
            fig_wind.update_layout(
                title="💨 Dự báo Tốc độ gió (có dải bất định)",
                xaxis_title="Thời gian", yaxis_title="Tốc độ gió (m/s)",
                yaxis=dict(rangemode='nonnegative'),
                hovermode="x unified", template="plotly_white"
            )
            st.plotly_chart(fig_wind, width="stretch")

        with col_wind2:
            # Biểu đồ hoa gió polar (Wind Rose)
            fig_polar = go.Figure()

            # Quá khứ (48h gần nhất)
            past_display_polar = min(48, len(past_wspd))
            fig_polar.add_trace(go.Scatterpolar(
                r=past_wspd[-past_display_polar:],
                theta=past_wdir[-past_display_polar:],
                mode='markers',
                marker=dict(color='#94A3B8', size=5, opacity=0.5),
                name=f'Quá khứ ({past_display_polar}h)',
            ))

            # Dự báo (với gradient màu theo thời gian)
            fig_polar.add_trace(go.Scatterpolar(
                r=pred_wspd,
                theta=pred_wdir,
                mode='markers+lines',
                marker=dict(
                    color=list(range(forecast_hours)),
                    colorscale='YlOrRd',
                    size=8,
                    colorbar=dict(title="Giờ", len=0.6),
                    line=dict(width=1, color='white')
                ),
                line=dict(color='rgba(249,115,22,0.3)', width=1),
                name='Dự báo',
            ))

            fig_polar.update_layout(
                title="🧭 Hoa gió Dự báo (hướng & tốc độ)",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        title="m/s",
                        range=[0, max(max(pred_wspd) * 1.2, 5)]
                    ),
                    angularaxis=dict(
                        direction="clockwise",
                        rotation=90,
                        tickmode='array',
                        tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                        ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    )
                ),
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15)
            )
            st.plotly_chart(fig_polar, width="stretch")

        # ═══════════════════════════════════════════════════════════════
        # BIỂU ĐỒ 7: HEAT INDEX (cảm giác nóng thực tế)
        # ═══════════════════════════════════════════════════════════════
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
            title="🌡️ Chỉ số Heat Index (Cảm giác nóng thực tế)",
            xaxis_title="Thời gian", yaxis_title="Heat Index (°C)",
            template="plotly_white"
        )
        st.plotly_chart(fig_hi, width="stretch")

        # ═══════════════════════════════════════════════════════════════
        # BẢNG DỮ LIỆU CHI TIẾT
        # ═══════════════════════════════════════════════════════════════
        with st.expander("📊 Bảng dữ liệu chi tiết từng giờ"):
            cardinal_dirs = [degrees_to_cardinal(d) for d in pred_wdir]
            cardinal_vn   = [CARDINAL_VN.get(c, c) for c in cardinal_dirs]

            res_df = pd.DataFrame({
                "Thời gian"           : [t.strftime('%H:%M %d/%m') for t in future_times],
                "Nhiệt độ (°C)"       : pred_temp.round(1),
                "±σ Nhiệt độ"         : std_temp.round(2),
                "Heat Index (°C)"     : hi_arr.round(1),
                "Độ ẩm (%)"           : pred_hum.round(1),
                "Áp suất (hPa)"       : pred_pres.round(1),
                "Lượng mưa (mm)"      : pred_prec.round(2),
                "±σ Mưa"              : std_prec.round(2),
                "Tốc độ gió (m/s)"    : pred_wspd.round(1),
                "±σ Gió"              : std_wspd.round(2),
                "Hướng gió (°)"       : pred_wdir.round(0).astype(int),
                "Hướng gió"           : cardinal_vn,
            })
            st.dataframe(res_df, width="stretch")


def main():
    st.sidebar.title('🤖 Trợ lý L-GRU AI')
    run_weather_app()

if __name__ == '__main__':
    main()
