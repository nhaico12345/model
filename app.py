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

# --- Kiến trúc L-GRU MỚI (dùng cho model thời tiết Hà Tĩnh 1940-2026) ---
class WeatherLGRUCell(nn.Module):
    """Cell L-GRU mới: tách riêng W_rh (hidden) và W_rx (input) cho từng cổng."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Cổng Reset
        self.W_Cr = nn.Parameter(torch.Tensor(hidden_size))
        self.W_rh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_rx = nn.Linear(input_size, hidden_size)
        self.ln_r = nn.LayerNorm(hidden_size)

        # Candidate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        # Cổng Update
        self.W_Cz = nn.Parameter(torch.Tensor(hidden_size))
        self.W_zh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_zx = nn.Linear(input_size, hidden_size)
        self.ln_z = nn.LayerNorm(hidden_size)

        # Cổng Output
        self.W_Co = nn.Parameter(torch.Tensor(hidden_size))
        self.W_oh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ox = nn.Linear(input_size, hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)

        self.mish = Mish()

    def forward(self, x_t, states):
        h_prev, c_prev = states

        # Reset gate
        r_t = torch.sigmoid(self.ln_r(self.W_rh(h_prev) + self.W_rx(x_t) + self.W_Cr * c_prev))

        # Candidate
        r_hx = torch.cat([x_t, r_t * h_prev], dim=1)
        c_tilde = self.mish(self.W_c(r_hx))

        # Update gate
        z_t = torch.sigmoid(self.ln_z(self.W_zh(h_prev) + self.W_zx(x_t) + self.W_Cz * c_prev))
        c_t = z_t * c_prev + (1 - z_t) * c_tilde

        # Output gate
        o_t = torch.sigmoid(self.ln_o(self.W_oh(h_prev) + self.W_ox(x_t) + self.W_Co * c_t))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class _WeatherRNNCells(nn.Module):
    """Wrapper chứa 'cells' ModuleList để key names khớp với 'rnn.cells.*' trong checkpoint."""
    def __init__(self):
        super().__init__()
        self.cells = nn.ModuleList()

class WeatherLGRUModel(nn.Module):
    """Model L-GRU mới cho dự báo thời tiết Hà Tĩnh (4 features in/out).
    Kiến trúc khớp đúng với state dict keys: rnn.cells.0.*, rnn.cells.1.*, output_layer.*
    """
    def __init__(self, input_size=4, hidden_size=128, output_size=4, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Dùng _WeatherRNNCells để tạo key 'rnn.cells.*' khớp với checkpoint
        self.rnn = _WeatherRNNCells()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.rnn.cells.append(WeatherLGRUCell(in_sz, hidden_size))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        states = [
            (torch.zeros(batch_size, self.hidden_size).to(x.device),
             torch.zeros(batch_size, self.hidden_size).to(x.device))
            for _ in range(self.num_layers)
        ]
        h_t = None
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.rnn.cells):
                h_t, c_t = cell(x_t, states[i])
                states[i] = (h_t, c_t)
                x_t = h_t
        return self.output_layer(h_t)

def predict_autoregressive(model, input_seq, steps, device):
    current_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_seq)
            predictions.append(pred.cpu().numpy()[0])
            pred_unsqueeze = pred.unsqueeze(1)
            current_seq = torch.cat([current_seq[:, 1:, :], pred_unsqueeze], dim=1)
            
    return np.array(predictions)

# -------------------------------------------------------------
# 2. HÀM XỬ LÝ CACHE MODEL VÀ SCALER (TÁCH BIỆT CHO 2 PHÂN HỆ)
# -------------------------------------------------------------
@st.cache_resource
def load_weather_model_and_scaler():
    """Load model thời tiết Hà Tĩnh mới (4 features) và fit scaler từ dữ liệu training."""
    from sklearn.preprocessing import MinMaxScaler
    model_path = "best_weather_model.pth"
    if not os.path.exists(model_path):
        return None, None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLGRUModel(input_size=4, hidden_size=128, output_size=4, num_layers=2).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state']
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # Tạo scaler phù hợp với dữ liệu huấn luyện Hà Tĩnh 1940-2026
    # Dùng min/max chuẩn từ bộ dữ liệu gốc (weather_trainn.csv)
    features = ['temperature', 'humidity', 'pressure', 'precipitation']
    scaler = MinMaxScaler()
    train_path = "weather_trainn.csv"
    if os.path.exists(train_path):
        # Đọc theo chunks để tiết kiệm RAM với file >700k dòng
        data_min = None
        data_max = None
        for chunk in pd.read_csv(train_path, usecols=features, chunksize=50000):
            chunk_min = chunk[features].min().values
            chunk_max = chunk[features].max().values
            data_min = chunk_min if data_min is None else np.minimum(data_min, chunk_min)
            data_max = chunk_max if data_max is None else np.maximum(data_max, chunk_max)
        # Fit scaler bằng min/max đã tính
        dummy = np.vstack([data_min, data_max])
        scaler.fit(dummy)
    else:
        # Hardcode min/max đã biết từ dữ liệu thực khi file train không tồn tại
        # [temperature, humidity, pressure, precipitation]
        data_min = np.array([7.2,  30.0,  975.8, 0.0])
        data_max = np.array([40.8, 100.0, 1035.1, 77.9])
        dummy = np.vstack([data_min, data_max])
        scaler.fit(dummy)

    return model, device, scaler

@st.cache_resource
def load_finance_scaler(df):
    # CHUYỂN LỆNH IMPORT VÀO ĐÂY ĐỂ TRÁNH LỖI ATEXIT CỦA JOBLIB
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    scaler.fit(df[['Close']].values)
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

    # Load model mới kèm scaler từ checkpoint
    model, device, scaler = load_weather_model_and_scaler()

    if model is None:
        st.error("❌ Không tìm thấy file model thời tiết `best_weather_model.pth`.")
        st.stop()
    if scaler is None:
        st.error("❌ Không thể khôi phục Scaler từ checkpoint model.")
        st.stop()
    
    forecast_hours = st.sidebar.slider("Dự báo bao nhiêu giờ tiếp theo?", min_value=1, max_value=48, value=24)
    
    st.markdown("### 📥 1. Tải lên dữ liệu thời tiết 24-48 giờ qua của Hà Tĩnh")
    st.info("💡 File CSV cần có các cột: `time`, `temperature`, `humidity`, `pressure`, `precipitation`")
    
    uploaded_file = st.file_uploader("Chọn file định dạng CSV hoặc Excel", type=["csv", "xlsx"], key="weather_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success("Tải dữ liệu thành công!")
        
        with st.expander("👁️ Xem trước dữ liệu tải lên", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        # Model mới dùng 4 features
        features = ['temperature', 'humidity', 'pressure', 'precipitation']
        missing = [c for c in features if c not in df.columns]
        if missing:
            # Nếu thiếu precipitation thì tự điền 0
            if missing == ['precipitation']:
                df['precipitation'] = 0.0
                st.warning("⚠️ Cột `precipitation` không có trong file — tự động điền giá trị 0.")
            else:
                st.error(f"❌ File thiếu các cột bắt buộc: {missing}")
                st.stop()
            
        if st.button("🚀 BẮT ĐẦU DỰ BÁO", use_container_width=True, type="primary", key="btn_weather"):
            with st.spinner("Bộ AI đang xử lý mô phỏng..."):
                data = df[features].values.astype(np.float32)
                data_scaled = scaler.transform(data)
                
                preds_scaled = predict_autoregressive(model, data_scaled, steps=forecast_hours, device=device)
                preds = scaler.inverse_transform(preds_scaled)
                
                try:
                    last_time = pd.to_datetime(df['time'].iloc[-1])
                except Exception:
                    last_time = pd.Timestamp.now()
                    
                future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
                past_times = pd.to_datetime(df['time'].iloc[-48:]).tolist() if 'time' in df.columns else [last_time - timedelta(hours=48-i) for i in range(min(48, len(df)))]
                past_temp = df['temperature'].iloc[-48:].values
                
                st.divider()
                st.markdown("### 🌟 Tổng quan Dự báo")
                
                current_temp = past_temp[-1]
                max_pred_temp = np.max(preds[:, 0])
                min_pred_temp = np.min(preds[:, 0])
                avg_pred_temp = np.mean(preds[:, 0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("🌡️ Nhiệt độ Hiện tại", f"{current_temp:.1f} °C")
                with col2: st.metric("🔥 Cao nhất", f"{max_pred_temp:.1f} °C", f"{max_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
                with col3: st.metric("❄️ Thấp nhất", f"{min_pred_temp:.1f} °C", f"{min_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
                with col4: st.metric("💧 Độ ẩm Trung bình", f"{np.mean(preds[:, 1]):.0f} %")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = avg_pred_temp, domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Nhiệt độ Trung bình Dự báo", 'font': {'size': 20}}, number = {'suffix': " °C", 'font': {'size': 40}},
                    gauge = {
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
                fig.add_trace(go.Scatter(x=past_times, y=past_temp, mode='lines+markers', name='Thực tế hiện tại', line=dict(color='#2563EB', width=3)))
                fig.add_trace(go.Scatter(x=[past_times[-1]] + list(future_times), y=[past_temp[-1]] + list(preds[:, 0]), mode='lines+markers', name='Dự báo (L-GRU Hà Tĩnh)', line=dict(color='#EF4444', width=3, dash='dash')))
                fig.update_layout(title="Biểu đồ Dự báo Nhiệt độ Hà Tĩnh", xaxis_title="Thời gian (Giờ)", yaxis_title="Nhiệt độ (°C)", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Biểu đồ lượng mưa
                fig_rain = go.Figure()
                fig_rain.add_trace(go.Bar(x=future_times, y=preds[:, 3].clip(min=0), name='Lượng mưa dự báo', marker_color='#3B82F6'))
                fig_rain.update_layout(title="Biểu đồ Dự báo Lượng mưa Hà Tĩnh", xaxis_title="Thời gian (Giờ)", yaxis_title="Lượng mưa (mm)", template="plotly_white")
                st.plotly_chart(fig_rain, use_container_width=True)
                
                with st.expander("📊 Bảng dữ liệu chi tiết từng giờ"):
                    res_df = pd.DataFrame({
                        "Thời gian": future_times,
                        "Nhiệt độ (°C)": preds[:, 0].round(2),
                        "Độ ẩm (%)": preds[:, 1].round(2),
                        "Áp suất (hPa)": preds[:, 2].round(2),
                        "Lượng mưa (mm)": preds[:, 3].clip(min=0).round(2)
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
    
    forecast_days = st.sidebar.slider("Dự báo bao nhiêu ngày tới?", min_value=1, max_value=30, value=7)
    
    st.markdown(f"### 📥 Tải lên dữ liệu test của {model_choice} (Định dạng CSV)")
    uploaded_file = st.file_uploader("Chọn file CSV dữ liệu", type=["csv"], label_visibility="collapsed", key="finance_uploader")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("Dữ liệu cần phải chứa cột 'Date' và 'Close'")
            st.stop()
            
        with st.expander("👁️ Xem trước dữ liệu", expanded=False):
            st.dataframe(df.tail(10), width='stretch')
            
        model_path = model_files[model_choice]
        model_data = load_finance_model(model_path)
        
        if model_data[0] is None:
            st.error(f"❌ Không tìm thấy trọng số tại {model_path}")
            st.stop()
            
        model, device = model_data
        scaler = load_finance_scaler(df)

        if st.button("🚀 BẮT ĐẦU DỰ BÁO", width='stretch', type="primary", key="btn_finance"):
            with st.spinner("AI đang phân tích chuỗi dữ liệu..."):
                seq_len = min(60, len(df))
                recent_data = df['Close'].values[-seq_len:].reshape(-1, 1)
                
                data_scaled = scaler.transform(recent_data)
                preds_scaled = predict_autoregressive(model, data_scaled, steps=forecast_days, device=device)
                preds = scaler.inverse_transform(preds_scaled)
                
                st.markdown("### 🌟 Tổng quan Dự báo L-GRU")
                
                # --- PHẦN MỚI 1: TÍNH TOÁN VÀ HIỂN THỊ METRIC CARDS ---
                current_price = df['Close'].iloc[-1]
                final_predicted_price = preds[-1, 0] # Giá ở ngày cuối cùng của chu kỳ dự báo
                price_change = final_predicted_price - current_price
                pct_change = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💵 Giá hiện tại", f"${current_price:,.2f}")
                with col2:
                    st.metric(f"🎯 Dự báo (sau {forecast_days} ngày)", f"${final_predicted_price:,.2f}", f"{price_change:+,.2f} ({pct_change:+.2f}%)")
                with col3:
                    # Hiển thị tín hiệu dễ hiểu cho người không chuyên
                    if pct_change > 2:
                        st.info("🚀 **Tín hiệu AI:** Xu hướng TĂNG RÕ RỆT (Tích cực)")
                    elif pct_change < -2:
                        st.error("📉 **Tín hiệu AI:** Xu hướng GIẢM (Thận trọng)")
                    else:
                        st.warning("⚖️ **Tín hiệu AI:** Đi ngang / Biến động nhẹ")
                
                st.divider()
                st.markdown("### 📊 Phân tích Biểu đồ Trực quan")
                
                # Chuẩn bị dữ liệu thời gian
                last_date = pd.to_datetime(df['Date'].iloc[-1])
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                past_dates = pd.to_datetime(df['Date'].iloc[-30:])
                past_prices = df['Close'].iloc[-30:].values
                
                # --- PHẦN MỚI 2: NÂNG CẤP BIỂU ĐỒ PLOTLY ---
                fig = go.Figure()
                
                # Line quá khứ
                fig.add_trace(go.Scatter(
                    x=past_dates, y=past_prices, 
                    mode='lines+markers', name='Dữ liệu Thực tế', 
                    line=dict(color='#64748B', width=2) # Đổi màu xám nhạt hơn để nhường sân khấu cho phần dự báo
                ))
                
                # Line dự báo (Thêm fill='tozeroy' để tạo bóng mờ)
                fig.add_trace(go.Scatter(
                    x=[past_dates.iloc[-1]] + future_dates, 
                    y=[past_prices[-1]] + list(preds[:, 0]), 
                    mode='lines+markers', name='Dự báo AI (L-GRU)', 
                    line=dict(color='#10B981' if pct_change >= 0 else '#EF4444', width=3), # Đổi màu xanh/đỏ tùy xu hướng
                    fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)' if pct_change >= 0 else 'rgba(239, 68, 68, 0.1)'
                ))
                
                # Thêm vạch kẻ dọc ngăn cách Hiện tại và Tương lai
                # SỬA LỖI Ở ĐÂY: Chuyển Timestamp sang mili-giây để tránh lỗi TypeError của Plotly
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
                
                st.plotly_chart(fig, width='stretch')
                
                # Bảng chi tiết
                with st.expander("🧮 Xem chi tiết bảng giá dự báo từng ngày"):
                    res_df = pd.DataFrame({
                        "Ngày": [d.strftime('%Y-%m-%d') for d in future_dates],
                        "Giá dự báo ($)": preds[:, 0].round(2)
                    })
                    st.dataframe(res_df, width='stretch', hide_index=True)
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