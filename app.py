import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
from datetime import timedelta

st.set_page_config(
    page_title="L-GRU Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# -------------------------------------------------------------
# 1. KHỞI TẠO KIẾN TRÚC MẠNG TÙY CHỈNH THEO THUẬT TOÁN AI
# -------------------------------------------------------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

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

# -------------------------------------------------------------
# 2. HÀM XỬ LÝ CACHE MODEL VÀ SCALER
# -------------------------------------------------------------
@st.cache_resource
def load_scaler():
    train_path = r"d:\tranining model AI\manymodel\weather_train.csv"
    if not os.path.exists(train_path):
        st.error(f"❌ Không tìm thấy bộ dữ liệu huấn luyện gốc tại {train_path} để thiết lập Scaler.")
        return None
        
    df_train = pd.read_csv(train_path)
    features = ['temperature', 'humidity', 'pressure']
    data = df_train[features].values
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler

@st.cache_resource
def load_ai_model():
    model_path = r"d:\tranining model AI\manymodel\best_ai_model.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy trọng số model tại {model_path}")
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomAIModel(input_size=3, hidden_size=256, output_size=3, num_layers=3).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    
    # Xử lý khóa "_orig_mod." do compile() của Pytorch
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '')
        new_state_dict[new_k] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, device

# -------------------------------------------------------------
# 3. HÀM DỰ BÁO
# -------------------------------------------------------------
def predict_autoregressive(model, input_seq, steps, device):
    current_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device) # [1, seq_len, 3]
    predictions = []
    
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_seq) # [1, 3]
            predictions.append(pred.cpu().numpy()[0])
            
            # Đẩy giá trị dự báo mới vào chuỗi và loại giá trị cũ nhất
            pred_unsqueeze = pred.unsqueeze(1) # [1, 1, 3]
            current_seq = torch.cat([current_seq[:, 1:, :], pred_unsqueeze], dim=1)
            
    return np.array(predictions)

# -------------------------------------------------------------
# 4. GIAO DIỆN CHÍNH
# -------------------------------------------------------------
def main():
    # Logo và Tiêu đề
    col1, col2 = st.columns([1, 4])
    with col1:
        logo_path = r"d:\tranining model AI\manymodel\LogoHTU.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
    with col2:
        st.markdown("<h1 style='color: #1E3A8A;'>L-GRU: Thuật toán AI Lai mới nhằm Cải thiện Dự báo Chuỗi Thời gian</h1>", unsafe_allow_html=True)
        st.write("**Bản thử nghiệm Beta - Dự báo Thời tiết Khu vực Hà Tĩnh**")
        
    st.divider()
    
    # Load Model và Scaler
    scaler = load_scaler()
    model, device = load_ai_model()
    
    if scaler is None or model is None:
        st.stop()
        
    st.sidebar.header("Tùy chọn Dự báo")
    forecast_hours = st.sidebar.slider("Dự báo bao nhiêu giờ tiếp theo?", min_value=1, max_value=48, value=24)
    
    st.markdown("### 📥 1. Tải lên dữ liệu thời tiết 24-48 giờ qua của Hà Tĩnh")
    st.info("💡 Bạn có thể chạy file `download_test_data.py` để tự động tải file `hatinh_weather_test.csv` mới nhất.")
    
    uploaded_file = st.file_uploader("Chọn file định dạng CSV hoặc Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Tải dữ liệu thành công!")
        
        with st.expander("👁️ Xem trước dữ liệu tải lên", expanded=False):
            st.dataframe(df.head(10), width='stretch')
            
        features = ['temperature', 'humidity', 'pressure']
        if not all(col in df.columns for col in features):
            st.error(f"File phải chứa các cột: {features}")
            st.stop()
            
        # Nút bấm bắt đầu
        if st.button("🚀 BẮT ĐẦU DỰ BÁO BẰNG L-GRU", width='stretch', type="primary"):
            with st.spinner("Bộ AI đang xử lý mô phỏng..."):
                data = df[features].values.astype(np.float32)
                data_scaled = scaler.transform(data)
                
                # Dự báo
                preds_scaled = predict_autoregressive(model, data_scaled, steps=forecast_hours, device=device)
                preds = scaler.inverse_transform(preds_scaled)
                
                # Chuẩn bị thời gian cho biểu đồ
                try:
                    last_time = pd.to_datetime(df['time'].iloc[-1])
                except Exception:
                    # Rơi vào trường hợp không có cột time hoặc format sai
                    last_time = pd.Timestamp.now()
                    
                future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
                past_times = pd.to_datetime(df['time'].iloc[-48:]).tolist() if 'time' in df.columns else [last_time - timedelta(hours=48-i) for i in range(min(48, len(df)))]
                past_temp = df['temperature'].iloc[-48:].values
                
                st.divider()
                st.markdown("### 🌟 1. Tổng quan Dự báo (Dành cho mọi người)")
                
                current_temp = past_temp[-1]
                max_pred_temp = np.max(preds[:, 0])
                min_pred_temp = np.min(preds[:, 0])
                avg_pred_temp = np.mean(preds[:, 0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🌡️ Nhiệt độ Hiện tại", f"{current_temp:.1f} °C")
                with col2:
                    st.metric("🔥 Cao nhất sắp tới", f"{max_pred_temp:.1f} °C", f"{max_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
                with col3:
                    st.metric("❄️ Thấp nhất sắp tới", f"{min_pred_temp:.1f} °C", f"{min_pred_temp - current_temp:+.1f} °C", delta_color="inverse")
                with col4:
                    st.metric("💧 Độ ẩm Trung bình", f"{np.mean(preds[:, 1]):.0f} %")
                
                st.info("💡 **Mẹo xem nhanh:** Biểu đồ đồng hồ bên dưới cho biết mức nhiệt độ trung bình trong những giờ tới. Màu xanh là mát, màu vàng là ấm, màu đỏ là nóng.")
                
                # Biểu đồ Gauge (Đồng hồ đo)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_pred_temp,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Nhiệt độ Trung bình Dự báo", 'font': {'size': 20}},
                    number = {'suffix': " °C", 'font': {'size': 40}},
                    gauge = {
                        'axis': {'range': [10, 45], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "rgba(0,0,0,0.3)"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [10, 20], 'color': '#60A5FA'}, # Lạnh/Mát
                            {'range': [20, 28], 'color': '#34D399'}, # Ấm / Dễ chịu
                            {'range': [28, 35], 'color': '#fbbf24'}, # Hơi Nóng
                            {'range': [35, 45], 'color': '#F87171'}  # Rất Nóng
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_pred_temp
                        }
                    }
                ))
                fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, width='stretch')
                
                st.divider()
                st.markdown("### 📈 2. Phân tích Xu hướng Chi tiết (Biểu đồ Đường)")
                
                # Biểu đồ Plotly
                fig = go.Figure()
                
                # Đường dữ liệu thực tế tải lên
                fig.add_trace(go.Scatter(
                    x=past_times, 
                    y=past_temp,
                    mode='lines+markers',
                    name='Thực tế hiện tại',
                    line=dict(color='#2563EB', width=3),
                    marker=dict(size=6)
                ))
                
                # Đường dự báo AI
                fig.add_trace(go.Scatter(
                    x=[past_times[-1]] + list(future_times),
                    y=[past_temp[-1]] + list(preds[:, 0]), # Nối điểm cuối thực tế lên dự báo
                    mode='lines+markers',
                    name='Dự báo (L-GRU AI)',
                    line=dict(color='#EF4444', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                fig.update_layout(
                    title="Biểu đồ Dự báo Nhiệt độ Hà Tĩnh",
                    xaxis_title="Thời gian (Giờ)",
                    yaxis_title="Nhiệt độ (°C)",
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Bảng chi tiết
                with st.expander("📊 Bảng dữ liệu chi tiết từng giờ"):
                    res_df = pd.DataFrame({
                        "Thời gian": future_times,
                        "Nhiệt độ (°C)": preds[:, 0].round(2),
                        "Độ ẩm (%)": preds[:, 1].round(2),
                        "Áp suất (hPa)": preds[:, 2].round(2)
                    })
                    st.dataframe(res_df, width='stretch')

if __name__ == '__main__':
    main()
