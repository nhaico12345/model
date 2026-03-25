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

st.set_page_config(page_title="L-GRU Finance Forecast", page_icon="📈", layout="wide")

# -------------------------------------------------------------
# 1. KIẾN TRÚC MẠNG L-GRU 
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
# 2. XỬ LÝ MODEL VÀ SCALER
# -------------------------------------------------------------
@st.cache_resource
def load_scaler(df):
    scaler = MinMaxScaler()
    # Fit scaler trên cột Close thực tế để inverse transform sau dự báo
    scaler.fit(df[['Close']].values)
    return scaler

@st.cache_resource
def load_ai_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy trọng số tại {model_path}")
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Chuyển đổi input_size=1 và output_size=1 thay vì 3 như thời tiết
    model = CustomAIModel(input_size=1, hidden_size=256, output_size=1, num_layers=3).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Tương thích với Pytorch compile
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, device

# -------------------------------------------------------------
# 3. HÀM DỰ BÁO TỰ QUAY LUI (AUTOREGRESSIVE)
# -------------------------------------------------------------
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
# 4. GIAO DIỆN CHÍNH
# -------------------------------------------------------------
def main():
    st.markdown("<h1 style='color: #1E3A8A;'>📈 AI L-GRU: Dự báo Tài chính</h1>", unsafe_allow_html=True)
    st.write("**Áp dụng thuật toán L-GRU cho chuỗi thời gian giá cổ phiếu & tiền điện tử**")
    st.divider()

    # Sidebar điều khiển
    st.sidebar.header("🔧 Cài đặt Model")
    model_choice = st.sidebar.selectbox("Chọn mô hình dự báo:", ["Bitcoin", "SP500", "Tesla"])
    
    model_files = {
        "Bitcoin": "modeltaichinh/best_bitcoin_model.pth",
        "SP500": "modeltaichinh/best_sp500_model.pth",
        "Tesla": "modeltaichinh/best_tesla_model.pth"
    }
    
    forecast_days = st.sidebar.slider("Dự báo bao nhiêu ngày tới?", min_value=1, max_value=30, value=7)
    
    # Upload dữ liệu test
    st.markdown(f"### 📥 Tải lên dữ liệu test của {model_choice} (Định dạng CSV)")
    st.info(f"Bạn có thể dùng file `{model_choice}_Test_Data.csv` sinh ra từ file tải dữ liệu.")
    
    uploaded_file = st.file_uploader("Chọn file CSV dữ liệu", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("Dữ liệu cần phải chứa cột 'Date' và 'Close'")
            st.stop()
            
        with st.expander("👁️ Xem trước dữ liệu", expanded=False):
            st.dataframe(df.tail(10), width='stretch')
            
        # Nạp model & scaler
        model_path = model_files[model_choice]
        model_data = load_ai_model(model_path)
        if model_data is None: st.stop()
        model, device = model_data
        
        scaler = load_scaler(df)

        if st.button("🚀 BẮT ĐẦU DỰ BÁO", width='stretch', type="primary"):
            with st.spinner("AI đang tính toán chuỗi dữ liệu..."):
                # Chuẩn bị dữ liệu (Lấy 60 ngày gần nhất để làm input sequence)
                seq_len = min(60, len(df))
                recent_data = df['Close'].values[-seq_len:].reshape(-1, 1)
                
                # Scaler data -> Dự báo -> Inverse transform về giá trị thực
                data_scaled = scaler.transform(recent_data)
                preds_scaled = predict_autoregressive(model, data_scaled, steps=forecast_days, device=device)
                preds = scaler.inverse_transform(preds_scaled)
                
                # Hiển thị Biểu đồ
                st.markdown("### 📊 Kết quả Dự báo L-GRU")
                
                last_date = pd.to_datetime(df['Date'].iloc[-1])
                future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                # Vẽ 30 ngày quá khứ + tương lai
                past_dates = pd.to_datetime(df['Date'].iloc[-30:])
                past_prices = df['Close'].iloc[-30:].values
                
                fig = go.Figure()
                # Line quá khứ
                fig.add_trace(go.Scatter(x=past_dates, y=past_prices, mode='lines+markers', 
                                         name='Thực tế', line=dict(color='#2563EB', width=2)))
                # Line dự báo
                fig.add_trace(go.Scatter(x=[past_dates.iloc[-1]] + future_dates, 
                                         y=[past_prices[-1]] + list(preds[:, 0]),
                                         mode='lines+markers', name='Dự báo AI', 
                                         line=dict(color='#EF4444', width=2, dash='dash')))
                
                fig.update_layout(title=f"Dự báo giá {model_choice} ({forecast_days} ngày tới)",
                                  xaxis_title="Thời gian (Ngày)",
                                  yaxis_title="Giá ($)", hovermode="x unified", template="plotly_white")
                
                st.plotly_chart(fig, width='stretch')

if __name__ == '__main__':
    main()