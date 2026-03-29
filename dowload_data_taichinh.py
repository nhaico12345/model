import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def download_clean_and_save(ticker, file_name, start_date, end_date):
    print(f"🔄 Đang xử lý dữ liệu cho {ticker}...")
    
    # 1. Tải dữ liệu từ Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Rút trích cột Giá đóng cửa (Close)
    # Tinh chỉnh để tương thích với cấu trúc mới của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Lúc này df['Close'] đã là DataFrame, chứa cột tên là mã Ticker (vd: BTC-USD)
        df = df['Close'].copy()
        df.columns = ['Close'] # Đổi tên cột thành 'Close' cho chuẩn
    else:
        df = df[['Close']].copy()
        
    # 2. Làm sạch dữ liệu (Data Cleaning): 
    # Loại bỏ các hàng bị thiếu dữ liệu (NaN) do ngày lễ hoặc lỗi hệ thống
    df = df.dropna()
    
    # 3. Chuẩn hóa dữ liệu (Data Normalization):
    # Đưa dải giá trị về khoảng [0, 1] cho mô hình L-GRU học hiệu quả nhất
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Scaled_Close'] = scaler.fit_transform(df[['Close']])
    
    # 4. Xuất ra file CSV
    # Lấy đường dẫn thư mục hiện tại đang chạy code
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    
    # Lưu file
    df.to_csv(file_path)
    print(f"✅ Đã xuất file thành công tại: {file_path}")
    print(f"   Tổng số dòng dữ liệu đã làm sạch: {len(df)} dòng.\n")
    
    return df

# Thiết lập thời gian (2018 - 2024 để lấy trọn 3 chu kỳ thị trường)
START = '2010-01-01'
END = '2026-03-23'

print("BẮT ĐẦU TẢI VÀ CHUẨN BỊ DỮ LIỆU...\n" + "-"*40)

# Chạy tải và lưu cho 3 loại tài sản
download_clean_and_save('BTC-USD', 'Bitcoin_Ready_Data.csv', START, END)
download_clean_and_save('TSLA', 'Tesla_Ready_Data.csv', START, END)
download_clean_and_save('^GSPC', 'SP500_Ready_Data.csv', START, END)

print("-" * 40)
print("🎉 Quá trình hoàn tất! Hãy kiểm tra thư mục chứa file code này để xem 3 file CSV.")