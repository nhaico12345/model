import urllib.request
import json
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("⏳ Đang tải dữ liệu thời tiết thực tế cho Hà Tĩnh (48 giờ qua)...")
    # Tọa độ Hà Tĩnh: 18.3333 N, 105.9000 E
    url = "https://api.open-meteo.com/v1/forecast?latitude=18.3333&longitude=105.9000&past_days=2&hourly=temperature_2m,relative_humidity_2m,surface_pressure&timezone=Asia%2FBangkok"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        hourly = data['hourly']
        
        df = pd.DataFrame({
            'time': hourly['time'],
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'pressure': hourly['surface_pressure']
        })
        
        # Lấy tối đa 48 giờ gần nhất có dữ liệu thực
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%dT%H:00')
        df_past = df[df['time'] <= now_str]
        
        if len(df_past) >= 48:
            df_final = df_past.tail(48)
        else:
            df_final = df.head(48) # fallback
            
        # Xử lý missing data nếu có
        df_final = df_final.interpolate(method='linear').ffill().bfill()
        
        outfile = "hatinh_weather_test.csv"
        df_final.to_csv(outfile, index=False)
        print(f"✅ Đã lưu {len(df_final)} giờ dữ liệu thời tiết Hà Tĩnh vào file: '{outfile}'")
        print("Mẫu dữ liệu:")
        print(df_final.head())
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra khi tải dữ liệu: {e}")

if __name__ == '__main__':
    main()
