# =============================================================================
# CHƯƠNG TRÌNH TRAIN AI NHẬN DIỆN GIỌNG NÓI (PHIÊN BẢN ONE-CLICK)
# Hỗ trợ: .wav, .mp3, .m4a
# =============================================================================

# 1. Cài đặt các thư viện cần thiết (Chạy siêu tốc)
!pip install -q librosa numpy scikit-learn joblib resampy

import librosa
import numpy as np
import os
import glob
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CẤU HÌNH ---
DATASET_ZIP = "dataset_giong_noi.zip" # Tên file zip bạn upload lên
DATA_DIR = "dataset_giong_noi"      # Tên thư mục sau khi giải nén

# --- HÀM TRÍCH XUẤT ĐẶC TRƯNG MFCC ---
def extract_features(file_path):
    try:
        # Load file âm thanh (tự động nhận diện mp3, m4a, wav)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Trích xuất MFCC (Lấy 40 đặc trưng quan trọng nhất)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Lấy trung bình cộng theo thời gian
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc file {file_path}: {e}")
        return None

# --- BƯỚC 1: XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
print("▶️ BẮT ĐẦU XỬ LÝ DỮ LIỆU...")

# Tự động giải nén nếu chưa giải nén
if os.path.exists(DATASET_ZIP):
    print(f"--> Đang giải nén {DATASET_ZIP}...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("--> Giải nén xong!")
elif not os.path.exists(DATA_DIR):
    print(f"❌ LỖI: Không tìm thấy file '{DATASET_ZIP}' cũng không thấy thư mục '{DATA_DIR}'!")
    print("👉 Hãy upload file zip lên Colab trước khi chạy!")
    # Dừng chương trình tại đây nếu không có dữ liệu
    raise SystemExit

features = []
labels = []

# Định nghĩa cấu trúc dữ liệu cần tìm
# Cấu trúc: (Tên thư mục con, Nhãn gán cho AI)
# Nhãn 1 = Chủ nhà, Nhãn 0 = Người lạ
structure = [
    (os.path.join(DATA_DIR, "chu_nha"), 1),
    (os.path.join(DATA_DIR, "nguoi_la"), 0)
]

# Các đuôi file chấp nhận
extensions = ["*.wav", "*.mp3", "*.m4a"]

total_files = 0

for folder_path, label in structure:
    print(f"--> Đang quét thư mục: {folder_path}...")

    if not os.path.exists(folder_path):
        print(f"⚠️ Cảnh báo: Không thấy thư mục {folder_path}. Bỏ qua.")
        continue

    # Quét tất cả các đuôi file
    files_found = []
    for ext in extensions:
        files_found.extend(glob.glob(os.path.join(folder_path, ext)))

    # Xử lý từng file
    for file in files_found:
        data = extract_features(file)
        if data is not None:
            features.append(data)
            labels.append(label)
            total_files += 1

# Kiểm tra nếu không có dữ liệu thì báo lỗi ngay
if len(features) == 0:
    print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy bất kỳ file âm thanh nào!")
    print("👉 Kiểm tra lại xem trong thư mục 'chu_nha' và 'nguoi_la' có file chưa?")
    raise SystemExit

print(f"✅ ĐÃ XỬ LÝ XONG: Tổng cộng {len(features)} mẫu dữ liệu hợp lệ.")

# --- BƯỚC 2: CHUẨN BỊ TRAIN ---
X = np.array(features)
y = np.array(labels)

# Chia dữ liệu: 80% học - 20% thi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BƯỚC 3: HUẤN LUYỆN MODEL (MLP) ---
print("\n▶️ ĐANG HUẤN LUYỆN AI (Training)...")

# Cấu hình mạng Nơ-ron:
# - hidden_layer_sizes=(128, 64): 2 lớp ẩn giúp AI thông minh hơn
# - max_iter=500: Học tối đa 500 lần
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu', solver='adam', random_state=1)

model.fit(X_train, y_train)

# --- BƯỚC 4: ĐÁNH GIÁ KẾT QUẢ ---
print("\n▶️ KẾT QUẢ ĐÁNH GIÁ:")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🏆 ĐỘ CHÍNH XÁC: {accuracy * 100:.2f}%")
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Người lạ', 'Chủ nhà']))

# --- BƯỚC 5: LƯU MODEL ---
model_filename = 'model_giong_noi.pkl'
joblib.dump(model, model_filename)
print(f"✅ Đã lưu model thành công vào file: {model_filename}")
print("👉 Bạn hãy tải file này về máy tính để dùng cho Project!")