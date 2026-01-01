# --- CODE TEST THỬ MỘT FILE BẤT KỲ ---
def test_new_voice(file_path):
    # 1. Load lại model đã lưu
    loaded_model = joblib.load('model_giong_noi.pkl')

    # 2. Xử lý file âm thanh mới
    feature = extract_features(file_path)
    
    if feature is not None:
        # Reshape để đúng định dạng đầu vào
        feature = feature.reshape(1, -1)
        
        # 3. AI Phán đoán
        prediction = loaded_model.predict(feature)
        probability = loaded_model.predict_proba(feature) # Xem độ tự tin
        
        confidence = np.max(probability) * 100
        
        if prediction[0] == 1:
            print(f"KẾT QUẢ: ✅ ĐÚNG LÀ CHỦ NHÀ! (Độ tin cậy: {confidence:.2f}%)")
        else:
            print(f"KẾT QUẢ: ❌ CẢNH BÁO - NGƯỜI LẠ! (Độ tin cậy: {confidence:.2f}%)")

# Thay tên file của bạn vào đây để test
test_new_voice("/content/Recording (22).m4a")