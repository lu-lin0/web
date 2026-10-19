import os
import json
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 1. 載入完整的 config 設定
from config import (
    gestures,  
    TSL_GESTURES_TW,
    TFC,
    FEATURE_F,
    MODEL_PATH,
    LANGUAGE
)

app = FastAPI(title=f"即時手語辨識 API ({LANGUAGE})")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 載入模型 (根據 config 中的 MODEL_PATH)
# ==========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}，請確認檔案放置於專案根目錄。")

print(f"正在載入 {LANGUAGE} 模型 [{MODEL_PATH}]...")
model = load_model(MODEL_PATH)
print("模型載入成功！")
print("模型輸入 shape:", model.input_shape)
print("模型輸出 shape:", model.output_shape)


# ==========================================
# WebSocket 即時辨識 Endpoint
# ==========================================
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ 新的 WebSocket 連線已建立")

    # 為每個獨立連線建立滑動視窗緩衝區 (長度為 30 幀)
    sequence_buffer = deque(maxlen=30)

    try:
        while True:
            # 接收前端發送的 JSON 數據
            data_text = await websocket.receive_text()
            data = json.loads(data_text)

            # ------------------------------------------------------
            # 1. 處理短影片播放結束指令 (Flush)
            # ------------------------------------------------------
            if data.get("action") == "flush":
                if len(sequence_buffer) >= 5:  # 只要累積 5 幀以上就強制輸出
                    padded_seq = list(sequence_buffer)
                    while len(padded_seq) < 30:
                        padded_seq.append([0.0] * 198)  # 用 0 補齊至 30 幀

                    input_data = np.expand_dims(np.array(padded_seq, dtype=np.float32), axis=0)
                    prediction = model(input_data, training=False).numpy()[0]
                    class_idx = int(np.argmax(prediction))
                    confidence = float(prediction[class_idx])

                    raw_label = gestures[class_idx] if class_idx < len(gestures) else "Unknown"
                    zh_label = TSL_GESTURES_TW.get(raw_label, raw_label)

                    await websocket.send_json({
                        "prediction": zh_label,
                        "confidence": round(confidence * 100, 1)
                    })
                continue

            landmarks = data.get("landmarks", [])

            # 驗證維度是否為 198
            if len(landmarks) != 198:
                await websocket.send_json({
                    "prediction": "數據格式錯誤",
                    "confidence": 0
                })
                continue

            # 全 0 檢查 (代表未偵測到手部)
            if np.all(np.array(landmarks) == 0):
                sequence_buffer.clear()  # 手離開畫面時清空佇列
                await websocket.send_json({
                    "prediction": "等待手勢中...",
                    "confidence": 0
                })
                continue

            # 存入滑動視窗
            sequence_buffer.append(landmarks)

            # ------------------------------------------------------
            # 2. 短影片 & 即時推論邏輯 (零填充 Zero-Padding)
            # ------------------------------------------------------
            # 當影格達 10 幀以上（約 0.5 秒）即開始預測，未滿 30 幀自動補 0
            if len(sequence_buffer) >= 10:
                padded_seq = list(sequence_buffer)
                while len(padded_seq) < 30:
                    padded_seq.append([0.0] * 198)

                input_data = np.expand_dims(np.array(padded_seq, dtype=np.float32), axis=0)
                
                # 💡 改用 direct call 替代 model.predict() 以提高執行效率
                prediction = model(input_data, training=False).numpy()[0]
                class_idx = int(np.argmax(prediction))
                confidence = float(prediction[class_idx])

                # 💡 修正變數名稱：使用 config 匯入的 gestures 變數
                raw_label = gestures[class_idx] if class_idx < len(gestures) else "Unknown"
                zh_label = TSL_GESTURES_TW.get(raw_label, raw_label)

                # 回傳結果給前端
                await websocket.send_json({
                    "prediction": zh_label,
                    "confidence": round(confidence * 100, 1)
                })
            else:
                progress = int((len(sequence_buffer) / 30) * 100)
                await websocket.send_json({
                    "prediction": f"分析中 ({progress}%)",
                    "confidence": 0
                })

    except WebSocketDisconnect:
        print("🔌 WebSocket 連線已中斷")
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)