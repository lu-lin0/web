import os
import json
from collections import deque
import numpy as np
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
    print("WebSocket 客戶端已連線")

    # 為連線建立滑動序列 Buffer (長度由 config.TFC 決定，預設為 30)
    sequence = deque(maxlen=TFC)

    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            landmarks = data.get("landmarks", [])

            # 驗證關鍵點維度是否為 config 中設定的 FEATURE_F (198維)
            if len(landmarks) != FEATURE_F:
                await websocket.send_json({
                    "prediction": f"特徵維度不符 (期待 {FEATURE_F}, 收到 {len(landmarks)})",
                    "confidence": 0.0
                })
                continue

            keypoints = np.array(landmarks, dtype=np.float32)
            sequence.append(keypoints)

            # 當收集滿 30 幀時開始進行模型推論
            if len(sequence) == TFC:
                input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

                # 模型預測
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_index = int(np.argmax(prediction))
                probability = float(prediction[predicted_index])

                # 取得英文類別標籤與對應中文
                action_en = gestures[predicted_index]
                
                # 若為 TSL 則優先查詢 TSL_GESTURES_TW 字典
                action_zh = TSL_GESTURES_TW.get(action_en, action_en)

                await websocket.send_json({
                    "prediction": action_zh,
                    "confidence": round(probability * 100, 2)
                })
            else:
                await websocket.send_json({
                    "prediction": f"累積畫面中 ({len(sequence)}/{TFC})...",
                    "confidence": 0.0
                })

    except WebSocketDisconnect:
        print("WebSocket 客戶端已斷開連線")
    except Exception as e:
        print(f"WebSocket 發生錯誤: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)