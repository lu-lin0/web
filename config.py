import numpy as np


# ==============================
# 檔案位置 (請記得要改!!!!!)
# ==============================

# 原始影片資料夾
VIDEO_PATH = "video"

# MediaPipe特徵輸出資料夾
PREPROCESS_PATH = "MP_Data"

# 美式(ASL)或台式手語(TSL)
LANGUAGE = "TSL"


# ==============================
# 編輯者ID (請記得要改!!!!!)
# ==============================
# 趙 → "A"
# 邱 → "B"
# 呂 → "C"

editor_ID = "A"



# ==============================
# 手語名稱 (請記得要改!!!!!)
# ==============================

# 台式手語種類
TSL_GESTURES = np.array(["you", "I", "know", "happy", "sorry", "thank", "wait", "Miss", "You're welcome", "he;she;it", "can", "Mr", "goodbye", "Hello", "It doesn't matter"])

# 顯示在螢幕上的中文名稱
TSL_GESTURES_TW = {"you": "你", "I": "我", "know": "知道", "happy": "高興", "sorry": "對不起", "thank": "謝謝", "wait": "等一下", "Miss": "小姐", "You're welcome": "不客氣", "he;she;it": "他", "can": "可以", "Mr": "先生", "goodbye": "再見", "Hello": "你好", "It doesn't matter": "沒關係"}

# 美式手語種類
ASL_GESTURES = np.array(["book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black"])

if LANGUAGE == "ASL":
    gestures = ASL_GESTURES
    MODEL_PATH = "model_ASL.h5"

elif LANGUAGE == "TSL":
    gestures = TSL_GESTURES
    MODEL_PATH = "model_TSL_baseline.h5"



# 每支影片統一取出的frame數量
TFC = 30



# ==============================
# MediaPipe Holistic設定
# ==============================

# 模型複雜度
MODEL_COMPLEXITY = 1

# 最低偵測信心
MIN_DC = 0.5

# 最低追蹤信心
MIN_TC = 0.5

# 視窗大小
DISPLAY_WIDTH = 900


# (部位的左右之分皆以本人視角為主)
# 只使用上半身(骨架)關鍵點
POSE_INDEX = [
    11,     #左肩
    12,     #右肩
    13,     #左手肘
    14,     #右手肘
    15,     #左手腕
    16      #右手腕
]



# 臉部關鍵點 (提取大概的五官、臉部輪廓)
FACE_INDEX = {

    # 【右眉】(從左下角開始逆時針)
    "right_eyebrow": [
        46, 52, 55, 107, 105, 70
    ],
    # 【左眉】(從右下角開始順時針)
    "left_eyebrow": [
        276, 282, 285, 336, 334, 300
    ],

    # 【右眼】(從最左邊開始逆時針)
    "right_eye": [
        #眼睛輪廓
        33, 145, 133, 158, 159, 
        #中心點
        # 468
    ],

    # 【左眼】(從最左邊開始逆時針)
    "left_eye": [
        #眼睛輪廓
        362, 374, 263, 386, 385, 
        #中心點
        # 473
    ],

    # 【鼻子】(從最上面開始逆時針)--最高處為6
    "nose": [
        # 整體輪廓
        6, 129, 98, 2, 327, 358, 
    ],

    # 【嘴巴】
    "mouth": [
        # 整體輪廓--最高處為0, 最低處為17
        61, 91, 17, 321, 291, 267, 0, 37,
        # 13, 14 為內唇的最高、低處
    ],

    # 【額頭】(從最低處開始順時針)--最高處為10,　最低處為9
    "forehead": [
        9, 284, 297, 10, 67, 54, 
    ],

    # 【下巴】(從最低處開始逆時針)--最高處為18, 最低處為152
    "chin": [
        152, 400, 431, 406,
        18, 182, 211, 176
    ],

    # 【臉部輪廓】(從最低處開始逆時針)--最高處為10, 最低處為152
    # "face_contour": [
    #     10, 103, 54, 21, 93, 58, 150, 152, 
    #     379, 288, 401, 323, 251, 284, 332
    # ],
    
}



# ==============================
# 特徵值
# ==============================

# 肩膀到手臂: 6個關鍵點，使用(x,y,z,visibility)數值
POSE_F = len(POSE_INDEX) * 4


# 手掌: 21個關鍵點，使用(x,y,z)數值
HAND_F = 21 * 3

# 手到臉部: 8個部位，使用(x,y,z)數值
FACE_F = 8 * 3


# 身體 + 左手 + 右手
FEATURE_F = POSE_F + HAND_F + HAND_F + FACE_F + FACE_F



#測試用
print("==============================")
print("手勢:", gestures)
print("每段動作包含的Frame數量:", TFC)
print("一個Frame總共多少特徵值:", FEATURE_F)
print("==============================")