from flask import Flask, render_template, request
import subprocess
import threading
import time
import os
import cv2
import face_recognition_english

app = Flask(__name__)

# 用于存储未知人脸检测的开始时间
unknown_start_time = None

@app.route('/')
def index():
    global unknown_start_time
    unknown_start_time = None
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    mode = request.form.get('mode')
    if mode not in ['1', '2']:
        return "Invalid mode selection."
    def run_face_recognition():
        global unknown_start_time
        try:
            # 调用原 Python 脚本，并传递模式参数
            subprocess.run(['python', 'face_recognition_english.py', mode])
        except Exception as e:
            print(f"Error running face recognition: {e}")
    # 启动一个新线程来运行人脸识别程序
    threading.Thread(target=run_face_recognition).start()
    return "Face recognition started."

@app.route('/save_face', methods=['GET', 'POST'])
def save_face():
    if request.method == 'POST':
        save_choice = request.form.get('save_choice')
        if save_choice == 'y':
            new_name = request.form.get('new_name')
            # 读取临时保存的帧
            frame = cv2.imread('temp_frame.jpg')
            # 读取临时保存的人脸位置
            face_location = None
            with open('temp_face_location.txt', 'r') as f:
                line = f.readline()
                top, right, bottom, left = map(int, line.strip().split(','))
                face_location = (top, right, bottom, left)
            if frame is not None and face_location is not None:
                new_encoding = face_recognition_english.save_new_face(frame, face_location, new_name)
                # 更新已知人脸编码和名称列表
                known_encodings, known_names = face_recognition_english.load_all_known_faces()
                known_encodings.append(new_encoding)
                known_names.append(new_name)
                print(f"The face of {new_name} has been saved.")
                return "Face saved successfully."
            else:
                return "Failed to save face: frame or face location is None."
        else:
            return "Face not saved."
    return render_template('save_face.html')

if __name__ == '__main__':
    app.run(debug=True)
    