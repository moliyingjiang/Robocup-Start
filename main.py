import tkinter as tk
from tkinter import Canvas
import cv2
from PIL import Image, ImageTk
import threading
from v5 import init,predict_img
from utils.general import check_img_size, non_max_suppression, scale_coords
import time
import sys
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

class CameraApp:

    '''

    定义一些基础的变量以及TK图标库以及标签

    '''
    def __init__(self, root):
        self.root = root
        self.root.title("图像识别程序")
        self.root.geometry("800x600")
        self.image_frame = tk.Canvas(root, bg="gray", width=550, height=400)
        self.image_frame.place(x=20, y=20)
        self.model_message = "图像显示"
        self.image_frame.create_text(275, 200, text="图像显示", font=("Helvetica", 12),tags="pre_display_none")
        custom_label = tk.Label(root, text="识别结果输出区", font=("Helvetica", 14))
        custom_label.place(x=580, y=5)
        
        self.frame = tk.Frame(root, bg="gray", width=200, height=200)
        self.frame.place(x=590, y=200)  # Adjust the position
        self.bordered_label = tk.Label(self.frame, text="识别状态程序输出", font=("Helvetica", 16), bg="gray", fg="black", relief="solid", bd=2)
        self.bordered_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_frame = Canvas(root, bg="gray", width=200, height=300)
        self.result_frame.place(x=580, y=50)
        self.bordered_label = tk.Label(self.result_frame, text="识别状态程序输出", font=("Helvetica", 16), bg="gray",fg="black", relief="solid", bd=2)
        # self.result_frame.create_text(100, 20, text="识别程序状态输出", font=("Helvetica", 16))

        self.start_button = tk.Button(root, text="开始", command=self.start_camera, bg="orange", width=5, height=2, font=("Helvetica", 16))
        self.start_button.place(x=350, y=450)
        self.red_button = tk.Button(root, text="➔", command=self.kill_predict, bg="red", fg="black", width=5, height=2, font=("Helvetica", 24, "bold"))
        self.red_button.place(x=700, y=500)

        self.tip_text = tk.Label(root, text="灰色矩形框为可显示输出数据框，黄色矩形按钮表示开始按钮", font=("Helvetica", 10))
        self.tip_text.place(x=20, y=570)
        self.done_lock = True
        self.cap = None
        self.camera_thread = None
        self.is_running = False

        self.tags = []
    
    def predict_text_display(self,predict_text):
        self.result_frame.delete("predict_text_display")
        self.result_frame.create_text(100, 14, text=predict_text, font=("Helvetica", 10),tags="predict_text_display")
        self.root.update_idletasks()  # Update display

    def start_camera(self):
        self.model_message = "模型加载中..."
        self.image_frame.create_text(275, 250, text=self.model_message, font=("Helvetica", 12),tags="pre_display_none") # 创建并显示：模型加载中...
        self.root.update_idletasks()  # Update display 更新画面
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            self.capture_loop()
            # 删除这里，线程运行的方式会导致画面闪动
            # self.camera_thread = threading.Thread(target=self.capture_loop) 
            # self.camera_thread.start()
    
    def capture_loop(self):
        predict_text_arr = []
        predict_text_display_count = 40
        device, half, model, names, colors = init()
        # 允许检测
        while self.is_running:
            ret, frame = self.cap.read() # 读取图像
            if not ret: # 无图像正常退出
                break
            if self.done_lock:
                self.image_frame.delete("pre_display_none") # 删除一开始DeBug用到的TK初始文字显示
                self.done_lock = False # 模型加载完成只显示一次
                 
                '''
                TK display
                '''
                 # TK显示模型加载完毕
                self.tip_text = tk.Label(root, text="模型加载完毕！", font=("Helvetica", 10))
                self.tip_text.place(x=350, y=515)
            
            '''
            
            动态显示"模型加载中......."
            
            '''
            predict_text_display_count -= 1 # 保证动态区间控制的值在发生变化
            if predict_text_display_count % 4 == 0: # 显示计数区间值隔帧动态--(根据自己的实际帧率调整，帧率高就调高点，帧率低就调低点)
                predict_text_arr.append(".")
                predict_text = f'模型推理中{"".join(predict_text_arr)}' # 将列表转化为字符串方便显示
                if len(predict_text_arr) <= 6: # 限制六个省略号
                    self.predict_text_display(predict_text)
            # 避免计数区间值的干涸与超值
            if predict_text_display_count == 0:
                predict_text_display_count = 40
                predict_text_arr.clear()

            ''''''
             # 调用V5进行检测
            img, pred = predict_img([frame], device, half, model)
            
            yolo_results = [] # 定义一个存储yolo输出标签的列表
            yolo_last_results = [] # 定义一个用于tkinter读取显示的最后的列表
            self.tags.clear() # 用于存储已读标签避免TK重复读取重复显示

            '''

            定义一个要显示并计算其计算数量的标签(如果不想限制，则设置Read_count_args_Lock为False)

            '''
            count_args = config['count_args']
            Read_count_args_Lock = config['Read_count_args_Lock']

            # 获取V5输出结果
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # use frame.shape instead of im0.shape
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if (conf >= config['conf'] and names[int(cls)] in count_args and Read_count_args_Lock) or (conf >= config['conf'] and not Read_count_args_Lock):
                            # print(label)
                            yolo_results.append(label)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            '''
                            
                            画框和标签
                            
                            '''
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 0, 255), 3)
                            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            '''

            计算"your_args"出现的次数并转移合适信息到TK列表中

            '''
            for temp in yolo_results:
                cont_temp = self.tags
                count_result = yolo_results.count(temp) # 计算物体出现的次数
                if temp not in cont_temp: # 如果出现的这个标签没有被读取和添加过
                    if count_result > 1: # 如果数量超过一个，则设置标签数量为复数
                        yolo_last_results.append(f'{temp}s，数量：{count_result}个.') # 添加信息到TK读取的列表
                    else:
                        yolo_last_results.append(f'{temp}, 数量：{count_result}个.') # 同上
                self.tags.append(temp) # 添加已读标签到self.tags中

            '''

            显示图像以及检测结果

            '''
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 避免图像蓝色调过重的问题/修复1
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_frame.delete("image")  # Clear previous image
            self.image_frame.create_image(275, 200, image=img_tk, tags="image")  # Add new image
            # threading.Thread(target = TEXT,args=(yolo_results,)).start()
            self.result_frame.delete("done") # Clear previous tag
            self.result_frame.delete("yolo_result_font") # Clear previous tag
            
            '''

            将检测结果显示在输出框中

            '''
            if len(yolo_last_results) != 0:
                print(yolo_last_results)
            pre_out_size = 43 # 设置初始的标签显示位置
            for temp in yolo_last_results:
                # print(temp)
                self.result_frame.create_text(100, pre_out_size, text=temp, font=("Helvetica", 9),tags = "yolo_result_font") # Add new tag to display
                pre_out_size += 33 # 自动换行间隔设定
            
            '''

            非常重要的地方，一定要更新画面

            '''
            self.root.update_idletasks()  # Update display

        self.cap.release()

    def kill_predict(self):
        sys.exit()

if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
