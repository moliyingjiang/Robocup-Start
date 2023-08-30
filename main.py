import tkinter as tk
from tkinter import Canvas
import cv2
from PIL import Image, ImageTk
import threading
from v5 import init,predict_img
from utils.general import check_img_size, non_max_suppression, scale_coords

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像识别程序")
        self.root.geometry("800x600")

        self.image_frame = tk.Canvas(root, bg="gray", width=550, height=400)
        self.image_frame.place(x=20, y=20)
        self.image_frame.create_text(275, 200, text="图像显示", font=("Helvetica", 16))

        custom_label = tk.Label(root, text="识别结果输出区", font=("Helvetica", 18))
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


        self.red_button = tk.Button(root, text="➔", bg="red", fg="black", width=5, height=2, font=("Helvetica", 24, "bold"))
        self.red_button.place(x=700, y=500)

        self.tip_text = tk.Label(root, text="灰色矩形框为可显示输出数据框，黄色矩形按钮表示开始按钮", font=("Helvetica", 10))
        self.tip_text.place(x=20, y=570)

        self.cap = None
        self.camera_thread = None
        self.is_running = False

    def start_camera(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            self.camera_thread = threading.Thread(target=self.capture_loop)
            self.camera_thread.start()

    def capture_loop(self):
        device, half, model, names, colors = init()
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            img, pred = predict_img([frame], device, half, model)
            ocr_results = []
            yolo_results = []
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # use frame.shape instead of im0.shape
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        if conf >= 0.6:
                            print(label)
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 0, 255), 3)

                            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (0, 0, 255), 3)
                            
            # Display the frame with bounding boxes and labels
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_frame.delete("image")  # Clear previous image
            self.image_frame.create_image(275, 200, image=img_tk, tags="image")  # Add new image
            self.root.update_idletasks()  # Update display

        self.cap.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
