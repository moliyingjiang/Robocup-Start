## __START__

##### ''''''

##### '''''''

##### v5.py # 运行yolov5所需要的初始化方法和预测方法
##### main.py # 包含Tkinter包装和yolov5识别结果的可视化
##### config.yaml # 包含一些基本的配置，包括模型路径、置信度、类别等
##### nohup.out # 终端输出重定向到此
##### nohup-just-now.txt # 上次运行main.py之后终端输出重定向的文件nohup.out的备份
##### output.txt # last_pre.sh第一次处理生成
##### last-output.txt # last_pre.sh调用pre.py进行第二次也就是最终处理后生成的我们所需要的文件

##### ''''''

### 1、自己操作--仅检测/无重定向

#### '''
##### conda activate "YourEnv" # 激活自己所配置好的yolov5及Tkinter的环境
##### python main.py # 运行main.py即可调用摄像头进行识别

##### ''''''

### 2、自己操作--检测+重定向输出

#### '''
##### conda activate "YourEnv" # 激活自己所配置好的yolov5及Tkinter的环境
##### nohup python main.py # 运行main.py即可调用摄像头进行识别，使用nohup进行终端输出重定向，定向位置为nohup.out
##### mv nohup.out output.txt
##### python pre.py

##### ''''''

### 3、便捷操作(仅三步)

#### '''
##### conda activate "YourEnv" # 激活自己所配置好的yolov5及Tkinter的环境
##### bash run.sh # 运行run.sh进行yolov5检测输出重定向和上次重定向文件nohup.out的备份
##### bash last_pre.sh # 运行last_pre.sh进行output.txt的生成 并自动运行pre.py对output.txt进行终端输出的提取重构为last-output.txt
##### mkdir ~/last-output # 在主目录创建一个存储我们所要最终结果的文件
##### cp last-output.txt ~/last-output # 复制最终结果的文件进去

##### ''''''
##### '''''''

## __END__

