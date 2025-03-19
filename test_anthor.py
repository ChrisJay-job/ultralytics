from ultralytics import YOLO  # 导入 YOLO 模型库
import os  # 用于文件和目录操作
import cv2  # 用于图像处理和操作
import random  # 用于生成随机颜色
from datetime import datetime  # 用于获取当前时间

# 加载训练好的 YOLO 模型
model = YOLO('yolo11n.pt')  # 替换为你训练好的模型权重文件路径

# 输入图片目录
input_path = 'datasets-anchor/video/'  # 替换为存放图片的输入目录路径
output_path = 'runs_output_anchor_images_vedio'  # 替换为存放图片的输出目录路径
output_images_dir = '%s/output_images/' % output_path # 保存标注截图的输出目录
output_images_anchor_txt_dir = '%s/output_images/output_txt/' % output_path  # 保存图片预测结果（坐标）的 txt 文件目录
output_video_dir = '%s/output_videos/' % output_path  # 保存标注后视频的目录
output_video_image_dir = '%s/output_videos/output_video_anchor_image' % output_path  # 保存标注帧图片的目录
output_video_anchor_txt_dir = '%s/output_videos/output_video_anchor_txt' % output_path  # 保存视频（抽帧）预测结果（坐标）的 txt 文件目录

# 创建输出目录（如果不存在则自动创建）
os.makedirs(output_images_dir, exist_ok=True)  # 创建图片保存 txt 文件的目录
os.makedirs(output_images_anchor_txt_dir, exist_ok=True)  # 创建保存图片预测结果（坐标）的 txt 文件目录
os.makedirs(output_video_dir, exist_ok=True)   # 保存标注后视频的目录
os.makedirs(output_video_image_dir, exist_ok=True)   # 保存标注帧图片的目录
os.makedirs(output_video_anchor_txt_dir, exist_ok=True)  # 保存视频（抽帧）预测结果（坐标）的 txt 文件目录

# 定义类别名称列表（COCO 数据集类别）
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 定义类别与颜色的固定映射
fixed_colors = {
    "person": (0, 255, 0),    # 绿色
    "car": (255, 0, 0),       # 蓝色
    "apple": (128, 0, 128),   # 紫色
    "banana": (0, 255, 255)   # 黄色
}

# 为每个类别生成随机颜色，用于绘制边框
colors = [tuple(random.choices(range(256), k=3)) for _ in range(len(class_names))]


# 检查输入路径是否为文件夹
if os.path.isdir(input_path):  # 如果输入路径是文件夹
    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]  # 筛选支持的视频文件格式
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # 筛选支持的图片文件格式

    if not video_files:  # 如果没有找到任何视频文件
        print(f"No video files found in directory '{input_path}'.")  # 打印提示信息
    else:
        # 处理视频文件
        if video_files:
            print(f"Found {len(video_files)} video files.")  # 打印找到的视频文件数量

            for video_file in video_files:  # 遍历每个视频文件
                video_path = os.path.join(input_path, video_file)  # 构造完整视频路径
                print(f"Processing video: {video_path}")  # 打印当前处理的视频路径

                cap = cv2.VideoCapture(video_path)  # 打开视频文件
                if not cap.isOpened():  # 检查视频是否成功打开
                    print(f"Error: Unable to open video file '{video_file}'.")  # 打印错误信息
                    continue  # 跳过该视频文件

                # 动态获取视频参数
                fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧宽度
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧高度
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

                # 根据视频扩展名选择适当的编码器
                video_extension = os.path.splitext(video_file)[1].lower()  # 获取视频文件扩展名
                if video_extension in ['.mp4', '.mov']:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码器
                elif video_extension in ['.avi']:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI 编码器
                elif video_extension in ['.mkv', '.flv', '.wmv']:
                    fourcc = cv2.VideoWriter_fourcc(*'X264')  # MKV/FLV 编码器
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 默认使用 MP4 编码器

                # 设置输出视频路径和初始化视频写入对象
                output_video_path = os.path.join(output_video_dir, video_file)  # 构造输出视频路径
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # 初始化视频写入对象

                # 打开对应的 txt 文件，用于记录检测结果
                txt_path = os.path.join(output_video_anchor_txt_dir, f"{os.path.splitext(video_file)[0]}.txt")
                with open(txt_path, 'a') as f_txt:  # 以追加模式打开 txt 文件
                    frame_id = 0  # 初始化帧编号

                    while cap.isOpened():  # 循环读取视频帧
                        ret, frame = cap.read()  # 读取一帧
                        if not ret:  # 如果读取失败，退出循环
                            print(f"End of video: {video_file}")  # 打印提示视频结束
                            break

                        frame_id += 1  # 当前帧编号加 1
                        video_time = frame_id / fps  # 当前帧的时间戳（秒）
                        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")  # 获取当前时间，格式化为文件名友好的形式

                        print(f"Processing frame {frame_id}/{total_frames} at {video_time:.2f}s...")  # 打印帧处理进度

                        # 使用 YOLO 模型对当前帧进行预测
                        results = model.predict(source=frame, device='0', conf=0.25, save=False, verbose=False)

                        # 遍历检测结果
                        for result in results:
                            for box in result.boxes:  # 遍历每个检测框
                                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 提取边界框坐标（左上角和右下角）
                                confidence = box.conf.tolist()[0]  # 检测框的置信度
                                class_id = int(box.cls.tolist()[0])  # 检测到的类别 ID
                                class_name = class_names[class_id]  # 获取类别名称

                                # 确定颜色，优先使用固定颜色映射
                                color = fixed_colors.get(class_name, colors[class_id])

                                # 绘制检测框和标签在框内
                                label = f"{class_name} {confidence:.2f}"  # 设置标签内容
                                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获取文本尺寸
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 绘制边框
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - 5), color, -1)  # 绘制标签背景
                                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 在框内绘制标签

                                # 将检测结果写入 txt 文件
                                f_txt.write(f"{current_time}, {video_time:.2f}s, {class_name}, {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}, {confidence:.2f}\n")

                        # 将处理后的帧写入输出视频
                        out.write(frame)

                        # 保存标注帧图片
                        frame_image_path = os.path.join(output_video_image_dir, f"{current_time}_{frame_id}.jpg")  # 构造帧图片保存路径
                        cv2.imwrite(frame_image_path, frame)  # 保存当前帧图片

                cap.release()  # 释放视频资源
                out.release()  # 关闭视频写入对象
                print(f"Finished processing video: {video_file}")  # 打印完成处理信息

# elif os.path.isdir(input_path) or os.path.isfile(input_path):
#     # 图片输入情况
#     image_paths = [input_path] if os.path.isfile(input_path) else [
#         os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.jpg', '.png'))
#     ]  # 如果是文件则生成单文件列表，若为目录则获取目录下所有图片

    # 处理图片文件
    if image_files:
        print(f"Found {len(image_files)} image files.")  # 打印找到的图片文件数量

        for image_file in image_files:  # 遍历每张图片
            image_path = os.path.join(input_path, image_file)  # 构造完整图片路径
            print(f"Processing image: {image_path}")  # 打印当前处理的图片路径

            # 读取图片
            image = cv2.imread(image_path)  # 使用 OpenCV 读取图片
            if image is None:  # 检查图片是否成功读取
                print(f"Error: Unable to read image file '{image_file}'.")  # 打印错误信息
                continue  # 跳过该图片

            # 使用 YOLO 模型对图片进行预测
            results = model.predict(source=image, device='0', conf=0.25, save=False, verbose=False)

            # 打开图片对应的 txt 文件以追加模式写入
            txt_path = os.path.join(output_images_anchor_txt_dir, f"{image_file}[0].txt")
            with open(txt_path, 'a') as f_txt:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
                # img = cv2.imread(image_path)  # 读取原始图片
                for result in results:
                    for box in result.boxes:  # 遍历每个检测框
                        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 提取边界框坐标
                        confidence = box.conf.tolist()[0]  # 检测框置信度
                        class_id = int(box.cls.tolist()[0])  # 检测到的类别 ID
                        class_name = class_names[class_id]  # 类别名称

                        # 确定颜色，优先使用固定颜色映射
                        color = fixed_colors.get(class_name, colors[class_id])

                        # 绘制检测框和标签在框内
                        label = f"{class_name} {confidence:.2f}"  # 设置标签内容
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获取文本尺寸
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 绘制边框
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - 5),color, -1)  # 绘制标签背景
                        cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1)  # 在框内绘制标签

                        # 写入检测结果到 txt 文件
                        f_txt.write(f"{current_time}, {class_name}, {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}, {confidence:.2f}\n")

                output_image_path = os.path.join(output_images_dir, f"{image_file}.jpg")  # 输出图片路径
                cv2.imwrite(output_image_path, image)  # 保存标注后的图片
                print(f"Processed image: {image_file}")  # 打印处理进度

        print("Image processing complete.")  # 输出处理完成信息


else:
    print(f"Error: '{input_path}' is not a valid directory.")  # 如果路径无效，打印错误信息
