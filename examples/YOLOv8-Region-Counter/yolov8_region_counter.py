# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse                                             # 用于解析命令行参数。
from collections import defaultdict                         # 用于创建具有默认值的字典。
from pathlib import Path                                    # 用于方便地处理文件路径。

import cv2                                                  # 用于处理计算机视觉任务。
import numpy as np                                          # 处理数组和数学计算。
from shapely.geometry import Polygon                        # 用于定义和处理多边形。
from shapely.geometry.point import Point                    # 用于处理点坐标。

from ultralytics import YOLO                                # 用于创建 YOLO 模型。
from ultralytics.utils.files import increment_path          # 用于递增文件路径，避免覆盖已存在的文件。
from ultralytics.utils.plotting import Annotator, colors    # 用于在图像上绘制标注及处理颜色。

# 全局变量
# 创建一个默认值为列表的字典，用于存储每个追踪对象的历史轨迹。
track_history = defaultdict(list)
# 初始化当前选中的区域为空。
current_region = None
# 定义需要计数的区域，包括多边形区域和矩形区域的定义，包括名称、几何形状、计数、拖动状态和颜色信息。
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):           # 定义一个回调函数，用于处理鼠标事件，允许用户与图像的区域进行交互。
    """
    Handles mouse events for region manipulation.

    Args:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Notes:
        - This function is intended to be used as a callback for OpenCV mouse events.
        - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region                            # 声明 current_region 为全局变量，以便在函数内修改它。

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:              # 遍历所有计数区域，检查鼠标点击点是否在区域内。如果在区域内，设置 current_region 为所选区域，并将其拖动状态设置为 True，同时记录鼠标点击的初始位置（偏移量）。
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):   # 如果有区域正在被拖动，更新区域位置。
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:              # 检查鼠标移动事件。
            # 计算鼠标当前坐标与初始坐标的差值，并根据这个差值更新多边形区域的新位置，重新创建 Polygon 对象，同时更新偏移量。
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:      # 检查左键抬起事件。
        # 如果某个区域正在被拖动，结束拖动状态。
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


# 主要运行函数
def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):  # 定义 run 函数，接受多个参数，用于设置模型权重、视频源、设备类型、显示和保存结果的选项等。
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0     # 初始化视频帧计数器。

    # Check source path
    # 检查视频源路径是否存在，如果不存在则抛出 FileNotFoundError。
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")                              # 创建 YOLO 模型实例，使用指定的权重文件。
    model.to("cuda") if device == "0" else model.to("cpu")  # 根据设备参数选择将模型加载到 GPU（如果设备为 "0"）或 CPU。

    # Extract classes names
    names = model.model.names           # 提取模型中检测的类名称。 

    # Video setup
    # 打开视频文件并获取帧宽度、高度和帧率，以及视频编码格式。
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    # 使用 increment_path 生成输出目录，确保目录存在。
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 创建视频写入对象，将处理后的视频保存到指定的输出目录。
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    # 进入视频处理循环，读取每一帧视频，直到视频读取结束。
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1        # 帧计数器加一。

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)     # 对当前帧进行对象跟踪，返回检测结果。

        if results[0].boxes.id is not None:     # 如果检测到的框有有效的 ID：
            # 提取边界框坐标、追踪 ID 和类别列表。
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            # 创建 Annotator 实例，用于在图像上绘制标注。
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            # 遍历检测到的每个框、ID 和类别。
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # 在图像上绘制边框和标签。
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                # 计算边界框中心点坐标。
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                # 获取当前追踪 ID 对应的轨迹列表，并在列表中添加当前边界框中心点。
                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                # 如果轨迹超过 30 个点，则移除最旧的点。
                if len(track) > 30:
                    track.pop(0)
                
                # 将轨迹点合并并绘制多条轨迹线。
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                # 检查当前检测到的物体中心是否在计数区域内，如果是，则增加该区域的计数。
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        # 绘制计数区域的信息。
        for region in counting_regions:
            # 获取区域的计数值和颜色信息。
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            # 提取多边形的坐标和质心位置。
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            # 计算区域计数标签的文字大小和位置。
            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            # 确定文本框的绘制位置。
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            # 在图像上绘制一个填充矩形，作为文字背景。
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            # 在图像上放置区域计数文本。
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            # 绘制区域的边界线。
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
        # 如果需要显示图像，创建窗口并设置鼠标回调。
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            # 显示处理后的视频帧。
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        # 如果需要保存图像，将当前帧写入视频文件。
        if save_img:
            video_writer.write(frame)

        # 在处理新一帧前重置每个区域的计数。
        for region in counting_regions:
            region["counts"] = 0

        # 检查是否按下 "q" 键，如果按下则退出循环。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # 清理并释放系统资源，关闭所有窗口。
    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


# 参数解析函数
def parse_opt():    # 定义参数解析函数。
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()              # 创建参数解析器实例。
    # 添加参数：weights，指定模型权重的路径。
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    # 添加参数：device，指定使用的设备类型。
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加参数：source，指定视频文件路径为必需项。
    parser.add_argument("--source", type=str, required=True, help="video file path")
    # 添加参数：view-img，是否显示处理结果。
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加参数：save-img，是否保存处理结果。
    parser.add_argument("--save-img", action="store_true", help="save results")
    # 添加参数：exist-ok，是否允许重名的输出目录。
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 添加参数：classes，允许过滤特定类别的检测。
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # 添加参数：line-thickness，指定边界框的线条厚度。
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    # 添加参数：track-thickness，指定追踪线的厚度。
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    # 添加参数：region-thickness，指定绘制区域边界的厚度。
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    # 解析命令行参数并返回。
    return parser.parse_args()


# 主函数
def main(opt):      # 定义 main 函数，调用 run 并传入解析的参数。
    """Main function."""
    run(**vars(opt))


# 入口点
if __name__ == "__main__":      # 
    opt = parse_opt()
    main(opt)
