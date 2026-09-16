"""相机/视频源 — Windows(DSHOW) / Linux(V4L2) 自动选择 + 文件回放。

用法:
    cam = VisionCamera(cam_idx=0, width=1280, height=720)   # 相机
    cam = VisionCamera(video_path="run.mp4")                 # 视频文件
    ok, frame = cam.read()
"""
import sys
import cv2


class VisionCamera:
    def __init__(self, cam_idx=0, width=1280, height=720,
                 video_path=None, width_640=False):
        self.video_path = video_path
        self.cap = None
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return
        # Windows 用 DSHOW，Linux (Jetson) 用 V4L2
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(cam_idx, api)
        # USB2.0 带宽下 YUY2 到 1280×720 只有 10fps（未压缩 1.84MB/帧），
        # MJPEG 有 30fps。必须先设格式再设分辨率，否则驱动会按默认格式重排。
        # 不支持 MJPG 的相机（多为笔记本内置）会忽略，退回默认格式。
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width_640:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # V4L2 偶发第一帧无效，先抓几帧唤醒
        for _ in range(3):
            ok, _ = self.cap.read()
            if ok:
                break

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

    @property
    def size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            w, h = 1280, 720
        return w, h

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    @property
    def fourcc(self):
        """驱动实际生效的像素格式（4 字符），用于确认 MJPG 有没有设上。"""
        if self.cap is None:
            return "----"
        v = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
