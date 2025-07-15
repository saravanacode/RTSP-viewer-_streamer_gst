import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
from ultralytics import YOLO
import queue
import threading
import time
import torch 

class RTSPYOLOPlayer:
    def __init__(self, rtsp_url, model_path="best_5k.pt"):
        Gst.init(None)
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        import torch

        if torch.cuda.is_available():
            self.model.to("cuda")  # Moves model to GPU

    def create_pipeline(self):
        self.pipeline = Gst.parse_launch(f"""
            rtspsrc location={self.rtsp_url} latency=100 !
            rtph264depay ! h264parse ! avdec_h264 ! videoconvert !
            video/x-raw, format=BGR ! appsink name=appsink emit-signals=true sync=false max-buffers=2 drop=true
        """)

        appsink = self.pipeline.get_by_name("appsink")
        appsink.connect("new-sample", self.on_new_sample)
        return True

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            data = np.frombuffer(map_info.data, np.uint8)
            frame = data.reshape((height, width, 3))
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())
            else:
                self.frame_queue.get_nowait()  # Drop oldest
                self.frame_queue.put_nowait(frame.copy())

        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def run(self):
        if not self.create_pipeline():
            print("Failed to create pipeline.")
            return

        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        print("Press 'q' to stop.")
        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    resized = cv2.resize(frame, (640, 480))
                    results = self.model(resized, verbose=False)[0]
                    annotated = results.plot()
                    cv2.imshow("YOLOv8 RTSP", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:cctv%404753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    player = RTSPYOLOPlayer(rtsp_url)
    player.run()
