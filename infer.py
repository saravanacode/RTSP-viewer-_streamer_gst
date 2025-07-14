#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# RTSP GStreamer with Inference and Display

import sys
import time
import threading
import gi

# GStreamer initialization must come before importing Gst
gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

# Initialize GStreamer
Gst.init(None)

# Other imports
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

class RTSPInferencePlayer:
    def __init__(self, rtsp_url, latency=300):
        self.rtsp_url = rtsp_url
        self.latency = latency
        self.image_arr = None
        self.running = False
        self.pipeline = None
        self.main_loop = None
        
        # Load the YOLO segmentation model
        try:
            self.model = YOLO('best_5k.pt')  # YOLO
            print("Segmentation model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
    def gst_to_opencv(self, sample):
        """Convert GStreamer sample to OpenCV format"""
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get video info
        structure = caps.get_structure(0)
        format_str = structure.get_value('format')
        height = structure.get_value('height')
        width = structure.get_value('width')
        
        print(f"Format: {format_str}, Size: {width}x{height}")
        
        # Extract buffer data
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer")
            return None
            
        try:
            # Create numpy array from buffer
            if format_str == 'BGR':
                arr = np.ndarray(
                    (height, width, 3),
                    buffer=map_info.data,
                    dtype=np.uint8
                )
            elif format_str == 'RGB':
                arr = np.ndarray(
                    (height, width, 3),
                    buffer=map_info.data,
                    dtype=np.uint8
                )
                # Convert RGB to BGR for OpenCV
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                print(f"Unsupported format: {format_str}")
                return None
                
            return arr.copy()  # Make a copy since we'll unmap the buffer
        finally:
            buf.unmap(map_info)
    
    def new_buffer_callback(self, sink, data):
        """Callback for new buffer from appsink"""
        sample = sink.emit("pull-sample")
        if sample:
            arr = self.gst_to_opencv(sample)
            if arr is not None:
                # Apply inference here
                processed_frame = self.apply_inference(arr)
                self.image_arr = processed_frame
        return Gst.FlowReturn.OK
    
    def apply_inference(self, frame):
        """Apply segmentation model inference"""
        if self.model is None:
            return frame  # Return original frame if model failed to load
            
        try:
            # Run inference using YOLO
            results = self.model(frame, verbose=False)  # Process the frame
            
            # Get the annotated frame with segmentation masks
            annotated_frame = results[0].plot()
            
            # Add timestamp overlay
            cv2.putText(annotated_frame, f"RTSP Stream - {time.strftime('%H:%M:%S')}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            print(f"Error applying inference: {e}")
            return frame
    
    def create_pipeline(self):
        """Create GStreamer pipeline for RTSP"""
        # Create pipeline elements using gst-launch style
        pipeline_str = (
            "rtspsrc name=source "
            "! rtph264depay name=depay "
            "! h264parse "
            "! avdec_h264 "
            "! videoconvert "
            "! videoscale "
            "! video/x-raw,format=BGR "
            "! appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )
        
        # Create pipeline from string
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f"Failed to create pipeline: {e}")
            return False
            
        # Get elements by name
        source = self.pipeline.get_by_name("source")
        sink = self.pipeline.get_by_name("sink")
        
        if not source or not sink:
            print("Failed to get pipeline elements")
            return False
        
        # Configure source
        source.set_property("location", self.rtsp_url)
        source.set_property("latency", self.latency)
        source.set_property("protocols", 4)  # TCP
        source.set_property("retry", 50)
        source.set_property("timeout", 20)
        
        # Connect callback
        sink.connect("new-sample", self.new_buffer_callback, None)
        
        return True
    
    def on_pad_added(self, src, new_pad, depay):
        """Handle dynamic pad creation"""
        print(f"Pad added: {new_pad.get_name()}")
        
        # Check if this is a video pad
        caps = new_pad.get_current_caps()
        if caps is None:
            caps = new_pad.query_caps(None)
        
        if caps is None:
            print("Failed to get caps from pad")
            return
            
        structure = caps.get_structure(0)
        media_type = structure.get_name()
        
        print(f"Media type: {media_type}")
        
        if media_type.startswith("application/x-rtp"):
            sink_pad = depay.get_static_pad("sink")
            if sink_pad is not None and not sink_pad.is_linked():
                ret = new_pad.link(sink_pad)
                if ret == Gst.PadLinkReturn.OK:
                    print("Source pad linked successfully")
                else:
                    print(f"Failed to link source pad: {ret}")
            else:
                print("Sink pad already linked or not found")
    
    def bus_message_handler(self, bus, message):
        """Handle bus messages"""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}")
            print(f"Debug: {debug}")
            self.main_loop.quit()
        elif msg_type == Gst.MessageType.EOS:
            print("End of stream")
            self.main_loop.quit()
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
        
        return True
    
    def display_thread(self):
        """Thread for displaying frames"""
        while self.running:
            if self.image_arr is not None:
                cv2.imshow("RTSP with Inference", self.image_arr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
            time.sleep(0.01)  # Small delay to prevent high CPU usage
    
    def play(self):
        """Start playing the stream"""
        if not self.create_pipeline():
            return False
        
        # Set up bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_message_handler)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set pipeline to playing state")
            return False
        
        self.running = True
        
        # Start display thread
        display_thread = threading.Thread(target=self.display_thread)
        display_thread.daemon = True
        display_thread.start()
        
        # Create main loop
        self.main_loop = GLib.MainLoop()
        
        try:
            print("Starting playback... Press 'q' in the video window to quit")
            self.main_loop.run()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

def main():
    # RTSP URL configuration
    rtsp_url = "rtsp://admin:cctv%404753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    
    # Create and start player
    player = RTSPInferencePlayer(rtsp_url, latency=300)
    player.play()

if __name__ == "__main__":
    main()