import logging
import time

from rtsp_gst.classes import BUFFER, FPS_CALCULATOR, STATE_CHANGE, \
    STATE, TAGS, RTSP_CAPS, X_RTP_JITTERBUFFER_STATS, X_RTP_BIN_STATS
    
from rtsp_gst import Gst

class BUS_HANDLERS():

    fps_calculcator = FPS_CALCULATOR()
    fps = 0

    def on_status_changed(self, bus, message):
        """
           STATUS Change Bus Callback
        """
        state = message.parse_state_changed()
        self.state = STATE_CHANGE(
            STATE(state.oldstate),
            STATE(state.newstate),
            STATE(state.pending)
        )
        logging.debug(f"State | {self.state}")

    def on_eos(self, bus, message):
        """
           EOS Bus Callback
        """
        logging.warning(f"EOS | {message}")

    def on_qos(self, live, running_time, stream_time, timestamp, duration):
        """
           QOS Bus Callback
        """
        logging.debug(f"QOS | {running_time} {stream_time} {timestamp} {duration}")

    def on_element(self, bus, message):
        """
           ELEMENT Bus Callback
        """
        logging.debug(f"ELEMENT | {message.get_structure ()}")

    def on_info(self, bus, message):
        """
           INFO Bus Callback
        """
        logging.info(f"EOS | {message.parse_info_details()}")

    def on_error(self, bus, message, player):
        """
           ERROR Bus Callback
        """
        gerror, debug = message.parse_error()
        logging.error(gerror)
        if player.restart_on_error:
            logging.warning("Restart Pipeline on Error")
            player.pipeline.set_state(Gst.State.NULL) 
            player.pipeline.set_state(Gst.State.PLAYING) 
        else:
            player.loop.quit()

    def on_stream_status(self, bus, message):
        """
           STREAM Bus Callback
        """
        logging.debug(f"STREAM | {message.parse_stream_status()}")

    def on_tag(self, bus, message):
        """
           TAG Bus Callback
        """
        taglist = message.parse_tag()
        self.tags = TAGS(
            taglist.get_uint("bitrate").value,
            taglist.get_uint("minimum-bitrate").value,
            taglist.get_uint("maximum-bitrate").value,
            taglist.get_string("video-codec").value,
            message.src.name
        )

    def on_handoff(self, identity, buffer):
        """
           identity element callback, used to calculate FPS
        """
        self.identity_buffer = BUFFER(buffer.dts, buffer.duration,
                             buffer.offset, buffer.offset_end, buffer.pts)
        self.calculate_fps()

    def calculate_fps(self):
        """
            Calculate FPS
        """
        self.fps = self.fps_calculcator()

    def on_pad_added(self, rtspsrc, pad):
        caps = pad.get_current_caps()
        structure = caps.get_structure(0)

        if structure.get_string("media") == "video":
            # Default resolution in case missing
            frames_raw = structure.get_string("a-framesize")
            frames = ["0", "0"]
            if frames_raw:
                try:
                    frames = frames_raw.split("-")
                except Exception as e:
                    logging.warning(f"Error splitting a-framesize: {e}")

            self.rtsp_video_caps = RTSP_CAPS(
                structure.get_int("payload").value if structure.has_field("payload") else 96,
                structure.get_int("clock-rate").value if structure.has_field("clock-rate") else 90000,
                structure.get_string("packetization-mode") or "1",
                structure.get_string("encoding-name") or "H264",
                structure.get_string("profile-level-id") or "baseline",
                frames_raw or "unknown",
                float(structure.get_string("a-framerate") or "0.0"),
                int(frames[0]),
                int(frames[1])
            )

            logging.info(f"RTSP CAPS (VIDEO) | {self.rtsp_video_caps}")

        self.rtspsrc.link(self.rtph264depay)


    def on_new_manger(self, rtspsrc, manager):
        """
            Callback to recive manger to handler jitterbuffer callback
        """
        manager.connect("new-jitterbuffer", self.on_new_jitterbuffer)

    def on_new_jitterbuffer(self, rtpbin, jitterbuffer, session, ssrc):
        """
            Callback to sync messages on jitterbuffer
        """
        jitterbuffer.connect(
            "handle-sync", self.jitterbuffer_sync, jitterbuffer)

    def jitterbuffer_sync(self, buffer, rtp_struct, jitterbuffer):
        self.rtpbin_stats = X_RTP_BIN_STATS(
            rtp_struct.get_uint64("base-time").value,
            rtp_struct.get_uint64("sr-ext-rtptime").value,
            rtp_struct.get_uint("clock-rate").value,
            rtp_struct.get_uint("base-rtptime").value,
        )

        struct = jitterbuffer.get_properties("stats")[0]
        self.jitterbuffer_stats = X_RTP_JITTERBUFFER_STATS(
            struct.get_uint64("num-pushed").value,
            struct.get_uint64("num-lost").value,
            struct.get_uint64("num-late").value,
            struct.get_uint64("num-duplicates").value,
            struct.get_uint64("avg-jitter").value / 1000000,
            struct.get_uint64("rtx-count").value,
            struct.get_uint64("rtx-success-count").value,
            struct.get_uint64("rtx-per-packet").value,
            struct.get_uint64("rtx-rtt").value,
            self.rtpbin_stats
        )

        # Safely access bitrate, fps, resolution
        bitrate = getattr(self.tags, "bitrate", "N/A")
        fps = self.fps or "N/A"
        if hasattr(self, "rtsp_video_caps"):
            res = f"{self.rtsp_video_caps.frame_height}x{self.rtsp_video_caps.frame_width}"
        else:
            res = "N/A"

        logging.info(f"JB-Lost: {self.jitterbuffer_stats.num_lost} | JB-Avg: {self.jitterbuffer_stats.avg_jitter}ms | Bitrate: {bitrate} | Fps: {fps} | Resolution: {res}")
