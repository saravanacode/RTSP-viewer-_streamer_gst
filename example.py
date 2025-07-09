from rtsp_gst import GST_RTSP_PLAYER, RTSP_CONFIG, GstRtsp, Gst

if __name__ == '__main__':
    grp = GST_RTSP_PLAYER(
        rtspsrc_config = RTSP_CONFIG(
            # Try both URL formats
            #location = "rtsp://admin:cctv4753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
            #location = "rtsp://admin:cctv:4753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
            # location = "rtsp://admin:cctv%4753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
            location = "rtsp://admin:cctv%404753@103.240.103.135:5054/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
            latency = 300,
            # Use TCP instead of UDP
            protocols = GstRtsp.RTSPLowerTrans.TCP
        ),
        restart_on_error = True,
        # loglevel = "DEBUG"
        # decoder = Gst.ElementFactory.make("d3d11videosink")
        # decoder = d3d11videosink
        # sink = Gst.ElementFactory.make("avdec_h264")
        # sink = avdec_h264
    )
    grp.play()
