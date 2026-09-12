class CameraProfiles:
    @staticmethod
    def get_hikvision_rtsp(ip, user="admin", password="password", channel=1, stream="main"):
        # stream: 'main' -> 01, 'sub' -> 02
        stream_code = "01" if stream == "main" else "02"
        return f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}{stream_code}"

    @staticmethod
    def get_unv_rtsp(ip, user="admin", password="password", stream="main"):
        # stream: 'main' -> s0, 'sub' -> s1
        stream_code = "s0" if stream == "main" else "s1"
        return f"rtsp://{user}:{password}@{ip}:554/unicast/c1/{stream_code}/live"

if __name__ == "__main__":
    print("Hikvision RTSP:", CameraProfiles.get_hikvision_rtsp("192.168.1.100"))
    print("Uniview RTSP  :", CameraProfiles.get_unv_rtsp("192.168.1.105"))
