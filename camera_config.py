import os

class CameraProfiles:
    @staticmethod
    def get_hikvision_rtsp(ip: str, user: str = "admin", password: str = "password", channel: int = 1, stream: str = "main") -> str:
        # stream: 'main' -> 01, 'sub' -> 02
        stream_code = "01" if stream == "main" else "02"
        return f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}{stream_code}"

    @staticmethod
    def get_unv_rtsp(ip: str, user: str = "admin", password: str = "password", stream: str = "main") -> str:
        # stream: 'main' -> s0, 'sub' -> s1
        stream_code = "s0" if stream == "main" else "s1"
        return f"rtsp://{user}:{password}@{ip}:554/unicast/c1/{stream_code}/live"

    @staticmethod
    def get_dahua_rtsp(ip: str, user: str = "admin", password: str = "password", channel: int = 1, stream: str = "main") -> str:
        # stream: 'main' -> subtype=0, 'sub' -> subtype=1
        subtype = 0 if stream == "main" else 1
        return f"rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

    @staticmethod
    def get_generic_rtsp(ip: str, port: int = 554, path: str = "live", user: str = "", password: str = "") -> str:
        auth = f"{user}:{password}@" if user and password else ""
        return f"rtsp://{auth}{ip}:{port}/{path}"

if __name__ == "__main__":
    print("Hikvision RTSP :", CameraProfiles.get_hikvision_rtsp("192.168.1.100"))
    print("Uniview RTSP   :", CameraProfiles.get_unv_rtsp("192.168.1.105"))
    print("Dahua RTSP     :", CameraProfiles.get_dahua_rtsp("192.168.1.110"))