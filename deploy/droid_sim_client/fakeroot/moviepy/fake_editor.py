"""假的 moviepy.editor：联调时不真正写视频文件。"""


class ImageSequenceClip:
    def __init__(self, frames, fps=10):
        self.frames = frames
        self.fps = fps

    def write_videofile(self, filename, codec=None, **kwargs):
        print(f"[sim moviepy] 跳过写视频: {filename} ({len(self.frames)} 帧 @ {self.fps}fps)", flush=True)
