
import cv2

def save_video(video_path,frames):
    print("video recording..")
    height, width, layers  = frames[0].shape
    size = (width,height)
    fps = 15
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), size)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()