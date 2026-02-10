from moviepy.editor import VideoFileClip
import os

# Paths inside the container
VIDEO_PATH = "/home/jovyan/work/src/static/dashboard_demo.mp4"
OUTPUT_PATH = "/home/jovyan/work/docs/capturas/dashboard_demo.gif"

print(f"Loading video from {VIDEO_PATH}...")

if not os.path.exists(VIDEO_PATH):
    print("Error: Video file not found!")
    exit(1)

try:
    clip = VideoFileClip(VIDEO_PATH)
    # Take a 10-second subclip from 00:05 to 00:15
    subclip = clip.subclip(5, 15)
    # Resize to width 600px to keep file size manageable
    subclip = subclip.resize(width=600)
    
    print(f"Writing GIF to {OUTPUT_PATH}...")
    subclip.write_gif(OUTPUT_PATH, fps=10)
    print("Success!")
except Exception as e:
    print(f"Error processing video: {e}")
