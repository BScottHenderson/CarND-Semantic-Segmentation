import cv2
import os

# image_folder = './data/data_road/testing/image_2'
# video_name = './data/data_road/testing/image_2.avi'

image_folder = './runs/1554575624.8604581'
video_name = './runs/1554575624.8604581.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
