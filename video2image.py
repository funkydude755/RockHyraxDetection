import cv2
from os import listdir, mkdir
from os.path import isfile, join

def getFrame(sec, path):
    path = path
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        print(".")
        cv2.imwrite("/home/ok/Downloads/Nature_video/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

for video_path in ['/home/ok/OAI/Bunnys/RockHyrexDetection/y2mate.com - ein_gedi_nature_reserve_and_national_park_2017_israel__v4gvnXURgE_360p.mp4']:
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    frameRate = 1 #//it will capture image in each 0.5 second
    count=1
    mkdir('/home/ok/Downloads/Nature_video/')
    success = getFrame(sec, video_path)
    print("start working")
    while success:
        print('frame: {}'.format(count))
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, video_path)

