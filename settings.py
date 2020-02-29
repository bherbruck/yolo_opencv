
VIDEO = 0
WEIGHTS = './models/weights/yolov3-tiny.weights'
CONFIG = './models/cfg/yolov3-tiny.cfg'

SCALE = 0.00392
CON_THRESHOLD = 0.2
NMS_THRESHOLD = 0.3

with open('./models/labels/yolov3.txt', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
