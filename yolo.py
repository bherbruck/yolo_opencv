import cv2
import numpy as np
import settings


def run(video, weights, config, classes, scale, con_threshold, nms_threshold):
    cap = cv2.VideoCapture(video)
    net = cv2.dnn.readNet(weights, config)
    colors = get_classes_colors(classes)

    while True:
        _, img = cap.read()
        height, width = img.shape[:2]

        detections = detect(img, net, scale, con_threshold, nms_threshold)
        for detection in detections:
            cls = str(classes[detection['cls']])
            con = int(detection['con'] * 100)
            draw_bounding_box(img,
                              f'{cls} {con}%',
                              int(detection['x']),
                              int(detection['y']),
                              int(detection['w']),
                              int(detection['h']),
                              color=colors[detection['cls']])

        cv2.imshow('video', img)

        # press Escape to exit
        key = cv2.waitKey(1)
        if key == 27:
            break


def detect(img, net, scale, con_threshold, nms_threshold):
    blob = cv2.dnn.blobFromImage(img,
                                 scalefactor=scale,
                                 size=(224, 224),
                                 mean=(0, 0, 0),
                                 swapRB=True,
                                 crop=False)
    net.setInput(blob)
    class_ids = []
    confidences = []
    boxes = []
    height, width = img.shape[:2]
    outputs = net.forward(get_output_layers(net))
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= con_threshold:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(detection[0] * width) - w / 2
                y = int(detection[1] * height) - h / 2
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, con_threshold, nms_threshold)

    return [{'x': boxes[i][0],
             'y': boxes[i][1],
             'w': boxes[i][2],
             'h': boxes[i][3],
             'cls': class_ids[i],
             'con': confidences[i]}
            for ii in indices
            for i in ii]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_bounding_box(img, label, x, y, w, h, color=(0, 255, 0)):
    x2 = x + w
    y2 = y + h
    cv2.rectangle(img,
                  (x, y),
                  (x2, y2),
                  color,
                  2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_classes_colors(classes):
    return np.random.uniform(0, 255, size=(len(classes), 3))


if __name__ == '__main__':
    run(video=settings.VIDEO,
        weights=settings.WEIGHTS,
        config=settings.CONFIG,
        classes=settings.CLASSES,
        scale=settings.SCALE,
        con_threshold=settings.CON_THRESHOLD,
        nms_threshold=settings.NMS_THRESHOLD)
