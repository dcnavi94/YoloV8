from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8l.pt")

nombresDeClase = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

img = cv2.imread("1.jpg")
cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)

resultados = modelo(img, stream=False, verbose=False)

for r in resultados:
    cajas = r.boxes
    
    for caja in cajas:
        porcentajeCoincidencia = int(caja.conf[0]*100)
        clase = int(caja.cls[0])
        # print(nombresDeClase[clase], porcentajeCoincidencia)
        
        x1, y1, x2, y2 = caja.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.putText(img, nombresDeClase[clase], (x1,y1+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
cv2.imshow("Resultados", img)
cv2.waitKey()
cv2.destroyAllWindows()