from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8n.pt")

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

cap = cv2.VideoCapture(1)
cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)

while True:
    capturaOK, frame = cap.read()
    if not capturaOK:
        break
    
    resultados = modelo(frame, stream=True, verbose=False)
    
    for r in resultados:
        cajas = r.boxes
        
        for caja in cajas:
            porcentajeCoincidencia = int(caja.conf[0]*100)
            clase = int(caja.cls[0])
            # print(nombresDeClase[clase], porcentajeCoincidencia)
            
            x1, y1, x2, y2 = caja.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 3)
            cv2.putText(frame, nombresDeClase[clase], (x1,y1+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
    cv2.imshow("Resultados", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()