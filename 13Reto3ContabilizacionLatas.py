import cv2
from ultralytics import YOLO

# 1. Carga tu modelo (entrenado para detectar latas)
model_path = "yolov8x.pt"   # o "yolov8x.pt" si usas COCO y la clase 'bottle'
modelo = YOLO(model_path)

# 2. Abre el vídeo
video_path = "./videos/latas.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Define la ROI (caja) donde contarás las latas
#    (x1, y1) es la esquina superior izquierda, (x2, y2) la inferior derecha
x1, y1, x2, y2 = 100, 200, 800, 600

# 4. Identificador de clase para 'lata'
#    Si tu modelo está entrenado en COCO, la clase 'bottle' es 39:
can_class = 39  

cv2.namedWindow("Conteo de latas", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5. Detección
    results = modelo(frame)[0]                # detect()
    boxes = results.boxes.xyxy.cpu().numpy()  # array de [x1,y1,x2,y2]
    classes = results.boxes.cls.cpu().numpy() # array de clases
    confidences = results.boxes.conf.cpu().numpy()

    count = 0
    # 6. Recorre cada detección válida
    for (bx1, by1, bx2, by2), cls, conf in zip(boxes, classes, confidences):
        if cls == can_class and conf > 0.3:   # umbral de confianza
            # centro del bounding box
            cx, cy = int((bx1+bx2)/2), int((by1+by2)/2)
            # ¿Está dentro de la ROI?
            if x1 < cx < x2 and y1 < cy < y2:
                count += 1
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

    # 7. Dibuja la ROI y el conteo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(frame,
                f"Latas en caja: {count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,0,255),
                3)

    cv2.imshow("Conteo de latas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
