from ultralytics import YOLO
from ultralytics.solutions import object_counter as oc
import cv2

objetosConteoEntrada = {"car":0, "truck":0}
objetosConteoSalida = {"car":0, "truck":0}

modelo = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("./carros1.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# linea = [(0,h-800), (w,h-800)]
area = [(0,h-900), (w,h-900), (w,h-800), (0,h-800)]

contador = oc.ObjectCounter()
contador.set_args(view_img=False, reg_pts=area, classes_names=modelo.names,
                  view_in_counts=False, view_out_counts=False,
                  objetosConteoEntrada=objetosConteoEntrada,
                  objetosConteoSalida=objetosConteoSalida)

cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    tracks = modelo.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1,
                          classes=[2,7], verbose=False)
    contador.start_counting(frame, tracks)
    
    conteoEntrada = contador.objetosConteoEntrada
    conteoSalida = contador.objetosConteoSalida
    print(conteoEntrada, conteoSalida)
    
    cv2.putText(frame, "carros entraron: "+str(conteoEntrada["car"]), 
                (100,130), cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,255), 5)
    cv2.putText(frame, "camiones entraron: "+str(conteoEntrada["truck"]), 
                (100,300), cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,255), 5)
    cv2.putText(frame, "carros salieron: "+str(conteoSalida["car"]), 
                (100,470), cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,255), 5)
    cv2.putText(frame, "camiones salieron: "+str(conteoSalida["truck"]), 
                (100,640), cv2.FONT_HERSHEY_TRIPLEX, 5, (0,0,255), 5)
    
    cv2.imshow("Resultado", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()