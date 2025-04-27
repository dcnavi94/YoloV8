from ultralytics import YOLO

if __name__ == "__main__":
    modelo =YOLO("yolov8m.pt")
    modelo.train(data="./datasets/placas/data.yaml", epochs=50, batch=8,
                 optimizer='Adam', lr0=0.0001, pretrained=True)