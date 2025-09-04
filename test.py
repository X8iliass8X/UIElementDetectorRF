from ultralytics import YOLO
model = YOLO('last.pt')

dict = model.names

for i in range(len(dict)):
    print(i, dict[i])