import cv2
import torch


model = torch.hub.load("ultralytics/yolov5","yolov5n",pretrained = True)

def detector():
    captura = cv2.VideoCapture(0)
    while captura.isOpened():
        status,frame = captura.read() #Llama frames constantemente 

        if not status:
            break

        prediccion = model(frame)# crea inferencias 
        df = prediccion.pandas().xyxy[0]# transforma a formato de pandas, xyxy es las cordenadas xmin,ymin,xmax,ymax 
        
        df = df[df["confidence"]> 0.5]# Quita todos los que sean mayores a 0.5
        
        for i in range(df.shape[0]):
           bbox = df.iloc[i][["xmin","ymin","xmax","ymax"]].values.astype(int)
           if df.iloc[i]['name'] == "person":
             cv2.rectangle(frame,(bbox[0],bbox[1],bbox[2],bbox[3]),(255,0,0),2)#pinta un rectangulo mediante las cordenadas en el frame 
             cv2.putText(frame, f"{df.iloc[i]['name']}:{round(df.iloc[i]['confidence'],4)}",
                        (bbox[0],bbox[1]-15),
                        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
             cantidad = df.shape
             if cantidad[0] > 2:
                print("Sobrepasa la cantidad maxima del elevador")   
              
               

        cv2.imshow("frame",frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    captura.release()


detector()        