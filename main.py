
#Importar librerias
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Leer imagenes para entrenar algoritmo
path = './venv/ImagesAttendance'

#Lista de imagenes
images = []

#Lista de Nombres que se registran dependiendo nombre de imagen
classNames = []

#Leemos directorio
myList = os.listdir(path)


# Metodo para leer imagenes, registrar nombres de personas formateando el nombre del archivo
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)


#Codificacion de rostro
# Encontrar Encodings, meterlos a una lista y regresarla.
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR->RGB
        encode = face_recognition.face_encodings(img)[0] #Guardar primer valor de face_recognition.face_encodings
        encodeList.append(encode)
    return encodeList


#Metodo para registrar fecha y hora en las que una cara registrada se encuentra en el video
def markAttendance(name):
    with open('./venv/Attendance.csv', 'r+') as f: #Leer archivo .csv para registrar asistencia
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')

#Llamar metodo para codificar caras
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Abrir video de entrada/captura (En este caso es la webcam de la pc)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read() #Leer entrada de video
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#En cada frame en donde se encuentran las caras y se guarda en imagen salida
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) #Codificacion de la cara actual en el frame exacto

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #Comparar caras para ver si hay un match
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #Distancia entre caras
        matchIndex = np.argmin(faceDis)  #Indices de valores minimos en un axis
        if matches[matchIndex]:
            name = classNames[matchIndex].upper() #Regisrar nombre de persona encontrada

            #Creando rectangulos para indicar match en video
            y1, x2, y2, x1 = faceLoc #Guardando coordenadas de cara encontrada
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 #Reescalando
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #Rectangulo de arriba
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED) #Rectangulo del texto
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #Poner texto en rectangulo de arriba
            now = datetime.now() #Hora
            dtString = now.strftime('%H:%M:%S') # Formato a la hora
            print(name, 'Found!!', 'Date: ', dtString) # Imprimir en consola
            markAttendance(name) #Llamar metodo para registar asistencia
    cv2.imshow('Webcam', img) #Enseñar ventana de video
    cv2.waitKey(1)
