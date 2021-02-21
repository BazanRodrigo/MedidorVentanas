
import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, request, Response, jsonify, render_template
import jsonpickle
from werkzeug.utils import secure_filename
import io as StringIO
import sys
import base64
from io import BytesIO
import io
import json
from PIL import Image

# construct the argument parse and parse the arguments

confthres = 0.3
nmsthres = 0.1
yolo_path = 'MedidorVentanas'

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def creando_archivo():
    tiempo_segundos = time.time()
    tiempocadena = str(time.ctime(tiempo_segundos))
    tiempo_cadena = tiempocadena.replace(':','-')
    tiempo_cadena = tiempo_cadena + '.txt'
    return tiempo_cadena

def get_predection(image,net,LABELS,COLORS,tiempo_cadena):        
    f = open(tiempo_cadena,'w')
    f.close()            
    print(tiempo_cadena)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            #print(scores)
            classID = np.argmax(scores)
            #print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                #print(box)
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            f = open(tiempo_cadena,'a')
            f.write((str(LABELS[classIDs[i]])+' ('+str(boxes[i][0]) +', '+ str(boxes[i][1])+', '+str(boxes[i][2])+', '+ str(boxes[i][3])+')'+'\n'))
            f.close()          
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #print(boxes)
            #print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            cv2.imwrite('prediction.jpg', image)    
    return image

def medidor(results):
    coinexist = False
    windowexist = False
    f = open(results, "r")
    for linea in f:
        if 'CoinAi' in linea:
            coindata = linea
            coinexist = True
        if 'WindowAi' in linea:
            windowdata = linea      
            windowexist = True
    f.close()
    if windowexist and coinexist:
        coinparini = coindata.index('(') +1
        coinparfin = coindata.index(')') 
        #Sacamos los datos de los parentecis
        coinsize = coindata[coinparini:coinparfin]

        winparini = windowdata.index('(')+1
        winparfin = windowdata.index(')')
        #Sacamos los datos de los parentecis
        windowsize = windowdata[winparini:winparfin]
        coma =(windowsize.index(','))
        leftxwin = int(windowsize[:coma])#topy
        coma += 1
        windowsize = (windowsize[coma:])#topy)
        coma =(windowsize.index(','))
        topywin  = int(windowsize[:coma])#topy
        coma += 1
        windowsize = (windowsize[coma:])#topy)
        coma =(windowsize.index(','))
        widthwin  = int(windowsize[:coma])#topy
        coma += 1
        windowsize = (windowsize[coma:])#topy)
        heightwin  = int(windowsize[:])#topy
        windowsize = (windowsize[:])#topy)
        coma =(coinsize.index(','))
        leftxcoin = int(coinsize[:coma])#topy
        coma += 1
        coinsize = (coinsize[coma:])#topy)
        coma =(coinsize.index(','))
        topycoin  = int(coinsize[:coma])#topy
        coma += 1
        coinsize = (coinsize[coma:])#topy)
        coma =(coinsize.index(','))
        widthcoin  = int(windowsize[:coma])#topy
        coma += 1
        coinsize = (coinsize[coma:])#topy)
        heightcoin  = int(coinsize[:])#topy
        coma += 1
        coinsize = (coinsize[:])#topy)


        if(widthcoin<=heightcoin):
            mmperpix = widthcoin/2.8
            alto = (widthwin/mmperpix)
            ancho = (heightwin/mmperpix)
        else:
            mmperpix = heightcoin/2.8
            alto = (widthwin/mmperpix)
            ancho = (heightwin/mmperpix)
        resultados = 'La ventana mide ', alto,'cm', ancho, 'cm'
        print(resultados)
        return resultados
    else:
        print()
        resultados = 'No se detecto con claridad la moneda o la ventana Reintenta contemplando las recomendaciones'
        return resultados


labelsPath="obj.names"
cfgpath="yolov4.cfg"
wpath="yolov4.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
# Initialize the Flask application
app = Flask(__name__)

@app.route("/")
def upload_file():
	# renderiamos la plantilla "formulario.html"
	return render_template('formulario.html')

app.config['UPLOAD_FOLDER'] = './'

def treat_as_plain_text(response):
    response.headers["content-type"] = "text/plain"
    return response

# route http posts to this method
@app.route('/upload', methods=['POST'])
def main():
    # load our input image and grab its spatial dimensions
    #image = cv2.imread("./test1.jpg")
	if request.method == 'POST':
	  # obtenemos el archivo del input "archivo"
		f = request.files['archivo']
		filename = secure_filename(f.filename)
		# Guardamos el archivo en el directorio "Archivos PDF"
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	    # Retornamos una respuesta satisfactoria
		imgenSubida = os.path.join('./',filename)
		img = Image.open(imgenSubida)
		npimg=np.array(img)
		image=npimg.copy()
		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		archivo = creando_archivo()
		res=get_predection(image,nets,Lables,Colors,archivo)
		#meididas = (medidor(archivo))
	return render_template('medicionfinal.html', prediccion = medidor(archivo))

    

    # start flask app
if __name__ == '__main__':
    app.run(debug=True)
#https://medium.com/analytics-vidhya/object-detection-using-yolo-v3-and-deploying-it-on-docker-and-minikube-c1192e81ae7a