from flask import Flask, jsonify, request
from predictChessPiece.predictor import loadParams, predict, convertToNumpyArray
from PIL import Image

import numpy as np
import json
import os
import urllib.request
import imghdr
import base64
import collections

app = Flask(__name__)

# Activate app debuging
app.debug = False

@app.route('/predict', methods=['POST'])
def predictImage():
    response = []    
    if request.method != 'POST':
        response['status'] = False
        response['message'] = 'Invalid request'
    else:
        requiredPredictions = request.get_json()
        parameters = loadParams()

        for data in requiredPredictions:            
            # Implement prediction function here
            # Get current image
            rowResponse = collections.OrderedDict()
            
            key = data['PositionID']
            rawImgData = data['PositionImageByte']
            
            # Decode to base64            
            #b64img = base64.b64decode(image)
            #Image.frombytes("L", (64,64), b64img,"raw","L",0,1)

            imageArray = convertToNumpyArray(rawImgData)
            image = np.array(Image.fromarray(imageArray).resize((64,64),Image.BILINEAR)).reshape((1, 64*64*3)).T
                        
            predictedPiece = predict(image, parameters)
            rowResponse["PositionID"]=key
            rowResponse["PredictedPieceID"]=str(np.squeeze(predictedPiece))
            response.append(rowResponse)
  
    return json.dumps(response)