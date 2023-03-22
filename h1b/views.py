from django.shortcuts import render
from django.http import HttpResponse
import pickle
import pandas as pd
import numpy as np

def predictor(request):
    return render(request,'index1.html')

def check(request):
    return render(request,'newform2.html')

def forminfo(request):
    EMPLOYER_NAME=request.POST["emp"]
    JOB_TITLE=request.POST["job"]
    SOC_NAME=request.POST["soc"]
    FULL_TIME_POSITION=request.POST["full"]
    PREVAILING_WAGE=request.POST["wage"]
    WORKSITE=request.POST["work"]
    if len(EMPLOYER_NAME) == 0:
            emp_name = None
    if len(JOB_TITLE)  == 0:
            score = None
    if len(SOC_NAME) == 0:
            soc_name = None
    if len(FULL_TIME_POSITION)  == 0:
            ft = None
    if len(PREVAILING_WAGE)  == 0:
            wage = None
    if len(WORKSITE)  == 0:
            worksite = None
    if(FULL_TIME_POSITION == 'Y'):
        FULL_TIME_POSITION=1
    elif(FULL_TIME_POSITION =='N'):
        FULL_TIME_POSITION=0
    data={'EMPLOYER_NAME' : EMPLOYER_NAME,'JOB_TITLE' :JOB_TITLE,'SOC_NAME':SOC_NAME,'FULL_TIME_POSITION':FULL_TIME_POSITION,'PREVAILING_WAGE':PREVAILING_WAGE,'WORKSITE':WORKSITE}
    user_input=pd.DataFrame(data,index=[0])

    model = pickle.load(open('model/ml_model.sav', 'rb'))
    scaled = pickle.load(open('model/scaler.sav', 'rb'))
    with open('model/encoder.sav', 'rb') as f:
        enc= pd.read_pickle(f)
    #enc=pickle.load(open('model/encoder.sav', 'rb'))
    user_input['PREVAILING_WAGE']=scaled.transform(user_input[['PREVAILING_WAGE']])
    enc1=enc.transform(user_input)

    prediction = model.predict(enc1)


    if prediction<0.5:
        prediction="no"
    elif prediction>=0.5:
        prediction="yes"
    return render(request,'newform2.html',{'prediction': prediction})
