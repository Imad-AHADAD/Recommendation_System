from django.shortcuts import render, redirect

import numpy as np
import pandas as pd
import pickle

from .ML_Function import preprocess_data

def index(request):
    if request.method == "POST":
        Requirements = request.POST.get("Requirements")
        project_Category = request.POST.get("project Category")
        Requirement_Category = request.POST.get("Requirement Category")
        Risk_Target_Category = request.POST.get("Risk Target Category")
        Probability = float(request.POST.get("Probability"))
        Magnitude_of_Risk = request.POST.get("Magnitude of Risk")
        Impact = request.POST.get("Impact")
        Dimension_of_Risk = request.POST.get("Dimension of Risk")
        Afftecting_No_of_Modules = int(float(request.POST.get("Afftecting No of Modules")))
        Fixing_Duration = int(float(request.POST.get("Fixing Duration")))
        Fix_Cost = int(float(request.POST.get("Fix Cost")))
        Priority = float(request.POST.get("Priority"))
        preprocess_data
        
        # Load the model from the pickle file
        with open('playground/deploy/svc_model.pkl', 'rb') as file:
            svc_model = pickle.load(file)
        
        # Load the function from the file
        with open('playground/deploy/preprocess_function.pkl', 'rb') as file:
            preprocessSample = pickle.load(file)

        # creating our data sample (one row)
        X_input = pd.array([Requirements , project_Category  , Requirement_Category , Risk_Target_Category , Probability , Magnitude_of_Risk , Impact, Dimension_of_Risk , Afftecting_No_of_Modules , Fixing_Duration ,  Fix_Cost , Priority , 1])
    
        colonnes = ["Requirements", "project_Category", "Requirement_Category", "Risk_Target_Category","Probability", "Magnitude_Risk", "Impact", "Dimension_Risk", "Afftecting_No_Modules", "Fixing_Duration", "Fix_Cost", "Priority","Risk_Level"]
        # Convert single_sample_data into a DataFrame with column names
        single_sample_df = pd.DataFrame([X_input], columns=colonnes)
        # Preprocess the single sample
        preprocessed_sample = preprocessSample(single_sample_df)
        # Extract the target column from single_sample_df
        target_column = preprocessed_sample['Risk_Level']
        preprocessed_sample = preprocessed_sample.drop('Risk_Level', axis=1)

        # Reshape the sample to match the expected input shape
        expected_length = 670
        num_columns = preprocessed_sample.shape[1]
        # Reshape the sample to match the expected input shape
        reshaped_sample = np.reshape(preprocessed_sample.values, (1, -1))
        reshaped_sample = np.pad(preprocessed_sample.values, (0, expected_length - num_columns), mode='constant')

        # Make a prediction
        predict = svc_model.predict(reshaped_sample)[0].astype(int)

        if predict>2:
            return redirect('risk-high-index', predict=predict)
        else:
            return redirect('risk-low-index', predict=predict)

    return render(request, "index.html")

def index1(request, predict):
    return render(request, "index3.html", {'predict': predict})

def index2(request, predict):
    return render(request, "result1.html", {'predict': predict})
