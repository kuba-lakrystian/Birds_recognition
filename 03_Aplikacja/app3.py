import base64
import os

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import librosa
import numpy as np
import pandas as pd
import keras




app = dash.Dash(__name__)



app.layout = html.Div(
    [
        html.H1("Wybór plików"),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Wybierz plik dźwiękowy z dźwiękiem ptaka"]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.H2("Twój ptak to:"),
        html.Ul(id="file-list"),
    ],
    style={"max-width": "500px"},
)


def extract_features(files):
  try:
      file_name = os.path.join(os.path.abspath('C:\\Users\\Krystian\\Desktop\\github\\Birds_recognition\\Birds_recognition')+'\\'+files)
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
      stft = np.abs(librosa.stft(X))
      chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
      contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
      tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
      sr=sample_rate).T,axis=0)
  
  except ZeroDivisionError:
      mfccs = np.zeros(40)
      chroma = np.zeros(12)
      mel = np.zeros(128)
      contrast = np.zeros(7)
      tonnetz = np.zeros(6)

  return mfccs, chroma, mel, contrast, tonnetz


@app.callback(
    Output("file-list", "children"),
    Input("upload-data", "filename"),
)
def update_output(uploaded_filenames):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None:
        for filename in uploaded_filenames:
            train_features = extract_features(filename)
            
            features_train = []
            
            features_train.append(np.concatenate((
                train_features[0],
                train_features[1], 
                train_features[2], 
                train_features[3],
                train_features[4]), axis=0))
            
            X_train = np.array(features_train)
            
            selector = pd.read_pickle(r'C:\\Users\\Krystian\\Desktop\\github\\Birds_recognition\\Birds_recognition\\00_Model\\selector.pickle')
            X_train_pruned = selector.transform(X_train)
            
            ss = pd.read_pickle(r'C:\\Users\\Krystian\\Desktop\\github\\Birds_recognition\\Birds_recognition\\00_Model\\ss.pickle')
            X_train_pruned = ss.transform(X_train_pruned)
            
            model = keras.models.load_model('C:\\Users\\Krystian\\Desktop\\github\\Birds_recognition\\Birds_recognition\\00_Model\\model_ann.h5')
            
            predictions_train = model.predict_classes(X_train_pruned)
            
            lb = pd.read_pickle(r'C:\\Users\\Krystian\\Desktop\\github\\Birds_recognition\\Birds_recognition\\00_Model\\lb.pickle')
            predictions_train = lb.inverse_transform(predictions_train)
            
            #return [html.Li(train_features[0])]

            return [html.Li(predictions_train)]            
    else:
        return [html.Li("Nie wybrano pliku dźwiękowego")]

        



if __name__ == "__main__":
    app.run_server(debug=True)