from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import sys

IN_COLAB = False

try:
  import google.colab
  IN_COLAB = True
except:
  pass

import os

if IN_COLAB == True:
  sys.path.insert(0, os.path.abspath('functions'))
  sys.path.insert(0, os.path.abspath(''))
  PROJECT_PATH = '/content/gdrive/MyDrive/Tesi/'
else:
  sys.path.insert(0, os.path.abspath('functions'))
  sys.path.insert(0, os.path.abspath(''))
  PROJECT_PATH = '/Users/edoardorossi/Documents/Universita/Tesi/Tesi_GDrive/'

from pkl import store_data, load_data

def compress(df):

  # Normalizzazione
  scaler = MinMaxScaler()
  df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
  scaler_fit = {'scaler_fit' : scaler}
  store_data(scaler_fit, 'pre_processing_fits.pkl')

  # Carico n_compoments da *.pkl
  ld_path = os.path.join(PROJECT_PATH, '_.pkl')

  if IN_COLAB == True:
    ld_path = os.path.join(PROJECT_PATH, '*.pkl')

  ld = load_data(ld_path)
  
  n_components = ld['n_components']

  # converto df_norm in un array numpy
  np_df = df_norm.values
  np_df.reshape(1,-1)

  #pca
  pca = PCA(n_components)
  principalComponents = pca.fit_transform(np_df)

  principalDf = pd.DataFrame(data = principalComponents)
  finalDf = principalDf
  pca_fit = {'pca_fit' : principalComponents}
  store_data(pca_fit, 'pre_processing_fits.pkl')


  return finalDf
