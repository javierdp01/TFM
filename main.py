import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Import custom modules/classes
#from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from Código.dataloader.SpectrumObject import SpectrumObject
import pymzml
from data_augementator import DataAugmenter

# Global constants and configuration:
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# Define the dataset structure parameters.
semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases_list = CLASSES  # same order for iteration
medios = ['Medio Ch', 'Medio Br', 'Medio Cl', 'Medio Sc', 'GU']

# Use a particular condition for training. For example, here training samples are selected when:
#   medio == 'Medio Ch' and semana == 'Semana 1'
training_media  = 'Medio Ch'
training_week   = 'Semana 1'
n_biomarkers = 10

# Base path for the data (adjust as needed)
base_path = 'C:/Users/javie/Desktop/TFM/DATA/ClostriRepro/ClostriRepro/Reproducibilidad No extracción'

###########################################
# Data Loading
###########################################
baseline_samples = []          # Will hold SpectrumObject instances (training samples)
baseline_id_label = []         # IDs extracted from file names
Y_train = []                   # Class labels

print("Loading training data ...")
for medio in medios:
    for semana in semanas:
        for clase in clases_list:
            ruta = f"{base_path}/{medio}/{semana}/{clase}"
            if os.path.exists(ruta):
                for f in os.listdir(ruta):
                    ruta_f = os.path.join(ruta, f)
                    # Select training samples from specified condition.
                    if medio == training_media and semana == training_week:
                        baseline_id_label.append(f.split('_')[0])
                        if 'mzml' in ruta_f:
                            run = pymzml.run.Reader(ruta_f)
                            spectro = [r for r in run]
                            s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            baseline_samples.append(s)
                            Y_train.append(clase)
                        else:
                            carpetas = [subf for subf in os.listdir(ruta_f)]
                            if carpetas:
                                ruta_sub = os.path.join(ruta_f, carpetas[0])
                                # Look for 'fid' and 'acqu' files in subfolders.
                                fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                if fid_files and acqu_files:
                                    ruta_fid = fid_files[0]
                                    ruta_acqu = acqu_files[0]
                                    s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                    baseline_samples.append(s)
                                    Y_train.append(clase)

# Encode the string labels to integer indices.
label_mapping = {label: idx for idx, label in enumerate(CLASSES)}
Y_train = np.array([label_mapping[label] for label in Y_train])



# Preprocesado: llamamos a la función preprocesado para realizar un data augmentation

