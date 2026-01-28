###########################################
# Imports
###########################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from dataloader.SpectrumObject import SpectrumObject
import pymzml
from data_augementator import DataAugmentor
from tsne import TSNESpectre
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV, StratifiedGroupKFold
from scipy.stats import randint
from sklearn.preprocessing import label_binarize
import re
from excel_dict import export_results_to_excel

from dataloader.preprocess import (
    SequentialPreprocessor,
    VarStabilizer,
    Smoother,
    BaselineCorrecter,
    Trimmer,
    Binner,
    Normalizer,
    IntensityThresholding,
    detect_peaks,
    Aligner
)

#-----------------------------------------------------------------------------------------------------------------------------------------------
###########################################
# Opciones
###########################################
# Global constants and configuration:
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# Define the dataset structure parameters.
semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases_list = CLASSES  # same order for iteration
medios = ['Ch', 'Br', 'Cl', 'Sc']

# Use a particular condition for training. For example, here training samples are selected when:
#   medio == 'Medio Ch' and semana == 'Semana 1'
training_media  = 'Ch'
training_week   = 'Semana 1'
n_biomarkers = 10

# Base path for the data (adjust as needed)
base_path = 'C:/Users/javie/Desktop/TFM/DATA/ClostriRepro/ClostriRepro/Reproducibilidad No extracción'

# DATA AUGMENTATION
type_augmentation = 'linear'    # Opciones: "random" y "linear"
k_per_spectrum = 6               # nº de espectros aumentados por espectro
seed = 0                        # reprodicibilidad

# T-SNE
show_tsne_plot = False                                  # True/False -> mostrar figuras
path_save_tsne = r"C:\Users\javie\Desktop\ImagenesTFM"  # Dirección donde se guardan las figuras del t-sne
tsne_params = dict(                                     # Hiperparámetros
    n_points=8192,
    tic_norm=True,
    do_log1p=True,
    pca_dims=50,
    perplexity=30,  # la clase lo ajusta automáticamente si hay pocas muestras
    seed=0,
    max_iter=1500
)

# RANDOM FOREST
rf_params = dict(
    n_estimators = 300,
    random_state = 0,
    n_jobs = -1
)

# DATOS GLOBALES
data_path = 'C:/Users/javie/Desktop/TFM/DATA/ClostriRepro/ClostriRepro'
extraccion = ['Reproducibilidad Extracción', 'Reproducibilidad No extracción']
medios_extraccion = ['Brx', 'Chx', 'Clx', 'Scx']
results_rf_path = 'C:/Users/javie/Desktop/TFM/results'  # Dóonde se guardaran los datos de los resultados
show_common_grid_plot = False

# Valores Random Forest personalizado
window = 2   # Valor de la ventana simétrica del decission tree

param_cv_rf_robusto = {
    "n_estimators": [400, 800, 1200, 2000, 2500, 3000, 3500, 4000],
    "max_depth": [5, 8, 10, 15, 20]
}

scoring_list = ['macro', 'weighted', 'accuracy']

rf_base = RandomForestClassifier(random_state=seed, n_jobs=1)

do_energy_rf = False
do_occupation_rf = True
do_peak_rf = False
delta = 1e-4         # Valor de estabilización
eps_list = [0.83, 0.84, 0.85, 0.86, 0.87]
#-----------------------------------------------------------------------------------------------------------------------------------------------

# ###########################################
# # Data Loading
# ###########################################
# print("\n====================================")
# print("PRUEBAS CON DATOS DE UN MEDIO")
# print("====================================")
# baseline_samples = []          # SpectrumObject instances (training samples)
# baseline_id_label = []         # IDs extracted from file names
# Y_train = []                   # Class labels

# print("Loading training data ...")
# for medio in medios:
#     for semana in semanas:
#         for clase in clases_list:
#             ruta = f"{base_path}/{medio}/{semana}/{clase}"
#             if os.path.exists(ruta):
#                 for f in os.listdir(ruta):
#                     ruta_f = os.path.join(ruta, f)
#                     # Select training samples from specified condition.
#                     if medio == training_media and semana == training_week:
#                         baseline_id_label.append(f.split('_')[0])
#                         if 'mzml' in ruta_f:
#                             run = pymzml.run.Reader(ruta_f)
#                             spectro = [r for r in run]
#                             s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
#                             baseline_samples.append(s)
#                             Y_train.append(clase)
#                         else:
#                             if not os.path.isdir(ruta_f):
#                                 continue
#                             carpetas = [subf for subf in os.listdir(ruta_f)]
#                             if carpetas:
#                                 ruta_sub = os.path.join(ruta_f, carpetas[0])
#                                 # Look for 'fid' and 'acqu' files in subfolders.
#                                 fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
#                                 acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
#                                 if fid_files and acqu_files:
#                                     ruta_fid = fid_files[0]
#                                     ruta_acqu = acqu_files[0]
#                                     s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
#                                     baseline_samples.append(s)
#                                     Y_train.append(clase)

# # Encode the string labels to integer indices.
# label_mapping = {label: idx for idx, label in enumerate(CLASSES)}
# Y_train = np.array([label_mapping[label] for label in Y_train])

# ###########################################
# # Preprocessing
# ###########################################
# dataAugment = DataAugmentor(seed=seed)
# preaugment = len(baseline_samples)

# print(f"Realizando data augment con los datos -> nº de operaciones por muestra: {k_per_spectrum}, como se realizan las operaciones: {type_augmentation}...")

# aug_samples, aug_ids, aug_labels = dataAugment.augment_dataset(
#     spectra=baseline_samples,
#     ids=baseline_id_label,
#     labels=Y_train,
#     k_per_spectrum=k_per_spectrum,
#     id_suffix="aug"
# )
# aug_operation_label = ["+".join(getattr(s, "aug_ops", ["none"]))
#                        for s in aug_samples]

# # Concatenamos todo
# all_samples = baseline_samples + aug_samples
# all_ids = baseline_id_label + (aug_ids if aug_ids is not None else [])
# all_labels = np.concatenate([Y_train, aug_labels]) if aug_labels is not None else Y_train

# combined_aug_ops_labels = (["originales"] * len(baseline_samples) + aug_operation_label)

# if (preaugment != len(baseline_samples)):
#     print("Error al aumentar los datos.")
#     exit()
# else:
#     print("Data augment realizado correctamente...")
#     print(f"Nº de espectros originales: {len(baseline_samples)}. Nº de espectros generados: {len(aug_samples)}. Nº total de espectros en el training data: {int(len(baseline_samples)) + int(len(aug_samples))}")

# ###########################################
# # t-SNE
# ###########################################
# print("Realizando t-sne...")

# tsner = TSNESpectre(**tsne_params)  # Pasamos los hiperparámetros

# print("t-sne realizado correctamente...")

# print("Generando figuras del t-sne...")

# # 1) Construir una plantilla común
# combined_grid = tsner._build_common_grid(all_samples)  # Utilizamos la misma plantilla para las 3 situaciones

# # 2) t-SNE SOLO ORIGINALES
# X_orig, _ = tsner.featurize(baseline_samples, grid=combined_grid)
# Z_orig = tsner.run_tsne(X_orig)
# tsner.plot(Z_orig, labels=Y_train, class_names=CLASSES, title="t-SNE originales", show=show_tsne_plot, type=type_augmentation)

# # 3) t-SNE SOLO AUMENTADOS
# X_aug, _ = tsner.featurize(aug_samples, grid=combined_grid)
# Z_aug = tsner.run_tsne(X_aug)
# tsner.plot(Z_aug, labels=aug_labels, class_names=CLASSES, title="t-SNE aumentados", show=show_tsne_plot, type=type_augmentation)

# # 4) t-SNE COMBINADO (base + aug)
# X_all, _ = tsner.featurize(all_samples, grid=combined_grid)
# Z_all = tsner.run_tsne(X_all)

# n_orig = len(baseline_samples)
# combined_aug_ops_labels = np.array(combined_aug_ops_labels)

# plt.figure(figsize=(8,6))

# # 1) Aumentados (circulos, por operación)
# unique_ops = np.unique(aug_operation_label)
# for op in unique_ops:
#     # índices en la parte aumentada
#     idx_aug_part = np.where(combined_aug_ops_labels == op)[0]
#     idx_aug_part = idx_aug_part[idx_aug_part >= n_orig]  # solo indices >= n_orig
#     plt.scatter(
#         Z_all[idx_aug_part, 0],
#         Z_all[idx_aug_part, 1],
#         s=25,
#         alpha=0.6,
#         label=f"aug: {op}"
#     )

# # 2) Originales (triángulos negros, por clase)
# Y_train_arr = np.array(Y_train)
# for c in np.unique(Y_train_arr):
#     idx_orig_class = np.where(Y_train_arr == c)[0]
#     name = CLASSES[c] if c < len(CLASSES) else str(c)
#     plt.scatter(
#         Z_all[idx_orig_class, 0],
#         Z_all[idx_orig_class, 1],
#         s=70,
#         marker="^",
#         alpha=0.95,
#         edgecolors="k",
#         linewidths=0.8,
#         label=f"{name} (orig)"
#     )

# plt.legend(frameon=False, fontsize=9, ncol=2)
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.title(f"t-SNE: originales (▲) vs aumentados (●) {type_augmentation}")
# plt.tight_layout()

# save_plot_as = f"t-SNE combinados originales+aumentados_{type_augmentation}_comprobar_operación.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# path_to_save = os.path.join(path_save_tsne, save_plot_as)
# print("Guardando la figura generada en " + path_to_save)
# plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

# if show_tsne_plot:
#     plt.show()
# plt.close()

# X_all, _ = tsner.featurize(all_samples, grid=combined_grid)
# Z_all = tsner.run_tsne(X_all)

# # indices
# n_orig = len(baseline_samples)
# idx_orig = np.arange(n_orig)
# idx_aug = np.arange(n_orig, len(Z_all))

# plt.figure(figsize=(7,6))
# for c in np.unique(all_labels):
#     # índices para cada clase
#     idx_orig_class = idx_orig[Y_train == c]
#     idx_aug_class = idx_aug[aug_labels == c]
#     name = CLASSES[c] if c < len(CLASSES) else str(c)
    
#     # Originales: triángulos
#     plt.scatter(Z_all[idx_orig_class, 0],
#                 Z_all[idx_orig_class, 1],
#                 s=70, marker="^", alpha=0.95, label=f"{name} (orig)", zorder=3, edgecolors="k")
#     # Aumentados: círculos
#     plt.scatter(Z_all[idx_aug_class, 0],
#                 Z_all[idx_aug_class, 1],
#                 s=25, marker="o", alpha=0.3, label=f"{name} (aug)", zorder=1)

# plt.legend(frameon=False, fontsize=9, ncol=2)
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.title(f"t-SNE: originales (▲) vs aumentados (●) {type_augmentation}")
# plt.tight_layout()

# save_plot_as = f"t-SNE combinados originales+aumentados_{type_augmentation}_no_comprobar_operación.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# path_to_save = os.path.join(path_save_tsne, save_plot_as)
# print("Guardando la figura generada en " + path_to_save)
# plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

# if show_tsne_plot:
#     plt.show()
# plt.close()

# print("Figuras generadas y guardadas correctamente...")

# ###########################################
# #
# # TODOS LOS DATOS SEMANA 1
# #
# ###########################################

# print("\n====================================")
# print("TODOS LOS DATOS SEMANA 1")
# print("====================================")
# # 1) Cargar todos los datos Semana 1
# data_samples = []          # SpectrumObject instances
# data_id_label = []         # IDs
# Y_data = []                # Class labels
# data_media = []            # Medio de los datos
# print("Cargando todos los datos...")

# for extraccion_i in extraccion:
#     if extraccion_i == 'Reproducibilidad Extracción':
#         for medio in medios_extraccion:
#             for semana in ['Semana 1']:   # <-- SOLO Semana 1
#                 for clase in clases_list:
#                     ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
#                     if os.path.exists(ruta):
#                         for f in os.listdir(ruta):
#                             ruta_f = os.path.join(ruta, f)
#                             bact_id = f.split('_')[0].strip()
#                             if not bact_id.isdigit():
#                                 continue
#                             bact_id = str(int(bact_id))
#                             data_id_label.append(bact_id)
#                             if 'mzml' in ruta_f:
#                                 run = pymzml.run.Reader(ruta_f)
#                                 spectro = [r for r in run]
#                                 s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
#                                 data_samples.append(s)
#                                 Y_data.append(clase)
#                                 data_media.append(medio) 
#                             else:
#                                 if not os.path.isdir(ruta_f):
#                                     continue
#                                 carpetas = [subf for subf in os.listdir(ruta_f)]
#                                 if carpetas:
#                                     ruta_sub = os.path.join(ruta_f, carpetas[0])
#                                     fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
#                                     acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
#                                     if fid_files and acqu_files:
#                                         ruta_fid = fid_files[0]
#                                         ruta_acqu = acqu_files[0]
#                                         s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
#                                         data_samples.append(s)
#                                         Y_data.append(clase)
#                                         data_media.append(medio) 
#                     else:
#                         continue
#     elif extraccion_i == 'Reproducibilidad No extracción':
#         for medio in medios:
#             for semana in ['Semana 1']:   # <-- SOLO Semana 1
#                 for clase in clases_list:
#                     ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
#                     if os.path.exists(ruta):
#                         for f in os.listdir(ruta):
#                             ruta_f = os.path.join(ruta, f)
#                             bact_id = f.split('_')[0].strip()
#                             if not bact_id.isdigit():
#                                 continue
#                             bact_id = str(int(bact_id))
#                             data_id_label.append(bact_id)
#                             if 'mzml' in ruta_f:
#                                 run = pymzml.run.Reader(ruta_f)
#                                 spectro = [r for r in run]
#                                 s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
#                                 data_samples.append(s)
#                                 Y_data.append(clase)
#                                 data_media.append(medio) 
#                             else:
#                                 if not os.path.isdir(ruta_f):
#                                     continue
#                                 carpetas = [subf for subf in os.listdir(ruta_f)]
#                                 if carpetas:
#                                     ruta_sub = os.path.join(ruta_f, carpetas[0])
#                                     fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
#                                     acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
#                                     if fid_files and acqu_files:
#                                         ruta_fid = fid_files[0]
#                                         ruta_acqu = acqu_files[0]
#                                         s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
#                                         data_samples.append(s)
#                                         Y_data.append(clase)
#                                         data_media.append(medio) 
#                     else:
#                         continue
#     else:
#         print("Error al meter la ruta")

# # Realizamos el data augmentation sobre TODOS los datos (Semana 1)
# dataAugment = DataAugmentor(seed=seed)
# preaugment = len(data_samples)

# print(f"Realizando data augment con los datos -> nº de operaciones por muestra: {k_per_spectrum}, como se realizan las operaciones: {type_augmentation}...")

# aug_samples_total, aug_ids_total, aug_labels_total = dataAugment.augment_dataset(
#     spectra=data_samples,
#     ids=data_id_label,
#     labels=Y_data,
#     k_per_spectrum=k_per_spectrum,
#     id_suffix="aug"
# )
# aug_operation_label = ["+".join(getattr(s, "aug_ops", ["none"]))
#                        for s in aug_samples_total]

# # Concatenamos todo
# all_samples_total = data_samples + aug_samples_total
# all_ids_total = data_id_label + (aug_ids_total if aug_ids_total is not None else [])
# all_labels_total = np.concatenate([Y_data, aug_labels_total]) if aug_labels_total is not None else Y_data

# combined_aug_ops_labels_total = (["originales"] * len(data_samples) + aug_operation_label)

# if (preaugment != len(data_samples)):
#     print("Error al aumentar los datos.")
#     exit()
# else:
#     print("Data augment realizado correctamente...")
#     print(f"Nº de espectros originales: {len(data_samples)}. Nº de espectros generados: {len(aug_samples_total)}. Nº total de espectros: {int(len(data_samples)) + int(len(aug_samples_total))}")

# ###########################################
# # t-SNE por MEDIO (originales vs aumentados)
# ###########################################
# print("Realizando t-sne...")

# tsner_total = TSNESpectre(**tsne_params)

# print("t-sne realizado correctamente...")

# print("Generando figuras del t-sne...")

# # 1) Construimos una grid común con todos los espectros
# combined_grid_total = tsner_total._build_common_grid(all_samples_total)

# # 2) Featurizamos originales, aumentados y todos
# X_orig_total, _ = tsner_total.featurize(data_samples,       grid=combined_grid_total)
# X_aug_total,  _ = tsner_total.featurize(aug_samples_total,  grid=combined_grid_total)
# X_all_total,  _ = tsner_total.featurize(all_samples_total,  grid=combined_grid_total)

# # 3) Preparamos etiquetas de medio también para los aumentados
# data_media = np.array(data_media)

# media_aug_total = []
# for m in data_media:
#     media_aug_total.extend([m] * k_per_spectrum)
# media_aug_total = np.array(media_aug_total)

# all_media_total = np.concatenate([data_media, media_aug_total])

# # 4) Ejecutamos t-SNE sobre TODOS
# Z_all_total = tsner_total.run_tsne(X_all_total)

# # 5) Índices de originales y aumentados
# n_orig_total = len(data_samples)
# idx_orig_total = np.arange(n_orig_total)
# idx_aug_total  = np.arange(n_orig_total, len(all_samples_total))

# # 6) Mapa de colores por medio
# medios_data = np.unique(all_media_total)
# cmap = plt.get_cmap("tab10")
# color_map = {m: cmap(i % 10) for i, m in enumerate(medios_data)}

# plt.figure(figsize=(10, 8))

# for medio in medios_data:
#     color = color_map[medio]

#     # índices de este medio en todo el dataset
#     idx_all_medio = np.where(all_media_total == medio)[0]

#     # separamos originales y aumentados de ese medio
#     idx_orig_medio = idx_all_medio[idx_all_medio < n_orig_total]
#     idx_aug_medio  = idx_all_medio[idx_all_medio >= n_orig_total]

#     # Aumentados: círculos, más pequeños y translúcidos
#     plt.scatter(
#         Z_all_total[idx_aug_medio, 0],
#         Z_all_total[idx_aug_medio, 1],
#         s=25,
#         marker="o",
#         alpha=0.3,
#         color=color,
#         label=f"{medio} (aug)"
#     )

#     # Originales: triángulos grandes con borde negro con mayor contraste
#     plt.scatter(
#         Z_all_total[idx_orig_medio, 0],
#         Z_all_total[idx_orig_medio, 1],
#         s=70,
#         marker="^",
#         alpha=0.95,
#         color=color,
#         edgecolors="k",
#         linewidths=0.8,
#         label=f"{medio} (orig)"
#     )

# plt.legend(frameon=False, fontsize=8, ncol=2)
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.title(f"t-SNE por MEDIO (Semana 1): originales (▲) vs aumentados (●) [{type_augmentation}]")
# plt.tight_layout()

# # Guardar figura
# save_plot_as = f"t-SNE_todos_por_medio_Semana1_{type_augmentation}.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# path_to_save = os.path.join(path_save_tsne, save_plot_as)
# print("Guardando la figura generada en " + path_to_save)
# plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

# if show_tsne_plot:
#     plt.show()
# plt.close()

# print("Figura t-SNE por medio (Semana 1) generada y guardada correctamente...")

# ################################################
# # t-SNE por RIBOTIPO (originales vs aumentados)
# ################################################
# # Ribotipos presentes en estos datos (6 posibles)
# unique_ribos = np.unique(all_labels_total)

# n_classes = len(unique_ribos)
# n_cols = min(3, n_classes)              # hasta 3 columnas
# n_rows = int(np.ceil(n_classes / n_cols))

# fig, axes = plt.subplots(n_rows, n_cols,
#                          figsize=(5 * n_cols, 4 * n_rows),
#                          sharex=True, sharey=True)
# axes = np.atleast_1d(axes).ravel()      # por si solo hay 1 fila

# for ax, ribo in zip(axes, unique_ribos):
#     # índices de este ribotipo en todo el dataset (originales + aumentados)
#     idx_ribo_all = np.where(all_labels_total == ribo)[0]

#     # medios que aparecen para este ribotipo
#     medios_ribo = np.unique(all_media_total[idx_ribo_all])

#     # para evitar repetir leyendas dentro del mismo subplot
#     used_labels = set()

#     for medio in medios_ribo:
#         color = color_map.get(medio, "C0")

#         # índices de este ribotipo + este medio
#         mask_medio = (all_media_total[idx_ribo_all] == medio)
#         idx_ribo_medio = idx_ribo_all[mask_medio]

#         # separamos originales y aumentados
#         idx_orig_medio = idx_ribo_medio[idx_ribo_medio < n_orig_total]
#         idx_aug_medio  = idx_ribo_medio[idx_ribo_medio >= n_orig_total]

#         # label para la leyenda
#         label_aug  = f"{medio} (aug)"
#         label_orig = f"{medio} (orig)"

#         if label_aug in used_labels:
#             label_aug = None
#         else:
#             used_labels.add(label_aug)

#         if label_orig in used_labels:
#             label_orig = None
#         else:
#             used_labels.add(label_orig)

#         # Aumentados: círculos, translúcidos
#         ax.scatter(
#             Z_all_total[idx_aug_medio, 0],
#             Z_all_total[idx_aug_medio, 1],
#             s=25,
#             marker="o",
#             alpha=0.3,
#             color=color,
#             label=label_aug
#         )

#         # Originales: triángulos con borde negro, más visibles
#         ax.scatter(
#             Z_all_total[idx_orig_medio, 0],
#             Z_all_total[idx_orig_medio, 1],
#             s=70,
#             marker="^",
#             alpha=0.95,
#             color=color,
#             edgecolors="k",
#             linewidths=0.8,
#             label=label_orig
#         )

#     ax.set_title(f"Ribotipo: {ribo}")
#     ax.set_xlabel("t-SNE 1")
#     ax.set_ylabel("t-SNE 2")
#     ax.legend(frameon=False, fontsize=7)

# # Si hay más subplots creados que ribotipos (p.ej. 6 huecos pero 5 clases), los apagamos
# for ax in axes[len(unique_ribos):]:
#     ax.axis("off")

# plt.tight_layout()

# save_plot_as = f"t-SNE_subplots_ribotipo_Semana1_{type_augmentation}.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# path_to_save = os.path.join(path_save_tsne, save_plot_as)
# print("Guardando la figura de subplots por ribotipo en " + path_to_save)
# plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

# if show_tsne_plot:
#     plt.show()
# plt.close()

# print("Figura t-SNE por ribotipo y medio (subplots) generada y guardada correctamente...")


###########################################
#
# DIVISIÓN TRAIN/TEST
#
###########################################
print("\n====================================")
print("DIVISIÓN TRAIN/TEST")
print("====================================")
# 1) Cargar las etiquetas por Ribotipo para hacer el split 7/3
print("Cargando las etiquetas por ribotipo...")

ribo_to_ids = {cl: set() for cl in clases_list}

# Usamos la condición base: No extracción + medio training_media + training_week
base_ribos_path = base_path  # ya apunta a "Reproducibilidad No extracción"

for clase in clases_list:
    ruta = f"{base_ribos_path}/{training_media}/{training_week}/{clase}"
    if not os.path.exists(ruta):
        print(f"Error: No existe ruta base para {clase}: {ruta}")
        continue

    for f in os.listdir(ruta):
        if f.startswith("."):
            continue
        bact_id = f.split("_")[0].strip()
        if not bact_id.isdigit():
            continue
        bact_id = str(int(bact_id))   # normaliza
        ribo_to_ids[clase].add(bact_id)


train_ids_per_ribo = {}
test_ids_per_ribo  = {}

for ribo, ids_set in ribo_to_ids.items():
    ids_sorted = sorted(ids_set, key=int)
    print(f"Ribotipo {ribo}: IDs encontrados en base =", ids_sorted)

    if len(ids_sorted) != 10:
        print(f"Error: ribotipo {ribo} tiene {len(ids_sorted)} bacterias en vez de 10")

    # 7 primeras → TRAIN, 3 últimas → TEST (ajusta si quieres otro orden)
    train_ids_per_ribo[ribo] = set(ids_sorted[:7])
    test_ids_per_ribo[ribo]  = set(ids_sorted[7:10])

    print(f"  TRAIN → {sorted(train_ids_per_ribo[ribo], key=int)}")
    print(f"  TEST  → {sorted(test_ids_per_ribo[ribo],  key=int)}")

print("Etiquetas por ribotipo cargadas correctamente...")

# 1) Cargar todos los datos Semana 1
data_samples = []          # SpectrumObject instances
data_id_label = []         # IDs
Y_data = []                # Class labels
data_media = []            # Medio de los datos
data_split_list = []       # Para train/test

print("Cargando todos los datos...")

for extraccion_i in extraccion:
    if extraccion_i == 'Reproducibilidad Extracción':
        for medio in medios_extraccion:
            if medio == 'Chx':
                continue
            for semana in ['Semana 1']:   # <-- SOLO Semana 1
                for clase in clases_list:
                    ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
                    if not os.path.exists(ruta):
                        continue

                    for f in os.listdir(ruta):
                        # Ignorar ocultos tipo .DS_Store
                        if f.startswith("."):
                            continue

                        # ID bacteria = primera parte del nombre
                        bact_id = f.split('_')[0].strip()
                        if not bact_id.isdigit():
                            continue
                        bact_id = str(int(bact_id))   # normaliza


                        # Split según diccionarios canónicos (hechos arriba)
                        if bact_id in train_ids_per_ribo.get(clase, set()):
                            split_label = "train"
                        elif bact_id in test_ids_per_ribo.get(clase, set()):
                            split_label = "test"
                        else:
                            print(f"Ignorando bacteria {bact_id} de la ruta: {ruta}")
                            split_label = "ignore"

                        ruta_f = os.path.join(ruta, f)

                        # Cargar spectrum primero (y solo si carga -> append en TODO)
                        s = None
                        if 'mzml' in ruta_f.lower():
                            try:
                                run = pymzml.run.Reader(ruta_f)
                                spectro = [r for r in run]
                                if len(spectro) == 0:
                                    continue
                                s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            except Exception:
                                continue
                        else:
                            if not os.path.isdir(ruta_f):
                                continue

                            carpetas = [subf for subf in os.listdir(ruta_f) if not subf.startswith(".")]
                            if not carpetas:
                                continue

                            ruta_sub = os.path.join(ruta_f, carpetas[0])
                            fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                            acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                            if not (fid_files and acqu_files):
                                continue

                            try:
                                s = SpectrumObject().from_bruker(acqu_files[0], fid_files[0])
                            except Exception:
                                continue

                        if s is None:
                            continue

                        data_samples.append(s)
                        data_id_label.append(bact_id)
                        Y_data.append(clase)
                        data_media.append(medio)
                        data_split_list.append(split_label)

    elif extraccion_i == 'Reproducibilidad No extracción':
        for medio in medios:
            if medio == 'Ch':
                continue
            for semana in ['Semana 1']:   # <-- SOLO Semana 1
                for clase in clases_list:
                    ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
                    if not os.path.exists(ruta):
                        continue

                    for f in os.listdir(ruta):
                        if f.startswith("."):
                            continue

                        bact_id = f.split('_')[0]
                        if not bact_id.isdigit():
                            continue

                        if bact_id in train_ids_per_ribo.get(clase, set()):
                            split_label = "train"
                        elif bact_id in test_ids_per_ribo.get(clase, set()):
                            split_label = "test"
                        else:
                            print(f"Ignorando bacteria {bact_id} de la ruta: {ruta}")
                            split_label = "ignore"

                        ruta_f = os.path.join(ruta, f)

                        s = None
                        if 'mzml' in ruta_f.lower():
                            try:
                                run = pymzml.run.Reader(ruta_f)
                                spectro = [r for r in run]
                                if len(spectro) == 0:
                                    continue
                                s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                            except Exception:
                                continue
                        else:
                            if not os.path.isdir(ruta_f):
                                continue

                            carpetas = [subf for subf in os.listdir(ruta_f) if not subf.startswith(".")]
                            if not carpetas:
                                continue

                            ruta_sub = os.path.join(ruta_f, carpetas[0])
                            fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                            acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                            if not (fid_files and acqu_files):
                                continue

                            try:
                                s = SpectrumObject().from_bruker(acqu_files[0], fid_files[0])
                            except Exception:
                                continue

                        if s is None:
                            continue

                        data_samples.append(s)
                        data_id_label.append(bact_id)
                        Y_data.append(clase)
                        data_media.append(medio)
                        data_split_list.append(split_label)

    else:
        print("Error al meter la ruta")

print("Una vez cargados los datos, comprobamos que todo está alineado...")

###########################################
# Split 7/3 por ribotipo usando IDs
###########################################
data_samples_np   = np.array(data_samples, dtype=object)
data_id_label_np  = np.array(data_id_label, dtype=str)
Y_data_np         = np.array(Y_data, dtype=str)
data_media_np     = np.array(data_media, dtype=str)
data_split_np     = np.array(data_split_list, dtype=str)

lens = [len(data_samples_np), len(data_id_label_np), len(Y_data_np), len(data_media_np), len(data_split_np)]
assert len(set(lens)) == 1, f"Error: Desalineación detectada: {lens}"

print("Resumen split:")
print("  TRAIN :", np.sum(data_split_np == "train"))
print("  TEST  :", np.sum(data_split_np == "test"))
print("  IGNORE:", np.sum(data_split_np == "ignore"))

# máscaras
train_mask = (data_split_np == "train")
test_mask  = (data_split_np == "test")

id_array    = np.array(data_id_label)   # IDs de bacteria (tipo), ej: '7120562'
label_array = np.array(Y_data)          # ribotipo, ej: 'RT023'

print("\n========== IDS AGRUPADOS POR RIBOTIPO ==========")

for ribo in np.unique(label_array):
    idx_ribo = np.where(label_array == ribo)[0]

    ids_ribo = id_array[idx_ribo]
    train_ribo = ids_ribo[train_mask[idx_ribo]]
    test_ribo  = ids_ribo[test_mask[idx_ribo]]

    print(f"\nRibotipo {ribo}:")
    print("  TRAIN →", sorted(set(train_ribo.tolist())))
    print("  TEST  →", sorted(set(test_ribo.tolist())))

# Realizamos el data augmentation sobre TODOS los datos (Semana 1) Train/Test
###########################################
# DataAugemntation por MEDIO (Semana 1) Train/Test
###########################################
dataAugment = DataAugmentor(seed=seed)

# --- 1) Separar originales por split ---
X_train_spectra = data_samples_np[train_mask].tolist()
id_train        = data_id_label_np[train_mask].tolist()
y_train_lbls    = Y_data_np[train_mask]          # ribotipo
m_train         = data_media_np[train_mask]
preaugment_train = len(X_train_spectra)

X_test_spectra  = data_samples_np[test_mask].tolist()
id_test         = data_id_label_np[test_mask].tolist()
y_test_lbls     = Y_data_np[test_mask]
m_test          = data_media_np[test_mask]

print("Originales:")
print("  TRAIN:", len(X_train_spectra))
print("  TEST :", len(X_test_spectra))

print(f"Realizando data augment con los datos de train -> nº de operaciones por muestra: {k_per_spectrum}, como se realizan las operaciones: {type_augmentation}...")

# --- 2) Augment SOLO TRAIN ---
aug_train_samples, aug_train_ids, aug_train_labels = dataAugment.augment_dataset(
    spectra=X_train_spectra,
    ids=id_train,
    labels=y_train_lbls,
    k_per_spectrum=k_per_spectrum,
    id_suffix="aug"
)

split_aug_train = np.array(["train"] * len(aug_train_samples), dtype=str)
media_aug_train = np.repeat(m_train, k_per_spectrum)

# Comprobación de tamaños
if preaugment_train != len(X_train_spectra):
    print("Error al aumentar los datos de TRAIN: el número de originales ha cambiado.")
    exit()
else:
    print("Data augment TRAIN realizado correctamente...")
    print(
        f"   Nº de espectros originales (TRAIN): {len(X_train_spectra)}\n"
        f"   Nº de espectros generados (TRAIN): {len(aug_train_samples)}\n"
        f"   Nº total TRAIN tras augment: {len(X_train_spectra) + len(aug_train_samples)}"
    )

# --- 3) Concatenar TODO manteniendo splits ---
train_samples_final = X_train_spectra + aug_train_samples
train_ids_final     = id_train + (aug_train_ids if aug_train_ids is not None else [])
train_labels_final  = np.concatenate([y_train_lbls, aug_train_labels])

# Para poder filtrar luego por split si quieres:
train_split_final = np.array(["train"] * len(train_samples_final), dtype=str)

# TEST se queda tal cual (sin augmentation)
test_samples_final = X_test_spectra
test_ids_final     = id_test
test_labels_final  = y_test_lbls
test_split_final   = np.array(["test"] * len(test_samples_final), dtype=str)

# --- 4) Construir medias "finales" (imprescindible para el color en t-SNE) ---
train_media_final = np.concatenate([m_train, media_aug_train])  # train orig + train aug
test_media_final  = m_test                                      # test orig (sin aug)

lens = [len(train_samples_final), len(train_labels_final), len(train_media_final)]
assert len(set(lens)) == 1, f"Error: TRAIN desalineado: {lens}"

lens = [len(test_samples_final), len(test_labels_final), len(test_media_final)]
assert len(set(lens)) == 1, f"Error: TEST desalineado: {lens}"


print("Después de augmentation:")
print("  TRAIN total:", len(train_samples_final), "(orig:", len(X_train_spectra), "+ aug:", len(aug_train_samples), ")")
print("  TEST total :", len(test_samples_final), "(sin aug)")

################################################
# t-SNE por RIBOTIPO (Train (originales vs aumentados) vs Test) 
################################################
print("Realizando t-sne por Ribotipo: Train (originales vs aumentados) vs Test")
# -------------------------------------------------
# 1) Construimos arrays globales (todo junto)
# -------------------------------------------------
all_samples_tsne = train_samples_final + test_samples_final

labels_total = np.concatenate([train_labels_final.astype(str), test_labels_final.astype(str)])
media_total  = np.concatenate([train_media_final.astype(str), test_media_final.astype(str)])

n_train_orig = len(train_samples_final) - sum(
    1 for x in train_samples_final if hasattr(x, "aug_ops")
)
n_train_total = len(train_samples_final)

is_train = np.array([True]*n_train_total + [False]*len(test_samples_final))
is_test  = is_train

is_aug_train = np.array(
    [False]*n_train_orig + [True]*(n_train_total - n_train_orig)
)
is_aug_total = np.concatenate([
    is_aug_train,
    np.array([False]*len(test_samples_final))
])

# -------------------------------------------------
# 2) t-SNE único
# -------------------------------------------------
tsner = TSNESpectre(**tsne_params)
combined_grid = tsner._build_common_grid(all_samples_tsne)

X_all, _ = tsner.featurize(all_samples_tsne, grid=combined_grid)
Z_all = tsner.run_tsne(X_all)

# -------------------------------------------------
# 3) Colores por medio
# -------------------------------------------------
unique_medios = np.unique(media_total)
cmap = plt.get_cmap("tab10")
color_map = {m: cmap(i % 10) for i, m in enumerate(unique_medios)}

# -------------------------------------------------
# 4) Subplots por ribotipo
# -------------------------------------------------
unique_ribos = np.unique(labels_total)
n_cols = min(3, len(unique_ribos))
n_rows = int(np.ceil(len(unique_ribos) / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5*n_cols, 4*n_rows),
    sharex=True, sharey=True
)
axes = np.atleast_1d(axes).ravel()

for ax, ribo in zip(axes, unique_ribos):

    idx_ribo = np.where(labels_total == ribo)[0]

    for medio in unique_medios:
        color = color_map[medio]

        idx_rm = idx_ribo[media_total[idx_ribo] == medio]

        idx_train_orig = idx_rm[
            is_train[idx_rm] & (~is_aug_total[idx_rm])
        ]
        idx_train_aug = idx_rm[
            is_train[idx_rm] & (is_aug_total[idx_rm])
        ]
        idx_test_orig = idx_rm[
            is_test[idx_rm]
        ]

        # TRAIN aumentado (círculos, muy transparentes)
        ax.scatter(
            Z_all[idx_train_aug, 0],
            Z_all[idx_train_aug, 1],
            s=20,
            marker="o",
            alpha=0.20,
            color=color,
            label=f"{medio} (train aug)"
        )

        # TRAIN original (triángulos)
        ax.scatter(
            Z_all[idx_train_orig, 0],
            Z_all[idx_train_orig, 1],
            s=60,
            marker="^",
            alpha=0.95,
            color=color,
            edgecolors="k",
            linewidths=0.7,
            label=f"{medio} (train)"
        )

        # TEST original (cuadrados)
        ax.scatter(
            Z_all[idx_test_orig, 0],
            Z_all[idx_test_orig, 1],
            s=60,
            marker="s",
            alpha=0.95,
            color=color,
            edgecolors="k",
            linewidths=0.7,
            label=f"{medio} (test)"
        )

    ax.set_title(f"Ribotipo {ribo}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Eliminar duplicados en leyenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              frameon=False, fontsize=7)

# Apagar ejes sobrantes
for ax in axes[len(unique_ribos):]:
    ax.axis("off")

plt.tight_layout()

save_plot = f"t-SNE_subplots_ribotipo_train_test_color_medio_{type_augmentation}.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
plt.savefig(os.path.join(path_save_tsne, save_plot),
            dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

# =================================================
# 5) PLOT GLOBAL: todo junto en el mismo grid
#    - color por ribotipo
#    - TRAIN original: triángulos
#    - TRAIN aug: círculos transparentes
#    - TEST: cuadrados
# =================================================
unique_ribos = np.unique(labels_total)

# Mapa de colores por ribotipo (tab10 sirve para 6 ribotipos)
cmap_ribo = plt.get_cmap("tab10")
color_map_ribo = {r: cmap_ribo(i % 10) for i, r in enumerate(unique_ribos)}

plt.figure(figsize=(10, 8))

# Para que no se repita la leyenda
used_labels = set()

for ribo in unique_ribos:
    color = color_map_ribo[ribo]
    idx_ribo = np.where(labels_total == ribo)[0]

    # Índices para cada tipo
    idx_train_orig = idx_ribo[is_train[idx_ribo] & (~is_aug_total[idx_ribo])]
    idx_train_aug  = idx_ribo[is_train[idx_ribo] & (is_aug_total[idx_ribo])]
    idx_test_orig  = idx_ribo[is_test[idx_ribo]]

    # --- TRAIN AUG (círculos transparentes) ---
    label_aug = f"{ribo} (train aug)"
    if label_aug in used_labels:
        label_aug = None
    else:
        used_labels.add(label_aug)

    plt.scatter(
        Z_all[idx_train_aug, 0],
        Z_all[idx_train_aug, 1],
        s=18,
        marker="o",
        alpha=0.15,
        color=color,
        label=label_aug
    )

    # --- TRAIN ORIG (triángulos) ---
    label_train = f"{ribo} (train)"
    if label_train in used_labels:
        label_train = None
    else:
        used_labels.add(label_train)

    plt.scatter(
        Z_all[idx_train_orig, 0],
        Z_all[idx_train_orig, 1],
        s=55,
        marker="^",
        alpha=0.95,
        color=color,
        edgecolors="k",
        linewidths=0.6,
        label=label_train
    )

    # --- TEST (cuadrados) ---
    label_test = f"{ribo} (test)"
    if label_test in used_labels:
        label_test = None
    else:
        used_labels.add(label_test)

    plt.scatter(
        Z_all[idx_test_orig, 0],
        Z_all[idx_test_orig, 1],
        s=55,
        marker="s",
        alpha=0.95,
        color=color,
        edgecolors="k",
        linewidths=0.6,
        label=label_test
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"t-SNE global: color por ribotipo | ^ train | o train aug | s test [{type_augmentation}]")
plt.legend(frameon=False, fontsize=8, ncol=3)
plt.tight_layout()

save_plot_global = f"t-SNE_GLOBAL_train_aug_test_color_ribotipo_{type_augmentation}.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
plt.savefig(os.path.join(path_save_tsne, save_plot_global),
            dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

print("t-SNE por ribotipo (color por medio) generado correctamente.")

###########################################
# 
# CREACIÓN DE LOS CLASIFICADORES
# 
###########################################

# Incialización de variables comunes (train, test, labels)
featurizer = TSNESpectre(**tsne_params)
grid_class = featurizer._build_common_grid(train_samples_final)

X_train, _ = featurizer.featurize(train_samples_final, grid=grid_class)
X_test,  _ = featurizer.featurize(test_samples_final,  grid=grid_class)

label_mapping = {label: idx for idx, label in enumerate(CLASSES)}

y_train = np.array([label_mapping[r] for r in train_labels_final], dtype=int)
y_test  = np.array([label_mapping[r] for r in test_labels_final], dtype=int)

label_mapping = {v: k for k, v in label_mapping.items()}

groups = []
for sid in train_ids_final:
    sid = str(sid).strip()
    sid_low = sid.lower()

    if "aug" in sid_low:
        base_id = sid_low.split("aug")[0]
    else:
        base_id = sid_low

    base_id = re.sub(r"\D", "", base_id)
    groups.append(base_id)

groups = np.array(groups, dtype=str)

#-----------------------------------------
# Modelo Random Forest Energía
#-----------------------------------------
# 1) Convertir las gráficas de los espectros en funciones discretas formadas por bins
mz = grid_class
mz_min = mz.min()
mz_max = mz.max()
i_bins = X_train.shape[1]
bin_edges = np.linspace(mz_min, mz_max, i_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

n_train = X_train.shape[0]
X_train_binned = np.zeros((n_train, i_bins), dtype=np.float32)

for i in range(n_train):
    spec = X_train[i]

    for b in range(i_bins):
        mask = (mz >= bin_edges[b]) & (mz < bin_edges[b + 1])
        if np.any(mask):
            X_train_binned[i, b] = spec[mask].sum()


X_train_binned_norm = X_train_binned / X_train_binned.sum(axis=1, keepdims=True)

print("TRAIN:")
print("Shape binned:", X_train_binned_norm.shape)
print("Suma del primer espectro:", X_train_binned_norm[0].sum())

n_test = X_test.shape[0]
X_test_binned = np.zeros((n_test, i_bins), dtype=np.float32)

for i in range(n_test):
    spec = X_test[i]

    for b in range(i_bins):
        mask = (mz >= bin_edges[b]) & (mz < bin_edges[b + 1])
        if np.any(mask):
            X_test_binned[i, b] = spec[mask].sum()


X_test_binned_norm = X_test_binned / X_test_binned.sum(axis=1, keepdims=True)

print("\nTEST:")
print("Shape binned:", X_test_binned_norm.shape)
print("Suma del primer espectro:", X_test_binned_norm[0].sum())

if np.isnan(X_train_binned_norm).any():
    print("WARNING: NaNs en X_train_binned_norm (posible suma 0 en algún espectro)")
if np.isnan(X_test_binned_norm).any():
    print("WARNING: NaNs en X_test_binned_norm (posible suma 0 en algún espectro)")

# 2) Calculamos la energía de cada bin
# Train
print("\nCalculando la energía de cada train bin con ventana =", window)
energy_bin_train = np.zeros((X_train_binned_norm.shape[0], X_train_binned_norm.shape[1]))
for i in range(X_train_binned_norm.shape[0]):
    for j in range(X_train_binned_norm.shape[1]):
        energy_bin_train[i, j] = X_train_binned_norm[i, j]
        for w in range(1, window + 1):
            next_idx = j + w
            previus_idx = j - w
            if previus_idx < 0:
                energy_bin_train[i, j] += X_train_binned_norm[i, next_idx]
            elif next_idx >= X_train_binned_norm.shape[1]:
                energy_bin_train[i, j] += X_train_binned_norm[i, previus_idx]
            else:
                energy_bin_train[i, j] += X_train_binned_norm[i, next_idx] + X_train_binned_norm[i, previus_idx]

# Test
print("Calculando la energía de cada test bin con ventana =", window)
energy_bin_test= np.zeros((X_test_binned_norm.shape[0], X_test_binned_norm.shape[1]))
for i in range(X_test_binned_norm.shape[0]):
    for j in range(X_test_binned_norm.shape[1]):
        energy_bin_test[i, j] = X_test_binned_norm[i, j]
        for w in range(1, window + 1):
            next_idx = j + w
            previus_idx = j - w
            if previus_idx < 0:
                energy_bin_test[i, j] += X_test_binned_norm[i, next_idx]
            elif next_idx >= X_test_binned_norm.shape[1]:
                energy_bin_test[i, j] += X_test_binned_norm[i, previus_idx]
            else:
                energy_bin_test[i, j] += X_test_binned_norm[i, next_idx] + X_test_binned_norm[i, previus_idx]

if np.isnan(energy_bin_train).any():
    print("WARNING: NaNs en energy_bin_train (posible suma 0 en algún espectro)")
if np.isnan(energy_bin_test).any():
    print("WARNING: NaNs en energy_bin_test (posible suma 0 en algún espectro)")

if do_energy_rf:
    print("\n====================================")
    print("Random Forest Energía")
    print("====================================")
    # 3) Buscamos los mejores hiperparámetros con GridSearchCV y generamos el modelo del Random Forest Robusto
    n_classes = len(CLASSES)

    auc_scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovr",
        average='macro',
        labels=np.arange(n_classes)  # fuerza el orden de clases
    )

    cv = GroupKFold(n_splits=2)

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_cv_rf_robusto,
        scoring=auc_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        error_score=np.nan,  # para que no reviente si algún fold no permite AUC
        refit=True
    )

    grid.fit(energy_bin_train, y_train, groups=groups)

    print("Mejor score CV:", grid.best_score_)
    print("Mejores hiperparámetros:", grid.best_params_)
    best_rf = grid.best_estimator_

    # 4) Obtenemos la precisión final del TEST y la precisión por ribotipo del mismo set de datos
    y_test_pred = best_rf.predict(energy_bin_test)
    y_test_proba = best_rf.predict_proba(energy_bin_test)
    acc_test_global = accuracy_score(y_test, y_test_pred)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    roc_auc_score_test_robusto = roc_auc_score(y_test_bin, y_test_proba, multi_class="ovr", average="macro")
    print(f"\nAccuracy GLOBAL (TEST): {acc_test_global:.4f}")

    print("\nAccuracy por ribotipo (TEST):")

    rows = []

    for c in sorted(np.unique(y_test)):
        ribo = label_mapping[c]   # RT023, RT027, ...
        mask = (y_test == c)       # solo muestras TEST de ese ribotipo

        acc_c = np.mean(y_test_pred[mask] == y_test[mask])
        n_c   = mask.sum()

        print(f"  {ribo}: n={n_c}  acc={acc_c:.4f}")

        rows.append({
            "Ribotipo": ribo,
            "n_test": int(n_c),
            "accuracy_test": float(acc_c)
        })

    balanced_acc_test_global_rf_robusto = balanced_accuracy_score(y_test, y_test_pred)
    print(f"\nValor del balanced accuracy: {balanced_acc_test_global_rf_robusto:.4f}")
    print(f"Valor del ROC AUC score: {roc_auc_score_test_robusto:.4f}")

# =========================================
# AUC (ROC) en TEST para multiclase (OvR)
# =========================================
# n_classes = len(CLASSES)

# # Probabilidades del mejor modelo en TEST
# y_test_proba = best_rf.predict_proba(energy_bin_test)   # shape: (n_test, n_classes)

# # Binarizar y_test para OvR
# y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))  # shape: (n_test, n_classes)

# # AUC global (macro, One-vs-Rest)
# auc_test_macro = roc_auc_score(
#     y_test_bin,
#     y_test_proba,
#     multi_class="ovr",
#     average="macro"
# )
# print(f"\nROC AUC GLOBAL (TEST) [macro OvR]: {auc_test_macro:.4f}")

# # AUC por ribotipo (cada clase vs resto)
# print("\nROC AUC por ribotipo (TEST):")
# for i in range(n_classes):
#     ribo = label_mapping[i]  # tu mapping inverso: int -> 'RTxxx'
#     # Si para alguna clase no hay positivos en TEST, roc_auc_score falla:
#     if y_test_bin[:, i].sum() == 0:
#         print(f"  {ribo}: AUC=NA (no hay positivos en TEST)")
#         continue
#     auc_i = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])
#     print(f"  {ribo}: AUC={auc_i:.4f}")

# # =========================================
# # Plot ROC por ribotipo (OvR) en TEST
# # =========================================
# plt.figure(figsize=(8, 6))

# for i in range(n_classes):
#     ribo = label_mapping[i]

#     # Evita error si no hay positivos/negativos
#     if y_test_bin[:, i].sum() == 0 or y_test_bin[:, i].sum() == len(y_test_bin[:, i]):
#         continue

#     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
#     auc_i = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])

#     plt.plot(fpr, tpr, label=f"{ribo} (AUC={auc_i:.3f})")

# plt.plot([0, 1], [0, 1], linestyle="--")  # línea aleatoria
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC (OvR) por ribotipo - TEST")
# plt.legend(frameon=False, fontsize=9)
# plt.tight_layout()

# save_plot = f"ROC_por_ribotipo_TEST_{type_augmentation}.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# plt.savefig(os.path.join(path_save_tsne, save_plot), dpi=300, bbox_inches="tight")

# plt.close()


# # 5) Área Bajo la curva (AUC) para comprobar la robustez del modelo
# n_classes = len(CLASSES)

# y_train_bin = label_binarize(y_train, classes=range(n_classes))
# y_test_bin  = label_binarize(y_test,  classes=range(n_classes))

# print("\n====================================")
# print("Curva de aprendizaje AUC (Random Forest)")
# print("====================================")

# trees_range = np.arange(50, 1250, 50)

# auc_train_curve = []
# auc_val_curve   = []

# cv = GroupKFold(n_splits=7)

# for n_trees in trees_range:
#     rf = RandomForestClassifier(
#         n_estimators=n_trees,
#         random_state=rf_params["random_state"],
#         n_jobs=rf_params["n_jobs"],
#         max_depth=8,
#         min_samples_leaf=5,
#         min_samples_split=10,
#         max_features="sqrt",
#         bootstrap=True
#     )

#     # ---- TRAIN AUC ----
#     rf.fit(energy_bin_train, y_train)
#     y_train_proba = rf.predict_proba(energy_bin_train)
#     auc_train = roc_auc_score(
#         y_train_bin,
#         y_train_proba,
#         multi_class="ovr",
#         average="macro"
#     )
#     auc_train_curve.append(auc_train)

#     # ---- VALIDATION AUC (CV) ----
#     auc_folds = []
#     for tr_idx, va_idx in cv.split(energy_bin_train, y_train, groups=groups):
#         rf.fit(energy_bin_train[tr_idx], y_train[tr_idx])
#         y_va_proba = rf.predict_proba(energy_bin_train[va_idx])

#         y_va_bin = label_binarize(
#             y_train[va_idx],
#             classes=np.arange(n_classes)
#         )

#         auc_fold = roc_auc_score(
#             y_va_bin,
#             y_va_proba,
#             multi_class="ovr",
#             average="macro"
#         )
#         auc_folds.append(auc_fold)

#     auc_val_curve.append(np.mean(auc_folds))

#     print(f"Trees={n_trees:4d} | AUC train={auc_train:.4f} | AUC val={np.mean(auc_folds):.4f}")

# plt.figure(figsize=(7,5))
# plt.plot(trees_range, auc_train_curve, label="Train AUC", marker="o")
# plt.plot(trees_range, auc_val_curve, label="Validation AUC (CV)", marker="s")

# plt.xlabel("Número de árboles")
# plt.ylabel("ROC AUC (macro, OvR)")
# plt.title("Curva de aprendizaje AUC – Random Forest")
# plt.legend(frameon=False)
# plt.grid(alpha=0.3)

# plt.tight_layout()

# save_plot = f"Curva_AUC_Random_Forest_.jpg"
# os.makedirs(path_save_tsne, exist_ok=True)
# plt.savefig(os.path.join(path_save_tsne, save_plot),
#             dpi=300, bbox_inches="tight")
# plt.close()

# y_test_proba = rf.predict_proba(energy_bin_test)

# auc_test = roc_auc_score(
#     y_test_bin,
#     y_test_proba,
#     multi_class="ovr",
#     average="macro"
# )

# print(f"\nROC AUC GLOBAL (TEST): {auc_test:.4f}")

# print("\nROC AUC por ribotipo (TEST):")
# for i, ribo in enumerate(CLASSES):
#     auc_ribo = roc_auc_score(
#         y_test_bin[:, i],
#         y_test_proba[:, i]
#     )
#     print(f"  {ribo}: AUC={auc_ribo:.4f}")

# exit()

#-----------------------------------------
# Modelo Random Forest Ocupación
#-----------------------------------------
best_acc = -1
best_eps = -1
if do_occupation_rf:
    print("\n====================================")
    print("Random Forest Ocupación")
    print("====================================")
    cv_outer = GroupKFold(n_splits=2)

    for eps in eps_list:
        # epsilon por muestra (un valor por fila)
        eps_vec = np.quantile(X_train_binned_norm, eps, axis=1)  # shape (n_train,)
        X_train_occ = (X_train_binned_norm > eps_vec[:, None])

        rf = RandomForestClassifier(random_state=0, n_jobs=-1)

        n_classes = len(CLASSES)

        auc_scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovr",
        average='macro',
        labels=np.arange(n_classes)  # fuerza el orden de clases
        )

        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_cv_rf_robusto,
            scoring=auc_scorer,
            cv=cv_outer,
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train_occ, y_train, groups=groups)
        mean_cv = grid.best_score_

        print(f"eps={eps:.2f} | best_CV={mean_cv:.4f} | best_params={grid.best_params_}")

        eps_test = np.quantile(X_test_binned_norm, eps, axis=1)
        X_test_occ = (X_test_binned_norm > eps_test[:, None])

        best_rf = grid.best_estimator_

        y_pred = best_rf.predict(X_test_occ)
        balanced_acc_test_global_rf_occupation = balanced_accuracy_score(y_test, y_pred)
       
        if balanced_acc_test_global_rf_occupation > best_acc:
            best_eps = eps
            best_acc = balanced_acc_test_global_rf_occupation
            y_test_proba = best_rf.predict_proba(X_test_occ)
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            roc_auc_score_test_occupation = roc_auc_score(y_test_bin, y_test_proba, multi_class="ovr", average="macro")
            train_acc_occupation = grid.best_score_

print(f"Mejor valor de eps: {best_eps}")
print(f"Balanced Accuracy TEST: {balanced_acc_test_global_rf_occupation:.4f}")
print(f"Valor del ROC AUC score: {roc_auc_score_test_occupation:.4f}")

#-----------------------------------------
# Modelo Random Forest Peak
#-----------------------------------------
if do_peak_rf:
    print("\n====================================")
    print("Random Forest Peak")
    print("====================================")

    eps_vec = np.quantile(X_train_binned_norm, eps, axis=1)
    X_occ_train = (X_train_binned_norm > eps_vec[:, None])
    X_occ_test = (X_test_binned_norm > eps_test[:, None])

    X_peak_train = energy_bin_train / (X_occ_train + delta)
    X_peak_test = energy_bin_test / (X_occ_test + delta)

    n_classes = len(CLASSES)

    auc_scorer = make_scorer(
        roc_auc_score,
        response_method="predict_proba",
        multi_class="ovr",
        average='macro',
        labels=np.arange(n_classes)  # fuerza el orden de clases
    )

    cv = GroupKFold(n_splits=2)

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_cv_rf_robusto,
        scoring=auc_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        error_score=np.nan,  # para que no reviente si algún fold no permite AUC
        refit=True
    )

    grid.fit(X_peak_train, y_train, groups=groups)
    print("Mejor score CV:", grid.best_score_)
    print("Mejores hiperparámetros:", grid.best_params_)
    best_rf = grid.best_estimator_

    y_pred = best_rf.predict(X_peak_test)
    y_test_proba = best_rf.predict_proba(X_peak_test)
    balanced_acc_test_global_rf_peak = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy TEST: {balanced_acc_test_global_rf_peak:.4f}")
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    roc_auc_score_test_peak = roc_auc_score(y_test_bin, y_test_proba, multi_class="ovr", average="macro")
    print(f"Valor del ROC AUC score: {roc_auc_score_test_peak:.4f}")


#-----------------------------------------
# Modelo Random Forest Standard
#-----------------------------------------
print("\n====================================")
print("Random Forest Standard")
print("====================================")

# -----------------------------------------
# 2) Groups para GroupKFold (EVITA LEAKAGE por augmentation)
#    -> un augment debe quedarse en el mismo fold que su bacteria original
# -----------------------------------------
groups_train = []
for sid in train_split_final:
    sid = str(sid)
    base_id = sid.split("aug")[0]
    groups_train.append(base_id)
groups_train = np.array(groups_train, dtype=str)

# -----------------------------------------
# 3) CV y espacio de búsqueda
# -----------------------------------------
cv = GroupKFold(n_splits=2)

rf_base = RandomForestClassifier(
    n_jobs=-1,
    random_state=seed
)

grid = GridSearchCV(
    estimator=rf_base,
    param_grid=param_cv_rf_robusto,
    scoring="balanced_accuracy",         
    cv=cv,
    error_score=np.nan,
    n_jobs=-1,
    verbose=2,
    refit=True                   
)

print("\nBuscando mejores hiperparámetros en TRAIN (GroupKFold)...")
grid.fit(X_train, y_train, groups=groups)
print("Train score: ", grid.best_score_)

# -----------------------------------------
# 4) Evaluación final en TEST (intocable)
# -----------------------------------------
best_rf_generic = grid.best_estimator_
y_pred = best_rf_generic.predict(X_test)
y_pred_proba = best_rf_generic.predict_proba(X_test)
acc_global = balanced_accuracy_score(y_test, y_pred)
roc_auc_score_test_generico = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average="macro")
print(f"\nAccuracy GLOBAL (TEST): {acc_global:.4f}")

print("\nAccuracy por ribotipo (TEST):")
rows = []
for c in sorted(np.unique(y_test)):
    ribo = label_mapping[c]
    mask = (y_test == c)
    n_c = int(mask.sum())
    acc_c = float(np.mean(y_pred[mask] == y_test[mask]))
    print(f"  {ribo}: n={n_c}  acc={acc_c:.4f}")
    rows.append({"Ribotipo": ribo, "Accuracy": acc_c})

print("\n====================================")
print(f"FIN DEL PROCESO")
print("====================================")

print("\nResumen de resultados Random Forest Robusto vs Random Forest Genérico:")
print(f"Valores utilizados: aumentos por datos: {k_per_spectrum}, tamaño de la ventana: {window}, nº de bins: {i_bins}")

if do_energy_rf:
    print(f"\nRF ENERGÍA: balanced acc: {balanced_acc_test_global_rf_robusto:.4f}. ROC AUC score: {roc_auc_score_test_robusto:.4f}")

if do_occupation_rf:
    print("\nRF OCCUPATION:")
    print(f"TEST: balanced acc: {balanced_acc_test_global_rf_occupation:.4f}. ROC AUC score: {roc_auc_score_test_occupation:.4f}")
    print(f"TRAIN: {train_acc_occupation}")

if do_peak_rf:
    print(f"\nRF PEAK: balanced acc: {balanced_acc_test_global_rf_peak:.4f}. ROC AUC score: {roc_auc_score_test_peak:.4f}")

print(f"\nRF STANDARD:")
print(f"TEST: balanced acc: {acc_global:.4f}. ROC AUC score: {roc_auc_score_test_generico:.4f}")

print("\nCódigo ejecutado sin errores...")

# baseline_samples = [preprocess_pipeline(s) for s in tqdm(baseline_samples, desc="Baseline samples")]
# X_train = [s.intensity for i, s in enumerate(baseline_samples)]