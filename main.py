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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import randint

#-----------------------------------------------------------------------------------------------------------------------------------------------
###########################################
# Opciones
###########################################
# Global constants and configuration:
CLASSES = ['RT023', 'RT027', 'RT078', 'RT106', 'RT165', 'RT181']

# Define the dataset structure parameters.
semanas = ['Semana 1', 'Semana 2', 'Semana 3']
clases_list = CLASSES  # same order for iteration
medios = ['Ch', 'Br', 'Cl', 'Sc'] # Quitados por ahora: Ch (posicion 0)

# Use a particular condition for training. For example, here training samples are selected when:
#   medio == 'Medio Ch' and semana == 'Semana 1'
training_media  = 'Ch'
training_week   = 'Semana 1'
n_biomarkers = 10

# Base path for the data (adjust as needed)
base_path = 'C:/Users/javie/Desktop/TFM/DATA/ClostriRepro/ClostriRepro/Reproducibilidad No extracción'

# DATA AUGMENTATION
type_augmentation = 'linear'    # Opciones: "random" y "linear"
k_per_spectrum = 36             # nº de espectros aumentados por espectro
seed = 0                        # reprodicibilidad

# T-SNE
show_tsne_plot = False                                  # True/False -> mostrar figuras
path_save_tsne = r"C:\Users\javie\Desktop\ImagenesTFM"  # Dirección donde se guardan las figuras del t-sne
tsne_params = dict(                                     # Hiperparámetros
    n_points=4096,
    tic_norm=True,
    do_log1p=True,
    pca_dims=50,
    perplexity=30,  # la clase lo ajusta automáticamente si hay pocas muestras
    seed=0,
    max_iter=1500
)

# RANDOM FOREST
rf_params = dict(               # Hiperparámetros
    n_estimators = 300,
    random_state = 0,
    n_jobs = -1
)

# DATOS GLOBALES
data_path = 'C:/Users/javie/Desktop/TFM/DATA/ClostriRepro/ClostriRepro'
extraccion = ['Reproducibilidad Extracción', 'Reproducibilidad No extracción']
medios_extraccion = ['Brx', 'Chx', 'Chx_24h', 'Clx', 'Fax', 'Scx'] # Quitados por ahora: Chx (posicion 1)
results_rf_path = 'C:/Users/javie/Desktop/TFM/results'  # Dóonde se guardaran los datos de los resultados
n_ejecuciones_rf = 1
#-----------------------------------------------------------------------------------------------------------------------------------------------

###########################################
# Data Loading
###########################################
print("\n====================================")
print("PRUEBAS CON DATOS DE UN MEDIO")
print("====================================")
baseline_samples = []          # SpectrumObject instances (training samples)
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
                            if not os.path.isdir(ruta_f):
                                continue
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

###########################################
# Preprocessing
###########################################
dataAugment = DataAugmentor(seed=seed)
preaugment = len(baseline_samples)

print(f"Realizando data augment con los datos -> nº de operaciones por muestra: {k_per_spectrum}, como se realizan las operaciones: {type_augmentation}...")

aug_samples, aug_ids, aug_labels = dataAugment.augment_dataset(
    spectra=baseline_samples,
    ids=baseline_id_label,
    labels=Y_train,
    k_per_spectrum=k_per_spectrum,
    id_suffix="aug"
)
aug_operation_label = ["+".join(getattr(s, "aug_ops", ["none"]))
                       for s in aug_samples]

# Concatenamos todo
all_samples = baseline_samples + aug_samples
all_ids = baseline_id_label + (aug_ids if aug_ids is not None else [])
all_labels = np.concatenate([Y_train, aug_labels]) if aug_labels is not None else Y_train

combined_aug_ops_labels = (["originales"] * len(baseline_samples) + aug_operation_label)

if (preaugment != len(baseline_samples)):
    print("Error al aumentar los datos.")
    exit()
else:
    print("Data augment realizado correctamente...")
    print(f"Nº de espectros originales: {len(baseline_samples)}. Nº de espectros generados: {len(aug_samples)}. Nº total de espectros en el training data: {int(len(baseline_samples)) + int(len(aug_samples))}")

###########################################
# t-SNE
###########################################
print("Realizando t-sne...")

tsner = TSNESpectre(**tsne_params)  # Pasamos los hiperparámetros

print("t-sne realizado correctamente...")

print("Generando figuras del t-sne...")

# 1) Construir una plantilla común
combined_grid = tsner._build_common_grid(all_samples)  # Utilizamos la misma plantilla para las 3 situaciones

# 2) t-SNE SOLO ORIGINALES
X_orig, _ = tsner.featurize(baseline_samples, grid=combined_grid)
Z_orig = tsner.run_tsne(X_orig)
tsner.plot(Z_orig, labels=Y_train, class_names=CLASSES, title="t-SNE originales", show=show_tsne_plot, type=type_augmentation)

# 3) t-SNE SOLO AUMENTADOS
X_aug, _ = tsner.featurize(aug_samples, grid=combined_grid)
Z_aug = tsner.run_tsne(X_aug)
tsner.plot(Z_aug, labels=aug_labels, class_names=CLASSES, title="t-SNE aumentados", show=show_tsne_plot, type=type_augmentation)

# 4) t-SNE COMBINADO (base + aug)
X_all, _ = tsner.featurize(all_samples, grid=combined_grid)
Z_all = tsner.run_tsne(X_all)

n_orig = len(baseline_samples)
combined_aug_ops_labels = np.array(combined_aug_ops_labels)

plt.figure(figsize=(8,6))

# 1) Aumentados (circulos, por operación)
unique_ops = np.unique(aug_operation_label)
for op in unique_ops:
    # índices en la parte aumentada
    idx_aug_part = np.where(combined_aug_ops_labels == op)[0]
    idx_aug_part = idx_aug_part[idx_aug_part >= n_orig]  # solo indices >= n_orig
    plt.scatter(
        Z_all[idx_aug_part, 0],
        Z_all[idx_aug_part, 1],
        s=25,
        alpha=0.6,
        label=f"aug: {op}"
    )

# 2) Originales (triángulos negros, por clase)
Y_train_arr = np.array(Y_train)
for c in np.unique(Y_train_arr):
    idx_orig_class = np.where(Y_train_arr == c)[0]
    name = CLASSES[c] if c < len(CLASSES) else str(c)
    plt.scatter(
        Z_all[idx_orig_class, 0],
        Z_all[idx_orig_class, 1],
        s=70,
        marker="^",
        alpha=0.95,
        edgecolors="k",
        linewidths=0.8,
        label=f"{name} (orig)"
    )

plt.legend(frameon=False, fontsize=9, ncol=2)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"t-SNE: originales (▲) vs aumentados (●) {type_augmentation}")
plt.tight_layout()

save_plot_as = f"t-SNE combinados originales+aumentados_{type_augmentation}_comprobar_operación.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
path_to_save = os.path.join(path_save_tsne, save_plot_as)
print("Guardando la figura generada en " + path_to_save)
plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

X_all, _ = tsner.featurize(all_samples, grid=combined_grid)
Z_all = tsner.run_tsne(X_all)

# indices
n_orig = len(baseline_samples)
idx_orig = np.arange(n_orig)
idx_aug = np.arange(n_orig, len(Z_all))

plt.figure(figsize=(7,6))
for c in np.unique(all_labels):
    # índices para cada clase
    idx_orig_class = idx_orig[Y_train == c]
    idx_aug_class = idx_aug[aug_labels == c]
    name = CLASSES[c] if c < len(CLASSES) else str(c)
    
    # Originales: triángulos
    plt.scatter(Z_all[idx_orig_class, 0],
                Z_all[idx_orig_class, 1],
                s=70, marker="^", alpha=0.95, label=f"{name} (orig)", zorder=3, edgecolors="k")
    # Aumentados: círculos
    plt.scatter(Z_all[idx_aug_class, 0],
                Z_all[idx_aug_class, 1],
                s=25, marker="o", alpha=0.3, label=f"{name} (aug)", zorder=1)

plt.legend(frameon=False, fontsize=9, ncol=2)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"t-SNE: originales (▲) vs aumentados (●) {type_augmentation}")
plt.tight_layout()

save_plot_as = f"t-SNE combinados originales+aumentados_{type_augmentation}_no_comprobar_operación.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
path_to_save = os.path.join(path_save_tsne, save_plot_as)
print("Guardando la figura generada en " + path_to_save)
plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

print("Figuras generadas y guardadas correctamente...")

###########################################
# Random Forest - Training
###########################################
# print("Realizando Random Forest...")

# # Utilizaremos los datos ya calculados en t-sne y adaptaré las Y para que esté más organizando
# y_orig = np.array(Y_train)
# y_aug  = np.array(aug_labels)
# y_all  = np.array(all_labels)

# # 1) RF solo datos originales
# rf_orig = RandomForestClassifier(**rf_params)
# rf_orig.fit(X_orig, y_orig)
# train_acc_orig = rf_orig.score(X_orig, y_orig)  # accuracy en train

# # 2) RF solo datos aumentados
# rf_aug = RandomForestClassifier(**rf_params)
# rf_aug.fit(X_aug, y_aug)
# train_acc_aug = rf_aug.score(X_aug, y_aug)

# # 3) RF con datos originales + aumentados
# rf_all = RandomForestClassifier(**rf_params)
# rf_all.fit(X_all, y_all)
# train_acc_all = rf_all.score(X_all, y_all)

# print("Todos los Random Forest se han realizado correctamente...")

# # 4) Comprobamos los resultados
# print("RESULTADOS RANDOM FOREST:")
# print("[RF SOLO ORIGINALES]: %.3f" % train_acc_orig)
# print("[RF SOLO AUMENTADOS]: %.3f" % train_acc_aug)
# print("[RF ORIGINALES + AUMENTADOS]: %.3f" % train_acc_all)

###########################################
#
# TODOS LOS DATOS SEMANA 1
#
###########################################
print("\n====================================")
print("TODOS LOS DATOS SEMANA 1")
print("====================================")
# 1) Cargar todos los datos Semana 1
data_samples = []          # SpectrumObject instances
data_id_label = []         # IDs
Y_data = []                # Class labels
data_media = []            # Medio de los datos
print("Cargando todos los datos...")

for extraccion_i in extraccion:
    if extraccion_i == 'Reproducibilidad Extracción':
        for medio in medios_extraccion:
            for semana in ['Semana 1']:   # <-- SOLO Semana 1
                for clase in clases_list:
                    ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
                    if os.path.exists(ruta):
                        for f in os.listdir(ruta):
                            ruta_f = os.path.join(ruta, f)
                            bact_id = f.split('_')[0].strip()
                            if not bact_id.isdigit():
                                continue
                            bact_id = str(int(bact_id))
                            data_id_label.append(bact_id)
                            if 'mzml' in ruta_f:
                                run = pymzml.run.Reader(ruta_f)
                                spectro = [r for r in run]
                                s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                                data_samples.append(s)
                                Y_data.append(clase)
                                data_media.append(medio) 
                            else:
                                if not os.path.isdir(ruta_f):
                                    continue
                                carpetas = [subf for subf in os.listdir(ruta_f)]
                                if carpetas:
                                    ruta_sub = os.path.join(ruta_f, carpetas[0])
                                    fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                    acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                    if fid_files and acqu_files:
                                        ruta_fid = fid_files[0]
                                        ruta_acqu = acqu_files[0]
                                        s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                        data_samples.append(s)
                                        Y_data.append(clase)
                                        data_media.append(medio) 
                    else:
                        continue
    elif extraccion_i == 'Reproducibilidad No extracción':
        for medio in medios:
            for semana in ['Semana 1']:   # <-- SOLO Semana 1
                for clase in clases_list:
                    ruta = f"{data_path}/{extraccion_i}/{medio}/{semana}/{clase}"
                    if os.path.exists(ruta):
                        for f in os.listdir(ruta):
                            ruta_f = os.path.join(ruta, f)
                            bact_id = f.split('_')[0].strip()
                            if not bact_id.isdigit():
                                continue
                            bact_id = str(int(bact_id))
                            data_id_label.append(bact_id)
                            if 'mzml' in ruta_f:
                                run = pymzml.run.Reader(ruta_f)
                                spectro = [r for r in run]
                                s = SpectrumObject(mz=spectro[0].mz, intensity=spectro[0].i)
                                data_samples.append(s)
                                Y_data.append(clase)
                                data_media.append(medio) 
                            else:
                                if not os.path.isdir(ruta_f):
                                    continue
                                carpetas = [subf for subf in os.listdir(ruta_f)]
                                if carpetas:
                                    ruta_sub = os.path.join(ruta_f, carpetas[0])
                                    fid_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'fid'))
                                    acqu_files = glob(os.path.join(ruta_sub, '*', '1SLin', 'acqu'))
                                    if fid_files and acqu_files:
                                        ruta_fid = fid_files[0]
                                        ruta_acqu = acqu_files[0]
                                        s = SpectrumObject().from_bruker(ruta_acqu, ruta_fid)
                                        data_samples.append(s)
                                        Y_data.append(clase)
                                        data_media.append(medio) 
                    else:
                        continue
    else:
        print("Error al meter la ruta")

# Realizamos el data augmentation sobre TODOS los datos (Semana 1)
dataAugment = DataAugmentor(seed=seed)
preaugment = len(data_samples)

print(f"Realizando data augment con los datos -> nº de operaciones por muestra: {k_per_spectrum}, como se realizan las operaciones: {type_augmentation}...")

aug_samples_total, aug_ids_total, aug_labels_total = dataAugment.augment_dataset(
    spectra=data_samples,
    ids=data_id_label,
    labels=Y_data,
    k_per_spectrum=k_per_spectrum,
    id_suffix="aug"
)
aug_operation_label = ["+".join(getattr(s, "aug_ops", ["none"]))
                       for s in aug_samples_total]

# Concatenamos todo
all_samples_total = data_samples + aug_samples_total
all_ids_total = data_id_label + (aug_ids_total if aug_ids_total is not None else [])
all_labels_total = np.concatenate([Y_data, aug_labels_total]) if aug_labels_total is not None else Y_data

combined_aug_ops_labels_total = (["originales"] * len(data_samples) + aug_operation_label)

if (preaugment != len(data_samples)):
    print("Error al aumentar los datos.")
    exit()
else:
    print("Data augment realizado correctamente...")
    print(f"Nº de espectros originales: {len(data_samples)}. Nº de espectros generados: {len(aug_samples_total)}. Nº total de espectros: {int(len(data_samples)) + int(len(aug_samples_total))}")

###########################################
# t-SNE por MEDIO (originales vs aumentados)
###########################################
print("Realizando t-sne...")

tsner_total = TSNESpectre(**tsne_params)

print("t-sne realizado correctamente...")

print("Generando figuras del t-sne...")

# 1) Construimos una grid común con todos los espectros
combined_grid_total = tsner_total._build_common_grid(all_samples_total)

# 2) Featurizamos originales, aumentados y todos
X_orig_total, _ = tsner_total.featurize(data_samples,       grid=combined_grid_total)
X_aug_total,  _ = tsner_total.featurize(aug_samples_total,  grid=combined_grid_total)
X_all_total,  _ = tsner_total.featurize(all_samples_total,  grid=combined_grid_total)

# 3) Preparamos etiquetas de medio también para los aumentados
data_media = np.array(data_media)

media_aug_total = []
for m in data_media:
    media_aug_total.extend([m] * k_per_spectrum)
media_aug_total = np.array(media_aug_total)

all_media_total = np.concatenate([data_media, media_aug_total])

# 4) Ejecutamos t-SNE sobre TODOS
Z_all_total = tsner_total.run_tsne(X_all_total)

# 5) Índices de originales y aumentados
n_orig_total = len(data_samples)
idx_orig_total = np.arange(n_orig_total)
idx_aug_total  = np.arange(n_orig_total, len(all_samples_total))

# 6) Mapa de colores por medio
medios_data = np.unique(all_media_total)
cmap = plt.get_cmap("tab10")
color_map = {m: cmap(i % 10) for i, m in enumerate(medios_data)}

plt.figure(figsize=(10, 8))

for medio in medios_data:
    color = color_map[medio]

    # índices de este medio en todo el dataset
    idx_all_medio = np.where(all_media_total == medio)[0]

    # separamos originales y aumentados de ese medio
    idx_orig_medio = idx_all_medio[idx_all_medio < n_orig_total]
    idx_aug_medio  = idx_all_medio[idx_all_medio >= n_orig_total]

    # Aumentados: círculos, más pequeños y translúcidos
    plt.scatter(
        Z_all_total[idx_aug_medio, 0],
        Z_all_total[idx_aug_medio, 1],
        s=25,
        marker="o",
        alpha=0.3,
        color=color,
        label=f"{medio} (aug)"
    )

    # Originales: triángulos grandes con borde negro con mayor contraste
    plt.scatter(
        Z_all_total[idx_orig_medio, 0],
        Z_all_total[idx_orig_medio, 1],
        s=70,
        marker="^",
        alpha=0.95,
        color=color,
        edgecolors="k",
        linewidths=0.8,
        label=f"{medio} (orig)"
    )

plt.legend(frameon=False, fontsize=8, ncol=2)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"t-SNE por MEDIO (Semana 1): originales (▲) vs aumentados (●) [{type_augmentation}]")
plt.tight_layout()

# Guardar figura
save_plot_as = f"t-SNE_todos_por_medio_Semana1_{type_augmentation}.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
path_to_save = os.path.join(path_save_tsne, save_plot_as)
print("Guardando la figura generada en " + path_to_save)
plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

print("Figura t-SNE por medio (Semana 1) generada y guardada correctamente...")

################################################
# t-SNE por RIBOTIPO (originales vs aumentados)
################################################
# Ribotipos presentes en estos datos (6 posibles)
unique_ribos = np.unique(all_labels_total)

n_classes = len(unique_ribos)
n_cols = min(3, n_classes)              # hasta 3 columnas
n_rows = int(np.ceil(n_classes / n_cols))

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4 * n_rows),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()      # por si solo hay 1 fila

for ax, ribo in zip(axes, unique_ribos):
    # índices de este ribotipo en todo el dataset (originales + aumentados)
    idx_ribo_all = np.where(all_labels_total == ribo)[0]

    # medios que aparecen para este ribotipo
    medios_ribo = np.unique(all_media_total[idx_ribo_all])

    # para evitar repetir leyendas dentro del mismo subplot
    used_labels = set()

    for medio in medios_ribo:
        color = color_map.get(medio, "C0")

        # índices de este ribotipo + este medio
        mask_medio = (all_media_total[idx_ribo_all] == medio)
        idx_ribo_medio = idx_ribo_all[mask_medio]

        # separamos originales y aumentados
        idx_orig_medio = idx_ribo_medio[idx_ribo_medio < n_orig_total]
        idx_aug_medio  = idx_ribo_medio[idx_ribo_medio >= n_orig_total]

        # label para la leyenda
        label_aug  = f"{medio} (aug)"
        label_orig = f"{medio} (orig)"

        if label_aug in used_labels:
            label_aug = None
        else:
            used_labels.add(label_aug)

        if label_orig in used_labels:
            label_orig = None
        else:
            used_labels.add(label_orig)

        # Aumentados: círculos, translúcidos
        ax.scatter(
            Z_all_total[idx_aug_medio, 0],
            Z_all_total[idx_aug_medio, 1],
            s=25,
            marker="o",
            alpha=0.3,
            color=color,
            label=label_aug
        )

        # Originales: triángulos con borde negro, más visibles
        ax.scatter(
            Z_all_total[idx_orig_medio, 0],
            Z_all_total[idx_orig_medio, 1],
            s=70,
            marker="^",
            alpha=0.95,
            color=color,
            edgecolors="k",
            linewidths=0.8,
            label=label_orig
        )

    ax.set_title(f"Ribotipo: {ribo}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=7)

# Si hay más subplots creados que ribotipos (p.ej. 6 huecos pero 5 clases), los apagamos
for ax in axes[len(unique_ribos):]:
    ax.axis("off")

plt.tight_layout()

save_plot_as = f"t-SNE_subplots_ribotipo_Semana1_{type_augmentation}.jpg"
os.makedirs(path_save_tsne, exist_ok=True)
path_to_save = os.path.join(path_save_tsne, save_plot_as)
print("Guardando la figura de subplots por ribotipo en " + path_to_save)
plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

if show_tsne_plot:
    plt.show()
plt.close()

print("Figura t-SNE por ribotipo y medio (subplots) generada y guardada correctamente...")


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

                        # ✅ Append alineado (todo a la vez)
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
is_test  = ~is_train

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

print("t-SNE por ribotipo (color por medio) generado correctamente.")


###########################################
# Random Forest (predicción de RIBOTIPO)
# - Entrena con TRAIN (orig + aug)
# - Test SIN augmentation
# - Accuracy global + accuracy por ribotipo (solo en TEST)
###########################################
print("\n====================================")
print("Random Forest: predicción de RIBOTIPO")
print("====================================")

# 1) Inicialización
inv_label_mapping = {v: k for k, v in label_mapping.items()}

featurizer = TSNESpectre(**tsne_params)
grid_rf = featurizer._build_common_grid(train_samples_final)

X_train_rf, _ = featurizer.featurize(train_samples_final, grid=grid_rf)
X_test_rf,  _ = featurizer.featurize(test_samples_final,  grid=grid_rf)

# 2) Etiquetas (Ribotipo)
y_train_rf = np.array([label_mapping[r] for r in train_labels_final], dtype=int)
y_test_rf  = np.array([label_mapping[r] for r in test_labels_final], dtype=int)

# 3) Entrenar RF
n_test_por_ribotipo = {}
acc_por_ribotipo = {inv_label_mapping[c]: [] for c in np.unique(y_test_rf)}

for i in range(n_ejecuciones_rf):
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_rf, y_train_rf)

    # 4) Predicción en TEST
    y_pred = rf.predict(X_test_rf)

    # 5) Accuracy global (TEST)
    acc_global = accuracy_score(y_test_rf, y_pred)
    print(f"\nAccuracy GLOBAL (TEST): {acc_global:.4f}")

    # 6) Accuracy por ribotipo (TEST)
    print("\nAccuracy por ribotipo (TEST):")
    for c in sorted(np.unique(y_test_rf)):
        ribo = inv_label_mapping[c]
        mask = (y_test_rf == c)
        acc_c = float(np.mean(y_pred[mask] == y_test_rf[mask]))
        n_c = int(mask.sum())

        acc_por_ribotipo[ribo].append(acc_c)
        n_test_por_ribotipo[ribo] = n_c
        print(f"  {ribo}: n={n_c}  acc={acc_c:.4f}")

rows = []

print("\nAccuracy MEDIA por ribotipo (TEST):")
for ribo, acc_list in acc_por_ribotipo.items():
    acc_mean = float(np.mean(acc_list))
    n_c = n_test_por_ribotipo[ribo]

    print(f"  {ribo}: n={n_c}  acc_mean={acc_mean:.4f}")

    rows.append({
        "Ribotipo": ribo,
        "Accuracy": acc_mean
    })

# 7) Guardar CSV con resultados por ribotipo
df_xlsx = pd.DataFrame(rows)

os.makedirs(path_save_tsne, exist_ok=True)
out_xlsx = os.path.join(
    path_save_tsne,
    f"rf_accuracy_por_ribotipo_MEDIA_{type_augmentation}.xlsx"
)

df_xlsx.to_excel(out_xlsx, index=False)

print("\nXLSX guardado en:", out_xlsx)

# # -----------------------------------------
# # 1) Featurize (misma grid TRAIN->TEST)
# # -----------------------------------------
# inv_label_mapping = {v: k for k, v in label_mapping.items()}
# featurizer = TSNESpectre(**tsne_params)
# grid_rf = featurizer._build_common_grid(train_samples_final)

# X_train_rf, _ = featurizer.featurize(train_samples_final, grid=grid_rf)
# X_test_rf,  _ = featurizer.featurize(test_samples_final,  grid=grid_rf)

# y_train_rf = np.array([label_mapping[r] for r in train_labels_final], dtype=int)
# y_test_rf  = np.array([label_mapping[r] for r in test_labels_final], dtype=int)

# # -----------------------------------------
# # 2) Groups para GroupKFold (EVITA LEAKAGE por augmentation)
# #    -> un augment debe quedarse en el mismo fold que su bacteria original
# # -----------------------------------------
# groups_train = []
# for sid in train_ids_final:
#     sid = str(sid)
#     # Si tus aug_ids llevan sufijos tipo "12345_aug_7", esto lo normaliza a "12345"
#     base_id = sid.split("_")[0]
#     groups_train.append(base_id)
# groups_train = np.array(groups_train, dtype=str)

# # -----------------------------------------
# # 3) CV y espacio de búsqueda
# # -----------------------------------------
# cv = GroupKFold(n_splits=5)

# rf_base = RandomForestClassifier(
#     n_jobs=-1,
#     random_state=0
# )

# # Distributions (ajusta rangos si quieres)
# param_distributions = {
#     "n_estimators": randint(300, 1501),        # 300..1500
#     "max_depth": [None] + list(range(5, 61, 5)),
#     "max_features": ["sqrt", "log2", None],
#     "min_samples_split": randint(2, 21),       # 2..20
#     "min_samples_leaf": randint(1, 11),        # 1..10
#     "bootstrap": [True, False],
#     "class_weight": [None, "balanced"]         # útil si hay desbalance
# }

# # OJO: n_iter controla cuántas combinaciones pruebas
# n_iter_search = 20

# search = RandomizedSearchCV(
#     estimator=rf_base,
#     param_distributions=param_distributions,
#     n_iter=n_iter_search,
#     scoring="accuracy",          # precisión global
#     cv=cv,
#     n_jobs=-1,
#     random_state=0,
#     verbose=2,
#     refit=True                   # al final re-entrena con best_params en TODO el TRAIN
# )

# print("\nBuscando mejores hiperparámetros en TRAIN (GroupKFold)...")
# search.fit(X_train_rf, y_train_rf, groups=groups_train)

# print("\nMejores hiperparámetros encontrados:")
# print(search.best_params_)
# print(f"Mejor accuracy CV (TRAIN): {search.best_score_:.4f}")

# best_rf = search.best_estimator_

# # -----------------------------------------
# # 4) Evaluación final en TEST (intocable)
# # -----------------------------------------
# y_pred = best_rf.predict(X_test_rf)

# acc_global = accuracy_score(y_test_rf, y_pred)
# print(f"\nAccuracy GLOBAL (TEST): {acc_global:.4f}")

# print("\nAccuracy por ribotipo (TEST):")
# rows = []
# for c in sorted(np.unique(y_test_rf)):
#     ribo = inv_label_mapping[c]
#     mask = (y_test_rf == c)
#     n_c = int(mask.sum())
#     acc_c = float(np.mean(y_pred[mask] == y_test_rf[mask]))
#     print(f"  {ribo}: n={n_c}  acc={acc_c:.4f}")
#     rows.append({"Ribotipo": ribo, "Accuracy": acc_c})

# # Guardar resultados a XLSX como pediste: 1a col ribotipo, 2a col accuracy
# df_xlsx = pd.DataFrame(rows)[["Ribotipo", "Accuracy"]]
# out_xlsx = os.path.join(path_save_tsne, f"rf_bestparams_accuracy_por_ribotipo_{type_augmentation}.xlsx")
# df_xlsx.to_excel(out_xlsx, index=False)
# print("\nXLSX guardado en:", out_xlsx)

print("Código ejecutado sin errores...")