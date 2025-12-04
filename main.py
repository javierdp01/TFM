###########################################
# Imports
###########################################
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from dataloader.SpectrumObject import SpectrumObject
import pymzml
from data_augementator import DataAugmentor
from tsne import TSNESpectre
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


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
k_per_spectrum = 36             # nº de espectros aumentados por espectro
seed = 0                        # reprodicibilidad

# T-SNE
show_tsne_plot = False                                  # True/False -> mostrar figuras
path_save_tsne = r'C:\Users\javie\Desktop\ImagenesTFM'  # Dirección donde se guardan las figuras del t-sne
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
medios_extraccion = ['Brx', 'Chx', 'Chx_24h', 'Clx', 'Fax', 'Scx']
#-----------------------------------------------------------------------------------------------------------------------------------------------

###########################################
# Data Loading
###########################################
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
print("Realizando Random Forest...")

# Utilizaremos los datos ya calculados en t-sne y adaptaré las Y para que esté más organizando
y_orig = np.array(Y_train)
y_aug  = np.array(aug_labels)
y_all  = np.array(all_labels)

# 1) RF solo datos originales
rf_orig = RandomForestClassifier(**rf_params)
rf_orig.fit(X_orig, y_orig)
train_acc_orig = rf_orig.score(X_orig, y_orig)  # accuracy en train

# 2) RF solo datos aumentados
rf_aug = RandomForestClassifier(**rf_params)
rf_aug.fit(X_aug, y_aug)
train_acc_aug = rf_aug.score(X_aug, y_aug)

# 3) RF con datos originales + aumentados
rf_all = RandomForestClassifier(**rf_params)
rf_all.fit(X_all, y_all)
train_acc_all = rf_all.score(X_all, y_all)

print("Todos los Random Forest se han realizado correctamente...")

# 4) Comprobamos los resultados
print("RESULTADOS RANDOM FOREST:")
print("[RF SOLO ORIGINALES]: %.3f" % train_acc_orig)
print("[RF SOLO AUMENTADOS]: %.3f" % train_acc_aug)
print("[RF ORIGINALES + AUMENTADOS]: %.3f" % train_acc_all)

###########################################
# Todos los datos (solo Semana 1)
###########################################
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
                            data_id_label.append(f.split('_')[0])
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
                            data_id_label.append(f.split('_')[0])
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


print("Código ejecutado sin errores...")
