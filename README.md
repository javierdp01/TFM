# TFM
Trabajo Fin de Máster

# Data Augmentation:
- Se han generado unos getters y unos setters para obtener y establecer los datos de forma más rápida y sencilla de comprender
- Se han generado distintas funciones que se encargan de realizar las diferentess operaciones utilizadas durante el aumento de los datos. Estas operaciones son:
    * intensity_scale: se encarga de multiplicar todo el espectro por un factor aleatorio, con una probabilidad p de que suceda. Esto permite simular variaciones de intensidad debido a fluctuaciones, a la sesibilidad del detector e incluso a diferencias en la ganancia del instrumento.
    * mz_shift: se trata de un desplazamiento en el eje x un valor aleatorio para posteriormente reinterpolar el espectro en la malla original para así mantener el mismo tamaño, permitiendo simular errores de calibración.
    * gaussian_noise: se encarga de añadir ruido blanco al espectro gracias a calcular la potencia media de la señal para así simular el ruido eléctrico del detector.
    * baseline_poly: genera una función polinómica aleatoria y la suma al espectro para así simular problemas de corrección de línea base en mediciones realies.
    * peak_broadening: aplica una convolución gaussiana de anchura aleatoria para ensanchar los picos, simulando diferencias en resolución intrumental durante la adquicisión del espectro.
    * random_dropout: se encarga de poner a 0 ciertos picos para simular pérdidas de información de la señal.
    * spikes: añade picos en posiciones aleatorios para simular descargas, interferencias o contaminación producida por los intrumentos al obtener el espectro de la señal.

- Se ha generado un "main" de la clase encargado de realizar las distintas operaciones ya mencionadas al espectro que se le pasa y devuelve 3 variables:
    * aug_specs: son los nuevos datos generados con las operaciones
    * aug_id: son los ids correspondientes a los nuevos datos
    * aug_labels: son las etiquetas correspondientes a los nuevos datos
  
   Como estos datos no incluyen los obtenidos en la medición, se deberan concatenar los antiguos resultados con los nuevos para así completar el data augmentation.

- Para comprobar si el data augmentation mantiene la posición original con cierta desviación se utilizó la herramienta t-sne para ver las distancias entre las distintas muestras y una vez comprobado que se mantienen se pasará a cargar todos los datos.

# TODOS LOS DATOS:
- Se ha partido por cargar los datos de la semana 1 y se ha hecho lo mismo que antes a diferencia de que se ha realizado el t-sne para comprobar las distancias separándolas por ribotipo para así tener una idea más clara de la situación en la que nos encontramos.
- Tras evaluar la situación, se ha decidido dividir las 10 muestras de cada medio y ribotipo en 2 partes, 7 muestras para realizar el entrenamiento y 3 muestras para realizar la validación. Las muestras del entrenamiento serán las únicas que se aumentarán.
- Como primer clasificador se ha utilizado un RF en el que se realizará una CrossValidation para evaluar hiperparámetros y así obtener la mejor métrica de precisión posible
