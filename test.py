import pandas as pd

# Datos de ejemplo
data = {'Columna1': [1, 2, 3], 'Columna2': [4, 5, 6]}
df = pd.DataFrame(data)

# Guardar en Excel
df.to_excel('datos.xlsx', index=False, sheet_name='Hoja1')
