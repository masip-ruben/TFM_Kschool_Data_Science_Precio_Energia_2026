import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
from win32com.client import DispatchEx, constants  # pip install pywin32
#_______________________________________________________________________________________________________________________________
#_________________________FUNCIONES ESPECÍFICAS PARA GENERAR DATAFRAMES a partir DE ARCHIVOS DESCARGADOS_________________________
#________________________________________________________________________________________________


#_____________________________________________________________________________________
# 04. MUNICIPIOS
#_____________________________________________________________________________________

def cargar_df_poblacion(ruta):
    df_municipios = pd.read_csv(ruta, sep=";", encoding="latin1")

    df_municipios['LONGITUD_ETRS89'] = df_municipios['LONGITUD_ETRS89'].str.replace(",", ".").astype(float)
    df_municipios['LATITUD_ETRS89'] = df_municipios['LATITUD_ETRS89'].str.replace(",", ".").astype(float)

        
    df_municipios['ALTITUD'] = pd.to_numeric(df_municipios['ALTITUD'],errors="coerce").astype("Int64")


    # Eliminar las islas Canarias, Baleares, Ceuta y Melilla
    df_municipios = df_municipios[~df_municipios['COD_PROV'].isin([7,35,38,51,52])]
    
    return df_municipios



 # 04.01. FUNCIÓN CREAR Mallado de coordenadas 
def crear_grid_poblacion(df, nx=10, ny=10):
    long = df["LONGITUD_ETRS89"].values
    lat = df["LATITUD_ETRS89"].values
    pob = df["POBLACION_MUNI"].values

    # Calcular histogram 2D ponderado por población
    H, xedges, yedges = np.histogram2d(long, lat, bins=[nx, ny], weights=pob)

    # Representar con imshow (heatmap)
    plt.figure(figsize=(8,6))
    plt.imshow(
        H.T, origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="turbo"
    )
    plt.colorbar(label="Población total en celda")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Población agregada en celdas de la grid")
    plt.show()
    
    # Crear lista de resultados
    rows = []
    for i in range(nx):
        for j in range(ny):
            if H[i, j] > 0:  # solo celdas con población
                mid_long = (xedges[i] + xedges[i+1]) / 2
                mid_lat = (yedges[j] + yedges[j+1]) / 2
                rows.append([mid_long, mid_lat, H[i, j]])

    # Convertir en DataFrame
    grid_df = pd.DataFrame(rows, columns=["LONG_centro", "LAT_centro", "POB_suma"])

    return grid_df, xedges, yedges
    





#_____________________________________________________________________________________
# 02. DEMANDA REAL: FUNCIÓN ESPECÍFICA PARA ARCHIVO descargado por Alejandro de DEMNADA REAL
#_____________________________________________________________________________________
# FUNCIÓN B1_1: Generar DataFrame, limpiar y guardar CSV
def generate_df_demanda_ale(directorio, nombre_archivo, extension):
    ruta = directorio / f"{nombre_archivo}{extension}"
    df = pd.read_excel (ruta)
    title_1="Demanda Real_Ale"
             
    # Eliminamos columnas que no dan datos
    df.drop(columns=['id', 'name','geoid','geoname'], inplace=True)
    
    # Damos nuevo nombre a cabeceras y Exploramos
    df.columns = ["Demanda MWh", "Date"]
    
    # Crear nuevas columnas
    df['Year'] = df['Date'].str[0:4].astype(int)
    df['Month'] = df['Date'].str[5:7].astype(int)
    df['Day'] = df['Date'].str[8:10].astype(int)
    df['Time'] = df['Date'].str[11:13].astype(int)
  
      
    # Guardar CSV limpio (en ambas carpetas)
    df.to_csv(nombre_archivo + ".csv", index=False)  
    df.to_csv(f"{directorio}/{nombre_archivo}.csv", index=False)
    
    return df 

#_____________________________________________________________________________________
# 01.PRECIOS MARGINALES: FUNCIÓN ESPECÍFICA PARA ARCHIVO descargado desde OMIE
#_____________________________________________________________________________________

# FUNCIÓN B01_2: Generar DataFrame total desde los parciales y guardar CSV

def generate_df_prices_total(directorio, nombre_archivos, extension):
    #"""
    # Genara 
    # df: DataFrame con columnas ['Year', 'Month', 'Day', 'Time', 'Price_1']
    #A partir de txt de cada año no separados por comas y con líneas no váidas
    #Ejemplo de uso:
    
    #directorio = Path(r"C:\Users\Elena\OneDrive\Desktop\Máster Datos\_TFM\Datos\02. Demanda y Precios\PRECIOS\omie_es_es_file_HORARIOS")
    #extension=".txt"
    #nombres_archivos = ["marginal_2019_TOTAL", "marginal_2020_TOTAL", "marginal_2021_TOTAL", "marginal_2022_TOTAL", "marginal_2023_TOTAL"]
    #df_total = generate_df_prices(directorio, nombres_archivos, extension)
    #"""
    df_total = pd.DataFrame() # Inicializar df_total
    for archivo in nombre_archivos:
        df_total = pd.concat([df_total, generate_df_prices_partial(directorio, archivo, extension)], ignore_index=True)    
        
    df_total.drop(columns=['Price_2'], inplace=True)  # Eliminar columna Price_2
    
    # Guardar CSV limpio (en ambas carpetas)
    # df_total.to_csv("marginal_TOTAL.csv", index=False)

    df_total.to_csv(f"{directorio}/Precios_TOTAL.csv", index=False, sep=";")

    return df_total

# FUNCIÓN B01_1: Generar DataFrame parcial desde TXT, limpiar y guardar CSV
def generate_df_prices_partial(directorio, nombre_archivo, extension):
    
    data = []
    with open(f"{directorio}/{nombre_archivo}{extension}") as f:
        for line in f:
            cols = line.strip().split(";")     # separar por ";"
            cols = [c if c != "" else "0" for c in cols]  # convertir vacíos en 0
            data.append(cols)

    # Convertir a DataFrame rellenando con 0 donde falte
    df = pd.DataFrame(data).fillna(0)

    # Guardar RAW como CSV (en ambas carpetas) 
    # df.to_csv(nombre_archivo + "_raw.csv", index=False)
    df.to_csv(f"{directorio}/{nombre_archivo}_raw.csv", index=False, sep=";")

    # Damos nombre a cabeceras y Exploramos
    df.columns = ["Year", "Month", "Day", "Time", "Price_1", "Price_2", "NULL"]
    
    
    # Convertir Year a numérico, forzando errores a NaN
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.dropna(subset=["Year"], inplace=True)
    df.drop(columns=["NULL"], inplace=True)  # Eliminar columna NULL
    
    # Convertir a int
    df["Year"] = df["Year"].astype(int)
    df = df.astype({"Year": int, "Month": int, "Day": int, "Time": int, "Price_1": float, "Price_2": float})
    
    # explora_df(df,title_1=df.iloc[1,0])    
    
    # Guardar CSV limpio (en ambas carpetas)
    # df.to_csv(nombre_archivo + ".csv", index=False)  
    # df.to_csv(f"{directorio}/{nombre_archivo}.csv", index=False)
    return df 





#_____________________________________________________________________________________
# 03.BALANCE Eléctrico: FUNCIÓN ESPECÍFICA PARA ARCHIVO descargado desde OMIE - Datos Diarios
#_____________________________________________________________________________________

# FUNCIÓN B03_2: Generar DataFrame total desde los parciales y guardar CSV

def generate_df_balance_total (directorio, nombres_archivos, extension):
    
    df_balance_total = pd.DataFrame() # Inicializar df_total
    for archivo in nombres_archivos:
        df_balance_total = pd.concat([df_balance_total, generate_df_balance_partial(directorio, archivo, extension)], ignore_index=True)


    df_balance_total['DayOfYear'] = pd.to_datetime({'year': df_balance_total['Year'], 'month': df_balance_total['Month'], 'day': df_balance_total['Day']}).dt.dayofyear
        
      
      # Imputamos en aquellos casos donde sea Null la demanda
    df_balance_total.loc[df_balance_total['Demanda'].isna(), 'Demanda'] = (
    df_balance_total.loc[df_balance_total['Demanda'].isna(), ['Generacion renovable', 'Generacion no renovable', 'Saldo almacenamiento', 'Enlace Peninsula-Baleares','Saldo I. internacionales']].sum(axis=1)
)
      
    # Crear el nuevo DataFrame con las columnas que te interesan
    df_balance_red = df_balance_total[[
        'Year', 'Month', 'Day', 'DayOfYear',
        'Demanda', 'Eolica', 
        'Solar fotovoltaica', 'Solar termica'
    ]].copy()  # copia para evitar advertencias de asignación

    # Crear la nueva columna 'Solar' como suma de las dos columnas solares
    df_balance_red['Solar'] = (
        df_balance_red['Solar fotovoltaica'] + df_balance_red['Solar termica']
    )
    # Si ya no quieres mantener las dos columnas originales:
    df_balance_red = df_balance_red.drop(columns=['Solar fotovoltaica', 'Solar termica'])
        
    # Guardar CSV limpio 
    #df_balance_total.to_csv("balance_TOTAL.csv", index=False)  
    # df_balance_total.to_csv(f"{directorio}/balance_TOTAL.csv", index=False, float_format="%.6f")
    
    df_balance_total.to_csv(
    f"{directorio}/balance_TOTAL.csv",
    sep=";",          # separador de columnas
    index=False,
    encoding="utf-8",
    decimal=",",      # usa coma como separador decimal en el archivo
    float_format="%.6f"  # controla cuántos decimales (ajusta a tu necesidad)
)

    df_balance_red.to_csv(
        f"{directorio}/balance_REDUCIDO.csv",
        sep=";",          # separador de columnas
        index=False,
        encoding="utf-8",
        decimal=",",      # usa coma como separador decimal en el archivo
        float_format="%.6f"  # controla cuántos decimales (ajusta a tu necesidad)
    )
    
    return df_balance_total, df_balance_red

# FUNCIÓN B03_1: Generar DataFrame parcial desde xlsx
    
def generate_df_balance_partial(directorio, nombre_archivo, extension): 
    ruta = directorio / f"{nombre_archivo}{extension}"
    df = pd.read_excel(ruta)
    #df = pd.read_excel(ruta, converters=lambda x: str(x).replace(",", "."))
    ultima_columna = df.columns[(df.notna().any())][-1]
        # Leer el rango específico de celdas (A1:ABD33)
        # Ajustar según la estructura del archivo
        # La primera fila (A1) es la cabecera, las filas 2 a 4 son metadatos y se saltan
        # Se leen 29 filas (de la 5 a la 33 inclusive) y las columnas desde A hasta ABD
        # La primera columna (A) se usa como índice
        # df = pd.read_excel(ruta, header=0, skiprows=4, nrows=29, usecols="A:ABD", index_col=0)
        
        # Leer el rango específico de celdas (A1:ABD33) usando usecols con range
        # Esto es útil si no se conoce la letra de la última columna o si puede variar
        # Aquí se asume que la última columna con datos es ABD (columna 56, índice 55)
        # Ajustar ultima_columna si es necesario
    ultima_columna_index = df.columns.get_loc(ultima_columna) + 1  # +1 porque range es exclusivo en el extremo superior
        
        # Leer el archivo Excel con los parámetros especificados    
    df = pd.read_excel(
        ruta,
        header=0,            # La primera fila del rango será la cabecera
        skiprows=4,          # Salta las primeras 4 filas (A1:A4)
        nrows=29,            # Lee 29 filas (de la 5 a la 33 inclusive)
        usecols=range(0, ultima_columna_index),     # Columnas desde A hasta ABD
        index_col=0          # La primera columna como índice
    )

   
    df=df.transpose().reset_index()
    # tRANSPONER EL DATAFRAME

    # Convertir a float todas las columnas menos la primera
    df.replace("-", 0, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype('float64')
        
        # Crear nuevas columnas
    df['Year'] = df['index'].str[7:9].astype(int)+2000
        
    meses_dict = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    df['Month'] = df['index'].str[3:6].map(meses_dict)
    df['Day'] = df['index'].str[0:2].astype(int)
    
    df = df.rename(columns={'Demanda en b.c.': 'Demanda'})

    # Eliminamos las dos columnas que no aportan
    df.drop(columns=['index'], inplace=True)
    # df.drop(df.columns[-4], axis=1, inplace=True)

    # Mover las tres últimas columnas al principio
    cols = list(df.columns)
    new_order = cols[-3:] + [cols[-4], cols[-5]] + cols[:-5]
    df = df[new_order]
    
    # Normalizar nombres de columnas para eliminar acentos
    df.columns = [unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("utf-8") for c in df.columns]
    return df 





#_____________________________________________________________________________________
# 06.Potencia instalada del PRETOR
#_____________________________________________________________________________________


def convertir_xls_a_xlsx(carpeta_1: str) -> list[Path]:
    """
    Abre cada .xls con Excel (COM) y lo guarda como .xlsx en la misma carpeta.
    Requiere Windows + Excel instalado: pip install pywin32
    Devuelve la lista de rutas .xlsx generadas.
    """

    carpeta = Path(carpeta_1)
    xls_files = sorted(f for f in carpeta.glob("*.xls") if f.is_file())
    if not xls_files:
        return []

    excel = DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    generados = []
    try:
        for f in xls_files:
            destino = f.with_suffix(".xlsx")
            wb = excel.Workbooks.Open(str(f))
            # 51 = xlOpenXMLWorkbook (xlsx, sin macros)
            wb.SaveAs(str(destino), FileFormat=51)
            wb.Close(SaveChanges=False)
            generados.append(destino)
    finally:
        excel.Quit()
    return generados



def concat_excels(carpeta_1: str, carpeta_2: str) -> pd.DataFrame:
    carpeta = Path(carpeta_1)

    # 1) Convertir .xls -> .xlsx (si hay)
    try:
        convertir_xls_a_xlsx(carpeta)
    except Exception as e:
        print(f"⚠️ No se pudo convertir con Excel COM: {e}")

    # 2) Leer todos los Excel modernos
    archivos = sorted([f for f in carpeta.glob("*") if f.suffix.lower() in (".xlsx", ".xlsm")])
    if not archivos:
        raise FileNotFoundError("No hay .xlsx/.xlsm en la carpeta (ni conversión exitosa).")

    frames = []
    for f in archivos:
        df = pd.read_excel(f, engine="openpyxl")
        if not df.empty:
            df = df.copy()
            df["__archivo__"] = f.name
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    
    # Normalizar nombres de columnas para eliminar acentos
    out.columns = [unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("utf-8") for c in out.columns]
    cols_a_limpiar = out.columns[1:4]

    # Quitar acentos y caracteres especiales solo en esas columnas
    for col in cols_a_limpiar:
        out[col] = out[col].apply(
            lambda x: ''.join(
                ch for ch in unicodedata.normalize('NFKD', str(x))
                if not unicodedata.combining(ch)
            ) if isinstance(x, str) else x
        )
        
    out = out.rename(columns={'Nombre de Instalacion': 'Nombre',
                              'Municipio de la Instalacion': 'Municipio',
                              'Provincia de la Instalacion': 'Provincia',
                              'Potencia Instalada KW': 'Potencia',
                              'Grupo Normativo': 'Grupo',
                              'Fecha de Inscripcion Definitiva': 'Fecha'})
    
    out.to_excel(f"{carpeta_2}/PRETOR_TOTAL.xlsx", index=False, engine="openpyxl")
    # Crear el nuevo DataFrame con las columnas que te interesan
    out_red = out[[
        'Municipio', 'Potencia', 'Grupo', 'Fecha',
        'Provincia'
    ]].copy()  # copia para evitar advertencias de asignación

    
    

    # Aplicar al DataFrame
   
    grupo_map = {
    "b.1": 'Solar',
    "b.1.1": 'Solar',
    "b.1.2": 'Solar',
    "b.2": 'Eolico',
    "b.2.1": 'Eolico',
    "b.2.2": 'Eolico'
    }

    out_red["Grupo"] = out_red["Grupo"].map(grupo_map)

    
    
    # --- 2) Filtrar por Grupo y fecha válida ---
    # fecha no nula, no vacía, y distinta de "n/D" (cualquier capitalización)
    fecha_str = out_red['Fecha'].astype(str).str.strip()
    mask_fecha_valida = out_red['Fecha'].notna() & (fecha_str != '') & (fecha_str.str.lower() != 'n/d')

    mask_grupo_valido = out_red['Grupo'].isin(['Solar', 'Eolico'])

    out_red = out_red.loc[mask_grupo_valido & mask_fecha_valida].copy()

    # (Opcional) Parsear la fecha a datetime y descartar las que no se puedan parsear
    out_red['Fecha'] = pd.to_datetime(out_red['Fecha'], errors='coerce', dayfirst=True)
    out_red = out_red[out_red['Fecha'].notna()]


    out_red.to_excel(f"{carpeta_2}/PRETOR_REDUCIDO.xlsx", index=False, engine="openpyxl")
    out_red.to_csv(f"{carpeta_2}/PRETOR_REDUCIDO.csv", index=False)
    
    return out





 #_________________________________________________________________________________________________
 #_________________________FUNCIÓN transforma datos HORARIOS en DIARIOS_________________________   
 #________________________________________________________________________________________________


def generate_df_variable_diaria(df,variable):
    """
    Esta función toma un DataFrame con datos horarios y genera un DataFrame con datos diarios.
    Agrupa los datos por fecha y guarda las columnas horarias como variables, calcula la suma, la media, el máximo y el mínimo para cada día.

    Parámetros:
    df_variable_horaria (pd.DataFrame): DataFrame con datos horarios. Debe tener una columna 'Year', 'Month', 'Day', 'Time' y 'Variable'.
    Retorna:
    pd.DataFrame: DataFrame con datos diarios que incluye los datos horarios, la suma, la media, el máximo y el mínimo para cada día.
    """
    
    # Asegurarse de que la columna 'Timer' es de tipo entero
    df['Time'] = df['Time'].astype(int)    
    df['Time_label'] = df['Time'].apply(lambda x: f"Time_{x:02d}")
    
    # Pivotar: una fila por día, columnas = horas
    df_diario = df.pivot_table(
        index=["Year", "Month", "Day"],
        columns="Time_label",
        values=variable
    ).reset_index()

    

    # Agrupar por fecha y calcular la media, el máximo y el mínimo
    # Añadir columnas estadísticas por fila
    df_diario[f"{variable}_D"] = df_diario.loc[:, df_diario.columns[3:]].sum(axis=1)   # suma de todas las horas
    df_diario["Max"]   = df_diario.loc[:, df_diario.columns[3:-1]].max(axis=1)   # máximo
    df_diario["Min"]   = df_diario.loc[:, df_diario.columns[3:-2]].min(axis=1)   # mínimo
    df_diario["Mean"]   = df_diario.loc[:, df_diario.columns[3:-3]].mean(axis=1)   # media

    df_diario['DayOfYear'] = pd.to_datetime({'year': df_diario['Year'], 'month': df_diario['Month'], 'day': df_diario['Day']}).dt.dayofyear

    return df_diario


