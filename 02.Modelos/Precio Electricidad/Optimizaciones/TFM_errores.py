import pandas as pd
import numpy as np


# Gráficos
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Preprocesado y modelado

from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import gaussian_kde


def wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def estudio_residuos_errores(X_train,X_test,y_train,y_test,y_train_pred,y_test_pred):
    # Generamos dataframes que nos sirven para el análisis de salidas, residuos y errores
    # Para el análisis de las salidas
    X_train_plot = X_train.copy()
    X_test_plot  = X_test.copy()

    # Crear columna fecha
    X_train_plot['Fecha'] = pd.to_datetime(
        dict(year=X_train_plot['Year'],
            month=X_train_plot['Month'],
            day=X_train_plot['Day'])
    )

    X_test_plot['Fecha'] = pd.to_datetime(
        dict(year=X_test_plot['Year'],
            month=X_test_plot['Month'],
            day=X_test_plot['Day'])
    )

    # TRAIN
    train_predict_analysis = pd.DataFrame({
        "Fecha": X_train_plot['Fecha'] ,
        "y_real": y_train,
        "y_pred": y_train_pred
    })

    train_predict_analysis["residuo"] = train_predict_analysis["y_real"] - train_predict_analysis["y_pred"]
    train_predict_analysis["residuo_relativo"] = (train_predict_analysis["y_real"] - train_predict_analysis["y_pred"])/ train_predict_analysis["y_real"]*100
    train_predict_analysis["y_q"] = pd.cut(train_predict_analysis["y_real"],bins=4)

    # TEST
    test_predict_analysis = pd.DataFrame({
        "Fecha": X_test_plot['Fecha'] ,
        "y_real": y_test,
        "y_pred": y_test_pred
    })

    test_predict_analysis["error"] = test_predict_analysis["y_real"] - test_predict_analysis["y_pred"]
    test_predict_analysis["error_relativo"] = (test_predict_analysis["y_real"] - test_predict_analysis["y_pred"])/ test_predict_analysis["y_real"]*100
    test_predict_analysis["y_q"] = pd.cut(test_predict_analysis["y_real"],bins=4)

    # Ordenamos por fecha
    train_predict_analysis = train_predict_analysis.sort_values("Fecha")
    test_predict_analysis  = test_predict_analysis.sort_values("Fecha")
    
    
    ### 2.1 Gráfica Y real vs Y Predicha (TEST and TRAIN)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    
    # 🔹 TÍTULO PRINCIPAL
    fig.suptitle("Series Actual vs Predicted", fontsize=16, fontweight="bold")

    # 1️⃣ TRAIN
    axes[0].plot(
        train_predict_analysis["Fecha"],
        train_predict_analysis["y_real"],
        label="Real",
        color="green"
    )

    axes[0].plot(
        train_predict_analysis["Fecha"],
        train_predict_analysis["y_pred"],
        label="Predicción"
    )

    axes[0].set_title("TRAIN Set")
    axes[0].set_ylabel("Precio electricidad (EUR/MWh)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 1️⃣ TEST
    axes[1].plot(
        test_predict_analysis["Fecha"],
        test_predict_analysis["y_real"],
        label="Real",
        color="green"
    )

    axes[1].plot(
        test_predict_analysis["Fecha"],
        test_predict_analysis["y_pred"],
        label="Predicción"
    )

    axes[1].set_title("TEST Set")
    axes[1].set_ylabel("Precio electricidad (EUR/MWh)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    
    ### 2.2 Análisis de residuos
    
    y_real=train_predict_analysis["y_real"]
    y_pred=train_predict_analysis["y_pred"]
    fecha=train_predict_analysis["Fecha"]
    diferencia=train_predict_analysis["residuo"]
    diferencia_relativa=train_predict_analysis["residuo_relativo"]

    # Error RMSE 
    rmse_train = root_mean_squared_error(
            y_true  = y_real,
            y_pred  = y_pred
    )
    print("")
    print(f"El error (rmse) de train es: {rmse_train:.2f} EUR/MWh")

    mape_train = mean_absolute_percentage_error(y_real, y_pred) * 100
    print(f"MAPE de train = {mape_train:.2f} %")

    wape_train = wape(y_real, y_pred) * 100
    print(f"WAPE de train = {wape_train:.2f} %")
    
    r2_train = r2_score(y_real, y_pred)
    print(f"R2 train = {r2_train:.4f}")

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    
    # 🔹 TÍTULO PRINCIPAL
    fig.suptitle("TRAIN Set", fontsize=16, fontweight="bold")

    # 1️⃣ Residuos vs Fecha
    axes[0, 0].plot(
        fecha,
        diferencia
    )
    axes[0, 0].axhline(0, color="black", linestyle="--")
    axes[0, 0].set_title("Residuos")
    axes[0, 0].set_ylabel("Residuo")

    # 2️⃣ Residuos relativos vs Fecha
    axes[0, 1].plot(
        fecha,
        diferencia_relativa
    )
    axes[0, 1].axhline(0, color="black", linestyle="--")
    axes[0, 1].set_title("Residuos relativos (%)")
    axes[0, 1].set_ylabel("Residuo relativo (%)")


    # 3️⃣ Distribución residuos (hist + KDE)
    axes[1, 0].hist(diferencia, bins=30, density=True, alpha=0.3)

    kde = gaussian_kde(diferencia)
    x = np.linspace(diferencia.min(), diferencia.max(), 500)
    axes[1, 0].plot(x, kde(x), color="navy", linewidth=1)

    axes[1, 0].set_title("Distribución de los residuos")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].set_ylabel("Densidad")


    # 4️⃣ Distribución residuos relativos (hist + KDE)
    res_rel = train_predict_analysis["residuo_relativo"]

    axes[1, 1].hist(diferencia_relativa, bins=30, density=True, alpha=0.3)

    kde = gaussian_kde(diferencia_relativa)
    x = np.linspace(diferencia_relativa.min(), diferencia_relativa.max(), 500)
    axes[1, 1].plot(x, kde(x), color="navy", linewidth=1)

    axes[1, 1].set_title("Distribución de los residuos relativos (%)")
    axes[1, 1].set_xlabel("Residuo relativo (%)")
    axes[1, 1].set_ylabel("Densidad")


    axes[2, 0].scatter(
        y_real,
        y_pred)

    axes[2, 0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
    axes[2, 0].set_xlabel('Valores reales')
    axes[2, 0].set_ylabel('Valores predichos')
    axes[2, 0].set_title('Predicted vs Actual')
    axes[2, 0].grid(True)

    axes[2, 1].scatter(
        y_pred,
        diferencia)

    axes[2, 1].axhline(0,c="k")
    axes[2, 1].set_xlabel('Valores predichos')
    axes[2, 1].set_ylabel('Residuos')
    axes[2, 1].set_title('Residuos vs Predichos')
    axes[2, 1].grid(True)
    plt.tight_layout()
    plt.show()
    
    
    ### 2.2 Análisis de errores
    
    y_real=test_predict_analysis["y_real"]
    y_pred=test_predict_analysis["y_pred"]
    fecha=test_predict_analysis["Fecha"]
    diferencia=test_predict_analysis["error"]
    diferencia_relativa=test_predict_analysis["error_relativo"]


    # Error de test del modelo 
    rmse_test = root_mean_squared_error(
            y_true  = y_real,
            y_pred  = y_pred
    )
    print("")
    print(f"El error (rmse) de test es: {rmse_test:.2f} EUR/MWh")

    mape_test = mean_absolute_percentage_error(y_real, y_pred) * 100
    print(f"MAPE de test = {mape_test:.2f} %")

    wape_test = wape(y_real, y_pred) * 100
    print(f"WAPE de test = {wape_test:.2f} %")

    r2_test = r2_score(y_real, y_pred)
    print(f"R2 test = {r2_test:.4f}")


    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    #  TÍTULO PRINCIPAL
    fig.suptitle("TEST Set", fontsize=16, fontweight="bold")

    # 1️⃣ Residuos vs Fecha
    axes[0, 0].plot(
        fecha,
        diferencia
    )
    axes[0, 0].axhline(0, color="black", linestyle="--")
    axes[0, 0].set_title("Errores")
    axes[0, 0].set_ylabel("Error")

    # 2️⃣ Residuos relativos vs Fecha
    axes[0, 1].plot(
        fecha,
        diferencia_relativa
    )
    axes[0, 1].axhline(0, color="black", linestyle="--")
    axes[0, 1].set_title("Errores relativos (%)")
    axes[0, 1].set_ylabel("Error relativo (%)")


    #  Distribución residuos (hist + KDE)
    axes[1, 0].hist(diferencia, bins=30, density=True, alpha=0.3)

    kde = gaussian_kde(diferencia)
    x = np.linspace(diferencia.min(), diferencia.max(), 500)
    axes[1, 0].plot(x, kde(x), color="navy", linewidth=1)

    axes[1, 0].set_title("Distribución de los errores")
    axes[1, 0].set_xlabel("Error")
    axes[1, 0].set_ylabel("Densidad")


    #  Distribución residuos relativos (hist + KDE)

    axes[1, 1].hist(diferencia_relativa, bins=30, density=True, alpha=0.3)

    kde = gaussian_kde(diferencia_relativa)
    x = np.linspace(diferencia_relativa.min(), diferencia_relativa.max(), 500)
    axes[1, 1].plot(x, kde(x), color="navy", linewidth=1)

    axes[1, 1].set_title("Distribución de los errores relativos (%)")
    axes[1, 1].set_xlabel("Error relativo (%)")
    axes[1, 1].set_ylabel("Densidad")


    axes[2, 0].scatter(
        y_real,
        y_pred)

    axes[2, 0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
    axes[2, 0].set_xlabel('Valores reales')
    axes[2, 0].set_ylabel('Valores predichos')
    axes[2, 0].set_title('Predicted vs Actual')
    axes[2, 0].grid(True)

    axes[2, 1].scatter(
        y_pred,
        diferencia)

    axes[2, 1].axhline(0,c="k")
    axes[2, 1].set_xlabel('Valores predichos')
    axes[2, 1].set_ylabel('Errores')
    axes[2, 1].set_title('Errores vs Predichos')
    axes[2, 1].grid(True)
    plt.tight_layout()
    plt.show()

def time_idx_to_date(time_idx, start_date="2019-01-01"):
    """
    Convierte un índice de tiempo (entero) en una fecha.
    
    Parámetros
    ----------
    time_idx : int o array-like
        Índice(s) de tiempo donde 1 = start_date
    start_date : str o datetime
        Fecha inicial (por defecto 2019-01-01)
    
    Retorna
    -------
    pandas.Timestamp o pandas.DatetimeIndex
    """
    start = pd.to_datetime(start_date)
    return start + pd.to_timedelta(time_idx - 1, unit="D")



def estudio_residuos_errores_index(X_train,X_test,y_train,y_test,y_train_pred,y_test_pred):
    # Generamos dataframes que nos sirven para el análisis de salidas, residuos y errores
    # Para el análisis de las salidas
        X_train_plot= X_train.copy()
        X_test_plot  = X_test.copy()

    
        # TRAIN
        train_predict_analysis = pd.DataFrame({
            "time_idx": X_train_plot['time_idx'] ,
            "y_real": y_train,
            "y_pred": y_train_pred
        })

        train_predict_analysis["residuo"] = train_predict_analysis["y_real"] - train_predict_analysis["y_pred"]
        train_predict_analysis["residuo_relativo"] = (train_predict_analysis["y_real"] - train_predict_analysis["y_pred"])/ train_predict_analysis["y_real"]*100
        train_predict_analysis["y_q"] = pd.cut(train_predict_analysis["y_real"],bins=4)

        # TEST
        test_predict_analysis = pd.DataFrame({
            "time_idx": X_test_plot['time_idx'] ,
            "y_real": y_test,
            "y_pred": y_test_pred
        })

        test_predict_analysis["error"] = test_predict_analysis["y_real"] - test_predict_analysis["y_pred"]
        test_predict_analysis["error_relativo"] = (test_predict_analysis["y_real"] - test_predict_analysis["y_pred"])/ test_predict_analysis["y_real"]*100
        test_predict_analysis["y_q"] = pd.cut(test_predict_analysis["y_real"],bins=4)

        # Ordenamos por fecha
        train_predict_analysis = train_predict_analysis.sort_values("time_idx")
        test_predict_analysis  = test_predict_analysis.sort_values("time_idx")


        train_predict_analysis["date"] = time_idx_to_date(train_predict_analysis["time_idx"])
        test_predict_analysis["date"] = time_idx_to_date(test_predict_analysis["time_idx"])

        ### 2.1 Gráfica Y real vs Y Predicha (TEST and TRAIN)
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
        
        #  TÍTULO PRINCIPAL
        fig.suptitle("Series Actual vs Predicted", fontsize=16, fontweight="bold")

        #  TRAIN
        axes[0].plot(
            train_predict_analysis["date"],
            train_predict_analysis["y_real"],
            label="Real",
            color="green"
        )

        axes[0].plot(
            train_predict_analysis["date"],
            train_predict_analysis["y_pred"],
            label="Predicción"
        )

        axes[0].set_title("TRAIN Set")
        axes[0].set_ylabel("Precio electricidad (EUR/MWh)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        #  TEST
        axes[1].plot(
            test_predict_analysis["date"],
            test_predict_analysis["y_real"],
            label="Real",
            color="green"
        )

        axes[1].plot(
            test_predict_analysis["date"],
            test_predict_analysis["y_pred"],
            label="Predicción"
        )

        axes[1].set_title("TEST Set")
        axes[1].set_ylabel("Precio electricidad (EUR/MWh)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
                
        # Resolución trimestral (cada 3 meses)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
                
        ### 2.2 Análisis de residuos
        
        y_real=train_predict_analysis["y_real"]
        y_pred=train_predict_analysis["y_pred"]
        date=train_predict_analysis["date"]
        diferencia=train_predict_analysis["residuo"]
        diferencia_relativa=train_predict_analysis["residuo_relativo"]

        # Error RMSE 
        rmse_train = root_mean_squared_error(
                y_true  = y_real,
                y_pred  = y_pred
        )
        print("")
        print(f"El error (rmse) de train es: {rmse_train:.2f} EUR/MWh")

        mape_train = mean_absolute_percentage_error(y_real, y_pred) * 100
        print(f"MAPE de train = {mape_train:.2f} %")

        wape_train = wape(y_real, y_pred) * 100
        print(f"WAPE de train = {wape_train:.2f} %")
        
        r2_train = r2_score(y_real, y_pred)
        print(f"R2 train = {r2_train:.4f}")

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
        
        #  TÍTULO PRINCIPAL
        fig.suptitle("TRAIN Set", fontsize=16, fontweight="bold")

        #  Residuos vs Fecha
        axes[0, 0].plot(
            date,
            diferencia
        )
        axes[0, 0].axhline(0, color="black", linestyle="--")
        axes[0, 0].set_title("Residuos")
        axes[0, 0].set_ylabel("Residuo")

        #  Residuos relativos vs Fecha
        axes[0, 1].plot(
            date,
            diferencia_relativa
        )
        axes[0, 1].axhline(0, color="black", linestyle="--")
        axes[0, 1].set_title("Residuos relativos (%)")
        axes[0, 1].set_ylabel("Residuo relativo (%)")


        #  Formato temporal trimestral para ambos ejes
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        
        #  Distribución residuos (hist + KDE)
        axes[1, 0].hist(diferencia, bins=30, density=True, alpha=0.3)

        kde = gaussian_kde(diferencia)
        x = np.linspace(diferencia.min(), diferencia.max(), 500)
        axes[1, 0].plot(x, kde(x), color="navy", linewidth=1)

        axes[1, 0].set_title("Distribución de los residuos")
        axes[1, 0].set_xlabel("Residuo")
        axes[1, 0].set_ylabel("Densidad")


        #  Distribución residuos relativos (hist + KDE)
        res_rel = train_predict_analysis["residuo_relativo"]

        axes[1, 1].hist(diferencia_relativa, bins=30, density=True, alpha=0.3)

        kde = gaussian_kde(diferencia_relativa)
        x = np.linspace(diferencia_relativa.min(), diferencia_relativa.max(), 500)
        axes[1, 1].plot(x, kde(x), color="navy", linewidth=1)

        axes[1, 1].set_title("Distribución de los residuos relativos (%)")
        axes[1, 1].set_xlabel("Residuo relativo (%)")
        axes[1, 1].set_ylabel("Densidad")


        axes[2, 0].scatter(
            y_real,
            y_pred)

        axes[2, 0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
        axes[2, 0].set_xlabel('Valores reales')
        axes[2, 0].set_ylabel('Valores predichos')
        axes[2, 0].set_title('Predicted vs Actual')
        axes[2, 0].grid(True)

        axes[2, 1].scatter(
            y_pred,
            diferencia)

        axes[2, 1].axhline(0,c="k")
        axes[2, 1].set_xlabel('Valores predichos')
        axes[2, 1].set_ylabel('Residuos')
        axes[2, 1].set_title('Residuos vs Predichos')
        axes[2, 1].grid(True)
        plt.tight_layout()
        plt.show()
        
        
        ### 2.2 Análisis de errores
        
        y_real=test_predict_analysis["y_real"]
        y_pred=test_predict_analysis["y_pred"]
        date=test_predict_analysis["date"]
        diferencia=test_predict_analysis["error"]
        diferencia_relativa=test_predict_analysis["error_relativo"]


        # Error de test del modelo 
        rmse_test = root_mean_squared_error(
                y_true  = y_real,
                y_pred  = y_pred
        )
        print("")
        print(f"El error (rmse) de test es: {rmse_test:.2f} EUR/MWh")

        mape_test = mean_absolute_percentage_error(y_real, y_pred) * 100
        print(f"MAPE de test = {mape_test:.2f} %")

        wape_test = wape(y_real, y_pred) * 100
        print(f"WAPE de test = {wape_test:.2f} %")

        r2_test = r2_score(y_real, y_pred)
        print(f"R2 test = {r2_test:.4f}")


        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
        # 🔹 TÍTULO PRINCIPAL
        fig.suptitle("TEST Set", fontsize=16, fontweight="bold")

        #  Residuos vs Fecha
        axes[0, 0].plot(
            date,
            diferencia
        )
        axes[0, 0].axhline(0, color="black", linestyle="--")
        axes[0, 0].set_title("Errores")
        axes[0, 0].set_ylabel("Error")

        #  Residuos relativos vs Fecha
        axes[0, 1].plot(
            date,
            diferencia_relativa
        )
        axes[0, 1].axhline(0, color="black", linestyle="--")
        axes[0, 1].set_title("Errores relativos (%)")
        axes[0, 1].set_ylabel("Error relativo (%)")
        
        # Formato temporal trimestral para ambos ejes
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Distribución residuos (hist + KDE)
        axes[1, 0].hist(diferencia, bins=30, density=True, alpha=0.3)

        kde = gaussian_kde(diferencia)
        x = np.linspace(diferencia.min(), diferencia.max(), 500)
        axes[1, 0].plot(x, kde(x), color="navy", linewidth=1)

        axes[1, 0].set_title("Distribución de los errores")
        axes[1, 0].set_xlabel("Error")
        axes[1, 0].set_ylabel("Densidad")


        # Distribución residuos relativos (hist + KDE)

        axes[1, 1].hist(diferencia_relativa, bins=30, density=True, alpha=0.3)

        kde = gaussian_kde(diferencia_relativa)
        x = np.linspace(diferencia_relativa.min(), diferencia_relativa.max(), 500)
        axes[1, 1].plot(x, kde(x), color="navy", linewidth=1)

        axes[1, 1].set_title("Distribución de los errores relativos (%)")
        axes[1, 1].set_xlabel("Error relativo (%)")
        axes[1, 1].set_ylabel("Densidad")


        axes[2, 0].scatter(
            y_real,
            y_pred)

        axes[2, 0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
        axes[2, 0].set_xlabel('Valores reales')
        axes[2, 0].set_ylabel('Valores predichos')
        axes[2, 0].set_title('Predicted vs Actual')
        axes[2, 0].grid(True)

        axes[2, 1].scatter(
            y_pred,
            diferencia)

        axes[2, 1].axhline(0,c="k")
        axes[2, 1].set_xlabel('Valores predichos')
        axes[2, 1].set_ylabel('Errores')
        axes[2, 1].set_title('Errores vs Predichos')
        axes[2, 1].grid(True)
        plt.tight_layout()
        plt.show()