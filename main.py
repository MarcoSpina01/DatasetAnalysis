import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.special import softmax
from numpy.random import seed
from keras.layers import Input, Dropout
from keras.layers.core import Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json
from scipy.special import softmax
import seaborn as sns


def main():
    readLoss()


def readLoss():
    loss_df = pd.read_csv("AutoEncoder_loss_p_column.csv", index_col=None)

    loss_df = pd.DataFrame(loss_df)

    softmax_result = softmax(loss_df, axis=1)

    loss_df = pd.DataFrame(
        data=softmax_result, columns=loss_df.columns, index=loss_df.index
    )

    for column in loss_df.columns:
        plt.subplots(figsize=(15, 7))
        plt.plot(loss_df.index, loss_df[f"{column}"], label=f"{column} loss")
        plt.legend(loc="upper right")

        plt.show()

    plt.subplots(figsize=(15, 7))

    df_label = [
        "Torque",
        "Cut lag",
        "Cut speed",
        "Cut position",
        "Film position",
        "Film speed",
        "Film lag",
        "VAX",
    ]

    plt.stackplot(
        loss_df.index,
        loss_df["pCut::Motor_Torque"],
        loss_df["pCut::CTRL_Position_controller::Lag_error"],
        loss_df["pCut::CTRL_Position_controller::Actual_speed"],
        loss_df["pCut::CTRL_Position_controller::Actual_position"],
        loss_df["pSvolFilm::CTRL_Position_controller::Actual_position"],
        loss_df["pSvolFilm::CTRL_Position_controller::Actual_speed"],
        loss_df["pSvolFilm::CTRL_Position_controller::Lag_error"],
        loss_df["pSpintor::VAX_speed"],
        labels=df_label,
    )

    plt.legend(loc="upper center", ncol=8)

    plt.ylim(0, 1)

    plt.subplots(figsize=(15, 7))

    df_label = [
        "Torque",
        "Cut lag",
        "Cut speed",
        "Cut position",
        "Film position",
        "Film speed",
        "Film lag",
        "VAX",
    ]

    loss_df = loss_df[400000:600000]

    plt.stackplot(
        loss_df.index,
        loss_df["pCut::Motor_Torque"],
        loss_df["pCut::CTRL_Position_controller::Lag_error"],
        loss_df["pCut::CTRL_Position_controller::Actual_speed"],
        loss_df["pCut::CTRL_Position_controller::Actual_position"],
        loss_df["pSvolFilm::CTRL_Position_controller::Actual_position"],
        loss_df["pSvolFilm::CTRL_Position_controller::Actual_speed"],
        loss_df["pSvolFilm::CTRL_Position_controller::Lag_error"],
        loss_df["pSpintor::VAX_speed"],
        labels=df_label,
    )

    plt.legend(loc="upper center", ncol=8)

    plt.ylim(0, 1)

    for column in loss_df.columns:
        plt.subplots(figsize=(15, 7))

        sns.distplot((loss_df[f"{column}"]), bins=15).set_title(
            f"Contribution Distribution"
        )
        plt.xlim(0, 1)

        plt.show()


def autoEncLossForParameters():
    main_df = pd.read_csv("OneYearComplete.csv")
    main_df = main_df.drop(["Num", "timestamp"], axis=1)
    main_df = handle_non_numeric(main_df)
    X = main_df

    scaler = preprocessing.MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X = preprocessing.scale(X)

    train_percentage = 0.15
    train_size = int(len(main_df.index) * train_percentage)

    X_train = X[:train_size]

    seed(10)

    act_func = "elu"

    model = Sequential()

    model.add(
        Dense(
            50,
            activation=act_func,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(0.0),
            input_shape=(X_train.shape[1],),
        )
    )

    model.add(Dense(10, activation=act_func, kernel_initializer="glorot_uniform"))

    model.add(Dense(50, activation=act_func, kernel_initializer="glorot_uniform"))

    model.add(Dense(X_train.shape[1], kernel_initializer="glorot_uniform"))

    model.compile(loss="mse", optimizer="adam")

    NUM_EPOCHS = 50
    BATCH_SIZE = 200

    history = model.fit(
        np.array(X_train),
        np.array(X_train),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=0.1,
        verbose=1,
    )
    X_pred = model.predict(np.array(X))
    X_pred = pd.DataFrame(X_pred, columns=main_df.columns)
    X_pred.index = pd.DataFrame(main_df).index

    X = pd.DataFrame(X, columns=main_df.columns)
    X.index = pd.DataFrame(main_df).index

    loss_df = pd.DataFrame()

    main_df.drop("Mode", axis=1, inplace=True)

    for column in main_df.columns:
        loss_df[f"{column}"] = (X_pred[f"{column}"] - X[f"{column}"]).abs()

        plt.subplots(figsize=(15, 7))
        plt.plot(loss_df.index, loss_df[f"{column}"], label=f"{column} loss")
        plt.legend(loc="upper right")

        plt.show()

    loss_df = pd.DataFrame(loss_df)

    loss_df.to_csv("AutoEncoder_loss_p_column.csv", index=False)


def autoEncLoss():
    k_anomaly = pd.read_csv("KM_Distance.csv")
    score = pd.read_csv("SVM_Score.csv")
    enc_loss = pd.read_csv("AutoEncoder_loss.csv")

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    main_df = pd.read_csv("OneYearComplete.csv")

    main_df["AutoEnc_loss"] = corr["AutoEnc_loss"]

    main_df["timestamp"] = pd.to_datetime(main_df["timestamp"])

    main_df["month"] = main_df["timestamp"].dt.month

    months = main_df["month"].unique()

    main_df["AutoEnc_loss"] = corr["AutoEnc_loss"]

    for month in months:
        month_df = main_df.groupby("month").get_group(month)

        month_df = main_df.groupby("month").get_group(month)

        upper_threshold = np.full((month_df["AutoEnc_loss"].size, 1), 0.1)
        high_density_threshold = np.full((month_df["AutoEnc_loss"].size, 1), 0.05)

        plt.subplots(figsize=(15, 7))
        plt.plot(
            month_df.index,
            month_df["AutoEnc_loss"],
            label=f"AutoEnc_loss month_{month}",
        )
        plt.plot(month_df.index, upper_threshold, label="Upper Threshold")
        plt.plot(month_df.index, high_density_threshold, label="Highest Density")
        plt.legend(loc="upper right")
        plt.ylim(0, 1.3)

        plt.show()

        plt.subplots(figsize=(15, 7))
        sns.distplot((month_df["AutoEnc_loss"]), bins=15).set_title(
            f"Month {month} Loss Distribution"
        )
        plt.xlim([-1.2, 1.2])
        plt.show()


def Compare2():
    k_anomaly = pd.read_csv("KM_Distance.csv")
    score = pd.read_csv("SVM_Score.csv")
    enc_loss = pd.read_csv("AutoEncoder_loss.csv")

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    lower_threshold = np.full((corr["SVM_score"].size, 1), 0)
    upper_threshold = np.full((corr["SVM_score"].size, 1), 18000)
    high_density_threshold = np.full((corr["SVM_score"].size, 1), 13250)

    plt.subplots(figsize=(15, 7))

    plt.plot(corr.index, corr["SVM_score"], "k", markersize=1, label="OCSVM_score")

    plt.plot(
        corr.index,
        corr["SVM_score"].rolling(100).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )

    plt.plot(corr.index, lower_threshold, label="Lower Threshold")
    plt.plot(corr.index, upper_threshold, label="Upper Threshold")
    plt.plot(corr.index, high_density_threshold, label="Highest Density")
    plt.legend(loc="upper right")

    plt.show()

    lower_threshold = np.full((corr["KM_cluster_distance"].size, 1), 1.2)
    upper_threshold = np.full((corr["KM_cluster_distance"].size, 1), 17.5)
    high_density_threshold = np.full((corr["KM_cluster_distance"].size, 1), 2.5)

    plt.subplots(figsize=(15, 7))

    plt.plot(
        corr.index,
        corr["KM_cluster_distance"],
        "k",
        markersize=1,
        label="KM_cluster_distance",
    )
    plt.plot(
        corr.index,
        corr["KM_cluster_distance"].rolling(100).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )
    plt.plot(corr.index, lower_threshold, label="Lower Threshold")
    plt.plot(corr.index, upper_threshold, label="Upper Threshold")
    plt.plot(corr.index, high_density_threshold, label="Highest Density")
    plt.legend(loc="upper right")
    plt.show()

    lower_threshold = np.full((corr["AutoEnc_loss"].size, 1), 0)
    upper_threshold = np.full((corr["AutoEnc_loss"].size, 1), 0.1)
    high_density_threshold = np.full((corr["AutoEnc_loss"].size, 1), 0.05)

    plt.subplots(figsize=(15, 7))

    plt.plot(corr.index, corr["AutoEnc_loss"], "k", markersize=1, label="AutoEnc_loss")
    plt.plot(
        corr.index,
        corr["AutoEnc_loss"].rolling(100).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )
    plt.plot(corr.index, lower_threshold, label="Lower Threshold")
    plt.plot(corr.index, upper_threshold, label="Upper Threshold")
    plt.plot(corr.index, high_density_threshold, label="Highest Density")
    plt.legend(loc="upper right")
    plt.show()

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    plt.subplots(figsize=(15, 7))

    plt.plot(corr["KM_cluster_distance"], corr["SVM_score"], "b.", markersize=1)
    plt.xlabel("KM")
    plt.ylabel("SVM")
    plt.show()

    plt.subplots(figsize=(15, 7))

    plt.plot(corr["AutoEnc_loss"], corr["SVM_score"], "b.", markersize=1)
    plt.xlabel("Encoder")
    plt.ylabel("SVM")
    plt.show()

    plt.subplots(figsize=(15, 7))

    plt.plot(corr["AutoEnc_loss"], corr["KM_cluster_distance"], "b.", markersize=1)
    plt.xlabel("Encoder")
    plt.ylabel("KM")
    plt.show()


def Compare():
    k_anomaly = pd.read_csv("KM_Distance.csv")
    score = pd.read_csv("SVM_Score.csv")
    enc_loss = pd.read_csv("AutoEncoder_loss.csv")

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    plt.subplots(figsize=(15, 7))

    plt.plot(corr.index, corr["SVM_score"], "g.", markersize=1, label="OCSVM_score")

    plt.plot(
        corr.index,
        corr["SVM_score"].rolling(1000).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )

    plt.legend(loc="upper right")

    plt.show()

    plt.subplots(figsize=(15, 7))

    plt.plot(
        corr.index,
        corr["KM_cluster_distance"],
        "g.",
        markersize=1,
        label="KM_cluster_distance",
    )
    plt.plot(
        corr.index,
        corr["KM_cluster_distance"].rolling(1000).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )

    plt.legend(loc="upper right")
    plt.show()

    plt.subplots(figsize=(15, 7))

    plt.plot(corr.index, corr["AutoEnc_loss"], "g.", markersize=1, label="AutoEnc_loss")
    plt.plot(
        corr.index,
        corr["AutoEnc_loss"].rolling(1000).mean(),
        "r",
        markersize=1,
        label="Moving Mean",
    )

    plt.legend(loc="upper right")
    plt.show()

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    plt.subplots(figsize=(10, 7))

    sns.distplot(corr["SVM_score"].head(160000), bins=15)

    plt.show()

    plt.subplots(figsize=(10, 7))
    sns.distplot(corr["KM_cluster_distance"].head(160000), bins=15)
    plt.show()

    plt.subplots(figsize=(10, 7))
    sns.distplot(corr["AutoEnc_loss"].head(160000), bins=15)
    plt.show()

    corr = pd.DataFrame()
    corr["SVM_score"] = score["score"]
    corr["KM_cluster_distance"] = k_anomaly["0"]
    corr["AutoEnc_loss"] = enc_loss["Loss_mae"]

    plt.subplots(figsize=(10, 7))
    sns.distplot(corr["SVM_score"], bins=15)
    plt.show()

    plt.subplots(figsize=(10, 7))
    sns.distplot(corr["KM_cluster_distance"], bins=15)
    plt.show()

    plt.subplots(figsize=(10, 7))
    sns.distplot(corr["AutoEnc_loss"], bins=15)
    plt.show()


def AutoEncoder():
    main_df = pd.read_csv("OneYearComplete.csv")
    main_df = main_df.drop(["Num", "timestamp"], axis=1)
    main_df = handle_non_numeric(main_df)
    X = main_df

    scaler = preprocessing.MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X = preprocessing.scale(X)

    train_percentage = 0.15
    train_size = int(len(main_df.index) * train_percentage)

    X_train = X[:train_size]
    # ----------------------------------------------------------------------------------

    # Seed for random batch validation and training
    seed(10)

    # Elu activatoin function
    act_func = "elu"

    # Input layer
    model = Sequential()

    # First hidden layer, connected to input vector X.
    model.add(
        Dense(
            50,
            activation=act_func,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(0.0),
            input_shape=(X_train.shape[1],),
        )
    )
    # Second hidden layer
    model.add(Dense(10, activation=act_func, kernel_initializer="glorot_uniform"))
    # Thrid hidden layer
    model.add(Dense(50, activation=act_func, kernel_initializer="glorot_uniform"))

    # Input layer
    model.add(Dense(X_train.shape[1], kernel_initializer="glorot_uniform"))

    # Loss function and Optimizer choice
    model.compile(loss="mse", optimizer="adam")

    # Train model for 50 epochs, batch size of 200
    NUM_EPOCHS = 50
    BATCH_SIZE = 200

    history = model.fit(
        np.array(X_train),
        np.array(X_train),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=0.1,
        verbose=1,
    )
    plt.subplots(figsize=(15, 7))

    plt.plot(history.history["loss"], "b", label="Training loss")
    plt.plot(history.history["val_loss"], "r", label="Validation loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss, [mse]")

    plt.show()

    X_pred = model.predict(np.array(X_train))

    X_pred = pd.DataFrame(X_pred, columns=main_df.columns)
    X_pred.index = pd.DataFrame(X_train).index

    scored = pd.DataFrame(index=pd.DataFrame(X_train).index)
    scored["Loss_mae"] = np.mean(np.abs(X_pred - X_train), axis=1)

    plt.subplots(figsize=(15, 7))
    sns.distplot(scored["Loss_mae"], bins=15, kde=True, color="blue")

    X_pred = model.predict(np.array(X))
    X_pred = pd.DataFrame(X_pred, columns=main_df.columns)
    X_pred.index = pd.DataFrame(X).index

    scored = pd.DataFrame(index=pd.DataFrame(X).index)
    scored["Loss_mae"] = np.mean(np.abs(X_pred - X), axis=1)

    plt.subplots(figsize=(15, 7))

    scored.to_csv("AutoEncoder_loss.csv")

    plt.plot(scored["Loss_mae"], "b", label="Prediction Loss")

    plt.legend(loc="upper right")
    plt.xlabel("Sample")
    plt.ylabel("Loss, [mse]")

    plt.subplots(figsize=(15, 7))
    enc_loss = pd.read_csv("AutoEncoder_loss.csv")
    plt.plot(
        enc_loss.index,
        enc_loss["Loss_mae"],
        "g.",
        markersize=1,
        label="AutoEncoder Loss",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Sample")
    plt.show()

    plt.subplots(figsize=(15, 7))
    k_anomaly = pd.read_csv("KM_Distance.csv")
    plt.plot(
        k_anomaly.index, k_anomaly["0"], "g.", markersize=1, label="KM cluster Distance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Sample")
    plt.show()

    plt.subplots(figsize=(15, 7))
    score = pd.read_csv("SVM_Score.csv")
    plt.plot(score.index, score["score"], "g.", markersize=1, label="OCSVM score")
    plt.legend(loc="upper right")
    plt.xlabel("Sample")
    plt.show()

    plt.subplots(figsize=(15, 7))

    k_anomaly = pd.read_csv("KM_Distance.csv")
    score = pd.read_csv("SVM_Score.csv")
    enc_loss = pd.read_csv("AutoEncoder_loss.csv")

    k_distance = k_anomaly / k_anomaly.max()
    svm_score = score / score.max()

    plt.plot(enc_loss.index, enc_loss["Loss_mae"], label="AutoEncoder Loss")
    plt.plot(svm_score.index, svm_score["score"], label="OCSVM score")
    plt.plot(k_distance.index, k_distance["0"], label="Kmeans Euclidean Dist")

    plt.gca().legend(("AutoEncoder Loss", "OCSVM score", "Kmeans Euclidean Dist"))


def viewChronological():
    data = pd.read_csv("OneYearComplete.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    motor_torque = data["pCut::Motor_Torque"]
    timestamps = data["timestamp"]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, motor_torque, label="Coppia del Motore")
    plt.title("Andamento temporale della Coppia del Motore")
    plt.xlabel("Timestamp")
    plt.ylabel("Coppia del Motore")
    plt.legend()
    plt.grid(True)
    plt.show()


# METODOLOGIA NON FUNZIONANTE
def viewColumnDistribution():
    main_df = pd.read_csv("OneYearComplete.csv")
    main_df[
        [
            "pCut::Motor_Torque",
            "pCut::CTRL_Position_controller::Actual_position",
            "pCut::CTRL_Position_controller::Lag_error",
            "pCut::CTRL_Position_controller::Actual_speed",
        ]
    ].hist(bins=20, figsize=(12, 8))
    plt.show()
    data = pd.read_csv("OneYearComplete.csv")

    percentuale_addestramento = 0.7
    numero_totale_righe = len(data)
    numero_righe_addestramento = int(percentuale_addestramento * numero_totale_righe)

    train_data = data[:numero_righe_addestramento]
    test_data = data[numero_righe_addestramento:]

    feature_di_addestramento = [
        "pCut::Motor_Torque",
        "pCut::CTRL_Position_controller::Lag_error",
        "pCut::CTRL_Position_controller::Actual_position",
        "pCut::CTRL_Position_controller::Actual_speed",
        "Mode",
    ]

    # Creazione del modello di regressione lineare
    modello_regressione = LinearRegression()
    print("vai")

    # Addestramento del modello
    modello_regressione.fit(
        train_data[feature_di_addestramento], train_data["Obiettivo_da_predire"]
    )
    print("ok")

    previsioni = modello_regressione.predict(test_data[feature_di_addestramento])
    print("si")

    mse = mean_squared_error(test_data["Obiettivo_da_predire"], previsioni)
    r2 = r2_score(test_data["Obiettivo_da_predire"], previsioni)

    plt.scatter(test_data["Obiettivo_da_predire"], previsioni)
    plt.xlabel("Osservato")
    plt.ylabel("Previsto")
    plt.title("Grafico di Dispersione delle Previste vs. Osservate")
    plt.show()

    # Visualizzazione grafica degli errori residui
    residui = test_data["Obiettivo_da_predire"] - previsioni
    plt.scatter(test_data["Timestamp_Completo"], residui)
    plt.xlabel("Timestamp")
    plt.ylabel("Errori Residui")
    plt.title("Grafico degli Errori Residui")
    plt.show()

    print(f"MSE: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


def format_and_replace_dates(input_file, output_file):
    df = pd.read_csv(input_file)

    def format_date(date_str):
        month_day = date_str[:5]
        time = date_str[6:]

        month_mapping = {
            1: "03",  # Marzo
            2: "04",  # Aprile
            3: "05",  # Maggio
            4: "06",  # Giugno
            5: "07",  # Luglio
            6: "08",  # Agosto
            7: "09",  # Settembre
            8: "10",  # Ottobre
            9: "11",  # Novembre
            10: "12",  # Dicembre
            11: "01",  # Gennaio
            12: "02",  # Febbraio
        }

        formatted_month = month_mapping[int(month_day.split("-")[0])]

        day = month_day[3:]

        formatted_date = f"2023-{formatted_month}-{day}{time}"

        formatted_date = pd.to_datetime(formatted_date, format="%Y-%m-%d%H%M%S")

        return formatted_date

    df["Date_Time"] = df["Date_Time"].apply(format_date)

    df.to_csv(output_file, index=False)


def createDataFrame():
    directory_path = "C:/Users/crowh/PycharmProjects/OneYearIndustrialComponent/input"

    data_frames = []

    file_name_pattern = re.compile(r"(\d{2}-\d{2}T\d{6})_(\d+)_mode(\w+)\.csv")

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            match = file_name_pattern.match(filename)
            if match:
                date_time = match.group(1)
                num = match.group(2)
                mode = match.group(3)

                df["Date_Time"] = date_time
                df["Num"] = num
                df["Mode"] = mode

            data_frames.append(df)

    concatenated_df = pd.concat(data_frames, ignore_index=True)

    concatenated_df.to_csv("One_year_compiled.csv", index=False)


def Kmeans():
    main_df = pd.read_csv("OneYearComplete.csv")
    main_df = main_df.drop(["Num", "timestamp"], axis=1)
    main_df = handle_non_numeric(main_df)
    X = main_df

    scaler = preprocessing.MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X = preprocessing.scale(X)
    print("pre ok")
    # -------------------------------------------------------------------

    train_percentage = 0.15
    train_size = int(len(main_df.index) * train_percentage)
    X_train = X[:train_size]

    kmeans = KMeans(n_clusters=1)
    print("kmeans def")

    kmeans.fit(X_train)
    print("kmeans train")

    k_anomaly = main_df.copy()

    k_anomaly = pd.DataFrame(kmeans.transform(X))

    k_anomaly.to_csv("KM_Distance.csv")

    plt.subplots(figsize=(15, 7))

    plt.plot(k_anomaly.index, k_anomaly[0], "g", markersize=1)


def createScore():
    main_df = pd.read_csv("OneYearComplete.csv")
    main_df = main_df.drop(["Num", "timestamp"], axis=1)
    tqdm.write("Dataset letto")

    main_df = handle_non_numeric(main_df)
    tqdm.write("Modalità trasformate")

    X = main_df

    scaler = preprocessing.MinMaxScaler()
    tqdm.write("Preprocessore definito")

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    tqdm.write("Pre-elaborazione completata")

    X = preprocessing.scale(X)
    tqdm.write("Scaling completato")

    X_train = X[:200000]
    tqdm.write("Training set suddiviso")

    ocsvm = OneClassSVM(nu=0.25, gamma=0.05)
    ocsvm.fit(X_train)
    tqdm.write("Addestramento OneClass SVM completato")

    score = pd.DataFrame()
    tqdm.write("Dataframe creato")

    score["score"] = ocsvm.score_samples(X)
    tqdm.write("Score ritornati")

    plt.subplots(figsize=(15, 7))
    score["score"].plot()

    # Saving score dataframe
    score.to_csv("SVM_Score.csv")
    tqdm.write("Dataframe salvato")


def readScore():
    score = pd.read_csv("SVM_Score.csv")
    plt.subplots(figsize=(15, 7))
    score["score"].plot()


def createOCSVM():
    main_df = pd.read_csv("OneYearComplete.csv")
    tqdm.write("Dataset letto")

    main_df = main_df.drop(["Num", "timestamp"], axis=1)

    main_df = handle_non_numeric(main_df)
    tqdm.write("Modalità trasformate")

    X = main_df

    scaler = preprocessing.MinMaxScaler()
    tqdm.write("Preprocessore definito")

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    tqdm.write("Pre-elaborazione completata")

    X = preprocessing.scale(X)
    tqdm.write("Scaling completato")

    X_train = X[:700000]
    tqdm.write("Training set suddiviso")

    ocsvm = OneClassSVM(nu=0.05, gamma=1)
    ocsvm.fit(X_train)
    tqdm.write("Addestramento OneClass SVM completato")

    df = main_df.copy()
    tqdm.write("DataFrame copiato")

    anomalies = []
    for dato in tqdm(X):
        predizione = ocsvm.predict([dato])
        anomalies.append(predizione[0])

    df["anomaly"] = anomalies
    tqdm.write("Previsione delle anomalie completata")

    df.to_csv("Labled_df2.csv")
    tqdm.write("Salvataggio del DataFrame completato")


def readLabled():
    df = pd.read_csv("Labled_df2.csv", index_col=0)
    scat_1 = df.groupby("anomaly").get_group(1)
    scat_0 = df.groupby("anomaly").get_group(-1)

    plt.subplots(figsize=(15, 7))

    plt.plot(
        scat_1.index,
        scat_1["pCut::Motor_Torque"],
        "g.",
        markersize=1,
        label="Normal Motor Torque",
    )

    plt.plot(
        scat_0.index,
        scat_0["pCut::Motor_Torque"],
        "r.",
        markersize=1,
        label="Anomaly in Motor Torque",
    )

    plt.legend()


def handle_non_numeric(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


if __name__ == "__main__":
    main()
