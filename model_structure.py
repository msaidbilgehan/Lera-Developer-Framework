import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


def LSTM_For_Predictive_Manufacturing_Intelligence(input_shape, dense_size, xtrain, ytrain, xtest, ytest, epochs=250, batch_size=32, model_save_path=''):

    model = Sequential([
        LSTM(units=112, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=56, return_sequences=True),
        Dropout(0.1),
        LSTM(units=28, return_sequences=True),
        Dropout(0.1),
        LSTM(units=14),
        Dropout(0.05),
        Dense(units=dense_size)
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'], run_eagerly=True)
    # model.summary()

    history = model.fit(
        xtrain,
        ytrain,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(xtest, ytest),
        verbose=0
    )
    model.save("../predictive-manufacturing-intelligence/model/lstm_model.h5")
    return model

def show_Pred_vs_Actual_Simple(model, scaler, test_sequences, columns, target_dates='', threshold=5):

    predictions = model.predict(test_sequences)
    pred = predictions[0]

    pred_df = pd.DataFrame([pred], columns=columns)
    unnormalized_pred = np.round(scaler.inverse_transform(pred_df)).astype(int)
    unnormalized_pred[unnormalized_pred < threshold] = 0

    result_df = pd.DataFrame({
        'PCB Name': columns,
        'Predicted Count': unnormalized_pred[0]
    })
    # stdo(1, f"Predicted Data (Day {day_index+1}):\n{result_df}")
    result_df["Predicted Date"] = target_dates

    result_df = result_df.pivot_table(index="Predicted Date", columns="PCB Name", values="Predicted Count")
    result_df.columns.name = None
    result_df.index.name = "FullDateTime"
    result_df = result_df.reset_index()

    return result_df

def LGBM_For_Predictive_Manufacturing_Intelligence(df_combined, lgbm_test_dataset):
    features_lgbm = ["PCB_enc", "PCB_Component_enc", 'dayofmonth', 'month', 'year', 'hour', 'minute']
    # Ensure numeric types for required columns
    for col in ['dayofmonth', 'month', 'year']:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        lgbm_test_dataset[col] = pd.to_numeric(lgbm_test_dataset[col], errors='coerce')
    X_lgbm = df_combined[features_lgbm]
    y_lgbm = df_combined[["PCB_Component_enc", 'label']]

    # split_date_lgbm = pd.Timestamp('2025-08-6')
    # train_idx_lgbm = df_combined['FullDateTime'] # < split_date_lgbm
    # test_idx_lgbm = df_combined['FullDateTime'] # >= split_date_lgbm

    X_train = X_lgbm
    y_train = y_lgbm
    # X_test = X_lgbm[test_idx_lgbm]
    X_test = lgbm_test_dataset
    # y_test = y_lgbm[test_idx_lgbm]

    # test_dates = df_combined.loc[test_idx_lgbm, 'FullDateTime']

    lgbm_model = MultiOutputRegressor(LGBMRegressor())
    lgbm_model.fit(X_train, y_train)
    lgbm_y_pred = lgbm_model.predict(X_test)
    lgbm_y_pred = np.round(lgbm_y_pred).astype(int)

    # for i, column in enumerate(y_lgbm.columns):

    #     # r2 = r2_score(y_test.iloc[:, i], lgbm_y_pred[:, i])
    #     # mae = mean_absolute_error(y_test.iloc[:, i], lgbm_y_pred[:, i])
    #     # print(f"{column} - R2 Score: {r2:.4f}, MAE: {mae:.4f}")

    return lgbm_y_pred
