import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


file_path = "healthcare-dataset-stroke-data.csv"
dataframe = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "fedesoriano/stroke-prediction-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

def base_pp(): # use BMI for regression
    """
    Preprocesses the dataset for regression tasks. Returns features and target variable.
    If include_stroke is False, the 'stroke' feature will be dropped to prevent data leakage when predicting BMI.
    """
    # drop rows with missing BMI values, since we can't use them for regression
    df = dataframe.copy().dropna(subset=['bmi'])

    # remove the single "Other" gender row to keep gender binary
    df = df[df['gender'] != 'Other'].copy()

    # remove identifiers before feature preprocessing
    df = df.drop(columns=['id'])

    # make 'Residence_type' into 'residence_type' to be consistent with other features
    df = df.rename(columns={'Residence_type': 'residence_type'})

    # encode binary categorical features as 0/1
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['residence_type'] = df['residence_type'].map({'Rural': 0, 'Urban': 1})

    # one-hot encode the remaining categorical features
    df = df.join(
        pd.get_dummies(
            df[['work_type', 'smoking_status']],
            prefix=['work_type', 'smoking_status'],
            dtype=int,
        )
    ).drop(columns=['work_type', 'smoking_status'])

    return df

def preprocessing(*, scaling=True, pca = False, outlier_removal = False, feature_selection = []):
    df = base_pp()

    # Standardization scaling
    if scaling:
        scaler = StandardScaler()
        scale_cols = ['age', 'avg_glucose_level', 'bmi']
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        if outlier_removal:
            z_score_threshold = 3
            outlier_mask = (df[scale_cols].abs() <= z_score_threshold).all(axis=1)
            df = df.loc[outlier_mask].copy()


    # Dimensionality reduction using PCA
    if pca:
        target = df['stroke'].copy()
        features = df.drop(columns=['stroke'])
        pca_model = PCA(n_components=2)
        reduced_features = pca_model.fit_transform(features)
        df = pd.DataFrame(
            reduced_features,
            columns=['pc1', 'pc2'],
            index=df.index,
        )

        df['stroke'] = target

    # Feature selection / removal
    for feature in feature_selection:
        if feature in df.columns:
            df = df.drop(columns=[feature])
        else:
            print(f"{feature} is not a column name")

    return df


def main():
    df = preprocessing()



if __name__ == "__main__":
    main()
