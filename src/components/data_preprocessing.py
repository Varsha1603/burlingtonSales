import os
import sys
import pandas as pd
import category_encoders as ce
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

@dataclass
class DataPreprocessingConfig:
    """
    Data Preprocessing Configuration class.
    """
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_train_data.csv")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_test_data.csv")


class DataPreprocessing:
    def __init__(self):
        self.config = DataPreprocessingConfig()
        self.label_encoder = LabelEncoder()
        self.ordinal_encoder = OrdinalEncoder()
        logging.info("DataPreprocessing class initialized.")

    def remove_outliers(self, df, columns):
        try:
            logging.info("Starting outlier removal process.")
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.where(df[col] < lower_bound, lower_bound,
                                   np.where(df[col] > upper_bound, upper_bound, df[col]))
            logging.info("Outliers capped using IQR method for columns: %s", columns)
            return df
        except Exception as e:
            logging.error("Error in remove_outliers: %s", str(e))
            raise CustomException(e, sys)

    def fill_null_values(self, df, numerical_cols, categorical_cols):
        try:
            logging.info("Filling null values in numerical columns: %s", numerical_cols)
            for col in numerical_cols:
                df[col].fillna(df[col].median(), inplace=True)

            logging.info("Filling null values in categorical columns: %s", categorical_cols)
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)

            logging.info("Null values filled successfully.")
            return df
        except Exception as e:
            logging.error("Error in fill_null_values: %s", str(e))
            raise CustomException(e, sys)

    def drop_redundant_features(self, df, drop_columns):
        try:
            logging.info("Dropping redundant features: %s", drop_columns)
            df.drop(columns=drop_columns, axis=1, inplace=True)
            logging.info("Dropped columns: %s", drop_columns)
            return df
        except Exception as e:
            logging.error("Error in drop_redundant_features: %s", str(e))
            raise CustomException(e, sys)

    def label_encode(self, df, cols):
        try:
            logging.info("Label encoding columns: %s", cols)
            for col in cols:
                df[col] = self.label_encoder.fit_transform(df[col])
            logging.info("Label encoding completed for columns: %s", cols)
            return df
        except Exception as e:
            logging.error("Error in label_encode: %s", str(e))
            raise CustomException(e, sys)

    def one_hot_encode(self, df, cols):
        try:
            logging.info("One-hot encoding columns: %s", cols)
            df = pd.get_dummies(df, columns=cols, drop_first=True)
            logging.info("One-hot encoding completed for columns: %s", cols)
            return df
        except Exception as e:
            logging.error("Error in one_hot_encode: %s", str(e))
            raise CustomException(e, sys)

    def target_encode(self, df, cols, target):
        try:
            logging.info("Starting target encoding for columns: %s", cols)
            encoder = ce.TargetEncoder(cols=cols)
            df[cols] = encoder.fit_transform(df[cols], df[target])
            logging.info("Target encoding completed for columns: %s", cols)
            return df
        except Exception as e:
            logging.error("Error in target_encode: %s", str(e))
            raise CustomException(e, sys)

    def frequency_encode(self, df, cols):
        try:
            logging.info("Starting frequency encoding for columns: %s", cols)
            for col in cols:
                freq = df[col].value_counts() / len(df)
                df[col] = df[col].map(freq)
            logging.info("Frequency encoding completed for columns: %s", cols)
            return df
        except Exception as e:
            logging.error("Error in frequency_encode: %s", str(e))
            raise CustomException(e, sys)

    def binary_encode(self, df, cols):
        try:
            logging.info("Starting binary encoding for columns: %s", cols)
            encoder = ce.BinaryEncoder(cols=cols)
            df = encoder.fit_transform(df)
            logging.info("Binary encoding completed for columns: %s", cols)
            return df
        except Exception as e:
            logging.error("Error in binary_encode: %s", str(e))
            raise CustomException(e, sys)

    def run_preprocessing(self, df):
        try:
            logging.info("Data preprocessing started.")
            
            numerical_cols = ['item_price', 'item_cnt_day']
            categorical_cols = []  # Update if categorical columns are present
            drop_columns = []      # Specify column names to drop, if any
            target_col = 'item_cnt_day'  # Example target column
            encode_cols = ['shop_id', 'item_id']

            # Read the train and test data
            df_train = pd.read_csv("artifacts/train.csv")
            df_test = pd.read_csv("artifacts/test.csv")

            logging.info("Data loaded successfully.")

            # Apply preprocessing steps
            df_train = self.remove_outliers(df_train, columns=numerical_cols)
            df_train = self.fill_null_values(df_train, numerical_cols, categorical_cols)
            df_train = self.drop_redundant_features(df_train, drop_columns)
            df_train = self.target_encode(df_train, cols=encode_cols, target=target_col)
            df_train = self.frequency_encode(df_train, cols=encode_cols)
            df_train = self.binary_encode(df_train, cols=encode_cols)

            df_test = self.remove_outliers(df_test, columns=numerical_cols)
            df_test = self.fill_null_values(df_test, numerical_cols, categorical_cols)
            df_test = self.drop_redundant_features(df_test, drop_columns)
            df_test = self.target_encode(df_test, cols=encode_cols, target=target_col)
            df_test = self.frequency_encode(df_test, cols=encode_cols)
            df_test = self.binary_encode(df_test, cols=encode_cols)

            # Save the training and testing sets to CSV files
            df_train.to_csv(self.config.transformed_train_data_path, index=False)
            df_test.to_csv(self.config.transformed_test_data_path, index=False)
            logging.info(" Transformed Train and Test datasets saved to CSV files")

            logging.info("Data preprocessing completed successfully.")
            return self.config.transformed_train_data_path, self.config.transformed_test_data_path
            

        except Exception as e:
            logging.error("Error in run_preprocessing: %s", str(e))
            raise CustomException(e, sys)
if __name__ == "__main__":
    data_preprocessor = DataPreprocessing()
    df_train, df_test = data_preprocessor.run_preprocessing(pd.DataFrame())
    print("Preprocessing completed successfully.")
