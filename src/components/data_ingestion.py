import os
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration class.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    Data Ingestion class for handling the data ingestion process.
    """
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.
        """
        logging.info("Data Ingestion method starts")
        try:
            # Read the dataset from the URL
            df = pd.read_csv("notebook\competitive-data-science-predict-future-sales\sales_train.csv")
            logging.info("Dataset read as dataframe")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info("Raw data saved to CSV")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and Test split completed")

            # Save the training and testing sets to CSV files
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info("Train and Test datasets saved to CSV files")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()