import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
            categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipelines', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    import numpy as np

    def initiate_data_transformation(train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = DataTransformation().get_data_transformer_object()

            target_column_name = 'Item_Outlet_Sales'
            numerical_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Reshape target feature arrays to match the dimensions of input feature arrays
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)

            logging.info('Saving preprocessing object.')

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Add a column of zeros to the test data
            test_arr = np.concatenate((test_arr, np.zeros((len(test_arr), 1))), axis=1)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     train_path = "data/train.csv"
#     test_path = "data/test.csv"

#     train_arr, test_arr, preprocessor_obj_file_path = initiate_data_transformation(train_path, test_path)
