import os
import sys

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
import great_expectations as gx

def get_features_expectations(validator):
    ex1 = validator.expect_column_values_to_not_be_null(column="price", meta={"dimension": "Completeness"})
    ex2 = validator.expect_column_values_to_be_between(column="price", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex3 = validator.expect_column_values_to_be_of_type(column="price", type_="float64", meta={"dimension": "Datatype"})

    ex4 = validator.expect_column_values_to_not_be_null(column="item_seq_number", meta={"dimension": "Completeness"})
    ex5 = validator.expect_column_values_to_be_between(column="item_seq_number", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex6 = validator.expect_column_values_to_be_of_type(column="item_seq_number", type_="float64", meta={"dimension": "Datatype"})

    ex7 = validator.expect_column_values_to_not_be_null(column="image_top_1", meta={"dimension": "Completeness"})
    ex8 = validator.expect_column_values_to_be_between(column="image_top_1", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex9 = validator.expect_column_values_to_be_of_type(column="image_top_1", type_="float64", meta={"dimension": "Datatype"})

    ex10 = validator.expect_column_values_to_not_be_null(column="title_length", meta={"dimension": "Completeness"})
    ex11 = validator.expect_column_values_to_be_between(column="title_length", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex12 = validator.expect_column_values_to_be_of_type(column="title_length", type_="float64", meta={"dimension": "Datatype"})

    ex13 = validator.expect_column_values_to_not_be_null(column="description_length", meta={"dimension": "Completeness"})
    ex14 = validator.expect_column_values_to_be_between(column="description_length", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex15 = validator.expect_column_values_to_be_of_type(column="description_length", type_="float64", meta={"dimension": "Datatype"})

    ex16 = validator.expect_column_values_to_not_be_null(column="params_length", meta={"dimension": "Completeness"})
    ex17 = validator.expect_column_values_to_be_between(column="params_length", min_value=0, max_value=1, meta={"dimension": "Range"})
    ex18 = validator.expect_column_values_to_be_of_type(column="params_length", type_="float64", meta={"dimension": "Datatype"})

    ex19 = validator.expect_column_values_to_not_be_null(column="day_of_week_sin", meta={"dimension": "Completeness"})
    ex20 = validator.expect_column_values_to_be_between(column="day_of_week_sin", min_value=-1, max_value=1, meta={"dimension": "Range"})
    ex21 = validator.expect_column_values_to_be_of_type(column="day_of_week_sin", type_="float64", meta={"dimension": "Datatype"})

    ex22 = validator.expect_column_values_to_not_be_null(column="day_of_week_cos", meta={"dimension": "Completeness"})
    ex23 = validator.expect_column_values_to_be_between(column="day_of_week_cos", min_value=-1, max_value=1, meta={"dimension": "Range"})
    ex24 = validator.expect_column_values_to_be_of_type(column="day_of_week_cos", type_="float64", meta={"dimension": "Datatype"})

    ex25 = validator.expect_column_values_to_not_be_null(column="day_of_month_sin", meta={"dimension": "Completeness"})
    ex26 = validator.expect_column_values_to_be_between(column="day_of_month_sin", min_value=-1, max_value=1, meta={"dimension": "Range"})
    ex27 = validator.expect_column_values_to_be_of_type(column="day_of_month_sin", type_="float64", meta={"dimension": "Datatype"})

    ex28 = validator.expect_column_values_to_not_be_null(column="day_of_month_cos", meta={"dimension": "Completeness"})
    ex29 = validator.expect_column_values_to_be_between(column="day_of_month_cos", min_value=-1, max_value=1, meta={"dimension": "Range"})
    ex30 = validator.expect_column_values_to_be_of_type(column="day_of_month_cos", type_="float64", meta={"dimension": "Datatype"})

    validator.save_expectation_suite(discard_failed_expectations=False)
    expectations = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18,
                     ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26, ex27, ex28, ex29, ex30]
    return expectations

def validate_features(data: pd.DataFrame, version: str):
    with initialize(config_path="../configs", job_name="validate_features", version_base=None):
        cfg = compose(config_name="main")
        services_path = os.path.join(cfg.paths.root_path, 'services')

        context = gx.get_context(project_root_dir=services_path)
        ds = context.sources.add_or_update_pandas(name="pandas_features")
        da = ds.add_dataframe_asset(name="sample_preprocessed")

        batch_request = da.build_batch_request(dataframe=data)
        context.add_or_update_expectation_suite("features_data_validation")

        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name="features_data_validation"
        )

        expectations = get_features_expectations(validator)

        for i, expectation in enumerate(expectations):
            print(f"Expectation {i + 1}: {('Success' if expectation['success'] else 'Failed')}")
            assert expectation["success"], f"Expectation {i + 1} failed"
        
        checkpoint = context.add_or_update_checkpoint(
            name="features_data_validation",
            config_version=version,
            validations=[
                {
                    "batch_request": batch_request, 
                    "expectation_suite_name": "features_data_validation"
                }
            ]
        )

        checkpoint_result = checkpoint.run()
        if checkpoint_result["success"]:
            return data
    return None

