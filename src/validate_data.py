import sys

import hydra
import pandas as pd
from omegaconf import DictConfig
import great_expectations as gx

def get_expectations(validator):
    ex1 = validator.expect_column_values_to_not_be_null(column="region", meta={"dimension": "Completeness"})
    ex2 = validator.expect_column_unique_value_count_to_be_between(column="region", min_value=28, max_value=28,
                                                                   meta={"dimension": "Cardinality"})
    ex3 = validator.expect_column_values_to_be_of_type(column="region", type_="object", meta={"dimension": "Datatype"})
    ex4 = validator.expect_column_values_to_not_be_null(column="city", meta={"dimension": "Completeness"})
    ex5 = validator.expect_column_values_to_be_of_type(column="city", type_="object", meta={"dimension": "Datatype"})
    ex6 = validator.expect_column_values_to_not_be_null(column="parent_category_name",
                                                        meta={"dimension": "Completeness"})
    ex7 = validator.expect_column_unique_value_count_to_be_between(column="parent_category_name", min_value=9,
                                                                   max_value=9, meta={"dimension": "Cardinality"})
    ex8 = validator.expect_column_values_to_be_of_type(column="parent_category_name", type_="object",
                                                       meta={"dimension": "Datatype"})
    ex9 = validator.expect_column_values_to_not_be_null(column="category_name", meta={"dimension": "Completeness"})
    ex10 = validator.expect_column_unique_value_count_to_be_between(column="category_name", min_value=47, max_value=47,
                                                                    meta={"dimension": "Cardinality"})
    ex11 = validator.expect_column_values_to_be_of_type(column="category_name", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex12 = validator.expect_column_values_to_not_be_null(column="user_type", meta={"dimension": "Completeness"})
    ex13 = validator.expect_column_unique_value_count_to_be_between(column="user_type", min_value=3, max_value=3,
                                                                    meta={"dimension": "Cardinality"})
    ex14 = validator.expect_column_values_to_be_of_type(column="user_type", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex15 = validator.expect_column_values_to_be_of_type(column="price", type_="float64", meta={"dimension": "Datatype"})
    ex16 = validator.expect_column_values_to_be_between(column="price", min_value=0, meta={"dimension": "Validity"})
    ex17 = validator.expect_column_values_to_be_of_type(column="item_seq_number", type_="int64",
                                                        meta={"dimension": "Datatype"})
    ex18 = validator.expect_column_values_to_be_between(column="item_seq_number", min_value=0,
                                                        meta={"dimension": "Validity"})
    ex19 = validator.expect_column_values_to_be_of_type(column="image_top_1", type_="float64",
                                                        meta={"dimension": "Datatype"})
    ex20 = validator.expect_column_values_to_be_between(column="image_top_1", min_value=0,
                                                        meta={"dimension": "Validity"})
    ex21 = validator.expect_column_values_to_be_of_type(column="param_1", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex22 = validator.expect_column_values_to_be_of_type(column="param_2", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex23 = validator.expect_column_values_to_be_of_type(column="param_3", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex24 = validator.expect_column_values_to_not_be_null(column="title", meta={"dimension": "Completeness"})
    ex25 = validator.expect_column_values_to_be_of_type(column="title", type_="object", meta={"dimension": "Datatype"})
    ex26 = validator.expect_column_value_lengths_to_be_between(column="title", max_value=50,
                                                               meta={"dimension": "Validity"})
    ex27 = validator.expect_column_values_to_not_match_regex(column="title", regex=r"^\s*$",
                                                             meta={"dimension": "Validity"})
    ex28 = validator.expect_column_values_to_be_of_type(column="description", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex29 = validator.expect_column_values_to_be_of_type(column="deal_probability", type_="float64",
                                                        meta={"dimension": "Datatype"})
    ex30 = validator.expect_column_min_to_be_between(column="deal_probability", min_value=0,
                                                     meta={"dimension": "Validity"})
    ex31 = validator.expect_column_max_to_be_between(column="deal_probability", max_value=1,
                                                     meta={"dimension": "Validity"})
    ex32 = validator.expect_column_values_to_not_be_null(column="deal_probability", meta={"dimension": "Completeness"})
    ex33 = validator.expect_column_values_to_be_of_type(column="activation_date", type_="datetime64[ns]",
                                                        meta={"dimension": "Datatype"})
    ex34 = validator.expect_column_values_to_not_be_null(column="activation_date", meta={"dimension": "Completeness"})
    ex35 = validator.expect_column_values_to_be_of_type(column="image", type_="object", meta={"dimension": "Datatype"})
    ex36 = validator.expect_column_values_to_be_of_type(column="item_id", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex37 = validator.expect_column_values_to_be_unique(column="item_id", meta={"dimension": "Uniqueness"})
    ex38 = validator.expect_column_values_to_not_be_null(column="item_id", meta={"dimension": "Completeness"})
    ex39 = validator.expect_column_values_to_be_of_type(column="user_id", type_="object",
                                                        meta={"dimension": "Datatype"})
    ex40 = validator.expect_column_values_to_not_be_null(column="user_id", meta={"dimension": "Completeness"})
    # Save the expectation suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    expectations = [
        ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18, ex19, ex20,
        ex21, ex22, ex23, ex24, ex25, ex26, ex27, ex28, ex29, ex30, ex31, ex32, ex33, ex34, ex35, ex36, ex37, ex38,
        ex39, ex40,
    ]
    return expectations

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def validate_initial_dataset(cfg: DictConfig) -> bool:
    context = gx.get_context(project_root_dir="../services")
    df = pd.read_csv(f"{cfg.datasets.sample_output_dir}/{cfg.datasets.sample_filename}", parse_dates=["activation_date"])

    ds = context.sources.add_or_update_pandas(name="pandas_datasource")
    da = ds.add_dataframe_asset(name="sample")

    batch_request = da.build_batch_request(dataframe=df)
    batches = da.get_batch_list_from_batch_request(batch_request)

    context.add_or_update_expectation_suite("initial_data_validation")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="initial_data_validation"
    )

    expectations = get_expectations(validator)

    for i, expectation in enumerate(expectations):
        print(f"Expectation {i + 1}: {('Success' if expectation['success'] else 'Failed')}")

    checkpoint = context.add_or_update_checkpoint(
        name="initial_data_validation_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "initial_data_validation"
            }
        ]
    )

    checkpoint_result = checkpoint.run()

    if not checkpoint_result.success:
        print("Validataion has not succeeded")
        sys.exit(1)

    context.build_data_docs()
    context.open_data_docs()

    sys.exit(0)

if __name__ == "__main__":
    validate_initial_dataset()
