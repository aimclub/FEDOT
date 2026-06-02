from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
import numpy as np

if __name__ == "__main__":
    data = np.array(
        [
            [0.0, 10.0, "A",0.0],
            [1.0, 10.0, "B",1.0],
            [2.0, np.nan, "C",0.0],
            [3.0, 13.0, "A",2.0],
        ], dtype=object
    )
    td = TensorDataCreator.create(data, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td, {PreprocessingStepEnum.imputation: None})

# if __name__ == "__main__":
#     X = np.array([
#         [1, 2, "A", 3],
#         [4, np.nan, "B", 6],
#         [7, 8, "C", 9]
#     ], dtype=object)

#     td = TensorDataCreator.create(X, backend_name="cpu")

#     strategy = {
#         PreprocessingStepEnum.scaling: None,
#         PreprocessingStepEnum.imputation: [{
#             "method": ImputationMethodEnum.constant,
#             "features_idx": [1],
#             "step_args": {"constant": 3}
#         }],
#     }
#     service = OptionalTabularService()
#     preprocessed_data = service.fit_transform(td, strategy)