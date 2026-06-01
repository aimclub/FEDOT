from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
import numpy as np

if __name__ == "__main__":
    data = np.array(
        [
            [0.0, 10.0, "A",0.0],
            [1.0, 11.0, "B",1.0],
            [2.0, 12.0, "C",0.0],
            [3.0, 13.0, "A",2.0],
        ],
    )
    td = TensorDataCreator.create(data, backend_name="cpu")
    print(td.ready_fingerprint)