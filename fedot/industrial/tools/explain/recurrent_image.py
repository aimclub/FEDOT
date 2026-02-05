import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from fedot.industrial.core.operation.dummy.dummy_operation import init_input_data
from fedot.industrial.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor
from fedot.industrial.tools.loader import DataLoader


def plot_recurrence_matrix(
        dataset_name,
        save: bool = False,
        show: bool = True):
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    input_data = init_input_data(train_data[0], train_data[1])

    # strides = [1, 5, 10]
    # windows = [5, 10, 30]
    #
    # cls_dict = {f'class_{cls}': np.where(input_data.target == cls)[0] for cls in np.unique(input_data.target)}

    os.mkdir(f'{dataset_name}')
    for subset_name, subset in zip(['train', 'test'], [train_data, test_data]):

        # show_class_balance(subset[1])

        input_data = init_input_data(subset[0], subset[1])

        os.mkdir(f'{dataset_name}/{subset_name}')
        path_to_save = f'{dataset_name}/{subset_name}'

        for cls in np.unique(input_data.target):
            os.mkdir(f'{path_to_save}/{cls}')

        params = {'window_size': 10,
                  'stride': 1,
                  'image_mode': True}
        recur = RecurrenceExtractor(params)
        output = recur.transform(input_data)

        for idx, image in enumerate(output.predict):
            image_mode = 'L' if len(image.shape) == 2 else 'RGB'
            Image.fromarray(image, mode=image_mode).save(
                f'{path_to_save}/{output.target[idx][0]}/image_{idx}.png')

    # for cls in np.unique(input_data.target):
    #     fig, axs = plt.subplots(len(windows), len(strides), figsize=(20, 20))
    #     fig.suptitle(f'{dataset_name}, class {cls}', fontsize=40)
    #     for window_idx, window in enumerate(windows):
    #         for stride_idx, stride in enumerate(strides):
    #             params = {'window_size': window,
    #                       'stride': stride,
    #                       'image_mode': True}
    #             recur = RecurrenceExtractor(params)
    #
    #             random_sample_idx = np.random.choice(cls_dict[f'class_{cls}'].flatten(), 1)[0]
    #             mtrx, _ = recur.generate(input_data.features[random_sample_idx])
    #             axs[window_idx, stride_idx].imshow(mtrx)
    #             axs[window_idx, stride_idx].set_title(f'window: {window}, stride: {stride}', fontsize=30)
    #
    # plt.tight_layout()
    # if save:
    #     plt.savefig(f'{dataset_name}_recurrence_matrix.png')
    # if show:
    #     plt.show()


def show_class_balance(target):
    plt.hist(target, bins=len(np.unique(target)))
    plt.show()


if __name__ == '__main__':
    ds = 'Epilepsy'
    plot_recurrence_matrix(dataset_name=ds)
