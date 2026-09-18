class WindowCutter:
    """
    Window cutter.
        input format: dict with "data" and "labels" fields
        output: the same dict but with additional windows_list and labels for it
    """

    def __init__(self, window_len, window_step):
        # super().__init__(name="Window Cutter", operation="window cutting")
        self.input_dict = None
        self.window_len = window_len
        self.window_step = window_step
        self.output_window_list = []

    def load_data(self, input_dict: dict) -> None:
        self.input_dict = input_dict

    def get_windows(self) -> list:
        return self.output_window_list

    def run(self) -> None:
        """
        Cut data to windows
        :return: none
        """
        self.output_window_list = self._cut_ts_to_windows(self.input_dict)

    def _cut_ts_to_windows(self, ts: dict) -> list:
        start_idx = 0
        end_idx = len(ts[list(ts.keys())[0]]) - self.window_len
        temp_windows_list = []
        for i in range(start_idx, end_idx, self.window_step):
            temp_window_dict = {}
            for key in ts.keys():
                temp_window = []
                for j in range(i, i + self.window_len):
                    temp_window.append(ts[key][j])
                temp_window_dict[key] = temp_window
        temp_windows_list.append(temp_window_dict)
        return temp_windows_list
