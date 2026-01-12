from base.slider import GraySlider

class ModelSlider(GraySlider):
    def __init__(self, parent, width=200, height=10, command=None, name="Model"):
        self.name = name
        super().__init__(parent, height=height, min_val=0, max_val=359, command=command, name=name)

class SimSlider(GraySlider):
    def __init__(self, parent, width=200, height=10, command=None, name="Sim"):
        self.name = name
        super().__init__(parent, height=height, min_val=0, max_val=1, command=command, name=name)