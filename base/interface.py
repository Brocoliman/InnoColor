from tkinter import Frame, Button, Canvas

class Interface(Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.parent = parent
        self.controller = controller
        self.dataset = self.controller.datasets[self.controller.active_ds]

        self.standard_order = [] # for Xbox controller standard selection
        self.standard_idx = 0 # current button idx

        self.buttons_with_open_menus = []
        self.xbox_ystatus = True  # Track the Y-axis status for standard order movement

    def onShow(self):
        pass
    
    def onHide(self):
        pass

    def onAxisChange(self, axis, value):
        # Context menu on: context menu selection (1) only
        # No context menu on: data management: standard selection (0)
        # No context menu on: data creation and camera: mslider and sslider (0) and (2) respectively
        pass

    def onButtonPress(self, button):
        # A: selection for context menu and standard selection (0)
        # B: going back / close context menu (1)
        # X: model button (3)
        # Y: sim button (4)
        # Bars: control standard button selection (6) (7)
        pass

    def onButtonRelease(self, button):
        pass   

    def onHatChange(self, hat, value):
        # left and right: data creation image selection
        # up and down: context menu selection
        pass

    def switchActiveDataset(self):
        self.dataset = self.controller.datasets[self.controller.active_ds]
    
    def highlight(self, new_button, prev_button):
        # Highlight the button when selected
        new_button.config(bg="#5996f7")
        prev_button.config(bg="white" if isinstance(prev_button, Button) else "black")

    def go_to_data_management(self):
        from interfaces.data_management import DataManagementInterface
        self.controller.show_frame(DataManagementInterface)
    
    def go_to_camera(self):
        from interfaces.camera_interface import CameraInterface
        self.controller.show_frame(CameraInterface)
    
    def go_to_data_creation(self):
        from interfaces.data_creation import DataCreationInterface
        self.controller.show_frame(DataCreationInterface)
    
    def next_button(self):
        prev_button = self.standard_order[self.standard_idx]
        self.standard_idx = (self.standard_idx + 1) % len(self.standard_order)
        new_button = self.standard_order[self.standard_idx]
        self.highlight(new_button, prev_button)
    
    def prev_button(self):
        prev_button = self.standard_order[self.standard_idx]
        self.standard_idx = (self.standard_idx - 1) % len(self.standard_order)
        new_button = self.standard_order[self.standard_idx]
        self.highlight(new_button, prev_button)
    
    def xbox_joystick(self, value): # for standard button order selection
        if self.xbox_ystatus and value > 0.8:
            self.next_button()
        if self.xbox_ystatus and value < -0.8:
            self.prev_button()
        if value < 0.2 and value > -0.2:
            self.xbox_ystatus = True