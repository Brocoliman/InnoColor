import tkinter as tk
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Set the working directory to the script's location

from interfaces.camera_interface import CameraInterface
from interfaces.data_management import DataManagementInterface
from interfaces.data_creation import DataCreationInterface
from dataset import InnoColorDataset
from settings import *

# For xbox adapter
from threading import Thread
import pygame
import time


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.width = WIDTH
        self.height = HEIGHT

        # Set the default window size to 1920x1080 (Full HD) and prevent resizing
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)  # Lock the window size
        self.title("InnoColor")

        self.initDatasets()
        self.initFrames()
        self.initXBox()  # Initialize Xbox controller

        # Show the first frame (CameraInterface) when the app starts
        self.show_frame(DataManagementInterface)

    def initFrames(self):
        # Load Frames
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)  # Makes the container expand
        self.frames = {}
        for F in (CameraInterface, DataManagementInterface, DataCreationInterface):
            frame = F(parent=self.container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")  # Stretch the frame to fill the container

        # Make the container components flexible and resizable
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self.active_frame = self.frames[CameraInterface]  # Set the default active frame
    
    def initDatasets(self):
        self.active_ds = 0
        self.datasets = [InnoColorDataset("DS1", 'ds1.pth'), InnoColorDataset("DS2", 'ds2.pth'), InnoColorDataset("DS3", 'ds3.pth')]
        self.datasets[self.active_ds].load() # initial actions (this needs to be done before any interface is loaded; data_creation.py relies on loaded dataset to start on correct idx)

    def switch_active_ds(self, idx):
        self.active_ds = idx
        self.datasets[self.active_ds].load()
        #self.frames[DataCreationInterface].switchActiveDataset()
        del self.frames[DataCreationInterface]
        self.frames[DataCreationInterface] = DataCreationInterface(self.container, self)
        self.frames[DataCreationInterface].grid(row=0, column=0, sticky="nsew")
        self.frames[CameraInterface].switchActiveDataset()
        self.frames[DataManagementInterface].switchActiveDataset()
        print(len(self.datasets[self.active_ds]))

    def initXBox(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[ERROR] No Xbox controller detected!")
            return
        self.xbox = pygame.joystick.Joystick(0)
        self.xbox.init()

        # Sep thread
        self.running = True
        self.thread = Thread(target=self.listen_to_controller, daemon=True)
        self.thread.start()

    def show_frame(self, cont):
        for frame in self.frames:
            if frame != cont:
                # Hide all frames except the one we want to show
                self.frames[frame].onHide()
        self.frames[cont].onShow()
        self.frames[cont].tkraise()  # Bring the selected frame to the front
        self.active_frame = self.frames[cont]
    
    def listen_to_controller(self):
        """
        button 0: A
        button 1: B
        button 3: X
        button 4: Y
        button 6: LB
        button 7: RB
        axis 0: left stick x
        axis 1: left stick y
        axis 2: right stick x
        axis 3: right stick y
        """
        last_axis = {}
        last_hat = {}
        last_button = set()
        while self.running:
            pygame.event.pump()  # Update internal event state without fetching all events

            # Check axes (sticks and triggers)
            for i in range(self.xbox.get_numaxes()):
                axis_value = round(self.xbox.get_axis(i), 2)
                if last_axis.get(i) != axis_value:
                    last_axis[i] = axis_value
                    self.active_frame.onAxisChange(i, axis_value)

            # Check buttons
            for i in range(self.xbox.get_numbuttons()):
                button_state = self.xbox.get_button(i)
                if button_state == 1 and i not in last_button:
                    self.active_frame.onButtonPress(i)
                    last_button.add(i)
                elif button_state == 0 and last_axis.get(f"button_{i}") == 1:
                    self.active_frame.onButtonRelease(i)
                    last_button.remove(i)
                last_axis[f"button_{i}"] = button_state

            # Check D-pad
            for i in range(self.xbox.get_numhats()):
                hat_value = self.xbox.get_hat(i)
                if last_hat.get(i) != hat_value:
                    last_hat[i] = hat_value
                    self.active_frame.onHatChange(i, hat_value)

            time.sleep(XBOX_LISTEN_DELAY)  # Add a small delay to reduce CPU usage

        self.xbox.quit()
        pygame.quit()


# Start the application
if __name__ == "__main__":
    app = App()
    app.mainloop()  # This starts the Tkinter event loop, making the app interactive


    
