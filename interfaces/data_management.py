import tkinter as tk
from tkinter import Label, Canvas, Frame
from base.interface import Interface
from base.iconbutton import IconButton
from functools import partial
from settings import *

class DataManagementInterface(Interface):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.config(bg="white")

        # --- Top Navigation Bar ---
        self.top_bar = Frame(self, bg="white", height=50)
        self.top_bar.pack(fill="x", padx=10, pady=10)

        # Back Button (iPhone-style left arrow)
        self.back_button = IconButton(self.top_bar, icon_path="icons/icon_back.png", bg="white", width=50, height=66)
        self.back_button.pack(side="left", padx=5)
        self.back_button.bind("<Button-1>", lambda event: self.go_to_camera())
        self.back_button.invoke = lambda: self.go_to_camera()

        # Back Label
        self.back_label = Label(self.top_bar, text="Back to Camera", font=("Arial", 14), bg="white", fg="black")
        self.back_label.pack(side="left", padx=5)
        self.back_label.bind("<Button-1>", lambda event: self.go_to_camera())

        # --- Header Label ---
        Label(self, text="Data Management", font=("Arial", 20), bg="white", fg="black").pack(pady=10, fill="x")

        # --- Dataset Buttons Area ---
        datasets_frame = Frame(self, bg="white")
        datasets_frame.pack(pady=10)

        square_size = 120  # Square button size in pixels
        num_datasets = 3

        self.dataset_buttons = []
        def go_and_switch(idx):
            self.go_to_data_creation()
            self.controller.switch_active_ds(idx)
        for i in range(num_datasets):
            # Create a fixed-size square container
            square_size = 200  # Ensure this is large enough
            square_frame = Frame(datasets_frame, width=square_size, height=square_size, bg="white", bd=0, relief="solid")
            square_frame.pack_propagate(False)  # Lock size
            square_frame.grid(row=0, column=i, padx=100, pady=10)

            # Create the dataset button
            btn = IconButton(square_frame, command=partial(go_and_switch, i), icon_path="icons/icon_folder.png", 
                            width=180, height=120, bg="white")
            btn.pack(side="top", pady=(10, 0))  # Button at top with padding

            # Create the label
            label_text = self.controller.datasets[i].name or "Unnamed"  # Fallback if name is empty
            label = Label(square_frame, text=label_text, font=("Arial", 12), bg="white", fg="black")
            label.pack(side="bottom", pady=(0, 10))  # Label at bottom with padding

            self.dataset_buttons.append(btn)

        self.standard_order = [self.back_button, *self.dataset_buttons]

    def get_header_text(self):
        """ Returns formatted header text with the active dataset name. """
        active_ds = self.controller.get_active_ds()
        return f"Editing {active_ds.name}"

    def onButtonPress(self, button):
        if button == XBOX_A:
            if self.buttons_with_open_menus:
                self.buttons_with_open_menus[-1].menu_invoke()
            else:
                self.standard_order[self.standard_idx].invoke()
        if button == XBOX_B:
            if self.buttons_with_open_menus:
                for b in self.buttons_with_open_menus.copy():
                    b.close_menu()
            else:
                self.go_to_data_management()
        if button == XBOX_LB:
            self.prev_button()
        if button == XBOX_RB:
            self.next_button()

    def onAxisChange(self, axis, value):
        if axis == 1:
            self.xbox_joystick(value)
