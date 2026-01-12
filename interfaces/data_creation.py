import tkinter as tk
from tkinter import Label, Canvas, Frame
import numpy as np
from PIL import Image, ImageTk
import torch
from torchvision.transforms.functional import to_pil_image
from utils.toolbuttons import ModelButton, SimButton, HighlightButton
from utils.process import Process
from utils.mssliders import SimSlider, ModelSlider
from base.iconbutton import IconButton  # Assuming trash icon uses IconButton
from utils.data_carousel import Carousel
from utils.checkdim import checkdim
from base.interface import Interface
from settings import *

class DataCreationInterface(Interface):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.idx = len(self.dataset)-1

        # White background
        self.config(bg="white")

        ######################################################################################################################

        # Create Top Bar Frame
        self.top_bar = tk.Frame(self, bg="white", height=50//SCALE)
        self.top_bar.pack(fill="x", padx=10//SCALE, pady=10//SCALE)

        # Back Button (iPhone-style left arrow)
        self.back_button = IconButton(self.top_bar, icon_path="icons/icon_back.png", bg="white", width=50//SCALE, height=66//SCALE)
        self.back_button.pack(side="left", padx=5//SCALE)

        # Bind Click Event
        self.back_button.bind("<Button-1>", lambda event: self.go_to_data_management())
        self.back_button.invoke = self.go_to_data_management

        # Back Text (Next to Arrow)
        self.back_label = Label(self.top_bar, text="Back to My Datasets", font=("Arial", 14//SCALE), bg="white", fg="black")
        self.back_label.pack(side="left", padx=5//SCALE)
        self.back_label.bind("<Button-1>", lambda event: self.go_to_data_management())  # Make text clickable

        # Header Label (Dynamic Dataset Name)
        self.header_label = Label(self, text=self.get_header_text(), font=("Arial", 26//SCALE, "bold"), bg="white", fg="black")
        self.header_label.pack(fill="x")

        ######################################################################################################################
        ##### ROW 1: CONTROL BUTTONS & SLIDERS #####

        self.row1_container = Frame(self, bg="white")
        self.row1_container.pack(fill="x", padx=20//SCALE, pady=10//SCALE)

        # Use grid for even spacing
        self.row1_container.columnconfigure((0, 1, 2, 3), weight=1)  # Ensures even column distribution

        # Column 1: Model Button (Fixed Collision Box)
        self.button_container1 = Frame(self.row1_container, bg="white")
        self.button_container1.grid(row=0, column=0, padx=20//SCALE, pady=5//SCALE, sticky="nsew")
        self.button_container1.pack_propagate(False)  # Prevents frame from resizing beyond the button

        self.mbutton = ModelButton(self.button_container1, self, save_func=self.save_result, refresh_func=self.refresh_image, bg="white")
        self.mbutton.pack()

        # Column 2: Sliders (SimSlider on top, ModelSlider below) - Centered Vertically
        self.slider_container = Frame(self.row1_container, bg="white")
        self.slider_container.grid(row=0, column=1, rowspan=2, padx=20//SCALE, pady=5//SCALE, sticky="nsew")
        self.slider_container.pack_propagate(False)

        # Spacer Above Sliders (Centers them vertically)
        self.spacer_top = Frame(self.slider_container, height=40, bg="white")
        self.spacer_top.pack()

        self.sim_slider = SimSlider(self.slider_container, width=500//SCALE)
        self.sim_slider.pack(fill="x", padx=10//SCALE, pady=5//SCALE)

        # **NEW: Spacer Between Sliders**
        self.spacer_middle = Frame(self.slider_container, height=15//SCALE, bg="white")  # Adjust height as needed
        self.spacer_middle.pack()

        self.model_slider = ModelSlider(self.slider_container, width=500//SCALE)
        self.model_slider.pack(fill="x", padx=10//SCALE, pady=5//SCALE)

        # Spacer Below Sliders (Centers them vertically)
        self.spacer_bottom = Frame(self.slider_container, height=20//SCALE, bg="white")
        self.spacer_bottom.pack()

        # Column 3: Sim Button & Highlight Button (Side-by-Side)
        self.button_container2 = Frame(self.row1_container, bg="white")
        self.button_container2.grid(row=0, column=2, padx=20//SCALE, pady=5//SCALE, sticky="nsew")

        self.button_inner_container2 = Frame(self.button_container2, bg="white")
        self.button_inner_container2.pack()

        # Highlight Button (Right)
        self.hbutton = HighlightButton(self.button_inner_container2, self, refresh_func=self.refresh_image, bg="white")
        self.hbutton.pack(side="right", padx=5//SCALE, pady=5//SCALE)

        # Sim Button (Left)
        self.sbutton = SimButton(self.button_inner_container2, self, refresh_func=self.refresh_image, bg="white")
        self.sbutton.pack(side="right", padx=5//SCALE, pady=5//SCALE)
        self.sbutton.add_slider(self.sim_slider)

        # Column 4: Trash Can Button (Fix Clickbox)
        self.button_container3 = Frame(self.row1_container, bg="white")
        self.button_container3.grid(row=0, column=3, padx=20//SCALE, pady=5//SCALE, sticky="nsew")

        # Spacer
        self.spacer_right = Frame(self.button_container3, width=150//SCALE, bg="white")
        self.spacer_right.pack(side="right", padx=5//SCALE)
        self.trash_button = IconButton(self.button_container3, command=self.delete_current_image, icon_path="icons/icon_trash.png", bg="white", width=120, height=120)
        self.trash_button.pack(side="right")

        # Column 5: Save Dataset button
        self.save_button = IconButton(self.button_container3, command=self.dataset.save, icon_path="icons/icon_save.png", bg="white", width=105, height=120)
        self.save_button.pack(side="right")

        # Bind to process
        self.funcdisplay = Process(self.mbutton, self.sbutton, self.hbutton, optimize=False)
        self.funcsave = Process(self.mbutton, self.sbutton, self.hbutton, modelonly=True, topil=False)

        ######################################################################################################################
        ##### ROW 2: TWO LARGE 4:3 BOXES #####

        self.box_width = int(BOXHEIGHT * ASPECT)
        self.box_height = BOXHEIGHT

        self.row2_container = Frame(self, bg="white", height=self.box_height)  # Set a fixed height
        self.row2_container.pack(fill="x", padx=20//SCALE, pady=10//SCALE)  
        self.row2_container.pack_propagate(False)  # Prevent expansion

        # Left Box (Main Image Display)
        self.left_box = Label(self.row2_container, bg="white", highlightthickness=1)
        self.left_box.pack(side="left", padx=10//SCALE, pady=10//SCALE, expand=True)

        # Right Box (Secondary Display)
        self.right_box = Label(self.row2_container, bg="white", highlightthickness=1)
        self.right_box.pack(side="left", padx=10//SCALE, pady=10//SCALE, expand=True)

        ######################################################################################################################
        ##### ROW 3: Carousel #####

        self.row3_container = Frame(self, bg="white", height=216//SCALE)  # Ensure height is set
        self.row3_container.pack(side="bottom", fill="x", padx=50//SCALE, pady=0)  # Keep it always visible
        self.row3_container.pack_propagate(False)  # Prevent resizing issues

        # Left Arrow
        diameter = 112//SCALE
        self.left_button = IconButton(
            self.row3_container,
            command=self.prev_image,
            icon_path="icons/icon_arrow_left.png",
            bg="white",
            width=diameter,
            height=diameter
        )
        self.left_button.pack(side="left", padx=5//SCALE, pady=5//SCALE)

        # Carousel (Fixing width to prevent overflow)
        self.carousel = Carousel(self.row3_container, ds=self.dataset)
        self.carousel.pack(side="left", expand=True, fill="both")  # Expands correctly
        self.dataset.carousel = self.carousel # intertwine with carousel: if dataset element is deleted, carousel will update its index accordingly

        # Right Arrow
        self.right_button = IconButton(
            self.row3_container,
            command=self.next_image,
            icon_path="icons/icon_arrow_right.png",
            bg="white",
            width=diameter,
            height=diameter
        )
        self.right_button.pack(side="right", padx=5, pady=5)

        # Initial image update
        if self.idx != -1: self.refresh_image()  

        # Standard Selector
        self.standard_order = [self.back_button, self.mbutton, self.sbutton, self.hbutton, self.trash_button, self.save_button, self.left_button, self.right_button]

        ######################################################################################################################

    def next_image(self):
        self.idx = (self.idx + 1) % len(self.dataset)
        self.refresh_image()

    def prev_image(self):
        self.idx = (self.idx - 1) % len(self.dataset)
        self.refresh_image()

    def refresh_image(self): # when switch to new image; loads in X and renders updated Y
        # Update call to carousel
        self.carousel.set_active_image(self.idx) 

        # Update big display (Left Box)
        x = self.dataset[self.idx][0] # CHW
        checkdim(x, 'CHW')
        # check dimension before model
        img = to_pil_image(x)
        img = img.resize((self.box_width, self.box_height))  # Standardized size
        imgtk = ImageTk.PhotoImage(img)
        self.left_box.imgtk = imgtk
        self.left_box.configure(image=imgtk)

        
        # Update small display (Right Box)
        self.funcdisplay.M.args = self.dataset[self.idx][2]
        y_show = self.funcdisplay(x)
        y_show = y_show.resize((self.box_width, self.box_height))  # Standardized size
        imgtk = ImageTk.PhotoImage(y_show)
        self.right_box.imgtk = imgtk
        self.right_box.configure(image=imgtk)
    
    def save_result(self): # generates Y and saves it
        x = self.dataset[self.idx][0] # CHW
        y_save = self.funcsave(x) 
        self.dataset.update(self.idx, x, y_save, self.mbutton.args)  # Save the result to the dataset
        print(f"[EVENT] Saved image at index {self.idx} with model settings: {self.mbutton.args}")

    def get_header_text(self):
        """ Returns formatted header text with the active dataset name. """
        return f"Editing {self.dataset.name}"
    
    def delete_current_image(self):
        """ Deletes the current image from the dataset. """
        if self.dataset:
            self.dataset.delete(self.idx)
            self.idx = (self.idx - 1) % len(self.dataset)
            self.carousel.load_images()
            self.refresh_image() # update the showing image

    ######################################################################################################################
    ##### Navigation #####

    def switchActiveDataset(self):
        super().switchActiveDataset()
        #self.__init__(self.parent, self.controller)  # Reinitialize the interface with the new dataset

    def onShow(self):
        self.idx = len(self.dataset)-1
        self.carousel.load_images()  # Reload images in the carousel and add any images that were added
        self.carousel.set_active_image(self.idx) # set active image to the last one (must be after load image due to possible new images)
        self.refresh_image() # update the showing image (newest image)
    
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
        if button == XBOX_X:
            self.mbutton.toggle_menu()
        if button == XBOX_Y:
            self.sbutton.toggle_menu()
        if button == XBOX_LB:
            self.prev_button()
        if button == XBOX_RB:
            self.next_button()
    
    def onHatChange(self, hat, value):
        if value == XBOX_DLEFT: self.prev_image()
        if value == XBOX_DRIGHT: self.next_image()
        if self.buttons_with_open_menus:
            if value == XBOX_DUP or value == XBOX_DDOWN: 
                self.buttons_with_open_menus[-1].xbox_dpad(value[1])
    
    def onAxisChange(self, axis, value):
        if axis == XBOX_AXIS_LY:
            if self.buttons_with_open_menus:
                self.buttons_with_open_menus[-1].xbox_joystick(value)
            else:
                self.xbox_joystick(value)
        if axis == XBOX_AXIS_LX:
            self.model_slider.xbox_joystick(value)
        if axis == XBOX_AXIS_RX:
            self.sim_slider.xbox_joystick(value)
        
                

