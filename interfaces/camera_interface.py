import tkinter as tk
from tkinter import Frame, Label, Button, Canvas, Scale
import cv2
from PIL import Image, ImageTk

import torch

from utils.toolbuttons import ModelButton, SimButton, HighlightButton
from utils.process import Process
from utils.mssliders import SimSlider
from utils.camera_feed import CameraFeedDisplay
from utils.checkdim import checkdim
from base.interface import Interface
from settings import *

# sensor display
import serial, colour
from colour.plotting import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class CameraInterface(Interface):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.toggle_state = False  # Global state toggle

        ######################################################################################################################

        ########## Window and Camera config ##########

        # Fixed left sidebar width (140px), Right sidebar width adjustable
        self.left_sidebar_width = 140//SCALE
        self.right_sidebar_width = 180//SCALE

        # Set the current size of the window
        total_width, total_height = self.controller.width, self.controller.height

        # Calculate the usable space (subtract left and right sidebar widths)
        usable_width = max(1, total_width - self.left_sidebar_width - self.right_sidebar_width)
        usable_height = max(1, total_height)  

        # Aspect ratio of the camera feed 
        aspect_ratio = ASPECT

        # PRIORITIZE HEIGHT: Set the camera height to fully match the window height
        self.camera_height = usable_height  
        self.camera_width = int(self.camera_height * aspect_ratio)  # Adjust width based on aspect ratio

        # If width exceeds available space, adjust based on width instead
        if self.camera_width > usable_width:
            self.camera_width = usable_width
            self.camera_height = int(self.camera_width / aspect_ratio)  # Maintain aspect ratio

        # Ensure dimensions are at least 1px to prevent OpenCV errors
        self.camera_width = max(1, self.camera_width)
        self.camera_height = max(1, self.camera_height)

        ######################################################################################################################

        ########## GUI ##########

        # Container Frame for Camera Feed and Controls
        self.container = Frame(self, bg="#000000")
        self.container.pack(fill="both", expand=True)

        # Left Sidebar (empty, black background)
        self.left_sidebar = Frame(self.container, width=self.left_sidebar_width, bg="#000000")
        self.left_sidebar.pack(side="left", fill="y")

        # Right Sidebar (for controls like the buttons)
        self.right_sidebar = Frame(self.container, width=self.right_sidebar_width, bg="#000000")
        self.right_sidebar.pack(side="right", fill="y")

        ########## Video Feed Label (Now Initialized Before update_camera_feed()) ##########
        self.cap = cv2.VideoCapture(0)
        self.cam_feed = CameraFeedDisplay(self.container, self.cap, self.camera_width, self.camera_height, bg="#000000")
        self.cam_feed.pack(side="left", fill="both", expand=True)  # Fully fills available space

        ########## Slider #########
        self.slider_frame = Frame(self.container)  # Transparent Black Background
        self.slider_frame.place(relx=0.5, rely=0.98, anchor="s", relwidth=0.6)  # Centered at Bottom

        self.sim_slider = SimSlider(
            self.slider_frame, width=WIDTH*0.6, name="Severity"
        )
        self.sim_slider.pack(fill="x", expand=True)  


        ######################################################################################################################

        ########## Control Frame ##########
        self.control_frame = Frame(self.right_sidebar, bg="#000000")
        self.control_frame.pack(fill="both", expand=True, pady=(80//SCALE, 60//SCALE))  # Expands within the right sidebar

        ########## View Toggle Button (TOP) ##########
        self.tcx, self.tcy = 50//SCALE, 50//SCALE
        self.ta = 2//SCALE

        self.toggle_button_canvas = Canvas(self.control_frame, width=100//SCALE, height=100//SCALE, bg="#000000", highlightthickness=0)
        self.toggle_button_canvas.pack(side="top", padx=self.right_sidebar_width/2, pady=20//SCALE)  # Keep original padx

        # Draw a white triangle (upward-pointing by default)
        self.objA = self.toggle_button_canvas.create_polygon(
            self.tcx - 16 * self.ta, self.tcy + 16 * self.ta, 
            self.tcx, self.tcy - 16 * self.ta, 
            self.tcx + 16 * self.ta, self.tcy + 16 * self.ta, 
            fill="white"
        )
        self.objB = None

        # Make the button clickable
        self.toggle_button_canvas.bind("<Button-1>", self.toggle_view)
        self.toggle_button_canvas.invoke = self.toggle_view # for ease for xbox

        ########## Middle Spacer (to push the Capture Button downward) ##########
        self.middle_spacer = Frame(self.control_frame, bg="#000000")
        self.middle_spacer.pack(expand=True)  # This pushes the capture button down

        ########## Capture Button (PERFECTLY CENTERED with Original `padx`) ##########
        self.capture_button_canvas = Canvas(self.control_frame, width=80//SCALE, height=80//SCALE, bg="#000000", highlightthickness=0)
        self.capture_button_canvas.pack(pady=10//SCALE, padx=self.right_sidebar_width/2)  # Keep original padx

        # Draw iPhone-style button
        self.capture_button_canvas.create_oval(0, 0, 80//SCALE, 80//SCALE, fill="white", outline="gray", width=3//SCALE)
        self.capture_button_canvas.create_oval(15//SCALE, 15//SCALE, 65//SCALE, 65//SCALE, fill="red", outline="")

        # Make the button clickable
        self.capture_button_canvas.bind("<Button-1>", self.take_picture)
        self.capture_button_canvas.invoke = self.take_picture # for ease for xbox

        ########## Bottom Spacer (to push the Capture Button upward) ##########
        self.bottom_spacer = Frame(self.control_frame, bg="#000000")
        self.bottom_spacer.pack(expand=True)  # Pushes the capture button up

        ########## Nav Button (BOTTOM) ##########
        self.navbutton = Button(
            self.control_frame,
            command=self.go_to_data_management,
            bg="white",  # White button
            relief="flat",
        )
        self.navbutton.pack(side="bottom", padx=self.right_sidebar_width/2, pady=20)  # Keep original padx

        # Set default img
        thumbnail = Image.open('icons/icon_nav.png').resize((120//SCALE, 120//SCALE), Image.Resampling.LANCZOS)
        self.navbutton.imgtk = ImageTk.PhotoImage(thumbnail)
        self.navbutton.configure(image=self.navbutton.imgtk)


        ######################################################################################################################

        ########## Toolbar Frame ##########
        self.toolbar_container = Frame(self.left_sidebar, bg="#000000", pady=60//SCALE)
        self.toolbar_container.pack(fill="both", expand=True, pady=20//SCALE)

        ########## Model Button ##########
        self.mbutton = ModelButton(self.toolbar_container, self)
        self.mbutton.pack(anchor="center", padx=self.left_sidebar_width/2, pady=18//SCALE) 

        ########## Middle Spacer ##########
        self.middle_spacer = Frame(self.toolbar_container)
        self.middle_spacer.pack(expand=True)  # This pushes the capture button down

        ########## Sim Button ##########
        self.sbutton = SimButton(self.toolbar_container, self)
        self.sbutton.add_slider(self.sim_slider)
        self.sbutton.pack(anchor="center", padx=self.left_sidebar_width/2, pady=18//SCALE)

        ########## Bottom Spacer ##########
        self.bottom_spacer = Frame(self.toolbar_container)
        self.bottom_spacer.pack(expand=True)  # Pushes the capture button up

        ########## Highlight Button ##########
        self.hbutton = HighlightButton(self.toolbar_container, self)
        self.hbutton.pack(anchor="center", padx=self.left_sidebar_width/2, pady=18//SCALE)

        ########## Initiate image pathway and camera feed ##########

        self.cam_feed.func = Process(self.mbutton, self.sbutton, self.hbutton, optimize=True)  # Set the function to process the image
        self.capture_func = Process(self.mbutton, self.sbutton, self.hbutton, modelonly=True, topil=False) 
        self.cam_feed.update_feed()

        # Standard selector
        self.standard_order = [self.mbutton, self.sbutton, self.hbutton, self.toggle_button_canvas, self.capture_button_canvas, self.navbutton]

        ########## Sensor config ##########
        try:
            self.arduino = serial.Serial('COM12', 9600, timeout=1)
            self.fig = plt.Figure(figsize=(3, 2.5))
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.cam_feed)  # Changed from self.container to self.right_sidebar
            self.canvas.get_tk_widget().place(relx=0.68, rely=0.02, anchor="nw")  # Position in top-right
            
            # Initial plot setup
            self.setup_plot()
            
            # Start the update loop
            self.update_plot()
        except serial.serialutil.SerialException:
            print("[ERROR] Arduino port COM12 could not be found.")
            print("[ERROR] Color sensor will not be used.")

    ######################################################################################################################
        

    ########## View Toggle Function ##########
    def toggle_view(self, event=None):
        """Toggle the global state and update the toggle button appearance."""
        self.toggle_state = not self.toggle_state  # Swap between True and False

        # Change triangle direction based on state
        self.toggle_button_canvas.delete(self.objA) 
        self.toggle_button_canvas.delete(self.objB) 
        if self.toggle_state:
            # Dual view (2 triangle)
            self.objA = self.toggle_button_canvas.create_polygon(self.tcx+8*self.ta,self.tcy-16*self.ta,self.tcx+24*self.ta,self.tcy+16*self.ta,self.tcx-8*self.ta,self.tcy+16*self.ta, fill="white")
            self.objB = self.toggle_button_canvas.create_polygon(self.tcx-10*self.ta,self.tcy+16*self.ta,self.tcx-24*self.ta,self.tcy+16*self.ta,self.tcx-8*self.ta,self.tcy-16*self.ta,self.tcx-2*self.ta,self.tcy, fill="white")
        else:
            # Single view (1 triangle)
            self.objA = self.toggle_button_canvas.create_polygon(self.tcx-16*self.ta, self.tcy+16*self.ta, self.tcx, self.tcy-16*self.ta, self.tcx+16*self.ta, self.tcy+16*self.ta, fill="white")
        self.cam_feed.set_mode("dual" if self.toggle_state else "mono")  # Update camera feed mode
        print(f"View Toggled: State is now {'DUAL' if self.toggle_state else 'MONO'}")

    ########## Picture Function ##########
    def take_picture(self, event=None):
        """Capture a frame from the camera and save it."""
        ret, frame = self.cap.read()  # Capture frame (BGR format)
        assert ret, "Failed to capture image from camera"
        
        # Preprocess
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(frame)  # Now x is a tensor (HWC)
        x = torch.permute(x, (2, 0, 1)) # HWC -> CHW
        y = self.capture_func(x)
        m_info = self.mbutton.args # model args
        self.dataset.append(x, y, m_info)

        # Create a thumbnail image for the nav button
        pil_image = Image.fromarray(frame, "RGB")
        
        # Use LANCZOS resampling instead of ANTIALIAS
        thumbnail = pil_image.resize((120, 120), Image.Resampling.LANCZOS)
        
        # Set this thumbnail as the nav button image
        self.navbutton.imgtk = ImageTk.PhotoImage(image=thumbnail)
        self.navbutton.configure(image=self.navbutton.imgtk)

        print(f"[EVENT] Captured Picture and saved to dataset: {self.dataset.name}")
    
    ########## Sensor Function ##########

    def setup_plot(self):
        # Configure colour style and draw CIE 1976 UCS diagram
        colour_style()
        plot_chromaticity_diagram_CIE1976UCS(standalone=False, axes=self.ax, spectral_locus_labels=None)
        self.ax.set_title('Ambient Color')
        self.ax.grid(True)

    def draw_point(self, r, g, b):
        # Clear previous points but keep the chromaticity diagram
        for artist in self.ax.collections + self.ax.lines:
            artist.remove()
        
        # Convert RGB to xy coordinates
        rgb_normalized = np.array([[r/255, g/255, b/255]])
        xyz = colour.sRGB_to_XYZ(rgb_normalized)
        xy = colour.XYZ_to_xy(xyz)
        
        # Plot the point
        str_d = f'RGB({r}, {g}, {b})'
        self.ax.scatter(xy[0][0], xy[0][1], color='red', s=100, label=str_d)
        self.ax.legend()
        
        # Redraw the canvas
        self.canvas.draw()

    def update_plot(self):
        try:
            data = self.arduino.readline().decode('utf-8').strip()
            if data:
                values = data.split(',')
                r = int(values[0])
                g = int(values[1])
                b = int(values[2])
                self.draw_point(r, g, b)
        except Exception as e:
            print(f"Error: {e}")
        
        # Schedule the next update
        self.after(1000, self.update_plot)  # Update every 100ms

    def on_closing(self):
        self.arduino.close()
        self.controller.destroy()

    ########## Nav Function ##########
    def onShow(self):
        self.cam_feed.on = True

    def onHide(self):
        self.cam_feed.on = False

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
        if self.buttons_with_open_menus:
            if value == XBOX_DUP or value == XBOX_DDOWN: 
                self.buttons_with_open_menus[-1].xbox_dpad(value[1])
    
    def onAxisChange(self, axis, value):
        if axis == XBOX_AXIS_LY:
            if self.buttons_with_open_menus:
                self.buttons_with_open_menus[-1].xbox_joystick(value)
            else:
                self.xbox_joystick(value)
        if axis == XBOX_AXIS_RX:
            self.sim_slider.xbox_joystick(value)
    
    def go_to_data_management(self):
        self.dataset.save()
        from interfaces.data_management import DataManagementInterface
        self.controller.show_frame(DataManagementInterface)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
