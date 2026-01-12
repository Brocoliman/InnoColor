import threading
import time
import cv2
import torch
from PIL import Image, ImageTk
from torchvision.transforms.functional import to_pil_image
import tkinter as tk
from tkinter import Frame, Label
from settings import *

class CameraFeedDisplay(Frame):
    def __init__(self, parent, cap, cw, ch, **kwargs):
        super().__init__(parent, **kwargs)
        self.cap = cap
        self.camera_width = cw
        self.camera_height = ch
        self.update_interval = 1000 // 24  # 24 FPS
        self.on = True
        self.func = None  # External processing function
        self.mode = "mono"  # Default mode

        # UI Setup
        self.mono_label = tk.Label(self, bg="black")
        self.mono_label.grid(row=0, column=0, sticky="nsew")
        
        # Dual View (Hidden initially)
        self.left_label = tk.Label(self, bg="black")
        self.right_label = tk.Label(self, bg="black")

        # Camera thread
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update_camera, daemon=True)
        self.thread.start()
        self.update_feed()

    def set_mode(self, mode):
        """ Switch between mono and dual view """
        self.mode = mode
        if mode == "dual":
            self.mono_label.grid_remove()
            self.left_label.grid(row=0, column=0, sticky="nsew")
            self.right_label.grid(row=0, column=1, sticky="nsew")
            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=1)
        else:
            self.left_label.grid_remove()
            self.right_label.grid_remove()
            self.mono_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
            self.columnconfigure(1, weight=0)

    def toggle_view(self):
        """ Toggle between mono and dual view modes """
        if self.mode == "mono":
            self.set_mode("dual")
        else:
            self.set_mode("mono")

    def update_camera(self):
        """ Background thread to capture camera frames """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def update_feed(self):
        """ Tkinter UI update loop """
        if self.on:
            with self.lock:
                frame = self.frame if self.frame is not None else None

            if frame is not None:
                # Pre-process frame once
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.camera_width, self.camera_height))

                if self.mode == "mono":
                    # Mono view: Convert directly to PIL and display
                    img = Image.fromarray(frame)
                    if self.func is not None:
                        x = torch.from_numpy(frame).permute(2, 0, 1)
                        start_time = time.time()
                        result = self.func(x)
                        elapsed = time.time() - start_time
                        if elapsed <= MAX_FRAME_TIME:  # Only use result if under 30ms
                            img = result
                            if not isinstance(img, Image.Image):
                                img = to_pil_image(img.detach().cpu())
                        # Else, img remains the unprocessed frame
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.mono_label.imgtk = imgtk
                    self.mono_label.configure(image=imgtk)
                else:
                    # Dual view: Optimize by processing frame once
                    img = Image.fromarray(frame)
                    if self.func is not None:
                        x = torch.from_numpy(frame).permute(2, 0, 1)
                        start_time = time.time()
                        result = self.func(x)
                        elapsed = time.time() - start_time
                        if elapsed <= 0.03:  # Only use result if under 30ms
                            img = result
                            if not isinstance(img, Image.Image):
                                img = to_pil_image(img.detach().cpu())
                        # Else, img remains the unprocessed frame

                    # Calculate offset for cropping
                    offset = self.camera_width // 20

                    # Crop for left and right eyes without resizing
                    left_img = img.crop((offset, 0, self.camera_width - offset, self.camera_height))
                    right_img = img.crop((0, 0, self.camera_width - 2 * offset, self.camera_height))

                    # Convert to PhotoImage without resizing
                    left_imgtk = ImageTk.PhotoImage(image=left_img)
                    right_imgtk = ImageTk.PhotoImage(image=right_img)

                    self.left_label.imgtk = left_imgtk
                    self.left_label.configure(image=left_imgtk)
                    self.right_label.imgtk = right_imgtk
                    self.right_label.configure(image=right_imgtk)

        self.after(self.update_interval, self.update_feed)