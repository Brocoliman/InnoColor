import tkinter as tk
from PIL import Image, ImageTk
from settings import *
import os

class IconButton(tk.Button):
    def __init__(self, parent, command=None, icon_path=None, **kwargs):
        """
        A reusable image-based button for Tkinter with a left-click context menu.
        """
        # File icon load
        if icon_path is not None:
            if not os.path.exists(icon_path):
                print(f"Error: Icon file not found! Path: {os.path.abspath(icon_path)}")
            else:
                # Open the image with transparency
                image = Image.open(icon_path).convert("RGBA")

                # Create a new white background image with the same size
                new_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
                new_image.paste(image, (0, 0), image)  # Paste with transparency

                # Resize the image while maintaining aspect ratio
                new_image = new_image.resize((kwargs['width']//SCALE, kwargs['height']//SCALE), Image.Resampling.LANCZOS)

                # Convert to a format Tkinter understands
                self.icon = ImageTk.PhotoImage(new_image)



        # Initialize the Button UI adapter with left-click to open/close context menu
        super().__init__(
            parent,
            image=self.icon,
            command=command,  # Left-click toggles the menu
            borderwidth=0, highlightthickness=0, highlightbackground="blue", highlightcolor="blue",
            **kwargs
        )

        # Keep reference to prevent garbage collection
        self.image = self.icon