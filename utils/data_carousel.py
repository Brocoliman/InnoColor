import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from utils.checkdim import checkdim

class Carousel(tk.Frame):
    def __init__(self, parent, ds):
        super().__init__(parent)

        # Store dataset
        self.ds = ds
        self.dl = DataLoader(self.ds, batch_size=1, shuffle=False)

        # Canvas to hold images with a scrollbar
        self.canvas = tk.Canvas(self, height=225, bg="white", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        # Internal frame for images
        self.frame = tk.Frame(self.canvas, bg="white")
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        # Layout
        self.scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)

        # Image storage
        self.image_objects = []
        self.image_labels = []
        self.active_index = 0

        # Load images into carousel
        self.load_images()

        # Bind configure event to update scrollregion
        self.frame.bind("<Configure>", self.update_frame_size)

    def load_images(self):
        """Clears the canvas and reloads all images from the dataset."""
        for widget in self.frame.winfo_children():
            widget.destroy()

        self.image_objects.clear()
        self.image_labels.clear()

        for i, (x, y, m_info) in enumerate(self.dl):
            checkdim(x, 'BCHW')
            img = to_pil_image(x.squeeze(0))
            img = img.resize((180, 135))
            img_tk = ImageTk.PhotoImage(img)

            self.image_objects.append(img_tk)

            lbl = tk.Label(
                self.frame, image=img_tk, bg="white",
                borderwidth=2, relief="solid",
                highlightthickness=0, padx=5, pady=5
            )
            lbl.pack(side="left", padx=10, pady=20)
            self.image_labels.append(lbl)

        # Update immediately after loading
        self.update_idletasks()  # Ensure layout is updated
        self.update_frame_size()

        # Set initial active image
        self.set_active_image(self.active_index)

    def update_frame_size(self, event=None):
        """Ensures the frame is wide enough and updates scrollregion."""
        self.canvas.update_idletasks()  # Ensure frame size is current
        frame_width = self.frame.winfo_reqwidth()  # Requested width of frame
        canvas_width = self.canvas.winfo_width()

        # Update scrollregion to match the frame's full size
        self.canvas.configure(scrollregion=(0, 0, frame_width, 225))


    def set_active_image(self, index):
        """Centers the specified image and highlights it with a blue border."""
        if 0 <= index < len(self.dl):
            # Remove highlight from previous image
            self.image_labels[self.active_index].config(highlightthickness=0, highlightbackground="white")

            # Highlight new active image
            self.image_labels[index].config(highlightthickness=4, highlightbackground="blue")

            # Center the image
            self.center_image(index)

            self.active_index = index

    def center_image(self, index):
        """Moves the carousel so that the selected image is centered."""
        if not self.image_labels:
            return

        self.update_idletasks()  # Ensure layout is current

        selected_label = self.image_labels[index]
        frame_width = self.frame.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()

        # Calculate the offset to center the selected image
        x_offset = selected_label.winfo_x() + (selected_label.winfo_width() // 2)
        scroll_position = (x_offset - (canvas_width // 2)) / frame_width

        # Clamp scroll position between 0 and 1
        scroll_position = max(0, min(1.0, scroll_position))

        self.canvas.xview_moveto(scroll_position)