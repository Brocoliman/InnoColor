import tkinter as tk

class GraySlider(tk.Canvas):
    def __init__(self, parent, height=10, handle_radius=10, min_val=0, max_val=100, bg_color="#333333", handle_color="#bbbbbb", command=None, name="Slider"):
        super().__init__(parent, height=handle_radius * 2, bg=bg_color, highlightthickness=0)
        
        self.parent = parent  # Store parent reference
        self.height = height
        self.handle_radius = handle_radius
        self.min_val = min_val
        self.max_val = max_val
        self.command = command
        self.handle_color = handle_color
        self.name = name
        self.tooltip = None
        self.xbox_status = True
        self.slider_x = self.handle_radius  # Default starting position

        # Draw initial components (placeholder dimensions)
        self.track = self.create_line(0, handle_radius, 1, handle_radius, fill="#666666", width=height, capstyle="round")
        self.handle = self.create_oval(0, 0, handle_radius * 2, handle_radius * 2, fill=self.handle_color, outline="")

        # Bind resize event
        self.bind("<Configure>", self.on_resize)

        # Bind interaction events
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.show_tooltip)
        self.bind("<Leave>", self.hide_tooltip)
        self.bind("<ButtonRelease-1>", self.on_release)

    def on_resize(self, event=None):
        """ Adjusts slider width dynamically based on parent container. """
        new_width = self.winfo_width()
        if new_width > 10:  # Ensure valid size
            self.coords(self.track, self.handle_radius, self.handle_radius, new_width - self.handle_radius, self.handle_radius)
            self.update_handle_position(self.slider_x)  # Keep handle within bounds

    def on_drag(self, event):
        """ Move handle with mouse drag """
        new_x = max(self.handle_radius, min(event.x, self.winfo_width() - self.handle_radius))
        self.update_handle_position(new_x)

    def on_click(self, event):
        """ Move handle when clicked """
        new_x = max(self.handle_radius, min(event.x, self.winfo_width() - self.handle_radius))
        self.update_handle_position(new_x)

    def update_handle_position(self, new_x):
        """ Update handle position based on new x coordinate """
        self.slider_x = new_x
        self.coords(
            self.handle,
            self.slider_x - self.handle_radius, self.handle_radius - self.handle_radius,
            self.slider_x + self.handle_radius, self.handle_radius + self.handle_radius
        )

        # Move Tooltip with Handle
        if self.tooltip:
            self.tooltip.geometry(f"+{self.winfo_rootx() + self.slider_x}+{self.winfo_rooty() - 30}")
            self.tooltip_label.config(text=self.get_textvalue())

    def on_release(self, event):
        if self.command:
            # Calculate the value and call command function
            value = self.get_value()
            self.command(value)

    def get_value(self):
        """ Returns the current slider value based on position """
        if self.winfo_width() > 2 * self.handle_radius:
            value = self.min_val + (self.max_val - self.min_val) * ((self.slider_x - self.handle_radius) / (self.winfo_width() - 2 * self.handle_radius))
            return value
        return self.min_val  # Fallback if width is too small

    def get_textvalue(self):
        """ Returns the current slider value as a string """
        return f"{self.name}: {round(self.get_value(), 2)}"

    def show_tooltip(self, event=None):
        """ Show a custom tooltip near the slider handle """
        if not self.tooltip:
            self.tooltip = tk.Toplevel(self)
            self.tooltip.overrideredirect(True)  # No window borders
            self.tooltip.config(bg="black")

            self.tooltip_label = tk.Label(self.tooltip, text=self.get_textvalue(), bg="white", fg="black", font=("Arial", 12), padx=5, pady=2)
            self.tooltip_label.pack()

        # Ensure tooltip position is an integer
        tooltip_x = int(self.winfo_rootx() + self.slider_x)
        tooltip_y = int(self.winfo_rooty() - 30)

        self.tooltip.geometry(f"+{tooltip_x}+{tooltip_y}")

    def hide_tooltip(self, event=None):
        """ Hide the tooltip when the mouse leaves the slider """
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def xbox_joystick(self, value):
        """ Move handle with Xbox joystick in increments of 0.1 """
        step = (self.max_val - self.min_val) * 0.1  # Compute step as 10% of the range

        if self.xbox_status and value > 0.8:
            new_value = min(self.get_value() + step, self.max_val)  # Increment by 0.1
        elif self.xbox_status and value < -0.8:
            new_value = max(self.get_value() - step, self.min_val)  # Decrement by 0.1
        else:
            return  # No movement if joystick is centered

        # Convert value back to pixel position
        new_x = self.handle_radius + (new_value - self.min_val) * (self.winfo_width() - 2 * self.handle_radius) / (self.max_val - self.min_val)
        self.update_handle_position(new_x)  # Move handle

        if self.command:
            self.command(new_value)  # Call the command with the new value

