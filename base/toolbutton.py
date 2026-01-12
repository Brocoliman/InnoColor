import tkinter as tk
from PIL import Image, ImageTk
from functools import partial
from settings import *

class ToolButton(tk.Button):
    def __init__(self, parent, interface, args, name, icon_path, **kwargs): # width=120, height=120, bg="#000000", borderwidth=0, highlightthickness=0, padx=0, pady=0
        """
        A reusable image-based button for Tkinter with a left-click context menu.
        """
        self.name = name

        # Restore w/h defaults:
        if 'width' not in kwargs:
            kwargs['width'] = 120
        if 'height' not in kwargs:
            kwargs['height'] = 120

        # Load and resize the image
        image = Image.open(icon_path)
        image = image.resize((kwargs['width']//SCALE, kwargs['height']//SCALE))  # Resize image dynamically
        self.icon = ImageTk.PhotoImage(image)
        self.args = args
        self.menu_open = False  # Tracks menu state
        self.menu_xbox_idx = 0 # tracks position of xbox scroll on menu
        self.xbox_ystatus = True # if XBox joystick has been to the middle, resets after every context menu switch

        # Placeholder menu options: list of tuples (name, (*arguments))
        self.options = None
        self.interface = interface # reference to the interface to control # menus open
        
        # Initialize the Button UI adapter with left-click to open/close context menu
        super().__init__(
            parent,
            image=self.icon,
            command=self.toggle_menu,  # Left-click toggles the menu
            borderwidth=0, highlightthickness=0, highlightbackground="blue", highlightcolor="blue",
            **kwargs,
        )

        # Keep reference to prevent garbage collection
        self.image = self.icon

    def toggle_menu(self):
        """ Toggle the context menu on left-click """
        if self.menu_open:
            self.close_menu()
        else:
            self.open_menu()

    def open_menu(self):
        """ Open the modernized context menu with uniform spacing """
        self.menu = tk.Menu(self, tearoff=0, bg="white", fg="black", bd=2, relief="solid", font=("Arial", 12))
        self.menu_commands = []

        # Set global options for menus
        self.option_add("*Menu.background", "white")
        self.option_add("*Menu.activeBackground", "#e0e0e0")

        for index, (label, new_args) in enumerate(self.options):
            func = partial(self.update, new_args)
            is_curr_option = new_args[0] == self.args[0]
            activebkg = MENU_CURR_HOVER if is_curr_option else MENU_HOVER
            bkg = MENU_HOVER if is_curr_option else "white"
            if is_curr_option: self.menu_xbox_idx = index
            self.menu_commands.append(func)
            self.menu.add_command(
                label=label,
                command=func,
                activebackground=activebkg,  # Light gray hover for active items
                activeforeground="black",
                background=bkg,
                compound="left"  # Ensures text alignment consistency
            )

        # Get button position for accurate menu placement
        x = self.winfo_rootx() + self.winfo_width()
        y = self.winfo_rooty()

        # Display the menu at the button's position
        self.menu.post(x, y)
        self.menu_open = True  # Track that the menu is open
        self.interface.buttons_with_open_menus.append(self)  # Add to the set of buttons with open menus

    def close_menu(self):
        """ Close the context menu """
        if self.menu_open:
            self.menu.unpost()
            self.menu_open = False
            self.interface.buttons_with_open_menus.remove(self)
    
    def menu_invoke(self):
        self.menu_commands[self.menu_xbox_idx]()

    def xbox_menu_control(self, value: int):
        prev_idx = self.menu_xbox_idx
        self.menu_xbox_idx = max(0, self.menu_xbox_idx + value)
        new_idx = self.menu_xbox_idx
        prev_idx_old_color = self.menu.entrycget(prev_idx, "activebackground")
        prev_idx_new_color = MENU_HOVER if prev_idx_old_color == MENU_CURR_HOVER else MENU_DEFAULT
        self.menu.entryconfig(prev_idx, background=prev_idx_new_color)
        new_idx_old_color = self.menu.entrycget(new_idx, "activebackground")
        new_idx_new_color = MENU_CURR_HOVER if new_idx_old_color == MENU_HOVER else MENU_HOVER
        self.menu.entryconfig(new_idx, background=new_idx_new_color)

    def xbox_joystick(self, value):
        if self.xbox_ystatus and value > 0.8:
            self.xbox_menu_control(1)
        if self.xbox_ystatus and value < -0.8:
            self.xbox_menu_control(-1)
        if value < 0.2 and value > -0.2:
            self.xbox_ystatus = True

    def xbox_dpad(self, value):
        if value == 1:
            self.xbox_menu_control(-1)
        if value == -1:
            self.xbox_menu_control(+1)
        

    # Update settings of button: args include all settings
    # Also refresh feature for Data Creation Interface
    def update(self, new_args):
        # Update the object arguments as part of process pipeline (Camera interface side)
        self.args = [new_args[i] if new_args[i] is not None else self.args[i] for i in range(len(new_args)) ]

        self.close_menu()  # Close menu after selection

        print(f"[EVENT] Updated args for {self.name}: {self.args}{' - Refresh' if self.refresh_func else ''}")  # Debugging

    # Call the function: imgs can take multiple images
    def __call__(self, *imgs):
        return self.forward(*imgs)

    def forward(self, *imgs): # in and out as tensor
        """ Placeholder function to be implemented in subclasses """
        pass
