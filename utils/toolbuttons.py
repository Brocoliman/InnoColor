from base.toolbutton import ToolButton
import torch
import time

from cvdsim.fast import sim_image, change_setting
from utils.checkdim import checkdim
from model.hr.infer import generator as hr_x
from model.daltnet.infer import generator as daltnet_x
from model.distill.infer import generator as distill_x
from model.innocolor.infer import generator as innocolor_x

class ModelButton(ToolButton):
    """
    args:
    [0] get desired model name
    [1] get mslider value = 0 (mainly hue rotation additional adjustment)
    """
    def __init__(self, parent, interface, save_func=None, refresh_func=None, args=("none", 0), **kwargs): # default args is for camera interface
        super().__init__(parent, interface,
            args, "Model", icon_path='icons/icon_model2.png', **kwargs
        )
        self.options =  [
            ("   None   ", ("none", None)),
            ("   Hue Rotation   ", ("hr", None)),
            ("   Dalt-NET   ", ("dalt", None)),
            ("   InnoColor   ", ("innocolor", None)),
            ("   Distillation   ", ("distill", None)),
        ]
        self.save_res_func = save_func
        self.refresh_func = refresh_func
            
    def forward(self, *imgs): # input is already a tensor from process: BCHW
        img = imgs[0]
        checkdim(img, 'BCHW')
        modelname = self.args[0]
        value = self.args[1] if len(self.args) == 2 else 0
        if modelname=="none":
            pass
        elif modelname=="hr":
            img = hr_x(img)
        elif modelname=="dalt":
            img = daltnet_x(img)
        elif modelname=="innocolor":
            img = innocolor_x(img)
        elif modelname=="distill":
            img = distill_x(img)
        else:
            print('[ERROR] Model not found')
        checkdim(img, 'BCHW')
        return img
    
    def update(self, new_args):
        super().update(new_args)
        # There are different cases in CameraInterface and DataCreationInterface; we only need to implement 2nd
        # Camera Interface: current feed is marked; m_name is stored upon image capture (image capture incorporates m_name ALREADY)
        # Data Creation Interface: current image is marked; m_info are immediately sent into dataset (interface save_result is called)

        if self.save_res_func: self.save_res_func()
        if self.refresh_func: self.refresh_func()

    def add_slider(self, slider):
        rounding = 10
        slider.command = lambda lvl: self.update([None, int(rounding*lvl)/rounding])
    

class SimButton(ToolButton):
    def __init__(self, parent, interface, refresh_func=None, **kwargs):
        args = ("none", 0)
        super().__init__(
            parent, interface,
            args, "Sim", icon_path='icons/icon_sim2.png', **kwargs
        )
        self.options =  [
            ("   None   ", ("none", None)), 
            ("   Protan   ", ("p", None)), 
            ("   Deutan   ", ("d", None)),
            ("   Tritan   ", ("t", None))
        ]
        self.refresh_func = refresh_func
    
    def forward(self, *imgs):
        img = imgs[0]
        checkdim(img, 'BCHW')
        cvdtype = self.args[0] # p, d, t
        if cvdtype == "none": return img
        img = sim_image(img)
        checkdim(img, 'BCHW')
        return img
    
    def update(self, new_args):
        # Update Lookup table used
        super().update(new_args)
        if self.args[0] != "none": change_setting(self.args[0], self.args[1])
        if self.refresh_func: self.refresh_func()
    
    def add_slider(self, slider):
        rounding = 10
        slider.command = lambda lvl: self.update([None, int(rounding*lvl)/rounding])

class HighlightButton(ToolButton):
    def __init__(self, parent, interface, refresh_func=None, **kwargs):
        args = ("none",)
        super().__init__(
            parent, interface,
            args, "Highlight", icon_path='icons/icon_highlight.png',**kwargs
        )
        self.options =  [
            ("   None   ", ("none",)), 
            ("   Red   ", ("red",)), 
            ("   Green   ", ("green",)), 
            ("   Blue   ", ("blue",)), 
        ]
        self.refresh_func = refresh_func
    
    def forward(self, *imgs):
        target = imgs[0]  # image to overlap highlight annotations on
        src = imgs[1]  # original image to detect highlight regions from
        htype = self.args[0]
        checkdim(target, 'BCHW')
        if htype == "none": return target
        r,g,b=src[0]
        passives = 0.17
        base = 0.1
        if htype == "red":
            #mask = (r > 0.5) & (g < 0.3) & (b < 0.3) 
            mask = (r > base) & (r-2*g > passives) & (r-2*b > passives)
        if htype == "green":
            mask = (g > base) & (g - 2*b > passives) & (g - 2*r > passives)
        if htype == "blue":
            mask = (b > base) & (b - 2 * r> passives) & (b - 2 * g < passives)
        src = torch.where(mask.unsqueeze(0).expand_as(target), torch.ones_like(target), target)
        return src
    
    def update(self, new_args):
        # Update Lookup table used
        super().update(new_args)
        if self.refresh_func: self.refresh_func()