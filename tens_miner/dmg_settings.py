# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os.path

# Basic settings
application = 'dist/TensMiner.app'
appname = os.path.basename(application)

# Files to include in DMG
files = [application]

# Create symlink to Applications folder
symlinks = {'Applications': '/Applications'}

# Configure icon locations in DMG window
icon_locations = {
    appname:        (140, 120),
    'Applications': (500, 120)
}

# Window configuration
window_rect = ((100, 100), (640, 280))
default_view = 'icon-view'
show_icon_preview = False

# Icon view settings
arrange_by = None
grid_offset = (0, 0)
grid_spacing = 100
scroll_position = (0, 0)
label_pos = 'bottom'
text_size = 16
icon_size = 128

# Background
background = 'builtin-arrow'

# Include icon view settings
include_icon_view_settings = True
include_list_view_settings = False

# Icon view configuration
icon_view_settings = {
    'arrange_by':       arrange_by,
    'grid_offset':      grid_offset,
    'grid_spacing':     grid_spacing,
    'scroll_position':  scroll_position,
    'label_pos':       label_pos,
    'text_size':       text_size,
    'icon_size':       icon_size,
}