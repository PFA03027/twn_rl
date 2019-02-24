# -*- coding: utf-8 -*-
"""
TwoWheelMover
Created on Sun Jul 15 17:42:04 2018

@author: alpha

"""

from gym.envs.registration import register

register(
    id='twm_box_garden-v1',
    entry_point='myenv.env:TWN_BoxGardenEnv1',
)
register(
    id='twm_box_garden-v2',
    entry_point='myenv.env:TWN_BoxGardenEnv2',
)
register(
    id='twm_box_garden-v3',
    entry_point='myenv.env:TWN_BoxGardenEnv3',
)
register(
    id='twm_box_garden-v4',
    entry_point='myenv.env:TWN_BoxGardenEnv4',
)
register(
    id='twm_box_garden-v5',
    entry_point='myenv.env:TWN_BoxGardenEnv5',
)

