# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:24:01 2018

@author: alpha
"""

import multiprocessing as mp

class _internal_id_singleton():
    _singleton_instance_id = None
    
    def __init__( self ):
        pass
    
    @classmethod
    def get_new_id(cls):
        if cls._singleton_instance_id is None:
            cls._singleton_instance_id = mp.Value('i', 1)
        else:
            cls._singleton_instance_id.value += 1
        return cls._singleton_instance_id.value

    @classmethod
    def compare_id(cls, a, b):
        if (a is None) or (b is None):
            return False
        return (a == b)

