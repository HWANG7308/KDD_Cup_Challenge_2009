# -*- coding: utf-8 -*-
'''
logger

Author:
organize all code: S1802373
'''
import logging

def log():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('KDD2009')
    return logger
