"""
@author: mading
@license: (C) Copyright: LUCULENT Corporation Limited.
@contact: mading@luculent.net
@file: uuid.py
@time: 2023/11/6 11:39
@desc: 
"""
import uuid


def random_uuid() -> str:
    return str(uuid.uuid4().hex)