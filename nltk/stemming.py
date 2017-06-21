# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 01:14:22 2017

@author: aditya
"""

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

print stemmer.stem("responsiveness")