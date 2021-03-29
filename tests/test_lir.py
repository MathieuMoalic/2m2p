import pytest
import sys
sys.path.append("/home/mat/workfolder/paper1/lir")
from lir.lir import Lir 
import os
import mock

m = Lir("tests/test","tests/test_sim")
m.p
# print(os.getcwd())
def test_clean_path():
    assert m._clean_path('qwe') == os.getcwd()+'/qwe.h5'
    assert m._clean_path('qwe.h5') == os.getcwd()+'/qwe.h5'

def test_shape():
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(m.h5_path)
    assert m.shape('m') == (6, 1, 25, 25, 3)
    assert m.shape('table') == (7, 6)

os.remove(m.h5_path)