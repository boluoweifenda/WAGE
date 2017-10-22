import sys
class Log(object):
  def __init__(self, *args):
    self.f = file(*args)
    sys.stdout = self

  def write(self, data):
    self.f.write(data)
    sys.__stdout__.write(data)
