import numpy as np
import torch as t


def no_grad(func):
  def wrapper_nograd(*args, **kwargs):
    with t.no_grad():
      return func(*args, **kwargs)
  return wrapper_nograd
