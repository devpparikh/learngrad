import numpy as np
class Value:
  """
    stores a singal scalar value and its gradient
  """
  def __init__(self, data, children=(), op=''):
    
    # payload 
    self.data = data
    
    # children and op used to obtain this value 
    self._prev  = set(children)
    self._op = op

    #inital grad for a value = 0
    self.grad = 0
    self._backward = lambda : None   

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    out._backward = backward
    return out
  
  def __neg__(self):
    return self * -1

  def __sub__ (self, other):
    return self + (-other) 
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')

    def backward():
      self.grad += out.grad * (other * self.data ** (other -1))
    
    out._backward = backward
    return out
  
  def __truediv__(self, other):
    return self * other**-1

  def __rmul__(self, other):
    return self * other
  
  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return other + (-self)

  def __repr__(self): 
    return f"Value(data={self.data}, grad={self.grad})"
  
  def __rtruediv__(self, other):
        return other * self**-1

  def exp(self):
    out  = Value(np.exp(self.data), (self,), 'exp')

    def backward():
      self.grad += out.data * out.grad
    
    out._backward = backward
    return out
  

  def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1
    for val in reversed(topo):
      val._backward()
