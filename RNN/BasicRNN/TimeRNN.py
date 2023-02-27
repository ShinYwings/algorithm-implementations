import sys
sys.path.append("..")

import numpy as np
from RNN import RNN

class TimeRNN:
  def __init__(self, Wx, Wh, b, stateful=False):
    #초기화 메서드는 가중치, 편향, stateful이라는 boolean값을 인수로 받음
    #stateful=True : Time RNN계층이 은닉 상태를 유지한다.->아무리 긴 시계열 데이터라도 Time RNN계층의 순전파를 끊지 않고 전파한다.
    #stateful=False: Time RNN 계층은 은닉 상태를 '영행렬'로 초기화한다.상태가 없다.
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layers = None
    #layers : 다수의 RNN계층을 리스트로 저장하는 용도
    
    self.h, self.dh = None, None
    #h: forward() 메서드를 불렀을 때 마지막 RNN 계층의 은닉 상태를 저장
    #dh: backward()를 불렀을 때 하나 앞 블록의 은닉 상태의 기울기를 저장한다.
    self.stateful = stateful
    
  def set_state(self, h):
    self.h = h
    
  def reset_state(self):
    self.h = None

  def forward(self, xs):
    #아래로부터 입력 xs(T개 분량의 시계열 데이터를 하나로 모은 것)를 받는다.
    Wx, Wh, b = self.params
    N, T, D = xs.shape #N: 미니배치 크기, T: time_step 입력값  D: corpus 크기
    D, H = Wx.shape
    
    self.layers = []
    hs = np.empty((N, T, H), dtype='f')
    #출력값을 담을 그릇 hs를 준비한다.
    
    if not self.stateful or self.h is None:
      self.h = np.zeros((N, H), dtype='f')
      #h: RNN 계층의 은닉 상태. 
      #self.h=None: 처음 호출 시에는 원소가 모두 0인 영행렬로 초기화됨.
      #stateful=False: 항상 영행렬로 초기화
      
    for t in range(T):
      layer = RNN(*self.params)
      # *: 리스트의 원소들을 추출하여 메서드의 인수로 전달
      #self.params에 들어 있는 Wx, Wh, b를 추출하여 RNN 클래스의 __init__()메서드에 전달
      #RNN계층을 생성하여 인스턴스 변수 layers에 추가한다.
      self.h = layer.forward(xs[:,t,:], self.h)
      hs[:, t,:] = self.h
      self.layers.append(layer)
      
    return hs

  def backward(self, dhs):
    Wx, Wh, b = self.params
    N, T, H = dhs.shape
    D, H = Wx.shape
    
    dxs = np.empty((N, T, D), dtype='f')
    dh = 0
    grads = [0, 0, 0]
    for t in reversed(range(T)):
      layer = self.layers[t]
      dx, dh = layer.backward(dhs[:, t, :] + dh) #합산된 기울기
      #RNN계층의 순전파에서는 출력이 2개로 분기되는데 역전파에서는 
      #각 기울기가 합산되어 전해진다.
      dxs[:, t, :] = dx
      
      for i, grad in enumerate(layer.grads):
        grads[i] +=grad
        
    for i, grad in enumerate(grads):
      self.grads[i][...] = grad
    self.dh = dh
    
    return dxs