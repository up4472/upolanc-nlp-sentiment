from logging import Logger
from typing import Any

from torch import cuda
from torch import device

__device = None
__random_state = 63130192

def init_device (logger : Logger) -> None :
	global __device

	if cuda.is_available() :
		__device = device('cuda')

		logger.info(f'Using CUDA cores for neural networks...\n')
	else :
		__device = device('cpu')

		logger.info('Using CPU cores for neural networks...\n')

def get_device () -> Any :
	global __device
	return __device

def get_random_state () -> int :
	return __random_state
