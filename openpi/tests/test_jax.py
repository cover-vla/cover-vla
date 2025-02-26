import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import sys

def check_jax_gpu():
    print("JAX version:", jax.__version__)
    print("Python version:", sys.version)
    
    devices = jax.devices()
    print("Available devices:", devices)
    
    gpu_available = any(dev.platform == 'gpu' for dev in devices)
    print("GPU available:", gpu_available)
    
    if not gpu_available:
        print("No GPU found. JAX will run on CPU.")
        return False
    
    print("Default backend:", jax.default_backend())
    
    try:
        x = jnp.arange(10)
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
        if gpu_devices:
            y = jax.device_put(x, gpu_devices[0])
            result = jnp.sum(y)
            print("Sum computed on GPU:", result)
        else:
            print("No GPU devices found.")
            return False
    except Exception as e:
        print("Error performing GPU operation:", str(e))
        return False
    
    @jit
    def test_function(x):
        return jnp.sum(jnp.tanh(x))
    
    grad_test = grad(test_function)
    
    try:
        x = jnp.arange(1000, dtype=jnp.float32)
        result = grad_test(x)
        print("Gradient computation successful")
    except Exception as e:
        print("Error in gradient computation:", str(e))
        return False
    
    print("JAX is successfully installed and configured for GPU use.")
    return True

if __name__ == "__main__":
    check_jax_gpu()