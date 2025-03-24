import torch
from torch.utils._pytree import register_pytree_node

print(f"Using torch version: {torch.__version__}")
print(f"Torch location: {torch.__file__}")

# Define dummy functions for testing
def flatten_fn(x): return [x], None
def unflatten_fn(_, children): return children[0]

# Test registration without flatten_with_keys_fn
try:
    register_pytree_node(
        type("DummyClass", (), {}),
        flatten_fn,
        unflatten_fn
    )
    print("Successfully registered pytree node")
except TypeError as e:
    print(f"TypeError: {e}")