import hashlib
import random

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def hash_string(my_string: str, len: int=8) -> str:
    # Add length validation
    if len <= 0:
        raise ValueError("Length must be positive")
    if len > 64:  # SHA-256 produces 64 characters
        raise ValueError("Length cannot be greater than 64")
        
    hash_object = hashlib.sha256(my_string.encode())  # Encode the string to bytes
    hash_code = hash_object.hexdigest()  # Get the hexadecimal digest of the hash
    return hash_code[:len]  # Return the first `len` characters of the hash

def generate_friendly_name():
    adjectives = [
        "happy", "clever", "swift", "bright", "eager", "gentle", "bold", "calm", 
        "wise", "kind", "quick", "brave", "proud", "keen", "fair", "warm"
    ]
    nouns = [
        "panda", "falcon", "dolphin", "tiger", "eagle", "wolf", "bear", "fox",
        "owl", "lion", "hawk", "deer", "seal", "whale", "lynx", "robin"
    ]
    # Generate a random 3-digit number
    number = f"{random.randint(0, 999):03d}"
    
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{number}"
