
import hashlib

def hash_string(my_string: str, len: int=8) -> str:
    hash_object = hashlib.sha256(my_string.encode())  # Encode the string to bytes
    hash_code = hash_object.hexdigest()  # Get the hexadecimal digest of the hash
    return hash_code[:len]  # Return the first `len` characters of the hash
