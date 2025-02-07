import pytest
from pips.utils import hash_string, generate_friendly_name

def test_hash_string():
    # Test basic hashing
    test_str = "hello world"
    hash_8 = hash_string(test_str)
    hash_16 = hash_string(test_str, len=16)
    
    # Test length of output
    assert len(hash_8) == 8
    assert len(hash_16) == 16
    
    # Test consistency
    assert hash_string(test_str) == hash_string(test_str)
    assert hash_string(test_str, len=16) == hash_string(test_str, len=16)
    
    # Test different strings produce different hashes
    assert hash_string("hello") != hash_string("world")
    
    # Test empty string
    assert len(hash_string("")) == 8
    
    # Test with special characters
    special_str = "!@#$%^&*()"
    assert len(hash_string(special_str)) == 8
    
    # Test with unicode characters
    unicode_str = "Hello 世界"
    assert len(hash_string(unicode_str)) == 8

def test_hash_string_length_validation():
    # Test invalid lengths
    with pytest.raises(ValueError, match="Length must be positive"):
        hash_string("test", len=0)
    
    with pytest.raises(ValueError, match="Length must be positive"):
        hash_string("test", len=-1)
    
    with pytest.raises(ValueError, match="Length cannot be greater than 64"):
        hash_string("test", len=65)  # SHA-256 produces 64 characters
    
    # Test valid lengths
    assert len(hash_string("test", len=1)) == 1
    assert len(hash_string("test", len=64)) == 64

def test_generate_friendly_name():
    # Test basic name generation
    name = generate_friendly_name()
    
    # Test format: adjective-noun-number
    parts = name.split('-')
    assert len(parts) == 3
    
    # Test that the number is 3 digits
    assert len(parts[2]) == 3
    assert parts[2].isdigit()
    assert 0 <= int(parts[2]) <= 999
    
    # Test multiple generations are different
    names = {generate_friendly_name() for _ in range(100)}
    assert len(names) == 100  # All names should be unique
    
    # Test format consistency across multiple generations
    for _ in range(100):
        name = generate_friendly_name()
        parts = name.split('-')
        assert len(parts) == 3
        assert len(parts[2]) == 3
        assert parts[2].isdigit()
        assert 0 <= int(parts[2]) <= 999

def test_friendly_name_components():
    # Test that components are from predefined lists
    adjectives = {
        "happy", "clever", "swift", "bright", "eager", "gentle", "bold", "calm",
        "wise", "kind", "quick", "brave", "proud", "keen", "fair", "warm"
    }
    nouns = {
        "panda", "falcon", "dolphin", "tiger", "eagle", "wolf", "bear", "fox",
        "owl", "lion", "hawk", "deer", "seal", "whale", "lynx", "robin"
    }
    
    # Generate multiple names and check components
    for _ in range(50):
        name = generate_friendly_name()
        adj, noun, _ = name.split('-')
        assert adj in adjectives, f"Unknown adjective: {adj}"
        assert noun in nouns, f"Unknown noun: {noun}"

def test_friendly_name_deterministic():
    # Test that with the same random seed, we get the same name
    import random
    
    # Set seed and generate name
    random.seed(42)
    name1 = generate_friendly_name()
    
    # Reset seed and generate again
    random.seed(42)
    name2 = generate_friendly_name()
    
    # Names should be identical
    assert name1 == name2

def test_hash_string_different_types():
    # Test with different input types
    assert len(hash_string(str(123))) == 8
    assert len(hash_string(str(3.14))) == 8
    assert len(hash_string(str(True))) == 8
    assert len(hash_string(str(None))) == 8
    
    # Test with whitespace
    assert len(hash_string("  spaces  ")) == 8
    assert len(hash_string("\ttabs\t")) == 8
    assert len(hash_string("\nnewlines\n")) == 8
    
    # Test that different whitespace produces different hashes
    assert hash_string("test") != hash_string("test ")
    assert hash_string("test") != hash_string(" test")

def test_friendly_name_number_format():
    # Test that numbers are properly zero-padded
    for _ in range(100):
        name = generate_friendly_name()
        number = name.split('-')[2]
        
        # Check length
        assert len(number) == 3
        
        # Check zero-padding
        if int(number) < 100:
            assert number.startswith('0')
        if int(number) < 10:
            assert number.startswith('00') 