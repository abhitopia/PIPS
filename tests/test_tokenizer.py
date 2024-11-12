import pytest
import numpy as np
import torch
from pips.tokenizer import (
    Tokenizer, GridTokenizer, GridSerializer, ProgramTokenizer,
    ColorPermutationTokenizer, ArrayTransformTokenizer, ArcTokenizer,
    MODEL_INPUT, MODEL_OUTPUT
)
from pips.task import Example, ColorPermutation, ArrayTransform

@pytest.fixture
def basic_tokenizer():
    return Tokenizer(frozen=False)

@pytest.fixture
def grid_tokenizer():
    return GridTokenizer()

@pytest.fixture
def program_tokenizer():
    tokenizer = ProgramTokenizer()
    tokenizer.build(['test_program1', 'test_program2'])
    return tokenizer

@pytest.fixture
def arc_tokenizer():
    tokenizer = ArcTokenizer()
    examples = [Example(
        idx="test",
        task_id="task",
        dataset="test",
        input=np.array([[1]]),
        output=np.array([[1]]),
        program_id=prog,
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name,
    ) for prog in ['test_program1', 'test_program2']]
    tokenizer.build_program_tokenizer(examples)
    return tokenizer

@pytest.fixture
def sample_grid():
    return np.array([[1, 2], [3, 4]])

@pytest.fixture
def sample_example(sample_grid):
    return Example(
        idx="test_1",
        task_id="task_1",
        dataset="test",
        input=sample_grid,
        output=sample_grid,
        program_id="test_program1",
        color_perm=ColorPermutation.CPID.name,
        transform=ArrayTransform.IDENT.name
    )

@pytest.fixture
def color_perm_tokenizer():
    return ColorPermutationTokenizer()

@pytest.fixture
def array_transform_tokenizer():
    return ArrayTransformTokenizer()

class TestTokenizer:
    def test_add_token(self, basic_tokenizer):
        basic_tokenizer.add_token('test')
        assert 'test' in basic_tokenizer.token2idx
        assert basic_tokenizer.token2idx['test'] == 0
        assert basic_tokenizer.idx2token[0] == 'test'

    def test_frozen_tokenizer(self, basic_tokenizer):
        basic_tokenizer.frozen = True
        with pytest.raises(ValueError):
            basic_tokenizer.add_token('test')

    def test_encode_decode(self, basic_tokenizer):
        tokens = ['a', 'b', 'c']
        for token in tokens:
            basic_tokenizer.add_token(token)
        
        sequence = 'a b c'
        encoded = basic_tokenizer.encode(sequence)
        decoded = basic_tokenizer.decode(encoded)
        assert decoded == sequence

class TestGridTokenizer:
    def test_special_tokens(self, grid_tokenizer):
        assert grid_tokenizer.PAD_TOKEN == '<NOOP>'
        assert grid_tokenizer.BOS_TOKEN == '[['
        assert grid_tokenizer.EOS_TOKEN == ']]'
        assert grid_tokenizer.NEW_ROW_TOKEN == '['

    def test_encode_decode_grid(self, grid_tokenizer):
        grid_str = '[[ 1 2 ] [ 3 4 ]]'
        encoded = grid_tokenizer.encode(grid_str)
        decoded = grid_tokenizer.decode(encoded)
        assert decoded == grid_str

    def test_pad_token_removal(self, grid_tokenizer):
        sequence = f'{grid_tokenizer.PAD_TOKEN} 1 2 {grid_tokenizer.PAD_TOKEN}'
        encoded = grid_tokenizer.encode(sequence)
        decoded = grid_tokenizer.decode(encoded, remove_padding=True)
        assert grid_tokenizer.PAD_TOKEN not in decoded

class TestGridSerializer:
    def test_serialize_deserialize(self, sample_grid):
        serialized, indices = GridSerializer.serialize_array(sample_grid)
        deserialized = GridSerializer.deserialize_array(serialized)
        np.testing.assert_array_equal(sample_grid, deserialized)

    def test_indices_shape(self, sample_grid):
        _, indices = GridSerializer.serialize_array(sample_grid)
        assert len(indices) == sample_grid.shape[0] * (sample_grid.shape[1] + 2)

class TestArcTokenizer:
    def test_encode_decode_example(self, arc_tokenizer, sample_example):
        x, y = arc_tokenizer.encode(sample_example)
        decoded = arc_tokenizer.decode(x, y)
        
        assert decoded.idx == sample_example.idx
        assert decoded.task_id == sample_example.task_id
        assert decoded.program_id == sample_example.program_id
        assert decoded.color_perm == sample_example.color_perm
        assert decoded.transform == sample_example.transform
        np.testing.assert_array_equal(decoded.input, sample_example.input)
        np.testing.assert_array_equal(decoded.output, sample_example.output)

    def test_serialization(self, arc_tokenizer):
        data = arc_tokenizer.to_dict()
        loaded = ArcTokenizer.from_dict(data)
        assert arc_tokenizer == loaded

    def test_program_tokenizer_build(self, arc_tokenizer):
        # Reset the program tokenizer to be unfrozen before building
        arc_tokenizer.program_tokenizer.frozen = False
        
        programs = ['new_program1', 'new_program2']
        arc_tokenizer.build_program_tokenizer([Example(
            idx="test",
            task_id="task",
            dataset="test",
            input=np.array([[1]]),
            output=np.array([[1]]),
            program_id=prog,
            color_perm=ColorPermutation.CPID.name,
            transform=ArrayTransform.IDENT.name,
        ) for prog in programs])
        
        for program in programs:
            encoded = arc_tokenizer.program_tokenizer.encode(program)
            decoded = arc_tokenizer.program_tokenizer.decode(encoded)
            assert decoded == program

class TestModelIO:
    def test_model_input_squeeze_unsqueeze(self, arc_tokenizer, sample_example):
        x, _ = arc_tokenizer.encode(sample_example)
        x_unsqueezed = x.unsqueeze(0)
        x_squeezed = x_unsqueezed.squeeze(0)
        
        assert isinstance(x_unsqueezed, MODEL_INPUT)
        assert isinstance(x_squeezed, MODEL_INPUT)
        assert x_squeezed.grid.shape == x.grid.shape
        
    def test_model_output_squeeze_unsqueeze(self, arc_tokenizer, sample_example):
        _, y = arc_tokenizer.encode(sample_example)
        y_unsqueezed = y.unsqueeze(0)
        y_squeezed = y_unsqueezed.squeeze(0)
        
        assert isinstance(y_unsqueezed, MODEL_OUTPUT)
        assert isinstance(y_squeezed, MODEL_OUTPUT)
        assert y_squeezed.grid.shape == y.grid.shape 

class TestColorPermutationTokenizer:
    def test_encode_decode(self, color_perm_tokenizer):
        for perm in ColorPermutation:
            encoded = color_perm_tokenizer.encode(perm.name)
            decoded = color_perm_tokenizer.decode(encoded)
            assert decoded == perm.name
    
    def test_invalid_permutation(self, color_perm_tokenizer):
        with pytest.raises(KeyError):
            color_perm_tokenizer.encode("INVALID_PERM")
    
    def test_serialization(self, color_perm_tokenizer):
        data = color_perm_tokenizer.to_dict()
        loaded = ColorPermutationTokenizer.from_dict(data)
        assert color_perm_tokenizer == loaded

class TestArrayTransformTokenizer:
    def test_encode_decode(self, array_transform_tokenizer):
        for transform in ArrayTransform:
            encoded = array_transform_tokenizer.encode(transform.name)
            decoded = array_transform_tokenizer.decode(encoded)
            assert decoded == transform.name
    
    def test_invalid_transform(self, array_transform_tokenizer):
        with pytest.raises(KeyError):
            array_transform_tokenizer.encode("INVALID_TRANSFORM")
    
    def test_serialization(self, array_transform_tokenizer):
        data = array_transform_tokenizer.to_dict()
        loaded = ArrayTransformTokenizer.from_dict(data)
        assert array_transform_tokenizer == loaded