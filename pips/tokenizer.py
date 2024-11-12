import re
from typing import Dict, List, NamedTuple, Optional, Tuple
import json
import numpy as np
import torch
from .task import ArrayTransform, ColorPermutation, Example


class Tokenizer:
    def __init__(self, token2idx=None, idx2token=None, frozen=True) -> None:
        self.token2idx = token2idx if token2idx is not None else {}
        self.idx2token = idx2token if idx2token is not None else {}
        self.frozen = frozen
    
    def add_token(self, token):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def encode(self, sequence: str) -> List[int]:
        sequence = sequence.split(' ')
        return [self.token2idx[token] for token in sequence]

    def decode(self, sequence, remove_padding=True):
        tokens = [self.idx2token[idx] for idx in sequence]
        return ' '.join(tokens)
    
    def to_dict(self) -> Dict:
        return {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'frozen': self.frozen
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.token2idx = data['token2idx']
        obj.idx2token = data['idx2token']
        obj.frozen = data['frozen']
        return obj
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, Tokenizer), 'value must be an instance of Tokenizer'
        return self.token2idx == value.token2idx and self.idx2token == value.idx2token

    def __len__(self):
        return len(self.token2idx)
    

class GridTokenizer(Tokenizer):
    def __init__(self):
        self.PAD_TOKEN = '<NOOP>'
        self.BOS_TOKEN = '[['
        self.EOS_TOKEN = ']]'
        self.NEW_ROW_TOKEN = '['

        tokens = [self.PAD_TOKEN, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', ']', '[', ']]']
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)
        self.PAD_IDX = self.token2idx[self.PAD_TOKEN]
        self.BOS_IDX = self.token2idx[self.BOS_TOKEN]
        self.EOS_IDX = self.token2idx[self.EOS_TOKEN]
        self.NEW_ROW_IDX = self.token2idx[self.NEW_ROW_TOKEN]
        assert self.PAD_IDX == 0

    def decode(self, sequence, remove_padding=True):
        tokens = super().decode(sequence)
        if remove_padding:
            tokens = [token for token in tokens.split(' ') if token != self.PAD_TOKEN]
            return ' '.join(tokens)
    
        return tokens

class GridSerializer:
    @staticmethod
    def serialize_array(array: np.ndarray) -> str:
        list_of_lists = array.tolist()
        array_str = str(list_of_lists)
        array_str = array_str.replace('],' , ' ],').replace('[[', '[[ ').replace(']]', ' ]]').replace(' [', ' [ ').replace(',', '')
        num_tokens = len(array_str.split())
        assert num_tokens == array.shape[0] * (array.shape[1] + 2)
        indices = GridSerializer.indices(array)
        assert num_tokens == len(indices)
        return array_str, indices

    @staticmethod
    def indices(array: np.ndarray) -> List[Tuple[int, int]]:
        height, width = array.shape
        indices = np.indices((height, width + 2)).transpose(1, 2, 0)
        indices = indices.reshape(height*(width+2), 2)
        indices = [tuple(row) for row in indices]
        return indices

    @staticmethod
    def deserialize_array(array_str: str) -> np.ndarray:
        pattern = r'(?<=\d) (?=\d)'
        replacement = ', '
        result = re.sub(pattern, replacement, array_str)
        result = result.replace('] [', '], [')
        result = json.loads(result)
        return np.array(result)


class ProgramTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(frozen=False)

    def build(self, tokens: List[str]):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        for token in tokens:
            for t in token.strip().split(' '):
                if len(t) == 1:
                    print(f'Adding token: {token}')
                self.add_token(t)
        self.frozen = True


class ColorPermutationTokenizer(Tokenizer):
    def __init__(self):
        tokens = [cp.name for cp in list(ColorPermutation)]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)


class ArrayTransformTokenizer(Tokenizer):
    def __init__(self):
        tokens = [at.name for at in list(ArrayTransform)]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)


class MODEL_INPUT(NamedTuple):
    color_permutation: torch.Tensor
    array_transform: torch.Tensor
    program: torch.Tensor
    grid: torch.Tensor
    grid_indices: torch.Tensor
    meta: Optional[List[Dict[str, str]]]

    def unsqueeze(self, dim: int):
        return MODEL_INPUT(
            color_permutation=self.color_permutation.unsqueeze(dim),
            array_transform=self.array_transform.unsqueeze(dim),
            program=self.program.unsqueeze(dim),
            grid=self.grid.unsqueeze(dim),
            grid_indices=self.grid_indices.unsqueeze(dim),
            meta=self.meta
        )
    
    def squeeze(self, dim: int):
        return MODEL_INPUT(
            color_permutation=self.color_permutation.squeeze(dim),
            array_transform=self.array_transform.squeeze(dim),
            program=self.program.squeeze(dim),
            grid=self.grid.squeeze(dim),
            grid_indices=self.grid_indices.squeeze(dim),
            meta=self.meta
        )

class MODEL_OUTPUT(NamedTuple):
    grid: torch.Tensor
    grid_indices: torch.Tensor
    target_grid: Optional[torch.Tensor]
    
    def unsqueeze(self, dim: int):
        return MODEL_OUTPUT(
            grid=self.grid.unsqueeze(dim),
            grid_indices=self.grid_indices.unsqueeze(dim),
            target_grid=self.target_grid.unsqueeze(dim) if self.target_grid is not None else None
        )
    
    def squeeze(self, dim: int):
        return MODEL_OUTPUT(
            grid=self.grid.squeeze(dim),
            grid_indices=self.grid_indices.squeeze(dim),
            target_grid=self.target_grid.squeeze(dim) if self.target_grid is not None else None
        )


class ArcTokenizer:
    def __init__(self) -> None:
        self.grid_tokenizer = GridTokenizer()
        self.program_tokenizer = ProgramTokenizer()
        self.color_permutation_tokenizer = ColorPermutationTokenizer()
        self.array_transform_tokenizer = ArrayTransformTokenizer()

    def encode(self, example: Example) -> Tuple[MODEL_INPUT, MODEL_OUTPUT]:
        input_grid_ser, input_indices = GridSerializer.serialize_array(example.input.array)
        input_grid_encoded = self.grid_tokenizer.encode(input_grid_ser)
        output_grid_ser, output_indices = GridSerializer.serialize_array(example.output.array)
        output_grid_encoded = self.grid_tokenizer.encode(output_grid_ser)
        program_encoded = self.program_tokenizer.encode(example.program_id)
        color_permutation_encoded = self.color_permutation_tokenizer.encode(example.color_perm)
        array_transform_encoded = self.array_transform_tokenizer.encode(example.transform)

        x = MODEL_INPUT(
            color_permutation = torch.tensor(color_permutation_encoded, dtype=torch.long),
            array_transform = torch.tensor(array_transform_encoded, dtype=torch.long),
            program = torch.tensor(program_encoded, dtype=torch.long),
            grid = torch.tensor(input_grid_encoded, dtype=torch.long),
            grid_indices = torch.tensor(input_indices, dtype=torch.long),
            meta={'task_id': example.task_id,
                  'example_id': example.idx, 
                  'complexity': example.complexity,
                  'dataset': example.dataset}
        )
        y = MODEL_OUTPUT(
            grid = torch.tensor(output_grid_encoded, dtype=torch.long),
            grid_indices = torch.tensor(output_indices, dtype=torch.long),
            target_grid = torch.tensor(output_grid_encoded[:-1] + [self.grid_tokenizer.PAD_IDX], dtype=torch.long)
            )
        return x, y

    def decode(self, x: MODEL_INPUT, y: MODEL_OUTPUT=None) -> Example:

        if x.grid.dim() == 2:
            assert x.grid.size(0) == 1
            assert y.grid.size(0) == 1
            x = x.squeeze(0)
            y = y.squeeze(0)

        assert x.grid.dim() == 1 and y.grid.dim() == 1, 'Grids must be 1D tensors.'
  
        input_decoded = self.grid_tokenizer.decode(x.grid.tolist())
        output_decoded = self.grid_tokenizer.decode(y.grid.tolist()) if y else None
        program_decoded = self.program_tokenizer.decode(x.program.tolist())
        color_permutation_decoded = self.color_permutation_tokenizer.decode(x.color_permutation.tolist())
        array_transform_decoded = self.array_transform_tokenizer.decode(x.array_transform.tolist())

        example = Example(
            idx=x.meta['example_id'],
            task_id=x.meta['task_id'],
            dataset=x.meta['dataset'],
            input=GridSerializer.deserialize_array(input_decoded),
            output=GridSerializer.deserialize_array(output_decoded),
            program_id=program_decoded,
            color_perm=color_permutation_decoded,
            transform=array_transform_decoded
        )

        return example
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, ArcTokenizer), 'value must be an instance of Tokenizer'
        return self.grid_tokenizer == value.grid_tokenizer and \
            self.program_tokenizer == value.program_tokenizer and \
            self.color_permutation_tokenizer == value.color_permutation_tokenizer and \
            self.array_transform_tokenizer == value.array_transform_tokenizer

    def to_dict(self):
        assert self.program_tokenizer.frozen, 'ProgramTokenizer must be frozen before saving.'
        return {
            'color_permutation_tokenizer': self.color_permutation_tokenizer.to_dict(),
            'array_transform_tokenizer': self.array_transform_tokenizer.to_dict(),
            'program_tokenizer': self.program_tokenizer.to_dict(),
            'grid_tokenizer': self.grid_tokenizer.to_dict(),
        }
    
    def build_program_tokenizer(self, examples: List[Example]):
        programs = [example.program_id for example in examples]
        self.program_tokenizer.build(programs)
    
    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.color_permutation_tokenizer = ColorPermutationTokenizer.from_dict(data['color_permutation_tokenizer'])
        obj.array_transform_tokenizer = ArrayTransformTokenizer.from_dict(data['array_transform_tokenizer'])
        obj.program_tokenizer = ProgramTokenizer.from_dict(data['program_tokenizer'])
        obj.grid_tokenizer = GridTokenizer.from_dict(data['grid_tokenizer'])
        return obj
    