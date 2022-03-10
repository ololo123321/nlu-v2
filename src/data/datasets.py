import re
from typing import List, Callable, Pattern, Union
from abc import ABC, abstractmethod
import tqdm

from bert.tokenization import FullTokenizer

from src.data.io import load_collection, simplify, read_file_v3
from src.data.check import check_tokens_entities_alignment, check_arcs, check_split
from src.data.preprocessing import split_example_v2, enumerate_entities, get_connected_components
from src.data.base import Languages, Example
from src.utils import ModeKeys


class BaseDataset(ABC):
    """
    Здесь сосредоточена логика фильтрации и препроцессинга сырых документов.

    """
    def __init__(
            self,
            data: List[Example] = None,
            mode: str = None,
            tokenizer: FullTokenizer = None,
            tokens_expression: Union[str, Pattern] = None,
            ignore_bad_examples: bool = True,
            max_chunk_length: int = 512,
            window: int = 3,
            stride: int = 1,
            language: str = Languages.RU,
            fix_sent_pointers: bool = True
    ):
        self.data = data
        self.mode = mode
        self.tokenizer = tokenizer
        self.tokens_expression = tokens_expression
        if isinstance(self.tokens_expression, str):
            self.tokens_expression = re.compile(self.tokens_expression)
        self.ignore_bad_examples = ignore_bad_examples
        self.max_chunk_length = max_chunk_length  # TODO: без учёта cls и sep
        self.window = window
        self.stride = stride
        self.language = language
        self.fix_sent_pointers = fix_sent_pointers

    def load(self, data_dir: str, limit: int = None, read_fn: Callable = read_file_v3):
        self.data = load_collection(
            data_dir=data_dir,
            n=limit,
            tokens_expression=self.tokens_expression,
            ignore_bad_examples=self.ignore_bad_examples,
            read_fn=read_fn
        )

    def filter(self):
        """
        какие документы есть смысл использовать:
        * игнорить документы, где есть какие-то несогласованности из-за косячности файлов .ann и .txt
        * coref, re - в документе должно быть более одной сущности
        """
        self.data = [x for x in self.data if self._is_valid_example(x)]

    def preprocess(self):
        for x in tqdm.tqdm(self.data):
            self._preprocess_example(x)

    def check(self):  # TODO: криетрии корректности к документам и кусочкам могут быть разными
        """
        итоговая проверка корректности примеров
        """
        for x in self.data:
            assert self._is_valid_example(x)

    # TODO: криетрии корректности к документам и кусочкам могут быть разными
    @abstractmethod
    def _is_valid_example(self, x: Example) -> bool:
        """
        какой пример является корректным
        """

    @abstractmethod
    def _preprocess_example(self, x: Example) -> None:
        """
        весь препроцессинг реализуется здесь:
        * разделение на предложения
        * удаление дубликатов сущностей и рёбер
        """

    def _apply_bpe(self, x: Example):
        """
        word-piece токенизация
        """
        for t in x.tokens:
            t.pieces = self.tokenizer.tokenize(t.text)
            t.token_ids = self.tokenizer.convert_tokens_to_ids(t.pieces)


class CoreferenceResolutionDataset(BaseDataset):
    def _preprocess_example(self, x: Example) -> None:
        # remove redundant entities and edged
        x = simplify(x)

        if self.mode == ModeKeys.VALID:
            self._assign_chain_ids(x)

        # split documents in chunks
        # tokenize chunks, remove bad chunks
        chunks = split_example_v2(
            x,
            window=self.window,
            stride=self.stride,
            lang=self.language,
            tokens_expression=self.tokens_expression,
            fix_pointers=self.fix_sent_pointers
        )
        for chunk in chunks:
            check_split(chunk, window=self.window, fixed_sent_pointers=self.fix_sent_pointers)
            if len(chunk.entities) > 0:
                self._apply_bpe(chunk)
                if all(len(t.token_ids) > 0 for t in chunk.tokens) \
                        and (sum(len(t.pieces) for t in chunk.tokens) <= self.max_chunk_length):
                    enumerate_entities(chunk)
                    x.chunks.append(chunk)

    # TODO: доделать
    def _is_valid_example(self, x: Example) -> bool:
        try:
            check_tokens_entities_alignment(x)
            check_arcs(x, one_parent=False, one_child=True)
            if len(x.entities) > 1:
                return True
            else:
                return False
        except AssertionError:
            return False

    @staticmethod
    def _assign_chain_ids(x: Example):
        id2entity = {}
        g = {}
        for entity in x.entities:
            g[entity.id] = set()
            id2entity[entity.id] = entity
        for arc in x.arcs:
            g[arc.head].add(arc.dep)

        components = get_connected_components(g)

        for id_chain, comp in enumerate(components):
            for id_entity in comp:
                id2entity[id_entity].id_chain = id_chain
