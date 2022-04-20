import re
from typing import List, Callable, Pattern, Union, Dict
from abc import ABC, abstractmethod
import tqdm

from bert.tokenization import FullTokenizer

from src.data.io import load_collection, simplify, read_file_v3, from_conllu, to_conllu, to_brat_v2
from src.data.check import (
    check_tokens_entities_alignment,
    check_arcs,
    check_split,
    check_multi_class_ner_markup,
    check_multi_class_re_markup
)
from src.data.preprocessing import split_example_v2, enumerate_entities, get_connected_components
from src.data.base import Languages, Example, NO_LABEL
from src.utils import LoggerMixin, log


class BaseDataset(ABC, LoggerMixin):
    """
    Здесь сосредоточена логика фильтрации и препроцессинга сырых документов.

    """
    def __init__(
            self,
            data: List[Example] = None,
            tokenizer: FullTokenizer = None,
            tokens_expression: Union[str, Pattern] = None,
            ignore_bad_examples: bool = True,
            max_chunk_length: int = 512,
            window: int = 1,
            stride: int = 1,
            language: str = Languages.RU,
            fix_sent_pointers: bool = True,
            read_fn: Callable = None,
            logger_parent_name: str = None
    ):
        super().__init__(logger_parent_name=logger_parent_name)
        self.data = data
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
        if read_fn is not None:
            self.read_fn = read_fn
        else:
            self.read_fn = read_file_v3

    @log
    def load(self, path: str, limit: int = None):
        self.data = load_collection(
            data_dir=path,
            n=limit,
            tokens_expression=self.tokens_expression,
            ignore_bad_examples=self.ignore_bad_examples,
            read_fn=self.read_fn,
            verbose_fn=self.logger.info
        )
        return self

    @log
    def save(self, path: str) -> None:
        to_brat_v2(self.data, output_dir=path)

    @log
    def filter(self, doc_level=True, chunk_level=True):
        """
        какие документы есть смысл использовать:
        * число модельных токенов <= max_chunk_length
        * игнорить документы, где есть какие-то несогласованности из-за косячности файлов .ann и .txt
        * coref, re - в документе должно быть более одной сущности.
        примеры, не прошедшие условия, описанные в _is_valid_example, _is_valid_chunk отсеиваются
        """
        if not (doc_level or chunk_level):
            return self
        num_examples_init = len(self.data)
        num_chunks_init = 0
        num_chunks_new = 0
        data_new = []
        for x in self.data:
            num_chunks_init += len(x.chunks)
            if (doc_level and self._is_valid_example(x)) or (not doc_level):
                if chunk_level:
                    x.chunks = [
                        chunk for chunk in x.chunks if self._is_valid_length(chunk) and self._is_valid_chunk(chunk)
                    ]
                data_new.append(x)
                num_chunks_new += len(x.chunks)
        self.data = data_new
        num_examples_new = len(self.data)
        self.logger.info(f"{num_examples_new} / {num_examples_init} examples saved "
                         f"({num_examples_init - num_examples_new} removed).")
        self.logger.info(f"{num_chunks_new} / {num_chunks_init} chunks saved "
                         f"({num_chunks_init - num_chunks_new} removed).")
        return self

    @log
    def preprocess(self):
        self.data = [self._preprocess_example(x) for x in tqdm.tqdm(self.data)]
        return self

    @log
    def check(self, doc_level=True, chunk_level=True):
        """
        как filter, только вызывается AssertionError, если пример не прошёл условия,
        описанные в _is_valid_example, _is_valid_chunk
        """
        if not (doc_level or chunk_level):
            return self
        for x in self.data:
            if doc_level:
                assert self._is_valid_example(x)
            for chunk in x.chunks:
                if chunk_level:
                    assert self._is_valid_length(chunk)
                    assert self._is_valid_chunk(chunk)
        return self

    @log
    def clear(self):
        """
        подготовка примеров для инференса:
        * удаление сущностей (ner)
        * удаление отношений (re, cr)
        """
        for x in self.data:
            self._clear_example(x)
        return self

    @log
    def fit(self) -> Dict:
        """
        получение маппинга label -> int
        """
        return {}

    @abstractmethod
    def _is_valid_example(self, x: Example) -> bool:
        """
        какой пример является корректным.
        """

    def _is_valid_chunk(self, x: Example) -> bool:
        """
        кроме условий для документа нужно ещё проверить следующие условия:
        * корректность разбиения на предложения (см. условия в описании функции check_split)
        * отсутствие символов, которые не удётся токенизировать TODO: проверить, почему это важно
        """
        if self._is_valid_example(x):
            try:
                check_split(x, window=self.window, fixed_sent_pointers=self.fix_sent_pointers)
                if all(len(t.token_ids) > 0 for t in x.tokens):
                    return True
                else:
                    return False
            except AssertionError:
                return False
        else:
            return False

    @abstractmethod
    def _preprocess_example(self, x: Example) -> Example:
        """
        весь препроцессинг реализуется здесь:
        * разделение на предложения
        * удаление дубликатов сущностей и рёбер

        возвращает новый инстанс
        """

    @abstractmethod
    def _clear_example(self, x: Example) -> None:
        pass

    def _apply_bpe(self, x: Example) -> None:
        """
        word-piece токенизация
        """
        for t in x.tokens:
            t.pieces = self.tokenizer.tokenize(t.text)
            t.token_ids = self.tokenizer.convert_tokens_to_ids(t.pieces)

    def _is_valid_length(self, x: Example) -> bool:
        return sum(len(t.pieces) for t in x.tokens) <= self.max_chunk_length


def assign_chain_ids(x: Example):
    """
    x - документ, а не кусок!
    """
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


class CoreferenceResolutionDataset(BaseDataset):
    def _preprocess_example(self, x: Example) -> Example:
        # remove redundant entities and edged
        x = simplify(x)

        assign_chain_ids(x)

        # split documents in chunks
        # tokenize chunks, remove bad chunks
        x.chunks = split_example_v2(
            x,
            window=self.window,
            stride=self.stride,
            lang=self.language,
            tokens_expression=self.tokens_expression,
            fix_pointers=self.fix_sent_pointers
        )
        for chunk in x.chunks:
            self._apply_bpe(chunk)
            enumerate_entities(chunk)
        return x

    def _is_valid_example(self, x: Example) -> bool:
        """
        * спаны сущностей согласованы с текстом
        * сущности отношения должны быть в множестве сущностей документа
        * каждая сущность ссылается не более чем на одну
        """
        try:
            check_tokens_entities_alignment(x)
            check_arcs(x, one_parent=False, one_child=True)
            if len(x.entities) > 1:
                return True
            else:
                return False
        except AssertionError:
            return False

    def _clear_example(self, x: Example) -> None:
        x.arcs = []


class DependencyParsingDataset(BaseDataset):
    @log
    def load(self, path: str, limit: int = None, read_fn: Callable = read_file_v3):
        self.data = from_conllu(path=path, warn=False)
        return self

    @log
    def save(self, path: str) -> None:
        to_conllu(self.data, path)

    @log
    def fit(self) -> Dict:
        labels = set()
        for x in self.data:
            for chunk in x.chunks:
                for t in chunk.tokens:
                    labels.add(t.rel)
        res = {
            "rel_enc": {x: i for i, x in enumerate(sorted(labels))}
        }
        return res

    def _is_valid_example(self, x: Example) -> bool:
        """
        * делить на предложения не нужно -> не нужно проверять корректность разбиения на предложения
        * нет ни рёбер, ни сущностей -> не нужно проверять корректность этих товарищей
        """
        return True

    def _is_valid_chunk(self, x: Example) -> bool:
        """
        * нет ни рёбер, ни сущностей -> не нужно проверять корректность этих товарищей
        """
        return True

    def _preprocess_example(self, x: Example) -> Example:
        """
        * исходные документы уже разбиты на предложения -> делить на них не нужно
        * убирать лишние рёбра тоже не нужно, потому что их нет: синтаксические деревья заданы над токенами,
        а у каждого токена есть атрибуты id_head (номер родителя) и rel (тип отношения) -> упрощать граф не нужно
        """
        for chunk in x.chunks:
            self._apply_bpe(chunk)
        return x

    def _clear_example(self, x: Example) -> None:
        """
        пример изначально разбит на кусочки, поэтому чистим только их
        """
        for chunk in x.chunks:
            for t in chunk.tokens:
                t.reset()


class NerAsSequenceLabelingDataset(BaseDataset):
    def __init__(
            self,
            data: List[Example] = None,
            tokenizer: FullTokenizer = None,
            tokens_expression: Union[str, Pattern] = None,
            ignore_bad_examples: bool = True,
            max_chunk_length: int = 512,
            window: int = 1,
            stride: int = 1,
            language: str = Languages.RU,
            fix_sent_pointers: bool = True,
            read_fn: Callable = None,
            is_flat_ner: bool = True,
            logger_parent_name: str = None
    ):
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            tokens_expression=tokens_expression,
            ignore_bad_examples=ignore_bad_examples,
            max_chunk_length=max_chunk_length,
            window=window,
            stride=stride,
            language=language,
            fix_sent_pointers=fix_sent_pointers,
            read_fn=read_fn,
            logger_parent_name=logger_parent_name
        )
        self.is_flat_ner = is_flat_ner

    @log
    def fit(self) -> Dict:
        labels = set()
        for x in self.data:
            for t in x.tokens:
                if t.label != NO_LABEL:
                    labels.add(t.label)
        labels = [NO_LABEL] + sorted(labels)
        res = {
            "ner_enc": {x: i for i, x in enumerate(labels)}
        }
        return res

    def _is_valid_example(self, x: Example) -> bool:
        """
        * спаны сущностей согласованы с текстом
        * если плоский нер, то не должно быть вложенных сущностей
        """
        try:
            check_tokens_entities_alignment(x)
            check_multi_class_ner_markup(x)
            if self.is_flat_ner:
                entities_sorted = sorted(
                    x.entities, key=lambda e: (e.tokens[0].span_abs.start, e.tokens[-1].span_abs.end)
                )
                for i in range(len(entities_sorted) - 1):
                    e1 = entities_sorted[i]
                    e2 = entities_sorted[i + 1]
                    if e1.tokens[-1].span_abs.end > e2.tokens[0].span_abs.start:
                        self.logger.warning(f'[{x.id}] overlapping entities: {e1.id} and {e2.id}')
                        return False
                return True
            else:
                return True
        except AssertionError as e:
            self.logger.error(e)
            return False

    def _preprocess_example(self, x: Example) -> Example:
        simplify(x)
        x.assign_labels_to_tokens()
        x.chunks = split_example_v2(
            x,
            window=self.window,
            stride=self.stride,
            lang=self.language,
            tokens_expression=self.tokens_expression,
            fix_pointers=self.fix_sent_pointers
        )
        for chunk in x.chunks:
            self._apply_bpe(chunk)
        return x

    def _clear_example(self, x: Example) -> None:
        x.entities = []
        for t in x.tokens:
            t.reset()


class NerAsSpanPredictionDataset(NerAsSequenceLabelingDataset):
    @log
    def fit(self) -> Dict:
        labels = set()
        for x in self.data:
            for e in x.entities:
                labels.add(e.label)
        labels = [NO_LABEL] + sorted(labels)
        res = {
            "ner_enc": {x: i for i, x in enumerate(labels)}
        }
        return res

    def _preprocess_example(self, x: Example) -> Example:
        simplify(x)
        x.chunks = split_example_v2(
            x,
            window=self.window,
            stride=self.stride,
            lang=self.language,
            tokens_expression=self.tokens_expression,
            fix_pointers=self.fix_sent_pointers
        )
        for chunk in x.chunks:
            self._apply_bpe(chunk)
        return x

    def _clear_example(self, x: Example) -> None:
        x.entities = []


class RelationExtractionDataset(BaseDataset):
    @log
    def fit(self) -> Dict:
        labels_re = set()
        labels_ner = set()
        for x in self.data:
            for arc in x.arcs:
                labels_re.add(arc.rel)
            for e in x.entities:
                labels_ner.add(e.label)
        labels_re = [NO_LABEL] + sorted(labels_re)
        labels_ner = [NO_LABEL] + sorted(labels_ner)
        res = {
            "re_enc": {x: i for i, x in enumerate(labels_re)},
            "ner_enc": {x: i for i, x in enumerate(labels_ner)}
        }
        return res

    def _is_valid_example(self, x: Example) -> bool:
        """
        * спаны сущностей согласованы с текстом
        в данном таске сущности уже даны заранее и могут быть любыми, поэтому не нужно проверять вложенность
        """
        try:
            check_tokens_entities_alignment(x)
            check_multi_class_ner_markup(x)
            check_multi_class_re_markup(x)
            check_arcs(x, one_child=False, one_parent=False)  # TODO: в конфиг
            return True
        except AssertionError as e:
            self.logger.error(e)
            return False

    def _is_valid_chunk(self, x: Example) -> bool:
        if super()._is_valid_chunk(x):
            try:
                check_arcs(x, one_child=False, one_parent=False)  # TODO: в конфиг
                return True
            except AssertionError as e:
                self.logger.error(e)
                return False
        else:
            return False

    def _preprocess_example(self, x: Example) -> Example:
        x = simplify(x)
        x.chunks = split_example_v2(
            x,
            window=self.window,
            stride=self.stride,
            lang=self.language,
            tokens_expression=self.tokens_expression,
            fix_pointers=self.fix_sent_pointers
        )
        for chunk in x.chunks:
            self._apply_bpe(chunk)
            enumerate_entities(chunk)
        return x

    def _clear_example(self, x: Example) -> None:
        x.arcs = []


# for chunk in chunks:
#     check_split(chunk, window=self.window, fixed_sent_pointers=self.fix_sent_pointers)
#     if len(chunk.entities) > 0:
#         self._apply_bpe(chunk)
#         if all(len(t.token_ids) > 0 for t in chunk.tokens) \
#                 and (sum(len(t.pieces) for t in chunk.tokens) <= self.max_chunk_length):
#             enumerate_entities(chunk)
#             x.chunks.append(chunk)


# старая функция из src.utils
# def get_filtered_by_length_chunks(
#         examples: List[Example],
#         maxlen: int = None,
#         pieces_level: bool = False
# ) -> List[Example]:
#     res = []
#     ignored_ids = {}
#     for x in examples:
#         for chunk in x.chunks:
#             if maxlen is None:
#                 res.append(chunk)
#                 continue
#             if pieces_level:
#                 n = sum(len(t.pieces) for t in chunk.tokens)
#             else:
#                 n = len(chunk.tokens)
#             if n <= maxlen:
#                 res.append(chunk)
#             else:
#                 ignored_ids[chunk.id] = n
#     s = "pieces" if pieces_level else "tokens"
#     if len(ignored_ids) > 0:
#         print("number of ignored examples:", len(ignored_ids))
#         print(f"following examples are ignored due to their length is > {maxlen} {s}:")
#         for k, v in ignored_ids.items():
#             print(f"{k} has length {v} {s}, which is greater than {maxlen}")
#     else:
#         print(f"all examples have length <= {maxlen} {s}")
#     return res
