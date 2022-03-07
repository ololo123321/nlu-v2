from typing import Tuple, List
import tensorflow as tf
import numpy as np


tf = tf.compat.v1


def get_labels_mask(labels_2d: tf.Tensor, values: tf.Tensor, sequence_len: tf.Tensor) -> tf.Tensor:
    """
    Получение маски: mask[i, j] = any(labels[i, j] == v for v in values)
    :param labels_2d: tf.Tensor of shape [N, T] and type tf.int32 - label ids
    :param values: tf.Tensor of shape [num_ids] and type tf.int32 - start label ids. может быть пустым
    :param sequence_len: tf.Tensor of shape [N] and type tf.int32 - sequence lengths
    :return: tf.Tensor of shape [N, T] and type tf.bool
    """
    labels_3d = tf.tile(labels_2d[:, :, None], [1, 1, tf.shape(values)[0]])  # [N, T, num_bound_ids]
    mask_3d = tf.equal(labels_3d, values[None, None, :])  # [N, T, num_bound_ids]
    mask_2d = tf.reduce_any(mask_3d, axis=-1)  # [N, T]
    mask_2d = tf.logical_and(mask_2d, tf.sequence_mask(sequence_len))  # [N, T]
    return mask_2d


def get_padded_coords_2d(mask_2d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Получение кординат элементов True с паддингом (0, 0).
    В случае отсутствия элементов True возвращается нулевой тензор размерности [mask_2d.shape[0], 2]:

    mask_2d:
    [[False, True,  False],
     [True,  False, True],
     [False, False, False]]
    coords:
    [
     [[0, 1], [0, 0]],
     [[1, 0], [1, 2]],
     [[0, 0], [0, 0]]
    ]

    mask_2d:
    [[False, False,  False],
     [False, False, False],
     [False, False, False]]
    coords:
    [
     [[0, 0]],
     [[0, 0]],
     [[0, 0]]
    ]

    нотация:
    N - batch size
    T - number of tokens (max)
    E - number of elements (max)
    Es - number of elements (sum)

    :param mask_2d: Tensor of shape [N, T] and type bool
    :return: coords: Tensor of shape [N, E  or 1, 2],
             где E - наибольее число элементов True в строке mask_2d.
             (i, j) - начало или конец сущности, где 0 <= i < N; 0 <= j < T
    """
    # вывод координаты y
    num_elements = tf.reduce_sum(tf.cast(mask_2d, tf.int32), axis=-1)  # [N]
    sequence_mask = tf.sequence_mask(num_elements)  # [N, E]
    indices = tf.cast(tf.where(sequence_mask), tf.int32)  # [Es, 2]
    updates = tf.cast(tf.where(mask_2d)[:, -1], tf.int32)  # [Es]
    sequence_mask_shape = tf.shape(sequence_mask)  # [2]
    y_coord = tf.scatter_nd(indices, updates, shape=sequence_mask_shape)  # [N, E], int32

    # вывод координаты x
    # Пусть число примеров = 3, максимальное число элементов = 2
    batch_size = sequence_mask_shape[0]  # []
    num_elements_max = sequence_mask_shape[1]  # []
    x_coord = tf.range(batch_size, dtype=tf.int32)  # [N], [0, 1, 2]
    x_coord = tf.tile(x_coord[:, None], [1, num_elements_max])  # [N, E], [[0, 0], [1, 1], [2, 2]]

    # объединение координат x и y
    coords = tf.concat([x_coord[:, :, None], y_coord[:, :, None]], axis=-1)  # [N, E, 2]

    # фейковые координаты в случае отсутствия сущностей
    coords_dummy = tf.zeros([batch_size, 1, 2], dtype=tf.int32)  # [N, 1, 2]
    cond = tf.equal(tf.reduce_max(num_elements), 0)  # []
    coords = tf.cond(cond, true_fn=lambda: coords_dummy, false_fn=lambda: coords)  # [N, E or 1, 2]

    return coords, num_elements


def get_padded_coords_3d(mask_3d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    немного доработаная функция get_padded_coords_2d с учётом того, что маска - трёхмерная

    нотация:
    N - batch size
    T - number of tokens (max)
    E - number of elements (max)
    Es - number of elements (sum)

    :param mask_3d: tf.Tensor of shape [N, T, T] and type tf.bool
    :return:
    starts_coords: Tensor of shape [N, E or 1, 2] and type int32
    end_coords: Tensor of shape [N, E or 1, 2] and type int32
    num_entities: Tensor of shape [N] and type int32
    """
    num_entities = tf.reduce_sum(tf.cast(mask_3d, tf.int32), axis=[1, 2])  # [N]
    sequence_mask = tf.sequence_mask(num_entities)  # [N, E]
    sequence_mask_shape = tf.shape(sequence_mask)  # [2]
    coords = tf.cast(tf.where(mask_3d), tf.int32)  # [Es, 3]
    indices = tf.cast(tf.where(sequence_mask), tf.int32)  # [Es, 2]
    updates_start = coords[:, 1]  # [Es]
    updates_end = coords[:, 2]  # [Es]
    y_coord_start = tf.scatter_nd(indices, updates_start, shape=sequence_mask_shape)  # [N, E]
    y_coord_end = tf.scatter_nd(indices, updates_end, shape=sequence_mask_shape)  # [N, E]

    batch_size = sequence_mask_shape[0]  # []
    num_elements_max = sequence_mask_shape[1]  # []
    x_coord = tf.range(batch_size, dtype=tf.int32)  # [N]
    x_coord = tf.tile(x_coord[:, None], [1, num_elements_max])  # [N, E]

    start_coords = tf.concat([x_coord[:, :, None], y_coord_start[:, :, None]], axis=-1)  # [N, E, 2]
    end_coords = tf.concat([x_coord[:, :, None], y_coord_end[:, :, None]], axis=-1)  # [N, E, 2]

    # фейковые координаты в случае отсутствия сущностей
    coords_dummy = tf.zeros([batch_size, 1, 2], dtype=tf.int32)  # [N, 1, 2]
    cond = tf.equal(tf.reduce_max(num_entities), 0)  # []
    start_coords = tf.cond(cond, true_fn=lambda: coords_dummy, false_fn=lambda: start_coords)  # [N, E or 1, 2]
    end_coords = tf.cond(cond, true_fn=lambda: coords_dummy, false_fn=lambda: end_coords)  # [N, E or 1, 2]
    return start_coords, end_coords, num_entities


def get_batched_coords_from_labels(
        labels_2d: tf.Tensor,
        values: tf.Tensor,
        sequence_len: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    mask_2d = get_labels_mask(labels_2d=labels_2d, values=values, sequence_len=sequence_len)
    return get_padded_coords_2d(mask_2d=mask_2d)


def get_entity_embeddings(
        x: tf.Tensor,
        d_model: int,
        start_coords: tf.Tensor,
        end_coords: tf.Tensor = None
) -> tf.Tensor:
    """
    Векторизация сущностей. Предполагается, что границы сущностей известны.
    увидел такое представление здесь: https://arxiv.org/pdf/1911.03875.pdf

    :param x:
    :param d_model:
    :param start_coords: [batch_size, num_entities, 2]
    :param end_coords: [batch_size, num_entities, 2]
    :return:
    """

    coords_shape = tf.shape(start_coords)
    batch_size = coords_shape[0]
    num_entities_max = coords_shape[1]
    one = tf.tile([[[0, 1]]], [batch_size, num_entities_max, 1])
    x_i = tf.gather_nd(x, start_coords)  # [N, num_entities, D]
    x_i_minus_one = tf.gather_nd(x, end_coords - one)  # [N, num_entities, D]
    x_j = tf.gather_nd(x, end_coords)  # [N, num_entities, D]
    x_j_plus_one = tf.gather_nd(x, end_coords + one)  # [N, num_entities, D]

    d_model_half = d_model // 2
    x_start = x_j - x_i_minus_one
    x_start = x_start[..., :d_model_half]
    x_end = x_i - x_j_plus_one
    x_end = x_end[..., d_model_half:]

    x_span = tf.concat([x_start, x_end], axis=-1)  # [N, num_entities, D]

    return x_span


def get_entity_embeddings_concat(
        x: tf.Tensor,
        start_coords: tf.Tensor,
        end_coords: tf.Tensor = None
) -> tf.Tensor:
    x_start = tf.gather_nd(x, start_coords)  # [N, num_entities, D]
    x_end = tf.gather_nd(x, end_coords)  # [N, num_entities, D]
    x_span = tf.concat([x_start, x_end], axis=-1)  # [N, num_entities, D * 2]
    return x_span


def get_entity_embeddings_concat_half(
        x: tf.Tensor,
        d_model: int,
        start_coords: tf.Tensor,
        end_coords: tf.Tensor = None
) -> tf.Tensor:
    x_i = tf.gather_nd(x, start_coords)  # [N, num_entities, D]
    x_j = tf.gather_nd(x, end_coords)  # [N, num_entities, D]
    d_model_half = d_model // 2
    x_start = x_i[:, :, :d_model_half]
    x_end = x_j[:, :, d_model_half:]
    x_span = tf.concat([x_start, x_end], axis=-1)  # [N, num_entities, D]
    return x_span


# TODO: должна возвращать [batch_size, num_entities_max, span_size_max]
def get_span_indices(start_ids: tf.Tensor, end_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """

    :param start_ids: tf.Tensor of shape [batch_size, num_entities] and dtype tf.int32
    :param end_ids: tf.Tensor of shape [batch_size, num_entities] and dtype tf.int32
    :return: tf.Tensor of shape [batch_size, num_entities, max_span_size] and dtype tf.int32
    """
    span_sizes = end_ids - start_ids + 1  # [N, E]
    m = tf.reduce_max(span_sizes)  # []
    start_ids_shape = tf.shape(start_ids)  # [2]
    res = tf.range(m)  # [S]
    res = res[None, None, :]  # [1, 1, S]
    res = tf.tile(res, [start_ids_shape[0], start_ids_shape[1], 1])  # [N, E, S]
    res += start_ids[:, :, None]  # [N, E, S]
    sequence_mask = tf.sequence_mask(span_sizes, dtype=tf.int32)  # [N, E, S]
    res *= sequence_mask  # [N, E, S]
    return res, sequence_mask


# TODO: явно прокидывать indices и updates
# TODO: tensor_scatter_nd_update -> scatter_nd; deprecate no_label_id (must always be zero)
def get_dense_labels_from_indices(indices: tf.Tensor, shape: tf.Tensor, no_label_id: int = 0):
    """
    лейблы отношений.
    должно гарантироваться, что reduce_min(shape)) >= 1
    :param indices: tf.Tensor of shape [num_elements, ndims] - индексы логитов
    :param shape: tf.Tensor of shape [ndims] - размерность лейблов
    :param no_label_id: int
    :return:
    """
    labels = tf.broadcast_to(no_label_id, shape)  # [batch_size, num_entities, num_entities]
    labels = tf.tensor_scatter_nd_update(
        tensor=labels,
        indices=indices[:, :-1],
        updates=indices[:, -1],
    )  # [batch_size, num_entities, num_entities]
    return labels


def upper_triangular(n: int, dtype):
    x = tf.linalg.band_part(tf.ones((n, n)), 0, -1)
    x = tf.cast(x, dtype)
    return x

# def add_ones(x: tf.Tensor) -> tf.Tensor:
#     ones = tf.ones_like(x[..., :1])
#     x = tf.concat([x, ones], axis=-1)
#     return x


def bucket_distance(distances):
    """
    https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/coref_model.py#L395

    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/tf.log(2.0))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)


def noam_scheme(init_lr: int, global_step: int, warmup_steps: int = 4000):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def get_additive_mask(mask: tf.Tensor) -> tf.Tensor:
    """
    [True, False] -> [0.0, -1e9]
    :param mask: tf.Tensor of dtype tf.bool
    :return:
    """
    return (1.0 - tf.cast(mask, tf.float32)) * -1e9


def get_session() -> tf.Session:
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    return sess


def get_entities_representation(
        x: tf.Tensor,
        ner_labels: tf.Tensor,
        sparse_labels: bool,
        ff_attn=None,
        entity_emb_layer=None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    векторизация сущностей как конкатенация трёх векторов:
    * вектор первого токена сущности
    * вектор последнего токена сущности
    * взвешенный эмбеддинг всех токенов сущности
    :param x: tf.Tensor of shape [N, T, D]
    :param ner_labels: if sparse_labels: [num_entities_total, 4] and type int32, else: tf.Tensor of shape [N, T, T] and type int32
    :param sparse_labels
    :param ff_attn: Callable: ff_attn(x) -> y. x - [d_1, ..., d_k, D], y - [d1, ..., d_k, 1]
    :param entity_emb_layer: Callable: entity_emb_layer(x) -> y; x - [N, T], int32; y - [N, T, D], float32
    :return:
    """
    x_shape = tf.shape(x)

    if sparse_labels:
        indices = ner_labels[:, :-1]
        updates = ner_labels[:, -1]
        labels_shape = tf.concat([x_shape[:2], x_shape[1:2]], axis=0)  # [3], N, T, T
        ner_labels = tf.scatter_nd(indices=indices, updates=updates, shape=labels_shape)  # [N, T, T]

    # маскирование
    mask = upper_triangular(x_shape[1], dtype=tf.int32)  # [T, T]
    ner_labels_dense_masked = ner_labels * mask[None, :, :]  # [N, T, T]

    # векторизация сущностей
    features = []

    # start, end
    no_mention_id = 0
    span_mask = tf.not_equal(ner_labels_dense_masked, no_mention_id)  # [batch_size, T, T]
    start_coords, end_coords, num_entities = get_padded_coords_3d(mask_3d=span_mask)
    x_start = tf.gather_nd(x, start_coords)  # [batch_size, num_entities, d_model]
    x_end = tf.gather_nd(x, end_coords)  # [batch_size, num_entities, d_model]
    features.append(x_start)
    features.append(x_end)

    # attn
    if ff_attn is not None:
        grid, sequence_mask_span = get_span_indices(
            start_ids=start_coords[:, :, 1],
            end_ids=end_coords[:, :, 1]
        )  # ([batch_size, num_entities, span_size], [batch_size, num_entities, span_size])

        x_coord = tf.range(x_shape[0])[:, None, None, None]  # [batch_size, 1, 1, 1]
        grid_shape = tf.shape(grid)  # [3]
        x_coord = tf.tile(x_coord, [1, grid_shape[1], grid_shape[2], 1])  # [batch_size, num_entities, span_size, 1]
        y_coord = tf.expand_dims(grid, -1)  # [batch_size, num_entities, span_size, 1]
        coords = tf.concat([x_coord, y_coord], axis=-1)  # [batch_size, num_entities, span_size, 2]
        x_span = tf.gather_nd(x, coords)  # [batch_size, num_entities, span_size, d_model]
        w = ff_attn(x_span)  # [batch_size, num_entities, span_size, 1]
        sequence_mask_span = tf.expand_dims(sequence_mask_span, -1)
        w += get_additive_mask(sequence_mask_span)  # [batch_size, num_entities, span_size, 1]
        w = tf.nn.softmax(w, axis=2)  # [batch_size, num_entities, span_size, 1]
        x_span = tf.reduce_sum(x_span * w, axis=2)  # [batch_size, num_entities, d_model]
        features.append(x_span)

    # meta
    if entity_emb_layer is not None:
        entity_coords = tf.concat([start_coords, end_coords[:, :, -1:]], axis=-1)  # [batch_size, num_entities, 3]
        ner_labels_2d = tf.gather_nd(ner_labels, entity_coords)  # [batch_size, num_entities]
        ner_labels_2d *= tf.sequence_mask(num_entities, dtype=tf.int32)  # [batch_size, num_entities]
        x_emb = entity_emb_layer(ner_labels_2d)  # [batch_size, num_entities, d_emb]
        features.append(x_emb)

    x_entity = tf.concat(features, axis=-1)

    return x_entity, num_entities


def get_entity_pairs_mask(entity_sent_ids, i, j):
    """
    entity_sent_ids: np.ndarray of shape [num_entities]
    """
    num_entities = entity_sent_ids.shape[0]
    mask = np.zeros((num_entities, num_entities), dtype=np.int32)
    indices_i = np.where(entity_sent_ids == i)[0]
    indices_j = np.where(entity_sent_ids == j)[0]
    for i in indices_i:
        for j in indices_j:
            mask[i, j] = 1
            mask[j, i] = 1
    return mask


# 1 - общий случай
# 2 - первый кусок
# 3 - последний кусок
grid_2 = np.tri(2).T
grid_3 = np.tri(3).T
grid_3_alt = np.array([
    [2, 1, 1],
    [0, 1, 3],
    [0, 0, 3]
])
grid_4 = np.tri(4).T
grid_5 = np.array([
    [2, 2, 2, 1, 1],
    [0, 2, 1, 1, 3],
    [0, 0, 1, 3, 3],
    [0, 0, 0, 3, 3],
    [0, 0, 0, 0, 3]
])

pairs_2 = list(zip(*np.where(grid_2 == 1)))
pairs_3 = list(zip(*np.where(grid_3 == 1)))
pairs_3_1 = list(zip(*np.where(grid_3_alt == 1)))
pairs_3_2 = list(zip(*np.where(grid_3_alt == 2)))
pairs_3_3 = list(zip(*np.where(grid_3_alt == 3)))
pairs_4 = list(zip(*np.where(grid_4 == 1)))
pairs_5_1 = list(zip(*np.where(grid_5 == 1)))
pairs_5_2 = list(zip(*np.where(grid_5 == 2)))
pairs_5_3 = list(zip(*np.where(grid_5 == 3)))


# TODO: протестировать!
def get_sent_pairs_to_predict_for(end: int, is_first: bool, is_last: bool, window: int) -> List[Tuple]:
    assert 0 <= end < window, f"end: {end}, window: {window}, is_first: {is_first}, is_last: {is_last}"

    if window == 1:
        return [(0, 0)]
    elif window == 3:
        res = pairs_3_1.copy()
        if is_first:
            res += pairs_3_2
        if is_last:
            res += pairs_3_3
        return res
    elif window == 5:
        if end == 0:
            return [(0, 0)]
        elif end == 1:
            return pairs_2.copy()
        elif end == 2:
            return pairs_3.copy()
        elif end == 3:
            return pairs_4.copy()
        else:
            res = pairs_5_1.copy()
            if is_first:
                res += pairs_5_2
            if is_last:
                res += pairs_5_3
            return res
    else:
        raise NotImplementedError(f"expected window in {{1, 3, 5}}, but got {window}")


def get_sent_ids_to_predict_for(is_first, is_last, window):
    assert window in {1, 3, 5}
    if window == 1:
        return {0}
    elif window == 3:
        if is_first:
            return {0, 1}
        elif is_last:
            return {1, 2}
        else:
            return {1}
    elif window == 5:
        if is_first:
            return {0, 1, 2}
        elif is_last:
            return {2, 3, 4}
        else:
            return {2}
    else:
        raise NotImplementedError
