import torch

torch.manual_seed(42)

def check_tensor(x, name="tensor"):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} должен быть torch.Tensor, а получен {type(x).__name__}")


def safe_matmul(a, b):
    check_tensor(a, "a")
    check_tensor(b, "b")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Нужны 2D-матрицы, получены {a.dim()}D и {b.dim()}D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Нельзя умножить {tuple(a.shape)} на {tuple(b.shape)}: "
            f"{a.shape[1]} != {b.shape[0]}"
        )
    return a @ b


def safe_elementwise_mul(a, b):
    check_tensor(a, "a")
    check_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"Формы не совпадают: {tuple(a.shape)} и {tuple(b.shape)}")
    return a * b


def safe_reshape(x, shape):
    check_tensor(x, "x")
    needed = 1
    for dim in shape:
        needed *= dim
    if needed != x.numel():
        raise ValueError(
            f"Нельзя превратить {x.numel()} элементов в форму {shape} "
            f"({needed} элементов)"
        )
    return x.reshape(shape)


# 1.1 Создание тензоров

def create_tensors():
    rand_3x4 = torch.rand(3, 4)
    zeros_2x3x4 = torch.zeros(2, 3, 4)
    ones_5x5 = torch.ones(5, 5)
    range_4x4 = torch.arange(16).reshape(4, 4)
    return {
        "rand_3x4": rand_3x4,
        "zeros_2x3x4": zeros_2x3x4,
        "ones_5x5": ones_5x5,
        "range_4x4": range_4x4,
    }

# 1.2 Операции с тензорами

def tensor_operations(A, B):
    A_T = A.T
    matmul = safe_matmul(A, B)
    elementwise = safe_elementwise_mul(A, B.T)
    sum_A = A.sum()
    return {"A_T": A_T, "matmul": matmul, "elementwise": elementwise, "sum_A": sum_A}

# 1.3 Индексация и срезы

def indexing_and_slicing(t):
    check_tensor(t, "t")
    if t.shape != (5, 5, 5):
        raise ValueError(f"Ожидался тензор 5x5x5, получен {tuple(t.shape)}")

    first_row = t[0, 0, :]
    last_column = t[0, :, -1]
    mid = t.shape[0] // 2         
    start = (t.shape[1] - 2) // 2  
    center_2x2 = t[mid, start:start + 2, start:start + 2]
    even_indices = t[::2, ::2, ::2]

    return {
        "first_row": first_row,
        "last_column": last_column,
        "center_2x2": center_2x2,
        "even_indices": even_indices,
    }



# 1.4 Работа с формами

def reshape_examples():
    x = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    return {shape: safe_reshape(x, shape) for shape in shapes}


# ===== Демонстрация =====
print("----- 1.1 -----")
tensors = create_tensors()
for name, value in tensors.items():
    print(name, value.shape)
    print(value)

print("----- 1.2 -----")
A = torch.rand(3, 4)
B = torch.rand(4, 3)
ops = tensor_operations(A, B)
for name, value in ops.items():
    print(name, value.shape)
    print(value)

print("----- 1.3 -----")
cube = torch.arange(125).reshape(5, 5, 5)
parts = indexing_and_slicing(cube)
for name, value in parts.items():
    print(name, value.shape)
    print(value)

print("----- 1.4 -----")
for shape, value in reshape_examples().items():
    print(shape)
    print(value)

# ===== Тесты =====
# Умножение на примере, посчитанном руками: 1*5 + 2*7 = 19 и т.д.
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])
assert torch.equal(a @ b, torch.tensor([[19., 22.], [43., 50.]]))

# Формы получились те, что просили
assert tensors["zeros_2x3x4"].shape == (2, 3, 4)
assert ops["matmul"].shape == (3, 3)
assert parts["even_indices"].shape == (3, 3, 3)

# Срезы достали нужные числа
assert parts["first_row"].tolist() == [0, 1, 2, 3, 4]
assert parts["center_2x2"].tolist() == [[56, 57], [61, 62]]

print("Все тесты пройдены")