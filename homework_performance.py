import time
import torch


REPEATS = 3
WARMUP = 1

CUDA_AVAILABLE = torch.cuda.is_available()

# 3.1 Подготовка данных
SIZES = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
]


def create_matrices(shape):
    if len(shape) != 3 or any((not isinstance(d, int)) or d <= 0 for d in shape):
        raise ValueError(f"Ожидалась форма из трёх положительных целых, получено {shape}")
    a = torch.rand(shape, dtype=torch.float32)
    b = torch.rand(shape, dtype=torch.float32)
    return a, b

# 3.2 Функция измерения времени
def measure_time(func, device, repeats=REPEATS, warmup=WARMUP):
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device должен быть 'cpu' или 'cuda', получено {device!r}")
    if repeats < 1:
        raise ValueError("repeats должен быть >= 1")
    if device == "cuda" and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA недоступна на этом компьютере")
    for _ in range(warmup):
        func()

    #GPU
    if device == "cuda":
        torch.cuda.synchronize() 
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            func()
        end.record()
        torch.cuda.synchronize()  
        return start.elapsed_time(end) / repeats 
    #CPU 
    start = time.time()
    for _ in range(repeats):
        func()
    return (time.time() - start) * 1000 / repeats 


# 3.3 Сравнение операций

def get_operations(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Формы не совпадают: {tuple(a.shape)} и {tuple(b.shape)}")
    return {
        "Матричное умножение": lambda: torch.matmul(a, b),
        "Сложение": lambda: a + b,
        "Поэлементное умножение": lambda: a * b,
        "Транспонирование": lambda: a.transpose(1, 2),
        "Транспонирование + копия": lambda: a.transpose(1, 2).contiguous(),
        "Сумма всех элементов": lambda: a.sum(),
    }


def benchmark_size(shape):
    a_cpu, b_cpu = create_matrices(shape)
    cpu_ops = get_operations(a_cpu, b_cpu)

    gpu_ops = None
    if CUDA_AVAILABLE:
        a_gpu, b_gpu = a_cpu.to("cuda"), b_cpu.to("cuda")
        gpu_ops = get_operations(a_gpu, b_gpu)

    results = []
    for name, cpu_func in cpu_ops.items():
        cpu_ms = measure_time(cpu_func, "cpu")
        gpu_ms = measure_time(gpu_ops[name], "cuda") if gpu_ops else None
        speedup = cpu_ms / gpu_ms if gpu_ms else None
        results.append({"op": name, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": speedup})

    if CUDA_AVAILABLE:
        transfer_ms = measure_time(lambda: a_cpu.to("cuda"), "cuda")
        results.append({"op": "Передача CPU -> GPU", "cpu_ms": None,
                        "gpu_ms": transfer_ms, "speedup": None})
        del a_gpu, b_gpu
        torch.cuda.empty_cache()

    return results


def print_table(shape, results):
    def fmt(value, suffix=""):
        return "—" if value is None else f"{value:.2f}{suffix}"

    print(f"\nРазмер: {' x '.join(map(str, shape))}")
    print(f"{'Операция':<26}| {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
    print("-" * 66)
    for r in results:
        print(f"{r['op']:<26}| {fmt(r['cpu_ms']):>10} | {fmt(r['gpu_ms']):>10} | "
              f"{fmt(r['speedup'], 'x'):>10}")


# 3.4 Анализ результатов

def analyze(all_results):
    print("\n" + "=" * 66)
    print("Анализ")
    print("=" * 66)

    if not CUDA_AVAILABLE:
        print("GPU не найден, поэтому сравнить с CUDA нельзя — посчитано только CPU.")
        print("Видно, что матричное умножение — самая тяжёлая операция: "
              "ей нужно ~2*N операций на каждый элемент результата,")
        print("а сложению и умножению поэлементно — одна.")
        return

    for shape, results in all_results.items():
        compared = [r for r in results
                    if r["speedup"] is not None and r["op"] != "Транспонирование"]
        best = max(compared, key=lambda r: r["speedup"])
        slower = [r["op"] for r in compared if r["speedup"] < 1]
        size = " x ".join(map(str, shape))
        print(f"\n{size}: больше всего ускорилось «{best['op']}» ({best['speedup']:.1f}x)")
        if slower:
            print(f"  На GPU медленнее или так же: {', '.join(slower)}")

        transfer = next(r for r in results if r["op"] == "Передача CPU -> GPU")
        add = next(r for r in results if r["op"] == "Сложение")
        print(f"  Передача одной матрицы на GPU: {transfer['gpu_ms']:.2f} мс, "
              f"а сложение на GPU: {add['gpu_ms']:.2f} мс")


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

def run_tests():
    """Проверяет, что CPU и GPU считают одно и то же, и что проверки работают."""
    a, b = create_matrices((2, 3, 3))
    assert a.shape == (2, 3, 3)

    # Транспонирование действительно меняет местами индексы
    assert torch.equal(a.transpose(1, 2)[0, 0, 1], a[0, 1, 0])

    # Пакетное умножение = умножение каждой пары матриц по отдельности
    assert torch.allclose(torch.matmul(a, b)[1], a[1] @ b[1])

    # Время — положительное число
    assert measure_time(lambda: a + b, "cpu", repeats=2) >= 0

    if CUDA_AVAILABLE:
        # Результаты на GPU совпадают с CPU (с небольшой погрешностью float32)
        for name, cpu_func in get_operations(a, b).items():
            gpu_func = get_operations(a.cuda(), b.cuda())[name]
            assert torch.allclose(cpu_func(), gpu_func().cpu(), atol=1e-4), name

    try:
        create_matrices((10, -5, 5))
        raise AssertionError("Должна была быть ошибка формы")
    except ValueError:
        pass

    print("Все тесты пройдены ✅")


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def main():
    print(f"PyTorch {torch.__version__}")
    if CUDA_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU (CUDA) не найден — замеры только на CPU")
    print(f"Повторов на замер: {REPEATS}, прогрев: {WARMUP}")

    run_tests()

    all_results = {}
    for shape in SIZES:
        results = benchmark_size(shape)
        print_table(shape, results)
        all_results[shape] = results

    analyze(all_results)


if __name__ == "__main__":
    main()