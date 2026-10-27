import math
import torch

# 2.1 Простые вычисления с градиентами

def f_xyz(x, y, z):
    return x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z


def grads_xyz_autograd(x_val, y_val, z_val):
    x = torch.tensor(float(x_val), requires_grad=True)
    y = torch.tensor(float(y_val), requires_grad=True)
    z = torch.tensor(float(z_val), requires_grad=True)

    f = f_xyz(x, y, z)
    f.backward()

    return f.item(), x.grad.item(), y.grad.item(), z.grad.item()


def grads_xyz_analytic(x, y, z):
    return 2 * x + 2 * y * z, 2 * y + 2 * x * z, 2 * z + 2 * x * y

# 2.2 Градиент функции потерь MSE

def mse(y_pred, y_true):
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("y_pred и y_true должны быть torch.Tensor")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Формы не совпадают: {tuple(y_pred.shape)} и {tuple(y_true.shape)}")
    if y_true.numel() == 0:
        raise ValueError("Пустые тензоры: делить на n = 0 нельзя")

    n = y_true.numel()
    return ((y_pred - y_true) ** 2).sum() / n


def mse_grads_autograd(x, y_true, w_val, b_val):
    w = torch.tensor(float(w_val), requires_grad=True)
    b = torch.tensor(float(b_val), requires_grad=True)

    y_pred = w * x + b        
    loss = mse(y_pred, y_true)
    loss.backward()

    return loss.item(), w.grad.item(), b.grad.item()


def mse_grads_analytic(x, y_true, w, b):
    n = x.numel()
    error = w * x + b - y_true
    dw = (2 / n) * (error * x).sum()
    db = (2 / n) * error.sum()
    return dw.item(), db.item()

# 2.3 Цепное правило

def f_chain(x):
    return torch.sin(x ** 2 + 1)


def chain_grad_backward(x_val):
    x = torch.tensor(float(x_val), requires_grad=True)
    f = f_chain(x)
    f.backward()
    return x.grad.item()


def chain_grad_autograd_grad(x_val):
    x = torch.tensor(float(x_val), requires_grad=True)
    f = f_chain(x)
    (grad_x,) = torch.autograd.grad(f, x)
    return grad_x.item()


def chain_grad_analytic(x):
    return math.cos(x ** 2 + 1) * 2 * x

# Тесты

def run_tests():
    # 2.1
    for point in [(1, 2, 3), (0, 0, 0), (-1.5, 0.5, 2)]:
        _, gx, gy, gz = grads_xyz_autograd(*point)
        ax, ay, az = grads_xyz_analytic(*point)
        assert math.isclose(gx, ax, abs_tol=1e-5)
        assert math.isclose(gy, ay, abs_tol=1e-5)
        assert math.isclose(gz, az, abs_tol=1e-5)

    # 2.2 — пример, посчитанный на бумаге (см. main)
    x = torch.tensor([1., 2., 3., 4.])
    y = torch.tensor([2., 4., 6., 8.])
    loss, dw, db = mse_grads_autograd(x, y, w_val=1.0, b_val=0.0)
    assert math.isclose(loss, 7.5, abs_tol=1e-5)
    assert math.isclose(dw, -15.0, abs_tol=1e-5)
    assert math.isclose(db, -5.0, abs_tol=1e-5)

    # Своя MSE должна совпадать со встроенной в PyTorch
    y_pred = torch.rand(10)
    y_true = torch.rand(10)
    assert torch.isclose(mse(y_pred, y_true), torch.nn.functional.mse_loss(y_pred, y_true))

    # Если w и b уже идеальные (y = 2x), ошибка и градиенты равны нулю
    loss, dw, db = mse_grads_autograd(x, y, w_val=2.0, b_val=0.0)
    assert loss == 0 and dw == 0 and db == 0

    # Проверка обработки ошибок
    try:
        mse(torch.rand(3), torch.rand(4))
        raise AssertionError("Должна была быть ошибка формы")
    except ValueError:
        pass

    # 2.3 — три способа должны дать одно и то же
    for x_val in [0.0, 1.0, 2.0, -0.7]:
        a = chain_grad_backward(x_val)
        b = chain_grad_autograd_grad(x_val)
        c = chain_grad_analytic(x_val)
        assert math.isclose(a, c, abs_tol=1e-5)
        assert math.isclose(b, c, abs_tol=1e-5)

    print("Все тесты пройдены ✅")


# ---------------------------------------------------------------------------
# Демонстрация
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("2.1 f(x, y, z) = x^2 + y^2 + z^2 + 2xyz в точке (1, 2, 3)")
    print("=" * 60)
    f, gx, gy, gz = grads_xyz_autograd(1, 2, 3)
    ax, ay, az = grads_xyz_analytic(1, 2, 3)
    print(f"f(1, 2, 3) = {f}")
    print(f"{'':6}{'autograd':>10}{'вручную':>10}")
    print(f"{'df/dx':6}{gx:>10.4f}{ax:>10.4f}")
    print(f"{'df/dy':6}{gy:>10.4f}{ay:>10.4f}")
    print(f"{'df/dz':6}{gz:>10.4f}{az:>10.4f}")

    print("\n" + "=" * 60)
    print("2.2 MSE для y_pred = w*x + b")
    print("=" * 60)
    x = torch.tensor([1., 2., 3., 4.])
    y_true = torch.tensor([2., 4., 6., 8.])  # настоящая зависимость: y = 2x
    w0, b0 = 1.0, 0.0                         # стартовые (неправильные) параметры
    loss, dw, db = mse_grads_autograd(x, y_true, w0, b0)
    adw, adb = mse_grads_analytic(x, y_true, w0, b0)
    print(f"x = {x.tolist()}, y_true = {y_true.tolist()}, w = {w0}, b = {b0}")
    print(f"MSE = {loss:.4f}")
    print(f"{'':6}{'autograd':>10}{'вручную':>10}")
    print(f"{'dL/dw':6}{dw:>10.4f}{adw:>10.4f}")
    print(f"{'dL/db':6}{db:>10.4f}{adb:>10.4f}")
    print("Градиенты отрицательные -> чтобы ошибка уменьшилась, w и b надо увеличивать")

    print("\n" + "=" * 60)
    print("2.3 f(x) = sin(x^2 + 1), цепное правило, x = 2")
    print("=" * 60)
    print(f"через .backward():          {chain_grad_backward(2.0):.6f}")
    print(f"через torch.autograd.grad:  {chain_grad_autograd_grad(2.0):.6f}")
    print(f"вручную cos(x^2+1) * 2x:    {chain_grad_analytic(2.0):.6f}")

    print("\n" + "=" * 60)
    run_tests()


if __name__ == "__main__":
    main()