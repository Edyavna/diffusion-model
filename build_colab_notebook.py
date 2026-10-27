"""Build a Colab notebook with automatic download of the pinned NVIDIA network."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'diffusion_mnist.py').read_text(encoding='utf-8')
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
definitions = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        start = min([node.lineno] + [item.lineno for item in node.decorator_list])
        definitions[node.name] = ''.join(lines[start - 1:node.end_lineno]).rstrip() + '\n'

cells = []


def add(kind, body, **metadata):
    cell = dict(cell_type=kind, id=f'cell-{len(cells):03d}', metadata=metadata,
                source=textwrap.dedent(body).strip().splitlines(keepends=True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def md(body):
    add('markdown', body)


def code(body, **metadata):
    add('code', body, **metadata)


md('''
# Диффузионная модель MNIST — запуск в Google Colab

Этот ноутбук обучает **денойзер**: U-Net получает шумную картинку и уровень
шума, затем предсказывает чистую картинку. После обучения **метод Эйлера**
многократно использует эти предсказания, чтобы создать новую цифру из шума.

Ноутбук самостоятельный: загружать дополнительные `.py` файлы не нужно.
Используется та же DhariwalUNet и те же формулы EDM, что в нашем скрипте.
Интернет нужен для загрузки U-Net, библиотек (если отсутствуют) и MNIST.

## Как запустить

1. Открой https://colab.research.google.com/ и выбери **Файл → Загрузить блокнот**.
2. Загрузи `MNIST_Diffusion_Colab.ipynb`.
3. Выбери **Среда выполнения → Сменить среду выполнения → GPU**
   (`Runtime → Change runtime type → GPU`). Доступность ускорителя зависит от Colab.
4. Запускай ячейки сверху вниз кнопкой ▶ или сочетанием **Shift+Enter**.
5. В разделе настроек для первого знакомства можно поставить `EPOCHS = 1`.
   Для основного обучения оставь `6`. Одна эпоха не гарантирует хороших цифр.
6. После обучения посмотри график и примеры, затем скачай результаты в конце.

**Полное обучение запускается ячейкой «Обучение».** Предыдущие ячейки готовят
данные, определения функций и необученную модель.

Весов, обученных на MNIST, в файле нет. Локально проверены вычисления и короткий
проход ячеек на искусственных данных; запуск в облачном Colab и CUDA не проверялся.
Файлы обычной сессии Colab временные. Для сохранения между сессиями используй
необязательный Google Drive или скачай архив результатов.
''')

md('''
## 1. Подготовка библиотек

Используем готовые PyTorch и torchvision среды Colab. Ячейка устанавливает только
отсутствующие библиотеки и не заменяет установленную CUDA-сборку CPU-сборкой.
Если среда имеет несовместимую пару torch/torchvision, импорт покажет ошибку:
попробуй новую стандартную среду Colab, затем снова запусти ноутбук.
''')
code('''
import importlib.util
import subprocess
import sys

requirements = {
    'torch': 'torch>=2.6', 'torchvision': 'torchvision>=0.21',
    'numpy': 'numpy>=1.26', 'PIL': 'Pillow>=10',
    'matplotlib': 'matplotlib>=3.8', 'tqdm': 'tqdm>=4.66',
}
missing = [requirement for module, requirement in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
print('Библиотеки подготовлены. Python:', sys.version.split()[0])
''', tags=['setup'])

md('''
## 2. Загрузить U-Net

Загружаем готовую архитектуру из NVIDIA EDM. Просто запусти ячейку:
код сохранится в `vendor/`, вручную переносить файлы не нужно.
''')
code('''
from pathlib import Path
from urllib.request import urlopen
from hashlib import sha256

revision = '008a4e5316c8e3bfe61a62f874bddba254295afb'
base_url = f'https://raw.githubusercontent.com/NVlabs/edm/{revision}/'
vendor_dir = Path('vendor')
vendor_dir.mkdir(exist_ok=True)
downloads = [
    ('training/networks.py', 'edm_networks.py', '5db27dcd96674b95c72d5e6491b879cdc35e24039ada3411b4b46a28ed1fe284'),
    ('LICENSE.txt', 'EDM_LICENSE.txt', 'b7249043cca5e5c88f99730ddc40490a3284780dfe423b19f2c75bd5e2a8f03e'),
]
for remote, local, expected_hash in downloads:
    content = urlopen(base_url + remote, timeout=60).read()
    if sha256(content).hexdigest() != expected_hash:
        raise RuntimeError(f'Загруженный файл не совпадает с ожидаемой версией: {remote}')
    text = content.decode('utf-8')
    if local == 'edm_networks.py':
        text = text.replace('from torch_utils import persistence',
                            '# Persistence decorators omitted: this project saves state_dict checkpoints.')
        text = text.replace('@persistence.persistent_class', '')
    (vendor_dir / local).write_text(text, encoding='utf-8')
(vendor_dir / '__init__.py').touch()
(vendor_dir / 'SOURCE.md').write_text(base_url + 'training/networks.py', encoding='utf-8')
print('U-Net готова.')
''', tags=['vendor'])

md('''
## 3. Импорты и проверка устройства

**Тензор** — многомерный массив чисел. PyTorch выполняет операции над тензорами
и вычисляет градиенты для обучения. **GPU** ускоряет вычисления; CPU тоже
поддерживается, но полное обучение на нём может быть долгим.
''')
code('''
import csv
import gc
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage
from tqdm.auto import tqdm

from vendor.edm_networks import DhariwalUNet

version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
if version < (2, 6):
    raise RuntimeError('Нужен PyTorch >= 2.6. Используй актуальную стандартную среду Colab.')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('Устройство:', DEVICE)
if DEVICE.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('GPU не обнаружен. Включи GPU в настройках Colab до начала обучения.')
''', tags=['imports'])

md('''
## 4. Настройки

- `CONDITIONAL = True`: можно заказывать конкретную цифру.
- `EPOCHS`: число проходов по обучающему набору.
- `BATCH_SIZE`: сколько картинок обрабатывается за шаг обучения. При нехватке
  памяти снизь с `128` до `64` или `32`, затем начни новый запуск.
- `LEARNING_RATE`: масштаб обновления параметров.
- `SEED`: начальная настройка генераторов случайных чисел.
- `USE_DRIVE = True`: подключить Google Drive и сохранять результат после каждой эпохи.
  Colab запросит доступ к Drive. По умолчанию подключение выключено.
- `TRAIN_MODEL = False`: пропустить обучение и загрузить готовый `.pth`.
  Тогда укажи `CHECKPOINT_PATH` или включи `UPLOAD_CHECKPOINT`.
- `ALLOW_CPU_TRAINING = True`: разрешить обучение без GPU, если это осознанно нужно.

Начни с исходных значений. Размер пакета здесь 128 вместо 256 в исходном скрипте,
чтобы уменьшить расход памяти. Остальная архитектура и формулы сохранены.
''')
code('''
CONDITIONAL = True
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
SEED = 42
PREVIEW_STEPS = 50

TRAIN_MODEL = True
ALLOW_CPU_TRAINING = False
USE_DRIVE = False
UPLOAD_CHECKPOINT = False
CHECKPOINT_PATH = ''

if EPOCHS < 1 or BATCH_SIZE < 1 or LEARNING_RATE <= 0 or PREVIEW_STEPS < 2:
    raise ValueError('epochs и batch size должны быть положительными, steps >= 2.')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    result_root = Path('/content/drive/MyDrive/MNIST_Diffusion')
else:
    result_root = Path('mnist_diffusion_runs')
run_name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]
RUN_DIR = result_root / run_name
DATA_DIR = Path('mnist_data')
print('Папка результатов этого запуска:', RUN_DIR)
''', tags=['parameters'])

md('''
## 5. Данные: изображения цифр MNIST

MNIST содержит 60 000 обучающих и 10 000 тестовых примеров. Каждую картинку
уменьшаем до `16×16`. Один пакет имеет форму `[B, 1, 16, 16]`:
`B` — число изображений, `1` — один канал яркости, последние числа — высота и ширина.
При первом запуске данные скачаются автоматически.
''')
code(definitions['make_loaders'] + '\n' + definitions['calc_data_std'])
code('''
if TRAIN_MODEL:
    train_loader, test_loader = make_loaders(DATA_DIR, BATCH_SIZE, workers=0, device=DEVICE)
    example_images, example_labels = next(iter(test_loader))
    grid = make_grid(example_images[:40], nrow=10)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Настоящие изображения из MNIST')
    plt.show()
    print('Форма пакета:', tuple(example_images.shape))
    print('Метки первых изображений:', example_labels[:10].tolist())
else:
    print('Обучение отключено: скачивание MNIST пропущено.')
''', tags=['data'])

md('''
## 6. Денойзер: U-Net с обёрткой EDM

U-Net содержит обучаемые параметры. Обёртка `EDMPrecond` масштабирует числа
и собирает **оценку чистого изображения**. Она получает:

```text
шумную картинку + уровень шума sigma + класс цифры (для условной модели)
```

Метка `10` означает отсутствие условия. Это служебная метка, а не число десять.
Денойзер возвращает отдельное изображение и не меняет входной тензор на месте.
''')
code(definitions['EDMPrecond'])

md('''
## 7. Ошибка и обучение удалению шума

Мы сами добавляем шум к чистому изображению, поэтому знаем правильный ответ.

```python
noisy = clean + sigma * noise
prediction = model(noisy, sigma)
```

`edm_loss` сравнивает предсказание с оригиналом. При условном обучении примерно
10% меток заменяются на `10`, чтобы сеть умела работать и без указания цифры.
`evaluate` считает ошибку на проверочных данных, не обновляя параметры.

Loss — ошибка восстановления, а не процент правильных цифр. Тестовый MNIST здесь
используется для выбора лучшей эпохи, то есть играет роль validation.
Для независимой итоговой оценки нужен отдельный validation split.
''')
code(definitions['edm_loss'] + '\n' + definitions['evaluate'])

md('''
## 8. Генерация: что вычисляет метод Эйлера

При генерации исходной картинки нет — начинаем со случайного шума.
На каждом шаге денойзер предлагает оценку чистой картинки, а метод Эйлера
вычисляет следующее состояние `x`, соответствующее меньшему уровню шума:

```python
denoised = model(x, sigma)
x_next = x + (x - denoised) / sigma * (next_sigma - sigma)
```

Шаг `next_sigma - sigma` отрицателен: состояние движется в сторону оценки денойзера.
Для стохастического режима дополнительно добавляется шум и удваивается снос.
**Метод Эйлера не обучается. Обучаются веса U-Net.**

CFG усиливает влияние класса. В этой реализации `cfg=0` уже означает обычную
условную генерацию. Большие значения CFG не гарантируют лучшего качества.
''')
code(definitions['get_timesteps'] + '\n' + definitions['sample_euler'] + '\n' + definitions['save_samples'])

md('''
## 9. Сохранение и загрузка `.pth`

Checkpoint содержит веса и конфигурацию, необходимую для создания сети.
`last.pth` также хранит состояние Adam. `best.pth` — лучшие веса по проверочному loss.
Функции совместимы с checkpoint нашего локального Python-скрипта.
Автоматического продолжения прерванного обучения здесь нет: сохранённую модель
можно загрузить для генерации, а обучение запускается с нуля.
''')
code(definitions['save_checkpoint'] + '\n' + definitions['load_checkpoint'])

md('''
## 10. Создать необученную модель

Диапазон пикселей для обучения переводим из `[0,1]` в `[-1,1]`:
`clean = images * 2 - 1`. Измеряем разброс данных и передаём его как `sigma_data`.
Это характеристика данных; `sigma` в denoising loss — сила добавленного шума.
Adam использует градиенты, чтобы менять параметры U-Net.
''')
code('''
if TRAIN_MODEL:
    if DEVICE.type != 'cuda' and not ALLOW_CPU_TRAINING:
        raise RuntimeError(
            'Обучение остановлено: включи GPU в настройках Colab и снова запусти ячейки. '
            'Для осознанного запуска на CPU задай ALLOW_CPU_TRAINING=True.'
        )
    sigma_data = calc_data_std(train_loader) * 2
    model = EDMPrecond(sigma_data=sigma_data, conditional=CONDITIONAL).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print('Параметров:', f'{sum(p.numel() for p in model.parameters()):,}')
    print('sigma_data:', sigma_data)
    print('Модель пока не обучена. Следующая ячейка начнёт обучение.')
else:
    model = None
    print('Создание случайной модели пропущено: будем загружать готовые веса.')
''', tags=['create-model'])

md('''
## 11. Обучение

Для каждого пакета:

1. Сбросить предыдущие градиенты: `zero_grad`.
2. Зашумить данные и вычислить ошибку: `edm_loss`.
3. Вычислить градиенты: `backward`.
4. Изменить параметры: `step`.

После каждой эпохи сохраняются `last.pth`, `best.pth`, `metrics.csv` и PNG.
На графике появится средняя ошибка по эпохам; первые изображения могут быть плохими.
При исчерпании памяти GPU уменьши `BATCH_SIZE`, затем начни новый запуск с ячейки
настроек. Повторный запуск этой ячейки в той же папке не перезаписывает веса.
''')
code('''
history = []
if TRAIN_MODEL:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / 'last.pth').exists():
        raise FileExistsError('В этой папке уже есть обучение. Начни новый запуск с ячейки настроек.')
    config = dict(conditional=CONDITIONAL, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE, seed=SEED, device=str(DEVICE),
                  torch_version=str(torch.__version__), sigma_data=sigma_data)
    (RUN_DIR / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best_loss = math.inf
    with (RUN_DIR / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'test_loss'])
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total, seen = 0.0, 0
            progress = tqdm(train_loader, desc=f'Эпоха {epoch}/{EPOCHS}')
            for images, labels in progress:
                clean = images.to(DEVICE) * 2 - 1
                labels = labels.to(DEVICE) if model.conditional else None
                optimizer.zero_grad(set_to_none=True)
                loss = edm_loss(model, clean, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Loss стал NaN или бесконечностью.')
                loss.backward()
                optimizer.step()
                total += loss.item() * len(images)
                seen += len(images)
                progress.set_postfix(loss=f'{loss.item():.4f}')
            metrics = dict(epoch=epoch, train_loss=total / seen,
                           test_loss=evaluate(model, test_loader, DEVICE))
            history.append(metrics)
            writer.writerow(metrics)
            stream.flush()
            save_checkpoint(model, RUN_DIR / 'last.pth', epoch, optimizer, metrics)
            if metrics['test_loss'] < best_loss:
                best_loss = metrics['test_loss']
                save_checkpoint(model, RUN_DIR / 'best.pth', epoch, metrics=metrics)
            preview_labels = torch.arange(40, device=DEVICE) % 10 if model.conditional else None
            preview = sample_euler(model, count=40, labels=preview_labels,
                                   steps=PREVIEW_STEPS, seed=SEED)
            preview_path = RUN_DIR / f'epoch_{epoch:03d}.png'
            save_samples(preview, preview_path)
            print(f"Эпоха {epoch}: train={metrics['train_loss']:.5f}, "
                  f"test={metrics['test_loss']:.5f}")
            display(DisplayImage(filename=str(preview_path), width=800))
    plt.figure(figsize=(8, 4))
    plt.plot([row['epoch'] for row in history], [row['train_loss'] for row in history], label='train')
    plt.plot([row['epoch'] for row in history], [row['test_loss'] for row in history], label='test / validation')
    plt.xlabel('Эпоха')
    plt.ylabel('Denoising loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    print('Обучение завершено. Веса:', RUN_DIR / 'best.pth')
else:
    print('Обучение отключено настройкой TRAIN_MODEL=False.')
''', tags=['train'])

md('''
## 12. Загрузить лучшую модель

После обучения автоматически загружается `best.pth`. Теперь используется лучшая
по loss эпоха, которая может отличаться от последней.

Если хочешь использовать ранее сохранённые веса в новой сессии:
в настройках укажи `TRAIN_MODEL=False` и путь `CHECKPOINT_PATH` на Drive
либо `UPLOAD_CHECKPOINT=True` для загрузки с компьютера. Затем выполни ячейки
по порядку. Обучение и загрузка MNIST будут пропущены.
''')
code('''
if TRAIN_MODEL:
    selected_checkpoint = RUN_DIR / 'best.pth'
else:
    if UPLOAD_CHECKPOINT:
        from google.colab import files
        uploaded = files.upload()
        checkpoint_names = [name for name in uploaded if name.endswith(('.pth', '.pt'))]
        if len(checkpoint_names) != 1:
            raise ValueError('Загрузи ровно один checkpoint .pth или .pt.')
        selected_checkpoint = Path(checkpoint_names[0])
    elif CHECKPOINT_PATH:
        selected_checkpoint = Path(CHECKPOINT_PATH)
    else:
        raise ValueError('Укажи CHECKPOINT_PATH или включи UPLOAD_CHECKPOINT в настройках.')
    RUN_DIR.mkdir(parents=True, exist_ok=True)
if model is not None:
    del model
if 'optimizer' in globals():
    del optimizer
gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()
model = load_checkpoint(selected_checkpoint, DEVICE)
print('Загружено:', selected_checkpoint)
print('Условная модель:', model.conditional)
''', tags=['load-checkpoint'])

md('''
## 13. Сгенерировать выбранную цифру

Изменяй `DIGIT`, `CFG`, `GENERATION_SEED` и повторно запускай только эту ячейку.
Повторное обучение не требуется. `DIGIT=None` создаёт классы 0–9 по порядку.
При безусловной модели метки не используются и CFG автоматически равен нулю.

`STOCHASTIC=False` — детерминированный процесс; `True` — добавление шума на шагах.
`GENERATION_STEPS` — число обновлений Эйлера. `COUNT` — число изображений.
''')
code('''
DIGIT = 7
COUNT = 40
CFG = 2.0
GENERATION_STEPS = 50
GENERATION_SEED = 123
STOCHASTIC = False

if DIGIT is not None and DIGIT not in range(10):
    raise ValueError('DIGIT должен быть от 0 до 9 или None.')
if model.conditional:
    labels = (torch.arange(COUNT, device=DEVICE) % 10 if DIGIT is None else
              torch.full((COUNT,), DIGIT, device=DEVICE, dtype=torch.long))
    actual_cfg = CFG
else:
    labels, actual_cfg = None, 0.0
    print('Безусловная модель: выбор цифры и CFG не применяются.')
generated = sample_euler(model, count=COUNT, labels=labels, steps=GENERATION_STEPS,
                         cfg=actual_cfg, stochastic=STOCHASTIC, seed=GENERATION_SEED)
if not torch.isfinite(generated).all():
    raise FloatingPointError('При генерации получились NaN или бесконечности.')
generated_path = RUN_DIR / 'generated.png'
save_samples(generated, generated_path)
generation_settings = dict(digit=DIGIT if model.conditional else None,
                           labels=labels.cpu().tolist() if labels is not None else None,
                           cfg=actual_cfg, steps=GENERATION_STEPS, seed=GENERATION_SEED,
                           count=COUNT, stochastic=STOCHASTIC,
                           checkpoint=str(selected_checkpoint))
(RUN_DIR / 'generated.json').write_text(json.dumps(generation_settings, indent=2), encoding='utf-8')
display(DisplayImage(filename=str(generated_path), width=800))
print('PNG сохранён:', generated_path)
''', tags=['generate'])

md('''
## 14. Посмотреть именно работу денойзера

Здесь берём настоящее изображение, сами добавляем шум и **один раз** вызываем
денойзер. Метод Эйлера в этой ячейке не используется.

Строки картинки: **оригинал → шумное изображение → предсказание денойзера**.
Изменяй `DEMO_SIGMA`, чтобы увидеть влияние уровня шума. При очень сильном шуме
сеть не обязана восстановить исходное изображение: информации может не хватать.
Изображения при показе обрезаются до `[0,1]`, но вход денойзера не обрезается.
''')
code('''
DEMO_SIGMA = 0.5
if DEMO_SIGMA <= 0:
    raise ValueError('DEMO_SIGMA должен быть положительным.')
if TRAIN_MODEL:
    demo_images, demo_labels = next(iter(test_loader))
    clean = demo_images[:8].to(DEVICE) * 2 - 1
    demo_labels = demo_labels[:8].to(DEVICE) if model.conditional else None
    demo_rng = torch.Generator(device=DEVICE).manual_seed(123)
    noisy = clean + DEMO_SIGMA * torch.randn(clean.shape, device=DEVICE, generator=demo_rng)
    model.eval()
    with torch.inference_mode():
        denoised = model(noisy, DEMO_SIGMA, demo_labels)
    comparison = torch.cat([clean, noisy, denoised])
    comparison_path = RUN_DIR / 'denoising.png'
    save_samples(comparison, comparison_path, columns=8)
    display(DisplayImage(filename=str(comparison_path), width=800))
    print('Сверху вниз: оригинал, шум, оценка денойзера.')
else:
    print('Демонстрация пропущена: в режиме готовых весов MNIST не загружался.')
''', tags=['denoising-demo'])

md('''
## 15. Скачать результаты

Запускать необязательно. Поставь `DOWNLOAD_RESULTS=True` и выполни ячейку,
чтобы скачать ZIP с весами, метриками, настройками, PNG и исходником сети.
Архив также включает код функций модели, чтобы checkpoint можно было восстановить.
При использовании Google Drive файлы уже сохранены в выбранной папке Drive.
Сам ноутбук сохраняй отдельно через меню Colab.
''')
code('''
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    from google.colab import files
    # Собираем определения из выполненных ячеек, включая пользовательские правки.
    import ast
    names = ['EDMPrecond', 'edm_loss', 'get_timesteps', 'sample_euler',
             'save_samples', 'save_checkpoint', 'load_checkpoint']
    definitions = {}
    for source in get_ipython().history_manager.input_hist_raw:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                definitions[node.name] = ast.unparse(node)
    if set(names) - definitions.keys():
        raise RuntimeError('Сначала выполни ячейки с определениями модели и функций.')
    imports = ('from pathlib import Path\\nimport torch\\nfrom torch import nn\\n'
               'from torch.nn import functional as F\\n'
               'from torchvision.utils import save_image\\n'
               'from vendor.edm_networks import DhariwalUNet\\n\\n')
    export_source = imports + '\\n\\n'.join(definitions[name] for name in names)
    (RUN_DIR / 'model_definition.py').write_text(export_source, encoding='utf-8')
    shutil.copytree('vendor', RUN_DIR / 'vendor', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    if not (RUN_DIR / 'best.pth').exists():
        shutil.copy2(selected_checkpoint, RUN_DIR / 'best.pth')
    (RUN_DIR / 'RESTORE.txt').write_text(
        'Install torch>=2.6, torchvision>=0.21, numpy and Pillow.\\n'
        'From this folder run:\\n'
        'from model_definition import load_checkpoint\\n'
        'model = load_checkpoint("best.pth", "cpu")\\n', encoding='utf-8')
    archive_path = shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR)
    files.download(archive_path)
else:
    print('Для скачивания архива установи DOWNLOAD_RESULTS=True и повтори ячейку.')
''', tags=['download'])

md('''
## Что ты выполнил

- Загрузил изображения цифр как тензоры.
- Обучил U-Net предсказывать чистую картинку по шумной.
- Использовал фиксированные формулы EDM для масштабирования и ошибки.
- Получил новые цифры из шума методом Эйлера.
- Для условной модели использовал метку класса и CFG.
- Сохранил веса для генерации без повторного обучения.

**Обучается U-Net. Метод Эйлера, правила добавления шума и коэффициенты EDM
заданы программой и не обучаются.**

Справка Colab: https://research.google.com/colaboratory/faq.html
''')

notebook = dict(cells=cells, metadata={
    'colab': {'name': 'MNIST_Diffusion_Colab.ipynb', 'provenance': []},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
    'accelerator': 'GPU',
}, nbformat=4, nbformat_minor=5)
path = ROOT / 'MNIST_Diffusion_Colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
for index, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell_{index}', 'exec')
print(f'Created {path.name}: {len(cells)} cells, {path.stat().st_size:,} bytes')
